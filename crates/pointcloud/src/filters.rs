//! Canonical `f64` point-cloud filters.
//!
//! These are the single implementations behind both `cv_3d::filters` (which
//! re-exports them) and the `f32` [`PointCloud`](cv_core::PointCloud) entry
//! points in [`crate::point_cloud`] (which convert to `f64` and call them).
//!
//! Every function here treats a point with a non-finite coordinate (NaN or
//! infinite) as having no position: it is left out of every search tree and
//! every statistic, so it can never change another point's result.

use hashbrown::HashMap;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// Whether a point has a position at all (every coordinate finite).
pub(crate) fn is_finite_point(p: &Point3<f64>) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

/// An `f64` KD-tree over the finite points of a cloud.
///
/// `cv_math::spatial::KDTree` rather than an `f32` tree such as `rstar` or
/// `cv_3d::spatial::KDTree`: those store `f32` coordinates and squared
/// distances, which changes which neighbours are selected and every distance
/// the filters report. The math tree computes `sqrt((dx*dx + dy*dy) + dz*dz)`
/// in `f64`, bit-identical to a brute-force scan's `sqrt` of the squared
/// distance.
pub(crate) struct FiniteKnn {
    tree: cv_math::spatial::KDTree,
    /// `index[j]` is the position in the original slice of tree point `j`.
    index: Vec<usize>,
}

impl FiniteKnn {
    /// Builds the tree over the points listed in `finite` (indices into
    /// `points`, in ascending order). Returns `None` when `finite` is empty.
    pub(crate) fn new(points: &[Point3<f64>], finite: &[usize]) -> Option<Self> {
        if finite.is_empty() {
            return None;
        }
        let coords: Vec<Vec<f64>> = finite
            .iter()
            .map(|&i| {
                let p = &points[i];
                vec![p.x, p.y, p.z]
            })
            .collect();
        let tree = cv_math::spatial::KDTree::new(&coords).ok()?;
        Some(Self {
            tree,
            index: finite.to_vec(),
        })
    }

    /// The `k` nearest finite points to `p`, as `(original index, distance)`
    /// in ascending distance.
    pub(crate) fn query(&self, p: &Point3<f64>, k: usize) -> Vec<(usize, f64)> {
        let mut nn = self.tree.query(&[p.x, p.y, p.z], k);
        for hit in &mut nn {
            hit.0 = self.index[hit.0];
        }
        nn
    }
}

/// Indices of the points that have a position, in ascending order.
pub(crate) fn finite_indices(points: &[Point3<f64>]) -> Vec<usize> {
    (0..points.len())
        .filter(|&i| is_finite_point(&points[i]))
        .collect()
}

/// Statistical outlier removal (Open3D's `remove_statistical_outlier`).
///
/// For each point, computes the mean distance to its `nb_neighbors` nearest
/// points, **the point itself included** (Open3D searches `KNN(nb_neighbors)`,
/// whose first hit is the query point at distance 0, and averages over all of
/// them). Points whose mean distance is not below
/// `global_mean + std_ratio * global_std` are outliers, where `global_std` is
/// the sample standard deviation (`n - 1`) of the per-point means.
/// `nb_neighbors` of 0 is treated as 1, and a value larger than the cloud uses
/// the whole cloud.
///
/// One deliberate difference: Open3D also drops every point whose mean is
/// exactly 0 (all of its neighbours coincide with it), because it uses
/// "mean > 0" as its validity flag. Such a point sits inside a cluster of
/// duplicates and is kept here.
///
/// # Non-finite input
///
/// A point with a NaN or infinite coordinate has no distance to measure: it is
/// left out of the search tree, the mean and the standard deviation, and is
/// never kept. (A NaN mean used to make the threshold NaN, so only the points
/// whose mean was exactly 0 survived.) Fewer than two finite points are all
/// kept.
///
/// # Returns
/// `(inlier_points, inlier_indices)`, indices ascending.
pub fn statistical_outlier_removal(
    points: &[Point3<f64>],
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    let finite = finite_indices(points);
    if finite.len() < 2 {
        let kept = finite.iter().map(|&i| points[i]).collect();
        return (kept, finite);
    }
    let Some(tree) = FiniteKnn::new(points, &finite) else {
        return (Vec::new(), Vec::new());
    };

    // The neighbourhood includes the point. This used to be `nb_neighbors`
    // *other* points averaged over `nb_neighbors`, the population standard
    // deviation and an inclusive threshold, so the same parameters kept a
    // different set from Open3D: measured on 500 spread points, (5, 0.5) kept
    // 358 where Open3D keeps 364, and (20, 0.3) kept 343 where it keeps 340.
    let k = nb_neighbors.max(1).min(finite.len());

    // Mean distance over the k nearest points of every finite point
    // (parallel): O(n log n) instead of an O(n) scan per point. A duplicate of
    // the query can come back in its place; both are at distance 0, so the
    // mean is the same.
    let mean_dists: Vec<f64> = finite
        .par_iter()
        .map(|&i| {
            let nn = tree.query(&points[i], k);
            nn.iter().map(|&(_, d)| d).sum::<f64>() / nn.len() as f64
        })
        .collect();

    // Global mean and sample standard deviation.
    let n = mean_dists.len() as f64;
    let global_mean = mean_dists.iter().sum::<f64>() / n;
    let variance = mean_dists
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let global_std = variance.sqrt();

    let threshold = global_mean + std_ratio * global_std;

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (&i, &md) in finite.iter().zip(&mean_dists) {
        if md < threshold || md == 0.0 {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}

/// Result of [`voxel_downsample`]: one entry per occupied voxel, in sorted
/// voxel-key order. `normals` / `colors` are `Some` exactly when the
/// corresponding input was given.
#[derive(Debug, Clone)]
pub struct VoxelDownsampleResult {
    pub points: Vec<Point3<f64>>,
    pub normals: Option<Vec<Vector3<f64>>>,
    pub colors: Option<Vec<Vector3<f64>>>,
}

/// Voxel downsampling (Open3D's `voxel_down_sample`).
///
/// Groups points into cubic voxels of side `voxel_size` on the global grid
/// (key `floor(p / voxel_size)`, computed as `floor(p * (1 / voxel_size))` in
/// `f64`) and replaces each voxel's contents with the centroid. Optionally
/// averages normals (and re-normalises them; normals that cancel fall back to
/// the voxel's first normal, then to `+Z`) and colors. The output is in sorted
/// voxel-key order, so it is the same on every run. Normal or colour slices
/// shorter than `points` contribute only the entries they have.
///
/// This is intentionally *not* `cv_3d::spatial::VoxelGrid`: that grid is
/// `f32`, carries no normals/colors, and keys voxels relative to its `origin`.
/// `cv_3d::filters::voxel_downsample`, `cv_3d::gpu::voxel_downsample` and the
/// `f32` [`crate::point_cloud::voxel_down_sample`] all call this function.
///
/// # Invalid and non-finite input
///
/// An empty cloud, or a voxel size that is zero, negative, NaN or infinite,
/// gives an empty result (with empty `normals` / `colors` when they were
/// given). A point with a non-finite coordinate is skipped: it joins no voxel.
pub fn voxel_downsample(
    points: &[Point3<f64>],
    normals: Option<&[Vector3<f64>]>,
    colors: Option<&[Vector3<f64>]>,
    voxel_size: f64,
) -> VoxelDownsampleResult {
    // NaN is not `<= 0`. A non-finite size used to fall through, and
    // `(NaN).floor() as i64` is 0, so every non-finite point landed in voxel
    // `(0, 0, 0)` and poisoned the centroid of whatever finite point was there.
    if points.is_empty() || !voxel_size.is_finite() || voxel_size <= 0.0 {
        return VoxelDownsampleResult {
            points: Vec::new(),
            normals: normals.map(|_| Vec::new()),
            colors: colors.map(|_| Vec::new()),
        };
    }

    let inv = 1.0 / voxel_size;

    // Group points by voxel key.
    let mut voxels: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            continue;
        }
        let key = (
            (p.x * inv).floor() as i64,
            (p.y * inv).floor() as i64,
            (p.z * inv).floor() as i64,
        );
        voxels.entry(key).or_default().push(i);
    }

    let has_normals = normals.is_some();
    let has_colors = colors.is_some();

    // Sort voxel keys so output ordering is deterministic across runs
    // (HashMap iteration order is randomized per process).
    let mut sorted_keys: Vec<(i64, i64, i64)> = voxels.keys().copied().collect();
    sorted_keys.sort_unstable();

    let mut out_points = Vec::with_capacity(voxels.len());
    let mut out_normals = if has_normals {
        Some(Vec::with_capacity(voxels.len()))
    } else {
        None
    };
    let mut out_colors = if has_colors {
        Some(Vec::with_capacity(voxels.len()))
    } else {
        None
    };

    for key in &sorted_keys {
        let indices = &voxels[key];
        let n = indices.len() as f64;

        // Average position.
        let mut sum = Vector3::zeros();
        for &i in indices {
            sum += points[i].coords;
        }
        out_points.push(Point3::from(sum / n));

        // Average normals (then re-normalise).
        if let (Some(norms), Some(out_n)) = (&normals, &mut out_normals) {
            let mut nsum = Vector3::zeros();
            // Indices come from the voxel keys, and the caller may pass fewer
            // normals than points. `voxel_downsample(&[P;2], Some(&[n;1]),
            // None, 2.0)` panicked on element 1 of a 1-element slice. Every other
            // `filters` function validates its optional inputs; this one did not.
            for &i in indices {
                if let Some(n) = norms.get(i) {
                    nsum += n;
                }
            }
            let len = nsum.norm();
            if len > 1e-15 {
                out_n.push(nsum / len);
            } else {
                // Opposing normals cancelled: fall back to the first normal
                // in the voxel rather than an arbitrary +Z.
                let Some(first) = norms.get(indices[0]).copied() else {
                    out_n.push(Vector3::new(0.0, 0.0, 1.0));
                    continue;
                };
                let fl = first.norm();
                if fl > 1e-15 {
                    out_n.push(first / fl);
                } else {
                    out_n.push(Vector3::new(0.0, 0.0, 1.0));
                }
            }
        }

        // Average colors.
        if let (Some(cols), Some(out_c)) = (&colors, &mut out_colors) {
            let mut csum = Vector3::zeros();
            for &i in indices {
                if let Some(c) = cols.get(i) {
                    csum += c;
                }
            }
            out_c.push(csum / n);
        }
    }

    VoxelDownsampleResult {
        points: out_points,
        normals: out_normals,
        colors: out_colors,
    }
}
