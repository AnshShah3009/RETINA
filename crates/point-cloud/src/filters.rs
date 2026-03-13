//! Open3D-equivalent point cloud filtering operations.
//!
//! Standalone functions operating on `&[Point3<f64>]` slices — no dependency on
//! HAL, GPU, or the `PointCloud` struct from cv-scientific.
//!
//! All heavy loops use **rayon** for automatic parallelism.

use hashbrown::HashMap;
use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

// ── Result types ─────────────────────────────────────────────────────────────

/// Result of voxel downsampling, optionally including normals and colors.
#[derive(Debug, Clone)]
pub struct VoxelDownsampleResult {
    pub points: Vec<Point3<f64>>,
    pub normals: Option<Vec<Vector3<f64>>>,
    pub colors: Option<Vec<Vector3<f64>>>,
}

/// Oriented bounding box computed via PCA.
#[derive(Debug, Clone)]
pub struct OrientedBoundingBox {
    /// Center of the bounding box.
    pub center: Point3<f64>,
    /// Three principal axes (unit vectors), ordered by decreasing extent.
    pub axes: [Vector3<f64>; 3],
    /// Half-extents along each principal axis.
    pub extents: Vector3<f64>,
}

// ── Statistical Outlier Removal ──────────────────────────────────────────────

/// Statistical outlier removal (Open3D equivalent).
///
/// For each point, computes the mean distance to its `nb_neighbors` nearest
/// neighbours.  Points whose mean distance exceeds
/// `global_mean + std_ratio * global_std` are classified as outliers.
///
/// # Returns
/// `(inlier_points, inlier_indices)`.
pub fn statistical_outlier_removal(
    points: &[Point3<f64>],
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    if points.len() <= nb_neighbors {
        let indices: Vec<usize> = (0..points.len()).collect();
        return (points.to_vec(), indices);
    }

    let k = nb_neighbors.min(points.len() - 1);

    // Compute mean distance to k nearest neighbours for every point (parallel).
    let mean_dists: Vec<f64> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let mut dists: Vec<f64> = points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| dist_sq(p, q))
                .collect();
            dists.select_nth_unstable_by(k - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let sum: f64 = dists[..k].iter().map(|d| d.sqrt()).sum();
            sum / k as f64
        })
        .collect();

    // Global mean and standard deviation.
    let n = mean_dists.len() as f64;
    let global_mean = mean_dists.iter().sum::<f64>() / n;
    let variance = mean_dists
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f64>()
        / n;
    let global_std = variance.sqrt();

    let threshold = global_mean + std_ratio * global_std;

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (i, &md) in mean_dists.iter().enumerate() {
        if md <= threshold {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}

// ── Radius Outlier Removal ───────────────────────────────────────────────────

/// Radius outlier removal (Open3D equivalent).
///
/// Removes points that have fewer than `min_neighbors` neighbours within the
/// given `radius`.
///
/// # Returns
/// `(inlier_points, inlier_indices)`.
pub fn radius_outlier_removal(
    points: &[Point3<f64>],
    radius: f64,
    min_neighbors: usize,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    let r2 = radius * radius;

    let counts: Vec<usize> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            points
                .iter()
                .enumerate()
                .filter(|&(j, q)| j != i && dist_sq(p, q) <= r2)
                .count()
        })
        .collect();

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        if c >= min_neighbors {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}

// ── Voxel Downsampling ───────────────────────────────────────────────────────

/// Voxel downsampling (Open3D equivalent).
///
/// Groups points into cubic voxels of side `voxel_size` and replaces each
/// voxel's contents with the centroid.  Optionally averages normals (and
/// re-normalises) and colors.
pub fn voxel_downsample(
    points: &[Point3<f64>],
    normals: Option<&[Vector3<f64>]>,
    colors: Option<&[Vector3<f64>]>,
    voxel_size: f64,
) -> VoxelDownsampleResult {
    if points.is_empty() || voxel_size <= 0.0 {
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
        let key = (
            (p.x * inv).floor() as i64,
            (p.y * inv).floor() as i64,
            (p.z * inv).floor() as i64,
        );
        voxels.entry(key).or_default().push(i);
    }

    let has_normals = normals.is_some();
    let has_colors = colors.is_some();

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

    for indices in voxels.values() {
        let n = indices.len() as f64;

        // Average position.
        let mut sum = Vector3::zeros();
        for &i in indices {
            sum += points[i].coords;
        }
        out_points.push(Point3::from(sum / n));

        // Average normals (then re-normalise).
        if let (Some(norms), Some(ref mut out_n)) = (&normals, &mut out_normals) {
            let mut nsum = Vector3::zeros();
            for &i in indices {
                nsum += norms[i];
            }
            let len = nsum.norm();
            if len > 1e-15 {
                out_n.push(nsum / len);
            } else {
                out_n.push(Vector3::new(0.0, 0.0, 1.0));
            }
        }

        // Average colors.
        if let (Some(cols), Some(ref mut out_c)) = (&colors, &mut out_colors) {
            let mut csum = Vector3::zeros();
            for &i in indices {
                csum += cols[i];
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

// ── Uniform Downsampling ─────────────────────────────────────────────────────

/// Uniform downsampling: keep every `every_k`-th point.
pub fn uniform_downsample(points: &[Point3<f64>], every_k: usize) -> Vec<Point3<f64>> {
    if every_k == 0 {
        return points.to_vec();
    }
    points.iter().step_by(every_k).copied().collect()
}

// ── Crop to AABB ─────────────────────────────────────────────────────────────

/// Crop point cloud to an axis-aligned bounding box.
///
/// # Returns
/// `(cropped_points, inlier_indices)`.
pub fn crop_aabb(
    points: &[Point3<f64>],
    min_bound: &Point3<f64>,
    max_bound: &Point3<f64>,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    let mut out_points = Vec::new();
    let mut out_indices = Vec::new();
    for (i, p) in points.iter().enumerate() {
        if p.x >= min_bound.x
            && p.x <= max_bound.x
            && p.y >= min_bound.y
            && p.y <= max_bound.y
            && p.z >= min_bound.z
            && p.z <= max_bound.z
        {
            out_points.push(*p);
            out_indices.push(i);
        }
    }
    (out_points, out_indices)
}

// ── Bounding Boxes ───────────────────────────────────────────────────────────

/// Compute the axis-aligned bounding box.
///
/// # Returns
/// `(min_corner, max_corner)`.
///
/// # Panics
/// Panics if `points` is empty.
pub fn compute_aabb(points: &[Point3<f64>]) -> (Point3<f64>, Point3<f64>) {
    assert!(!points.is_empty(), "compute_aabb: empty point cloud");
    let mut min = points[0];
    let mut max = points[0];
    for p in &points[1..] {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    (min, max)
}

/// Compute an oriented bounding box using PCA on the point cloud.
///
/// The three principal axes of the cloud define the OBB orientation; the
/// extents are the half-widths of the cloud projected onto each axis.
pub fn compute_obb(points: &[Point3<f64>]) -> OrientedBoundingBox {
    assert!(!points.is_empty(), "compute_obb: empty point cloud");

    let n = points.len() as f64;

    // Centroid.
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    centroid /= n;

    // 3x3 covariance matrix.
    let mut cov = Matrix3::<f64>::zeros();
    for p in points {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    cov /= n;

    // Eigen decomposition (nalgebra's symmetric eigen — fine for 3x3).
    let eig = cov.symmetric_eigen();

    // Sort eigenvalues descending (largest extent first).
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let axes = [
        eig.eigenvectors.column(order[0]).into_owned(),
        eig.eigenvectors.column(order[1]).into_owned(),
        eig.eigenvectors.column(order[2]).into_owned(),
    ];

    // Project points onto axes and find extents.
    let mut mins = [f64::MAX; 3];
    let mut maxs = [f64::MIN; 3];
    for p in points {
        let d = p.coords - centroid;
        for (k, ax) in axes.iter().enumerate() {
            let proj = d.dot(ax);
            mins[k] = mins[k].min(proj);
            maxs[k] = maxs[k].max(proj);
        }
    }

    let extents = Vector3::new(
        (maxs[0] - mins[0]) / 2.0,
        (maxs[1] - mins[1]) / 2.0,
        (maxs[2] - mins[2]) / 2.0,
    );

    // Adjust center to be the midpoint of the projected extents.
    let center_offset = Vector3::new(
        (maxs[0] + mins[0]) / 2.0,
        (maxs[1] + mins[1]) / 2.0,
        (maxs[2] + mins[2]) / 2.0,
    );
    let center = Point3::from(
        centroid
            + axes[0] * center_offset.x
            + axes[1] * center_offset.y
            + axes[2] * center_offset.z,
    );

    OrientedBoundingBox {
        center,
        axes,
        extents,
    }
}

// ── Normal Estimation (PCA, standalone) ──────────────────────────────────────

/// Estimate point cloud normals using PCA over k nearest neighbours.
///
/// This is a standalone, CPU-only implementation that does not depend on the
/// HAL or GPU.  Uses the analytic 3x3 eigensolver (trigonometric eigenvalues +
/// best cross-product eigenvector) for the minimum eigenvector.
///
/// Normals are oriented towards the positive-z half-space by default (the
/// convention used by Open3D when no viewpoint is specified).
pub fn estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    if points.is_empty() {
        return Vec::new();
    }
    let k = k.min(points.len() - 1).max(1);

    points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            // Find k nearest neighbours (brute force).
            let mut dists: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, q)| (j, dist_sq(p, q)))
                .collect();
            dists.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Build covariance matrix from k nearest.
            let mut cov = Matrix3::<f64>::zeros();
            let mut centroid = Vector3::zeros();
            for &(j, _) in &dists[..k] {
                centroid += points[j].coords;
            }
            centroid += p.coords;
            centroid /= (k + 1) as f64;

            for &(j, _) in &dists[..k] {
                let d = points[j].coords - centroid;
                cov += d * d.transpose();
            }
            let d = p.coords - centroid;
            cov += d * d.transpose();

            // Analytic min-eigenvector via symmetric 3x3 eigensolver.
            let normal = min_eigenvector_3x3(&cov);

            // Orient towards +Z (Open3D default when no viewpoint).
            if normal.z < 0.0 {
                -normal
            } else {
                normal
            }
        })
        .collect()
}

/// Analytic minimum eigenvector of a symmetric 3x3 matrix.
///
/// Uses the Geometric Tools / Open3D `RobustEigenSymmetric3x3` method:
/// trigonometric eigenvalues + best cross-product eigenvector.
fn min_eigenvector_3x3(m: &Matrix3<f64>) -> Vector3<f64> {
    // Eigenvalues of a 3x3 symmetric matrix via Cardano's formula.
    let a00 = m[(0, 0)];
    let a01 = m[(0, 1)];
    let a02 = m[(0, 2)];
    let a11 = m[(1, 1)];
    let a12 = m[(1, 2)];
    let a22 = m[(2, 2)];

    let c0 = a00 * a11 * a22 + 2.0 * a01 * a02 * a12
        - a00 * a12 * a12
        - a11 * a02 * a02
        - a22 * a01 * a01;
    let c1 = a00 * a11 - a01 * a01 + a00 * a22 - a02 * a02 + a11 * a22 - a12 * a12;
    let c2 = a00 + a11 + a22;

    let c2_over_3 = c2 / 3.0;
    let a_val = c1 / 3.0 - c2_over_3 * c2_over_3;
    let half_b = 0.5 * (c0 + c2_over_3 * (2.0 * c2_over_3 * c2_over_3 - c1));

    // Clamp to avoid NaN from sqrt of negative due to numerical noise.
    let q = (a_val * a_val * a_val).min(0.0);
    let sqrt_neg_q = (-q).sqrt();
    let magnitude = if sqrt_neg_q > 1e-30 {
        sqrt_neg_q
    } else {
        1e-30
    };
    let angle = (-half_b / magnitude).clamp(-1.0, 1.0).acos() / 3.0;
    let two_sqrt_neg_a = 2.0 * (-a_val).max(0.0).sqrt();

    let mut evals = [
        c2_over_3 + two_sqrt_neg_a * angle.cos(),
        c2_over_3 - two_sqrt_neg_a * (angle + std::f64::consts::FRAC_PI_3).cos(),
        c2_over_3 - two_sqrt_neg_a * (angle - std::f64::consts::FRAC_PI_3).cos(),
    ];

    // Sort to find minimum eigenvalue.
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lambda_min = evals[0];

    // Eigenvector: best cross-product of rows of (M - lambda_min * I).
    let shifted = Matrix3::new(
        a00 - lambda_min,
        a01,
        a02,
        a01,
        a11 - lambda_min,
        a12,
        a02,
        a12,
        a22 - lambda_min,
    );

    let r0 = Vector3::new(shifted[(0, 0)], shifted[(0, 1)], shifted[(0, 2)]);
    let r1 = Vector3::new(shifted[(1, 0)], shifted[(1, 1)], shifted[(1, 2)]);
    let r2 = Vector3::new(shifted[(2, 0)], shifted[(2, 1)], shifted[(2, 2)]);

    // Pick the cross product with the largest magnitude.
    let c01 = r0.cross(&r1);
    let c02 = r0.cross(&r2);
    let c12 = r1.cross(&r2);

    let n01 = c01.norm_squared();
    let n02 = c02.norm_squared();
    let n12 = c12.norm_squared();

    let best = if n01 >= n02 && n01 >= n12 {
        c01
    } else if n02 >= n12 {
        c02
    } else {
        c12
    };

    let len = best.norm();
    if len > 1e-15 {
        best / len
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}

// ── Transform / Paint ────────────────────────────────────────────────────────

/// Transform a point cloud in place by a 4x4 homogeneous matrix.
pub fn transform_points(points: &mut [Point3<f64>], transform: &nalgebra::Matrix4<f64>) {
    for p in points.iter_mut() {
        let h = transform * nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
        let w = if h.w.abs() > 1e-15 { h.w } else { 1.0 };
        *p = Point3::new(h.x / w, h.y / w, h.z / w);
    }
}

/// Create a uniform color array for `num_points` points.
pub fn paint_uniform(num_points: usize, color: &Vector3<f64>) -> Vec<Vector3<f64>> {
    vec![*color; num_points]
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn dist_sq(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a small cluster around the origin.
    fn cluster(n: usize, spacing: f64) -> Vec<Point3<f64>> {
        let mut pts = Vec::with_capacity(n * n * n);
        let size = (n as f64 - 1.0) * spacing;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(Point3::new(
                        i as f64 * spacing - size / 2.0,
                        j as f64 * spacing - size / 2.0,
                        k as f64 * spacing - size / 2.0,
                    ));
                }
            }
        }
        pts
    }

    #[test]
    fn test_statistical_outlier_removal() {
        // 3x3x3 = 27 points, plus 3 far outliers.
        let mut points = cluster(3, 0.1);
        points.push(Point3::new(100.0, 0.0, 0.0));
        points.push(Point3::new(0.0, 100.0, 0.0));
        points.push(Point3::new(0.0, 0.0, 100.0));

        let (filtered, indices) = statistical_outlier_removal(&points, 5, 2.0);

        // The 3 outliers should be removed.
        assert_eq!(
            filtered.len(),
            27,
            "Expected 27 inliers, got {}",
            filtered.len()
        );
        assert_eq!(indices.len(), 27);

        // No outlier index should be present.
        for &idx in &indices {
            assert!(idx < 27, "Outlier index {} should not be in inliers", idx);
        }
    }

    #[test]
    fn test_radius_outlier_removal() {
        // Dense cluster + 2 isolated points.
        let mut points = cluster(3, 0.1); // 27 points, max extent ~0.2
        points.push(Point3::new(50.0, 0.0, 0.0));
        points.push(Point3::new(-50.0, 0.0, 0.0));

        let (filtered, indices) = radius_outlier_removal(&points, 0.5, 2);

        // Isolated points have 0 neighbours within 0.5 → removed.
        assert_eq!(filtered.len(), 27);
        for &idx in &indices {
            assert!(idx < 27);
        }
    }

    #[test]
    fn test_voxel_downsample_cube() {
        // 8 points at corners of a unit cube → 1 voxel with size 2.0 → 1 point at centroid.
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];

        let result = voxel_downsample(&points, None, None, 2.0);
        assert_eq!(
            result.points.len(),
            1,
            "8 corners in one voxel → 1 output point"
        );

        let p = &result.points[0];
        assert!(
            (p.x - 0.5).abs() < 1e-10,
            "centroid x should be 0.5, got {}",
            p.x
        );
        assert!(
            (p.y - 0.5).abs() < 1e-10,
            "centroid y should be 0.5, got {}",
            p.y
        );
        assert!(
            (p.z - 0.5).abs() < 1e-10,
            "centroid z should be 0.5, got {}",
            p.z
        );
    }

    #[test]
    fn test_voxel_downsample_with_normals() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.1, 0.0, 0.0)];
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0)];

        let result = voxel_downsample(&points, Some(&normals), None, 1.0);
        assert_eq!(result.points.len(), 1);

        let n = &result.normals.as_ref().unwrap()[0];
        // Average of (0,0,1) and (0,1,0) normalised = (0, 1/sqrt2, 1/sqrt2).
        let expected = Vector3::new(0.0, 1.0, 1.0).normalize();
        assert!(
            (n - expected).norm() < 1e-10,
            "Averaged normal should be re-normalised, got {:?}",
            n
        );
    }

    #[test]
    fn test_uniform_downsample() {
        let points: Vec<Point3<f64>> = (0..10).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();

        let down = uniform_downsample(&points, 2);
        // Every 2nd: indices 0, 2, 4, 6, 8.
        assert_eq!(down.len(), 5);
        assert!((down[0].x - 0.0).abs() < 1e-15);
        assert!((down[1].x - 2.0).abs() < 1e-15);
        assert!((down[4].x - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_crop_aabb() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];
        let min_b = Point3::new(-0.5, -0.5, -0.5);
        let max_b = Point3::new(1.5, 1.5, 1.5);

        let (cropped, indices) = crop_aabb(&points, &min_b, &max_b);
        assert_eq!(cropped.len(), 2); // (0,0,0) and (1,1,1) inside
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_compute_aabb() {
        let points = vec![
            Point3::new(-1.0, 2.0, 3.0),
            Point3::new(4.0, -5.0, 6.0),
            Point3::new(0.0, 0.0, 0.0),
        ];
        let (min_b, max_b) = compute_aabb(&points);
        assert!((min_b.x - (-1.0)).abs() < 1e-15);
        assert!((min_b.y - (-5.0)).abs() < 1e-15);
        assert!((min_b.z - 0.0).abs() < 1e-15);
        assert!((max_b.x - 4.0).abs() < 1e-15);
        assert!((max_b.y - 2.0).abs() < 1e-15);
        assert!((max_b.z - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_estimate_normals_knn_plane() {
        // Points on the XY plane (z=0): normals should be approximately (0, 0, +/-1).
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
            }
        }

        let normals = estimate_normals_knn(&points, 8);
        assert_eq!(normals.len(), points.len());

        for (i, n) in normals.iter().enumerate() {
            // Normal should be close to (0, 0, 1) (we orient towards +Z).
            assert!(
                n.z.abs() > 0.9,
                "Point {} normal z-component should be ~1.0, got {:?}",
                i,
                n
            );
            // Should be unit length.
            assert!(
                (n.norm() - 1.0).abs() < 1e-10,
                "Normal {} should be unit length, norm = {}",
                i,
                n.norm()
            );
        }
    }

    #[test]
    fn test_transform_points() {
        let mut points = vec![Point3::new(1.0, 0.0, 0.0)];
        // Translation by (10, 20, 30).
        let mut t = nalgebra::Matrix4::identity();
        t[(0, 3)] = 10.0;
        t[(1, 3)] = 20.0;
        t[(2, 3)] = 30.0;

        transform_points(&mut points, &t);
        assert!((points[0].x - 11.0).abs() < 1e-10);
        assert!((points[0].y - 20.0).abs() < 1e-10);
        assert!((points[0].z - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_paint_uniform() {
        let colors = paint_uniform(5, &Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(colors.len(), 5);
        for c in &colors {
            assert!((c.x - 1.0).abs() < 1e-15);
            assert!(c.y.abs() < 1e-15);
            assert!(c.z.abs() < 1e-15);
        }
    }

    #[test]
    fn test_compute_obb() {
        // Points along the X axis — OBB should be elongated along X.
        let points: Vec<Point3<f64>> = (0..20).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();

        let obb = compute_obb(&points);
        // The largest extent axis should be mostly aligned with X.
        let ax = obb.axes[0];
        assert!(
            ax.x.abs() > 0.9,
            "Primary OBB axis should align with X, got {:?}",
            ax
        );
        // Largest half-extent should be ~9.5.
        assert!(
            obb.extents.x > 5.0,
            "Largest half-extent should be > 5, got {}",
            obb.extents.x
        );
    }
}
