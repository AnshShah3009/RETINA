//! FPFH (Fast Point Feature Histogram) feature computation
//!
//! Implements the full FPFH pipeline from Rusu (ICRA 2009):
//! 1. Estimate normals if not already present (via PCA over kNN).
//! 2. Build a voxel-grid spatial index for radius search.
//! 3. Compute SPFH (Simplified Point Feature Histograms) for every point.
//! 4. Weight neighbour SPFHs to produce FPFH for every point.
//!
//! Returns one 33-bin feature vector per point.

#![allow(deprecated)]

use cv_core::point_cloud::PointCloud;
use cv_core::{Error, Result};
use nalgebra::{Point3, Vector3};

/// FPFH (Fast Point Feature Histogram) feature
#[derive(Debug, Clone)]
pub struct FPFHFeature {
    pub histogram: [f32; 33], // 33-dimensional histogram
}

/// Compute FPFH features for a point cloud.
///
/// Implements the full FPFH pipeline from Rusu (ICRA 2009):
/// 1. Estimate normals if not already present (via PCA over kNN).
/// 2. Build a voxel-grid spatial index for radius search.
/// 3. Compute SPFH (Simplified Point Feature Histograms) for every point.
/// 4. Weight neighbour SPFHs to produce FPFH for every point.
///
/// Returns one 33-bin feature vector per point.
pub fn compute_fpfh_features(cloud: &PointCloud, radius: f32) -> Result<Vec<FPFHFeature>> {
    let n = cloud.points.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // A zero or non-finite radius divides by zero below: `p.x / voxel_size` is
    // +/-inf, `.floor() as i32` saturates to `i32::MAX`, and the neighbour loop's
    // `vx + dx` then overflows - a panic on ordinary bad input from a public API.
    // (Only for clouds with at least one point, since the empty case returned
    // above.)
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "FPFH radius must be finite and > 0, got {radius}"
        )));
    }

    // --- normals ---
    // If the cloud already has normals, use them; otherwise estimate via PCA.
    let normals: Vec<Vector3<f32>> = if let Some(ref normals) = cloud.normals {
        normals.clone()
    } else {
        {
            let (normals, unsolved) = estimate_normals_pca(&cloud.points, radius);
            if unsolved > 0 {
                // Returning zero-information descriptors here is worse than an
                // error: a downstream FGR ratio test rejects them, but any other
                // consumer of this public function silently gets a descriptor
                // that looks valid and carries no information. Measured: a cloud
                // with 6 points spread 0.5 m apart at radius 0.05 gave 0 of 6
                // features with any non-zero bin, reported as `Ok`.
                return Err(Error::InvalidInput(format!(
                    "cannot estimate normals for {unsolved} of {n} points: fewer \
                     than 3 neighbours within radius {radius}. Increase the radius \
                     or use a denser cloud."
                )));
            }
            normals
        }
    };

    if normals.len() != n {
        return Err(Error::RuntimeError(
            "Normal count does not match point count".to_string(),
        ));
    }

    let points = &cloud.points;
    let radius_sq = radius * radius;

    // --- voxel-grid spatial index for radius search ---
    let voxel_size = radius;
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 4 + 1);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    // Helper: collect all neighbours within `radius` of `center` (excluding self).
    let radius_search = |center: &Point3<f32>, self_idx: usize| -> Vec<usize> {
        let vx = (center.x / voxel_size).floor() as i32;
        let vy = (center.y / voxel_size).floor() as i32;
        let vz = (center.z / voxel_size).floor() as i32;

        let mut result = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != self_idx {
                                let p = &points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= radius_sq {
                                    result.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    };

    // Open3D's `ComputeFPFHFeature`, in f64 as there.
    let points64: Vec<Vector3<f64>> = points.iter().map(|p| p.coords.cast::<f64>()).collect();
    let normals64: Vec<Vector3<f64>> = normals.iter().map(|v| v.cast::<f64>()).collect();
    let neighborhoods: Vec<Vec<usize>> = (0..n).map(|i| radius_search(&points[i], i)).collect();

    // --- Step 1: Compute SPFH for every point ---
    let all_spfh: Vec<[f64; 33]> = (0..n)
        .map(|i| compute_spfh(i, &neighborhoods[i], &points64, &normals64))
        .collect();

    // --- Step 2: Weight SPFHs to produce FPFH ---
    let fpfh_features: Vec<FPFHFeature> = (0..n)
        .map(|i| {
            let fpfh = weight_spfh(i, &neighborhoods[i], &points64, &all_spfh);
            FPFHFeature {
                histogram: fpfh.map(|v| v as f32),
            }
        })
        .collect();

    Ok(fpfh_features)
}

/// Estimate normals via PCA over radius-based neighbours.
///
/// This is a self-contained fallback so the registration crate does not
/// depend on `cv-3d`.
fn estimate_normals_pca(points: &[Point3<f32>], radius: f32) -> (Vec<Vector3<f32>>, usize) {
    let n = points.len();
    let mut normals = vec![Vector3::z(); n];

    if n == 0 {
        return (normals, 0);
    }

    // Points with fewer than three neighbours have no plane to fit: PCA needs
    // three points to determine a normal, and a single arbitrary direction
    // produces a histogram with no information in it. Those keep the `(0, 0, 1)`
    // default, which is why a sparse cloud used to yield descriptors that were
    // entirely zero while reporting success.
    let mut unsolved = 0usize;
    let voxel_size = radius;
    let radius_sq = radius * radius;

    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 4 + 1);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    for (i, center) in points.iter().enumerate() {
        let vx = (center.x / voxel_size).floor() as i32;
        let vy = (center.y / voxel_size).floor() as i32;
        let vz = (center.z / voxel_size).floor() as i32;

        let mut cov = nalgebra::Matrix3::<f32>::zeros();
        let mut centroid = Vector3::zeros();
        let mut count = 0u32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            let p = &points[idx];
                            let dist_sq = (center.x - p.x).powi(2)
                                + (center.y - p.y).powi(2)
                                + (center.z - p.z).powi(2);
                            if dist_sq <= radius_sq {
                                centroid += p.coords;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        if count < 3 {
            unsolved += 1;
            continue; // keep the (0, 0, 1) default
        }

        centroid /= count as f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            let p = &points[idx];
                            let dist_sq = (center.x - p.x).powi(2)
                                + (center.y - p.y).powi(2)
                                + (center.z - p.z).powi(2);
                            if dist_sq <= radius_sq {
                                let diff = p.coords - centroid;
                                cov += diff * diff.transpose();
                            }
                        }
                    }
                }
            }
        }

        // The normal is the eigenvector corresponding to the smallest eigenvalue.
        let eig = cov.symmetric_eigen();
        let mut min_idx = 0;
        let mut min_val = eig.eigenvalues[0].abs();
        for k in 1..3 {
            if eig.eigenvalues[k].abs() < min_val {
                min_val = eig.eigenvalues[k].abs();
                min_idx = k;
            }
        }

        let mut normal = eig.eigenvectors.column(min_idx).into_owned();
        let norm = normal.norm();
        if norm > 1e-6 {
            normal /= norm;
        } else {
            normal = Vector3::z();
        }

        // Orient towards positive-z half-space (convention)
        if normal.z < 0.0 {
            normal = -normal;
        }

        normals[i] = normal;
    }

    (normals, unsolved)
}

/// Open3D's `ComputePairFeatures` (after PCL's `computePairFeatures`): the
/// Darboux-frame angles `(theta, alpha, phi)` of the pair `(p1, n1)`,
/// `(p2, n2)`, or zeros for coincident points or a degenerate frame (which
/// Open3D still bins).
///
/// The source of the frame is whichever point's normal makes the smaller angle
/// with the line joining them, so the features of a pair do not depend on
/// which point is the query; `v = d x n_source`, as PCL and Open3D.
fn pair_features(
    p1: &Vector3<f64>,
    n1: &Vector3<f64>,
    p2: &Vector3<f64>,
    n2: &Vector3<f64>,
) -> [f64; 3] {
    let mut dp = p2 - p1;
    let dist = dp.norm();
    if dist == 0.0 {
        return [0.0; 3];
    }
    let angle1 = n1.dot(&dp) / dist;
    let angle2 = n2.dot(&dp) / dist;
    let (src, tgt, phi) = if angle1.abs().acos() > angle2.abs().acos() {
        dp = -dp;
        (n2, n1, -angle2)
    } else {
        (n1, n2, angle1)
    };
    let v = dp.cross(src);
    let v_norm = v.norm();
    if v_norm == 0.0 {
        return [0.0; 3];
    }
    let v = v / v_norm;
    let w = src.cross(&v);
    let alpha = v.dot(tgt);
    let theta = w.dot(tgt).atan2(src.dot(tgt));
    [theta, alpha, phi]
}

/// Bin of `value` in 11 equal bins over `[lo, hi]`, clamped.
fn bin11(value: f64, lo: f64, hi: f64) -> usize {
    ((11.0 * (value - lo) / (hi - lo)).floor() as i64).clamp(0, 10) as usize
}

/// Simplified Point Feature Histogram of point `i` (Open3D's
/// `ComputeSPFHFeature`): bins 0-10 theta over `[-pi, pi]`, 11-21 alpha and
/// 22-32 phi over `[-1, 1]`; each neighbour adds `100 / k` to one bin of each,
/// so each sub-histogram sums to 100.
///
/// This used to differ from Open3D in every respect that shows in the
/// numbers: the bins were ordered (alpha, phi, theta), the frame was
/// `v = n x d` (mirroring alpha and theta), there was no source/target swap
/// (so a pair's features depended on which point was asked), and each
/// sub-histogram summed to 1.
fn compute_spfh(
    i: usize,
    neighbors: &[usize],
    points: &[Vector3<f64>],
    normals: &[Vector3<f64>],
) -> [f64; 33] {
    let mut histogram = [0.0f64; 33];
    if neighbors.is_empty() {
        return histogram;
    }
    let increment = 100.0 / neighbors.len() as f64;
    for &j in neighbors {
        let [theta, alpha, phi] = pair_features(&points[i], &normals[i], &points[j], &normals[j]);
        histogram[bin11(theta, -std::f64::consts::PI, std::f64::consts::PI)] += increment;
        histogram[11 + bin11(alpha, -1.0, 1.0)] += increment;
        histogram[22 + bin11(phi, -1.0, 1.0)] += increment;
    }
    histogram
}

/// Fast Point Feature Histogram of point `i` (Open3D's `ComputeFPFHFeature`):
/// the neighbours' SPFHs weighted by the inverse *squared* distance, each
/// sub-histogram of that sum scaled to 100, plus the point's own SPFH.
///
/// It used `1 / (k |d|)` weights and scaled the whole 33-bin sum to 100.
fn weight_spfh(
    i: usize,
    neighbors: &[usize],
    points: &[Vector3<f64>],
    all_spfh: &[[f64; 33]],
) -> [f64; 33] {
    let mut fpfh = [0.0f64; 33];
    let mut sums = [0.0f64; 3];
    for &j in neighbors {
        let dist_sq = (points[j] - points[i]).norm_squared();
        if dist_sq == 0.0 {
            continue;
        }
        for bin in 0..33 {
            let value = all_spfh[j][bin] / dist_sq;
            sums[bin / 11] += value;
            fpfh[bin] += value;
        }
    }
    for bin in 0..33 {
        if sums[bin / 11] != 0.0 {
            fpfh[bin] *= 100.0 / sums[bin / 11];
        }
        fpfh[bin] += all_spfh[i][bin];
    }
    fpfh
}
