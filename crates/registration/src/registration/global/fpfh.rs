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

    // Open3D's `ComputeFPFHFeature`, in f64 as there; the descriptor itself is
    // `cv-pointcloud`'s, shared with `cv_pointcloud::compute_fpfh_feature`.
    let points64: Vec<Vector3<f64>> = points.iter().map(|p| p.coords.cast::<f64>()).collect();
    let normals64: Vec<Vector3<f64>> = normals.iter().map(|v| v.cast::<f64>()).collect();
    let neighborhoods: Vec<Vec<usize>> = (0..n).map(|i| radius_search(&points[i], i)).collect();

    Ok(
        cv_pointcloud::fpfh_from_neighbourhoods(&points64, &normals64, &neighborhoods)
            .into_iter()
            .map(|fpfh| FPFHFeature {
                histogram: fpfh.map(|v| v as f32),
            })
            .collect(),
    )
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
