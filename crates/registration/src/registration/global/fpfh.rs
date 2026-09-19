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

    // --- normals ---
    // If the cloud already has normals, use them; otherwise estimate via PCA.
    let normals: Vec<Vector3<f32>> = if let Some(ref normals) = cloud.normals {
        normals.clone()
    } else {
        estimate_normals_pca(&cloud.points, radius)
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

    // --- Step 1: Compute SPFH for every point ---
    let all_spfh: Vec<[f32; 33]> = (0..n)
        .map(|i| {
            let neighbors = radius_search(&points[i], i);
            compute_spfh(&points[i], &normals[i], &neighbors, points, &normals)
        })
        .collect();

    // --- Step 2: Weight SPFHs to produce FPFH ---
    let fpfh_features: Vec<FPFHFeature> = (0..n)
        .map(|i| {
            let neighbors = radius_search(&points[i], i);
            let histogram = weight_spfh(&points[i], &all_spfh[i], &neighbors, points, &all_spfh);
            FPFHFeature { histogram }
        })
        .collect();

    Ok(fpfh_features)
}

/// Estimate normals via PCA over radius-based neighbours.
///
/// This is a self-contained fallback so the registration crate does not
/// depend on `cv-3d`.
fn estimate_normals_pca(points: &[Point3<f32>], radius: f32) -> Vec<Vector3<f32>> {
    let n = points.len();
    let mut normals = vec![Vector3::z(); n];

    if n == 0 {
        return normals;
    }

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
            continue; // keep default (0,0,1)
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

    normals
}

/// Compute Simple Point Feature Histogram for a single point.
///
/// Implements the SPFH computation from Rusu (ICRA 2009) using the Darboux
/// frame to compute three angular features (alpha, phi, theta) for each
/// pair (source_point, neighbor), then bins them into an 11-bin histogram
/// per feature (33 bins total).
fn compute_spfh(
    point: &Point3<f32>,
    normal: &Vector3<f32>,
    neighbors: &[usize],
    points: &[Point3<f32>],
    normals: &[Vector3<f32>],
) -> [f32; 33] {
    let mut histogram = [0.0f32; 33];
    let mut count = 0u32;

    for &neighbor_idx in neighbors {
        if neighbor_idx >= points.len() {
            continue;
        }

        let neighbor = &points[neighbor_idx];
        let d = neighbor - point;
        let dist = d.norm();

        if dist < 1e-6 {
            continue;
        }

        let n_target = &normals[neighbor_idx];

        // Darboux frame (Rusu ICRA 2009):
        //   u = n_source
        //   v = u x (p_target - p_source) / ||p_target - p_source||
        //   w = u x v
        let u = *normal;
        let v_raw = u.cross(&d);
        let v_norm = v_raw.norm();
        if v_norm < 1e-6 {
            continue;
        }
        let v = v_raw / v_norm;
        let w = u.cross(&v);

        // Angular features:
        //   alpha = v . n_target
        //   phi   = u . d / ||d||
        //   theta = atan2(w . n_target, u . n_target)
        let alpha = v.dot(n_target);
        let phi = u.dot(&d) / dist;
        let theta = (w.dot(n_target)).atan2(u.dot(n_target));

        // Bin into 11 bins per feature
        // alpha in [-1, 1], phi in [-1, 1], theta in [-PI, PI]
        let alpha_bin = ((alpha + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
        let phi_bin = ((phi + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
        let theta_bin = ((theta + std::f32::consts::PI) * (11.0 / (2.0 * std::f32::consts::PI)))
            .floor()
            .clamp(0.0, 10.0) as usize;

        histogram[alpha_bin] += 1.0;
        histogram[11 + phi_bin] += 1.0;
        histogram[22 + theta_bin] += 1.0;
        count += 1;
    }

    // Normalize by count (each bin becomes a fraction)
    if count > 0 {
        let inv_k = 1.0 / count as f32;
        for h in &mut histogram {
            *h *= inv_k;
        }
    }

    histogram
}

/// Weight SPFH with neighbors to produce FPFH (Rusu ICRA 2009).
///
/// FPFH(p) = SPFH(p) + (1/k) * sum_{i=1}^{k} (1/||p - p_i||) * SPFH(p_i)
///
/// where k is the number of neighbors and p_i are the neighbor points.
fn weight_spfh(
    point: &Point3<f32>,
    own_spfh: &[f32; 33],
    neighbors: &[usize],
    points: &[Point3<f32>],
    all_spfh: &[[f32; 33]],
) -> [f32; 33] {
    let mut fpfh = *own_spfh;

    let k = neighbors.len();
    if k == 0 {
        return fpfh;
    }

    let inv_k = 1.0 / k as f32;

    for &neighbor_idx in neighbors {
        let dist = (points[neighbor_idx] - point).norm();
        if dist < 1e-6 {
            continue;
        }

        let w = inv_k / dist;
        let neighbor_spfh = &all_spfh[neighbor_idx];
        for bin in 0..33 {
            fpfh[bin] += neighbor_spfh[bin] * w;
        }
    }

    // Normalize to sum to 100 (standard FPFH convention)
    let sum: f32 = fpfh.iter().sum();
    if sum > 1e-6 {
        let scale = 100.0 / sum;
        for h in &mut fpfh {
            *h *= scale;
        }
    }

    fpfh
}
