//! Global Registration
//!
//! RANSAC-based global registration that doesn't require initial alignment.
//! Uses FPFH (Fast Point Feature Histograms) for feature matching.
//!
//! FPFH feature computation lives in `fpfh.rs`.
//! RANSAC and FGR registration live in `ransac.rs`.

#![allow(deprecated)]

pub mod fpfh;
pub mod ransac;

pub use fpfh::*;
pub use ransac::*;

use cv_core::point_cloud::PointCloud;

/// ISS (Intrinsic Shape Signatures) feature detector
/// Alternative to FPFH, good for scenes with repeated structures
#[derive(Debug, Clone)]
pub struct ISSFeature {
    pub keypoint_index: usize,
    pub descriptor: [f32; 16], // Simplified 16D descriptor
    pub covariance_eigenvalues: [f32; 3],
}

impl Default for ISSFeature {
    fn default() -> Self {
        Self {
            keypoint_index: 0,
            descriptor: [0.0; 16],
            covariance_eigenvalues: [0.0; 3],
        }
    }
}

/// ISS feature detector parameters
#[derive(Debug, Clone)]
pub struct ISSDetector {
    /// Radius for computing scatter matrix
    pub saliency_radius: f32,
    /// Minimum eigenvalue threshold
    pub min_eigenvalue: f32,
    /// Radius for non-maximum suppression
    pub non_max_radius: f32,
    /// Minimum number of neighbors
    pub min_neighbors: usize,
}

impl Default for ISSDetector {
    fn default() -> Self {
        Self {
            saliency_radius: 0.1,
            min_eigenvalue: 0.001,
            non_max_radius: 0.05,
            min_neighbors: 5,
        }
    }
}

/// Compute ISS keypoints and features
pub fn compute_iss_features(cloud: &PointCloud, detector: ISSDetector) -> Vec<ISSFeature> {
    let points = &cloud.points;
    let n = points.len();

    if n == 0 {
        return Vec::new();
    }

    let saliency_radius_sq = detector.saliency_radius * detector.saliency_radius;
    let non_max_radius_sq = detector.non_max_radius * detector.non_max_radius;

    // Build spatial index
    let voxel_size = detector.saliency_radius;
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 10);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    // Compute saliency (determinant of scatter matrix) for each point
    let mut saliencies: Vec<(usize, f32)> = Vec::with_capacity(n);

    for (i, center) in points.iter().enumerate() {
        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        // Gather neighbors
        let mut neighbors: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != i {
                                let p = points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= saliency_radius_sq {
                                    neighbors.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        if neighbors.len() < detector.min_neighbors {
            saliencies.push((i, 0.0));
            continue;
        }

        // Compute scatter matrix
        let mut scatter = nalgebra::Matrix3::zeros();
        let mut centroid = nalgebra::Vector3::zeros();

        for &idx in &neighbors {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }
        centroid /= neighbors.len() as f32;

        for &idx in &neighbors {
            let p = points[idx];
            let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
            scatter += diff * diff.transpose();
        }

        scatter /= neighbors.len() as f32;

        // Compute eigenvalues
        let eigenvals = scatter.eigenvalues();
        if let Some(eigs) = eigenvals {
            // Use determinant as saliency
            let det = eigs[0] * eigs[1] * eigs[2];
            saliencies.push((i, det));
        } else {
            saliencies.push((i, 0.0));
        }
    }

    // Non-maximum suppression
    let mut keypoints: Vec<usize> = Vec::new();

    // Sort by saliency (descending)
    saliencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; n];

    for (idx, _saliency) in &saliencies {
        if suppressed[*idx] {
            continue;
        }

        keypoints.push(*idx);
        let center = points[*idx];

        // Suppress neighbors within non_max_radius
        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &j in indices {
                            let pj = points[j];
                            let dist_sq = (center.x - pj.x).powi(2)
                                + (center.y - pj.y).powi(2)
                                + (center.z - pj.z).powi(2);
                            if dist_sq <= non_max_radius_sq {
                                suppressed[j] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute descriptors for keypoints
    let mut features = Vec::with_capacity(keypoints.len());

    for &keypoint_idx in &keypoints {
        let center = points[keypoint_idx];

        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        // Get neighbors for descriptor
        let mut neighbors: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != keypoint_idx {
                                let p = points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= saliency_radius_sq {
                                    neighbors.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute covariance for descriptor
        let mut cov = nalgebra::Matrix3::zeros();
        let mut centroid = nalgebra::Vector3::zeros();

        for &idx in &neighbors {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }

        if !neighbors.is_empty() {
            centroid /= neighbors.len() as f32;

            for &idx in &neighbors {
                let p = points[idx];
                let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
                cov += diff * diff.transpose();
            }
        }

        let eigenvals = cov.eigenvalues();
        let mut eigenvalues = [0.0f32; 3];
        if let Some(eigs) = eigenvals {
            eigenvalues = [eigs[0], eigs[1], eigs[2]];
        }

        // Create simple descriptor from eigenvalues
        let mut descriptor = [0.0f32; 16];
        if !neighbors.is_empty() {
            let e1 = eigenvalues[0].max(1e-10);
            let e2 = eigenvalues[1].max(1e-10);
            let e3 = eigenvalues[2].max(1e-10);
            let sum = (e1 + e2 + e3).max(1e-10);

            // Normalized eigenvalues as descriptor
            descriptor[0] = e1 / sum;
            descriptor[1] = e2 / sum;
            descriptor[2] = e3 / sum;
            descriptor[3] = neighbors.len() as f32;

            // Add spatial features
            let dx = centroid[0] - center.x;
            let dy = centroid[1] - center.y;
            let dz = centroid[2] - center.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-10);

            descriptor[4] = dx / dist;
            descriptor[5] = dy / dist;
            descriptor[6] = dz / dist;

            // Fill remaining with eigenvalue ratios
            descriptor[7] = (e1 * e2 / (e3 * e3)).min(100.0);
            descriptor[8] = (e1 * e3 / (e2 * e2)).min(100.0);
            descriptor[9] = (e2 * e3 / (e1 * e1)).min(100.0);
            descriptor[10] = (e1 / e3).min(100.0);
            descriptor[11] = (e2 / e3).min(100.0);
            descriptor[12] = (e1 - e2).abs() / e3.max(1e-10);
            descriptor[13] = (e1 - e3).abs() / e2.max(1e-10);
            descriptor[14] = (e2 - e3).abs() / e1.max(1e-10);
            descriptor[15] = (e1 - e2 - e3).abs() / e1.max(1e-10);
        }

        features.push(ISSFeature {
            keypoint_index: keypoint_idx,
            descriptor,
            covariance_eigenvalues: eigenvalues,
        });
    }

    features
}
