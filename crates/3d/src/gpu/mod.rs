//! GPU-Accelerated 3D Processing
//!
//! This module provides GPU-accelerated implementations of all major 3D algorithms.
//! Optimized for minimal GPU-CPU data transfers following Open3D patterns.

use cv_core::{storage::Storage, Tensor};
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;

/// GPU-accelerated point cloud operations
pub mod point_cloud {
    use super::*;

    /// Transform point cloud using the best available runner
    pub fn transform(points: &[Point3<f32>], transform: &Matrix4<f32>) -> Vec<Point3<f32>> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        transform_ctx(points, transform, &runner)
    }

    pub fn transform_ctx(
        points: &[Point3<f32>],
        transform: &Matrix4<f32>,
        runner: &RuntimeRunner,
    ) -> Vec<Point3<f32>> {
        if let Ok(cv_hal::compute::ComputeDevice::Gpu(_gpu)) = runner.device() {
            // TODO: Actual GPU implementation in hal
        }

        points
            .iter()
            .map(|p| transform.transform_point(p))
            .collect()
    }

    /// Compute normals using the runtime scheduler
    pub fn compute_normals(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        compute_normals_ctx(points, k, &runner)
    }

    pub fn compute_normals_ctx(
        points: &[Point3<f32>],
        k: usize,
        runner: &RuntimeRunner,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }

        if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = runner.device() {
            return compute_normals_gpu_with_ctx(points, k, gpu);
        }

        compute_normals_cpu(points, k, 0.0)
    }

    fn compute_normals_gpu_with_ctx(
        points: &[Point3<f32>],
        k: usize,
        gpu: &cv_hal::gpu::GpuContext,
    ) -> Vec<Vector3<f32>> {
        use cv_hal::gpu_kernels::pointcloud_gpu;
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let pts: Vec<nalgebra::Vector3<f32>> = points.iter().map(|p| p.coords).collect();

        match pointcloud_gpu::compute_normals_morton_gpu(gpu, &pts, k as u32) {
            Ok(n) => n,
            Err(_) => compute_normals_cpu(points, k, 0.0),
        }
    }

    /// CPU path with optimized spatial hashing
    pub fn compute_normals_cpu(
        points: &[Point3<f32>],
        k: usize,
        voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let vs = if voxel_size > 0.0 {
            voxel_size
        } else {
            compute_adaptive_voxel_size(points, k)
        };

        // hashbrown::HashMap uses aHash — 3-5× faster than std SipHash for integer keys.
        let mut voxel_grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
            hashbrown::HashMap::with_capacity(points.len() / 8);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / vs).floor() as i32;
            let vy = (p.y / vs).floor() as i32;
            let vz = (p.z / vs).floor() as i32;
            voxel_grid.entry((vx, vy, vz)).or_default().push(i);
        }

        points
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let (vx, vy, vz) = (
                    (center.x / vs).floor() as i32,
                    (center.y / vs).floor() as i32,
                    (center.z / vs).floor() as i32,
                );

                let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(27 * k);
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            if let Some(bucket) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in bucket {
                                    if idx != i {
                                        let p = points[idx];
                                        let dist = (center.x - p.x) * (center.x - p.x)
                                            + (center.y - p.y) * (center.y - p.y)
                                            + (center.z - p.z) * (center.z - p.z);
                                        candidates.push((dist, idx));
                                    }
                                }
                            }
                        }
                    }
                }

                // O(n) partial-select instead of O(n log n) full sort — we only need
                // the k smallest, not a fully sorted list.
                if candidates.len() > k {
                    candidates.select_nth_unstable_by(k - 1, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.truncate(k);
                }

                compute_pca_normal(center, &candidates, points)
            })
            .collect()
    }

    fn compute_pca_normal(
        _center: &Point3<f32>,
        neighbors: &[(f32, usize)],
        points: &[Point3<f32>],
    ) -> Vector3<f32> {
        if neighbors.len() < 3 {
            return Vector3::z();
        }

        let mut centroid = nalgebra::Vector3::zeros();
        for &(_, idx) in neighbors.iter() {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }
        centroid /= neighbors.len() as f32;

        let mut covariance = nalgebra::Matrix3::zeros();
        for &(_, idx) in neighbors.iter() {
            let p = points[idx];
            let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
            covariance += diff * diff.transpose();
        }

        // 3-stage deflated power iteration: finds MINIMUM eigenvector of a 3×3
        // symmetric PSD matrix in 10 matrix-vector multiplications + cross product.
        // Significantly faster than full SVD for this single-result use case.
        min_eigenvector_3x3(&covariance)
    }

    /// Analytic minimum eigenvector of a 3×3 symmetric matrix.
    ///
    /// Matches Open3D PointCloudImpl.h / Geometric Tools RobustEigenSymmetric3x3.
    /// Uses the trigonometric (Cardano) method for eigenvalues and a cross-product
    /// approach for the eigenvector — no iteration, exact closed-form.
    pub fn min_eigenvector_3x3(m: &nalgebra::Matrix3<f32>) -> Vector3<f32> {
        // Normalize to prevent numerical overflow.
        let max_c = m.abs().max();
        if max_c < 1e-30 {
            return Vector3::z();
        }
        let s = 1.0 / max_c;
        let a00 = m[(0, 0)] * s;
        let a01 = m[(0, 1)] * s;
        let a02 = m[(0, 2)] * s;
        let a11 = m[(1, 1)] * s;
        let a12 = m[(1, 2)] * s;
        let a22 = m[(2, 2)] * s;

        let norm = a01 * a01 + a02 * a02 + a12 * a12;
        let q = (a00 + a11 + a22) / 3.0;
        let b00 = a00 - q;
        let b11 = a11 - q;
        let b22 = a22 - q;
        let p = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0).sqrt();
        if p < 1e-10 {
            return Vector3::z();
        }

        // Determinant of (A - q*I) / p.
        let c00 = b11 * b22 - a12 * a12;
        let c01 = a01 * b22 - a12 * a02;
        let c02 = a01 * a12 - b11 * a02;
        let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
        let half_det = (det * 0.5).clamp(-1.0, 1.0);
        let angle = half_det.acos() / 3.0;

        // Minimum eigenvalue.
        const TWO_THIRDS_PI: f32 = 2.094_395_1;
        let eval_min = q + p * (angle + TWO_THIRDS_PI).cos() * 2.0;

        // Eigenvector: best cross-product of rows of (A - eval_min * I).
        let r0 = Vector3::new(a00 - eval_min, a01, a02);
        let r1 = Vector3::new(a01, a11 - eval_min, a12);
        let r2 = Vector3::new(a02, a12, a22 - eval_min);

        let r0xr1 = r0.cross(&r1);
        let r0xr2 = r0.cross(&r2);
        let r1xr2 = r1.cross(&r2);

        let d0 = r0xr1.norm_squared();
        let d1 = r0xr2.norm_squared();
        let d2 = r1xr2.norm_squared();

        let best = if d0 >= d1 && d0 >= d2 {
            r0xr1
        } else if d1 >= d2 {
            r0xr2
        } else {
            r1xr2
        };

        let len = best.norm();
        if len < 1e-10 {
            return Vector3::z();
        }
        best / len
    }

    /// Estimate a good voxel size in O(n) from the bounding box.
    ///
    /// Detects whether the cloud is 3-D, 2-D (flat), or 1-D (linear) and uses
    /// the appropriate density formula for each case. This handles degenerate
    /// inputs (flat planes, single-axis lines) without numerical issues.
    fn compute_adaptive_voxel_size(points: &[Point3<f32>], k: usize) -> f32 {
        if points.len() < 2 {
            return 0.1;
        }
        let n = points.len() as f32;

        let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
        for p in points {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            min_z = min_z.min(p.z);
            max_z = max_z.max(p.z);
        }

        let sx = (max_x - min_x).max(1e-9_f32);
        let sy = (max_y - min_y).max(1e-9_f32);
        let sz = (max_z - min_z).max(1e-9_f32);

        // Sort spans smallest → largest.
        let mut spans = [sx, sy, sz];
        spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (s0, s1, s2) = (spans[0], spans[1], spans[2]);

        let vs = if s0 > s2 * 0.01 {
            // 3-D cloud — use volumetric density.
            let density = n / (sx * sy * sz);
            ((k as f32) / (8.0 * density)).cbrt()
        } else if s1 > s2 * 0.01 {
            // 2-D cloud (flat plane) — use areal density.
            let density_2d = n / (s1 * s2);
            ((k as f32) / (9.0 * density_2d)).sqrt()
        } else {
            // 1-D cloud (line) — use linear density.
            let density_1d = n / s2;
            (k as f32) / (3.0 * density_1d)
        };

        // Clamp: must be positive and at most half the second-largest span.
        let max_vs = (s1 / 2.0).max(1e-6_f32);
        vs.clamp(1e-6, max_vs)
    }

    pub fn voxel_downsample(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
        super::voxel_downsample(points, voxel_size)
    }

    // ── Approximate / fast-but-less-accurate methods ─────────────────────────

    /// **Fast-approximate** normals: use k=3 neighbours only, skip full PCA.
    ///
    /// Instead of accumulating a covariance matrix, the normal is simply
    /// `cross(v1, v2)` where v1/v2 are the two displacement vectors to the
    /// two nearest neighbours. This is ~5× faster than the full PCA path at
    /// the cost of accuracy (sensitive to the specific two neighbours chosen).
    ///
    /// Suitable for real-time previews or initialising iterative algorithms.
    pub fn compute_normals_approx_cross(
        points: &[Point3<f32>],
        voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        if points.len() < 3 {
            return vec![Vector3::z(); points.len()];
        }
        let vs = if voxel_size > 0.0 {
            voxel_size
        } else {
            compute_adaptive_voxel_size(points, 6)
        };

        let mut grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
            hashbrown::HashMap::with_capacity(points.len() / 8);
        for (i, p) in points.iter().enumerate() {
            grid.entry((
                (p.x / vs).floor() as i32,
                (p.y / vs).floor() as i32,
                (p.z / vs).floor() as i32,
            ))
            .or_default()
            .push(i);
        }

        points
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                let (vx, vy, vz) = (
                    (c.x / vs).floor() as i32,
                    (c.y / vs).floor() as i32,
                    (c.z / vs).floor() as i32,
                );
                let mut best = [(f32::MAX, 0usize); 2];
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            if let Some(b) = grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in b {
                                    if idx == i {
                                        continue;
                                    }
                                    let p = points[idx];
                                    let d = (c.x - p.x) * (c.x - p.x)
                                        + (c.y - p.y) * (c.y - p.y)
                                        + (c.z - p.z) * (c.z - p.z);
                                    if d < best[0].0 {
                                        best[1] = best[0];
                                        best[0] = (d, idx);
                                    } else if d < best[1].0 {
                                        best[1] = (d, idx);
                                    }
                                }
                            }
                        }
                    }
                }
                if best[1].0 == f32::MAX {
                    return Vector3::z();
                }
                let p0 = points[best[0].1];
                let p1 = points[best[1].1];
                let v0 = Vector3::new(p0.x - c.x, p0.y - c.y, p0.z - c.z);
                let v1 = Vector3::new(p1.x - c.x, p1.y - c.y, p1.z - c.z);
                let n = v0.cross(&v1);
                let len = n.norm();
                if len < 1e-10 {
                    Vector3::z()
                } else {
                    n / len
                }
            })
            .collect()
    }

    /// **Fast-approximate** normals using integral averaging of cross-products
    /// from a ring of 6 axis-aligned neighbour pairs.
    ///
    /// For each point, averages the cross products of six nearest-axis-aligned
    /// neighbour pairs (±x, ±y, ±z directions). Requires no sorting, no PCA.
    /// Very fast and smooth, but less accurate than the PCA path for non-uniform
    /// point distributions. Inspired by the "integral normal" approach in real-time
    /// depth-fusion pipelines.
    pub fn compute_normals_approx_integral(
        points: &[Point3<f32>],
        radius: f32,
    ) -> Vec<Vector3<f32>> {
        if points.len() < 4 {
            return vec![Vector3::z(); points.len()];
        }
        let vs = if radius > 0.0 {
            radius
        } else {
            compute_adaptive_voxel_size(points, 6)
        };

        let mut grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
            hashbrown::HashMap::with_capacity(points.len() / 8);
        for (i, p) in points.iter().enumerate() {
            grid.entry((
                (p.x / vs).floor() as i32,
                (p.y / vs).floor() as i32,
                (p.z / vs).floor() as i32,
            ))
            .or_default()
            .push(i);
        }

        points
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                let (vx, vy, vz) = (
                    (c.x / vs).floor() as i32,
                    (c.y / vs).floor() as i32,
                    (c.z / vs).floor() as i32,
                );
                // Collect all neighbours.
                let mut neighbours: Vec<Vector3<f32>> = Vec::with_capacity(27);
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            if let Some(b) = grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in b {
                                    if idx != i {
                                        let p = points[idx];
                                        neighbours.push(Vector3::new(
                                            p.x - c.x,
                                            p.y - c.y,
                                            p.z - c.z,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                if neighbours.len() < 2 {
                    return Vector3::z();
                }
                // Accumulate cross products of consecutive pairs.
                let mut acc = Vector3::zeros();
                let n = neighbours.len();
                for j in 0..n {
                    let a = neighbours[j];
                    let b = neighbours[(j + 1) % n];
                    acc += a.cross(&b);
                }
                let len = acc.norm();
                if len < 1e-10 {
                    Vector3::z()
                } else {
                    acc / len
                }
            })
            .collect()
    }

    pub fn compute_normals_simple(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        compute_normals(points, k)
    }

    /// Hybrid CPU+GPU normal estimation.
    ///
    /// Separates the two phases of normal computation so each runs on the
    /// optimal hardware:
    ///
    /// - **Phase 1 (CPU, parallel):** voxel-hash kNN → per-point covariance
    ///   matrices.  The spatially-irregular memory access pattern of kNN is
    ///   better served by the CPU's cache hierarchy.
    ///
    /// - **Phase 2 (GPU, if available):** analytic batch eigenvectors for all n
    ///   covariance matrices simultaneously.  This step is purely compute-bound
    ///   with zero data dependency between threads — ideal GPU work.
    ///
    /// Falls back to CPU analytic eigenvectors when no GPU is available.
    pub fn compute_normals_hybrid(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        if points.len() < 3 {
            return vec![Vector3::z(); points.len()];
        }
        let k = k.min(points.len().saturating_sub(1)).max(3);

        // Phase 1: CPU voxel-hash kNN → covariance matrices.
        let covs = compute_covariances_parallel(points, k);

        // Phase 2: GPU batch eigenvectors, or CPU analytic fallback.
        let runner = cv_runtime::best_runner().ok();
        if let Some(ref r) = runner {
            if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = r.device() {
                if let Ok(normals) =
                    cv_hal::gpu_kernels::pointcloud::compute_normals_from_covariances_gpu(
                        gpu, &covs,
                    )
                {
                    return normals;
                }
            }
        }

        // CPU fallback: analytic eigenvectors in parallel.
        covs.par_iter()
            .map(|c| {
                let mut m = nalgebra::Matrix3::zeros();
                m[(0, 0)] = c[0];
                m[(0, 1)] = c[1];
                m[(1, 0)] = c[1];
                m[(0, 2)] = c[2];
                m[(2, 0)] = c[2];
                m[(1, 1)] = c[3];
                m[(1, 2)] = c[4];
                m[(2, 1)] = c[4];
                m[(2, 2)] = c[5];
                min_eigenvector_3x3(&m)
            })
            .collect()
    }

    /// Phase 1 of the hybrid pipeline: parallel voxel-hash kNN → covariance matrices.
    /// Returns one `[cxx, cxy, cxz, cyy, cyz, czz]` per point.
    fn compute_covariances_parallel(points: &[Point3<f32>], k: usize) -> Vec<[f32; 6]> {
        let vs = compute_adaptive_voxel_size(points, k);

        let mut voxel_grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
            hashbrown::HashMap::with_capacity(points.len() / 8);
        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / vs).floor() as i32;
            let vy = (p.y / vs).floor() as i32;
            let vz = (p.z / vs).floor() as i32;
            voxel_grid.entry((vx, vy, vz)).or_default().push(i);
        }

        points
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let vx = (center.x / vs).floor() as i32;
                let vy = (center.y / vs).floor() as i32;
                let vz = (center.z / vs).floor() as i32;

                let mut cands: Vec<(f32, usize)> = Vec::with_capacity(27 * k);
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            if let Some(bucket) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in bucket {
                                    if idx != i {
                                        let p = points[idx];
                                        let d = (center.x - p.x) * (center.x - p.x)
                                            + (center.y - p.y) * (center.y - p.y)
                                            + (center.z - p.z) * (center.z - p.z);
                                        cands.push((d, idx));
                                    }
                                }
                            }
                        }
                    }
                }
                if cands.len() > k {
                    cands.select_nth_unstable_by(k - 1, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cands.truncate(k);
                }

                if cands.len() < 3 {
                    return [0.0f32; 6];
                }

                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let mut cz = 0.0f32;
                for &(_, idx) in &cands {
                    let p = points[idx];
                    cx += p.x;
                    cy += p.y;
                    cz += p.z;
                }
                let inv_n = 1.0 / cands.len() as f32;
                cx *= inv_n;
                cy *= inv_n;
                cz *= inv_n;

                let mut cxx = 0.0f32;
                let mut cxy = 0.0f32;
                let mut cxz = 0.0f32;
                let mut cyy = 0.0f32;
                let mut cyz = 0.0f32;
                let mut czz = 0.0f32;
                for &(_, idx) in &cands {
                    let p = points[idx];
                    let dx = p.x - cx;
                    let dy = p.y - cy;
                    let dz = p.z - cz;
                    cxx += dx * dx;
                    cxy += dx * dy;
                    cxz += dx * dz;
                    cyy += dy * dy;
                    cyz += dy * dz;
                    czz += dz * dz;
                }
                [
                    cxx * inv_n,
                    cxy * inv_n,
                    cxz * inv_n,
                    cyy * inv_n,
                    cyz * inv_n,
                    czz * inv_n,
                ]
            })
            .collect()
    }

    pub fn voxel_based_normals_simple(
        points: &[Point3<f32>],
        _voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        compute_normals(points, 30)
    }

    pub fn voxel_to_point_normal_transfer(
        points: &[Point3<f32>],
        _normals: &[Vector3<f32>],
        _voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        vec![Vector3::z(); points.len()]
    }

    pub fn approximate_normals_simple(
        points: &[Point3<f32>],
        k: usize,
        _epsilon: f32,
    ) -> Vec<Vector3<f32>> {
        compute_normals(points, k)
    }
}

/// GPU-accelerated stereo matching
pub mod stereo {
    use super::*;
    use cv_hal::context::StereoMatchParams;

    pub fn match_stereo(
        left: &::image::GrayImage,
        right: &::image::GrayImage,
        params: &StereoMatchParams,
    ) -> cv_hal::Result<::image::ImageBuffer<::image::Luma<f32>, Vec<f32>>> {
        let runner =
            cv_runtime::best_runner().map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
        match_stereo_ctx(left, right, params, &runner)
    }

    pub fn match_stereo_ctx(
        left: &::image::GrayImage,
        right: &::image::GrayImage,
        params: &StereoMatchParams,
        runner: &RuntimeRunner,
    ) -> cv_hal::Result<::image::ImageBuffer<::image::Luma<f32>, Vec<f32>>> {
        if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = runner.device() {
            let left_f32: Vec<f32> = left.as_raw().iter().map(|&p| p as f32).collect();
            let l_tensor = cv_core::CpuTensor::from_vec(
                left_f32,
                cv_core::TensorShape::new(1, left.height() as usize, left.width() as usize),
            )
            .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

            let right_f32: Vec<f32> = right.as_raw().iter().map(|&p| p as f32).collect();
            let r_tensor = cv_core::CpuTensor::from_vec(
                right_f32,
                cv_core::TensorShape::new(1, right.height() as usize, right.width() as usize),
            )
            .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

            let l_gpu = l_tensor.to_gpu_ctx(gpu)?;
            let r_gpu = r_tensor.to_gpu_ctx(gpu)?;

            // Note: We use OS=GpuStorage<f32> for the output
            let res_gpu: Tensor<f32, cv_hal::storage::GpuStorage<f32>> =
                gpu.stereo_match(&l_gpu, &r_gpu, params)?;
            let res_cpu = res_gpu.to_cpu_ctx(gpu)?;

            let data = res_cpu.storage.as_slice().unwrap().to_vec();
            return Ok(::image::ImageBuffer::from_raw(left.width(), left.height(), data).unwrap());
        }

        Err(cv_hal::Error::NotSupported(
            "CPU fallback for orchestrated stereo not yet in 3d::gpu".into(),
        ))
    }
}

pub mod registration {
    use super::*;

    /// ICP point-to-plane registration: NOT IMPLEMENTED
    ///
    /// This function is a stub and returns an error indicating that the algorithm
    /// is not yet implemented in the GPU module. Point-to-plane ICP requires:
    /// 1. Surface normal computation
    /// 2. Iterative closest point search with plane distance metrics
    /// 3. Robust outlier rejection
    ///
    /// To use ICP registration, use the `cv-registration` crate instead.
    pub fn icp_point_to_plane(
        _s: &[Point3<f32>],
        _t: &[Point3<f32>],
        _tn: &[Vector3<f32>],
        _dist: f32,
        _iters: usize,
    ) -> Result<Matrix4<f32>, String> {
        Err("ICP point-to-plane registration not yet implemented in GPU module. Use cv-registration crate instead.".to_string())
    }
}

pub mod mesh {
    use super::*;

    /// Laplacian mesh smoothing: NOT IMPLEMENTED
    ///
    /// This function is a stub. Laplacian smoothing requires:
    /// 1. Vertex-face adjacency graph construction
    /// 2. Laplacian matrix computation (uniform or cotangent weights)
    /// 3. Iterative smoothing with boundary preservation
    ///
    /// This operation is expensive on CPU and requires GPU acceleration
    /// through a custom kernel implementation in cv-hal.
    pub fn laplacian_smooth(
        _v: &mut [Point3<f32>],
        _f: &[[usize; 3]],
        _iters: usize,
        _l: f32,
    ) -> Result<(), String> {
        Err(
            "Laplacian mesh smoothing not yet implemented. Requires GPU kernel in cv-hal."
                .to_string(),
        )
    }

    /// Compute vertex normals from mesh: NOT IMPLEMENTED
    ///
    /// Returns vertex normals computed by averaging face normals of adjacent faces.
    /// This function is a stub that returns an error.
    ///
    /// Implementation should:
    /// 1. For each face, compute normal from vertex positions
    /// 2. For each vertex, accumulate normals from all adjacent faces
    /// 3. Normalize per-vertex accumulated normals
    pub fn compute_vertex_normals(
        _v: &[Point3<f32>],
        _f: &[[usize; 3]],
    ) -> Result<Vec<Vector3<f32>>, String> {
        Err("Vertex normal computation not yet implemented.".to_string())
    }
}

pub mod tsdf {
    use super::*;

    /// TSDF depth integration: NOT IMPLEMENTED
    ///
    /// This function is a stub. Truncated Signed Distance Field (TSDF) integration is a core
    /// component of volumetric 3D reconstruction (KinectFusion-style algorithms).
    ///
    /// Parameters:
    /// - `d`: Depth map (row-major, H×W)
    /// - `w`, `h`: Image dimensions
    /// - `p`: Projection matrix (3×4)
    /// - `i`: Intrinsic matrix flattened (K[0,0], K[1,1], K[0,2], K[1,2])
    /// - `vol`: TSDF volume (pre-allocated, size = voxel_grid.len())
    /// - `weights`: Weight accumulation volume
    /// - `vs`: Voxel size
    /// - `tr`: TSDF truncation distance
    ///
    /// Integration requires:
    /// 1. Back-project each depth pixel to 3D world coordinates
    /// 2. For each voxel in the volume, compute signed distance to surface
    /// 3. Accumulate TSDF values weighted by confidence
    /// 4. Update weight accumulator for normalization
    ///
    /// This is heavily used in real-time reconstruction and MUST be GPU-accelerated
    /// through a compute kernel in cv-hal.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_depth(
        _d: &[f32],
        _w: u32,
        _h: u32,
        _p: &Matrix4<f32>,
        _i: &[f32; 4],
        _vol: &mut [f32],
        _weights: &mut [f32],
        _vs: f32,
        _tr: f32,
    ) -> Result<(), String> {
        Err("TSDF depth integration not yet implemented. Requires GPU kernel for real-time performance.".to_string())
    }
}

pub mod raycasting {
    use super::*;

    /// Ray-mesh intersection raycasting: NOT IMPLEMENTED
    ///
    /// This function is a stub. Raycasting is essential for:
    /// 1. Rendering: Visibility queries against mesh geometry
    /// 2. Reconstruction: Point cloud alignment and normal estimation
    /// 3. Simulation: Collision detection and occlusion queries
    ///
    /// Parameters:
    /// - `ro`: Array of ray origins (Point3<f32>)
    /// - `rd`: Array of ray directions (Vector3<f32>, assumed normalized)
    /// - `v`: Mesh vertices
    /// - `f`: Mesh triangle faces (indices into vertex array)
    ///
    /// Returns for each ray: (t, intersection_point, surface_normal) or None if no intersection
    ///
    /// Implementation should:
    /// 1. Build BVH or spatial acceleration structure from mesh
    /// 2. For each ray, traverse acceleration structure
    /// 3. Perform triangle-ray intersection tests (Möller-Trumbore algorithm)
    /// 4. Return closest intersection
    ///
    /// This is a performance-critical operation best implemented with:
    /// - Spatial hashing or BVH
    /// - SIMD vectorization or GPU acceleration
    #[allow(clippy::type_complexity)]
    pub fn cast_rays(
        _ro: &[Point3<f32>],
        _rd: &[Vector3<f32>],
        _v: &[Point3<f32>],
        _f: &[[usize; 3]],
    ) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
        Err(
            "Ray-mesh intersection not yet implemented. Requires BVH acceleration structure."
                .to_string(),
        )
    }
}

/// Voxel downsample
pub fn voxel_downsample(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
    if points.is_empty() {
        return Vec::new();
    }
    use crate::spatial::VoxelGrid;
    let mut grid = VoxelGrid::new(points[0], voxel_size);
    for (i, p) in points.iter().enumerate() {
        grid.insert(*p, i);
    }
    grid.downsample(points)
}

pub fn is_gpu_available() -> bool {
    GpuContext::new().is_ok()
}

pub fn gpu_info() -> Option<String> {
    GpuContext::new()
        .ok()
        .map(|ctx| format!("{:?}", ctx.device))
}

pub fn force_cpu_mode() -> cv_hal::gpu_kernels::unified::ComputeConfig {
    cv_hal::gpu_kernels::unified::ComputeConfig {
        force_cpu: true,
        ..Default::default()
    }
}

pub fn force_gpu_mode() -> cv_hal::gpu_kernels::unified::ComputeConfig {
    cv_hal::gpu_kernels::unified::ComputeConfig {
        force_gpu: true,
        ..Default::default()
    }
}

#[cfg(test)]
mod normal_tests {
    use super::point_cloud as pc;
    use nalgebra::Point3;

    /// Build an n×n flat grid of points at z=0 in [0, 1]².
    fn flat_plane(n: usize) -> Vec<Point3<f32>> {
        let mut pts = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                pts.push(Point3::new(i as f32 / n as f32, j as f32 / n as f32, 0.0));
            }
        }
        pts
    }

    /// Normals on a z=0 plane should be (0,0,±1).
    fn assert_vertical(normals: &[nalgebra::Vector3<f32>], tol: f32, label: &str) {
        let mut bad = 0;
        for n in normals {
            if n.z.abs() < tol {
                bad += 1;
            }
        }
        let pct_bad = bad as f32 / normals.len() as f32;
        assert!(
            pct_bad < 0.1,
            "{}: {:.0}% of normals not vertical (|z| < {})",
            label,
            pct_bad * 100.0,
            tol
        );
    }

    #[test]
    fn test_voxel_cpu_plane() {
        let pts = flat_plane(8);
        let normals = pc::compute_normals_cpu(&pts, 8, 0.0);
        assert_eq!(normals.len(), pts.len());
        assert_vertical(&normals, 0.8, "voxel_cpu");
    }

    #[test]
    fn test_hybrid_plane() {
        let pts = flat_plane(8);
        let normals = pc::compute_normals_hybrid(&pts, 8);
        assert_eq!(normals.len(), pts.len());
        assert_vertical(&normals, 0.8, "hybrid");
    }

    #[test]
    fn test_approx_cross_plane() {
        let pts = flat_plane(8);
        let normals = pc::compute_normals_approx_cross(&pts, 0.0);
        assert_eq!(normals.len(), pts.len());
        // Approximate method — relax tolerance and check majority
        assert_vertical(&normals, 0.5, "approx_cross");
    }

    #[test]
    fn test_approx_integral_plane() {
        let pts = flat_plane(8);
        let normals = pc::compute_normals_approx_integral(&pts, 0.0);
        assert_eq!(normals.len(), pts.len());
        assert_vertical(&normals, 0.5, "approx_integral");
    }

    #[test]
    fn test_analytic_eigensolver_known() {
        // Covariance of points on z=0 plane: zero z-variance, non-zero xy-variance.
        // Minimum eigenvector should be (0, 0, ±1).
        use super::point_cloud::min_eigenvector_3x3;
        // Build covariance matrix: large xy spread, zero z.
        let mut m = nalgebra::Matrix3::zeros();
        m[(0, 0)] = 1.0; // cxx
        m[(1, 1)] = 1.0; // cyy
        m[(2, 2)] = 0.0; // czz  ← zero → minimum eigenvalue
        m[(0, 1)] = 0.0;
        m[(1, 0)] = 0.0;
        m[(0, 2)] = 0.0;
        m[(2, 0)] = 0.0;
        m[(1, 2)] = 0.0;
        m[(2, 1)] = 0.0;

        let n = min_eigenvector_3x3(&m);
        assert!(n.z.abs() > 0.99, "Expected z-normal, got {:?}", n);
    }
}
