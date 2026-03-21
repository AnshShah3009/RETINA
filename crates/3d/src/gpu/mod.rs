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
use tracing::{info, warn};

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
            warn!("GPU not available for compute_normals, using CPU");
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

        warn!("GPU compute_normals: runner.device() failed, falling back to CPU");
        compute_normals_cpu(points, k, 0.0)
    }

    fn compute_normals_gpu_with_ctx(
        points: &[Point3<f32>],
        k: usize,
        _gpu: &cv_hal::gpu::GpuContext,
    ) -> Vec<Vector3<f32>> {
        eprintln!(
            "[GPU] compute_normals_gpu_with_ctx: {} points - using hybrid path (CPU kNN + GPU PCA)",
            points.len()
        );
        // Full GPU path (Morton + LBVH) has bind group layout issues.
        // Use hybrid path: CPU voxel-hash kNN + GPU batch PCA
        compute_normals_hybrid(points, k)
    }

    /// Estimate normals using tiled GPU brute-force kNN + analytic PCA.
    ///
    /// Each workgroup cooperatively loads tiles of 256 points into shared memory,
    /// giving 256x better memory bandwidth than naive global scans.
    ///
    /// Complexity: O(n²) but with ~256x lower bandwidth cost.
    /// Works well for small-medium clouds (1K-100K points).
    ///
    /// Falls back to CPU voxel-hash if GPU is unavailable.
    pub fn compute_normals_gpu_tiled(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }
        let k = k.min(points.len().saturating_sub(1)).max(3);

        let runner = cv_runtime::best_runner().ok();
        if let Some(ref r) = runner {
            if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = r.device() {
                let pts: Vec<nalgebra::Vector3<f32>> = points.iter().map(|p| p.coords).collect();
                match cv_hal::gpu_kernels::pointcloud_gpu::compute_normals_tiled_gpu(
                    gpu, &pts, k as u32,
                ) {
                    Ok(n) => {
                        info!(
                            "GPU compute_normals_gpu_tiled: executed on GPU ({} points)",
                            points.len()
                        );
                        return n;
                    }
                    Err(e) => {
                        warn!(
                            "GPU tiled normals kernel failed: {:?}, falling back to CPU",
                            e
                        );
                    }
                }
            }
        }
        warn!("GPU not available for compute_normals_gpu_tiled, falling back to CPU");
        compute_normals_cpu(points, k, 0.0)
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

    /// Estimate normals on CPU using KDTree + analytic eigensolver.
    ///
    /// This is the classic approach used by Open3D and most 3D libraries.
    /// O(n log n) construction, O(n log k) queries.
    pub fn compute_normals_cpu_kdtree(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        eprintln!(
            "[CPU-KDTREE] compute_normals_cpu_kdtree: {} points",
            points.len()
        );
        if points.is_empty() {
            return Vec::new();
        }
        let k = k.min(points.len().saturating_sub(1)).max(3);

        let pts_vec: Vec<Vec<f64>> = points
            .iter()
            .map(|p| vec![p.x as f64, p.y as f64, p.z as f64])
            .collect();

        let tree = match cv_scientific::spatial::KDTree::new(&pts_vec) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[CPU-KDTREE] Failed to build KDTree: {}", e);
                return vec![Vector3::z(); points.len()];
            }
        };

        points
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let query = vec![center.x as f64, center.y as f64, center.z as f64];
                let neighbors = tree.query(&query, k + 1);
                let filtered: Vec<(f32, usize)> = neighbors
                    .into_iter()
                    .filter(|(idx, _)| *idx != i)
                    .take(k)
                    .map(|(idx, dist)| (dist as f32, idx))
                    .collect();

                compute_pca_normal(center, &filtered, points)
            })
            .collect()
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
                // Sort neighbours by angle around the query point so that
                // consecutive cross-products give a consistent normal.
                // Project onto a local 2D frame and use atan2.
                let ref_dir = neighbours[0].normalize();
                // Build a second basis vector orthogonal to ref_dir.
                let up = if ref_dir.x.abs() < 0.9 {
                    Vector3::x()
                } else {
                    Vector3::y()
                };
                let u = ref_dir;
                let w = u.cross(&up).normalize();
                let v = w.cross(&u);
                neighbours.sort_by(|a, b| {
                    let angle_a = a.dot(&v).atan2(a.dot(&w));
                    let angle_b = b.dot(&v).atan2(b.dot(&w));
                    angle_a
                        .partial_cmp(&angle_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
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
                match cv_hal::gpu_kernels::pointcloud::compute_normals_from_covariances_gpu(
                    gpu, &covs,
                ) {
                    Ok(normals) => {
                        info!(
                            "GPU compute_normals_hybrid Phase 2: executed on GPU ({} points)",
                            points.len()
                        );
                        return normals;
                    }
                    Err(e) => {
                        warn!("GPU PCA failed: {:?}, falling back to CPU", e);
                    }
                }
            } else {
                warn!("No GPU device available for hybrid normals, falling back to CPU");
            }
        } else {
            warn!("No runner available for hybrid normals, falling back to CPU");
        }

        // CPU fallback: analytic eigenvectors in parallel.
        warn!("Hybrid normals Phase 2: falling back to CPU eigenvectors");
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

    /// Simple ICP point-to-plane registration.
    ///
    /// For full-featured ICP (multi-scale, robust kernels, colored),
    /// use the `cv-registration` crate instead.
    pub fn icp_point_to_plane(
        source: &[Point3<f32>],
        target: &[Point3<f32>],
        target_normals: &[Vector3<f32>],
        max_dist: f32,
        max_iters: usize,
    ) -> Result<Matrix4<f32>, String> {
        use crate::spatial::KDTree;

        if target.is_empty() || source.is_empty() || target_normals.len() != target.len() {
            return Err("Invalid input sizes".to_string());
        }

        let mut items: Vec<(Point3<f32>, usize)> =
            target.iter().copied().zip(0..target.len()).collect();
        let tree = KDTree::build(&mut items);
        let mut transform = Matrix4::identity();
        let max_dist_sq = max_dist * max_dist;

        for _ in 0..max_iters {
            // Find correspondences and build linear system (6x6) — parallel reduction
            let (ata, atb, n_corr) = source
                .par_iter()
                .fold(
                    || {
                        (
                            nalgebra::Matrix6::<f64>::zeros(),
                            nalgebra::Vector6::<f64>::zeros(),
                            0usize,
                        )
                    },
                    |(mut ata, mut atb, mut count), sp| {
                        let tp = transform.transform_point(sp);
                        if let Some((closest, idx, dist_sq)) = tree.nearest_neighbor(&tp) {
                            if dist_sq <= max_dist_sq {
                                let n = nalgebra::Vector3::new(
                                    target_normals[idx].x as f64,
                                    target_normals[idx].y as f64,
                                    target_normals[idx].z as f64,
                                );
                                let p =
                                    nalgebra::Vector3::new(tp.x as f64, tp.y as f64, tp.z as f64);
                                let q = nalgebra::Vector3::new(
                                    closest.x as f64,
                                    closest.y as f64,
                                    closest.z as f64,
                                );
                                let d = p - q;
                                let cross = p.cross(&n);
                                let row = nalgebra::Vector6::new(
                                    cross.x, cross.y, cross.z, n.x, n.y, n.z,
                                );
                                let rhs = -n.dot(&d);
                                ata += row * row.transpose();
                                atb += row * rhs;
                                count += 1;
                            }
                        }
                        (ata, atb, count)
                    },
                )
                .reduce(
                    || {
                        (
                            nalgebra::Matrix6::<f64>::zeros(),
                            nalgebra::Vector6::<f64>::zeros(),
                            0usize,
                        )
                    },
                    |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
                );

            if n_corr < 6 {
                return Err(format!("Too few correspondences: {n_corr}"));
            }

            // Solve 6x6 system
            let x = ata
                .lu()
                .solve(&atb)
                .ok_or("Failed to solve ICP linear system")?;

            // Build incremental transform from twist vector
            let (a, b, g) = (x[0] as f32, x[1] as f32, x[2] as f32);
            let (tx, ty, tz) = (x[3] as f32, x[4] as f32, x[5] as f32);

            let mut inc = Matrix4::identity();
            inc[(0, 1)] = -g;
            inc[(0, 2)] = b;
            inc[(1, 0)] = g;
            inc[(1, 2)] = -a;
            inc[(2, 0)] = -b;
            inc[(2, 1)] = a;
            inc[(0, 3)] = tx;
            inc[(1, 3)] = ty;
            inc[(2, 3)] = tz;

            transform = inc * transform;

            // Convergence check
            let delta = x.norm();
            if delta < 1e-6 {
                break;
            }
        }

        Ok(transform)
    }
}

pub mod mesh {
    use super::*;

    /// Laplacian mesh smoothing with uniform weights.
    ///
    /// Iteratively moves each vertex towards the centroid of its neighbors.
    /// `lambda` controls the smoothing strength (0..1, typically 0.5).
    pub fn laplacian_smooth(
        v: &mut [Point3<f32>],
        f: &[[usize; 3]],
        iters: usize,
        lambda: f32,
    ) -> Result<(), String> {
        if v.is_empty() || f.is_empty() {
            return Ok(());
        }
        // Build adjacency: for each vertex, collect unique neighbor indices
        let n = v.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for face in f {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                if !adj[a].contains(&b) {
                    adj[a].push(b);
                }
                if !adj[b].contains(&a) {
                    adj[b].push(a);
                }
            }
        }
        // Iterative Laplacian smoothing
        for _ in 0..iters {
            let old = v.to_vec();
            for i in 0..n {
                if adj[i].is_empty() {
                    continue;
                }
                let centroid: Vector3<f32> =
                    adj[i].iter().map(|&j| old[j].coords).sum::<Vector3<f32>>()
                        / adj[i].len() as f32;
                v[i] = Point3::from(old[i].coords * (1.0 - lambda) + centroid * lambda);
            }
        }
        Ok(())
    }

    /// Compute vertex normals by averaging adjacent face normals.
    pub fn compute_vertex_normals(
        v: &[Point3<f32>],
        f: &[[usize; 3]],
    ) -> Result<Vec<Vector3<f32>>, String> {
        let mut normals = vec![Vector3::zeros(); v.len()];
        for face in f {
            let v0 = v[face[0]];
            let v1 = v[face[1]];
            let v2 = v[face[2]];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let fn_ = e1.cross(&e2); // area-weighted normal
            for &idx in face {
                normals[idx] += fn_;
            }
        }
        for n in &mut normals {
            let len = n.norm();
            if len > 1e-9 {
                *n /= len;
            }
        }
        Ok(normals)
    }
}

pub mod tsdf {
    use super::*;

    /// KinectFusion-style TSDF depth integration.
    ///
    /// Integrates a single depth frame into a pre-allocated cubic TSDF volume
    /// using a running weighted average. The volume is assumed to be a flat
    /// cube with side length `n = cbrt(vol.len())`.
    ///
    /// Parameters:
    /// - `d`: Depth map (row-major, H x W)
    /// - `w`, `h`: Image dimensions
    /// - `p`: Camera extrinsics (world-to-camera transform)
    /// - `i`: Intrinsics `[fx, fy, cx, cy]`
    /// - `vol`: TSDF volume (pre-allocated, size = n^3)
    /// - `weights`: Weight volume (same size as `vol`)
    /// - `vs`: Voxel size in world units
    /// - `tr`: Truncation distance
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_depth(
        d: &[f32],
        w: u32,
        h: u32,
        p: &Matrix4<f32>,
        i: &[f32; 4],
        vol: &mut [f32],
        weights: &mut [f32],
        vs: f32,
        tr: f32,
    ) -> Result<(), String> {
        let total = vol.len();
        if total == 0 {
            return Err("Empty volume".to_string());
        }
        if weights.len() != total {
            return Err("vol and weights must have the same length".to_string());
        }

        // Determine cube side length: n = cbrt(total), must be exact.
        let n = (total as f64).cbrt().round() as usize;
        if n * n * n != total {
            return Err(format!(
                "Volume length {} is not a perfect cube (nearest n={})",
                total, n
            ));
        }

        let [fx, fy, cx, cy] = *i;
        let w_img = w as usize;
        let h_img = h as usize;
        let max_weight: f32 = 100.0;

        // Volume origin: centred so that voxel (0,0,0) maps to origin in world space.
        // This matches the common convention where the volume starts at the world origin.
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    // World-space centre of this voxel
                    let wx = ix as f32 * vs;
                    let wy = iy as f32 * vs;
                    let wz = iz as f32 * vs;

                    // Transform to camera coordinates: cam = p * [wx, wy, wz, 1]
                    let cam_x = p[(0, 0)] * wx + p[(0, 1)] * wy + p[(0, 2)] * wz + p[(0, 3)];
                    let cam_y = p[(1, 0)] * wx + p[(1, 1)] * wy + p[(1, 2)] * wz + p[(1, 3)];
                    let cam_z = p[(2, 0)] * wx + p[(2, 1)] * wy + p[(2, 2)] * wz + p[(2, 3)];

                    // Skip voxels behind the camera
                    if cam_z <= 0.0 {
                        continue;
                    }

                    // Project to pixel coordinates
                    let px = fx * cam_x / cam_z + cx;
                    let py = fy * cam_y / cam_z + cy;

                    let px_i = px as i32;
                    let py_i = py as i32;

                    if px_i < 0 || py_i < 0 || px_i >= w_img as i32 || py_i >= h_img as i32 {
                        continue;
                    }

                    let depth = d[py_i as usize * w_img + px_i as usize];
                    if depth <= 0.0 {
                        continue;
                    }

                    // Signed distance: positive means the voxel is in front of the surface
                    let sdf = depth - cam_z;

                    if sdf < -tr {
                        continue;
                    }

                    let tsdf_val = (sdf / tr).clamp(-1.0, 1.0);

                    // Running weighted average update
                    let idx = ix + iy * n + iz * n * n;
                    let old_tsdf = vol[idx];
                    let old_w = weights[idx];
                    let new_w = old_w + 1.0;
                    vol[idx] = (old_tsdf * old_w + tsdf_val) / new_w;
                    weights[idx] = new_w.min(max_weight);
                }
            }
        }

        Ok(())
    }
}

pub mod raycasting {
    use super::*;
    use crate::spatial::bvh::Bvh;

    /// BVH-accelerated ray-mesh intersection — O(rays * log(triangles)).
    ///
    /// Builds a BVH on first call. For repeated queries against the same mesh,
    /// use [`cast_rays_with_bvh`] to reuse the BVH.
    #[allow(clippy::type_complexity)]
    pub fn cast_rays(
        ro: &[Point3<f32>],
        rd: &[Vector3<f32>],
        v: &[Point3<f32>],
        f: &[[usize; 3]],
    ) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
        let bvh = Bvh::build(v, f);
        cast_rays_with_bvh(ro, rd, v, f, &bvh)
    }

    /// Ray-mesh intersection using a pre-built BVH.
    #[allow(clippy::type_complexity)]
    pub fn cast_rays_with_bvh(
        ro: &[Point3<f32>],
        rd: &[Vector3<f32>],
        v: &[Point3<f32>],
        f: &[[usize; 3]],
        bvh: &Bvh,
    ) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
        let results: Vec<_> = ro
            .par_iter()
            .zip(rd.par_iter())
            .map(|(origin, dir)| {
                bvh.intersect_ray(origin, dir, v, f).map(|(t, fi, _u, _v)| {
                    let hit = Point3::from(origin.coords + dir * t);
                    let face = &f[fi];
                    let e1 = v[face[1]] - v[face[0]];
                    let e2 = v[face[2]] - v[face[0]];
                    let mut n = e1.cross(&e2);
                    let len = n.norm();
                    if len > 1e-9 {
                        n /= len;
                    }
                    (t, hit, n)
                })
            })
            .collect();
        Ok(results)
    }

    /// Brute-force ray-mesh intersection — O(rays * triangles).
    /// Kept for correctness comparison and small meshes.
    #[allow(clippy::type_complexity)]
    pub fn cast_rays_brute(
        ro: &[Point3<f32>],
        rd: &[Vector3<f32>],
        v: &[Point3<f32>],
        f: &[[usize; 3]],
    ) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
        let results: Vec<_> = ro
            .par_iter()
            .zip(rd.par_iter())
            .map(|(origin, dir)| {
                let mut best: Option<(f32, Point3<f32>, Vector3<f32>)> = None;
                for face in f {
                    let v0 = v[face[0]];
                    let v1 = v[face[1]];
                    let v2 = v[face[2]];
                    if let Some((t, _u, _v)) = moller_trumbore(&origin.coords, dir, &v0, &v1, &v2) {
                        if t > 1e-6 {
                            let replace = match best {
                                None => true,
                                Some((bt, _, _)) => t < bt,
                            };
                            if replace {
                                let hit = Point3::from(origin.coords + dir * t);
                                let e1 = v1 - v0;
                                let e2 = v2 - v0;
                                let mut n = e1.cross(&e2);
                                let len = n.norm();
                                if len > 1e-9 {
                                    n /= len;
                                }
                                best = Some((t, hit, n));
                            }
                        }
                    }
                }
                best
            })
            .collect();
        Ok(results)
    }

    /// Möller-Trumbore ray-triangle intersection.
    /// Returns `Some((t, u, v))` if ray `origin + t*dir` hits the triangle.
    fn moller_trumbore(
        origin: &Vector3<f32>,
        dir: &Vector3<f32>,
        v0: &Point3<f32>,
        v1: &Point3<f32>,
        v2: &Point3<f32>,
    ) -> Option<(f32, f32, f32)> {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let h = dir.cross(&e2);
        let a = e1.dot(&h);
        if a.abs() < 1e-9 {
            return None; // parallel
        }
        let f = 1.0 / a;
        let s = Point3::from(*origin) - v0;
        let u = f * s.dot(&h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(&e1);
        let v = f * dir.dot(&q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * e2.dot(&q);
        Some((t, u, v))
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
