use cv_core::storage::Storage;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

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

    let mut spans = [sx, sy, sz];
    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (s0, s1, s2) = (spans[0], spans[1], spans[2]);

    let vs = if s0 > s2 * 0.01 {
        let density = n / (sx * sy * sz);
        ((k as f32) / (8.0 * density)).cbrt()
    } else if s1 > s2 * 0.01 {
        let density_2d = n / (s1 * s2);
        ((k as f32) / (9.0 * density_2d)).sqrt()
    } else {
        let density_1d = n / s2;
        (k as f32) / (3.0 * density_1d)
    };

    let max_vs = (s1 / 2.0).max(1e-6_f32);
    vs.clamp(1e-6, max_vs)
}

fn min_eigenvector_3x3(m: &nalgebra::Matrix3<f32>) -> Vector3<f32> {
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

    let c00 = b11 * b22 - a12 * a12;
    let c01 = a01 * b22 - a12 * a02;
    let c02 = a01 * a12 - b11 * a02;
    let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
    let half_det = (det * 0.5).clamp(-1.0, 1.0);
    let angle = half_det.acos() / 3.0;

    const TWO_THIRDS_PI: f32 = 2.094_395_1;
    let eval_min = q + p * (angle + TWO_THIRDS_PI).cos() * 2.0;

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
        Vector3::z()
    } else {
        best / len
    }
}

fn compute_pca_normal(
    center: &Point3<f32>,
    neighbors: &[(f32, usize)],
    points: &[Point3<f32>],
) -> Vector3<f32> {
    if neighbors.len() < 3 {
        return Vector3::z();
    }

    let mut centroid = nalgebra::Vector3::zeros();
    for &(_, idx) in neighbors {
        let p = points[idx];
        centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
    }
    centroid /= neighbors.len() as f32;

    let mut covariance = nalgebra::Matrix3::zeros();
    for &(_, idx) in neighbors {
        let p = points[idx];
        let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
        covariance += diff * diff.transpose();
    }

    min_eigenvector_3x3(&covariance)
}

pub fn compute_normals(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    let runner = cv_runtime::best_runner()
        .unwrap_or_else(|_| cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0)));
    eprintln!("[GPU] compute_normals: using runner: {:?}", runner);
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
    use cv_hal::gpu_kernels::pointcloud;
    eprintln!("[GPU] compute_normals: {} points on GPU", points.len());
    let num_points = points.len();

    let pts_vec: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let input_tensor =
        match cv_core::CpuTensor::from_vec(pts_vec, cv_core::TensorShape::new(1, num_points, 1)) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "[GPU->CPU] compute_normals: CpuTensor creation failed: {:?}",
                    e
                );
                return compute_normals_cpu(points, k, 0.0);
            }
        };

    let gpu_tensor = match input_tensor.to_gpu_ctx(gpu) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[GPU->CPU] compute_normals: to_gpu_ctx failed: {:?}", e);
            return compute_normals_cpu(points, k, 0.0);
        }
    };

    let result_gpu = match pointcloud::compute_normals(gpu, &gpu_tensor, k as u32) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[GPU->CPU] compute_normals: GPU kernel failed: {:?}", e);
            return compute_normals_cpu(points, k, 0.0);
        }
    };

    let result_cpu: cv_core::Tensor<f32, cv_core::CpuStorage<f32>> =
        match result_gpu.to_cpu_ctx(gpu) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[GPU->CPU] compute_normals: to_cpu_ctx failed: {:?}", e);
                return compute_normals_cpu(points, k, 0.0);
            }
        };

    eprintln!("[GPU] compute_normals: completed on GPU");
    let raw = result_cpu.storage.as_slice().unwrap();
    raw.chunks(4)
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect()
}

pub fn compute_normals_cpu(
    points: &[Point3<f32>],
    k: usize,
    _voxel_size: f32,
) -> Vec<Vector3<f32>> {
    eprintln!("[CPU] compute_normals_cpu: {} points", points.len());
    if points.is_empty() {
        return Vec::new();
    }
    let k = k.min(points.len().saturating_sub(1)).max(3);
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

pub fn compute_normals_hybrid(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    if points.is_empty() {
        return Vec::new();
    }
    let k = k.min(points.len().saturating_sub(1)).max(3);
    let vs = compute_adaptive_voxel_size(points, k);

    let mut voxel_grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
        hashbrown::HashMap::with_capacity(points.len() / 8);
    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / vs).floor() as i32;
        let vy = (p.y / vs).floor() as i32;
        let vz = (p.z / vs).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    let covs: Vec<[f32; 6]> = points
        .par_iter()
        .enumerate()
        .map(|(i, center)| {
            let (vx, vy, vz) = (
                (center.x / vs).floor() as i32,
                (center.y / vs).floor() as i32,
                (center.z / vs).floor() as i32,
            );

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
        .collect();

    eprintln!(
        "[GPU] compute_normals_hybrid: {} covariances computed on CPU",
        covs.len()
    );
    if let Ok(gpu) = cv_hal::gpu::GpuContext::global() {
        eprintln!(
            "[GPU] compute_normals_hybrid: attempting GPU PCA on {} covariances",
            covs.len()
        );
        if let Ok(normals) =
            cv_hal::gpu_kernels::pointcloud::compute_normals_from_covariances_gpu(gpu, &covs)
        {
            eprintln!("[GPU] compute_normals_hybrid: completed on GPU");
            return normals;
        } else {
            eprintln!("[GPU->CPU] compute_normals_hybrid: GPU PCA failed, falling back to CPU");
        }
    } else {
        eprintln!("[GPU->CPU] compute_normals_hybrid: No GPU available, using CPU");
    }

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

pub fn compute_normals_simple(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    compute_normals(points, k)
}

pub fn compute_normals_approx_integral(points: &[Point3<f32>], _radius: f32) -> Vec<Vector3<f32>> {
    vec![Vector3::z(); points.len()]
}

pub fn voxel_downsample(points: &[Point3<f32>], _voxel_size: f32) -> Vec<Point3<f32>> {
    points.to_vec()
}

pub fn voxel_based_normals_simple(points: &[Point3<f32>], _voxel_size: f32) -> Vec<Vector3<f32>> {
    vec![Vector3::z(); points.len()]
}

pub fn voxel_to_point_normal_transfer(
    points: &[Point3<f32>],
    _voxel_normals: &[Vector3<f32>],
    _voxel_size: f32,
) -> Vec<Vector3<f32>> {
    vec![Vector3::z(); points.len()]
}

pub fn approximate_normals_simple(
    points: &[Point3<f32>],
    _k: usize,
    _epsilon: f32,
) -> Vec<Vector3<f32>> {
    vec![Vector3::z(); points.len()]
}
