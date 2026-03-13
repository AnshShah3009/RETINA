use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Try Morton GPU normals; fall back to the fast CPU analytic path if no GPU is
/// available.  Used by `MlxContext` so Apple Silicon callers get Metal via wgpu,
/// and non-GPU callers still get correct results.
pub fn compute_normals_morton_gpu_or_cpu(
    points: &[nalgebra::Vector3<f32>],
    k: u32,
) -> Vec<nalgebra::Vector3<f32>> {
    if let Ok(gpu) = GpuContext::global() {
        if let Ok(normals) =
            crate::gpu_kernels::pointcloud_gpu::compute_normals_morton_gpu(gpu, points, k)
        {
            return normals;
        }
    }
    // CPU fallback: voxel-hash kNN + analytic eigensolver (same algorithm as
    // CpuBackend::pointcloud_normals and cv-3d compute_normals_cpu).
    normals_cpu_analytic(points, k as usize)
}

/// Voxel-hash kNN + analytic 3×3 eigensolver, fully on CPU.
/// Self-contained within `hal` (no dependency on `cv-3d`) for use by MlxContext.
pub fn normals_cpu_analytic(
    points: &[nalgebra::Vector3<f32>],
    k: usize,
) -> Vec<nalgebra::Vector3<f32>> {
    use rayon::prelude::*;

    if points.len() < 3 {
        return vec![nalgebra::Vector3::z(); points.len()];
    }
    let k = k.max(3).min(points.len().saturating_sub(1));
    let pts_p: Vec<nalgebra::Point3<f32>> =
        points.iter().map(|v| nalgebra::Point3::from(*v)).collect();

    // Adaptive voxel size from bounding box (dimensionality-aware).
    let vs = {
        let (mut minx, mut miny, mut minz) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut maxx, mut maxy, mut maxz) = (f32::MIN, f32::MIN, f32::MIN);
        for p in &pts_p {
            minx = minx.min(p.x);
            maxx = maxx.max(p.x);
            miny = miny.min(p.y);
            maxy = maxy.max(p.y);
            minz = minz.min(p.z);
            maxz = maxz.max(p.z);
        }
        let sx = (maxx - minx).max(1e-9_f32);
        let sy = (maxy - miny).max(1e-9_f32);
        let sz = (maxz - minz).max(1e-9_f32);
        let mut spans = [sx, sy, sz];
        spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (s0, s1, s2) = (spans[0], spans[1], spans[2]);
        let n = pts_p.len() as f32;
        let vs = if s0 > s2 * 0.01 {
            (k as f32 / (8.0 * n / (sx * sy * sz))).cbrt()
        } else if s1 > s2 * 0.01 {
            (k as f32 / (9.0 * n / (s1 * s2))).sqrt()
        } else {
            k as f32 / (3.0 * n / s2)
        };
        vs.clamp(1e-6, (s1 / 2.0).max(1e-6_f32))
    };

    let mut grid: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> =
        hashbrown::HashMap::with_capacity(pts_p.len() / 8);
    for (i, p) in pts_p.iter().enumerate() {
        grid.entry((
            (p.x / vs).floor() as i32,
            (p.y / vs).floor() as i32,
            (p.z / vs).floor() as i32,
        ))
        .or_default()
        .push(i);
    }

    pts_p
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
                        if let Some(b) = grid.get(&(vx + dx, vy + dy, vz + dz)) {
                            for &idx in b {
                                if idx != i {
                                    let p = &pts_p[idx];
                                    let d = (center.x - p.x).powi(2)
                                        + (center.y - p.y).powi(2)
                                        + (center.z - p.z).powi(2);
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
                return nalgebra::Vector3::z();
            }

            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            let mut cz = 0.0f32;
            for &(_, idx) in &cands {
                cx += pts_p[idx].x;
                cy += pts_p[idx].y;
                cz += pts_p[idx].z;
            }
            let inv = 1.0 / cands.len() as f32;
            cx *= inv;
            cy *= inv;
            cz *= inv;

            let mut cxx = 0.0f32;
            let mut cxy = 0.0f32;
            let mut cxz = 0.0f32;
            let mut cyy = 0.0f32;
            let mut cyz = 0.0f32;
            let mut czz = 0.0f32;
            for &(_, idx) in &cands {
                let dx = pts_p[idx].x - cx;
                let dy = pts_p[idx].y - cy;
                let dz = pts_p[idx].z - cz;
                cxx += dx * dx;
                cxy += dx * dy;
                cxz += dx * dz;
                cyy += dy * dy;
                cyz += dy * dz;
                czz += dz * dz;
            }

            // Analytic min eigenvector — Open3D / Geometric Tools algorithm.
            let max_c = cxx
                .abs()
                .max(cxy.abs())
                .max(cxz.abs())
                .max(cyy.abs())
                .max(cyz.abs())
                .max(czz.abs());
            if max_c < 1e-30 {
                return nalgebra::Vector3::z();
            }
            let s = 1.0 / max_c;
            let (a00, a01, a02, a11, a12, a22) =
                (cxx * s, cxy * s, cxz * s, cyy * s, cyz * s, czz * s);
            let norm = a01 * a01 + a02 * a02 + a12 * a12;
            let q = (a00 + a11 + a22) / 3.0;
            let (b00, b11, b22) = (a00 - q, a11 - q, a22 - q);
            let p_val = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0).sqrt();
            if p_val < 1e-10 {
                return nalgebra::Vector3::z();
            }
            let (c00, c01, c02) = (
                b11 * b22 - a12 * a12,
                a01 * b22 - a12 * a02,
                a01 * a12 - b11 * a02,
            );
            let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p_val * p_val * p_val);
            let half_det = (det * 0.5_f32).clamp(-1.0, 1.0);
            let angle = half_det.acos() / 3.0;
            const TPI: f32 = 2.094_395_1;
            let eval_min = q + p_val * (angle + TPI).cos() * 2.0;
            let r0 = nalgebra::Vector3::new(a00 - eval_min, a01, a02);
            let r1 = nalgebra::Vector3::new(a01, a11 - eval_min, a12);
            let r2 = nalgebra::Vector3::new(a02, a12, a22 - eval_min);
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
                nalgebra::Vector3::z()
            } else {
                best / len
            }
        })
        .collect()
}

pub fn transform_points(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    transform: &[[f32; 4]; 4],
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = input.shape.height;
    if num_points == 0 {
        return Ok(input.clone());
    }

    let byte_size = (num_points * 16) as u64; // float4 alignment
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PC Transform Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut transposed = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            transposed[c][r] = transform[r][c];
        }
    }

    let transform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Matrix"),
            contents: bytemuck::cast_slice(&transposed),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let num_points_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Transform Num Points"),
            contents: bytemuck::bytes_of(&(num_points as u32)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../gpu_kernels/pointcloud_transform.wgsl");
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PC Transform Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PC Transform BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PC Transform Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PC Transform Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PC Transform Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: num_points_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_points as u32).div_ceil(256);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), num_points * 4),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

pub fn compute_normals(
    ctx: &GpuContext,
    points: &Tensor<f32, GpuStorage<f32>>,
    k_neighbors: u32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = points.shape.height;
    if num_points == 0 {
        return Ok(points.clone());
    }

    // Download points from GPU → CPU, Morton-sort on CPU, then dispatch the Morton
    // shader.  This replaces the former O(n²) brute-force scan with an O(n log n +
    // n·window) approach while keeping the GPU busy for the PCA computation.
    use crate::tensor_ext::TensorToCpu;
    let cpu_tensor = points.to_cpu_ctx(ctx)?;
    let raw = cpu_tensor.as_slice()?;

    // Points are stored as vec4<f32> (16-byte aligned), so stride = 4 floats.
    let pts: Vec<nalgebra::Vector3<f32>> = raw
        .chunks(4)
        .map(|c| nalgebra::Vector3::new(c[0], c[1], c[2]))
        .collect();

    // Morton GPU normals: CPU sort → GPU PCA (correct O(n log n + n·window)).
    let normals_vec =
        crate::gpu_kernels::pointcloud_gpu::compute_normals_morton_gpu(ctx, &pts, k_neighbors)?;

    // Upload resulting normals back as a GPU Tensor (vec4 layout).
    let normals_data: Vec<[f32; 4]> = normals_vec.iter().map(|n| [n.x, n.y, n.z, 0.0]).collect();

    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Normals Output"),
            contents: bytemuck::cast_slice(&normals_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(buffer), num_points * 4),
        shape: points.shape,
        dtype: points.dtype,
        _phantom: PhantomData,
    })
}

/// Batch GPU eigenvector pass for the hybrid pipeline.
///
/// Takes **pre-computed** covariance matrices (one per point, upper triangle)
/// and computes the minimum eigenvector (surface normal) for each on the GPU.
///
/// `covs`: slice of `[cxx, cxy, cxz, cyy, cyz, czz]` per point, i.e. `6 * n` elements.
///
/// This is the GPU half of the hybrid `CPU kNN → GPU PCA` pipeline. The CPU
/// handles the spatially irregular kNN search; the GPU handles the fully-parallel,
/// compute-bound PCA step.
pub fn compute_normals_from_covariances_gpu(
    ctx: &GpuContext,
    covs: &[[f32; 6]],
) -> Result<Vec<nalgebra::Vector3<f32>>> {
    use crate::gpu_kernels::buffer_utils::{create_buffer, create_buffer_uninit, read_buffer};
    use wgpu::BufferUsages;

    let n = covs.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let device = ctx.device.clone();
    let queue = &ctx.queue;

    // Pack as two vec4s per point: (cxx,cxy,cxz,cyy) + (cyz,czz,0,0).
    let packed: Vec<[f32; 8]> = covs
        .iter()
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], 0.0, 0.0])
        .collect();

    let covs_buf = create_buffer(&device, &packed, BufferUsages::STORAGE);
    let normals_buf = create_buffer_uninit(
        &device,
        n * 16, // vec4<f32> per point
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let n_u32 = n as u32;
    let n_buf = create_buffer(&device, &[n_u32], BufferUsages::UNIFORM);

    let shader_source = include_str!("pointcloud_normals_batch_pca.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Batch PCA Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Batch PCA Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Batch PCA Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: covs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: normals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: n_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let result_raw: Vec<[f32; 4]> =
        pollster::block_on(read_buffer(device.clone(), queue, &normals_buf, 0, n * 16))?;

    Ok(result_raw
        .into_iter()
        .map(|v| nalgebra::Vector3::new(v[0], v[1], v[2]))
        .collect())
}
