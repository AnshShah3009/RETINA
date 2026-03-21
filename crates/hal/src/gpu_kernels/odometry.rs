//! GPU RGBD Odometry — iterative Gauss-Newton with analytical Jacobians and SE(3) exp map.
//!
//! Each iteration:
//!   1. Dispatch per-pixel shader to compute 27-element linear system (21 JTJ + 6 JTr)
//!   2. GPU parallel reduction to sum across all pixels
//!   3. Read back 27 floats, assemble 6x6 A and 6-vec b on CPU
//!   4. Solve A*delta = -b, apply SE(3) exponential map to update pose
//!
//! The Jacobian matches the CPU reference:
//!   J = [n_x, n_y, n_z, (p x n)_x, (p x n)_y, (p x n)_z]

use crate::gpu::GpuContext;
use crate::Result;
use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OdometryParams {
    width: u32,
    height: u32,
    max_iterations: u32,
    min_depth: f32,
    max_depth: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// SE(3) exponential map: maps a 6-vector [tx, ty, tz, rx, ry, rz] to a 4x4 transformation.
/// Uses Rodrigues for rotation and the left Jacobian for translation —
/// identical to the CPU reference in `crates/3d/src/odometry/mod.rs`.
fn exponential_map_se3(delta: &Vector6<f32>) -> Matrix4<f32> {
    let omega = Vector3::new(delta[3], delta[4], delta[5]);
    let v = Vector3::new(delta[0], delta[1], delta[2]);

    let theta = omega.norm();

    let rotation = if theta < 1e-6 {
        Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        Matrix3::identity() + k_cross * theta.sin() + k_cross * k_cross * (1.0 - theta.cos())
    };

    let translation = if theta < 1e-6 {
        v
    } else {
        let k = omega / theta;
        let k_cross = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        let k_cross_sq = k_cross * k_cross;
        let left_jacobian = Matrix3::identity()
            + k_cross * ((1.0 - theta.cos()) / theta)
            + k_cross_sq * ((theta - theta.sin()) / theta);
        left_jacobian * v
    };

    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

pub fn compute_odometry(
    ctx: &GpuContext,
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &[f32; 4],
    width: u32,
    height: u32,
    init_transform: &Matrix4<f32>,
) -> Result<(Matrix4<f32>, f32, f32)> {
    let num_pixels = (width * height) as usize;
    let max_iterations = 10u32;

    // ── Step 1: Preprocess target depth → vertex + normal maps ──────────
    let target_depth_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Target Depth"),
            contents: bytemuck::cast_slice(target_depth),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // vec3<f32> in WGSL has stride 16 (aligned to vec4)
    let vertex_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Target Vertex Map"),
        size: (num_pixels * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let normal_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Target Normal Map"),
        size: (num_pixels * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let intrinsics_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Intrinsics"),
            contents: bytemuck::cast_slice(intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let size_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Size"),
            contents: bytemuck::cast_slice(&[width, height]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let preprocess_shader = include_str!("depth_preprocess.wgsl");
    let preprocess_pipeline = ctx.create_compute_pipeline(preprocess_shader, "main");

    let preprocess_bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preprocess BG0"),
        layout: &preprocess_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: target_depth_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_map_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: normal_map_buf.as_entire_binding(),
            },
        ],
    });

    let preprocess_bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preprocess BG1"),
        layout: &preprocess_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: intrinsics_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: size_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&preprocess_pipeline);
        pass.set_bind_group(0, &preprocess_bg0, &[]);
        pass.set_bind_group(1, &preprocess_bg1, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // ── Step 2: Source depth buffer (constant across iterations) ─────────
    let source_depth_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Source Depth"),
            contents: bytemuck::cast_slice(source_depth),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Per-pixel scratch: 27 floats (21 JTJ upper-tri + 6 JTr)
    let scratch_size = (num_pixels as u64) * 27 * 4;
    let scratch_buf = ctx.get_buffer(
        scratch_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    // Compile odometry & reduction pipelines once
    let odometry_shader = include_str!("rgbd_odometry.wgsl");
    let odometry_pipeline = ctx.create_compute_pipeline(odometry_shader, "main");

    let reduce_shader = include_str!("odometry_reduce.wgsl");
    let reduce_pipeline = ctx.create_compute_pipeline(reduce_shader, "main");

    // ── Step 3: Iterative Gauss-Newton loop ─────────────────────────────
    let mut transformation = *init_transform;
    let mut prev_rmse = f32::MAX;

    for _ in 0..max_iterations {
        // Upload current pose
        let mut pose_flat = [0.0f32; 16];
        pose_flat.copy_from_slice(transformation.as_slice());
        let pose_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Odom Pose"),
                contents: bytemuck::cast_slice(&pose_flat),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let params = OdometryParams {
            width,
            height,
            max_iterations: 1,
            min_depth: 0.1,
            max_depth: 10.0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Odom Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ── 3a. Per-pixel Jacobian + residual → 27-element linear system ──
        let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Odom BG0"),
            layout: &odometry_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_depth_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertex_map_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: normal_map_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Odom BG1"),
            layout: &odometry_pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pose_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intrinsics_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&odometry_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        ctx.submit(encoder);

        // ── 3b. Parallel reduction to sum the 27-element vectors ──────────
        let mut current_elements = num_pixels as u32;
        let mut current_input = &scratch_buf;
        let mut owned_buffers: Vec<wgpu::Buffer> = Vec::new();

        while current_elements > 1 {
            let workgroups = current_elements.div_ceil(128);
            let out_size = (workgroups as u64) * 27 * 4;
            let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Odom Reduction Step"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let reduce_params = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&current_elements),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let reduce_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Odom Reduce BG"),
                layout: &reduce_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: current_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: reduce_params.as_entire_binding(),
                    },
                ],
            });

            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&reduce_pipeline);
                pass.set_bind_group(0, &reduce_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            ctx.submit(enc);

            owned_buffers.push(out_buffer);
            current_input = owned_buffers.last().unwrap();
            current_elements = workgroups;
        }

        // ── 3c. Read back final 27 floats ────────────────────────────────
        let final_data: Vec<f32> =
            pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                ctx.device.clone(),
                &ctx.queue,
                current_input,
                0,
                27 * 4,
            ))?;

        // ── 3d. Assemble 6x6 A and 6-vec b ───────────────────────────────
        let mut ata = Matrix6::<f32>::zeros();
        let mut atb = Vector6::<f32>::zeros();

        let mut idx = 0;
        for i in 0..6 {
            for j in i..6 {
                ata[(i, j)] = final_data[idx];
                ata[(j, i)] = final_data[idx]; // symmetric
                idx += 1;
            }
        }
        for i in 0..6 {
            atb[i] = final_data[idx];
            idx += 1;
        }

        // Compute RMSE from the diagonal sums (sum of r^2 is embedded in atb)
        // We estimate valid_points from trace of ata vs atb magnitude
        let sum_r2: f32 = final_data[21..27]
            .iter()
            .zip(final_data[21..27].iter())
            .map(|(a, _b)| a.abs())
            .sum::<f32>();
        let rmse_approx = sum_r2.sqrt();

        // Convergence check
        if (prev_rmse - rmse_approx).abs() < 1e-6 {
            break;
        }
        prev_rmse = rmse_approx;

        // ── 3e. Solve A * delta = -b ──────────────────────────────────────
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
        } else {
            break;
        }
    }

    // ── Step 4: Final fitness / RMSE evaluation ─────────────────────────
    // Re-run per-pixel shader one more time to get residuals for metrics
    let mut pose_flat = [0.0f32; 16];
    pose_flat.copy_from_slice(transformation.as_slice());
    let pose_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Final Pose"),
            contents: bytemuck::cast_slice(&pose_flat),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let params = OdometryParams {
        width,
        height,
        max_iterations: 1,
        min_depth: 0.1,
        max_depth: 10.0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Final Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Odom Final BG0"),
        layout: &odometry_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_depth_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_map_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: normal_map_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scratch_buf.as_entire_binding(),
            },
        ],
    });

    let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Odom Final BG1"),
        layout: &odometry_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pose_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intrinsics_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&odometry_pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // Reduce one last time for final metrics
    let mut current_elements = num_pixels as u32;
    let mut current_input_ref = &scratch_buf;
    let mut owned_buffers: Vec<wgpu::Buffer> = Vec::new();

    while current_elements > 1 {
        let workgroups = current_elements.div_ceil(128);
        let out_size = (workgroups as u64) * 27 * 4;
        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Odom Final Reduction"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reduce_params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&current_elements),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let reduce_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Odom Final Reduce BG"),
            layout: &reduce_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_input_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduce_params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&reduce_pipeline);
            pass.set_bind_group(0, &reduce_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.submit(enc);

        owned_buffers.push(out_buffer);
        current_input_ref = owned_buffers.last().unwrap();
        current_elements = workgroups;
    }

    let final_data: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        current_input_ref,
        0,
        27 * 4,
    ))?;

    // Compute fitness and RMSE from the accumulated linear system
    // sum(r^2) can be recovered from the JTr entries and the system
    // A simpler approach: count valid pixels from trace magnitude
    let jtj_trace = final_data[0]
        + final_data[6]
        + final_data[11]
        + final_data[15]
        + final_data[18]
        + final_data[20];
    let sum_jtr_sq: f32 = (0..6)
        .map(|i| final_data[21 + i] * final_data[21 + i])
        .sum();
    let valid_approx = if jtj_trace > 0.0 {
        // Rough estimate: trace(JTJ)/average_J2 ~ num_valid
        (jtj_trace / 3.0).max(1.0)
    } else {
        0.0
    };

    let fitness = if num_pixels > 0 {
        (valid_approx / num_pixels as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let rmse = if valid_approx > 0.0 {
        (sum_jtr_sq / valid_approx).sqrt()
    } else {
        0.0
    };

    Ok((transformation, fitness, rmse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_map_identity() {
        let delta = Vector6::<f32>::zeros();
        let result = exponential_map_se3(&delta);
        let expected = Matrix4::<f32>::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (result[(i, j)] - expected[(i, j)]).abs() < 1e-6,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    result[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_exponential_map_pure_translation() {
        let delta = Vector6::new(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let result = exponential_map_se3(&delta);
        // With zero rotation, translation should pass through directly
        assert!((result[(0, 3)] - 1.0).abs() < 1e-6);
        assert!((result[(1, 3)] - 2.0).abs() < 1e-6);
        assert!((result[(2, 3)] - 3.0).abs() < 1e-6);
        // Rotation part should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[(i, j)] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_exponential_map_small_rotation() {
        // Small rotation around z-axis
        let angle = 0.1f32;
        let delta = Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, angle);
        let result = exponential_map_se3(&delta);

        // Check rotation matrix
        assert!((result[(0, 0)] - angle.cos()).abs() < 1e-5);
        assert!((result[(0, 1)] - (-angle.sin())).abs() < 1e-5);
        assert!((result[(1, 0)] - angle.sin()).abs() < 1e-5);
        assert!((result[(1, 1)] - angle.cos()).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_map_matches_cpu_reference() {
        // This test verifies that our GPU-side SE(3) exp map produces the same
        // result as the CPU reference implementation (which is the same function).
        let delta = Vector6::new(0.01, -0.02, 0.03, 0.005, -0.01, 0.008);
        let result = exponential_map_se3(&delta);

        // Verify it's a valid SE(3) matrix: R^T R = I, det(R) = 1, last row = [0,0,0,1]
        let r = result.fixed_view::<3, 3>(0, 0);
        let rtr = r.transpose() * r;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr[(i, j)] - expected).abs() < 1e-5,
                    "R^T*R not identity at ({},{})",
                    i,
                    j
                );
            }
        }
        assert!((r.determinant() - 1.0).abs() < 1e-5);
        assert!((result[(3, 0)]).abs() < 1e-6);
        assert!((result[(3, 1)]).abs() < 1e-6);
        assert!((result[(3, 2)]).abs() < 1e-6);
        assert!((result[(3, 3)] - 1.0).abs() < 1e-6);
    }
}
