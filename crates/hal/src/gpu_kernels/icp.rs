use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ICPParams {
    num_src: u32,
    num_tgt: u32,
    max_dist_sq: f32,
}

pub fn icp_correspondences(
    ctx: &GpuContext,
    src: &Tensor<f32, GpuStorage<f32>>,
    tgt: &Tensor<f32, GpuStorage<f32>>,
    max_dist: f32,
) -> Result<Vec<(usize, usize, f32)>> {
    let num_src = src.shape.height;
    let num_tgt = tgt.shape.height;

    if num_src == 0 || num_tgt == 0 {
        return Ok(Vec::new());
    }

    let byte_size = (num_src * 16) as u64; // [src_idx, tgt_idx, dist_sq, valid]
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICP Correspondence Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ICPParams {
        num_src: num_src as u32,
        num_tgt: num_tgt as u32,
        max_dist_sq: max_dist * max_dist,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/icp_correspondence.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICP Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tgt.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
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
        let x = (num_src as u32).div_ceil(64);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    let raw_results: Vec<[f32; 4]> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &output_buffer,
            0,
            byte_size as usize,
        ))?;

    let mut correspondences = Vec::new();
    for res in raw_results {
        if res[3] > 0.5 {
            correspondences.push((res[0] as usize, res[1] as usize, res[2].sqrt()));
        }
    }

    Ok(correspondences)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    num_points: u32,
    _pad: [u32; 3],
    transform: [[f32; 4]; 4],
}

pub fn icp_accumulate(
    ctx: &GpuContext,
    source: &Tensor<f32, GpuStorage<f32>>,
    target: &Tensor<f32, GpuStorage<f32>>,
    target_normals: &Tensor<f32, GpuStorage<f32>>,
    correspondences: &[(u32, u32)],
    transform: &nalgebra::Matrix4<f32>,
) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
    use crate::gpu_kernels::buffer_utils::create_buffer;

    let num_corr = correspondences.len();
    if num_corr == 0 {
        return Ok((nalgebra::Matrix6::zeros(), nalgebra::Vector6::zeros()));
    }

    // 1. Prepare data
    let corr_data: Vec<[u32; 2]> = correspondences.iter().map(|&(s, t)| [s, t]).collect();
    let corr_buffer = create_buffer(&ctx.device, &corr_data, wgpu::BufferUsages::STORAGE);

    let ata_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("AtA Accumulator"),
        size: 36 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let atb_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Atb Accumulator"),
        size: 6 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Zero out buffers
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&ata_buffer, 0, None);
    encoder.clear_buffer(&atb_buffer, 0, None);

    let params = AccumulateParams {
        num_points: num_corr as u32,
        _pad: [0; 3],
        transform: (*transform).into(),
    };
    let params_buffer = create_buffer(&ctx.device, &[params], wgpu::BufferUsages::UNIFORM);

    // 2. Pipeline & Bind Group
    let shader_source = include_str!("../../shaders/icp_accumulate.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Accumulate Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: target_normals.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: corr_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: atb_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // 3. Dispatch
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_corr as u32).div_ceil(256);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    // 4. Read back and convert from fixed-point i32 to f32
    let ata_raw: Vec<i32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &ata_buffer,
        0,
        36 * 4,
    ))?;

    let atb_raw: Vec<i32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &atb_buffer,
        0,
        6 * 4,
    ))?;

    let mut ata = nalgebra::Matrix6::<f32>::zeros();
    for i in 0..6 {
        for j in 0..6 {
            ata[(i, j)] = ata_raw[i * 6 + j] as f32 / 1000000.0;
        }
    }

    let mut atb = nalgebra::Vector6::<f32>::zeros();
    for i in 0..6 {
        atb[i] = atb_raw[i] as f32 / 1000000.0;
    }

    Ok((ata, atb))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IcpDenseParams {
    width: u32,
    height: u32,
    max_dist: f32,
    max_angle: f32,
}

pub fn dense_step(
    ctx: &GpuContext,
    source_depth: &Tensor<f32, GpuStorage<f32>>,
    target_data: &Tensor<f32, GpuStorage<f32>>,
    intrinsics: &[f32; 4],
    initial_guess: &nalgebra::Matrix4<f32>,
    max_dist: f32,
    max_angle: f32,
) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
    let (h, w) = source_depth.shape.hw();
    let num_pixels = (w * h) as u32;

    let scratch_size = (num_pixels * 27 * 4) as u64;
    let scratch_buffer = ctx.get_buffer(
        scratch_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    let params = IcpDenseParams {
        width: w as u32,
        height: h as u32,
        max_dist,
        max_angle: max_angle.cos(),
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Dense Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let intrinsics_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Intrinsics"),
            contents: bytemuck::cast_slice(intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let inv_intrinsics = [
        1.0 / intrinsics[0],
        1.0 / intrinsics[1],
        intrinsics[2],
        intrinsics[3],
    ];
    let inv_intrinsics_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Inv Intrinsics"),
            contents: bytemuck::cast_slice(&inv_intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let mut pose_flat = [0.0f32; 16];
    pose_flat.copy_from_slice(initial_guess.as_slice());
    let pose_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Pose"),
            contents: bytemuck::cast_slice(&pose_flat),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/icp_dense.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICP Dense BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_depth.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_data.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scratch_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: intrinsics_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: inv_intrinsics_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: pose_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ICP Dense Compute"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((w as u32).div_ceil(16), (h as u32).div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // Reduction
    let reduce_shader = include_str!("../../shaders/icp_reduce.wgsl");
    let reduce_pipeline = ctx.create_compute_pipeline(reduce_shader, "main");

    let mut current_elements = num_pixels;
    let mut current_input = scratch_buffer;

    while current_elements > 1 {
        let workgroups = current_elements.div_ceil(128);
        let out_size = (workgroups * 27 * 4) as u64;
        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Step"),
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
            label: Some("ICP Reduce BG"),
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

        let mut red_enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = red_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&reduce_pipeline);
            pass.set_bind_group(0, &reduce_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.submit(red_enc);

        current_input = out_buffer;
        current_elements = workgroups;
    }

    let final_data: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &current_input, // Final result is here
        0,
        27 * 4,
    ))?;

    let mut ata = nalgebra::Matrix6::zeros();
    let mut atb = nalgebra::Vector6::zeros();

    let mut idx = 0;
    for i in 0..6 {
        for j in i..6 {
            ata[(i, j)] = final_data[idx];
            ata[(j, i)] = final_data[idx];
            idx += 1;
        }
    }
    for i in 0..6 {
        atb[i] = final_data[idx];
        idx += 1;
    }

    Ok((ata, atb))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColoredICPConfig {
    pub lambda_geometric: f32,
    pub max_correspondence_distance: f32,
    pub max_iterations: u32,
    pub relative_fitness: f32,
    pub relative_rmse: f32,
}

impl Default for ColoredICPConfig {
    fn default() -> Self {
        Self {
            lambda_geometric: 0.97,
            max_correspondence_distance: 0.07,
            max_iterations: 100,
            relative_fitness: 1e-6,
            relative_rmse: 1e-6,
        }
    }
}

pub fn colored_icp_kernel() -> &'static str {
    r#"
struct ColoredICPParams {
    lambda_geometric: f32,
    max_dist_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> source_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> target_points: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> target_normals: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> target_colors: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> color_gradients: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> transform: mat4x4<f32>;
@group(0) @binding(6) var<storage, read> params: ColoredICPParams;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

fn transform_point(p: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    let q = transform * vec4(p, 1.0);
    return q.xyz / q.w;
}

fn apply_rotation(p: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(
        transform[0][0] * p.x + transform[0][1] * p.y + transform[0][2] * p.z,
        transform[1][0] * p.x + transform[1][1] * p.y + transform[1][2] * p.z,
        transform[2][0] * p.x + transform[2][1] * p.y + transform[2][2] * p.z,
    );
}

fn skew(v: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0
    );
}

fn find_nearest_point(src: vec3<f32>, targets: array<vec3<f32>>, max_dist: f32) -> u32 {
    var min_dist = max_dist;
    var nearest_idx = 0u;
    
    let n = arrayLength(&targets);
    for (var i = 0u; i < n; i = i + 1u) {
        let diff = targets[i] - src;
        let dist = dot(diff, diff);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    
    return nearest_idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&source_points)) {
        return;
    }
    
    let sp = source_points[idx];
    let transformed = transform_point(sp, transform);
    
    let nearest = find_nearest_point(transformed, target_points, params.max_dist_sq);
    let tp = target_points[nearest];
    let tn = target_normals[nearest];
    
    let diff = transformed - tp;
    let dist_sq = dot(diff, diff);
    
    if (dist_sq > params.max_dist_sq || dist_sq < 1e-10) {
        for (var i = 0u; i < 21u; i = i + 1u) {
            output[idx * 21u + i] = 0.0;
        }
        return;
    }
    
    let sqrt_lambda = params.lambda_geometric;
    let sqrt_photo = 1.0 - params.lambda_geometric;
    
    let Rp = apply_rotation(sp, transform);
    
    let skew_Rp = skew(Rp);
    let J_geo = skew_Rp * tn;
    
    let residual_geo = dot(diff, tn);
    
    let color_grad = color_gradients[nearest];
    let tc = target_colors[nearest];
    
    let grad_c = vec3<f32>(color_grad.x, color_grad.y, 0.0);
    let dIdx = dot(grad_c, tn);
    
    let J_photo = dIdx * tn;
    let residual_photo = dot(tc - vec3<f32>(0.0, 0.0, 0.0), tn);
    
    let J0 = J_geo * sqrt_lambda;
    let J1 = tn * sqrt_lambda;
    let r = residual_geo * sqrt_lambda;
    
    var base = idx * 21u;
    
    output[base + 0u] = J0.x * J0.x;
    output[base + 1u] = J0.y * J0.y;
    output[base + 2u] = J0.z * J0.z;
    output[base + 3u] = J1.x * J1.x;
    output[base + 4u] = J1.y * J1.y;
    output[base + 5u] = J1.z * J1.z;
    output[base + 6u] = J0.x * J0.y;
    output[base + 7u] = J0.x * J0.z;
    output[base + 8u] = J0.x * J1.x;
    output[base + 9u] = J0.x * J1.y;
    output[base + 10u] = J0.x * J1.z;
    output[base + 11u] = J0.y * J0.z;
    output[base + 12u] = J0.y * J1.x;
    output[base + 13u] = J0.y * J1.y;
    output[base + 14u] = J0.y * J1.z;
    output[base + 15u] = J0.z * J1.x;
    output[base + 16u] = J0.z * J1.y;
    output[base + 17u] = J0.z * J1.z;
    output[base + 18u] = J1.x * J1.y;
    output[base + 19u] = J1.x * J1.z;
    output[base + 20u] = J1.y * J1.z;
    
    output[base + 0u] = J0.x * r;
    output[base + 1u] = J0.y * r;
    output[base + 2u] = J0.z * r;
    output[base + 3u] = J1.x * r;
    output[base + 4u] = J1.y * r;
    output[base + 5u] = J1.z * r;
}
"#
}

pub fn compute_color_gradients_kernel() -> &'static str {
    r#"
struct GradientParams {
    width: u32,
    height: u32,
    search_radius: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> colors: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> params: GradientParams;
@group(0) @binding(4) var<storage, read_write> gradients: array<vec3<f32>>;

fn get_color(colors: array<vec3<f32>>, width: u32, idx: u32) -> vec3<f32> {
    if (idx >= arrayLength(&colors)) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return colors[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&points)) {
        return;
    }
    
    let width = params.width;
    let radius = u32(params.search_radius);
    
    let x = idx % width;
    let y = idx / width;
    
    let c = colors[idx];
    let intensity = (c.r + c.g + c.b) / 3.0;
    
    var grad_x = 0.0;
    var grad_y = 0.0;
    var count = 0.0;
    
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
        for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
            let nx = u32(i32(x) + ox);
            let ny = u32(i32(y) + oy);
            if (nx < width && ny < params.height) {
                let nidx = ny * width + nx;
                if (nidx != idx && nidx < arrayLength(&colors)) {
                    let nc = colors[nidx];
                    let nintensity = (nc.r + nc.g + nc.b) / 3.0;
                    grad_x = grad_x + f32(ox) * (nintensity - intensity);
                    grad_y = grad_y + f32(oy) * (nintensity - intensity);
                    count = count + 1.0;
                }
            }
        }
    }
    
    if (count > 0.0) {
        grad_x = grad_x / count;
        grad_y = grad_y / count;
    }
    
    let n = normals[idx];
    let grad_mag = sqrt(grad_x * grad_x + grad_y * grad_y);
    if (grad_mag > 1e-6) {
        grad_x = grad_x / grad_mag;
        grad_y = grad_y / grad_mag;
    }
    
    gradients[idx] = vec3<f32>(grad_x, grad_y, grad_mag);
}
"#
}

pub struct GeneralizedICPConfig {
    pub max_correspondence_distance: f32,
    pub max_iterations: u32,
    pub relative_fitness: f32,
    pub relative_rmse: f32,
}

impl Default for GeneralizedICPConfig {
    fn default() -> Self {
        Self {
            max_correspondence_distance: 0.05,
            max_iterations: 100,
            relative_fitness: 1e-6,
            relative_rmse: 1e-6,
        }
    }
}

pub fn generalized_icp_kernel() -> &'static str {
    r#"
struct GICPParams {
    max_dist_sq: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> source_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> target_points: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> target_cov: array<mat3x3<f32>>;
@group(0) @binding(3) var<storage, read> transform: mat4x4<f32>;
@group(0) @binding(4) var<storage, read> params: GICPParams;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn transform_point(p: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    let q = transform * vec4(p, 1.0);
    return q.xyz / q.w;
}

fn apply_rotation(p: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(
        transform[0][0] * p.x + transform[0][1] * p.y + transform[0][2] * p.z,
        transform[1][0] * p.x + transform[1][1] * p.y + transform[1][2] * p.z,
        transform[2][0] * p.x + transform[2][1] * p.y + transform[2][2] * p.z,
    );
}

fn skew(v: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0
    );
}

fn find_correspondence(src: vec3<f32>, targets: array<vec3<f32>>, max_dist_sq: f32) -> vec4<u32> {
    var min_dist = max_dist_sq;
    var nearest_idx = 0u;
    
    let n = arrayLength(&targets);
    for (var i = 0u; i < n; i = i + 1u) {
        let diff = targets[i] - src;
        let dist = dot(diff, diff);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    
    return vec4<u32>(nearest_idx, 0u, 0u, 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&source_points)) {
        return;
    }
    
    let sp = source_points[idx];
    let transformed = transform_point(sp, transform);
    
    let corr = find_correspondence(transformed, target_points, params.max_dist_sq);
    let target_idx = corr.x;
    
    let tp = target_points[target_idx];
    let tcov = target_cov[target_idx];
    
    let diff = transformed - tp;
    let dist_sq = dot(diff, diff);
    
    if (dist_sq > params.max_dist_sq || dist_sq < 1e-10) {
        for (var i = 0u; i < 21u; i = i + 1u) {
            output[idx * 21u + i] = 0.0;
        }
        return;
    }
    
    let Rp = apply_rotation(sp, transform);
    let skew_Rp = skew(Rp);
    
    let cov_sum = tcov + tcov;
    let cov_diff = mat3x3<f32>(
        cov_sum[0][0], cov_sum[0][1], cov_sum[0][2],
        cov_sum[1][0], cov_sum[1][1], cov_sum[1][2],
        cov_sum[2][0], cov_sum[2][1], cov_sum[2][2],
    );
    
    let J = skew_Rp;
    let cov_diff_t = transpose(cov_diff);
    
    let J_cov = mat3x3<f32>(
        J[0][0] * cov_diff_t[0][0] + J[0][1] * cov_diff_t[1][0] + J[0][2] * cov_diff_t[2][0],
        J[0][0] * cov_diff_t[0][1] + J[0][1] * cov_diff_t[1][1] + J[0][2] * cov_diff_t[2][1],
        J[0][0] * cov_diff_t[0][2] + J[0][1] * cov_diff_t[1][2] + J[0][2] * cov_diff_t[2][2],
        J[1][0] * cov_diff_t[0][0] + J[1][1] * cov_diff_t[1][0] + J[1][2] * cov_diff_t[2][0],
        J[1][0] * cov_diff_t[0][1] + J[1][1] * cov_diff_t[1][1] + J[1][2] * cov_diff_t[2][1],
        J[1][0] * cov_diff_t[0][2] + J[1][1] * cov_diff_t[1][2] + J[1][2] * cov_diff_t[2][2],
        J[2][0] * cov_diff_t[0][0] + J[2][1] * cov_diff_t[1][0] + J[2][2] * cov_diff_t[2][0],
        J[2][0] * cov_diff_t[0][1] + J[2][1] * cov_diff_t[1][1] + J[2][2] * cov_diff_t[2][1],
        J[2][0] * cov_diff_t[0][2] + J[2][1] * cov_diff_t[1][2] + J[2][2] * cov_diff_t[2][2],
    );
    
    let residual = diff;
    
    var base = idx * 21u;
    
    output[base + 0u] = J_cov[0][0] * J_cov[0][0];
    output[base + 1u] = J_cov[0][1] * J_cov[0][1];
    output[base + 2u] = J_cov[0][2] * J_cov[0][2];
    output[base + 3u] = J_cov[1][0] * J_cov[1][0];
    output[base + 4u] = J_cov[1][1] * J_cov[1][1];
    output[base + 5u] = J_cov[1][2] * J_cov[1][2];
    output[base + 6u] = J_cov[2][0] * J_cov[2][0];
    output[base + 7u] = J_cov[2][1] * J_cov[2][1];
    output[base + 8u] = J_cov[2][2] * J_cov[2][2];
    
    output[base + 9u] = J_cov[0][0] * residual.x;
    output[base + 10u] = J_cov[0][1] * residual.y;
    output[base + 11u] = J_cov[0][2] * residual.z;
    output[base + 12u] = J_cov[1][0] * residual.x;
    output[base + 13u] = J_cov[1][1] * residual.y;
    output[base + 14u] = J_cov[1][2] * residual.z;
    output[base + 15u] = J_cov[2][0] * residual.x;
    output[base + 16u] = J_cov[2][1] * residual.y;
    output[base + 17u] = J_cov[2][2] * residual.z;
    
    for (var i = 18u; i < 21u; i = i + 1u) {
        output[base + i] = 0.0;
    }
}
"#
}

pub fn gicp_accumulate_kernel() -> &'static str {
    r#"
struct GICPAccumParams {
    num_correspondences: u32,
    num_source: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> contributions: array<f32>;
@group(0) @binding(1) var<storage, read> params: GICPAccumParams;
@group(0) @binding(2) var<storage, read_write> ata: array<f32>;
@group(0) @binding(3) var<storage, read_write> atb: array<f32>;

var<workgroup> shared_ata: array<f32, 21>;
var<workgroup> shared_atb: array<f32, 6>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx == 0u) {
        for (var i = 0u; i < 21u; i = i + 1u) {
            shared_ata[i] = 0.0;
        }
        for (var i = 0u; i < 6u; i = i + 1u) {
            shared_atb[i] = 0.0;
        }
    }
    
    workgroupBarrier();
    
    if (idx < params.num_source) {
        var base = idx * 21u;
        
        atomicAdd(&shared_ata[0], contributions[base + 0u]);
        atomicAdd(&shared_ata[1], contributions[base + 1u]);
        atomicAdd(&shared_ata[2], contributions[base + 2u]);
        atomicAdd(&shared_ata[3], contributions[base + 3u]);
        atomicAdd(&shared_ata[4], contributions[base + 4u]);
        atomicAdd(&shared_ata[5], contributions[base + 5u]);
        atomicAdd(&shared_ata[6], contributions[base + 6u]);
        atomicAdd(&shared_ata[7], contributions[base + 7u]);
        atomicAdd(&shared_ata[8], contributions[base + 8u]);
        
        atomicAdd(&shared_atb[0], contributions[base + 9u]);
        atomicAdd(&shared_atb[1], contributions[base + 10u]);
        atomicAdd(&shared_atb[2], contributions[base + 11u]);
        atomicAdd(&shared_atb[3], contributions[base + 12u]);
        atomicAdd(&shared_atb[4], contributions[base + 13u]);
        atomicAdd(&shared_atb[5], contributions[base + 14u]);
    }
    
    workgroupBarrier();
    
    if (idx == 0u) {
        ata[0] = shared_ata[0];
        ata[1] = shared_ata[1];
        ata[2] = shared_ata[2];
        ata[3] = shared_ata[3];
        ata[4] = shared_ata[4];
        ata[5] = shared_ata[5];
        ata[6] = shared_ata[6];
        ata[7] = shared_ata[7];
        ata[8] = shared_ata[8];
        
        atb[0] = shared_atb[0];
        atb[1] = shared_atb[1];
        atb[2] = shared_atb[2];
        atb[3] = shared_atb[3];
        atb[4] = shared_atb[4];
        atb[5] = shared_atb[5];
    }
}
"#
}
