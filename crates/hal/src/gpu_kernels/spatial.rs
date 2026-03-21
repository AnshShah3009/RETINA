use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn voxel_downsample(
    ctx: &GpuContext,
    points: &Tensor<f32, GpuStorage<f32>>,
    voxel_size: f32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = points.shape.height;

    if num_points == 0 {
        return Err(crate::Error::InvalidInput("Empty point cloud".into()));
    }

    let voxel_shader = r#"
struct VoxelParams {
    voxel_size: f32,
    num_points: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> params: VoxelParams;
@group(0) @binding(2) var<storage, read_write> voxel_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_points: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read_write> output_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let p = input_points[idx];
    
    let voxel_x = i32(floor(p.x / params.voxel_size));
    let voxel_y = i32(floor(p.y / params.voxel_size));
    let voxel_z = i32(floor(p.z / params.voxel_size));
    
    let hash = u32((voxel_x * 73856093 ^ voxel_y * 19349663 ^ voxel_z * 83492791) % 16777216);
    let slot = atomicAdd(&output_count[0], 1u);
    
    voxel_indices[idx] = slot;
    
    if (slot < params.num_points) {
        output_points[slot] = p;
    }
}
"#;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct VoxelParams {
        voxel_size: f32,
        num_points: u32,
        padding: [u32; 2],
    }

    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Params"),
            contents: bytemuck::bytes_of(&VoxelParams {
                voxel_size,
                num_points: num_points as u32,
                padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let voxel_indices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Voxel Indices"),
        size: (num_points * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_points = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Voxel Output Points"),
        size: (num_points * 12) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let count_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Voxel Count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let pipeline = ctx.create_compute_pipeline(voxel_shader, "main");

    let voxel_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Voxel Downsample BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: voxel_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_points.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: count_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Voxel Downsample"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &voxel_bind_group, &[]);
        pass.dispatch_workgroups((num_points as u32).div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    let count_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &count_buffer,
        0,
        4,
    ))?;

    let actual_count = count_data[0] as usize;

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_points), actual_count),
        shape: cv_core::TensorShape::new(1, actual_count, 3),
        dtype: points.dtype,
        _phantom: std::marker::PhantomData,
    })
}

pub fn spatial_hash_correspondences(
    ctx: &GpuContext,
    src: &Tensor<f32, GpuStorage<f32>>,
    tgt: &Tensor<f32, GpuStorage<f32>>,
    max_dist: f32,
    cell_size: f32,
) -> Result<Vec<(usize, usize, f32)>> {
    let num_src = src.shape.height;
    let num_tgt = tgt.shape.height;

    if num_src == 0 || num_tgt == 0 {
        return Ok(Vec::new());
    }

    let num_buckets = 8192u32;
    let max_per_bucket = 32u32;

    let byte_size = (num_src * 16) as u64;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct HashParams {
        num_points: u32,
        cell_size: f32,
        num_buckets: u32,
        max_per_bucket: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct CorrParams {
        num_src: u32,
        num_tgt: u32,
        max_dist_sq: f32,
        cell_size: f32,
        num_buckets: u32,
        max_per_bucket: u32,
    }

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hash Correspondence Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let hash_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hash Params"),
            contents: bytemuck::bytes_of(&HashParams {
                num_points: num_tgt as u32,
                cell_size,
                num_buckets,
                max_per_bucket,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let corr_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Corr Params"),
            contents: bytemuck::bytes_of(&CorrParams {
                num_src: num_src as u32,
                num_tgt: num_tgt as u32,
                max_dist_sq: max_dist * max_dist,
                cell_size,
                num_buckets,
                max_per_bucket,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let hash_counts = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hash Counts"),
        size: (num_buckets * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let hash_buckets = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hash Buckets"),
        size: (num_buckets as usize * max_per_bucket as usize * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let build_shader = r#"
@group(0) @binding(0) var<storage, read> tgt_points: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> params: BuildParams;
@group(0) @binding(2) var<storage, read_write> hash_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hash_buckets: array<u32>;

struct BuildParams {
    num_points: u32,
    cell_size: f32,
    num_buckets: u32,
    max_per_bucket: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let p = tgt_points[idx];
    
    let cell_x = i32(floor(p.x / params.cell_size));
    let cell_y = i32(floor(p.y / params.cell_size));
    let cell_z = i32(floor(p.z / params.cell_size));
    
    let hash = u32((cell_x * 73856093 ^ cell_y * 19349663 ^ cell_z * 83492791) % i32(params.num_buckets));
    let bucket_idx = atomicAdd(&hash_counts[hash], 1u);
    
    if (bucket_idx < params.max_per_bucket) {
        hash_buckets[hash * params.max_per_bucket + bucket_idx] = idx;
    }
}
"#;

    let build_pipeline = ctx.create_compute_pipeline(build_shader, "main");

    let build_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Hash Build BG"),
        layout: &build_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tgt.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hash_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: hash_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hash_buckets.as_entire_binding(),
            },
        ],
    });

    let mut build_enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Build Spatial Hash"),
        });
    {
        let mut pass = build_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&build_pipeline);
        pass.set_bind_group(0, &build_bg, &[]);
        pass.dispatch_workgroups((num_tgt as u32).div_ceil(256), 1, 1);
    }
    ctx.submit(build_enc);

    let correspondence_shader = r#"
struct CorrParams {
    num_src: u32,
    num_tgt: u32,
    max_dist_sq: f32,
    cell_size: f32,
    num_buckets: u32,
    max_per_bucket: u32,
}

@group(0) @binding(0) var<storage, read> src_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> tgt_points: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> hash_data: array<u32>;
@group(0) @binding(3) var<uniform> params: CorrParams;
@group(0) @binding(4) var<storage, read_write> correspondences: array<vec4<f32>>;

fn get_count(hash: u32) -> u32 { return hash_data[hash]; }
fn get_bucket(hash: u32, bucket_idx: u32) -> u32 { return hash_data[16384u + hash * 32u + bucket_idx]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_idx = global_id.x;
    if (src_idx >= params.num_src) { return; }

    let p_s = src_points[src_idx];
    
    var min_dist_sq = params.max_dist_sq;
    var best_tgt_idx = 0u;
    var found = false;

    let cell_x = i32(floor(p_s.x / params.cell_size));
    let cell_y = i32(floor(p_s.y / params.cell_size));
    let cell_z = i32(floor(p_s.z / params.cell_size));

    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
                let hash = u32((i32((cell_x + dx) * 73856093) ^ i32((cell_y + dy) * 19349663) ^ i32((cell_z + dz) * 83492791)) % i32(params.num_buckets));
                let count = min(get_count(hash), params.max_per_bucket);
                
                for (var j = 0u; j < count; j = j + 1u) {
                    let tgt_idx = get_bucket(hash, j);
                    if (tgt_idx >= params.num_tgt) { continue; }
                    
                    let p_t = tgt_points[tgt_idx];
                    let diff = p_s - p_t;
                    let d2 = dot(diff, diff);
                    
                    if (d2 < min_dist_sq) {
                        min_dist_sq = d2;
                        best_tgt_idx = tgt_idx;
                        found = true;
                    }
                }
            }
        }
    }

    if (found) {
        correspondences[src_idx] = vec4<f32>(f32(src_idx), f32(best_tgt_idx), min_dist_sq, 1.0);
    } else {
        correspondences[src_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
"#;

    let corr_pipeline = ctx.create_compute_pipeline(correspondence_shader, "main");

    let corr_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Correspondence BG"),
        layout: &corr_pipeline.get_bind_group_layout(0),
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
                resource: hash_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: corr_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut corr_enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Find Correspondences"),
        });
    {
        let mut pass = corr_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&corr_pipeline);
        pass.set_bind_group(0, &corr_bg, &[]);
        pass.dispatch_workgroups((num_src as u32).div_ceil(64), 1, 1);
    }
    ctx.submit(corr_enc);

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
