use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{DataType, Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LbvhNode {
    pub parent: i32,
    pub left: i32,
    pub right: i32,
    pub padding: i32,
    pub min_bound: [f32; 4],
    pub max_bound: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LbvhParams {
    num_elements: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
}

pub fn build_lbvh(
    ctx: &GpuContext,
    points: &Tensor<f32, GpuStorage<f32>>,
    sorted_indices: &Tensor<u32, GpuStorage<u32>>,
    morton_codes: &Tensor<u32, GpuStorage<u32>>,
) -> Result<Tensor<i32, GpuStorage<i32>>> {
    let num_elements = morton_codes.shape.height as u32;
    if num_elements < 2 {
        return Err(crate::Error::InvalidInput(
            "LBVH requires at least 2 points".into(),
        ));
    }

    let num_nodes = 2 * num_elements - 1;
    let node_buffer_size = (num_nodes as u64) * std::mem::size_of::<LbvhNode>() as u64;

    let buffer = ctx.get_buffer(
        node_buffer_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    let storage = GpuStorage::from_buffer(Arc::new(buffer), (num_nodes * 12) as usize);
    let shape = TensorShape::new(12, num_nodes as usize, 1);

    let nodes_tensor = Tensor {
        storage,
        shape,
        dtype: DataType::I32,
        _phantom: PhantomData,
    };

    let counters_buffer = ctx.get_buffer(
        (num_elements as u64) * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let params = LbvhParams {
        num_elements,
        padding1: 0,
        padding2: 0,
        padding3: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LBVH Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let shader_source = include_str!("../../shaders/lbvh_build.wgsl");
    let pipeline_init = ctx.create_compute_pipeline(shader_source, "init_nodes");
    let pipeline_tree = ctx.create_compute_pipeline(shader_source, "build_radix_tree");
    let pipeline_aabb = ctx.create_compute_pipeline(shader_source, "compute_aabbs");

    // Init/Tree Bind Group (uses 2, 3, 5) - we reuse the same layout
    let bg_tree = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LBVH Tree Bind Group"),
        layout: &pipeline_tree.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: morton_codes.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: nodes_tensor.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // AABB Bind Group (uses 0, 1, 3, 4, 5)
    let bg_aabb = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LBVH AABB Bind Group"),
        layout: &pipeline_aabb.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sorted_indices.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: nodes_tensor.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: counters_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LBVH Build Encoder"),
        });

    // Initialize counters
    encoder.clear_buffer(&counters_buffer, 0, None);

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBVH Build Pass"),
            timestamp_writes: None,
        });

        // 0. Initialize parent pointers to -1
        compute_pass.set_pipeline(&pipeline_init);
        compute_pass.set_bind_group(0, &bg_tree, &[]); // Reuses bindings 3, 5
        compute_pass.dispatch_workgroups(num_nodes.div_ceil(256), 1, 1);

        // 1. Build tree structure
        compute_pass.set_pipeline(&pipeline_tree);
        compute_pass.set_bind_group(0, &bg_tree, &[]);
        let workgroup_count_tree = (num_elements - 1).div_ceil(256);
        if workgroup_count_tree > 0 {
            compute_pass.dispatch_workgroups(workgroup_count_tree, 1, 1);
        }

        // 2. Compute AABBs
        compute_pass.set_pipeline(&pipeline_aabb);
        compute_pass.set_bind_group(0, &bg_aabb, &[]);
        let workgroup_count_aabb = num_elements.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroup_count_aabb, 1, 1);
    }

    ctx.submit(encoder);

    ctx.return_buffer(
        counters_buffer,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    Ok(nodes_tensor)
}
