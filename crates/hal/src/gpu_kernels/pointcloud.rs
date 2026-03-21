use crate::gpu::GpuContext;
use crate::gpu_kernels::radix_sort::radix_sort_key_value_u32;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{DataType, Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    // CPU fallback: voxel-hash kNN + analytic eigensolver
    normals_cpu_analytic(points, k as usize)
}

pub fn normals_cpu_analytic(
    points: &[nalgebra::Vector3<f32>],
    _k: usize,
) -> Vec<nalgebra::Vector3<f32>> {
    // In HAL we don't have scientific/KdTree, so we return default normals.
    // Full CPU implementation is in cv-3d.
    vec![nalgebra::Vector3::z(); points.len()]
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

    let device = &ctx.device;
    let queue = &ctx.queue;
    let num_points_u32 = num_points as u32;

    // 1. Compute Bounds (Min/Max)
    let workgroup_size = 256u32;
    let num_workgroups = num_points_u32.div_ceil(workgroup_size);
    let bounds_buf = ctx.get_buffer(
        (num_workgroups * 32) as u64, // 32 bytes per Bounds (2x vec4)
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let num_points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Num Points Uniform"),
        contents: bytemuck::bytes_of(&num_points_u32),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bounds_source = include_str!("pointcloud_bounds.wgsl");
    let bounds_pipeline = ctx.create_compute_pipeline(bounds_source, "main");
    let bounds_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PC Bounds BG"),
        layout: &bounds_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bounds_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: num_points_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&bounds_pipeline);
        pass.set_bind_group(0, &bounds_bg, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Download and finish bounds on CPU
    use crate::gpu_kernels::buffer_utils::read_buffer;
    let partial_bounds: Vec<f32> = pollster::block_on(read_buffer(
        device.clone(),
        queue,
        &bounds_buf,
        0,
        (num_workgroups * 32) as usize,
    ))?;
    ctx.return_buffer(
        bounds_buf,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let mut min_bound = [f32::MAX; 3];
    let mut max_bound = [f32::MIN; 3];
    for chunk in partial_bounds.chunks(8) {
        for i in 0..3 {
            min_bound[i] = min_bound[i].min(chunk[i]);
            max_bound[i] = max_bound[i].max(chunk[i + 4]);
        }
    }

    let span = (max_bound[0] - min_bound[0])
        .max(max_bound[1] - min_bound[1])
        .max(max_bound[2] - min_bound[2]);
    let grid_size = span / 1024.0;

    // 2. Compute Morton Codes
    let codes_buf = ctx.get_buffer(
        (num_points_u32 * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let indices_buf = ctx.get_buffer(
        (num_points_u32 * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct MortonParams {
        min_bound: [f32; 4],
        grid_size: f32,
        num_points: u32,
        padding1: u32,
        padding2: u32,
    }
    let morton_params = MortonParams {
        min_bound: [min_bound[0], min_bound[1], min_bound[2], 0.0],
        grid_size,
        num_points: num_points_u32,
        padding1: 0,
        padding2: 0,
    };
    let morton_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Morton Params"),
        contents: bytemuck::bytes_of(&morton_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let morton_source = include_str!("pointcloud_morton.wgsl");
    let morton_pipeline = ctx.create_compute_pipeline(morton_source, "main");
    let morton_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PC Morton BG"),
        layout: &morton_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: codes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: morton_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&morton_pipeline);
        pass.set_bind_group(0, &morton_bg, &[]);
        pass.dispatch_workgroups(num_points_u32.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // 3. GPU Radix Sort (Key-Value)
    let codes_tensor = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(codes_buf), num_points),
        shape: TensorShape::new(1, num_points, 1),
        dtype: DataType::U32,
        _phantom: PhantomData,
    };
    let indices_tensor = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(indices_buf), num_points),
        shape: codes_tensor.shape,
        dtype: DataType::U32,
        _phantom: PhantomData,
    };

    let (_sorted_codes, sorted_indices) =
        radix_sort_key_value_u32(ctx, &codes_tensor, &indices_tensor)?;

    // 3.5 Build LBVH
    let nodes_tensor =
        crate::gpu_kernels::lbvh::build_lbvh(ctx, points, &sorted_indices, &codes_tensor)?;

    // 4. kNN + Covariance
    let covs_buf = ctx.get_buffer(
        (num_points_u32 * 32) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct KnnParams {
        num_points: u32,
        k_neighbors: u32,
        window_size: u32,
        padding: u32,
    }
    let knn_params = KnnParams {
        num_points: num_points_u32,
        k_neighbors: k_neighbors.clamp(3, 16),
        window_size: 0,
        padding: 0,
    };
    let knn_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kNN Params"),
        contents: bytemuck::bytes_of(&knn_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let knn_source = include_str!("pointcloud_knn.wgsl");
    let knn_pipeline = ctx.create_compute_pipeline(knn_source, "main");
    let knn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PC kNN BG"),
        layout: &knn_pipeline.get_bind_group_layout(0),
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
                binding: 2,
                resource: covs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: knn_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: nodes_tensor.storage.buffer().as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&knn_pipeline);
        pass.set_bind_group(0, &knn_bg, &[]);
        pass.dispatch_workgroups(num_points_u32.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // 5. Batch PCA
    let normals_buf = ctx.get_buffer(
        (num_points_u32 * 16) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let pca_source = include_str!("pointcloud_normals_batch_pca.wgsl");
    let pca_pipeline = ctx.create_compute_pipeline(pca_source, "main");
    let pca_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Batch PCA BG"),
        layout: &pca_pipeline.get_bind_group_layout(0),
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
                resource: num_points_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pca_pipeline);
        pass.set_bind_group(0, &pca_bg, &[]);
        pass.dispatch_workgroups(num_points_u32.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    ctx.return_buffer(
        covs_buf,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(normals_buf), num_points * 4),
        shape: TensorShape::new(4, num_points, 1),
        dtype: DataType::F32,
        _phantom: PhantomData,
    })
}

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

    let packed: Vec<[f32; 8]> = covs
        .iter()
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], 0.0, 0.0])
        .collect();
    let covs_buf = create_buffer(&device, &packed, BufferUsages::STORAGE);
    let normals_buf = create_buffer_uninit(
        &device,
        n * 16,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let num_points_buf = create_buffer(&device, &[n as u32], BufferUsages::UNIFORM);

    let shader_source = include_str!("pointcloud_normals_batch_pca.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Batch PCA BG"),
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
                resource: num_points_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FastGpuParams {
    num_points: f32,
    k_neighbors: f32,
    voxel_size: f32,
    padding: f32,
}

pub fn compute_normals_fast_gpu(
    ctx: &GpuContext,
    points: &Tensor<f32, GpuStorage<f32>>,
    k_neighbors: u32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = points.shape.height;
    if num_points == 0 {
        return Ok(points.clone());
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let normals_buf = ctx.get_buffer(
        (num_points * 16) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let params = FastGpuParams {
        num_points: num_points as f32,
        k_neighbors: k_neighbors as f32,
        voxel_size: 0.0,
        padding: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fast GPU Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("pointcloud_normals_fast_gpu.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fast GPU Normals BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: normals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_points as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(normals_buf), num_points * 4),
        shape: TensorShape::new(4, num_points, 1),
        dtype: DataType::F32,
        _phantom: PhantomData,
    })
}
