use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SortParams {
    num_elements: u32,
    shift: u32,
    num_workgroups: u32,
    padding: u32,
}

pub fn radix_sort_u32(
    ctx: &GpuContext,
    input: &Tensor<u32, GpuStorage<u32>>,
) -> Result<Tensor<u32, GpuStorage<u32>>> {
    let num_elements = input.shape.len() as u32;
    if num_elements == 0 {
        return Ok(input.clone());
    }

    let workgroup_size = 256;
    let num_workgroups = num_elements.div_ceil(workgroup_size);
    let histogram_size = num_workgroups * 256;

    let buffer_size = (num_elements as u64) * 4;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let temp_buffer = ctx.get_buffer(buffer_size, usages);
    let histogram_buffer = ctx.get_buffer((histogram_size as u64) * 4, usages);

    let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Params"),
        size: std::mem::size_of::<SortParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load Shaders
    let sort_source = include_str!("radix_sort_keys.wgsl");
    let scan_source = include_str!("prefix_sum.wgsl");

    let result_buffer = ctx.get_buffer(buffer_size, usages);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Sort Input"),
        });
    encoder.copy_buffer_to_buffer(input.storage.buffer(), 0, &result_buffer, 0, buffer_size);
    ctx.submit(encoder);

    let mut in_ref: &wgpu::Buffer = &result_buffer;
    let mut out_ref: &wgpu::Buffer = &temp_buffer;

    let mut loop_res = Ok(());
    for pass in 0..4 {
        let shift = pass * 8;
        let params = SortParams {
            num_elements,
            shift,
            num_workgroups,
            padding: 0,
        };
        ctx.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // 1. Histogram Pass
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Histogram Pass"),
            });
        encoder.clear_buffer(&histogram_buffer, 0, None);

        let histogram_pipeline = ctx.create_compute_pipeline(sort_source, "histogram");
        let bg_hist = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hist BG"),
            layout: &histogram_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&histogram_pipeline);
            cpass.set_bind_group(0, &bg_hist, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.submit(encoder);

        // 2. GPU-Side Scan Histogram
        if let Err(e) = gpu_exclusive_scan(ctx, &histogram_buffer, histogram_size, scan_source) {
            loop_res = Err(e);
            break;
        }

        // 3. Scatter Pass
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Scatter Pass"),
            });
        let scatter_pipeline = ctx.create_compute_pipeline(sort_source, "scatter");
        let bg_scatter = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter BG"),
            layout: &scatter_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&scatter_pipeline);
            cpass.set_bind_group(0, &bg_scatter, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.submit(encoder);

        std::mem::swap(&mut in_ref, &mut out_ref);
    }

    let (final_buf, other_buf) = if std::ptr::eq(in_ref, &result_buffer) {
        (result_buffer, temp_buffer)
    } else {
        (temp_buffer, result_buffer)
    };

    ctx.return_buffer(other_buf, usages);
    ctx.return_buffer(histogram_buffer, usages);

    loop_res?;

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(final_buf), num_elements as usize),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: std::marker::PhantomData,
    })
}

pub fn radix_sort_key_value_u32(
    ctx: &GpuContext,
    keys: &Tensor<u32, GpuStorage<u32>>,
    values: &Tensor<u32, GpuStorage<u32>>,
) -> Result<(Tensor<u32, GpuStorage<u32>>, Tensor<u32, GpuStorage<u32>>)> {
    let num_elements = keys.shape.len() as u32;
    if num_elements == 0 {
        return Ok((keys.clone(), values.clone()));
    }

    let workgroup_size = 256;
    let num_workgroups = num_elements.div_ceil(workgroup_size);
    let histogram_size = num_workgroups * 256;

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    // We use packed Key-Value buffers to fit within iGPU storage limits (max 4 per stage)
    let kv_buffer_size = (num_elements as u64) * 8; // 8 bytes per KV pair
    let in_kv_buf = ctx.get_buffer(kv_buffer_size, usages);
    let out_kv_buf = ctx.get_buffer(kv_buffer_size, usages);
    let histogram_buffer = ctx.get_buffer((histogram_size as u64) * 4, usages);

    let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Params"),
        size: std::mem::size_of::<SortParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let sort_source = include_str!("radix_sort_kv.wgsl");
    let scan_source = include_str!("prefix_sum.wgsl");

    // 1. Pack Pass
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("KV Pack"),
        });
    let pack_pipeline = ctx.create_compute_pipeline(sort_source, "pack");

    // Using a special params for packing
    let pack_params = SortParams {
        num_elements,
        shift: 0,
        num_workgroups: 0,
        padding: 0,
    };
    ctx.queue
        .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&pack_params));

    let bg_pack = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pack BG"),
        layout: &pack_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: keys.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: values.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: in_kv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pack_pipeline);
        cpass.set_bind_group(0, &bg_pack, &[]);
        cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    let mut in_ref: &wgpu::Buffer = &in_kv_buf;
    let mut out_ref: &wgpu::Buffer = &out_kv_buf;

    let mut loop_res = Ok(());
    for pass in 0..4 {
        let shift = pass * 8;
        let params = SortParams {
            num_elements,
            shift,
            num_workgroups,
            padding: 0,
        };
        ctx.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Histogram Pass
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&histogram_buffer, 0, None);
        let histogram_pipeline = ctx.create_compute_pipeline(sort_source, "histogram");
        let bg_hist = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hist BG"),
            layout: &histogram_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&histogram_pipeline);
            cpass.set_bind_group(0, &bg_hist, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.submit(encoder);

        if let Err(e) = gpu_exclusive_scan(ctx, &histogram_buffer, histogram_size, scan_source) {
            loop_res = Err(e);
            break;
        }

        // Scatter Pass
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let scatter_pipeline = ctx.create_compute_pipeline(sort_source, "scatter");
        let bg_scatter = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter BG"),
            layout: &scatter_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&scatter_pipeline);
            cpass.set_bind_group(0, &bg_scatter, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.submit(encoder);

        std::mem::swap(&mut in_ref, &mut out_ref);
    }

    // Unpack Pass
    let result_keys_buf = ctx.get_buffer((num_elements as u64) * 4, usages);
    let result_values_buf = ctx.get_buffer((num_elements as u64) * 4, usages);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("KV Unpack"),
        });
    let unpack_pipeline = ctx.create_compute_pipeline(sort_source, "unpack");
    let bg_unpack = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Unpack BG"),
        layout: &unpack_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_ref.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_keys_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_values_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&unpack_pipeline);
        cpass.set_bind_group(0, &bg_unpack, &[]);
        cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    // Return temporary packed buffers
    ctx.return_buffer(in_kv_buf, usages);
    ctx.return_buffer(out_kv_buf, usages);
    ctx.return_buffer(histogram_buffer, usages);

    loop_res?;

    Ok((
        Tensor {
            storage: GpuStorage::from_buffer(Arc::new(result_keys_buf), num_elements as usize),
            shape: keys.shape,
            dtype: keys.dtype,
            _phantom: std::marker::PhantomData,
        },
        Tensor {
            storage: GpuStorage::from_buffer(Arc::new(result_values_buf), num_elements as usize),
            shape: values.shape,
            dtype: values.dtype,
            _phantom: std::marker::PhantomData,
        },
    ))
}

/// Recursively performs an exclusive prefix sum on the GPU.
pub fn gpu_exclusive_scan(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    num_elements: u32,
    scan_source: &str,
) -> Result<()> {
    if num_elements == 0 {
        return Ok(());
    }

    let block_size = 512;
    let num_workgroups = num_elements.div_ceil(block_size);

    // Create temporary block sums buffer
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let block_sums_buffer = ctx.get_buffer((num_workgroups as u64) * 4, usages);

    let n_elements_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scan NumElements"),
            contents: bytemuck::bytes_of(&num_elements),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    // 1. Dispatch Block Scan
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Scan Blocks"),
        });
    let scan_pipeline = ctx.create_compute_pipeline(scan_source, "scan_blocks");

    let bg_scan = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &scan_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: block_sums_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: n_elements_buffer.as_entire_binding(),
            },
        ],
        label: Some("Scan BG"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&scan_pipeline);
        cpass.set_bind_group(0, &bg_scan, &[]);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    ctx.submit(encoder);

    // 2. Scan Block Sums (Recursive)
    if num_workgroups > 1 {
        gpu_exclusive_scan(ctx, &block_sums_buffer, num_workgroups, scan_source)?;

        // 3. Add Offsets Back
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Add Offsets"),
            });
        let add_pipeline = ctx.create_compute_pipeline(scan_source, "add_offsets");

        let bg_add = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &add_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: n_elements_buffer.as_entire_binding(),
                },
            ],
            label: Some("Add BG"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&add_pipeline);
            cpass.set_bind_group(0, &bg_add, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.submit(encoder);
    }

    ctx.return_buffer(block_sums_buffer, usages);
    Ok(())
}
