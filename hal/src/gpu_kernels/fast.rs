use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    width: u32,
    height: u32,
    threshold: u32,
}

pub fn fast_detect(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    threshold: u8,
    non_max_suppression: bool,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU FAST currently only for grayscale".into(),
        ));
    }

    let out_len = w * h;
    let byte_size = (out_len.div_ceil(4) * 4) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FAST Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = FastParams {
        width: w as u32,
        height: h as u32,
        threshold: threshold as u32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FAST Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/fast.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FAST Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
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
        let x = (w as u32).div_ceil(4).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let score_map = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    };

    if non_max_suppression {
        let nms_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FAST NMS Output"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let nms_shader_source = include_str!("../../shaders/fast_nms.wgsl");
        let nms_pipeline = ctx.create_compute_pipeline(nms_shader_source, "main");

        let nms_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FAST NMS Bind Group"),
            layout: &nms_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_map.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut nms_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = nms_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&nms_pipeline);
            pass.set_bind_group(0, &nms_bind_group, &[]);
            let x = (w as u32).div_ceil(4).div_ceil(16);
            let y = (h as u32).div_ceil(16);
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.submit(nms_encoder);

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(nms_buffer), out_len),
            shape: score_map.shape,
            dtype: score_map.dtype,
            _phantom: PhantomData,
        })
    } else {
        Ok(score_map)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CollectParams {
    width: u32,
    height: u32,
    num_elements: u32,
    padding: u32,
}

pub fn extract_keypoints(
    ctx: &GpuContext,
    score_map: &Tensor<u8, GpuStorage<u8>>,
) -> Result<Vec<cv_core::KeyPointF32>> {
    let (h, w) = score_map.shape.hw();
    let num_pixels = w * h;
    if num_pixels == 0 {
        return Ok(Vec::new());
    }
    let num_u32 = num_pixels.div_ceil(4);

    // 1. Count pass
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let counts_buffer = ctx.get_buffer((num_u32 as u64) * 4, usages);

    let params = CollectParams {
        width: w as u32,
        height: h as u32,
        num_elements: num_u32 as u32,
        padding: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Collect Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("feature_collection.wgsl");
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feature Collection Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let count_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Count Points Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("count_points"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Count Points"),
        });
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Count Bind Group"),
            layout: &count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_map.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&count_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_u32 as u32).div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    // 2. Scan pass
    // Read the total count from CPU for now (most reliable way to allocate exact result buffer)
    let counts_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buffer,
        0,
        num_u32 * 4,
    ))?;

    let mut total = 0u32;
    let mut indices = Vec::with_capacity(num_u32);
    for &c in &counts_data {
        indices.push(total);
        if c <= 4 {
            // packed u32 check
            total += c;
        }
    }

    if total == 0 {
        ctx.return_buffer(counts_buffer, usages);
        return Ok(Vec::new());
    }

    if total > 1000000 {
        ctx.return_buffer(counts_buffer, usages);
        return Err(crate::Error::InvalidInput(format!(
            "Too many keypoints detected: {}",
            total
        )));
    }

    // Upload scanned indices
    ctx.queue
        .write_buffer(&counts_buffer, 0, bytemuck::cast_slice(&indices));

    // 3. Collect pass
    let kp_size = std::mem::size_of::<cv_core::KeyPointF32>();
    let kp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extracted Keypoints"),
        size: (total as u64) * kp_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let collect_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collect Points Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("collect_points"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Collect Points"),
        });
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collect Bind Group"),
            layout: &collect_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_map.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&collect_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_u32 as u32).div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    // Read back keypoints
    let kps = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &kp_buffer,
        0,
        (total as usize) * kp_size,
    ))?;

    ctx.return_buffer(counts_buffer, usages);

    Ok(kps)
}
