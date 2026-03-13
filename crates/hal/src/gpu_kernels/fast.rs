use crate::gpu::GpuContext;
use crate::Result;
use cv_core::storage::Storage;
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

pub fn fast_detect<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    threshold: T,
    nonmax_suppression: bool,
) -> Result<crate::GpuTensor<T>> {
    use crate::storage::GpuStorage;
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "FAST detection expects 1 channel image".into(),
        ));
    }

    let out_len = w * h;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FAST Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_data: [u32; 4] = [
        w as u32,
        h as u32,
        threshold.to_f32() as u32,
        if nonmax_suppression { 1u32 } else { 0u32 },
    ];

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FAST Params"),
            contents: bytemuck::bytes_of(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = match cv_core::DataType::from_type::<T>() {
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::F16) => include_str!("../../shaders/fast_f16.wgsl"),
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::BF16) => include_str!("../../shaders/fast_bf16.wgsl"),
        _ => {
            include_str!("../../shaders/fast_f32.wgsl")
        }
    };
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FAST Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>()
                    .unwrap()
                    .buffer()
                    .as_entire_binding(),
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
        let x = (w as u32).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let score_map = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData::<T>,
    };

    if nonmax_suppression {
        let nms_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FAST NMS Output"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_source_nms = match cv_core::DataType::from_type::<T>() {
            Ok(cv_core::DataType::F32) => include_str!("../../shaders/fast_nms_f32.wgsl"),
            Ok(_) => {
                return Err(crate::Error::NotSupported(
                    "Unsupported fast precision type".into(),
                ))
            }
            _ => {
                include_str!("../../shaders/fast_nms_f32.wgsl")
            }
        };
        let pipeline_nms = ctx.create_compute_pipeline(shader_source_nms, "main");

        let nms_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FAST NMS Bind Group"),
            layout: &pipeline_nms.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_map
                        .storage
                        .as_any()
                        .downcast_ref::<crate::storage::GpuStorage<T>>()
                        .unwrap()
                        .buffer()
                        .as_entire_binding(),
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
            pass.set_pipeline(&pipeline_nms);
            pass.set_bind_group(0, &nms_bind_group, &[]);
            let x = (w as u32).div_ceil(16);
            let y = (h as u32).div_ceil(16);
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.submit(nms_encoder);

        let nms_map = crate::GpuTensor {
            storage: GpuStorage::from_buffer(Arc::new(nms_buffer), out_len),
            shape: score_map.shape,
            dtype: cv_core::DataType::from_type::<T>().unwrap(),
            _phantom: PhantomData::<T>,
        };
        Ok(nms_map)
    } else {
        let res_gpu = crate::GpuTensor {
            storage: score_map.storage.clone(),
            shape: score_map.shape,
            dtype: cv_core::DataType::from_type::<T>().unwrap(),
            _phantom: PhantomData::<T>,
        };
        Ok(res_gpu)
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

pub fn extract_keypoints<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    score_map: &crate::GpuTensor<T>,
) -> Result<Vec<cv_core::KeyPointF32>> {
    use crate::storage::GpuStorage;
    let (h, w) = score_map.shape.hw();
    let num_pixels: u32 = (w * h) as u32;
    if num_pixels == 0 {
        return Ok(Vec::new());
    }

    // Buffer to get number of results
    let num_u32 = num_pixels.div_ceil(4);

    // 1. Count pass
    let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC; // Create a host accessible buffer. This depends on WGSL structs being packed to bytes, not float sizes of the GPU logic!
    let counts_buffer = ctx.get_buffer((num_u32 as u64) * 4, usages);

    let params = CollectParams {
        width: w as u32,
        height: h as u32,
        num_elements: num_u32,
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
                    resource: score_map
                        .storage
                        .as_any()
                        .downcast_ref::<GpuStorage<f32>>()
                        .unwrap()
                        .buffer()
                        .as_entire_binding(),
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
        pass.dispatch_workgroups(num_u32.div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    // 2. Scan pass
    let counts_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buffer,
        0,
        (num_u32 as usize) * 4,
    ))?;

    let mut total = 0u32;
    let mut indices = Vec::with_capacity(num_u32 as usize);
    for &c in &counts_data {
        indices.push(total);
        if c <= 4 {
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
                    resource: score_map
                        .storage
                        .as_any()
                        .downcast_ref::<GpuStorage<f32>>()
                        .unwrap()
                        .buffer()
                        .as_entire_binding(),
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
        pass.dispatch_workgroups(num_u32.div_ceil(256), 1, 1);
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
