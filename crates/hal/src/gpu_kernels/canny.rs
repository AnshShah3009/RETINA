use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use crate::Result;
use cv_core::Float;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CannyParams {
    width: u32,
    height: u32,
    low_threshold: f32,
    high_threshold: f32,
}

pub fn canny<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    low_threshold: T,
    high_threshold: T,
) -> Result<crate::GpuTensor<T>> {
    // Only f32 shader is available; return NotSupported for other types
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Canny GPU kernel only supports f32; other types not yet implemented".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let len = input.shape.len();
    let byte_size_f32 = (len * 4) as u64;

    // 1. Intermediate buffers (pooled)
    let mag_buffer = ctx.get_buffer(byte_size_f32, wgpu::BufferUsages::STORAGE);
    let dir_buffer = ctx.get_buffer(byte_size_f32, wgpu::BufferUsages::STORAGE);
    let nms_buffer = ctx.get_buffer(byte_size_f32, wgpu::BufferUsages::STORAGE);
    let final_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Canny Final"),
        size: byte_size_f32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = CannyParams {
        width: w as u32,
        height: h as u32,
        low_threshold: low_threshold.to_f32(),
        high_threshold: high_threshold.to_f32(),
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/canny.wgsl");
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Canny Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let gradients_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Canny Gradients"),
            layout: None,
            module: &shader_module,
            entry_point: Some("gradients"),
            compilation_options: Default::default(),
            cache: None,
        });

    let nms_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Canny NMS"),
            layout: None,
            module: &shader_module,
            entry_point: Some("nms"),
            compilation_options: Default::default(),
            cache: None,
        });

    let hysteresis_pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Canny Hysteresis"),
                layout: None,
                module: &shader_module,
                entry_point: Some("hysteresis"),
                compilation_options: Default::default(),
                cache: None,
            });

    // Pass 1: Gradients
    let bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny BG 1"),
        layout: &gradients_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mag_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dir_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Pass 2: NMS
    let bind_group_2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny BG 2"),
        layout: &nms_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: mag_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dir_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: nms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Pass 3: Hysteresis
    let bind_group_3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny BG 3"),
        layout: &hysteresis_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: final_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Canny Dispatch"),
        });
    let wg_x = (w as u32).div_ceil(16);
    let wg_y = (h as u32).div_ceil(16);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gradients"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&gradients_pipeline);
        pass.set_bind_group(0, &bind_group_1, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("NMS"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&nms_pipeline);
        pass.set_bind_group(0, &bind_group_2, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hysteresis"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&hysteresis_pipeline);
        pass.set_bind_group(0, &bind_group_3, &[]);
        let wg_x_hyst = (w as u32).div_ceil(4).div_ceil(16);
        pass.dispatch_workgroups(wg_x_hyst, wg_y, 1);
    }

    ctx.submit(encoder);

    // Return intermediate buffers to pool
    ctx.return_buffer(mag_buffer, wgpu::BufferUsages::STORAGE);
    ctx.return_buffer(dir_buffer, wgpu::BufferUsages::STORAGE);
    ctx.return_buffer(nms_buffer, wgpu::BufferUsages::STORAGE);

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(final_buffer), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
