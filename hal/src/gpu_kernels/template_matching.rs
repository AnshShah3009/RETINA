use crate::context::TemplateMatchMethod;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatchParams {
    img_w: u32,
    img_h: u32,
    templ_w: u32,
    templ_h: u32,
    method: u32,
}

pub fn match_template(
    ctx: &GpuContext,
    image: &Tensor<u8, GpuStorage<u8>>,
    template: &Tensor<u8, GpuStorage<u8>>,
    method: TemplateMatchMethod,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (img_h, img_w) = image.shape.hw();
    let (templ_h, templ_w) = template.shape.hw();

    let out_w = img_w - templ_w + 1;
    let out_h = img_h - templ_h + 1;
    let out_len = out_w * out_h;

    let byte_size = (out_len * 4) as u64;
    let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let params = MatchParams {
        img_w: img_w as u32,
        img_h: img_h as u32,
        templ_w: templ_w as u32,
        templ_h: templ_h as u32,
        method: match method {
            TemplateMatchMethod::SqDiff => 0,
            TemplateMatchMethod::SqDiffNormed => 1,
            TemplateMatchMethod::Ccorr => 2,
            TemplateMatchMethod::CcorrNormed => 3,
            TemplateMatchMethod::Ccoeff => 4,
            TemplateMatchMethod::CcoeffNormed => 5,
        },
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Match Template Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/match_template.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Match Template Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: image.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: template.storage.buffer().as_entire_binding(),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Match Template Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (out_w as u32).div_ceil(16);
        let y = (out_h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: TensorShape::new(1, out_h, out_w),
        dtype: cv_core::DataType::F32,
        _phantom: PhantomData,
    })
}
