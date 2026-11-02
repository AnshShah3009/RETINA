use crate::Result;
use crate::context::TemplateMatchMethod;
use crate::cpu::template_match::{TemplateTerms, is_normed};
use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use cv_core::{Float, TensorShape};
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
    channels: u32,
    /// `|T|` of the normalised methods (`TemplateTerms::norm`).
    t_norm: f32,
    _pad: u32,
}

/// `cv::matchTemplate` on the GPU, every method, any channel count.
///
/// Image and template are planar CHW with the same channel count; the score
/// map is one channel with every term summed over the channels, and the
/// normalised methods follow OpenCV's rule, degenerate windows included -
/// see `cpu::template_match`, which also prepares the template here
/// (`TemplateTerms`: centred per channel for the coefficient methods, and its
/// norm). The template is read back for that; it is the small operand. Each
/// invocation scores one window in `f32`, with the coefficient methods'
/// window variance taken about the window mean (two passes) rather than as
/// `sum(I^2) - sum(I)^2 / n`, which cancels badly in `f32`.
///
/// The kernel used to read plane 0 only and to compute `SqDiff` and `Ccorr`
/// alone: every other method returned a map of zeros.
pub fn match_template<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    image: &crate::GpuTensor<T>,
    template: &crate::GpuTensor<T>,
    method: TemplateMatchMethod,
) -> Result<crate::GpuTensor<T>> {
    // Only f32 WGSL shader available
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Template Matching GPU kernel only supports f32".into(),
        ));
    }

    let (img_h, img_w) = image.shape.hw();
    let (templ_h, templ_w) = template.shape.hw();
    let channels = image.shape.channels;

    // The CPU backend's checks, in its order. `cv::matchTemplate` rejects an
    // empty image or template (and an empty tensor is a zero-size buffer,
    // which cannot be bound), and requires the template's channel count to
    // be the image's.
    if image.shape.len() == 0 || template.shape.len() == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "match_template: empty image {:?} or template {:?}",
            image.shape, template.shape
        )));
    }
    if template.shape.channels != channels {
        return Err(crate::Error::InvalidInput(format!(
            "match_template: the image has {} channels, the template {}",
            channels, template.shape.channels
        )));
    }
    // A template larger than the image would underflow the output-size
    // subtraction below (debug panic / release over-allocation).
    if templ_w > img_w || templ_h > img_h {
        return Err(crate::Error::InvalidInput(format!(
            "Template ({}x{}) is larger than the image ({}x{})",
            templ_w, templ_h, img_w, img_h
        )));
    }

    let out_w = img_w - templ_w + 1;
    let out_h = img_h - templ_h + 1;
    let out_len = out_w * out_h;
    let out_shape = TensorShape::new(1, out_h, out_w);

    let t_plane = templ_w * templ_h;
    let tmpl: Vec<f32> = pollster::block_on(super::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        template.storage.buffer(),
        0,
        channels * t_plane * std::mem::size_of::<f32>(),
    ))?;
    let tmpl64: Vec<f64> = tmpl.iter().map(|&v| v as f64).collect();
    let terms = TemplateTerms::new(&tmpl64, channels, t_plane, method);
    let output = |buffer: wgpu::Buffer| cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(buffer), out_len),
        shape: out_shape,
        dtype: image.dtype,
        _phantom: PhantomData,
    };
    if terms.constant_ccoeff_normed {
        // OpenCV: 1 in every window, no correlation to compute.
        let ones = vec![1.0f32; out_len];
        return Ok(output(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Match Template Output"),
                contents: bytemuck::cast_slice(&ones),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        )));
    }
    let corr: Vec<f32> = terms.corr_template.iter().map(|&v| v as f32).collect();
    let corr_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Match Template Correlation Template"),
            contents: bytemuck::cast_slice(&corr),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let byte_size = (out_len * std::mem::size_of::<f32>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Match Template Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

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
        channels: channels as u32,
        t_norm: if is_normed(method) {
            terms.norm as f32
        } else {
            0.0
        },
        _pad: 0,
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
                resource: corr_buffer.as_entire_binding(),
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

    Ok(output(output_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::TemplateMatchMethod;
    use crate::gpu::GpuContext;
    use crate::tensor_ext::TensorToGpu;

    #[test]
    fn oversized_template_returns_error() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // no adapter available
        };

        let image =
            cv_core::CpuTensor::<f32>::from_vec(vec![0.0f32; 16], TensorShape::new(1, 4, 4))
                .unwrap()
                .to_gpu_ctx(&ctx)
                .unwrap();
        // Template (8x8) larger than the image (4x4) used to underflow.
        let template =
            cv_core::CpuTensor::<f32>::from_vec(vec![0.0f32; 64], TensorShape::new(1, 8, 8))
                .unwrap()
                .to_gpu_ctx(&ctx)
                .unwrap();

        let res = match_template(&ctx, &image, &template, TemplateMatchMethod::SqDiff);
        assert!(matches!(res, Err(crate::Error::InvalidInput(_))));
    }
}
