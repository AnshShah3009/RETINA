use crate::GpuTensor;
use crate::Result;
use crate::gpu::GpuContext;
use cv_core::TensorShape;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResizeParams {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub channels: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32, // pad to 32 bytes for WGSL uniform alignment
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Bilinear,
    Lanczos4,
}

pub fn resize<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
) -> Result<GpuTensor<T>> {
    resize_with_mode(
        ctx,
        input,
        new_width,
        new_height,
        InterpolationMode::Bilinear,
    )
}

/// Resize a planar CHW tensor on the GPU.
///
/// Element `(ch, y, x)` of a `TensorShape::new(c, h, w)` tensor is at
/// `ch * h * w + y * w + x`, as everywhere else in the workspace
/// (`Tensor::index`, `Tensor::from_image_rgb`, `gaussian_blur`, `cvt_color`);
/// each plane is resampled as an independent image.
///
/// A zero target size (or zero channels) returns an empty tensor of the
/// target shape, and an empty source with a non-empty target returns zeros of
/// the target shape, both without a dispatch. That is the CPU backend's
/// behaviour.
pub fn resize_with_mode<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
    mode: InterpolationMode,
) -> Result<GpuTensor<T>> {
    // Only f32 shaders are wired up (`resize_f16.wgsl` and `resize_bf16.wgsl`
    // have no caller), so other dtypes are refused.
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Resize GPU kernel only supports f32".into(),
        ));
    }
    let (src_h, src_w) = input.shape.hw();
    let c = input.shape.channels;
    let out_shape = TensorShape::new(c, new_height as usize, new_width as usize);
    let out_len = out_shape
        .checked_len()
        .ok_or_else(|| crate::Error::InvalidInput("Resize target size overflows".into()))?;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;

    // wgpu zero-initialises a new buffer, so an empty source needs no
    // dispatch to come back as zeros. A storage binding may not be empty,
    // hence the four-byte floor.
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resize Output Buffer"),
        size: byte_size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: out_shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    };
    if out_len == 0 || src_h == 0 || src_w == 0 {
        return Ok(output);
    }

    let params = ResizeParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w: new_width,
        dst_h: new_height,
        channels: c as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resize Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = match mode {
        InterpolationMode::Bilinear => include_str!("../../shaders/resize_f32.wgsl"),
        InterpolationMode::Lanczos4 => include_str!("../../shaders/lanczos4.wgsl"),
    };
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resize Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.storage.buffer().as_entire_binding(),
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
        // One invocation per output element; the plane is `global_id.z`.
        pass.dispatch_workgroups(new_width.div_ceil(16), new_height.div_ceil(16), c as u32);
    }
    ctx.submit(encoder);
    Ok(output)
}

pub fn resize_lanczos4<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
) -> Result<GpuTensor<T>> {
    resize_with_mode(
        ctx,
        input,
        new_width,
        new_height,
        InterpolationMode::Lanczos4,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::tensor_ext::TensorToGpu;

    #[test]
    fn non_f32_input_is_rejected() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // no adapter available
        };

        // f64 has no WGSL shader; this used to panic on the f32 downcast.
        let input = cv_core::CpuTensor::<f64>::from_vec(vec![0.0f64; 4], TensorShape::new(1, 2, 2))
            .unwrap()
            .to_gpu_ctx(&ctx)
            .unwrap();

        let res = resize(&ctx, &input, 1, 1);
        assert!(matches!(res, Err(crate::Error::NotSupported(_))));
    }
}
