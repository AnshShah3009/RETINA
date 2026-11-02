use crate::GpuTensor;
use crate::Result;
use crate::gpu::GpuContext;
#[cfg(feature = "half-precision")]
use crate::gpu_kernels::packed16::Packed16;
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

/// Uniform of `resize_packed16.wgsl`: `ResizeParams` with the 16-bit format.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParamsPacked16 {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
    format: u32,
    _pad0: u32,
    _pad1: u32,
}

/// One axis sample of `resize_packed16.wgsl`'s `taps` table: the CPU's
/// `cpu::bilinear::Tap`, weights widened to f32 (exact).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TapGpu {
    i0: u32,
    i1: u32,
    w0: f32,
    w1: f32,
}

/// The CPU backend's bilinear taps for `src -> dst` on both axes: `dst_w`
/// column taps, then `dst_h` row taps.
fn packed16_taps<T: cv_core::float::Float>(
    (src_w, src_h): (usize, usize),
    (dst_w, dst_h): (usize, usize),
) -> Vec<TapGpu> {
    let axis = |src, dst| {
        crate::cpu::bilinear::bilinear_taps::<T>(src, dst)
            .into_iter()
            .map(|t| TapGpu {
                i0: t.i0 as u32,
                i1: t.i1 as u32,
                w0: t.w0.to_f32(),
                w1: t.w1.to_f32(),
            })
    };
    axis(src_w, dst_w).chain(axis(src_h, dst_h)).collect()
}

/// Resize a planar CHW tensor on the GPU.
///
/// Element `(ch, y, x)` of a `TensorShape::new(c, h, w)` tensor is at
/// `ch * h * w + y * w + x`, as everywhere else in the workspace
/// (`Tensor::index`, `Tensor::from_image_rgb`, `gaussian_blur`, `cvt_color`);
/// each plane is resampled as an independent image.
///
/// f32 runs `resize_f32.wgsl` / `lanczos4.wgsl`. f16 and bf16 (with the
/// `half-precision` feature) run `resize_packed16.wgsl` on any device: its
/// bilinear entry is bit-identical to the CPU backend's `resize` for the same
/// type, and its Lanczos entry is the f32 Lanczos arithmetic on the widened
/// values, rounded once. Other types return `NotSupported`.
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
    // `None` is f32; `Some(format)` a 16-bit type bound as packed words,
    // `format` being the shader's `params.format` (`Packed16::code`).
    let packed: Option<u32> = match cv_core::DataType::from_type::<T>() {
        Ok(cv_core::DataType::F32) => None,
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::F16) => Some(Packed16::F16.code()),
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::BF16) => Some(Packed16::Bf16.code()),
        _ => {
            return Err(crate::Error::NotSupported(
                "Resize GPU kernel supports f32, f16 and bf16".into(),
            ));
        }
    };
    let (src_h, src_w) = input.shape.hw();
    let c = input.shape.channels;
    let (dst_w, dst_h) = (new_width as usize, new_height as usize);
    let out_shape = TensorShape::new(c, dst_h, dst_w);
    let out_len = out_shape
        .checked_len()
        .ok_or_else(|| crate::Error::InvalidInput("Resize target size overflows".into()))?;
    // A storage binding may be neither empty nor a partial word. Packed:
    // two values per u32 word, as `packed16::buffer_bytes` (which exists only
    // with `half-precision`).
    let byte_size = match packed {
        None => ((out_len * std::mem::size_of::<f32>()) as u64).max(4),
        Some(_) => (out_len.div_ceil(2).max(1) * 4) as u64,
    };

    // wgpu zero-initialises a new buffer, so an empty source needs no
    // dispatch to come back as zeros, and the packed shader can OR each half
    // into its word.
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resize Output Buffer"),
        size: byte_size,
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

    let (params, shader_source, entry_point): (Vec<u8>, &str, &str) = match packed {
        None => {
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
            let source = match mode {
                InterpolationMode::Bilinear => include_str!("../../shaders/resize_f32.wgsl"),
                InterpolationMode::Lanczos4 => include_str!("../../shaders/lanczos4.wgsl"),
            };
            (bytemuck::bytes_of(&params).to_vec(), source, "main")
        }
        Some(format) => {
            let params = ResizeParamsPacked16 {
                src_w: src_w as u32,
                src_h: src_h as u32,
                dst_w: new_width,
                dst_h: new_height,
                channels: c as u32,
                format,
                _pad0: 0,
                _pad1: 0,
            };
            let entry = match mode {
                InterpolationMode::Bilinear => "main_bilinear",
                InterpolationMode::Lanczos4 => "main_lanczos",
            };
            (
                bytemuck::bytes_of(&params).to_vec(),
                include_str!("../../shaders/resize_packed16.wgsl"),
                entry,
            )
        }
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resize Params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    // Only the packed bilinear entry reads a tap table; wgpu derives each
    // entry point's layout from the bindings it reaches, so the others must
    // not be given one.
    let taps_buffer = (packed.is_some() && mode == InterpolationMode::Bilinear).then(|| {
        let taps = packed16_taps::<T>((src_w, src_h), (dst_w, dst_h));
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Resize Taps"),
                contents: bytemuck::cast_slice(&taps),
                usage: wgpu::BufferUsages::STORAGE,
            })
    });

    let pipeline = ctx.create_compute_pipeline(shader_source, entry_point);

    let mut entries = vec![
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
    ];
    if let Some(taps) = &taps_buffer {
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: taps.as_entire_binding(),
        });
    }
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resize Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
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
