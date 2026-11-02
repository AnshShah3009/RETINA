use crate::GpuTensor;
use crate::Result;
use crate::context::ColorConversion;
use crate::gpu::GpuContext;
#[cfg(feature = "half-precision")]
use crate::gpu_kernels::packed16::{self, Packed16};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorParams {
    len: u32,
    code: u32,
    _pad0: u32,
    _pad1: u32,
    /// f32 bits of the CPU backend's weights (`T::from_f32(0.299)` etc.):
    /// the shader multiplies with integer operations.
    wr: u32,
    wg: u32,
    wb: u32,
    _pad2: u32,
}

/// Uniform of `color_cvt_packed16.wgsl`: the f32 layout plus the format and
/// the gray weights as the CPU backend uses them, `T::from_f32(0.299)` etc.,
/// widened back to f32 (exact).
#[cfg(feature = "half-precision")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorParamsPacked16 {
    len: u32,
    code: u32,
    format: u32,
    _pad0: u32,
    wr: f32,
    wg: f32,
    wb: f32,
    _pad1: u32,
}

/// Colour conversion of a planar CHW image. RGB / BGR to gray only.
///
/// f32 runs `color_cvt_f32.wgsl`, which multiplies with integer operations
/// and adds in a scaled domain where a subnormal could be flushed, so the
/// result is bit-identical to the CPU backend's for every input (a NaN result
/// is NaN, its payload aside). f16 and bf16 run `color_cvt_packed16.wgsl` on
/// every device (no `SHADER_F16` needed), which reproduces the CPU backend's
/// per-operation rounding, so the result is bit-identical to the CPU's for
/// the same type. Other types return `NotSupported`.
pub fn color_convert<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    conv: ColorConversion,
) -> Result<GpuTensor<T>> {
    let (h, w) = input.shape.hw();
    let num_pixels = h * w;

    let (out_channels, code_int) = match conv {
        ColorConversion::RgbToGray => (1, 0),
        ColorConversion::BgrToGray => (1, 1),
        ColorConversion::GrayToRgb => (3, 2),
    };

    if code_int == 2 {
        return Err(crate::Error::NotSupported("GPU GrayToRgb pending".into()));
    }
    // The shaders read three planes; with fewer they would read past the
    // buffer (robust access returns zeros) and give a wrong gray image.
    if input.shape.channels != 3 {
        return Err(crate::Error::InvalidInput(
            "RgbToGray requires 3 channels".into(),
        ));
    }

    let out_len = w * h * out_channels;
    // The f32 shader runs one invocation per pixel, the packed one one per
    // output word (two pixels).
    let (shader_source, params, byte_size, invocations): (&str, Vec<u8>, u64, u32) =
        match cv_core::DataType::from_type::<T>() {
            Ok(cv_core::DataType::F32) => {
                let params = ColorParams {
                    len: num_pixels as u32,
                    code: code_int,
                    _pad0: 0,
                    _pad1: 0,
                    wr: 0.299f32.to_bits(),
                    wg: 0.587f32.to_bits(),
                    wb: 0.114f32.to_bits(),
                    _pad2: 0,
                };
                (
                    include_str!("../../shaders/color_cvt_f32.wgsl"),
                    bytemuck::bytes_of(&params).to_vec(),
                    (out_len * std::mem::size_of::<f32>()) as u64,
                    num_pixels as u32,
                )
            }
            #[cfg(feature = "half-precision")]
            Ok(dt @ (cv_core::DataType::F16 | cv_core::DataType::BF16)) => {
                let format = if dt == cv_core::DataType::F16 {
                    Packed16::F16
                } else {
                    Packed16::Bf16
                };
                let params = ColorParamsPacked16 {
                    len: num_pixels as u32,
                    code: code_int,
                    format: format.code(),
                    _pad0: 0,
                    wr: packed16::rounded::<T>(0.299),
                    wg: packed16::rounded::<T>(0.587),
                    wb: packed16::rounded::<T>(0.114),
                    _pad1: 0,
                };
                (
                    include_str!("../../shaders/color_cvt_packed16.wgsl"),
                    bytemuck::bytes_of(&params).to_vec(),
                    packed16::buffer_bytes(out_len),
                    (num_pixels as u32).div_ceil(2),
                )
            }
            _ => {
                return Err(crate::Error::NotSupported(
                    "Color convert GPU kernel supports f32, f16 and bf16".into(),
                ));
            }
        };

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Color Convert Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Color Bind Group"),
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
        let (x, y, z) = super::dispatch_1d(invocations, 64);
        pass.dispatch_workgroups(x, y, z);
    }
    ctx.submit(encoder);

    let res_gpu = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: cv_core::TensorShape::new(out_channels, h, w),
        dtype: input.dtype,
        _phantom: PhantomData,
    };
    Ok(res_gpu)
}
