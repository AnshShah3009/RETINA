use crate::Result;
use crate::gpu::GpuContext;
#[cfg(feature = "half-precision")]
use crate::gpu_kernels::packed16::{self, Packed16};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone)]
#[repr(C)]
struct BilateralParams {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
    /// `bilateral_packed16.wgsl`'s `format` (0: f16, 1: bf16); padding for
    /// the f32 shader.
    format: u32,
    _pad1: u32,
    _pad2: u32, // pad to 32 bytes for WGSL uniform alignment
}

unsafe impl bytemuck::Pod for BilateralParams {}
unsafe impl bytemuck::Zeroable for BilateralParams {}

/// Bilateral filter of a single-channel image.
///
/// f32 runs `bilateral_f32.wgsl`. f16 and bf16 run `bilateral_packed16.wgsl`
/// on every device (no `SHADER_F16` needed): the same f32 arithmetic on the
/// exactly widened values, rounded to the type once at the end. Other types
/// return `NotSupported`.
pub fn bilateral_filter<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    d: i32,
    sigma_color: T,
    sigma_space: T,
) -> Result<crate::GpuTensor<T>> {
    let sigma_color_f = sigma_color.to_f32();
    let sigma_space_f = sigma_space.to_f32();
    // Same guard as the CPU filter: a sigma of 0 gives -inf coefficients and
    // NaN weights, so a non-positive (or NaN) sigma weights with 1 instead. The
    // radius below still comes from the caller's sigma_space, as on the CPU.
    let sigma_color_eff = if sigma_color_f > 0.0 {
        sigma_color_f
    } else {
        1.0
    };
    let sigma_space_eff = if sigma_space_f > 0.0 {
        sigma_space_f
    } else {
        1.0
    };
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Bilateral currently only for grayscale".into(),
        ));
    }

    let out_len = h * w * c;
    let (shader_source, format, byte_size): (&str, u32, u64) =
        match cv_core::DataType::from_type::<T>() {
            Ok(cv_core::DataType::F32) => (
                include_str!("../../shaders/bilateral_f32.wgsl"),
                0,
                (out_len * std::mem::size_of::<f32>()) as u64,
            ),
            #[cfg(feature = "half-precision")]
            Ok(cv_core::DataType::F16) => (
                include_str!("../../shaders/bilateral_packed16.wgsl"),
                Packed16::F16.code(),
                packed16::buffer_bytes(out_len),
            ),
            #[cfg(feature = "half-precision")]
            Ok(cv_core::DataType::BF16) => (
                include_str!("../../shaders/bilateral_packed16.wgsl"),
                Packed16::Bf16.code(),
                packed16::buffer_bytes(out_len),
            ),
            _ => {
                return Err(crate::Error::NotSupported(
                    "Bilateral GPU kernel supports f32, f16 and bf16".into(),
                ));
            }
        };

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bilateral Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let radius = if d <= 0 {
        (sigma_space_f * 1.5).ceil() as i32
    } else {
        d / 2
    };
    let params = BilateralParams {
        width: w as u32,
        height: h as u32,
        radius,
        sigma_color_sq_inv: -0.5 / (sigma_color_eff * sigma_color_eff),
        sigma_space_sq_inv: -0.5 / (sigma_space_eff * sigma_space_eff),
        format,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bilateral Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bilateral Bind Group"),
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
        let x = (w as u32).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let res_gpu = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    };
    Ok(res_gpu)
}
