use crate::GpuTensor;
use crate::Result;
use crate::context::ThresholdType;
use crate::gpu::GpuContext;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone)]
#[repr(C)]
struct ThresholdParams<T: bytemuck::Pod + bytemuck::Zeroable> {
    width: u32,
    height: u32,
    thresh: T,
    max_value: T,
    thresh_type: u32,
    len: u32,
    _pad0: u32,
    _pad1: u32, // pad to 32 bytes for WGSL uniform alignment
}

// SAFETY: `threshold` is only instantiated for `T: Float`, whose implementations
// (`f32`, `f64`, and the `half-precision` `f16`/`bf16`) are all plain scalar floats
// (size 2/4/8, power-of-two alignment). Each layout packs without padding: `thresh`
// sits at offset 8, `max_value` immediately after it, then `thresh_type`/`len`/`_pad0`/
// `_pad1` fill the 4-byte-aligned tail (sizes 28/32/40 for f16/bf16, f32 and f64
// respectively). Restricting the impl to `T: Float + Pod + Zeroable` matches the
// function's own bound and excludes e.g. `u8`, whose `ThresholdParams<u8>` layout has
// padding at bytes 10..12 and would make this impl unsound.
unsafe impl<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable> bytemuck::Pod
    for ThresholdParams<T>
{
}
unsafe impl<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable> bytemuck::Zeroable
    for ThresholdParams<T>
{
}
const _: () = assert!(std::mem::size_of::<ThresholdParams<f32>>() == 32);

/// Uniform for the two f16 shaders: the `threshold_f32.wgsl` layout with the
/// threshold and max value widened to f32 (exact), plus their f16 bits for
/// `threshold_f16_packed.wgsl`, which writes results back as raw bits.
#[cfg(feature = "half-precision")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ThresholdParamsF16 {
    width: u32,
    height: u32,
    thresh: f32,
    max_value: f32,
    thresh_type: u32,
    len: u32,
    thresh_bits: u32,
    max_bits: u32,
}

/// Which f16 shader `threshold` runs.
#[cfg(feature = "half-precision")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum F16Path {
    /// `threshold_f16.wgsl`, native f16 arithmetic. Needs `SHADER_F16`.
    Native,
    /// `threshold_f16_packed.wgsl`, two f16 per u32 word, compared in f32.
    Packed,
}

pub fn threshold<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    thresh: T,
    max_value: T,
    thresh_type: ThresholdType,
) -> Result<GpuTensor<T>> {
    let len = input.shape.len();

    // Validate the dtype *before* any `bytemuck::bytes_of` on the params
    // struct, so an unsupported `T` returns early instead of being
    // reinterpreted.
    match cv_core::DataType::from_type::<T>() {
        Ok(cv_core::DataType::F32) | Err(_) => {}
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::F16) => {
            let path = if ctx.supports_f16() {
                F16Path::Native
            } else {
                F16Path::Packed
            };
            return threshold_f16(ctx, input, thresh, max_value, thresh_type, path);
        }
        Ok(_) => {
            return Err(crate::Error::NotSupported(
                "Unsupported threshold precision type".into(),
            ));
        }
    }

    // Prepare params
    let params = ThresholdParams {
        width: input.shape.width as u32,
        height: input.shape.height as u32,
        thresh,
        max_value,
        thresh_type: thresh_type as u32,
        len: len as u32,
        _pad0: 0,
        _pad1: 0,
    };
    run(
        ctx,
        input,
        include_str!("../../shaders/threshold_f32.wgsl"),
        bytemuck::bytes_of(&params),
        len as u32,
    )
}

/// f16 threshold. `path` is chosen by the caller from `supports_f16`; the
/// tests force `Packed` to cover the fallback on a device that has the
/// feature. Both paths give the same bits.
#[cfg(feature = "half-precision")]
fn threshold_f16<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    thresh: T,
    max_value: T,
    thresh_type: ThresholdType,
    path: F16Path,
) -> Result<GpuTensor<T>> {
    debug_assert_eq!(std::mem::size_of::<T>(), 2);
    let len = input.shape.len();
    let bits = |v: &T| -> u32 {
        let b = bytemuck::bytes_of(v);
        u16::from_le_bytes([b[0], b[1]]) as u32
    };
    let params = ThresholdParamsF16 {
        width: input.shape.width as u32,
        height: input.shape.height as u32,
        thresh: thresh.to_f32(),
        max_value: max_value.to_f32(),
        thresh_type: thresh_type as u32,
        len: len as u32,
        thresh_bits: bits(&thresh),
        max_bits: bits(&max_value),
    };
    let (shader, invocations) = match path {
        F16Path::Native => (include_str!("../../shaders/threshold_f16.wgsl"), len as u32),
        F16Path::Packed => (
            include_str!("../../shaders/threshold_f16_packed.wgsl"),
            (len as u32).div_ceil(2),
        ),
    };
    run(ctx, input, shader, bytemuck::bytes_of(&params), invocations)
}

/// Bind `input`, a fresh output buffer and `params`, and dispatch `shader`
/// over `invocations` threads.
fn run<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    shader_source: &str,
    params: &[u8],
    invocations: u32,
) -> Result<GpuTensor<T>> {
    let len = input.shape.len();
    // A storage binding must be a multiple of 4 bytes, which an odd number
    // of f16 is not.
    let byte_size = ((len * std::mem::size_of::<T>()).div_ceil(4) * 4) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Threshold Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Threshold Params"),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Threshold Bind Group"),
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

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Threshold Dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Threshold Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let (x, y, z) = super::dispatch_1d(invocations, 64);
        pass.dispatch_workgroups(x, y, z);
    }

    ctx.submit(encoder);

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

/// Both f16 shaders against the f32 shader, on the real device. The public
/// `threshold` picks one path per device, so only here can the fallback be
/// run on a device that has `SHADER_F16`. Skips without an adapter.
#[cfg(all(test, feature = "half-precision"))]
mod tests {
    use super::*;
    use crate::tensor_ext::{TensorToCpu, TensorToGpu};
    use cv_core::{CpuTensor, Tensor, TensorShape};
    use half::f16;

    fn values(n: usize) -> Vec<f16> {
        let special = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            100.25,
            -100.25,
            65504.0,
            -65504.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            6e-8, // smallest f16 subnormal
            -6e-8,
            3e-5, // subnormal
        ];
        let mut s = 7u32;
        (0..n)
            .map(|i| {
                if i < special.len() {
                    return f16::from_f32(special[i]);
                }
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                // Any finite f16 bit pattern: clear one exponent bit of an
                // all-ones exponent (inf / NaN).
                let bits = (s >> 16) as u16;
                let bits = if (bits >> 10) & 0x1f == 0x1f {
                    bits & !0x4000
                } else {
                    bits
                };
                f16::from_bits(bits)
            })
            .collect()
    }

    #[test]
    fn f16_paths_match_the_f32_shader() {
        let Ok(ctx) = pollster::block_on(GpuContext::init_global()) else {
            println!("skipping: no GPU adapter");
            return;
        };
        let mut paths = vec![F16Path::Packed];
        if ctx.supports_f16() {
            paths.push(F16Path::Native);
        } else {
            println!("device lacks SHADER_F16: testing the packed fallback only");
        }
        // Odd and even lengths: the packed shader's last word is half full
        // for an odd one.
        for (h, w) in [(1usize, 1usize), (7, 143), (16, 64)] {
            let v = values(h * w);
            let v32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
            let t16: CpuTensor<f16> =
                Tensor::from_vec(v.clone(), TensorShape::new(1, h, w)).unwrap();
            let t32: CpuTensor<f32> = Tensor::from_vec(v32, TensorShape::new(1, h, w)).unwrap();
            let g16 = t16.to_gpu_ctx(ctx).unwrap();
            let g32 = t32.to_gpu_ctx(ctx).unwrap();
            for thresh in [0.0f32, -1.0, 100.25, 6e-8, 65504.0] {
                let (th, mx) = (f16::from_f32(thresh), f16::from_f32(200.5));
                for typ in [
                    ThresholdType::Binary,
                    ThresholdType::BinaryInv,
                    ThresholdType::Trunc,
                    ThresholdType::ToZero,
                    ThresholdType::ToZeroInv,
                ] {
                    let want = threshold(ctx, &g32, th.to_f32(), mx.to_f32(), typ).unwrap();
                    let want = want.to_cpu_ctx(ctx).unwrap();
                    let want = want.as_slice().unwrap();
                    for &path in &paths {
                        let got = threshold_f16(ctx, &g16, th, mx, typ, path).unwrap();
                        let got = got.to_cpu_ctx(ctx).unwrap();
                        let got = got.as_slice().unwrap();
                        assert_eq!(got.len(), want.len());
                        for i in 0..want.len() {
                            // Every result is an input, the threshold, the
                            // max value or 0, so it narrows to f16 exactly.
                            let w16 = f16::from_f32(want[i]);
                            assert_eq!(
                                got[i].to_bits(),
                                w16.to_bits(),
                                "{path:?} {typ:?} thresh {thresh} {h}x{w} [{i}]: input {}, \
                                 got {}, f32 shader gave {}",
                                v[i],
                                got[i],
                                want[i]
                            );
                        }
                    }
                }
            }
        }
    }
}
