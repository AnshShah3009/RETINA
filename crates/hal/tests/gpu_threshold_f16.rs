//! GPU `threshold` on f16 tensors, through the public `ComputeContext` entry.
//!
//! `threshold_f16.wgsl` never validated (it bitcast a u32 to f16), the device
//! never requested `SHADER_F16`, and `gpu_kernels::threshold` returned
//! `NotSupported` for f16 without trying. It now runs the native shader when
//! the device has `SHADER_F16` and a packed-u32 fallback otherwise; the
//! in-crate test `gpu_kernels::threshold::tests` runs both paths explicitly
//! against the f32 shader. Here the public entry is checked against the CPU
//! backend's f16 path, bit for bit. Needs `--features half-precision`; skips
//! without an adapter.
#![cfg(feature = "half-precision")]

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use half::f16;

#[test]
fn gpu_f16_threshold_matches_cpu_f16() {
    let Ok(gpu) = pollster::block_on(GpuContext::init_global()) else {
        println!("skipping: no GPU adapter");
        return;
    };
    println!("SHADER_F16 available: {}", gpu.supports_f16());
    let cpu = CpuBackend::new().unwrap();
    let (h, w) = (9usize, 31usize); // odd length
    let mut s = 3u32;
    let v: Vec<f16> = (0..h * w)
        .map(|i| match i {
            0 => f16::INFINITY,
            1 => f16::NEG_INFINITY,
            2 => f16::from_f32(-0.0),
            3 => f16::from_f32(12.5), // equal to a threshold below
            _ => {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                f16::from_f32(((s >> 8) as f32 / (1u32 << 24) as f32) * 400.0 - 200.0)
            }
        })
        .collect();
    let t: CpuTensor<f16> = Tensor::from_vec(v.clone(), TensorShape::new(1, h, w)).unwrap();
    let gt = t.to_gpu_ctx(gpu).unwrap();
    for thresh in [12.5f32, -3.0, 0.0] {
        let (th, mx) = (f16::from_f32(thresh), f16::from_f32(255.0));
        for typ in [
            ThresholdType::Binary,
            ThresholdType::BinaryInv,
            ThresholdType::Trunc,
            ThresholdType::ToZero,
            ThresholdType::ToZeroInv,
        ] {
            let want = cpu.threshold(&t, th, mx, typ).unwrap();
            let want = want.as_slice().unwrap();
            let got = gpu
                .threshold(&gt, th, mx, typ)
                .unwrap_or_else(|e| panic!("GPU f16 threshold {typ:?} failed: {e}"));
            let got = got.to_cpu_ctx(gpu).unwrap();
            let got = got.as_slice().unwrap();
            assert_eq!(got.len(), want.len());
            for i in 0..want.len() {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "{typ:?} thresh {thresh} [{i}]: input {}, GPU {}, CPU {}",
                    v[i],
                    got[i],
                    want[i]
                );
            }
        }
    }
}
