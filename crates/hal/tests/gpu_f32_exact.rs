//! f32 GPU kernels that must reproduce the CPU backend bit for bit, through
//! the public `ComputeContext` entries.
//!
//! * `threshold` compared floats, so the device flushed subnormal inputs and
//!   thresholds to zero (`0 > -1e-45` was false). Its Trunc was `min`, which
//!   WGSL leaves to the implementation for NaN; the CPU's was `min` as well,
//!   inconsistent with its other types and across platforms (see
//!   `cpu_f32_trunc_keeps_ties_and_nan_on_every_lane`).
//!
//! Results must match bit for bit, signed zero, subnormals and NaN payloads
//! included. Skips without an adapter.

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("skipping: no GPU adapter");
            None
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn tensor(v: Vec<f32>, c: usize, h: usize, w: usize) -> CpuTensor<f32> {
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

/// Equal bits, or (when `nan_class_only`) both NaN.
fn same(a: f32, b: f32, nan_class_only: bool) -> bool {
    a.to_bits() == b.to_bits() || (nan_class_only && a.is_nan() && b.is_nan())
}

fn assert_bits_eq(
    got: &[f32],
    want: &[f32],
    nan_class_only: bool,
    inputs: &dyn Fn(usize) -> String,
    what: &str,
) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad: Vec<usize> = (0..want.len())
        .filter(|&i| !same(got[i], want[i], nan_class_only))
        .collect();
    if let Some(&i) = bad.first() {
        panic!(
            "{what}: {} of {} differ, first at [{i}] ({}): GPU {:e} ({:#010x}), CPU {:e} ({:#010x})",
            bad.len(),
            want.len(),
            inputs(i),
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

/// Signed zeros, infinities, NaNs (both signs, quiet and signalling
/// payloads), subnormals, the normal boundary and ordinary values.
fn specials() -> Vec<f32> {
    [
        0x0000_0000u32, // +0
        0x8000_0000,    // -0
        0x0000_0001,    // smallest subnormal
        0x8000_0001,
        0x0000_0002,
        0x8000_0003,
        0x0012_3456, // mid subnormal
        0x8040_0000,
        0x007f_ffff, // largest subnormal
        0x807f_ffff,
        0x0080_0000, // smallest normal
        0x8080_0000,
        0x0080_0001,
        0x3f80_0000, // 1
        0xbf80_0000,
        0x4300_0000, // 128
        0x437f_0000, // 255
        0x7f7f_ffff, // largest finite
        0xff7f_ffff,
        0x7f80_0000, // inf
        0xff80_0000,
        0x7fc0_0000, // quiet NaN
        0xffc0_0001,
        0x7f80_0001, // signalling NaN
        0x7fa0_0000,
    ]
    .iter()
    .map(|&b| f32::from_bits(b))
    .collect()
}

/// Any f32 bit pattern, every exponent equally likely.
fn any_bits(r: &mut Lcg) -> f32 {
    f32::from_bits(r.next())
}

// ---------------------------------------------------------------- threshold

#[test]
fn f32_threshold_matches_cpu_bit_for_bit() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    let mut r = Lcg(3);
    let sp = specials();
    // Lengths that are not a multiple of 8 put values in the CPU's scalar
    // tail as well as its SIMD lanes; 4100 spans two 4096-element chunks.
    for (h, w) in [(1usize, 1usize), (1, 7), (5, 13), (3, 64), (41, 100)] {
        let n = h * w;
        // Every special at every position modulo 8, then a mix of specials,
        // values near them and arbitrary bit patterns.
        let mut v: Vec<f32> = (0..n)
            .map(|i| match i % 3 {
                0 => sp[(i / 3) % sp.len()],
                1 => f32::from_bits(sp[(i * 7) % sp.len()].to_bits().wrapping_add(1)),
                _ => any_bits(&mut r),
            })
            .collect();
        if n >= sp.len() {
            v[..sp.len()].copy_from_slice(&sp);
        }
        let t = tensor(v.clone(), 1, h, w);
        let gt = t.to_gpu_ctx(gpu).unwrap();
        for &thresh in &sp {
            for max in [255.0f32, -0.0, f32::from_bits(1), f32::NAN] {
                for typ in [
                    ThresholdType::Binary,
                    ThresholdType::BinaryInv,
                    ThresholdType::Trunc,
                    ThresholdType::ToZero,
                    ThresholdType::ToZeroInv,
                ] {
                    let want = cpu.threshold(&t, thresh, max, typ).unwrap();
                    let got = gpu.threshold(&gt, thresh, max, typ).unwrap();
                    let got = got.to_cpu_ctx(gpu).unwrap();
                    assert_bits_eq(
                        got.as_slice().unwrap(),
                        want.as_slice().unwrap(),
                        false,
                        &|i| format!("input {:e} ({:#010x})", v[i], v[i].to_bits()),
                        &format!(
                            "f32 threshold {typ:?} thresh {thresh:e} ({:#010x}) max {max} {h}x{w}",
                            thresh.to_bits()
                        ),
                    );
                }
            }
        }
    }
}

/// The CPU f32 Trunc is `v > t ? t : v` in the SIMD lanes and the scalar
/// tail alike, as for every other type. It was `min(v, t)`, which returned
/// the threshold for a NaN input (the f16 / bf16 paths keep the NaN) and left
/// a -0 / +0 tie to the platform: x86 returned the threshold, AArch64 -0, so
/// no GPU result could match both. Needs no adapter.
#[test]
fn cpu_f32_trunc_keeps_ties_and_nan_on_every_lane() {
    let cpu = CpuBackend::new().unwrap();
    // 11 elements: eight SIMD lanes and a scalar tail of three.
    let v = vec![
        -0.0f32,
        0.0,
        f32::NAN,
        1.0,
        -1.0,
        -0.0,
        0.0,
        f32::NAN,
        -0.0,
        0.0,
        f32::NAN,
    ];
    let t = tensor(v.clone(), 1, 1, v.len());
    // A tie or a NaN input is not above the threshold, so it is kept.
    for thresh in [0.0f32, -0.0] {
        let got = cpu
            .threshold(&t, thresh, 1.0, ThresholdType::Trunc)
            .unwrap();
        let got = got.as_slice().unwrap();
        for i in 0..v.len() {
            let w = if v[i] > thresh { thresh } else { v[i] };
            assert_eq!(
                got[i].to_bits(),
                w.to_bits(),
                "Trunc input {:?} thresh {thresh:?} [{i}]: got {:?}",
                v[i],
                got[i]
            );
        }
    }
    // A NaN threshold leaves every input as it is.
    let got = cpu
        .threshold(&t, f32::NAN, 1.0, ThresholdType::Trunc)
        .unwrap();
    for (g, x) in got.as_slice().unwrap().iter().zip(&v) {
        assert!(same(*g, *x, false), "Trunc NaN thresh: {x:?} -> {g:?}");
    }
}
