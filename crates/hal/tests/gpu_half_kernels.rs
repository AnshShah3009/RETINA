//! f16 and bf16 through the public `ComputeContext` entries of the GPU
//! kernels that select a 16-bit shader: `threshold` (bf16), `fast_detect`,
//! `cvt_color` and `bilateral_filter`.
//!
//! Before, `fast_detect`, `cvt_color` and `bilateral_filter` returned
//! `NotSupported` for both types (and bound the input as `GpuStorage<f32>`
//! with `.unwrap()` behind that guard), the `*_f16.wgsl` shaders they named
//! needed `SHADER_F16`, and the `*_bf16.wgsl` ones read one value per u32 word
//! or raced on shared output words. `threshold` returned `NotSupported` for
//! bf16 and `threshold_bf16.wgsl` compared raw bits. Each now runs a packed
//! shader on any device (see `gpu_kernels::packed16`).
//!
//! References and tolerances:
//! * `threshold`, `fast_detect` (with and without suppression) and
//!   `cvt_color`: the CPU backend for the same type, bit for bit.
//! * `bilateral_filter`: the GPU f32 path on the same (widened) values,
//!   narrowed to the type, bit for bit. Its `exp` is the device's, so the CPU
//!   is not a bit-exact reference even in f32; against the CPU f32 path
//!   narrowed, the result is within one unit in the last place of the type.
//!
//! Sizes include odd lengths and widths, where the last output word is half
//! full and two rows share a word. Needs `--features half-precision`; skips
//! without an adapter.
#![cfg(feature = "half-precision")]

use cv_core::float::Float;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ColorConversion, ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use half::{bf16, f16};

trait Half: Float + bytemuck::Pod + bytemuck::Zeroable + 'static {
    const NAME: &'static str;
    fn bits(self) -> u16;
    fn from_bits16(b: u16) -> Self;
}
impl Half for f16 {
    const NAME: &'static str = "f16";
    fn bits(self) -> u16 {
        self.to_bits()
    }
    fn from_bits16(b: u16) -> Self {
        f16::from_bits(b)
    }
}
impl Half for bf16 {
    const NAME: &'static str = "bf16";
    fn bits(self) -> u16 {
        self.to_bits()
    }
    fn from_bits16(b: u16) -> Self {
        bf16::from_bits(b)
    }
}

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("skipping: no GPU adapter");
            None
        }
    }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 8) as f32 / (1u32 << 24) as f32 * (hi - lo)
    }
}

fn tensor<T: Half>(v: &[T], c: usize, h: usize, w: usize) -> CpuTensor<T> {
    Tensor::from_vec(v.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

fn assert_bits_eq<T: Half>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad: Vec<usize> = (0..want.len())
        .filter(|&i| got[i].bits() != want[i].bits())
        .collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} differ, first at [{}]: GPU {} ({:#06x}), reference {} ({:#06x})",
        bad.len(),
        want.len(),
        bad[0],
        got[bad[0]],
        got[bad[0]].bits(),
        want[bad[0]],
        want[bad[0]].bits()
    );
}

/// An image with flat regions, edges, isolated spots and noise, so FAST finds
/// corners of both polarities and suppression has ties to break.
fn image<T: Half>(h: usize, w: usize, scale: f32, seed: u32) -> Vec<T> {
    let mut r = Lcg(seed);
    (0..h * w)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            let mut v = if (x / 6 + y / 5) % 2 == 0 { 0.2 } else { 0.8 };
            if (x * 7 + y * 3) % 11 == 0 {
                v = 1.0 - v;
            }
            v += r.range(-0.05, 0.05);
            T::from_f32(v * scale)
        })
        .collect()
}

// ---------------------------------------------------------------- threshold

fn bf16_specials() -> Vec<bf16> {
    [
        0x0000u16, // +0
        0x8000,    // -0
        0x3f80,    // 1
        0xbf80,    // -1
        0x7f80,    // inf
        0xff80,    // -inf
        0x7fc0,    // NaN
        0x0001,    // smallest subnormal
        0x8001,    // its negative
        0x007f,    // largest subnormal
        0x0080,    // smallest normal
        0x7f7f,    // largest finite
        0xff7f,    // its negative
        0x4148,    // 12.5
    ]
    .iter()
    .map(|&b| bf16::from_bits(b))
    .collect()
}

#[test]
fn bf16_threshold_matches_cpu() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    let mut r = Lcg(11);
    for (h, w) in [(1usize, 1usize), (1, 3), (9, 31), (16, 64)] {
        let mut v = bf16_specials();
        v.truncate(h * w);
        while v.len() < h * w {
            // Any bit pattern but NaN.
            let b = (r.next() >> 16) as u16;
            let b = if b & 0x7fff > 0x7f80 { b & 0xbfff } else { b };
            v.push(bf16::from_bits(b));
        }
        let t = tensor(&v, 1, h, w);
        let gt = t.to_gpu_ctx(gpu).unwrap();
        for th in [0x4148u16, 0x0000, 0x8000, 0xc040, 0x0001, 0x7f80, 0xff80] {
            let (th, mx) = (bf16::from_bits(th), bf16::from_f32(255.0));
            for typ in [
                ThresholdType::Binary,
                ThresholdType::BinaryInv,
                ThresholdType::Trunc,
                ThresholdType::ToZero,
                ThresholdType::ToZeroInv,
            ] {
                let want = cpu.threshold(&t, th, mx, typ).unwrap();
                let got = gpu
                    .threshold(&gt, th, mx, typ)
                    .unwrap_or_else(|e| panic!("GPU bf16 threshold {typ:?}: {e}"));
                let got = got.to_cpu_ctx(gpu).unwrap();
                assert_bits_eq(
                    got.as_slice().unwrap(),
                    want.as_slice().unwrap(),
                    &format!("bf16 threshold {typ:?} thresh {th} {h}x{w}"),
                );
            }
        }
    }
}

// --------------------------------------------------------------------- FAST

fn fast_matches_cpu<T: Half>() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    // Below 7 in either dimension the whole map is border; odd widths put a
    // row boundary inside an output word.
    let sizes = [(3usize, 5usize), (7, 7), (31, 45), (33, 64), (64, 33)];
    let mut corners = 0usize;
    for (k, &(h, w)) in sizes.iter().enumerate() {
        // A negative threshold makes `high < low`, so a circle pixel can be
        // both brighter and darker; the CPU counts it in both arcs.
        for (scale, thresholds) in [
            (255.0f32, [20.0f32, 7.5, 40.0, -7.5]),
            (1.0, [0.1, 0.03, 0.2, -0.03]),
        ] {
            let v: Vec<T> = image(h, w, scale, 100 + k as u32);
            let t = tensor(&v, 1, h, w);
            let gt = t.to_gpu_ctx(gpu).unwrap();
            for th in thresholds {
                let th = T::from_f32(th);
                for nms in [false, true] {
                    let want = cpu.fast_detect(&t, th, nms).unwrap();
                    let want = want.as_slice().unwrap();
                    let got = gpu
                        .fast_detect(&gt, th, nms)
                        .unwrap_or_else(|e| panic!("GPU {} FAST: {e}", T::NAME));
                    let got = got.to_cpu_ctx(gpu).unwrap();
                    assert_bits_eq(
                        got.as_slice().unwrap(),
                        want,
                        &format!("{} FAST thresh {th} nms {nms} {h}x{w}", T::NAME),
                    );
                    corners += want.iter().filter(|s| Float::to_f32(**s) > 0.0).count();
                }
            }
        }
    }
    // A comparison of two all-zero maps would prove nothing.
    assert!(corners > 100, "{} FAST: only {corners} corners", T::NAME);
}

#[test]
fn f16_fast_matches_cpu() {
    fast_matches_cpu::<f16>();
}

#[test]
fn bf16_fast_matches_cpu() {
    fast_matches_cpu::<bf16>();
}

// ---------------------------------------------------------------- cvt_color

fn cvt_color_matches_cpu<T: Half>() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    let mut r = Lcg(5);
    for (h, w) in [(1usize, 1usize), (1, 3), (5, 7), (16, 64), (33, 17)] {
        for scale in [255.0f32, 1.0] {
            let mut v: Vec<T> = (0..3 * h * w)
                .map(|_| T::from_f32(r.range(0.0, scale)))
                .collect();
            if v.len() > 5 {
                // Saturation, signed zero and an f16 subnormal.
                v[0] = T::from_f32(65504.0);
                v[1] = T::from_f32(-0.0);
                v[2] = T::from_bits16(0x0001);
                v[3] = T::from_f32(-3.25);
            }
            let t = tensor(&v, 3, h, w);
            let gt = t.to_gpu_ctx(gpu).unwrap();
            for code in [ColorConversion::RgbToGray, ColorConversion::BgrToGray] {
                let want = cpu.cvt_color(&t, code).unwrap();
                let got = gpu
                    .cvt_color(&gt, code)
                    .unwrap_or_else(|e| panic!("GPU {} cvt_color {code:?}: {e}", T::NAME));
                assert_eq!(got.shape, want.shape);
                let got = got.to_cpu_ctx(gpu).unwrap();
                assert_bits_eq(
                    got.as_slice().unwrap(),
                    want.as_slice().unwrap(),
                    &format!("{} cvt_color {code:?} scale {scale} {h}x{w}", T::NAME),
                );
            }
        }
    }
}

#[test]
fn f16_cvt_color_matches_cpu() {
    cvt_color_matches_cpu::<f16>();
}

#[test]
fn bf16_cvt_color_matches_cpu() {
    cvt_color_matches_cpu::<bf16>();
}

// --------------------------------------------------------- bilateral_filter

/// Distance in units in the last place between two finite values of the same
/// sign class, via the order of their bit patterns.
fn ulps<T: Half>(a: T, b: T) -> u32 {
    let key = |x: T| -> i32 {
        let b = x.bits() as i32;
        if b & 0x8000 != 0 { -(b & 0x7fff) } else { b }
    };
    (key(a) - key(b)).unsigned_abs()
}

fn bilateral_matches_f32<T: Half>() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    let mut worst_cpu = 0u32;
    for (k, &(h, w)) in [(1usize, 1usize), (3, 5), (17, 31), (32, 48), (45, 33)]
        .iter()
        .enumerate()
    {
        for scale in [255.0f32, 1.0] {
            let v: Vec<T> = image(h, w, scale, 7 + k as u32);
            let v32: Vec<f32> = v.iter().map(|x| Float::to_f32(*x)).collect();
            let t = tensor(&v, 1, h, w);
            let t32: CpuTensor<f32> = Tensor::from_vec(v32, TensorShape::new(1, h, w)).unwrap();
            let gt = t.to_gpu_ctx(gpu).unwrap();
            let gt32 = t32.to_gpu_ctx(gpu).unwrap();
            for (d, sc, ss) in [(5, 0.1f32, 2.0f32), (0, 0.3, 1.5), (3, 1.0, 1.0)] {
                let sc = sc * scale;
                let (sct, sst) = (T::from_f32(sc), T::from_f32(ss));
                // The f32 path with the same sigmas the 16-bit path sees.
                let (sc32, ss32) = (Float::to_f32(sct), Float::to_f32(sst));
                let want = gpu.bilateral_filter(&gt32, d, sc32, ss32).unwrap();
                let want = want.to_cpu_ctx(gpu).unwrap();
                let want: Vec<T> = want
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|&x| T::from_f32(x))
                    .collect();
                let got = gpu
                    .bilateral_filter(&gt, d, sct, sst)
                    .unwrap_or_else(|e| panic!("GPU {} bilateral: {e}", T::NAME));
                let got = got.to_cpu_ctx(gpu).unwrap();
                let got = got.as_slice().unwrap();
                let what = format!("{} bilateral d {d} sigmas {sc}/{ss} {h}x{w}", T::NAME);
                assert_bits_eq(got, &want, &what);

                let cpu32 = cpu.bilateral_filter(&t32, d, sc32, ss32).unwrap();
                for (i, (&g, &c)) in got.iter().zip(cpu32.as_slice().unwrap()).enumerate() {
                    let u = ulps(g, T::from_f32(c));
                    assert!(u <= 1, "{what} [{i}]: GPU {g}, CPU f32 {c}: {u} ulps");
                    worst_cpu = worst_cpu.max(u);
                }
            }
        }
    }
    println!(
        "{} bilateral: worst distance from the CPU f32 path narrowed: {worst_cpu} ulp",
        T::NAME
    );
}

#[test]
fn f16_bilateral_matches_f32_path() {
    bilateral_matches_f32::<f16>();
}

#[test]
fn bf16_bilateral_matches_f32_path() {
    bilateral_matches_f32::<bf16>();
}

// ------------------------------------------------------------ clean refusals

#[test]
fn unsupported_16_bit_requests_are_errors_not_panics() {
    let Some(gpu) = gpu() else { return };
    let v: Vec<f16> = (0..2 * 9 * 11).map(|i| f16::from_f32(i as f32)).collect();
    let two = tensor(&v, 2, 9, 11).to_gpu_ctx(gpu).unwrap();
    let gray = tensor(&v[..99], 1, 9, 11).to_gpu_ctx(gpu).unwrap();
    let one = f16::from_f32(1.0);
    assert!(gpu.bilateral_filter(&two, 5, one, one).is_err());
    assert!(gpu.fast_detect(&two, one, true).is_err());
    assert!(gpu.cvt_color(&two, ColorConversion::RgbToGray).is_err());
    assert!(gpu.cvt_color(&gray, ColorConversion::GrayToRgb).is_err());
}
