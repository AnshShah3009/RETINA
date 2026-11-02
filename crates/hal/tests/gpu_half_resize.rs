//! f16 and bf16 `resize` on the CPU and the GPU.
//!
//! The GPU used to refuse both types (`resize_f16.wgsl` computed the sampling
//! positions in f16 and `resize_bf16.wgsl` read one value per u32 word; no
//! host called either). `resize_packed16.wgsl` now runs both on any device.
//!
//! * Bilinear is bit-identical to the CPU backend for the same type: the host
//!   hands the shader the CPU's taps (source indices and weights, already
//!   rounded to the type) and the shader rounds every product and sum to the
//!   type in the CPU's order.
//! * Lanczos-4 has no CPU implementation. It is the f32 Lanczos shader's
//!   arithmetic on the exactly widened values, rounded once, so the reference
//!   is the f32 GPU result narrowed to the type.
//!
//! The CPU computed the sampling positions in the tensor's own type, which is
//! wrong for both 16-bit formats: `(x + 0.5) * src_w` overflows f16 above
//! 65504 (a 512-wide source resized to 256 sampled its last column from
//! output column 128 on), and bf16 cannot count past 256, so positions on a
//! wider image were off by whole pixels. Positions are now computed in f32
//! for 16-bit types and only the interpolation weights are rounded to the
//! type; `cpu_*_resize_tracks_f32` would fail on the old code.
//!
//! Needs `--features half-precision`; the GPU tests skip without an adapter.
#![cfg(feature = "half-precision")]

use cv_core::float::Float;
use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, Interpolation};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use half::{bf16, f16};

trait Half: Float + bytemuck::Pod + bytemuck::Zeroable + std::fmt::Display + 'static {
    const NAME: &'static str;
    /// A few units in the last place at 1.0.
    const TOL: f32;
    fn bits(self) -> u16;
}
impl Half for f16 {
    const NAME: &'static str = "f16";
    const TOL: f32 = 4.0 / 1024.0;
    fn bits(self) -> u16 {
        self.to_bits()
    }
}
impl Half for bf16 {
    const NAME: &'static str = "bf16";
    const TOL: f32 = 4.0 / 128.0;
    fn bits(self) -> u16 {
        self.to_bits()
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
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 8) as f32 / (1u32 << 24) as f32 * (hi - lo)
    }
}

/// Random values in `[lo, hi)`, rounded to `T`.
fn random<T: Half>(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<T> {
    let mut r = Lcg(seed);
    (0..len).map(|_| T::from_f32(r.range(lo, hi))).collect()
}

/// A period-4 sawtooth along x, so a sample taken a pixel off is far off.
fn sawtooth<T: Half>(c: usize, h: usize, w: usize) -> Vec<T> {
    (0..c * h * w)
        .map(|i| T::from_f32(((i % w) % 4) as f32 / 3.0 + (i / (h * w)) as f32))
        .collect()
}

fn tensor<T: Half>(v: Vec<T>, c: usize, h: usize, w: usize) -> CpuTensor<T> {
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

fn widen<T: Half>(t: &CpuTensor<T>) -> CpuTensor<f32> {
    let v = t
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| Float::to_f32(x))
        .collect();
    Tensor::from_vec(v, t.shape).unwrap()
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

/// (c, h, w) -> (nw, nh). Odd element counts leave the last output word half
/// full; odd widths make two rows (and two planes) share a word.
const CASES: &[((usize, usize, usize), (usize, usize))] = &[
    ((1, 4, 4), (2, 2)),
    ((1, 7, 9), (13, 5)),
    ((3, 7, 9), (5, 3)),
    ((3, 37, 41), (64, 29)),
    ((2, 1, 11), (3, 1)),
    ((1, 5, 1), (1, 9)),
    ((3, 16, 600), (300, 7)),
    ((1, 3, 513), (1025, 2)),
];

// ------------------------------------------------------------------ CPU

fn cpu_resize_tracks_f32<T: Half>() {
    let cpu = CpuBackend::new().unwrap();
    for &((c, h, w), (nw, nh)) in CASES {
        let input = tensor(sawtooth::<T>(c, h, w), c, h, w);
        let got = cpu.resize(&input, (nw, nh)).unwrap();
        let want = cpu.resize(&widen(&input), (nw, nh)).unwrap();
        let worst = got
            .as_slice()
            .unwrap()
            .iter()
            .zip(want.as_slice().unwrap())
            .map(|(&a, b)| (Float::to_f32(a) - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(
            worst <= T::TOL,
            "{} ({c},{h},{w}) -> {nw}x{nh}: worst relative error {worst} vs f32",
            T::NAME
        );
    }
}

#[test]
fn cpu_f16_resize_tracks_f32() {
    cpu_resize_tracks_f32::<f16>();
}

#[test]
fn cpu_bf16_resize_tracks_f32() {
    cpu_resize_tracks_f32::<bf16>();
}

// ------------------------------------------------------------------ GPU

fn gpu_resize<T: Float + bytemuck::Pod + std::fmt::Debug + Send + Sync + 'static>(
    g: &GpuContext,
    input: &CpuTensor<T>,
    nw: usize,
    nh: usize,
    interp: Interpolation,
) -> CpuTensor<T> {
    let on_gpu = input.to_gpu_ctx(g).unwrap();
    let out = g.resize_with(&on_gpu, (nw, nh), interp).unwrap();
    assert_eq!(out.shape, TensorShape::new(input.shape.channels, nh, nw));
    out.to_cpu_ctx(g).unwrap()
}

fn bilinear_matches_cpu<T: Half>() {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    for (k, &((c, h, w), (nw, nh))) in CASES.iter().enumerate() {
        for (data, what) in [
            (random::<T>(c * h * w, -2.0, 3.0, 7 + k as u32), "random"),
            (sawtooth::<T>(c, h, w), "sawtooth"),
        ] {
            let input = tensor(data, c, h, w);
            let want = cpu.resize(&input, (nw, nh)).unwrap();
            let got = gpu_resize(g, &input, nw, nh, Interpolation::Linear);
            assert_bits_eq(
                got.as_slice().unwrap(),
                want.as_slice().unwrap(),
                &format!("{} bilinear {what} ({c},{h},{w}) -> {nw}x{nh}", T::NAME),
            );
        }
    }
}

#[test]
fn gpu_f16_bilinear_resize_matches_cpu() {
    bilinear_matches_cpu::<f16>();
}

#[test]
fn gpu_bf16_bilinear_resize_matches_cpu() {
    bilinear_matches_cpu::<bf16>();
}

fn lanczos_matches_f32_narrowed<T: Half>() {
    let Some(g) = gpu() else { return };
    for (k, &((c, h, w), (nw, nh))) in CASES.iter().enumerate() {
        let input = tensor(random::<T>(c * h * w, -2.0, 3.0, 31 + k as u32), c, h, w);
        let reference: Vec<T> = gpu_resize(g, &widen(&input), nw, nh, Interpolation::Lanczos)
            .as_slice()
            .unwrap()
            .iter()
            .map(|&v| T::from_f32(v))
            .collect();
        let got = gpu_resize(g, &input, nw, nh, Interpolation::Lanczos);
        assert_bits_eq(
            got.as_slice().unwrap(),
            &reference,
            &format!("{} Lanczos ({c},{h},{w}) -> {nw}x{nh}", T::NAME),
        );
    }
}

#[test]
fn gpu_f16_lanczos_resize_matches_f32_path() {
    lanczos_matches_f32_narrowed::<f16>();
}

#[test]
fn gpu_bf16_lanczos_resize_matches_f32_path() {
    lanczos_matches_f32_narrowed::<bf16>();
}

/// Planar CHW for 16-bit tensors too: constant planes stay constant.
fn constant_planes_stay_constant<T: Half>() {
    let Some(g) = gpu() else { return };
    let (h, w) = (9usize, 7usize);
    let values = [0.0f32, 100.0, 200.0];
    let data: Vec<T> = values
        .iter()
        .flat_map(|&v| std::iter::repeat_n(T::from_f32(v), h * w))
        .collect();
    let input = tensor(data, 3, h, w);
    for interp in [Interpolation::Linear, Interpolation::Lanczos] {
        let out = gpu_resize(g, &input, 5, 11, interp);
        let d = out.as_slice().unwrap();
        for (ch, &v) in values.iter().enumerate() {
            let want = T::from_f32(v);
            assert!(
                d[ch * 55..(ch + 1) * 55]
                    .iter()
                    .all(|x| x.bits() == want.bits()),
                "{} {interp:?}: plane {ch} must stay {v}",
                T::NAME
            );
        }
    }
}

#[test]
fn gpu_f16_resize_keeps_constant_planes() {
    constant_planes_stay_constant::<f16>();
}

#[test]
fn gpu_bf16_resize_keeps_constant_planes() {
    constant_planes_stay_constant::<bf16>();
}

/// Empty target: empty tensor; empty source: zeros. CPU and GPU agree.
fn empty_sizes<T: Half>() {
    let cpu = CpuBackend::new().unwrap();
    let g = gpu();
    let cases: &[((usize, usize, usize), (usize, usize))] = &[
        ((3, 0, 0), (5, 3)),
        ((1, 0, 7), (3, 3)),
        ((3, 7, 0), (5, 4)),
        ((3, 5, 5), (0, 4)),
        ((3, 5, 5), (4, 0)),
        ((0, 5, 5), (3, 3)),
    ];
    for &((c, h, w), (nw, nh)) in cases {
        let input = tensor(vec![T::ONE; c * h * w], c, h, w);
        let shape = TensorShape::new(c, nh, nw);
        let zeros = vec![T::ZERO; shape.len()];
        let what = format!("{} ({c},{h},{w}) -> {nw}x{nh}", T::NAME);

        let out = cpu.resize(&input, (nw, nh)).unwrap();
        assert_eq!(out.shape, shape, "CPU {what}");
        assert_bits_eq(out.as_slice().unwrap(), &zeros, &format!("CPU {what}"));

        let Some(g) = g else { continue };
        let on_gpu = input.to_gpu_ctx(g).unwrap();
        for interp in [Interpolation::Linear, Interpolation::Lanczos] {
            let out = g.resize_with(&on_gpu, (nw, nh), interp).unwrap();
            assert_eq!(out.shape, shape, "GPU {interp:?} {what}");
            assert_eq!(out.storage.len(), zeros.len(), "GPU {interp:?} {what}");
            if !zeros.is_empty() {
                let back = out.to_cpu_ctx(g).unwrap();
                assert_bits_eq(
                    back.as_slice().unwrap(),
                    &zeros,
                    &format!("GPU {interp:?} {what}"),
                );
            }
        }
    }
}

#[test]
fn f16_empty_sizes() {
    empty_sizes::<f16>();
}

#[test]
fn bf16_empty_sizes() {
    empty_sizes::<bf16>();
}
