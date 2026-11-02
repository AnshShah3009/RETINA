//! GPU `morphology` must equal the CPU backend byte for byte.
//!
//! `shaders/morphology.wgsl` did not validate until the shader was rewritten
//! (it used `var v = if (..) {..} else {..}`, which WGSL does not have), so
//! the GPU path had never run. The CPU backend is the reference: anchor at
//! `(kw / 2, kh / 2)`, replicate border, every nonzero mask byte a live tap,
//! CHW planes processed independently. The cases below cover each of those:
//! one-sided and even-sized masks (off-centre sampling at the border), widths
//! that are not a multiple of four (rows straddling packed words), several
//! channels (planes starting mid-word), the CPU's separable rectangle path
//! (full masks above 5x5) and its 32-lane SIMD path (width >= 32), and
//! repeated iterations. Skips when no adapter is available.

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, MorphologyType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU morphology test (no adapter): {e}");
            None
        }
    }
}

fn lcg(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

/// Noise over blocks, so erode and dilate move real edges, not only noise.
fn image(c: usize, h: usize, w: usize, seed: u32) -> CpuTensor<u8> {
    let noise = lcg(c * h * w, seed);
    let px = (0..c * h * w)
        .map(|i| {
            let (x, y) = (i % w, (i / w) % h);
            let block = if (x / 5 + y / 3) % 2 == 0 { 180u8 } else { 40 };
            block.wrapping_add(noise[i] / 4)
        })
        .collect();
    Tensor::from_vec(px, TensorShape::new(c, h, w)).unwrap()
}

fn mask(kh: usize, kw: usize, taps: Vec<u8>) -> CpuTensor<u8> {
    assert_eq!(taps.len(), kh * kw);
    Tensor::from_vec(taps, TensorShape::new(1, kh, kw)).unwrap()
}

fn masks() -> Vec<(&'static str, CpuTensor<u8>)> {
    vec![
        ("1x1", mask(1, 1, vec![1])),
        ("3x3 full", mask(3, 3, vec![1; 9])),
        ("3x3 cross", mask(3, 3, vec![0, 1, 0, 1, 1, 1, 0, 1, 0])),
        // Only the left tap: offset (-1, 0). Replicate and the identity
        // border differ at column 0 for exactly this mask.
        ("1x3 left only", mask(1, 3, vec![1, 0, 0])),
        ("3x1 bottom only", mask(3, 1, vec![0, 0, 1])),
        // Even sizes put the anchor right of / below the centre.
        ("2x4 even", mask(2, 4, vec![1, 0, 1, 1, 0, 1, 0, 1])),
        ("4x2 even", mask(4, 2, vec![1, 1, 0, 1, 1, 0, 0, 1])),
        (
            "2x5 sparse",
            mask(2, 5, vec![0, 0, 1, 0, 7, 255, 0, 0, 0, 1]),
        ),
        ("1x9 line", mask(1, 9, vec![1; 9])),
        // Full rectangles above 5x5 take the CPU's separable path.
        ("7x7 full", mask(7, 7, vec![1; 49])),
        ("6x9 full", mask(6, 9, vec![1; 54])),
        // No live tap: every pixel is the reduction seed.
        ("3x3 empty", mask(3, 3, vec![0; 9])),
    ]
}

fn run(gpu: &GpuContext, cpu: &CpuBackend, c: usize, h: usize, w: usize, seed: u32) {
    let img = image(c, h, w, seed);
    let gimg = img.to_gpu_ctx(gpu).unwrap();
    for (mname, k) in masks() {
        let gk = k.to_gpu_ctx(gpu).unwrap();
        for typ in [
            MorphologyType::Erode,
            MorphologyType::Dilate,
            MorphologyType::Open,
            MorphologyType::Close,
        ] {
            for iterations in [1u32, 2, 3] {
                let want = cpu.morphology(&img, typ, &k, iterations).unwrap();
                let want = want.as_slice().unwrap();
                let got = gpu
                    .morphology(&gimg, typ, &gk, iterations)
                    .unwrap_or_else(|e| panic!("GPU {typ:?} {mname} failed: {e}"));
                let got = got.to_cpu_ctx(gpu).unwrap();
                let got = got.as_slice().unwrap();
                assert_eq!(got.len(), want.len());
                let bad: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
                assert!(
                    bad.is_empty(),
                    "{c}x{h}x{w} {typ:?} {mname} x{iterations}: {} of {} pixels differ, \
                     first at index {} (got {}, want {})",
                    bad.len(),
                    want.len(),
                    bad[0],
                    got[bad[0]],
                    want[bad[0]],
                );
            }
        }
    }
}

#[test]
fn gpu_morphology_matches_cpu_single_channel() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    for &(h, w, seed) in &[
        (1usize, 1usize, 1u32),
        (1, 7, 2),
        (5, 3, 3),
        (17, 23, 4),
        (31, 66, 5),
        (40, 40, 6),
    ] {
        run(gpu, &cpu, 1, h, w, seed);
    }
}

/// A 3-plane tensor whose plane size (13 * 11 = 143) is not a multiple of
/// four, so the second and third planes start in the middle of a word.
#[test]
fn gpu_morphology_matches_cpu_per_plane() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    run(gpu, &cpu, 3, 13, 11, 7);
    run(gpu, &cpu, 2, 9, 37, 8);
}

/// The row `[1, 2, 3]` dilated by a left-only tap is `[1, 1, 2]` with a
/// replicate border. The identity border the old shader used gives
/// `[0, 1, 2]`.
#[test]
fn gpu_morphology_replicates_the_edge_pixel() {
    let Some(gpu) = require_gpu() else { return };
    let img: CpuTensor<u8> = Tensor::from_vec(vec![1, 2, 3], TensorShape::new(1, 1, 3)).unwrap();
    let k = mask(1, 3, vec![1, 0, 0]);
    let out = gpu
        .morphology(
            &img.to_gpu_ctx(gpu).unwrap(),
            MorphologyType::Dilate,
            &k.to_gpu_ctx(gpu).unwrap(),
            1,
        )
        .unwrap();
    assert_eq!(out.to_cpu_ctx(gpu).unwrap().as_slice().unwrap(), &[1, 1, 2]);
}
