//! CPU kernels that go through `wide` SIMD types, on a 1920x1080 frame.
//!
//! Covers the `u8x32` morphology loop and the `f32x8` threshold, row
//! convolution and colour-conversion paths in `cv-hal`, plus `cv-imgproc`'s
//! own threshold and separable convolution. Used to check that a `wide`
//! upgrade does not slow these down; compare runs with `--save-baseline` /
//! `--baseline`.

use criterion::{Criterion, criterion_group, criterion_main};
use cv_core::TensorShape;
use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_hal::context::{ComputeContext, MorphologyType, ThresholdType as HalThreshold};
use cv_hal::cpu::CpuBackend;
use cv_hal::cpu::simd::{convolve_row_1d, rgb_to_gray_simd};
use cv_imgproc::{BorderMode, ThresholdType, gaussian_kernel_1d, separable_convolve, threshold};
use image::GrayImage;
use std::hint::black_box;

const W: usize = 1920;
const H: usize = 1080;

/// Deterministic, texture-like bytes (no RNG dependency).
fn frame_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((i % W) as u64 / 4 + (state & 63)) as u8
        })
        .collect()
}

fn gray_frame() -> GrayImage {
    GrayImage::from_raw(W as u32, H as u32, frame_bytes(W * H)).unwrap()
}

fn bench_imgproc(c: &mut Criterion) {
    let img = gray_frame();
    let mut group = c.benchmark_group("imgproc_1080p");
    for (name, typ) in [
        ("threshold_binary", ThresholdType::Binary),
        ("threshold_trunc", ThresholdType::Trunc),
        ("threshold_tozero", ThresholdType::ToZero),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| threshold(black_box(&img), 100, 255, typ))
        });
    }
    let k = gaussian_kernel_1d(1.5, 7);
    group.bench_function("separable_gaussian7", |b| {
        b.iter(|| separable_convolve(black_box(&img), &k, BorderMode::Replicate))
    });
    group.finish();
}

fn bench_hal(c: &mut Criterion) {
    let cpu = CpuBackend::new().unwrap();
    let mut group = c.benchmark_group("hal_cpu_1080p");

    let input: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(frame_bytes(W * H), TensorShape::new(1, H, W)).unwrap();
    let kernel: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(vec![1u8; 9], TensorShape::new(1, 3, 3)).unwrap();
    group.bench_function("erode_3x3", |b| {
        b.iter(|| {
            cpu.morphology(black_box(&input), MorphologyType::Erode, &kernel, 1)
                .unwrap()
        })
    });
    let cross: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
        vec![
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        ],
        TensorShape::new(1, 5, 5),
    )
    .unwrap();
    group.bench_function("dilate_cross5", |b| {
        b.iter(|| {
            cpu.morphology(black_box(&input), MorphologyType::Dilate, &cross, 1)
                .unwrap()
        })
    });

    let input_f32: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(
        frame_bytes(W * H).into_iter().map(f32::from).collect(),
        TensorShape::new(1, H, W),
    )
    .unwrap();
    group.bench_function("threshold_f32_binary", |b| {
        b.iter(|| {
            cpu.threshold(black_box(&input_f32), 100.0, 255.0, HalThreshold::Binary)
                .unwrap()
        })
    });

    let rgb = frame_bytes(W * H * 3);
    let mut gray = vec![0u8; W * H];
    group.bench_function("rgb_to_gray_simd", |b| {
        b.iter(|| rgb_to_gray_simd(black_box(&rgb), &mut gray))
    });

    let kernel_1d = gaussian_kernel_1d(1.5, 7);
    let radius = kernel_1d.len() / 2;
    let row: Vec<f32> = frame_bytes(W + 2 * radius)
        .into_iter()
        .map(f32::from)
        .collect();
    let mut out = vec![0.0f32; W];
    group.bench_function("convolve_row_1d_x1080", |b| {
        b.iter(|| {
            for _ in 0..H {
                convolve_row_1d(black_box(&row), &mut out, &kernel_1d, radius);
            }
        })
    });
    group.finish();
}

fn bench_core(c: &mut Criterion) {
    let a: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(
        frame_bytes(W * H).into_iter().map(f32::from).collect(),
        TensorShape::new(1, H, W),
    )
    .unwrap();
    let b2 = a.clone();
    c.bench_function("core_tensor_mul_1080p", |b| {
        b.iter(|| black_box(&a).mul(&b2).unwrap())
    });
}

criterion_group!(benches, bench_imgproc, bench_hal, bench_core);
criterion_main!(benches);
