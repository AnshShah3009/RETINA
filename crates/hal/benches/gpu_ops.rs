//! End-to-end cost of the wgpu kernels, measured the way a caller pays for
//! them: every iteration waits for the device to finish (or reads the result
//! back), so CPU-side encoding, pipeline lookup, buffer creation and the GPU
//! work itself all count.
//!
//! * `tiny_threshold_64x64`: a dispatch that does almost no GPU work, so the
//!   number is the fixed per-op overhead.
//! * `pipeline_1080p`: upload, Gaussian blur, Sobel, threshold, download.
//! * `blur_1080p`, `fast_nms_1080p`, `fast_extract_1080p`, `pyramid_1080p`.
//! * `farneback_512`, `tvl1_512`: the dense optical flow solvers.
//!
//! Skips (prints and returns) when no GPU adapter is available.
//!
//!   cargo bench -p cv-hal --bench gpu_ops

use criterion::{Criterion, criterion_group, criterion_main};
use cv_core::{CpuTensor, TensorShape};
use cv_hal::context::{ComputeContext, ThresholdType};
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::optical_flow::{FlowBlurMode, Tvl1Config, farneback, tvl1_optical_flow};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use std::hint::black_box;
use std::time::Duration;

/// Smooth, textured test frame with values in 0..255.
fn frame(w: usize, h: usize, shift: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let xf = x as f32 + shift;
            let yf = y as f32;
            let s = (xf * 0.07).sin() * (yf * 0.05).cos() + ((xf + yf) * 0.013).sin();
            let n = ((x * 7919 + y * 104_729) % 97) as f32 / 97.0;
            v.push(127.5 + 60.0 * s + 20.0 * n);
        }
    }
    v
}

fn upload(ctx: &GpuContext, w: usize, h: usize, shift: f32) -> cv_hal::GpuTensor<f32> {
    CpuTensor::from_vec(frame(w, h, shift), TensorShape::new(1, h, w))
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap()
}

fn wait(ctx: &GpuContext) {
    ctx.wait_idle().unwrap();
}

/// Checkerboard of 23x17 cells with a little noise: plenty of FAST corners.
fn blocks(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let on = x % 23 < 10 && y % 17 < 9;
            let n = ((x * 7919 + y * 104_729) % 13) as f32;
            v.push(if on { 200.0 } else { 40.0 } + n);
        }
    }
    v
}

fn bench_ops(c: &mut Criterion) {
    let Ok(ctx) = GpuContext::new() else {
        eprintln!("skipping gpu_ops: no GPU adapter");
        return;
    };
    let ctx = &ctx;

    let mut group = c.benchmark_group("gpu_ops");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let tiny = upload(ctx, 64, 64, 0.0);
    group.bench_function("tiny_threshold_64x64", |b| {
        b.iter(|| {
            let out = ctx
                .threshold(&tiny, 100.0f32, 255.0, ThresholdType::Binary)
                .unwrap();
            wait(ctx);
            black_box(out)
        });
    });

    let (w, h) = (1920usize, 1080usize);
    let host = CpuTensor::from_vec(frame(w, h, 0.0), TensorShape::new(1, h, w)).unwrap();
    group.bench_function("pipeline_1080p", |b| {
        b.iter(|| {
            let g = host.to_gpu_ctx(ctx).unwrap();
            let blurred = ctx.gaussian_blur(&g, 1.5f32, 7).unwrap();
            let (gx, _gy) = ctx.sobel(&blurred, 1, 1, 3).unwrap();
            let t = ctx
                .threshold(&gx, 10.0f32, 255.0, ThresholdType::Binary)
                .unwrap();
            black_box(t.to_cpu_ctx(ctx).unwrap())
        });
    });

    let img = upload(ctx, w, h, 0.0);
    group.bench_function("blur_1080p", |b| {
        b.iter(|| {
            let out = ctx.gaussian_blur(&img, 1.5f32, 7).unwrap();
            wait(ctx);
            black_box(out)
        });
    });

    group.bench_function("fast_nms_1080p", |b| {
        b.iter(|| {
            let out = ctx.fast_detect(&img, 20.0f32, true).unwrap();
            wait(ctx);
            black_box(out)
        });
    });

    let corners = CpuTensor::from_vec(blocks(w, h), TensorShape::new(1, h, w))
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap();
    let scores = ctx.fast_detect(&corners, 20.0f32, true).unwrap();
    group.bench_function("fast_extract_1080p", |b| {
        b.iter(|| black_box(cv_hal::gpu_kernels::fast::extract_keypoints(ctx, &scores).unwrap()));
    });

    group.bench_function("pyramid_1080p_8x1.2", |b| {
        b.iter(|| {
            let p = cv_hal::gpu_kernels::pyramid::build_pyramid(ctx, &img, 8, 1.2).unwrap();
            wait(ctx);
            black_box(p)
        });
    });

    group.bench_function("pyrdown_1080p_x4", |b| {
        b.iter(|| {
            let mut cur = img.clone();
            for _ in 0..4 {
                cur = ctx.pyramid_down(&cur).unwrap();
            }
            wait(ctx);
            black_box(cur)
        });
    });

    let (fw, fh) = (512usize, 512usize);
    let prev = upload(ctx, fw, fh, 0.0);
    let next = upload(ctx, fw, fh, 1.5);
    group.bench_function("farneback_512", |b| {
        b.iter(|| {
            let f = farneback(ctx, &prev, &next, 3, 0.5, 15, 3, 5, 1.2, FlowBlurMode::Box).unwrap();
            wait(ctx);
            black_box(f)
        });
    });
    group.finish();

    let mut slow = c.benchmark_group("gpu_ops_slow");
    slow.measurement_time(Duration::from_secs(10));
    slow.sample_size(10);
    slow.bench_function("tvl1_512", |b| {
        b.iter(|| {
            let f = tvl1_optical_flow(ctx, &prev, &next, Tvl1Config::default()).unwrap();
            wait(ctx);
            black_box(f)
        });
    });
    slow.finish();
}

criterion_group!(benches, bench_ops);
criterion_main!(benches);
