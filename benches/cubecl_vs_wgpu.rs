//! Benchmark: CubeCL kernels vs WGPU (WGSL shader) kernels vs CPU
//!
//! Compares three backend paths for identical operations:
//!   1. CPU — uses `CpuBackend` via the `ComputeContext` trait
//!   2. WGPU — calls the low-level gpu_kernel functions that dispatch WGSL compute shaders
//!   3. CubeCL — (gated behind `cubecl` feature) calls CubeCL-generated kernels
//!
//! Run:
//!   cargo bench --bench cubecl_vs_wgpu
//!   cargo bench --bench cubecl_vs_wgpu -- --quick      # fast smoke-test
//!   cargo bench --bench cubecl_vs_wgpu -- threshold    # single group

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cv_core::{CpuTensor, KeyPointF32, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use std::marker::PhantomData;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers: test-data generators
// ---------------------------------------------------------------------------

/// Create a 1-channel f32 gradient image on the CPU.
fn cpu_gradient_image(width: usize, height: usize) -> CpuTensor<f32> {
    let shape = TensorShape::new(1, height, width);
    let data: Vec<f32> = (0..height * width)
        .map(|i| {
            let x = (i % width) as f32;
            let y = (i / width) as f32;
            ((x + y) * 0.1).sin().abs() * 255.0
        })
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Upload a CpuTensor<f32> to the GPU, returning a GpuTensor.
fn upload_f32(ctx: &GpuContext, cpu: &CpuTensor<f32>) -> cv_hal::GpuTensor<f32> {
    let data = cpu.storage.as_slice().unwrap();
    let storage =
        cv_hal::storage::WgpuGpuStorage::from_slice_ctx(ctx, data).expect("GPU upload failed");
    Tensor {
        storage,
        shape: cpu.shape,
        dtype: cpu.dtype,
        _phantom: PhantomData,
    }
}

/// Upload a CpuTensor<u8> to the GPU.
fn upload_u8(
    ctx: &GpuContext,
    data: &[u8],
    shape: TensorShape,
) -> Tensor<u8, cv_hal::storage::WgpuGpuStorage<u8>> {
    let storage =
        cv_hal::storage::WgpuGpuStorage::from_slice_ctx(ctx, data).expect("GPU upload failed");
    Tensor {
        storage,
        shape,
        dtype: cv_core::DataType::U8,
        _phantom: PhantomData,
    }
}

/// Generate a sparse CSR matrix (row_ptr, col_indices, values) for a tridiagonal matrix.
fn sparse_tridiag(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0u32);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as u32);
            values.push(-1.0_f32);
        }
        col_indices.push(i as u32);
        values.push(2.0_f32);
        if i + 1 < n {
            col_indices.push((i + 1) as u32);
            values.push(-1.0_f32);
        }
        row_ptr.push(col_indices.len() as u32);
    }
    (row_ptr, col_indices, values)
}

/// Generate BRIEF test pattern (random-ish but deterministic).
fn brief_pattern(num_pairs: usize) -> Vec<cv_hal::gpu_kernels::brief::BRIEFPoint> {
    (0..num_pairs)
        .map(|i| {
            let seed = (i as f32) * 0.618;
            cv_hal::gpu_kernels::brief::BRIEFPoint {
                x1: (seed * 7.3).sin() * 12.0,
                y1: (seed * 3.7).cos() * 12.0,
                x2: (seed * 11.1).sin() * 12.0,
                y2: (seed * 5.3).cos() * 12.0,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU context helper — shared across all benchmarks in this file
// ---------------------------------------------------------------------------

/// Try to obtain a GpuContext. Returns None when no GPU is available.
fn try_gpu() -> Option<GpuContext> {
    GpuContext::new().ok()
}

// ---------------------------------------------------------------------------
// 1. Threshold
// ---------------------------------------------------------------------------

fn bench_threshold(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("threshold");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &(w, h) in &[(640, 480), (1920, 1080)] {
        let label = format!("{}x{}", w, h);
        let cpu_img = cpu_gradient_image(w, h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                black_box(
                    cpu_backend
                        .threshold(&cpu_img, 128.0_f32, 255.0, ThresholdType::Binary)
                        .unwrap(),
                )
            });
        });

        // -- WGPU (WGSL shader) --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            // Warmup: first call compiles the shader
            let _ = cv_hal::gpu_kernels::threshold::threshold(
                ctx,
                &gpu_img,
                128.0_f32,
                255.0,
                ThresholdType::Binary,
            );
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::threshold::threshold(
                        black_box(ctx),
                        black_box(&gpu_img),
                        128.0_f32,
                        255.0,
                        ThresholdType::Binary,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        // -- CubeCL (placeholder) --
        #[cfg(feature = "cubecl")]
        {
            // TODO: invoke CubeCL threshold kernel here once integrated
            // group.bench_with_input(BenchmarkId::new("cubecl", &label), &(), |b, _| { ... });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Sobel
// ---------------------------------------------------------------------------

fn bench_sobel(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("sobel");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &(w, h) in &[(640, 480), (1920, 1080)] {
        let label = format!("{}x{}", w, h);
        let cpu_img = cpu_gradient_image(w, h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| black_box(cpu_backend.sobel(&cpu_img, 1, 1, 3).unwrap()));
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            let _ = cv_hal::gpu_kernels::sobel::sobel(ctx, &gpu_img, 1, 1, 3);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::sobel::sobel(
                        black_box(ctx),
                        black_box(&gpu_img),
                        1,
                        1,
                        3,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL sobel kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Bilateral filter
// ---------------------------------------------------------------------------

fn bench_bilateral(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("bilateral_filter");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(15);

    for &(w, h) in &[(640, 480), (1920, 1080)] {
        let label = format!("{}x{}", w, h);
        let cpu_img = cpu_gradient_image(w, h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                black_box(
                    cpu_backend
                        .bilateral_filter(&cpu_img, 9, 75.0_f32, 75.0)
                        .unwrap(),
                )
            });
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            let _ =
                cv_hal::gpu_kernels::bilateral::bilateral_filter(ctx, &gpu_img, 9, 75.0_f32, 75.0);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::bilateral::bilateral_filter(
                        black_box(ctx),
                        black_box(&gpu_img),
                        9,
                        75.0_f32,
                        75.0,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL bilateral kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Resize (bilinear)
// ---------------------------------------------------------------------------

fn bench_resize(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("resize_bilinear");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Downscale 1920x1080 -> 960x540, and 3840x2160 -> 1920x1080
    for &(src_w, src_h, dst_w, dst_h) in &[(1920, 1080, 960, 540), (3840, 2160, 1920, 1080)] {
        let label = format!("{}x{}->{}x{}", src_w, src_h, dst_w, dst_h);
        let cpu_img = cpu_gradient_image(src_w, src_h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| black_box(cpu_backend.resize(&cpu_img, (dst_w, dst_h)).unwrap()));
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            let _ = cv_hal::gpu_kernels::resize::resize(ctx, &gpu_img, dst_w as u32, dst_h as u32);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::resize::resize(
                        black_box(ctx),
                        black_box(&gpu_img),
                        dst_w as u32,
                        dst_h as u32,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL resize kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Convolution 2D
// ---------------------------------------------------------------------------

fn bench_convolve_2d(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("convolve_2d");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // 3x3 edge-detect kernel
    let kernel_data = vec![-1.0_f32, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
    let kernel_shape = TensorShape::new(1, 3, 3);
    let cpu_kernel: CpuTensor<f32> = Tensor::from_vec(kernel_data.clone(), kernel_shape).unwrap();

    for &(w, h) in &[(640, 480), (1920, 1080)] {
        let label = format!("{}x{}_k3x3", w, h);
        let cpu_img = cpu_gradient_image(w, h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                black_box(
                    cpu_backend
                        .convolve_2d(&cpu_img, &cpu_kernel, BorderMode::Replicate)
                        .unwrap(),
                )
            });
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            let gpu_kernel = upload_f32(ctx, &cpu_kernel);

            let _ = cv_hal::gpu_kernels::convolve::convolve_2d(
                ctx,
                &gpu_img,
                &gpu_kernel,
                BorderMode::Replicate,
            );
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::convolve::convolve_2d(
                        black_box(ctx),
                        black_box(&gpu_img),
                        black_box(&gpu_kernel),
                        BorderMode::Replicate,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL convolve kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. NMS (pixel-level non-maximum suppression)
// ---------------------------------------------------------------------------

fn bench_nms(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("nms");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &(w, h) in &[(640, 480), (1920, 1080)] {
        let label = format!("{}x{}", w, h);
        let cpu_img = cpu_gradient_image(w, h);

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| black_box(cpu_backend.nms(&cpu_img, 30.0_f32, 3).unwrap()));
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_img = upload_f32(ctx, &cpu_img);
            let _ = cv_hal::gpu_kernels::nms::nms_pixel(ctx, &gpu_img, 30.0, 3);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::nms::nms_pixel(
                        black_box(ctx),
                        black_box(&gpu_img),
                        30.0,
                        3,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL NMS kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. SpMV (sparse matrix-vector multiply)
// ---------------------------------------------------------------------------

fn bench_spmv(c: &mut Criterion) {
    let cpu_backend = CpuBackend::new().unwrap();
    let gpu = try_gpu();

    let mut group = c.benchmark_group("spmv");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &n in &[10_000usize, 100_000] {
        let label = format!("n={}", n);
        let (row_ptr, col_idx, vals) = sparse_tridiag(n);
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let x_shape = TensorShape::new(1, n, 1);
        let cpu_x: CpuTensor<f32> = Tensor::from_vec(x_data.clone(), x_shape).unwrap();

        // -- CPU --
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                black_box(
                    cpu_backend
                        .spmv(
                            black_box(&row_ptr),
                            black_box(&col_idx),
                            black_box(&vals),
                            &cpu_x,
                        )
                        .unwrap(),
                )
            });
        });

        // -- WGPU --
        if let Some(ctx) = &gpu {
            let gpu_x = upload_f32(ctx, &cpu_x);
            let _ = cv_hal::gpu_kernels::sparse::spmv(ctx, &row_ptr, &col_idx, &vals, &gpu_x);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::sparse::spmv(
                        black_box(ctx),
                        black_box(&row_ptr),
                        black_box(&col_idx),
                        black_box(&vals),
                        black_box(&gpu_x),
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL SpMV kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 8. BRIEF descriptor
// ---------------------------------------------------------------------------

fn bench_brief(c: &mut Criterion) {
    let gpu = try_gpu();

    let mut group = c.benchmark_group("brief_descriptor");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let width = 640usize;
    let height = 480usize;
    let img_data: Vec<u8> = (0..width * height)
        .map(|i| ((i as f32 * 0.03).sin().abs() * 255.0) as u8)
        .collect();
    let img_shape = TensorShape::new(1, height, width);

    let pattern = brief_pattern(256);

    for &num_kp in &[100usize, 500, 2000] {
        let label = format!("kp={}", num_kp);
        let keypoints: Vec<KeyPointF32> = (0..num_kp)
            .map(|i| KeyPointF32 {
                x: 20.0 + (i % (width - 40)) as f32,
                y: 20.0 + (i / (width - 40).max(1)) as f32 % (height - 40) as f32,
                size: 7.0,
                angle: 0.0,
                response: 1.0,
                octave: 0,
                class_id: 0,
                padding: 0,
            })
            .collect();

        // BRIEF is GPU-only in this project, so we only have WGPU.
        if let Some(ctx) = &gpu {
            let gpu_img = upload_u8(ctx, &img_data, img_shape);

            // Warmup
            let _ = cv_hal::gpu_kernels::brief::compute_brief(ctx, &gpu_img, &keypoints, &pattern);
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::brief::compute_brief(
                        black_box(ctx),
                        black_box(&gpu_img),
                        black_box(&keypoints),
                        black_box(&pattern),
                    )
                    .unwrap();
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cubecl")]
        {
            // TODO: CubeCL BRIEF kernel
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

criterion_group!(
    cubecl_vs_wgpu,
    bench_threshold,
    bench_sobel,
    bench_bilateral,
    bench_resize,
    bench_convolve_2d,
    bench_nms,
    bench_spmv,
    bench_brief,
);
criterion_main!(cubecl_vs_wgpu);
