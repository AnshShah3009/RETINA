//! Benchmarks for stereo vision algorithms
//!
//! Compares CPU vs GPU performance for stereo matching operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{GrayImage, Luma};
use std::time::Duration;
#[cfg(feature = "gpu")]
use tokio::runtime::Runtime;

/// Create synthetic stereo pair with known disparity
fn create_stereo_pair(width: u32, height: u32, disparity: i32) -> (GrayImage, GrayImage) {
    let mut left = GrayImage::new(width, height);
    let mut right = GrayImage::new(width, height);

    // Create vertical stripes pattern
    for y in 0..height {
        for x in 0..width {
            let pattern = ((x / 10) % 2) * 200;
            left.put_pixel(x, y, Luma([pattern as u8]));

            // Shift pattern for right image
            let shifted_x = if x >= disparity as u32 {
                x - disparity as u32
            } else {
                0
            };
            let right_pattern = ((shifted_x / 10) % 2) * 200;
            right.put_pixel(x, y, Luma([right_pattern as u8]));
        }
    }

    (left, right)
}

fn benchmark_stereo_block_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("stereo_block_matching");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different image sizes
    for size in [64u32, 128, 256, 512] {
        let (left, right) = create_stereo_pair(size, size, 10);

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &(left.clone(), right.clone()),
            |b, (l, r)| {
                b.iter(|| {
                    use cv_stereo::BlockMatcher;
                    let matcher = BlockMatcher::new()
                        .with_block_size(11)
                        .with_disparity_range(0, 32);
                    let _ = matcher.compute(black_box(l), black_box(r));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_stereo_sgm(c: &mut Criterion) {
    let mut group = c.benchmark_group("stereo_sgm");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // SGM is more computationally intensive, test smaller sizes
    for size in [64u32, 128, 256] {
        let (left, right) = create_stereo_pair(size, size, 10);

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &(left.clone(), right.clone()),
            |b, (l, r)| {
                b.iter(|| {
                    use cv_stereo::SgmMatcher;
                    let matcher = SgmMatcher::new().with_disparity_range(0, 16);
                    let _ = matcher.compute(black_box(l), black_box(r));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_stereo_gpu_block_matching(c: &mut Criterion) {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = c;
        return;
    }

    #[cfg(feature = "gpu")]
    {
        let rt = Runtime::new().expect("tokio runtime for GPU benchmark");
        let gpu_matcher = rt.block_on(async {
            let _ = cv_hal::gpu::GpuContext::init_global().await;
            cv_stereo::GpuStereoMatcher::new(cv_stereo::GpuStereoAlgorithm::BlockMatching {
                block_size: 11,
            })
            .await
            .ok()
        });

        let mut group = c.benchmark_group("stereo_gpu_block_matching");
        group.measurement_time(Duration::from_secs(6));
        group.sample_size(10);

        for size in [128u32, 256] {
            let (left, right) = create_stereo_pair(size, size, 10);

            group.bench_with_input(
                BenchmarkId::new("cpu", format!("{}x{}", size, size)),
                &(left.clone(), right.clone()),
                |b, (l, r)| {
                    b.iter(|| {
                        let matcher = cv_stereo::BlockMatcher::new()
                            .with_block_size(11)
                            .with_disparity_range(0, 32);
                        let _ = matcher.compute(black_box(l), black_box(r));
                    });
                },
            );

            if let Some(matcher) = &gpu_matcher {
                group.bench_with_input(
                    BenchmarkId::new("gpu", format!("{}x{}", size, size)),
                    &(left.clone(), right.clone()),
                    |b, (l, r)| {
                        b.iter(|| {
                            let _ = matcher.compute_disparity(black_box(l), black_box(r), 0, 32);
                        });
                    },
                );
            }
        }

        group.finish();
    }
}

fn benchmark_image_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_processing");
    group.measurement_time(Duration::from_secs(5));

    for size in [256u32, 512, 1024] {
        let img = GrayImage::new(size, size);

        // Gaussian blur
        group.bench_with_input(
            BenchmarkId::new("gaussian_blur", format!("{}x{}", size, size)),
            &img,
            |b, i| {
                b.iter(|| cv_imgproc::gaussian_blur(black_box(i), black_box(1.0)));
            },
        );

        // Sobel edge detection
        group.bench_with_input(
            BenchmarkId::new("sobel_edges", format!("{}x{}", size, size)),
            &img,
            |b, i| {
                b.iter(|| {
                    let _ = cv_imgproc::sobel(black_box(i));
                });
            },
        );

        // Canny edge detection
        group.bench_with_input(
            BenchmarkId::new("canny_edges", format!("{}x{}", size, size)),
            &img,
            |b, i| {
                b.iter(|| cv_imgproc::canny(black_box(i), black_box(50u8), black_box(150u8)));
            },
        );
    }

    group.finish();
}

fn benchmark_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_detection");
    group.measurement_time(Duration::from_secs(5));

    for size in [256u32, 512, 1024] {
        // Create image with corners
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let is_white = ((x / 32) + (y / 32)) % 2 == 0;
                img.put_pixel(x, y, Luma([if is_white { 255 } else { 0 }]));
            }
        }

        // FAST detector
        group.bench_with_input(
            BenchmarkId::new("fast_detector", format!("{}x{}", size, size)),
            &img,
            |b, i| {
                b.iter(|| {
                    let _ =
                        cv_features::fast::fast_detect(black_box(i), black_box(20), black_box(500));
                });
            },
        );

        // Harris corners
        group.bench_with_input(
            BenchmarkId::new("harris_detector", format!("{}x{}", size, size)),
            &img,
            |b, i| {
                b.iter(|| {
                    let _ = cv_features::harris::harris_detect(
                        black_box(i),
                        black_box(3),
                        black_box(3),
                        black_box(0.04),
                        black_box(10000.0),
                    );
                });
            },
        );
    }

    group.finish();
}

fn benchmark_optical_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow");
    group.measurement_time(Duration::from_secs(10));

    for size in [128u32, 256, 512] {
        let (prev, next) = create_stereo_pair(size, size, 5);

        // Lucas-Kanade sparse flow
        let points: Vec<(f32, f32)> = (0..100)
            .map(|i| ((i * 10) as f32 % size as f32, (i * 7) as f32 % size as f32))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("lucas_kanade", format!("{}x{}", size, size)),
            &(prev.clone(), next.clone(), points.clone()),
            |b, (p, n, pts)| {
                b.iter(|| {
                    let _ =
                        cv_video::calc_optical_flow_lk(black_box(p), black_box(n), black_box(pts));
                });
            },
        );

        // Farneback dense flow
        group.bench_with_input(
            BenchmarkId::new("farneback", format!("{}x{}", size, size)),
            &(prev.clone(), next.clone()),
            |b, (p, n)| {
                b.iter(|| {
                    let _ = cv_video::calc_optical_flow_farneback(black_box(p), black_box(n));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_stereo_block_matching,
    benchmark_stereo_sgm,
    benchmark_stereo_gpu_block_matching,
    benchmark_image_processing,
    benchmark_feature_detection,
    benchmark_optical_flow
);
criterion_main!(benches);
