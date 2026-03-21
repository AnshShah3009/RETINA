//! Benchmarks for stereo vision algorithms
//!
//! Compares CPU vs GPU performance for stereo matching operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{GrayImage, Luma};
use std::time::Duration;

/// Create synthetic stereo pair with known disparity
fn create_stereo_pair(width: u32, height: u32, disparity: i32) -> (GrayImage, GrayImage) {
    let mut left = GrayImage::new(width, height);
    let mut right = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pattern = ((x / 10) % 2) * 200;
            left.put_pixel(x, y, Luma([pattern as u8]));

            let shifted_x = x.saturating_sub(disparity as u32);
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

    for size in [64u32, 128, 256, 512] {
        let (left, right) = create_stereo_pair(size, size, 10);

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &(left.clone(), right.clone()),
            |b, (l, r)| {
                b.iter(|| {
                    let _ = cv_calib3d::stereo_block_match(black_box(l), black_box(r), 11, 32);
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

    for size in [64u32, 128, 256] {
        let (left, right) = create_stereo_pair(size, size, 10);

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &(left.clone(), right.clone()),
            |b, (l, r)| {
                b.iter(|| {
                    let _ = cv_calib3d::stereo_sgm(black_box(l), black_box(r), 16);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_stereo_block_matching,
    benchmark_stereo_sgm
);
criterion_main!(benches);
