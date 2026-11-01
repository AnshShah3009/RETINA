//! Performance Benchmark Suite - CPU Baseline
//!
//! Run with:
//!   cargo bench --bench cubecl_perf

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

const BENCH_SIZES: &[usize] = &[1024, 4096, 16384];

fn bench_element_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_element_add");
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);

    for size in BENCH_SIZES {
        group.bench_with_input(BenchmarkId::new("size", *size), size, |b, &size| {
            b.iter(|| {
                let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
                let b_vec: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();
                let mut result = vec![0.0f32; size];
                for i in 0..size {
                    result[i] = a[i] + b_vec[i];
                }
                result
            });
        });
    }

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_relu");
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);

    for size in BENCH_SIZES {
        group.bench_with_input(BenchmarkId::new("size", *size), size, |b, &size| {
            b.iter(|| {
                let input: Vec<f32> = (0..size)
                    .map(|i| (i as f32 - size as f32 / 2.0) * 0.1)
                    .collect();
                let mut output = vec![0.0f32; size];
                for i in 0..size {
                    output[i] = input[i].max(0.0);
                }
                output
            });
        });
    }

    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_knn_8");
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);
    let sizes = [1024, 4096];

    for size in sizes.iter() {
        let n = *size;
        group.bench_with_input(BenchmarkId::new("points", n), &n, |b, &n| {
            b.iter(|| {
                let points: Vec<f32> = (0..n * 3).map(|i| (i as f32) * 0.01).collect();
                let queries = vec![0.0f32; 3];
                let k = 8;

                let qx = queries[0];
                let qy = queries[1];
                let qz = queries[2];

                let mut dists: Vec<(f32, usize)> = (0..n)
                    .map(|i| {
                        let dx = qx - points[i * 3];
                        let dy = qy - points[i * 3 + 1];
                        let dz = qz - points[i * 3 + 2];
                        (dx * dx + dy * dy + dz * dz, i)
                    })
                    .collect();

                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let mut results = vec![0.0f32; k];
                for i in 0..k.min(n) {
                    results[i] = dists[i].0;
                }
                results
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_element_add, bench_relu, bench_knn);
criterion_main!(benches);
