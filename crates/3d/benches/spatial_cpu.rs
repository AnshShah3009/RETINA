//! CPU spatial-index benchmarks: `KDTree` build and nearest / k-NN / radius
//! queries, each a serial batch of queries so the numbers are per-tree cost.
//!
//! The cloud is a deterministic noisy unit sphere; queries are cloud points
//! displaced off the surface, as ICP correspondence and normal-estimation
//! queries are.
//!
//! Run: `cargo bench -p cv-3d --bench spatial_cpu`

use criterion::{Criterion, criterion_group, criterion_main};
use cv_3d::spatial::KDTree;
use nalgebra::Point3;
use std::hint::black_box;
use std::time::Duration;

fn rng(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state % 20001) as f32 / 10000.0 - 1.0
}

fn sphere(n: usize, seed: u64) -> Vec<Point3<f32>> {
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut state = seed;
    (0..n)
        .map(|i| {
            let r = 1.0 + rng(&mut state) * 0.01;
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden;
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();
            Point3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            )
        })
        .collect()
}

fn bench_kdtree(c: &mut Criterion) {
    let points = sphere(100_000, 0x2545_f491_4f6c_dd1d);
    let mut state = 0x9e37_79b9u64;
    let queries: Vec<Point3<f32>> = sphere(10_000, 7)
        .into_iter()
        .map(|p| {
            p + nalgebra::Vector3::new(rng(&mut state), rng(&mut state), rng(&mut state)) * 0.02
        })
        .collect();
    let items: Vec<(Point3<f32>, usize)> =
        points.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let tree = KDTree::build(&mut items.clone());

    let mut g = c.benchmark_group("kdtree");
    g.sample_size(20).measurement_time(Duration::from_secs(5));
    g.bench_function("build_100k", |b| {
        b.iter(|| KDTree::build(black_box(&mut items.clone())))
    });
    g.bench_function("nearest_100k_x10k", |b| {
        b.iter(|| {
            queries
                .iter()
                .map(|q| tree.nearest_neighbor(black_box(q)).unwrap().1)
                .sum::<usize>()
        })
    });
    g.bench_function("knn10_100k_x10k", |b| {
        b.iter(|| {
            queries
                .iter()
                .map(|q| tree.k_nearest_neighbors(black_box(q), 10).len())
                .sum::<usize>()
        })
    });
    g.bench_function("radius0.03_100k_x10k", |b| {
        b.iter(|| {
            queries
                .iter()
                .map(|q| tree.search_radius(black_box(q), 0.03).len())
                .sum::<usize>()
        })
    });
    g.finish();
}

criterion_group!(benches, bench_kdtree);
criterion_main!(benches);
