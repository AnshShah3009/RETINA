/// Normal estimation benchmarks — all paths, multiple cloud sizes.
///
/// Paths measured (labelled to match Open3D equivalents):
///  1. rtree_pca       – RTree kNN  + analytic eigensolver  (≈ Open3D CPU/FLANN path)
///  2. voxel_cpu       – voxel-hash + analytic eigensolver  (our fast CPU path)
///  3. hybrid          – voxel-hash kNN (CPU) + batch PCA (GPU)
///  4. morton_gpu      – Morton sort (CPU) + kNN window + PCA (GPU full path)
///  5. depth_image     – O(n) cross-product from structured depth (RGBD path)
///
/// Cloud sizes: 1k / 10k / 100k points on a noisy sphere surface.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cv_3d::gpu::point_cloud as gpu_pc;
use cv_scientific::point_cloud as sci_pc;
use nalgebra::Point3;
use rand::Rng;
use std::time::Duration;

// ── point cloud generators ────────────────────────────────────────────────────

/// n points uniformly sampled from the surface of the unit sphere + small noise.
fn sphere_cloud(n: usize) -> Vec<Point3<f32>> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let theta = rng.random::<f32>() * std::f32::consts::TAU;
            let phi = (rng.random::<f32>() * 2.0 - 1.0).acos();
            let r = 1.0 + rng.random::<f32>() * 0.01;
            Point3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            )
        })
        .collect()
}

/// Synthetic depth image: H×W projection of the unit sphere onto the XY plane.
fn sphere_depth(width: usize, height: usize) -> Vec<f32> {
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let fx = width as f32; // focal length = image width (≈ 45° FoV)
    let fy = height as f32;

    (0..height * width)
        .map(|i| {
            let px = (i % width) as f32;
            let py = (i / width) as f32;
            let dx = (px - cx) / fx;
            let dy = (py - cy) / fy;
            let r2 = dx * dx + dy * dy;
            if r2 < 1.0 {
                (1.0 - r2).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

// ── individual path wrappers ──────────────────────────────────────────────────

fn bench_rtree_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/rtree_pca (≈Open3D CPU)");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        let mut pc = cv_core::PointCloud::new(pts);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                sci_pc::estimate_normals(black_box(&mut pc), 15);
            });
        });
    }
    group.finish();
}

fn bench_voxel_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/voxel_hash_cpu");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals_cpu(black_box(&pts), 15, 0.0));
        });
    }
    group.finish();
}

fn bench_kdtree_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/kdtree_cpu");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals_cpu_kdtree(black_box(&pts), 15));
        });
    }
    group.finish();
}

fn bench_tiled_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/tiled_gpu");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals_gpu_tiled(black_box(&pts), 15));
        });
    }
    group.finish();
}

fn bench_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/hybrid_cpu_knn_gpu_pca");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(15);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals_hybrid(black_box(&pts), 15));
        });
    }
    group.finish();
}

fn bench_morton_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/morton_gpu");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(15);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals(black_box(&pts), 15));
        });
    }
    group.finish();
}

fn bench_depth_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/depth_image_O(n)");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    // Comparable pixel counts to the point cloud sizes above.
    for &(w, h) in &[(32usize, 32usize), (100, 100), (320, 320)] {
        let depth = sphere_depth(w, h);
        let label = format!("{}x{}", w, h);
        group.bench_with_input(BenchmarkId::new("pixels", &label), &label, |b, _| {
            b.iter(|| {
                sci_pc::compute_normals_from_depth(
                    black_box(&depth),
                    w,
                    h,
                    w as f32,
                    h as f32,       // fx=fy=width
                    w as f32 / 2.0, // cx
                    h as f32 / 2.0, // cy
                )
            });
        });
    }
    group.finish();
}

fn bench_approx_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("normals/approx_integral (fast)");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = sphere_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gpu_pc::compute_normals_approx_integral(black_box(&pts), 0.0));
        });
    }
    group.finish();
}

// ── criterion entry point ─────────────────────────────────────────────────────

criterion_group!(
    normals_benches,
    bench_rtree_pca,
    bench_voxel_cpu,
    bench_kdtree_cpu,
    bench_tiled_gpu,
    bench_hybrid,
    bench_morton_gpu,
    bench_depth_image,
    bench_approx_integral,
);
criterion_main!(normals_benches);
