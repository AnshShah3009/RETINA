//! Benchmark: 3D pipeline operations (marching cubes, KDTree, ICP, Poisson, TSDF)
//!
//! Run:
//!   cargo bench --bench pipeline_3d
//!   cargo bench --bench pipeline_3d -- marching_cubes
//!   cargo bench --bench pipeline_3d -- kdtree

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cv_3d::mesh::reconstruction::poisson_reconstruction;
use cv_3d::spatial::KDTree;
use cv_3d::tsdf::CameraIntrinsics;
use cv_3d::TSDFVolume;
use cv_core::PointCloud;
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::WgpuGpuStorage;
use cv_registration::registration_icp_point_to_plane;
use nalgebra::{Matrix4, Point3, Vector3};
use std::marker::PhantomData;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fibonacci-sphere point cloud with outward normals.
fn sphere_cloud(n: usize, radius: f32) -> PointCloud {
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut pts = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for i in 0..n {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();
        let (sp, cp) = (phi.sin(), phi.cos());
        let (st, ct) = (theta.sin(), theta.cos());
        let p = Point3::new(radius * sp * ct, radius * sp * st, radius * cp);
        pts.push(p);
        nrm.push(Vector3::new(p.x, p.y, p.z).normalize());
    }
    PointCloud::new(pts).with_normals(nrm).unwrap()
}

/// Sphere SDF on a regular grid — TSDF+weight pairs, flat layout.
fn sphere_sdf(gs: usize, radius: f32) -> Vec<f32> {
    let vs = 2.2 * radius / gs as f32;
    let o = -1.1 * radius;
    let mut d = vec![0.0f32; gs * gs * gs * 2];
    for z in 0..gs {
        for y in 0..gs {
            for x in 0..gs {
                let px = o + (x as f32 + 0.5) * vs;
                let py = o + (y as f32 + 0.5) * vs;
                let pz = o + (z as f32 + 0.5) * vs;
                let dist = (px * px + py * py + pz * pz).sqrt() - radius;
                let idx = (z * gs * gs + y * gs + x) * 2;
                d[idx] = dist.clamp(-0.1, 0.1);
                d[idx + 1] = 1.0;
            }
        }
    }
    d
}

fn try_gpu() -> Option<GpuContext> {
    GpuContext::new().ok()
}

// ---------------------------------------------------------------------------
// 1. Marching Cubes (GPU)
// ---------------------------------------------------------------------------

fn bench_marching_cubes(c: &mut Criterion) {
    let gpu = try_gpu();
    let mut group = c.benchmark_group("marching_cubes");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(15);

    for &gs in &[64usize, 128] {
        let label = format!("{}^3", gs);
        let sdf = sphere_sdf(gs, 0.4);
        let vs = 2.2 * 0.4 / gs as f32;

        if let Some(ref ctx) = gpu {
            let storage =
                WgpuGpuStorage::from_slice_ctx(ctx, &sdf).expect("GPU upload");
            let shape = cv_core::TensorShape::new(gs * 2, gs, gs);
            let vol = cv_core::Tensor {
                storage,
                shape,
                dtype: cv_core::DataType::F32,
                _phantom: PhantomData,
            };

            // Warmup
            let _ = cv_hal::gpu_kernels::marching_cubes::extract_mesh(
                ctx, &vol, vs, 0.0, 500_000,
            );
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::marching_cubes::extract_mesh(
                        black_box(ctx),
                        black_box(&vol),
                        vs,
                        0.0,
                        500_000,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 2. KDTree queries
// ---------------------------------------------------------------------------

fn bench_kdtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_queries");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &n in &[10_000usize, 100_000] {
        let label = format!("n={}", n);
        let cloud = sphere_cloud(n, 1.0);
        let mut items: Vec<_> = cloud.points.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let tree = KDTree::build(&mut items);
        let q = Point3::new(0.5, 0.5, 0.5);

        group.bench_with_input(BenchmarkId::new("nearest", &label), &(), |b, _| {
            b.iter(|| black_box(tree.nearest_neighbor(black_box(&q))));
        });
        group.bench_with_input(BenchmarkId::new("knn_20", &label), &(), |b, _| {
            b.iter(|| black_box(tree.k_nearest_neighbors(black_box(&q), 20)));
        });
        group.bench_with_input(BenchmarkId::new("radius_0.1", &label), &(), |b, _| {
            b.iter(|| black_box(tree.search_radius(black_box(&q), 0.1)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 3. ICP Registration
// ---------------------------------------------------------------------------

fn bench_icp(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_registration");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &n in &[1_000usize, 10_000] {
        let label = format!("n={}", n);
        let target = sphere_cloud(n, 1.0);
        let offset = Vector3::new(0.05, 0.03, 0.02);
        let src_pts: Vec<_> = target.points.iter().map(|p| p + offset).collect();
        let source = PointCloud::new(src_pts)
            .with_normals(target.normals.clone().unwrap())
            .unwrap();
        let init = Matrix4::identity();

        group.bench_with_input(
            BenchmarkId::new("point_to_plane", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(registration_icp_point_to_plane(
                        black_box(&source),
                        black_box(&target),
                        0.5,
                        black_box(&init),
                        30,
                    ))
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Poisson Reconstruction
// ---------------------------------------------------------------------------

fn bench_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_reconstruction");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &n in &[10_000usize, 50_000] {
        let label = format!("n={}_d6", n);
        let cloud = sphere_cloud(n, 1.0);
        group.bench_with_input(BenchmarkId::new("depth_6", &label), &(), |b, _| {
            b.iter(|| black_box(poisson_reconstruction(black_box(&cloud), 6, 1.0)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. TSDF Integration
// ---------------------------------------------------------------------------

fn bench_tsdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsdf_integration");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for &(w, h) in &[(320usize, 240usize), (640, 480)] {
        let label = format!("{}x{}", w, h);
        let depth: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32 - 0.5;
                let y = (i / w) as f32 / h as f32 - 0.5;
                let r2 = x * x + y * y;
                if r2 < 0.2 {
                    1.0 - (0.2 - r2).sqrt() * 0.3
                } else {
                    0.0
                }
            })
            .collect();
        let intrinsics = CameraIntrinsics::new(
            w as f32, h as f32, w as f32 / 2.0, h as f32 / 2.0, w as u32, h as u32,
        );
        let extrinsics = Matrix4::identity();

        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                let mut vol = TSDFVolume::new(0.01, 0.03);
                vol.integrate(
                    black_box(&depth),
                    None,
                    black_box(&intrinsics),
                    black_box(&extrinsics),
                    w,
                    h,
                );
                black_box(&vol);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

criterion_group!(
    pipeline_3d,
    bench_marching_cubes,
    bench_kdtree,
    bench_icp,
    bench_poisson,
    bench_tsdf,
);
criterion_main!(pipeline_3d);
