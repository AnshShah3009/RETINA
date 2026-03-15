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
            let storage = WgpuGpuStorage::from_slice_ctx(ctx, &sdf).expect("GPU upload");
            let shape = cv_core::TensorShape::new(gs * 2, gs, gs);
            let vol = cv_core::Tensor {
                storage,
                shape,
                dtype: cv_core::DataType::F32,
                _phantom: PhantomData,
            };

            // Warmup
            let _ = cv_hal::gpu_kernels::marching_cubes::extract_mesh(ctx, &vol, vs, 0.0, 500_000);
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
        let mut items: Vec<_> = cloud
            .points
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
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

        group.bench_with_input(BenchmarkId::new("point_to_plane", &label), &(), |b, _| {
            b.iter(|| {
                black_box(registration_icp_point_to_plane(
                    black_box(&source),
                    black_box(&target),
                    0.5,
                    black_box(&init),
                    30,
                ))
            });
        });
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
            w as f32,
            h as f32,
            w as f32 / 2.0,
            h as f32 / 2.0,
            w as u32,
            h as u32,
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
// 6. GPU ICP (new implementation in crates/3d/src/gpu)
// ---------------------------------------------------------------------------

fn bench_gpu_icp(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_icp");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &n in &[1_000usize, 10_000] {
        let label = format!("n={}", n);
        let target = sphere_cloud(n, 1.0);
        let offset = Vector3::new(0.05, 0.03, 0.02);
        let src_pts: Vec<_> = target.points.iter().map(|p| p + offset).collect();
        let tgt_normals = target.normals.clone().unwrap();

        group.bench_with_input(BenchmarkId::new("point_to_plane", &label), &(), |b, _| {
            b.iter(|| {
                black_box(cv_3d::gpu::registration::icp_point_to_plane(
                    black_box(&src_pts),
                    black_box(&target.points),
                    black_box(&tgt_normals),
                    0.5,
                    30,
                ))
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Ray casting (brute-force — O(rays × triangles))
// ---------------------------------------------------------------------------

fn bench_raycasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("raycasting");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    // Build a simple mesh from sphere cloud via Poisson
    let cloud = sphere_cloud(5_000, 1.0);
    let mesh = poisson_reconstruction(&cloud, 5, 1.0);

    if let Some(ref mesh) = mesh {
        let verts = &mesh.vertices;
        let faces: Vec<[usize; 3]> = mesh.faces.clone();
        let n_faces = faces.len();

        for &n_rays in &[100usize, 1000] {
            let label = format!("{}rays_{}tris", n_rays, n_faces);
            let origins: Vec<_> = (0..n_rays)
                .map(|i| {
                    let t = i as f32 / n_rays as f32 * std::f32::consts::PI * 2.0;
                    Point3::new(3.0 * t.cos(), 3.0 * t.sin(), 0.0)
                })
                .collect();
            let dirs: Vec<_> = origins
                .iter()
                .map(|o| (Point3::origin() - o).normalize())
                .collect();

            group.bench_with_input(BenchmarkId::new("brute_force", &label), &(), |b, _| {
                b.iter(|| {
                    black_box(cv_3d::gpu::raycasting::cast_rays(
                        black_box(&origins),
                        black_box(&dirs),
                        black_box(verts),
                        black_box(&faces),
                    ))
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 8. Vertex normals + Laplacian smoothing
// ---------------------------------------------------------------------------

fn bench_mesh_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_ops");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(15);

    let cloud = sphere_cloud(10_000, 1.0);
    let mesh = poisson_reconstruction(&cloud, 6, 1.0);

    if let Some(ref mesh) = mesh {
        let n_v = mesh.vertices.len();
        let n_f = mesh.faces.len();
        let label = format!("{}v_{}f", n_v, n_f);

        group.bench_with_input(BenchmarkId::new("vertex_normals", &label), &(), |b, _| {
            b.iter(|| {
                black_box(cv_3d::gpu::mesh::compute_vertex_normals(
                    black_box(&mesh.vertices),
                    black_box(&mesh.faces),
                ))
            });
        });

        group.bench_with_input(
            BenchmarkId::new("laplacian_smooth_5iter", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut verts = mesh.vertices.clone();
                    black_box(cv_3d::gpu::mesh::laplacian_smooth(
                        black_box(&mut verts),
                        black_box(&mesh.faces),
                        5,
                        0.5,
                    ))
                });
            },
        );
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
    bench_gpu_icp,
    bench_raycasting,
    bench_mesh_ops,
);
criterion_main!(pipeline_3d);
