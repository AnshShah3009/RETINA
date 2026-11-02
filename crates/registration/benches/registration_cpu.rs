//! CPU registration benchmarks: FPFH features, point-to-plane ICP and
//! RANSAC on FPFH correspondences.
//!
//! Inputs are deterministic: a Fibonacci sphere with a little radial noise, so
//! normals and FPFH neighbourhoods are those of a real curved surface.
//!
//! Run: `cargo bench -p cv-registration --bench registration_cpu`

use criterion::{Criterion, criterion_group, criterion_main};
use cv_core::PointCloud;
use cv_registration::{
    compute_fpfh_features, registration_icp_point_to_plane,
    registration_ransac_based_on_feature_matching,
};
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
use std::hint::black_box;
use std::time::Duration;

/// `n` points on a unit sphere with deterministic radial noise of up to 1%,
/// and their outward normals.
fn sphere(n: usize) -> PointCloud {
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut pts = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for i in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = 1.0 + ((state % 2001) as f32 / 1000.0 - 1.0) * 0.01;
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();
        let p = Point3::new(
            r * phi.sin() * theta.cos(),
            r * phi.sin() * theta.sin(),
            r * phi.cos(),
        );
        pts.push(p);
        nrm.push(Vector3::new(p.x, p.y, p.z).normalize());
    }
    PointCloud::new(pts).with_normals(nrm).unwrap()
}

fn moved(cloud: &PointCloud, pose: &Matrix4<f32>) -> PointCloud {
    let pts = cloud
        .points
        .iter()
        .map(|p| pose.transform_point(p))
        .collect();
    let nrm = cloud
        .normals
        .as_ref()
        .unwrap()
        .iter()
        .map(|n| pose.transform_vector(n))
        .collect();
    PointCloud::new(pts).with_normals(nrm).unwrap()
}

fn small_pose() -> Matrix4<f32> {
    let r = Rotation3::from_euler_angles(0.02, -0.03, 0.025);
    let mut m = r.to_homogeneous();
    m[(0, 3)] = 0.03;
    m[(1, 3)] = -0.02;
    m[(2, 3)] = 0.01;
    m
}

fn bench_fpfh(c: &mut Criterion) {
    let cloud = sphere(10_000);
    let bare = PointCloud::new(cloud.points.clone());
    let mut g = c.benchmark_group("registration");
    g.sample_size(10).measurement_time(Duration::from_secs(6));
    g.bench_function("fpfh_10k_r0.1_given_normals", |b| {
        b.iter(|| compute_fpfh_features(black_box(&cloud), 0.1).unwrap())
    });
    g.bench_function("fpfh_10k_r0.1_estimated_normals", |b| {
        b.iter(|| compute_fpfh_features(black_box(&bare), 0.1).unwrap())
    });
    g.finish();
}

fn bench_icp(c: &mut Criterion) {
    let target = sphere(20_000);
    let source = moved(&target, &small_pose());
    let mut g = c.benchmark_group("registration");
    g.sample_size(10).measurement_time(Duration::from_secs(6));
    g.bench_function("icp_point_to_plane_20k_30it", |b| {
        b.iter(|| {
            registration_icp_point_to_plane(
                black_box(&source),
                black_box(&target),
                0.2,
                &Matrix4::identity(),
                30,
            )
        })
    });
    g.finish();
}

fn bench_ransac(c: &mut Criterion) {
    let target = sphere(3_000);
    let source = moved(&target, &small_pose());
    let tf = compute_fpfh_features(&target, 0.2).unwrap();
    let sf = compute_fpfh_features(&source, 0.2).unwrap();
    let mut g = c.benchmark_group("registration");
    g.sample_size(10).measurement_time(Duration::from_secs(6));
    g.bench_function("ransac_fpfh_3k_1000it", |b| {
        b.iter(|| {
            registration_ransac_based_on_feature_matching(
                black_box(&source),
                black_box(&target),
                &sf,
                &tf,
                0.05,
                3,
                1000,
            )
        })
    });
    g.finish();
}

criterion_group!(benches, bench_fpfh, bench_icp, bench_ransac);
criterion_main!(benches);
