//! `tsdf_extract_mesh` must put the mesh where `tsdf_integrate` put the
//! surface - and nowhere else.
//!
//! Two defects, shared by the CPU backend and the GPU marching-cubes shaders:
//!
//! 1. **Half-voxel shift.** Integration evaluates each voxel at its centre,
//!    `(i + 0.5) * voxel_size` (`CpuBackend::tsdf_integrate`,
//!    `tsdf_integrate.wgsl`), but the mesh placed cell corners at
//!    `i * voxel_size`, so every vertex came out half a voxel toward the
//!    origin.
//! 2. **Unobserved corners meshed.** A voxel no frame reached has weight zero
//!    and a placeholder value of 0. Where an observed negative voxel (just
//!    behind the surface) met an unobserved one (further behind, never seen)
//!    that was a "sign change", and a phantom back surface was emitted there.
//!
//! Measured before the fix on the fixture below (wall at z = 0.4, voxel 5 cm,
//! truncation 10 cm): 1350 vertices at z = 0.375 (the true wall, shifted half a
//! voxel) and 1350 at z = 0.471 (the phantom), none at 0.4.

use cv_core::storage::CpuStorage;
use cv_core::{Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::marching_cubes::Vertex;
use cv_hal::tensor_ext::TensorToGpu;

const VOX: usize = 16;
const VOXEL: f32 = 0.05;
const TRUNC: f32 = 0.1;
const WALL_Z: f32 = 0.4;
const IMG: usize = 64;

/// World-to-camera: camera at (0.4, 0.4, 0) looking down +z, so the wall at
/// camera depth 0.4 is the world plane z = 0.4 and the image covers the middle
/// of the volume.
const W2C: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, -0.4],
    [0.0, 1.0, 0.0, -0.4],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const INTRINSICS: [f32; 4] = [40.0, 40.0, 31.5, 31.5];

fn integrated_volume() -> Tensor<f32, CpuStorage<f32>> {
    let cpu = CpuBackend::new().expect("cpu backend");
    let depth = Tensor::from_vec(vec![WALL_Z; IMG * IMG], TensorShape::new(1, IMG, IMG)).unwrap();
    // Interleaved (tsdf, weight); everything starts unobserved.
    let mut vol = Tensor::from_vec(
        vec![0.0f32; VOX * VOX * VOX * 2],
        TensorShape::new(VOX * 2, VOX, VOX),
    )
    .unwrap();
    cpu.tsdf_integrate(&depth, &W2C, &INTRINSICS, &mut vol, VOXEL, TRUNC)
        .expect("integrate");
    vol
}

fn check_on_wall(label: &str, verts: &[Vertex]) {
    assert!(
        verts.len() > 300,
        "{label}: only {} vertices - the wall was not meshed",
        verts.len()
    );
    let off: Vec<f32> = verts
        .iter()
        .map(|v| v.pos()[2])
        .filter(|z| (z - WALL_Z).abs() > 0.5 * VOXEL * 0.5)
        .collect();
    let behind = verts
        .iter()
        .filter(|v| v.pos()[2] > WALL_Z + 0.25 * VOXEL)
        .count();
    let mut zs: Vec<f32> = off.clone();
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    assert!(
        off.is_empty(),
        "{label}: {} of {} vertices are more than a quarter voxel off the wall \
         z = {WALL_Z} ({behind} of them behind it); distinct z: {:?}",
        off.len(),
        verts.len(),
        &zs[..zs.len().min(6)]
    );
    // The wall spans the image footprint: x and y in ~[0.08, 0.72].
    let (lo, hi) = verts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| {
        (lo.min(v.pos()[0]), hi.max(v.pos()[0]))
    });
    assert!(
        hi - lo > 0.5,
        "{label}: the mesh only spans x in [{lo}, {hi}]"
    );
}

#[test]
fn cpu_mesh_lies_on_the_integrated_wall() {
    let cpu = CpuBackend::new().expect("cpu backend");
    let vol = integrated_volume();
    let verts = cpu.tsdf_extract_mesh(&vol, VOXEL, 0.0, 100_000).unwrap();
    check_on_wall("cpu", &verts);
}

/// A cell with any zero-weight corner is not meshed, even when its observed
/// corners disagree in sign with the placeholder.
#[test]
fn cpu_skips_cells_with_an_unobserved_corner() {
    let cpu = CpuBackend::new().expect("cpu backend");
    // 2x2x2: one observed negative corner, the rest unobserved placeholders.
    let mut data = vec![0.0f32; 2 * 2 * 2 * 2];
    data[0] = -0.5;
    data[1] = 1.0;
    let vol: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data, TensorShape::new(4, 2, 2)).unwrap();
    let verts = cpu.tsdf_extract_mesh(&vol, 1.0f32, 0.0f32, 1024).unwrap();
    assert!(
        verts.is_empty(),
        "{} vertices emitted from a cell with seven unobserved corners",
        verts.len()
    );
}

fn get_gpu_context() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    pollster::block_on(GpuContext::init_global()).ok()
}

#[test]
fn gpu_mesh_lies_on_the_integrated_wall() {
    let Some(gpu) = get_gpu_context() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let vol = integrated_volume().to_gpu_ctx(gpu).expect("upload volume");
    let verts = gpu.tsdf_extract_mesh(&vol, VOXEL, 0.0, 100_000).unwrap();
    check_on_wall("gpu", &verts);

    // The CPU and GPU meshes are the same surface: compare as sorted vertex
    // multisets, since the GPU emits in dispatch order.
    let cpu = CpuBackend::new().expect("cpu backend");
    let cpu_verts = cpu
        .tsdf_extract_mesh(&integrated_volume(), VOXEL, 0.0, 100_000)
        .unwrap();
    assert_eq!(
        verts.len(),
        cpu_verts.len(),
        "cpu and gpu emit different vertex counts"
    );
    let key = |v: &Vertex| {
        let p = v.pos();
        [
            (p[0] * 1e4).round() as i64,
            (p[1] * 1e4).round() as i64,
            (p[2] * 1e4).round() as i64,
        ]
    };
    let mut a: Vec<_> = verts.iter().map(key).collect();
    let mut b: Vec<_> = cpu_verts.iter().map(key).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b, "cpu and gpu vertex positions differ");
}

/// The two-pass kernel (`marching_cubes_count.wgsl` + `marching_cubes_emit.wgsl`)
/// carried the same two defects. It is not what `GpuContext::tsdf_extract_mesh`
/// dispatches, but it is public, so it is held to the same surface.
#[test]
fn gpu_two_pass_mesh_lies_on_the_integrated_wall() {
    let Some(gpu) = get_gpu_context() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let vol = integrated_volume().to_gpu_ctx(gpu).expect("upload volume");
    let verts =
        cv_hal::gpu_kernels::marching_cubes::extract_mesh_v2(gpu, &vol, VOXEL, 0.0).unwrap();
    check_on_wall("gpu two-pass", &verts);
}
