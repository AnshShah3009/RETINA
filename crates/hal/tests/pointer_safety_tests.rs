//! Type-guarded pointer-cast safety.
//!
//! Several kernels reinterpret a generic `Tensor<T, _>` as f32 through
//! `transmute` / `from_raw_parts` / a `[[T; 4]; 4]` -> `[[f32; 4]; 4]` pointer
//! cast. Those casts are sound only because a `TypeId` guard rejects every
//! `T` other than `f32` first. These tests drive the real entry points rather
//! than transmuting a value to itself: the f32 path must produce the correct
//! result, and a non-f32 instantiation must be rejected with an error instead
//! of reaching the cast.
//!
//! The layout assertions at the bottom are the prerequisite the casts rely on
//! (a `[[f32; 4]; 4]` must be 64 contiguous bytes), not a test of the casts.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::Error;
use nalgebra::Matrix4;
use pollster::block_on;

/// Initialize a GPU context if one is available. Returns None when there is no
/// adapter, so the GPU cases skip cleanly on machines without a GPU.
fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU case (no adapter): {}", e);
            None
        }
    }
}

#[test]
fn pointcloud_transform_f32_applies_the_transform() {
    let Some(gpu) = require_gpu() else { return };

    let points: Vec<f32> = vec![
        1.0, 2.0, 3.0, 1.0, //
        4.0, 5.0, 6.0, 1.0,
    ];
    let cpu: CpuTensor<f32> = Tensor::from_vec(points, TensorShape::new(1, 2, 4)).unwrap();
    let gpu_points = cpu.to_gpu_ctx(gpu).unwrap();

    let mut transform = [[0.0f32; 4]; 4];
    transform[0][0] = 1.0;
    transform[0][3] = 10.0;
    transform[1][1] = 1.0;
    transform[1][3] = -5.0;
    transform[2][2] = 1.0;
    transform[2][3] = 2.0;
    transform[3][3] = 1.0;

    let out = gpu
        .pointcloud_transform(&gpu_points, &transform)
        .expect("f32 pointcloud_transform must succeed");
    let out = out.to_cpu_ctx(gpu).unwrap();
    let got = out.storage.as_slice().unwrap();

    // This is the path that reinterprets the caller's [[T; 4]; 4] as [[f32; 4]; 4].
    assert!((got[0] - 11.0).abs() < 1e-5, "x: {}", got[0]);
    assert!((got[1] - -3.0).abs() < 1e-5, "y: {}", got[1]);
    assert!((got[2] - 5.0).abs() < 1e-5, "z: {}", got[2]);
    assert!((got[4] - 14.0).abs() < 1e-5, "x2: {}", got[4]);
    assert!((got[5] - 0.0).abs() < 1e-5, "y2: {}", got[5]);
    assert!((got[6] - 8.0).abs() < 1e-5, "z2: {}", got[6]);
}

#[test]
fn pointcloud_transform_f64_is_rejected_before_the_cast() {
    let Some(gpu) = require_gpu() else { return };

    let points: Vec<f64> = vec![1.0, 2.0, 3.0, 1.0];
    let cpu: CpuTensor<f64> = Tensor::from_vec(points, TensorShape::new(1, 1, 4)).unwrap();
    let gpu_points = cpu.to_gpu_ctx(gpu).unwrap();

    let mut transform = [[0.0f64; 4]; 4];
    transform[0][0] = 1.0;
    transform[3][3] = 1.0;

    // The guard must stop this: reinterpreting [[f64; 4]; 4] as [[f32; 4]; 4]
    // would read the wrong bytes entirely.
    match gpu.pointcloud_transform(&gpu_points, &transform) {
        Err(Error::NotSupported(_)) => {}
        Ok(_) => panic!("f64 pointcloud_transform must not silently reinterpret the transform"),
        Err(e) => panic!("expected NotSupported for f64, got {:?}", e),
    }
}

#[test]
fn icp_accumulate_f32_builds_a_system_and_f64_is_rejected() {
    let cpu = CpuBackend::new().expect("CPU backend");

    let mk = |data: Vec<f64>| -> CpuTensor<f64> {
        Tensor::from_vec(data, TensorShape::new(1, 2, 4)).unwrap()
    };
    let source = mk(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let target = mk(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let normals = mk(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let transform = Matrix4::<f64>::identity();
    let correspondences = [(0u32, 1u32)];

    // f64 is the rejected instantiation: the CPU path reinterprets the slices
    // as f32 through from_raw_parts, so it must refuse anything else.
    match cpu.icp_accumulate(&source, &target, &normals, &correspondences, &transform) {
        Err(Error::NotSupported(_)) => {}
        Ok(_) => panic!("f64 icp_accumulate must not reinterpret f64 slices as f32"),
        Err(e) => panic!("expected NotSupported for f64, got {:?}", e),
    }

    let mk32 = |data: Vec<f32>| -> CpuTensor<f32> {
        Tensor::from_vec(data, TensorShape::new(1, 2, 4)).unwrap()
    };
    let source32 = mk32(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let target32 = mk32(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let normals32 = mk32(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let (ata, atb) = cpu
        .icp_accumulate(
            &source32,
            &target32,
            &normals32,
            &correspondences,
            &Matrix4::<f32>::identity(),
        )
        .expect("f32 icp_accumulate must succeed");
    assert_eq!(atb.len(), 6, "the 6-DoF normal equations must be filled");
    assert!(
        ata.iter().any(|v| *v != 0.0),
        "the f32 path must accumulate a non-zero system"
    );
}

#[test]
fn optical_flow_lk_rejects_non_f32_pyramids() {
    let Some(gpu) = require_gpu() else { return };

    let level42: Vec<f64> = (0..42).map(|i| i as f64).collect();
    let level: CpuTensor<f64> = Tensor::from_vec(level42, TensorShape::new(1, 6, 7)).unwrap();
    let gpu_level = level.to_gpu_ctx(gpu).unwrap();
    let points = [[3.0f64, 3.0f64]];

    match gpu.optical_flow_lk(&[gpu_level.clone()], &[gpu_level], &points, 5, 3) {
        Err(Error::NotSupported(_)) => {}
        Ok(_) => panic!("f64 optical_flow_lk must not reinterpret f64 planes as f32"),
        Err(e) => panic!("expected NotSupported for f64, got {:?}", e),
    }
}

/// The casts assume a transform is 64 contiguous, 4-byte-aligned bytes and
/// that `[f32; 2]` is 8 bytes; if these ever change the reinterpretation sites
/// must change with them.
#[test]
fn layout_prerequisites_for_the_casts() {
    assert_eq!(std::mem::size_of::<f32>(), 4);
    assert_eq!(std::mem::align_of::<f32>(), 4);
    assert_eq!(std::mem::size_of::<[f32; 2]>(), 8);
    assert_eq!(std::mem::size_of::<[[f32; 4]; 4]>(), 64);
    // T == f32 (enforced by the TypeId guard) makes these identity equalities.
    assert_eq!(
        std::any::TypeId::of::<f32>(),
        std::any::TypeId::of::<f32>(),
        "the guard compares TypeId, so equal types must compare equal"
    );
    assert_ne!(std::any::TypeId::of::<f32>(), std::any::TypeId::of::<f64>());
}
