//! `gpu_kernels::spatial::voxel_downsample` keeps one point per occupied
//! voxel. Its shader failed wgpu validation (`vec3<bool>` used as a bool),
//! so it never ran; it also read and wrote points at WGSL's 16-byte vec3
//! stride while the tensors are packed xyz (GAPS g-x6). Skips without GPU.

use cv_core::storage::{CpuStorage, Storage};
use cv_core::{Tensor, TensorShape};
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;
use std::collections::HashSet;

#[test]
fn one_point_per_occupied_voxel() {
    let gpu = match GpuContext::global() {
        Ok(g) => g,
        Err(_) => match block_on(GpuContext::init_global()) {
            Ok(g) => g,
            Err(_) => return,
        },
    };
    let mut pts = Vec::new();
    for i in 0..2000u32 {
        let f = |k: u32| ((i.wrapping_mul(2654435761).wrapping_add(k * 97)) % 1000) as f32 / 100.0;
        pts.extend([f(1), f(2), f(3)]);
    }
    let vs = 1.0f32;
    let voxel = |x: f32, y: f32, z: f32| {
        (
            (x / vs).floor() as i32,
            (y / vs).floor() as i32,
            (z / vs).floor() as i32,
        )
    };
    let want: HashSet<_> = pts.chunks(3).map(|p| voxel(p[0], p[1], p[2])).collect();

    let t = Tensor::<f32, CpuStorage<f32>>::from_vec(pts.clone(), TensorShape::new(1, 2000, 3))
        .unwrap()
        .to_gpu_ctx(gpu)
        .unwrap();
    let out = cv_hal::gpu_kernels::spatial::voxel_downsample(gpu, &t, vs).unwrap();
    let out = out.to_cpu_ctx(gpu).unwrap();
    let data = out.storage.as_slice().unwrap();
    assert_eq!(data.len() % 3, 0);
    let got: Vec<_> = data.chunks(3).map(|p| voxel(p[0], p[1], p[2])).collect();
    let got_set: HashSet<_> = got.iter().copied().collect();
    assert_eq!(got.len(), got_set.len(), "a voxel was emitted twice");
    assert_eq!(got_set, want, "occupied voxels differ");
    // Every output point is one of the inputs.
    let inputs: HashSet<[u32; 3]> = pts
        .chunks(3)
        .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
        .collect();
    assert!(data.chunks(3).all(|p| inputs.contains(&[
        p[0].to_bits(),
        p[1].to_bits(),
        p[2].to_bits()
    ])));
}
