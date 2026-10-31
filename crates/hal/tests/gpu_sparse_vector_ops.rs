//! `gpu_kernels::sparse::dot` and `vec_scale` against plain Rust. Their
//! shader entry points use bind groups 2 and 3, but the host bound them at
//! index 0 (GAPS g-x7). Skips without an adapter.

use cv_core::storage::{CpuStorage, Storage};
use cv_core::{Tensor, TensorShape};
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    block_on(GpuContext::init_global()).ok()
}

#[test]
fn dot_and_vec_scale_match_cpu() {
    let Some(gpu) = require_gpu() else { return };
    let n = 10_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
    let b: Vec<f32> = (0..n)
        .map(|i| ((i * 53) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let expected: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let shape = TensorShape::new(1, 1, n);
    let ag = Tensor::<f32, CpuStorage<f32>>::from_vec(a.clone(), shape)
        .unwrap()
        .to_gpu_ctx(gpu)
        .unwrap();
    let bg = Tensor::<f32, CpuStorage<f32>>::from_vec(b, shape)
        .unwrap()
        .to_gpu_ctx(gpu)
        .unwrap();
    let got = cv_hal::gpu_kernels::sparse::dot(gpu, &ag, &bg).unwrap();
    assert!(
        (got as f64 - expected).abs() < 1e-2 * expected.abs().max(1.0),
        "dot {got} vs {expected}"
    );

    let mut vg = ag;
    cv_hal::gpu_kernels::sparse::vec_scale(gpu, 3.0, &mut vg).unwrap();
    let scaled = vg.to_cpu_ctx(gpu).unwrap();
    for (s, x) in scaled.storage.as_slice().unwrap().iter().zip(&a) {
        assert!((s - 3.0 * x).abs() < 1e-5, "vec_scale {s} vs {}", 3.0 * x);
    }
}
