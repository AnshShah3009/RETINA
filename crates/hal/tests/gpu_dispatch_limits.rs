//! Regression tests for 1-D GPU dispatches that exceed WebGPU's per-dimension
//! workgroup limit.
//!
//! WebGPU guarantees only `maxComputeWorkgroupsPerDimension = 65535`. Kernels
//! that dispatched `(n.div_ceil(wg), 1, 1)` over a data-sized `n` panicked with
//!
//! ```text
//! wgpu error: Validation Error ... dispatch group size dimension
//! ([129600, 1, 1]) must be less or equal to 65535
//! ```
//!
//! as soon as `n` crossed `65535 * wg` - a single 3840x2160 frame for a
//! 64-wide kernel. Each test here runs a kernel on an input sized to need more
//! than 65535 workgroups along x and compares against the CPU backend (or a
//! plain Rust reference where the CPU backend has no equivalent). They skip when
//! no adapter is available.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ColorConversion, ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::{dispatch_1d, MAX_WORKGROUPS_PER_DIM};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;

const W_4K: usize = 3840;
const H_4K: usize = 2160;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU dispatch-limit test (no adapter): {}", e);
            None
        }
    }
}

/// Deterministic, non-trivial f32 pattern in [0, 1).
fn pattern(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            (x >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    let mut bad = 0usize;
    let mut first = None;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        if (g - c).abs() > tol {
            bad += 1;
            first.get_or_insert((i, *g, *c));
        }
    }
    assert_eq!(
        bad,
        0,
        "{label}: {bad} of {} elements differ from CPU, first at {:?} (idx, gpu, cpu)",
        cpu.len(),
        first
    );
}

#[test]
fn dispatch_1d_never_exceeds_the_per_dimension_limit() {
    // Small counts are unchanged: one row of workgroups.
    assert_eq!(dispatch_1d(0, 64), (0, 1, 1));
    assert_eq!(dispatch_1d(1, 64), (1, 1, 1));
    assert_eq!(dispatch_1d(64 * 65535, 64), (65535, 1, 1));
    // One past the limit spills into y.
    assert_eq!(dispatch_1d(64 * 65535 + 1, 64), (65535, 2, 1));
    // 3840x2160 at 64 per workgroup = 129600 groups.
    assert_eq!(dispatch_1d((W_4K * H_4K) as u32, 64), (65535, 2, 1));
    for &(n, wg) in &[
        (65535 * 65535, 1u32),
        (u32::MAX, 64),
        (1 << 30, 256),
        (12345, 7),
    ] {
        let (x, y, z) = dispatch_1d(n, wg);
        assert!(x <= MAX_WORKGROUPS_PER_DIM && y <= MAX_WORKGROUPS_PER_DIM && z == 1);
        // Covers every invocation.
        assert!(x as u64 * y as u64 * wg as u64 >= n as u64, "{n}/{wg}");
    }
}

#[test]
fn threshold_f32_on_a_4k_frame_matches_cpu() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    let shape = TensorShape::new(1, H_4K, W_4K);
    let input: CpuTensor<f32> = Tensor::from_vec(pattern(shape.len(), 1), shape).unwrap();
    let expected = cpu
        .threshold(&input, 0.5f32, 1.0f32, ThresholdType::Binary)
        .unwrap();

    let got = gpu
        .threshold(
            &input.to_gpu_ctx(gpu).unwrap(),
            0.5f32,
            1.0f32,
            ThresholdType::Binary,
        )
        .unwrap()
        .to_cpu_ctx(gpu)
        .unwrap();

    assert_close(
        "threshold",
        got.storage.as_slice().unwrap(),
        expected.storage.as_slice().unwrap(),
        0.0,
    );
}

#[test]
fn cvt_color_rgb_to_gray_on_a_4k_frame_matches_cpu() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    let shape = TensorShape::new(3, H_4K, W_4K);
    let input: CpuTensor<f32> = Tensor::from_vec(pattern(shape.len(), 2), shape).unwrap();
    let expected = cpu.cvt_color(&input, ColorConversion::RgbToGray).unwrap();

    let got = gpu
        .cvt_color(&input.to_gpu_ctx(gpu).unwrap(), ColorConversion::RgbToGray)
        .unwrap()
        .to_cpu_ctx(gpu)
        .unwrap();

    assert_eq!(got.shape, expected.shape);
    assert_close(
        "cvt_color",
        got.storage.as_slice().unwrap(),
        expected.storage.as_slice().unwrap(),
        1e-5,
    );
}

#[test]
fn subtract_on_a_3_channel_4k_frame_matches_cpu() {
    // 3 * 3840 * 2160 = 24.9M elements at 256 per workgroup = 97200 groups.
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    let shape = TensorShape::new(3, H_4K, W_4K);
    let a: CpuTensor<f32> = Tensor::from_vec(pattern(shape.len(), 3), shape).unwrap();
    let b: CpuTensor<f32> = Tensor::from_vec(pattern(shape.len(), 4), shape).unwrap();
    let expected = cpu.subtract(&a, &b).unwrap();

    let got = gpu
        .subtract(&a.to_gpu_ctx(gpu).unwrap(), &b.to_gpu_ctx(gpu).unwrap())
        .unwrap()
        .to_cpu_ctx(gpu)
        .unwrap();

    assert_close(
        "subtract",
        got.storage.as_slice().unwrap(),
        expected.storage.as_slice().unwrap(),
        0.0,
    );
}

#[test]
fn pointcloud_transform_on_a_buffer_sized_cloud_matches_cpu() {
    // 2^24 points x 16 bytes = 256 MiB, the device's buffer cap. At 256 points
    // per workgroup that is 65536 groups - one past the limit.
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    let n = 1usize << 24;
    let mut data = pattern(n * 4, 5);
    for p in data.chunks_mut(4) {
        p[3] = 1.0;
    }
    let shape = TensorShape::new(1, n, 4);
    let input: CpuTensor<f32> = Tensor::from_vec(data, shape).unwrap();

    let transform = [
        [0.0f32, -1.0, 0.0, 0.25],
        [1.0, 0.0, 0.0, -0.5],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let expected = cpu.pointcloud_transform(&input, &transform).unwrap();

    let got = gpu
        .pointcloud_transform(&input.to_gpu_ctx(gpu).unwrap(), &transform)
        .unwrap()
        .to_cpu_ctx(gpu)
        .unwrap();

    assert_close(
        "pointcloud_transform",
        got.storage.as_slice().unwrap(),
        expected.storage.as_slice().unwrap(),
        1e-5,
    );
}

#[test]
fn icp_correspondences_with_many_source_points_match_cpu() {
    // 4.2M source points at 64 per workgroup = 65625 groups. The target set is
    // tiny so the brute-force search stays cheap.
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    let n_src = 4_200_000usize;
    let mut src = pattern(n_src * 4, 6);
    for p in src.chunks_mut(4) {
        p[3] = 1.0;
    }
    let tgt: Vec<f32> = vec![
        0.1, 0.1, 0.1, 1.0, //
        0.9, 0.1, 0.5, 1.0, //
        0.5, 0.9, 0.9, 1.0, //
    ];
    let src: CpuTensor<f32> = Tensor::from_vec(src, TensorShape::new(1, n_src, 4)).unwrap();
    let tgt: CpuTensor<f32> = Tensor::from_vec(tgt, TensorShape::new(1, 3, 4)).unwrap();
    let max_dist = 0.6f32;

    let mut expected = cpu.icp_correspondences(&src, &tgt, max_dist).unwrap();
    let mut got = gpu
        .icp_correspondences(
            &src.to_gpu_ctx(gpu).unwrap(),
            &tgt.to_gpu_ctx(gpu).unwrap(),
            max_dist,
        )
        .unwrap();
    expected.sort_by_key(|c| c.0);
    got.sort_by_key(|c| c.0);

    // Points past the old limit (index >= 64 * 65535) must be present.
    assert!(
        got.iter().any(|c| c.0 >= 64 * 65535),
        "no correspondence beyond the old dispatch limit"
    );
    assert_eq!(got.len(), expected.len(), "correspondence count");
    for (g, c) in got.iter().zip(&expected) {
        assert_eq!((g.0, g.1), (c.0, c.1));
        assert!((g.2 - c.2).abs() < 1e-5, "{g:?} vs {c:?}");
    }
}

#[test]
fn u8_threshold_kernel_via_dispatch_covers_an_8k_frame() {
    // The packed u8 threshold shader handles four pixels per invocation, so its
    // invocation count is ceil(len / 4), not len. 7680x4320 = 33.2M pixels =
    // 8.3M invocations = 129600 workgroups of 64.
    let Some(gpu) = require_gpu() else { return };

    let (w, h) = (7680usize, 4320usize);
    let len = w * h;
    let data: Vec<u8> = (0..len)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    let thresh = 127u32;
    let max_value = 255u32;
    let expected: Vec<u8> = data
        .iter()
        .map(|&v| {
            if v as u32 > thresh {
                max_value as u8
            } else {
                0
            }
        })
        .collect();

    let shape = TensorShape::new(1, h, w);
    let input: CpuTensor<u8> = Tensor::from_vec(data, shape).unwrap();
    let output: CpuTensor<u8> = Tensor::from_vec(vec![0u8; len], shape).unwrap();
    let input = input.to_gpu_ctx(gpu).unwrap();
    let output = output.to_gpu_ctx(gpu).unwrap();

    // Params { thresh, max_value, typ (0 = Binary), len }
    let uniforms: Vec<u8> = [thresh, max_value, 0u32, len as u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let workgroups = dispatch_1d((len as u32).div_ceil(4), 64);
    gpu.dispatch("threshold", &[&input, &output], &uniforms, workgroups)
        .unwrap();

    let got = output.to_cpu_ctx(gpu).unwrap();
    let got = got.storage.as_slice().unwrap();
    assert_eq!(got.len(), len);
    let bad = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
    let first = got.iter().zip(&expected).position(|(g, e)| g != e);
    assert_eq!(bad, 0, "{bad} of {len} pixels wrong, first at {first:?}");
}
