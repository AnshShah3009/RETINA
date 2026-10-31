//! `ComputeContext::canny` / `hough_lines` must agree between the GPU and CPU
//! backends: same edge mask bit for bit, same lines in the same order.
//! Skips when no adapter is available.

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU canny/hough test (no adapter): {}", e);
            None
        }
    }
}

fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..w * h)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

fn scene(w: usize, h: usize) -> Vec<u8> {
    let n = lcg_image(w, h, 11);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let mut v = 0.3 * x + 0.2 * y;
            if (x - 90.0).powi(2) + (y - 70.0).powi(2) < 45.0f32.powi(2) {
                v += 100.0;
            }
            if ((i % w) / 25 + (i / w) / 25) % 2 == 0 {
                v += 40.0;
            }
            v += (n[i] as f32 - 128.0) * 0.2;
            v.clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn tensor(px: &[u8], w: usize, h: usize) -> CpuTensor<f32> {
    let data: Vec<f32> = px.iter().map(|&p| p as f32).collect();
    Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap()
}

fn gpu_canny(gpu: &GpuContext, t: &CpuTensor<f32>, lo: f32, hi: f32) -> Vec<f32> {
    let out = gpu.canny(&t.to_gpu_ctx(gpu).unwrap(), lo, hi).unwrap();
    out.to_cpu_ctx(gpu).unwrap().as_slice().unwrap().to_vec()
}

#[test]
fn gpu_canny_matches_cpu_backend() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    for (name, w, h, px) in [
        ("scene", 203usize, 151usize, scene(203, 151)),
        ("noise", 64, 48, lcg_image(64, 48, 5)),
    ] {
        let t = tensor(&px, w, h);
        for &(lo, hi) in &[(20.0f32, 60.0f32), (60.0, 180.0), (300.0, 600.0)] {
            let c = cpu.canny(&t, lo, hi).unwrap();
            let c = c.as_slice().unwrap();
            let g = gpu_canny(gpu, &t, lo, hi);
            let bad = c.iter().zip(&g).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "{name} {lo}/{hi}: {bad} pixels differ");
            // And the CPU backend is the OpenCV reference on 8-bit input.
            let reference =
                cv_hal::cpu::canny_opencv::canny_u8(&px, w, h, lo as f64, hi as f64, false);
            let r: Vec<f32> = reference.iter().map(|&v| v as f32).collect();
            assert!(r == c, "{name} {lo}/{hi}: CPU backend is not canny_u8");
        }
    }
}

fn edge_map(w: usize, h: usize) -> Vec<u8> {
    let n = lcg_image(w, h, 9);
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let on = x == y || x == 20 || y == (x + 20) / 2 || x + 2 * y == 150 || n[i] > 252;
            if on {
                255
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn gpu_hough_lines_match_cpu_backend() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();
    let (w, h) = (160usize, 120usize);
    let t = tensor(&edge_map(w, h), w, h);
    let tg = t.to_gpu_ctx(gpu).unwrap();
    let pi = std::f64::consts::PI;
    for &(rho, div, thr) in &[
        (1.0f32, 180.0f64, 30u32),
        (2.0, 90.0, 30),
        (0.5, 360.0, 25),
        (1.0, 101.0, 40),
    ] {
        let theta = (pi / div) as f32;
        let c = cpu.hough_lines(&t, rho, theta, thr).unwrap();
        let g = gpu.hough_lines(&tg, rho, theta, thr).unwrap();
        let key = |v: &[cv_core::HoughLine]| -> Vec<(f32, f32, u32)> {
            v.iter().map(|l| (l.rho, l.theta, l.score)).collect()
        };
        assert!(!c.is_empty(), "rho {rho} pi/{div}: no CPU lines");
        assert_eq!(key(&g), key(&c), "rho {rho} pi/{div}: GPU lines differ");
    }
}

#[test]
fn gpu_hough_accumulator_is_bit_identical_including_overflow_recount() {
    // Accumulator-level check with the HAL convention, once with the normal
    // ambiguous-vote list and once with capacity 0, which forces the host
    // recount of the whole image.
    let Some(gpu) = require_gpu() else { return };
    let (w, h) = (317usize, 233usize);
    let edges = edge_map(w, h);
    let layout = cv_hal::cpu::hough_standard::HoughLayout::new(w, h, 1.0, 0.0174533);
    let (cos, sin) = layout.trig_tables();
    let mut expected = vec![0u32; layout.num_rho * layout.num_theta];
    for (i, _) in edges.iter().enumerate().filter(|(_, &e)| e != 0) {
        for t in 0..layout.num_theta {
            let k = layout.bin(i % w, i / w, cos[t], sin[t]);
            if let Some(j) = layout.index(t, k) {
                expected[j] += 1;
            }
        }
    }
    let sc: Vec<f32> = cos.iter().map(|c| c / layout.rho).collect();
    let ss: Vec<f32> = sin.iter().map(|s| s / layout.rho).collect();
    let exact = |x: usize, y: usize, n: usize| layout.index(n, layout.bin(x, y, cos[n], sin[n]));
    for cap in [1u32 << 20, 0] {
        let got = cv_hal::gpu_kernels::hough::hough_accumulate_u8_with_capacity(
            gpu,
            &edges,
            w,
            h,
            &sc,
            &ss,
            layout.num_rho,
            layout.rho_center as i32,
            &exact,
            cap,
        )
        .unwrap();
        let bad = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "capacity {cap}: {bad} bins differ");
    }
}
