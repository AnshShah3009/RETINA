//! `CpuBackend::stereo_match`: integer-valued images take the O(W*H*D)
//! running-sum SAD path (`cpu::stereo_sad`); the result must equal the dense
//! O(W*H*D*B^2) loop it replaced, transcribed below as `dense`, exactly.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::{Float, TensorShape};
use cv_hal::context::{ComputeContext, StereoMatchMethod, StereoMatchParams};
use cv_hal::cpu::CpuBackend;
use std::time::Instant;

/// The pre-existing dense SAD loop.
fn dense<T: Float>(l: &[T], r: &[T], w: usize, h: usize, p: &StereoMatchParams) -> Vec<T> {
    let (b, half) = (p.block_size, p.block_size / 2);
    let mut out = vec![T::from_f32(-1.0); w * h];
    for y in 0..h {
        if y < half || y + half >= h {
            continue;
        }
        for x in 0..w {
            if x < half || x + half >= w {
                continue;
            }
            let mut best_sad = T::from_f32(f32::MAX);
            let mut best = T::from_f32(-1.0);
            for d in 0..p.num_disparities {
                let disp = p.min_disparity + d;
                let rx = x as i32 - disp;
                if rx < half as i32 || rx + half as i32 >= w as i32 {
                    continue;
                }
                let rx = rx as usize;
                let mut sad = T::ZERO;
                for by in 0..b {
                    let sy = y - half + by;
                    for bx in 0..b {
                        sad += (l[sy * w + x - half + bx] - r[sy * w + rx - half + bx]).abs();
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best = T::from_f32(disp as f32);
                }
            }
            out[y * w + x] = best;
        }
    }
    out
}

fn noise(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u32
        })
        .collect()
}

/// A textured left image and a right image shifted by `shift` pixels, with
/// a little noise so costs are not all zero, plus a sprinkle of ties.
fn pair(w: usize, h: usize, shift: usize, seed: u64, max: u32) -> (Vec<f64>, Vec<f64>) {
    let n = noise(seed, w * h * 2);
    let l: Vec<f64> = (0..w * h).map(|i| (n[i] % (max + 1)) as f64).collect();
    let r: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let src = y * w + (x + shift).min(w - 1);
            let jitter = (n[w * h + i] % 3) as f64 - 1.0;
            (l[src] + jitter).clamp(0.0, max as f64)
        })
        .collect();
    (l, r)
}

fn run<T: Float + bytemuck::Pod + 'static>(
    l: &[f64],
    r: &[f64],
    w: usize,
    h: usize,
    p: &StereoMatchParams,
) {
    let lt: Vec<T> = l.iter().map(|&v| T::from_f64(v)).collect();
    let rt: Vec<T> = r.iter().map(|&v| T::from_f64(v)).collect();
    let want = dense(&lt, &rt, w, h, p);
    let lt: Tensor<T, CpuStorage<T>> = Tensor::from_vec(lt, TensorShape::new(1, h, w)).unwrap();
    let rt: Tensor<T, CpuStorage<T>> = Tensor::from_vec(rt, TensorShape::new(1, h, w)).unwrap();
    let got: Tensor<T, CpuStorage<T>> = CpuBackend::new()
        .unwrap()
        .stereo_match(&lt, &rt, p)
        .unwrap();
    let got = got.as_slice().unwrap();
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            Float::to_f64(*g) == Float::to_f64(*e),
            "{w}x{h} {p:?}: pixel ({}, {}) got {} want {}",
            i % w,
            i / w,
            Float::to_f64(*g),
            Float::to_f64(*e)
        );
    }
}

#[test]
fn running_sum_stereo_equals_dense_sad() {
    let mut seed = 0;
    for &(w, h) in &[(1usize, 1usize), (4, 3), (23, 11), (64, 21), (37, 40)] {
        for block in [1usize, 2, 3, 4, 5, 7, 9] {
            for (min_disparity, num_disparities) in
                [(0, 0), (0, 1), (0, 16), (-3, 9), (5, 12), (40, 4)]
            {
                let p = StereoMatchParams {
                    method: StereoMatchMethod::BlockMatching,
                    min_disparity,
                    num_disparities,
                    block_size: block,
                };
                seed += 1;
                // u8-like data (fast path), low-contrast data (many ties),
                let (l, r) = pair(w, h, 3, seed, 255);
                run::<f32>(&l, &r, w, h, &p);
                run::<f64>(&l, &r, w, h, &p);
                let (l, r) = pair(w, h, 2, seed, 2);
                run::<f32>(&l, &r, w, h, &p);
                // fractional data and huge integers (dense fallback).
                let lf: Vec<f64> = l.iter().map(|v| v * 0.37 + 0.1).collect();
                run::<f32>(&lf, &r, w, h, &p);
                let lb: Vec<f64> = l.iter().map(|v| v * 300_000.0).collect();
                let rb: Vec<f64> = r.iter().map(|v| v * 300_000.0).collect();
                run::<f32>(&lb, &rb, w, h, &p);
                run::<f64>(&lb, &rb, w, h, &p);
            }
        }
    }
}

/// `cargo test --release -p cv-hal --test stereo_sad_running_sum -- --ignored --nocapture`
#[test]
#[ignore]
fn stereo_timing() {
    let (w, h) = (640usize, 480usize);
    let (l, r) = pair(w, h, 17, 5, 255);
    let lt: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(
        l.iter().map(|&v| v as f32).collect(),
        TensorShape::new(1, h, w),
    )
    .unwrap();
    let rt: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(
        r.iter().map(|&v| v as f32).collect(),
        TensorShape::new(1, h, w),
    )
    .unwrap();
    let cpu = CpuBackend::new().unwrap();
    for (block, nd) in [(5usize, 64i32), (9, 64), (15, 64), (9, 128)] {
        let p = StereoMatchParams {
            method: StereoMatchMethod::BlockMatching,
            min_disparity: 0,
            num_disparities: nd,
            block_size: block,
        };
        let _: Tensor<f32, CpuStorage<f32>> = cpu.stereo_match(&lt, &rt, &p).unwrap();
        let reps = 3;
        let t = Instant::now();
        for _ in 0..reps {
            let o: Tensor<f32, CpuStorage<f32>> = cpu.stereo_match(&lt, &rt, &p).unwrap();
            std::hint::black_box(o);
        }
        println!(
            "stereo_match 640x480 block {block:2} disparities {nd:3}: {:8.2} ms",
            t.elapsed().as_secs_f64() / reps as f64 * 1e3
        );
    }
}
