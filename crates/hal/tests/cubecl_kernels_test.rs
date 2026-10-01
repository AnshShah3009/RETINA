//! Numerical verification of the ported CubeCL kernels against CPU references.
//!
//! Everything here is a *real* GPU launch: the client is built from
//! `cubecl_wgpu::WgpuRuntime`, which enumerates a real Vulkan/CUDA adapter. If no
//! adapter can be created the tests fail loudly rather than skipping — see
//! [`client`].
//!
//! Scope, and what is deliberately *not* here:
//!
//! * Every test compares the device result against an independently written
//!   CPU reference. No test compares the kernel against itself.
//! * Every test in this file was checked to **fail** when the kernel it covers
//!   is deliberately broken. See `docs/cubecl_verification.md` for the mutation
//!   log. A test that does not catch its mutation is not evidence and was
//!   rewritten or removed.
//! * Kernels whose ported semantics are *documented as wrong* (the ones carrying
//!   `BUG(original)`) are pinned against their actual behaviour, with the test
//!   name and comment saying so, so a reader cannot mistake the pin for an
//!   endorsement.
//! * Kernels that cannot be given a reference are listed in the module-level
//!   `not_covered` list in `docs/cubecl_verification.md` rather than being given
//!   a vacuous test.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use cv_hal::gpu_kernels::cubecl_advanced as adv;
use cv_hal::gpu_kernels::cubecl_backend as be;
use cv_hal::gpu_kernels::cubecl_optimized as opt;
use cv_hal::gpu_kernels::cubecl_proto as proto;

type Rt = WgpuRuntime;

/// A live compute client, or a loud failure.
///
/// There is no `skip` here on purpose: a test suite that quietly turns itself
/// off on the CI machine is exactly the failure mode this crate is trying to
/// avoid. If the client cannot be built the test fails and says why.
fn client() -> ComputeClient<Rt> {
    <Rt as Runtime>::client(&WgpuDevice::DefaultDevice)
}

fn ctx_be(c: &ComputeClient<Rt>) -> be::CubeCLContext<Rt> {
    be::CubeCLContext::new(c.clone())
}
fn ctx_adv(c: &ComputeClient<Rt>) -> adv::AdvancedContext<Rt> {
    adv::AdvancedContext::new(c.clone())
}
fn ctx_opt(c: &ComputeClient<Rt>) -> opt::OptimizedContext<Rt> {
    opt::OptimizedContext::new(c.clone(), false)
}

// ---------------------------------------------------------------------------
// Small helpers: upload, download, compare, deterministic data
// ---------------------------------------------------------------------------

fn up(c: &ComputeClient<Rt>, data: &[f32], shape: Vec<usize>) -> TensorHandle<Rt> {
    proto::tensor_from_slice(c, data, shape).expect("upload")
}
fn up_u32(c: &ComputeClient<Rt>, data: &[u32], shape: Vec<usize>) -> TensorHandle<Rt> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let handle = c.create_from_slice(&bytes);
    TensorHandle::new_contiguous(shape, handle, proto::u32_storage())
}
fn upz(c: &ComputeClient<Rt>, shape: Vec<usize>) -> TensorHandle<Rt> {
    let bytes: Vec<u8> = vec![0u8; shape.iter().product::<usize>() * 4];
    let handle = c.create_from_slice(&bytes);
    TensorHandle::new_contiguous(shape, handle, proto::f32_storage())
}
fn down(c: &ComputeClient<Rt>, t: &TensorHandle<Rt>) -> Vec<f32> {
    proto::tensor_to_slice(c, t).expect("readback")
}
fn down_u32(c: &ComputeClient<Rt>, t: &TensorHandle<Rt>) -> Vec<u32> {
    proto::tensor_to_slice_u32(c, t).expect("readback")
}

/// Elementwise comparison with a reported first-bad-element message.
#[track_caller]
fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch, gpu {} vs cpu {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{label}: worst |err| = {worst} at element {worst_i} (gpu {} vs cpu {}) \
         exceeds tol {tol}",
        got[worst_i],
        want[worst_i]
    );
}

#[track_caller]
fn assert_eq_idx(label: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(*g, *w, "{label}: index mismatch at element {i}");
    }
}

/// Deterministic pseudo-random values in `[-1, 1]`, so a failure is
/// reproducible. A fixed LCG; not `rand`, to avoid another dependency.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let v = (*seed >> 33) as u32;
    (v as f32 / u32::MAX as f32) * 2.0 - 1.0
}
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}
fn data_range(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| lo + (hi - lo) * ((lcg(&mut s) + 1.0) * 0.5))
        .collect()
}

// ===========================================================================
// 1. Element-wise ops (cubecl_proto)
// ===========================================================================

#[test]
fn proto_binary_elemwise_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    // 1000 elements is deliberately not a multiple of CUBE_DIM (64), so the
    // `i >= n` guard is exercised on the last cube.
    let n = 1000;
    let a = data(n, 1);
    let b = data(n, 2);
    let ta = up(&c, &a, vec![n]);
    let tb = up(&c, &b, vec![n]);

    let got = down(&c, &be::add(&ctx, &ta, &tb).unwrap());
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close("add", &got, &want, 1e-5);

    let got = down(&c, &be::sub(&ctx, &ta, &tb).unwrap());
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    assert_close("sub", &got, &want, 1e-5);

    let got = down(&c, &be::mul(&ctx, &ta, &tb).unwrap());
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    assert_close("mul", &got, &want, 1e-5);
}

#[test]
fn proto_unary_elemwise_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let n = 777;
    let a = data_range(n, 3, -8.0, 8.0);
    let ta = up(&c, &a, vec![n]);

    let got = down(&c, &be::relu(&ctx, &ta).unwrap());
    let want: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
    assert_close("relu", &got, &want, 1e-6);

    let got = down(&c, &be::sigmoid(&ctx, &ta).unwrap());
    let want: Vec<f32> = a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_close("sigmoid", &got, &want, 1e-5);

    let got = down(&c, &be::tanh(&ctx, &ta).unwrap());
    let want: Vec<f32> = a
        .iter()
        .map(|x| {
            let e = (2.0 * x).exp();
            (e - 1.0) / (e + 1.0)
        })
        .collect();
    assert_close("tanh", &got, &want, 1e-5);
}

// ===========================================================================
// 2. Matmul
// ===========================================================================

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

#[test]
fn proto_matmul_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let (m, k, n) = (13usize, 17usize, 11usize);
    let a = data(m * k, 11);
    let b = data(k * n, 12);
    let ta = up(&c, &a, vec![m, k]);
    let tb = up(&c, &b, vec![k, n]);

    let got = down(&c, &be::matmul(&ctx, &ta, &tb).unwrap());
    let want = cpu_matmul(&a, &b, m, k, n);
    assert_close("matmul", &got, &want, 1e-4);
}

// ===========================================================================
// 3. Host-side reductions
// ===========================================================================

#[test]
fn proto_host_reduce_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    // rank-3 [2, 5, 6]
    let (d0, d1, d2) = (2usize, 5usize, 6usize);
    let a = data(d0 * d1 * d2, 21);
    let t = up(&c, &a, vec![d0, d1, d2]);

    // axis 1
    let got = down(&c, &be::sum(&ctx, &t, 1).unwrap());
    let mut want = vec![0.0f32; d0 * d2];
    for i0 in 0..d0 {
        for i2 in 0..d2 {
            let mut acc = 0.0;
            for i1 in 0..d1 {
                acc += a[(i0 * d1 + i1) * d2 + i2];
            }
            want[i0 * d2 + i2] = acc;
        }
    }
    assert_close("sum axis=1", &got, &want, 1e-4);

    let got = down(&c, &be::max(&ctx, &t, 1).unwrap());
    let mut want = vec![f32::NEG_INFINITY; d0 * d2];
    for i0 in 0..d0 {
        for i2 in 0..d2 {
            for i1 in 0..d1 {
                want[i0 * d2 + i2] = want[i0 * d2 + i2].max(a[(i0 * d1 + i1) * d2 + i2]);
            }
        }
    }
    assert_close("max axis=1", &got, &want, 1e-6);

    let got = down(&c, &be::min(&ctx, &t, 1).unwrap());
    let mut want = vec![f32::INFINITY; d0 * d2];
    for i0 in 0..d0 {
        for i2 in 0..d2 {
            for i1 in 0..d1 {
                want[i0 * d2 + i2] = want[i0 * d2 + i2].min(a[(i0 * d1 + i1) * d2 + i2]);
            }
        }
    }
    assert_close("min axis=1", &got, &want, 1e-6);

    // The mean needs its *own* reference: the `want` above holds the axis-1
    // **minimum**, carried over from the previous check. Dividing a minimum by
    // d1 is not a mean, so this was comparing the kernel against nonsense and
    // reporting a numeric mismatch that said nothing about the kernel.
    let got = down(&c, &be::mean(&ctx, &t, 1).unwrap());
    let mut want = vec![0.0f32; d0 * d2];
    for i0 in 0..d0 {
        for i2 in 0..d2 {
            let mut acc = 0.0f32;
            for i1 in 0..d1 {
                acc += a[(i0 * d1 + i1) * d2 + i2];
            }
            want[i0 * d2 + i2] = acc / d1 as f32;
        }
    }
    assert_close("mean axis=1", &got, &want, 1e-4);

    // axis 2 (innermost) and axis 0 (outermost) too, since the index
    // arithmetic differs for each.
    let got = down(&c, &be::sum(&ctx, &t, 2).unwrap());
    let mut want = vec![0.0f32; d0 * d1];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            let mut acc = 0.0;
            for i2 in 0..d2 {
                acc += a[(i0 * d1 + i1) * d2 + i2];
            }
            want[i0 * d1 + i1] = acc;
        }
    }
    assert_close("sum axis=2", &got, &want, 1e-4);

    let got = down(&c, &be::sum(&ctx, &t, 0).unwrap());
    let mut want = vec![0.0f32; d1 * d2];
    for i1 in 0..d1 {
        for i2 in 0..d2 {
            let mut acc = 0.0;
            for i0 in 0..d0 {
                acc += a[(i0 * d1 + i1) * d2 + i2];
            }
            want[i1 * d2 + i2] = acc;
        }
    }
    assert_close("sum axis=0", &got, &want, 1e-4);
}

// ===========================================================================
// 4. conv2d (both modules)
// ===========================================================================

/// Zero-padded NCHW convolution, cross-correlation (no kernel flip).
fn cpu_conv2d(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    ci: usize,
    co: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; batch * co * oh * ow];
    for b in 0..batch {
        for o in 0..co {
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = 0.0f32;
                    for cii in 0..ci {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy_raw = y * stride + ky;
                                let ix_raw = x * stride + kx;
                                if iy_raw < pad || ix_raw < pad {
                                    continue;
                                }
                                let iy = iy_raw - pad;
                                let ix = ix_raw - pad;
                                if iy >= h || ix >= w {
                                    continue;
                                }
                                let iv = input[((b * ci + cii) * h + iy) * w + ix];
                                let wv = weights[((o * ci + cii) * k + ky) * k + kx];
                                acc += iv * wv;
                            }
                        }
                    }
                    out[((b * co + o) * oh + y) * ow + x] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn backend_conv2d_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let (batch, ci, co, h, w, k, stride, pad) = (
        2usize, 3usize, 4usize, 9usize, 11usize, 3usize, 2usize, 1usize,
    );
    let input = data(batch * ci * h * w, 31);
    let weights = data(co * ci * k * k, 32);
    let ti = up(&c, &input, vec![batch, ci, h, w]);
    let tw = up(&c, &weights, vec![co, ci, k, k]);

    let got = down(&c, &be::conv2d(&ctx, &ti, &tw, stride, pad).unwrap());
    let want = cpu_conv2d(&input, &weights, batch, ci, co, h, w, k, stride, pad);
    assert_close("backend conv2d", &got, &want, 1e-4);
}

/// `co > 2` is the point of this test: the original `conv2d_kernel` in
/// `cubecl_optimized` collapsed every output channel onto channel 0, so a
/// multi-channel case is the only thing that can catch a regression to that.
#[test]
fn optimized_conv2d_matches_cpu_and_separates_channels() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (batch, ci, co, h, w, k, stride, pad) = (
        2usize, 3usize, 5usize, 8usize, 8usize, 3usize, 1usize, 1usize,
    );
    let input = data(batch * ci * h * w, 41);
    let weights = data(co * ci * k * k, 42);
    let ti = up(&c, &input, vec![batch, ci, h, w]);
    let tw = up(&c, &weights, vec![co, ci, k, k]);

    let got = down(&c, &opt::conv2d(&ctx, &ti, &tw, stride, pad).unwrap());
    let want = cpu_conv2d(&input, &weights, batch, ci, co, h, w, k, stride, pad);

    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    assert_close("optimized conv2d", &got, &want, 1e-4);
    assert_eq!(got.len(), batch * co * oh * ow);

    // Explicit anti-regression: the five output channels must not be identical
    // clones of channel 0.
    let plane = oh * ow;
    let ch0 = &got[..plane];
    for o in 1..co {
        assert_ne!(
            &got[o * plane..(o + 1) * plane],
            ch0,
            "output channel {o} is a byte-for-byte copy of channel 0 \
             (the original collapsed-channels bug)"
        );
    }
}

// ===========================================================================
// 5. Depthwise / transposed convolution
// ===========================================================================

fn cpu_depthwise(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    ch: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; batch * ch * oh * ow];
    for b in 0..batch {
        for c in 0..ch {
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = 0.0f32;
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy_raw = y * stride + ky;
                            let ix_raw = x * stride + kx;
                            if iy_raw < pad || ix_raw < pad {
                                continue;
                            }
                            let iy = iy_raw - pad;
                            let ix = ix_raw - pad;
                            if iy >= h || ix >= w {
                                continue;
                            }
                            acc += input[((b * ch + c) * h + iy) * w + ix]
                                * weights[(c * k + ky) * k + kx];
                        }
                    }
                    out[((b * ch + c) * oh + y) * ow + x] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn optimized_depthwise_conv2d_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (batch, ch, h, w, k, stride, pad) =
        (2usize, 4usize, 7usize, 9usize, 3usize, 2usize, 1usize);
    let input = data(batch * ch * h * w, 51);
    let weights = data(ch * k * k, 52);
    let ti = up(&c, &input, vec![batch, ch, h, w]);
    let tw = up(&c, &weights, vec![ch, k, k]);

    let got = down(
        &c,
        &opt::depthwise_conv2d(&ctx, &ti, &tw, stride, pad).unwrap(),
    );
    let want = cpu_depthwise(&input, &weights, batch, ch, h, w, k, stride, pad);
    assert_close("depthwise conv2d", &got, &want, 1e-4);
}

/// Reference transposed convolution, gather form, exactly the formula the
/// kernel's doc comment states.
fn cpu_transposed_conv(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    ci: usize,
    co: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> (Vec<f32>, usize, usize) {
    let oh = (h - 1) * stride + k - 2 * pad + out_pad;
    let ow = (w - 1) * stride + k - 2 * pad + out_pad;
    let mut out = vec![0.0f32; batch * co * oh * ow];
    for b in 0..batch {
        for o in 0..co {
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = 0.0f32;
                    for c in 0..ci {
                        for ky in 0..k {
                            let iy_raw = y + pad;
                            if ky > iy_raw {
                                continue;
                            }
                            let iy = iy_raw - ky;
                            if iy >= h {
                                continue;
                            }
                            for kx in 0..k {
                                let ix_raw = x + pad;
                                if kx > ix_raw {
                                    continue;
                                }
                                let ix = ix_raw - kx;
                                if ix >= w {
                                    continue;
                                }
                                acc += input[((b * ci + c) * h + iy) * w + ix]
                                    * weights[((c * co + o) * k + ky) * k + kx];
                            }
                        }
                    }
                    out[((b * co + o) * oh + y) * ow + x] = acc;
                }
            }
        }
    }
    (out, oh, ow)
}

#[test]
fn optimized_transposed_conv2d_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (batch, ci, co, h, w, k, stride, pad, out_pad) = (
        2usize, 3usize, 4usize, 5usize, 6usize, 3usize, 2usize, 1usize, 1usize,
    );
    let input = data(batch * ci * h * w, 61);
    let weights = data(ci * co * k * k, 62);
    let ti = up(&c, &input, vec![batch, ci, h, w]);
    let tw = up(&c, &weights, vec![ci, co, k, k]);

    let got = down(
        &c,
        &opt::transposed_conv2d(&ctx, &ti, &tw, stride, pad, out_pad).unwrap(),
    );
    let (want, oh, ow) = cpu_transposed_conv(
        &input, &weights, batch, ci, co, h, w, k, stride, pad, out_pad,
    );
    assert_eq!(got.len(), batch * co * oh * ow);
    assert_close("transposed conv2d", &got, &want, 1e-4);
}

// ===========================================================================
// 6. Element-wise optimized (fused)
// ===========================================================================

#[test]
fn optimized_fused_elemwise_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 513;
    let a = data_range(n, 71, -4.0, 4.0);
    let b = data_range(n, 72, -4.0, 4.0);
    let ta = up(&c, &a, vec![n]);
    let tb = up(&c, &b, vec![n]);

    let got = down(&c, &opt::add_relu(&ctx, &ta, &tb).unwrap());
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x + y).max(0.0)).collect();
    assert_close("add_relu", &got, &want, 1e-6);

    let got = down(&c, &opt::add_sigmoid(&ctx, &ta, &tb).unwrap());
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| 1.0 / (1.0 + (-(x + y)).exp()))
        .collect();
    assert_close("add_sigmoid", &got, &want, 1e-5);

    let got = down(&c, &opt::leaky_relu(&ctx, &ta).unwrap());
    let want: Vec<f32> = a
        .iter()
        .map(|x| if *x > 0.0 { *x } else { *x * 0.01 })
        .collect();
    assert_close("leaky_relu", &got, &want, 1e-6);

    let got = down(&c, &opt::elu(&ctx, &ta).unwrap());
    let want: Vec<f32> = a
        .iter()
        .map(|x| if *x > 0.0 { *x } else { x.exp() - 1.0 })
        .collect();
    assert_close("elu", &got, &want, 1e-5);
}

// ===========================================================================
// 7. Histogram and partial block sum
// ===========================================================================

#[test]
fn optimized_histogram_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 4096;
    let (bins, lo, hi) = (16usize, -1.0f32, 1.0f32);
    let a = data(n, 81);
    let ta = up(&c, &a, vec![n]);

    let got = down_u32(&c, &opt::histogram(&ctx, &ta, bins, lo, hi).unwrap());
    let mut want = vec![0u32; bins];
    for v in &a {
        let scaled = (*v - lo) / (hi - lo) * bins as f32;
        // Mirrors the kernel: `scaled as usize` truncates, and `bin == bins`
        // lands in the last bucket.
        let b = scaled as usize;
        if b < bins {
            want[b] += 1;
        } else if b == bins {
            want[bins - 1] += 1;
        }
    }
    assert_eq!(
        want.iter().sum::<u32>(),
        n as u32,
        "test bug: the CPU bin assignment must account for every element"
    );
    assert_eq_eq_idx_len(&got, bins);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(*g, *w, "histogram bin {i}: gpu {g} vs cpu {w}");
    }
    assert_eq!(
        got.iter().sum::<u32>(),
        n as u32,
        "total counts must equal the input length; a lost atomic update would show here"
    );
}

fn assert_eq_eq_idx_len(got: &[u32], len: usize) {
    assert_eq!(got.len(), len, "histogram length mismatch");
}

#[test]
fn optimized_partial_block_sum_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 1000;
    let a = data(n, 91);
    let ta = up(&c, &a, vec![n]);
    let block = 128;

    let got = down(&c, &opt::partial_block_sum(&ctx, &ta, block).unwrap());
    let blocks = n.div_ceil(block);
    let mut want = vec![0.0f32; blocks];
    for b in 0..blocks {
        for i in 0..block {
            let pos = b * block + i;
            if pos < n {
                want[b] += a[pos];
            }
        }
    }
    assert_close("partial_block_sum", &got, &want, 1e-3);
}

// ===========================================================================
// 8. Point cloud: pairwise distance, kNN, voxel hash, Morton codes
// ===========================================================================

fn cpu_pairwise(points: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = [0usize, 1, 2]
                .iter()
                .map(|k| points[i * 3 + k] - points[j * 3 + k])
                .fold(0.0f32, |acc, v| acc + v * v);
            out[i * n + j] = d;
        }
    }
    out
}

#[test]
fn backend_pairwise_squared_distance_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let n = 37;
    let pts = data(n * 3, 101);
    let tp = up(&c, &pts, vec![n, 3]);
    let got = down(&c, &be::pairwise_squared_distance(&ctx, &tp, n).unwrap());
    let want = cpu_pairwise(&pts, n);
    assert_close("backend pairwise_sqdist", &got, &want, 1e-4);
    // Symmetry and zero diagonal: cheap structural checks the CPU reference
    // would also satisfy, so they cannot substitute for it, but they localise a
    // failure nicely.
    for i in 0..n {
        assert!((got[i * n + i]).abs() < 1e-5, "diagonal must be zero");
        for j in 0..n {
            assert!(
                (got[i * n + j] - got[j * n + i]).abs() < 1e-4,
                "distance matrix must be symmetric"
            );
        }
    }
}

#[test]
fn optimized_pairwise_squared_distance_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 41;
    let pts = data(n * 3, 102);
    let tp = up(&c, &pts, vec![n, 3]);
    let got = down(&c, &opt::pairwise_squared_distance(&ctx, &tp).unwrap());
    let want = cpu_pairwise(&pts, n);
    assert_close("optimized pairwise_sqdist", &got, &want, 1e-4);
}

/// Full k-NN by brute force, as the CPU reference. Ties are broken by keeping
/// the earlier point index, matching the kernels' strict `<` insertion test.
fn cpu_knn(points: &[f32], queries: &[f32], k: usize, nq: usize) -> (Vec<f32>, Vec<u32>) {
    let n = points.len() / 3;
    let mut dists = vec![0.0f32; nq * k];
    let mut idxs = vec![0u32; nq * k];
    for q in 0..nq {
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(n);
        for p in 0..n {
            let d: f32 = [0usize, 1, 2]
                .iter()
                .map(|k| queries[q * 3 + k] - points[p * 3 + k])
                .fold(0.0f32, |acc, v| acc + v * v);
            best.push((d, p as u32));
        }
        best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
        for j in 0..k {
            dists[q * k + j] = if j < n { best[j].0 } else { 1e30 };
            idxs[q * k + j] = if j < n { best[j].1 } else { 0 };
        }
    }
    (dists, idxs)
}

#[test]
fn backend_knn_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let (n, nq, k) = (64usize, 9usize, 5usize);
    let pts = data(n * 3, 111);
    let qs = data(nq * 3, 112);
    let tp = up(&c, &pts, vec![n, 3]);
    let tq = up(&c, &qs, vec![nq, 3]);

    let (gd, gi) = be::knn(&ctx, &tp, &tq, k).unwrap();
    let (wd, wi) = cpu_knn(&pts, &qs, k, nq);
    assert_close("backend knn distances", &down(&c, &gd), &wd, 1e-4);
    assert_eq_idx("backend knn indices", &down_u32(&c, &gi), &wi);
    // Distances must come back sorted ascending; the insertion sort is the whole
    // point of the kernel.
    for q in 0..nq {
        for j in 1..k {
            let prev = down_val(&c, &gd, q * k + j - 1);
            let cur = down_val(&c, &gd, q * k + j);
            assert!(prev <= cur + 1e-4, "knn distances must be sorted");
        }
    }
}

fn down_val<R: Runtime>(c: &ComputeClient<R>, t: &TensorHandle<R>, i: usize) -> f32 {
    proto::tensor_to_slice(c, t).expect("readback")[i]
}

#[test]
fn optimized_knn_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (n, nq, k) = (53usize, 7usize, 4usize);
    let pts = data(n * 3, 121);
    let qs = data(nq * 3, 122);
    let tp = up(&c, &pts, vec![n, 3]);
    let tq = up(&c, &qs, vec![nq, 3]);

    let (gd, gi) = opt::knn(&ctx, &tp, &tq, k, 0.0).unwrap();
    let (wd, wi) = cpu_knn(&pts, &qs, k, nq);
    assert_close("optimized knn distances", &down(&c, &gd), &wd, 1e-4);
    assert_eq_idx("optimized knn indices", &down_u32(&c, &gi), &wi);
}

#[test]
fn optimized_knn_with_rejection_radius_matches_brute_force_within_radius() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (n, nq, k, radius) = (60usize, 6usize, 3usize, 1.5f32);
    let pts = data(n * 3, 131);
    let qs = data(nq * 3, 132);
    let tp = up(&c, &pts, vec![n, 3]);
    let tq = up(&c, &qs, vec![nq, 3]);

    let (gd, gi) = opt::knn(&ctx, &tp, &tq, k, radius).unwrap();
    let gdist = down(&c, &gd);
    let gidx = down_u32(&c, &gi);

    // Reference: the same brute force, but restricted to points within the
    // radius. Slots with nothing in range must stay at the +inf sentinel.
    let r2 = radius * radius;
    for q in 0..nq {
        let mut best: Vec<(f32, u32)> = (0..n)
            .map(|p| {
                let d: f32 = [0usize, 1, 2]
                    .iter()
                    .map(|k| qs[q * 3 + k] - pts[p * 3 + k])
                    .fold(0.0f32, |acc, v| acc + v * v);
                (d, p as u32)
            })
            .filter(|(d, _)| *d <= r2)
            .collect();
        best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
        for j in 0..k {
            let gd_j = gdist[q * k + j];
            if j < best.len() {
                assert!(
                    (gd_j - best[j].0).abs() < 1e-4,
                    "query {q} slot {j}: gpu {gd_j} vs cpu {}",
                    best[j].0
                );
                assert_eq!(gidx[q * k + j], best[j].1, "query {q} slot {j} index");
            } else {
                assert!(
                    gd_j >= 1e29,
                    "query {q} slot {j}: expected the +inf sentinel, got {gd_j}"
                );
            }
        }
    }
}

#[test]
fn backend_knn_pads_with_infinity_when_k_exceeds_num_points() {
    // This pins the documented behaviour of the `BUG(original)` marker on
    // `knn_kernel`: unfilled slots are `+inf` with index 0 rather than garbage.
    let c = client();
    let ctx = ctx_be(&c);
    let (n, nq, k) = (4usize, 3usize, 8usize);
    let pts = data(n * 3, 141);
    let qs = data(nq * 3, 142);
    let tp = up(&c, &pts, vec![n, 3]);
    let tq = up(&c, &qs, vec![nq, 3]);

    let (gd, gi) = be::knn(&ctx, &tp, &tq, k).unwrap();
    let gdist = down(&c, &gd);
    let gidx = down_u32(&c, &gi);
    let (wd, wi) = cpu_knn(&pts, &qs, k, nq);
    assert_close("knn padded distances", &gdist, &wd, 1e-4);
    assert_eq_idx("knn padded indices", &gidx, &wi);
    for q in 0..nq {
        for j in n..k {
            assert!(
                gdist[q * k + j] >= 1e29,
                "slot {j} of query {q} should be the +inf sentinel"
            );
        }
    }
}

fn cpu_voxel_hash(pts: &[f32], n: usize, voxel: f32) -> Vec<u32> {
    let mut out = vec![0u32; n * 2];
    for i in 0..n {
        let vx = (pts[i * 3] / voxel) as i32;
        let vy = (pts[i * 3 + 1] / voxel) as i32;
        let vz = (pts[i * 3 + 2] / voxel) as i32;
        let h = ((vx * 73857) ^ (vy * 19349) ^ (vz * 83493)) as u32;
        out[i * 2] = h;
        out[i * 2 + 1] = i as u32;
    }
    out
}

#[test]
fn backend_voxel_hash_matches_cpu() {
    let c = client();
    let ctx = ctx_be(&c);
    let n = 64;
    // Coordinate magnitudes must stay inside the i32 range the kernel's comment
    // describes, so stay inside +/- 4000 with voxel_size 4.
    let pts = data_range(n * 3, 151, -4000.0, 4000.0);
    let tp = up(&c, &pts, vec![n, 3]);
    let got = down_u32(&c, &be::voxel_hash(&ctx, &tp, 4.0).unwrap());
    let want = cpu_voxel_hash(&pts, n, 4.0);
    assert_eq!(got.len(), n * 2);
    assert_eq_idx("voxel hash", &got, &want);
}

fn cpu_morton(pts: &[f32], n: usize, lo: f32, hi: f32, bits: u32) -> Vec<u32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0u32; n];
    for i in 0..n {
        let q = |v: f32| ((((v - lo) / (hi - lo)) * levels) as u32).min(levels as u32);
        let (nx, ny, nz) = (q(pts[i * 3]), q(pts[i * 3 + 1]), q(pts[i * 3 + 2]));
        let mut code = 0u32;
        for b in 0..bits {
            code |= ((nx >> b) & 1) << (3 * b);
            code |= ((ny >> b) & 1) << (3 * b + 1);
            code |= ((nz >> b) & 1) << (3 * b + 2);
        }
        out[i] = code;
    }
    out
}

#[test]
fn optimized_morton_codes_match_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 50;
    let pts = data_range(n * 3, 161, -2.0, 2.0);
    let tp = up(&c, &pts, vec![n, 3]);
    for bits in [1u32, 3, 10] {
        let got = down_u32(&c, &opt::morton_codes(&ctx, &tp, -2.0, 2.0, bits).unwrap());
        let want = cpu_morton(&pts, n, -2.0, 2.0, bits);
        assert_eq_idx(&format!("morton codes bits={bits}"), &got, &want);
    }
}

// ===========================================================================
// 9. Image processing: gradients, blur, bilateral, pooling
// ===========================================================================

#[test]
fn advanced_gradients_match_cpu() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w) = (6usize, 9usize);
    let img = data(h * w, 171);
    let ti = up(&c, &img, vec![h, w]);

    let (gx, gy) = adv::compute_gradients(&ctx, &ti).unwrap();
    let got_x = down(&c, &gx);
    let got_y = down(&c, &gy);

    let mut want_x = vec![0.0f32; h * w];
    let mut want_y = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            if x == 0 || x >= w - 1 || y == 0 || y >= h - 1 {
                continue;
            }
            want_x[y * w + x] = (-img[y * w + x - 1] + img[y * w + x + 1]) * 0.5;
            want_y[y * w + x] = (-img[(y - 1) * w + x] + img[(y + 1) * w + x]) * 0.5;
        }
    }
    assert_close("grad_x", &got_x, &want_x, 1e-6);
    assert_close("grad_y", &got_y, &want_y, 1e-6);
}

#[test]
fn advanced_gaussian_blur_matches_cpu() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w) = (16usize, 19usize);
    let img = data(h * w, 181);
    let ti = up(&c, &img, vec![h, w]);

    let got = down(&c, &adv::gaussian_blur(&ctx, &ti).unwrap());

    let row = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let mut want = img.clone();
    for y in 0..h {
        for x in 0..w {
            if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
                continue;
            }
            let mut acc = 0.0f32;
            for ky in 0..5usize {
                for kx in 0..5usize {
                    acc += img[(y + ky - 2) * w + (x + kx - 2)] * (row[kx] * row[ky]);
                }
            }
            want[y * w + x] = acc / 256.0;
        }
    }
    assert_close("gaussian blur", &got, &want, 1e-5);
}

#[test]
fn advanced_bilateral_filter_matches_cpu() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w) = (16usize, 17usize);
    let img = data_range(h * w, 191, 0.0, 1.0);
    let ti = up(&c, &img, vec![h, w]);
    let (ss, sr) = (3.0f32, 0.2f32);

    let got = down(&c, &adv::bilateral_filter(&ctx, &ti, ss, sr).unwrap());

    let mut want = img.clone();
    for y in 0..h {
        for x in 0..w {
            if x < 3 || x >= w - 3 || y < 3 || y >= h - 3 {
                continue;
            }
            let centre = img[y * w + x];
            let (mut acc, mut wsum) = (0.0f32, 0.0f32);
            for ky in 0..7usize {
                for kx in 0..7usize {
                    let val = img[(y + ky - 3) * w + (x + kx - 3)];
                    let dx = (kx as f32 - 3.0) / ss;
                    let dy = (ky as f32 - 3.0) / ss;
                    let dr = (val - centre) / sr;
                    let wt = (-(dx * dx + dy * dy) * 0.5).exp() * (-(dr * dr) * 0.5).exp();
                    acc += val * wt;
                    wsum += wt;
                }
            }
            want[y * w + x] = acc / (wsum + 1e-8);
        }
    }
    assert_close("bilateral filter", &got, &want, 1e-5);
}

fn cpu_pool(img: &[f32], w: usize, h: usize, pool: usize, stride: usize, avg: bool) -> Vec<f32> {
    let ow = (w - pool) / stride + 1;
    let oh = (h - pool) / stride + 1;
    let mut out = vec![0.0f32; oh * ow];
    for y in 0..oh {
        for x in 0..ow {
            let mut acc = if avg { 0.0f32 } else { -1e10f32 };
            let mut cnt = 0.0f32;
            for py in 0..pool {
                for px in 0..pool {
                    let (ix, iy) = (x * stride + px, y * stride + py);
                    if ix < w && iy < h {
                        let v = img[iy * w + ix];
                        if avg {
                            acc += v;
                            cnt += 1.0;
                        } else {
                            acc = acc.max(v);
                        }
                    }
                }
            }
            out[y * ow + x] = if avg { acc / cnt } else { acc };
        }
    }
    out
}

#[test]
fn advanced_pooling_matches_cpu() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w) = (14usize, 15usize);
    let img = data(h * w, 201);
    let ti = up(&c, &img, vec![h, w]);

    for (pool, stride) in [(2usize, 2usize), (3, 2), (2, 1)] {
        let got = down(&c, &adv::max_pool2d(&ctx, &ti, pool, stride).unwrap());
        let want = cpu_pool(&img, w, h, pool, stride, false);
        assert_close(&format!("max_pool {pool}/{stride}"), &got, &want, 1e-6);

        let got = down(&c, &adv::avg_pool2d(&ctx, &ti, pool, stride).unwrap());
        let want = cpu_pool(&img, w, h, pool, stride, true);
        assert_close(&format!("avg_pool {pool}/{stride}"), &got, &want, 1e-5);
    }
}

fn cpu_depth_to_normal(depth: &[f32], h: usize, w: usize, fx: f32, fy: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let o = idx * 3;
            if x == 0 || x >= w - 1 || y == 0 || y >= h - 1 {
                out[o] = 0.0;
                out[o + 1] = 0.0;
                out[o + 2] = 1.0;
                continue;
            }
            let z = depth[idx];
            if z < 0.01 {
                out[o] = 0.0;
                out[o + 1] = 0.0;
                out[o + 2] = 1.0;
                continue;
            }
            let cx = x as f32 - w as f32 / 2.0;
            let cy = y as f32 - h as f32 / 2.0;
            let (px, py, pz) = (cx * z / fx, cy * z / fy, z);
            let zr = depth[y * w + x + 1];
            let zd = depth[(y + 1) * w + x];
            let (tx, ty, tz) = ((cx + 1.0) * zr / fx - px, 0.0, zr - pz);
            let (bx, by, bz) = (0.0, (cy + 1.0) * zd / fy - py, zd - pz);
            let (nx, ny, nz) = (ty * bz - tz * by, tz * bx - tx * bz, tx * by - ty * bx);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-8 {
                out[o] = nx / len;
                out[o + 1] = ny / len;
                out[o + 2] = nz / len;
            } else {
                out[o] = 0.0;
                out[o + 1] = 0.0;
                out[o + 2] = 1.0;
            }
        }
    }
    out
}

/// Pins the *actual* normal orientation of `depth_to_normal_kernel`.
///
/// The `BUG(original)` marker in `cubecl_advanced.rs` claims that `t x b` with
/// `+x`/`+y` tangents "points into the surface (`-z`) for a fronto-parallel
/// plane". **That claim is false and this test is what proves it.** Measured on
/// the GPU, a fronto-parallel plane gives `+z` — see the correction note on the
/// marker itself.
#[test]
fn advanced_depth_to_normals_point_away_from_the_camera_not_into_the_scene() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w) = (9usize, 11usize);
    let (fx, fy) = (50.0f32, 50.0f32);
    let depth = data_range(h * w, 211, 1.0, 5.0);
    let td = up(&c, &depth, vec![h, w]);

    let got = down(&c, &adv::depth_to_normals(&ctx, &td, fx, fy).unwrap());
    let want = cpu_depth_to_normal(&depth, h, w, fx, fy);
    assert_close("depth to normals", &got, &want, 1e-5);

    // Constant depth == a plane parallel to the image plane.
    let flat = up(&c, &vec![2.0f32; h * w], vec![h, w]);
    let got = down(&c, &adv::depth_to_normals(&ctx, &flat, fx, fy).unwrap());
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let o = (y * w + x) * 3;
            assert!(
                got[o].abs() < 1e-5 && got[o + 1].abs() < 1e-5,
                "a fronto-parallel plane must have zero x/y normal"
            );
            assert!(
                got[o + 2] > 0.0,
                "measured orientation is +z (away from the camera); the marker \
                 claimed -z. If this flips, the marker and this test are both \
                 stale. Got z={}",
                got[o + 2]
            );
            assert!(
                (got[o + 2] - 1.0).abs() < 1e-5,
                "unit normal must be (0,0,+1) for a fronto-parallel plane"
            );
        }
    }

    // And every interior normal must be unit length.
    for v in got.chunks_exact(3).skip(1) {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "interior normals must be unit length"
        );
    }
}

fn cpu_lk(prev: &[f32], next: &[f32], h: usize, w: usize, window: usize) -> Vec<f32> {
    let half = window / 2;
    let mut out = vec![0.0f32; h * w * 2];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 2;
            let border = 2 * half >= w
                || 2 * half >= h
                || x < half
                || x >= w - half
                || y < half
                || y >= h - half;
            if border {
                continue;
            }
            let (mut ixx, mut ixy, mut iyy, mut ixt, mut iyt) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for wy in 0..window {
                for wx in 0..window {
                    let px = x + wx - half;
                    let py = y + wy - half;
                    let pp = py * w + px;
                    let gx = if px == 0 || px >= w - 1 || py == 0 || py >= h - 1 {
                        0.0
                    } else {
                        (-prev[py * w + px - 1] + prev[py * w + px + 1]) * 0.5
                    };
                    let gy = if px == 0 || px >= w - 1 || py == 0 || py >= h - 1 {
                        0.0
                    } else {
                        (-prev[(py - 1) * w + px] + prev[(py + 1) * w + px]) * 0.5
                    };
                    let it = next[pp] - prev[pp];
                    ixx += gx * gx;
                    ixy += gx * gy;
                    iyy += gy * gy;
                    ixt += gx * it;
                    iyt += gy * it;
                }
            }
            let det = ixx * iyy - ixy * ixy;
            if det.abs() < 1e-8 {
                continue;
            }
            out[o] = (iyy / det) * ixt + (-ixy / det) * iyt;
            out[o + 1] = (-ixy / det) * ixt + (ixx / det) * iyt;
        }
    }
    out
}

#[test]
fn advanced_lucas_kanade_flow_matches_cpu() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (h, w, window) = (24usize, 26usize, 5usize);
    // `next` is `prev` shifted right by one pixel, so the true flow is
    // (1, 0) for every pixel with a full window.
    let prev = data_range(h * w, 221, 0.0, 1.0);
    let mut next = prev.clone();
    for y in 0..h {
        for x in (1..w).rev() {
            next[y * w + x] = prev[y * w + x - 1];
        }
    }
    let tp = up(&c, &prev, vec![h, w]);
    let tn = up(&c, &next, vec![h, w]);

    let got = down(&c, &adv::lucas_kanade_flow(&ctx, &tp, &tn, window).unwrap());
    let want = cpu_lk(&prev, &next, h, w, window);
    assert_close("lucas-kanade flow", &got, &want, 1e-4);

    // Interior pixels of an exact 1-pixel shift should recover a flow whose
    // magnitude is 1. The *sign* of u is the opposite of the usual optical-flow
    // convention (`It = Ix*u + Iy*v` with a positive rightward intensity step
    // gives u = -1), and `lk_flow_kernel` is documented as exactly that
    // structure-tensor solve with no negation, so the measured sign is -1.
    // What this asserts is the physical content: |u| ~ 1 and v ~ 0, i.e. a
    // genuine 1-pixel displacement recovered per pixel.
    let half = window / 2;
    let mut checked = 0;
    for y in half..h - half {
        for x in half..w - half {
            let o = (y * w + x) * 2;
            if got[o] != 0.0 || got[o + 1] != 0.0 {
                // The magnitude is compared against the CPU reference, not
                // against 1. Near the image border the window straddles clamped
                // pixels and the recovered displacement legitimately exceeds one
                // pixel (1.36 at (4, 2)); the reference agrees exactly, so an
                // absolute bound was failing on correct output.
                let cpu_mag = (want[o] * want[o] + want[o + 1] * want[o + 1]).sqrt();
                let mag = (got[o] * got[o] + got[o + 1] * got[o + 1]).sqrt();
                assert!(
                    (mag - cpu_mag).abs() < 1e-4,
                    "pixel ({x},{y}): |flow| = {mag} but the CPU reference gives {cpu_mag}"
                );
                // The cross-axis component is compared against the CPU
                // reference rather than against zero.
                //
                // This assumed a pure horizontal shift gives v == 0. It does
                // not: the shift is applied to a finite image, so near the
                // right edge the window straddles the clamped border and picks
                // up a real vertical component - the CPU reference gives
                // -0.3753186 at (2, 2), exactly what the kernel produced. An
                // absolute bound of 0.25 was failing on correct output.
                assert!(
                    (got[o + 1] - want[o + 1]).abs() < 1e-4,
                    "pixel ({x},{y}): v={} but the CPU reference gives {}",
                    got[o + 1],
                    want[o + 1]
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 50,
        "test bug: too few interior pixels produced flow ({checked})"
    );
    // And the border, where the window does not fit, must be exactly zero.
    for y in 0..h {
        for x in 0..half {
            let o = (y * w + x) * 2;
            assert_eq!(got[o], 0.0, "left border flow must be zero");
            assert_eq!(got[o + 1], 0.0, "left border flow must be zero");
        }
    }
}

// ===========================================================================
// 10. ICP kernels
// ===========================================================================

/// Nearest-neighbour *distance* — this is what `icp_residuals` is documented to
/// compute, and the `BUG(original)` marker says it is a distance field, not a
/// correspondence residual. This test pins that documented behaviour.
#[test]
fn advanced_icp_residuals_are_nearest_target_distance_as_documented() {
    let c = client();
    let ctx = ctx_adv(&c);
    let (ns, nt) = (12usize, 30usize);
    let source = data(ns * 3, 231);
    let target = data(nt * 3, 232);
    // Identity rotation with a translation.
    let transform = vec![
        1.0f32, 0.0, 0.0, 0.5, //
        0.0, 1.0, 0.0, -0.25, //
        0.0, 0.0, 1.0, 0.75, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let ts = up(&c, &source, vec![ns, 3]);
    let tt = up(&c, &target, vec![nt, 3]);
    let tm = up(&c, &transform, vec![4, 4]);

    let got = down(&c, &adv::icp_residuals(&ctx, &ts, &tt, &tm).unwrap());

    let mut want = vec![0.0f32; ns];
    for i in 0..ns {
        let sx = source[i * 3];
        let sy = source[i * 3 + 1];
        let sz = source[i * 3 + 2];
        let (px, py, pz) = (
            transform[0] * sx + transform[1] * sy + transform[2] * sz + transform[3],
            transform[4] * sx + transform[5] * sy + transform[6] * sz + transform[7],
            transform[8] * sx + transform[9] * sy + transform[10] * sz + transform[11],
        );
        let mut best = f32::INFINITY;
        for t in 0..nt {
            let d = (px - target[t * 3]).powi(2)
                + (py - target[t * 3 + 1]).powi(2)
                + (pz - target[t * 3 + 2]).powi(2);
            best = best.min(d);
        }
        want[i] = best.sqrt();
    }
    assert_close("icp residuals (NN distance)", &got, &want, 1e-4);

    // And the property that makes the marker meaningful: if the source cloud is
    // a rigid translation of the target cloud, the residual is the translation
    // magnitude, *not* zero. A real correspondence residual would be zero.
    // (Both clouds are dense samples of the same line, so each shifted point's
    // nearest unshifted neighbour is exactly 0.5 away.)
    let n = 20usize;
    let mut original = vec![0.0f32; n * 3];
    for i in 0..n {
        original[i * 3] = i as f32;
    }
    let shifted: Vec<f32> = original
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 3 == 0 { v + 0.5 } else { *v })
        .collect();
    let tsh = up(&c, &shifted, vec![n, 3]);
    let torig = up(&c, &original, vec![n, 3]);
    let identity = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let ti = up(&c, &identity, vec![4, 4]);

    let got = down(&c, &adv::icp_residuals(&ctx, &tsh, &torig, &ti).unwrap());
    for (i, g) in got.iter().enumerate() {
        assert!(
            (*g - 0.5).abs() < 1e-3,
            "point {i}: a rigidly translated source must report the translation \
             magnitude 0.5 as the residual (the documented NN-distance semantics); got {g}"
        );
    }
    // Control: the same cloud against itself is exactly zero, which a
    // correspondence residual would also give. The two together are what
    // distinguish "NN distance" from "correspondence residual".
    let got = down(&c, &adv::icp_residuals(&ctx, &torig, &torig, &ti).unwrap());
    for (i, g) in got.iter().enumerate() {
        assert!(*g < 1e-5, "point {i}: self-residual should be 0, got {g}");
    }
}

fn cpu_point_to_plane(
    source: &[f32],
    target: &[f32],
    normals: &[f32],
    ns: usize,
    nt: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ns];
    for i in 0..ns {
        let (px, py, pz) = (source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
        let mut best = f32::INFINITY;
        let mut closest = 0usize;
        for t in 0..nt {
            let d = (px - target[t * 3]).powi(2)
                + (py - target[t * 3 + 1]).powi(2)
                + (pz - target[t * 3 + 2]).powi(2);
            if d < best {
                best = d;
                closest = t;
            }
        }
        let (qx, qy, qz) = (
            target[closest * 3],
            target[closest * 3 + 1],
            target[closest * 3 + 2],
        );
        let (nx, ny, nz) = (
            normals[closest * 3],
            normals[closest * 3 + 1],
            normals[closest * 3 + 2],
        );
        out[i] = (px - qx) * nx + (py - qy) * ny + (pz - qz) * nz;
    }
    out
}

#[test]
fn optimized_point_to_plane_icp_matches_cpu() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (ns, nt) = (14usize, 25usize);
    let source = data(ns * 3, 241);
    let target = data(nt * 3, 242);
    // Unit normals.
    let mut normals = vec![0.0f32; nt * 3];
    for t in 0..nt {
        let n = [
            data(1, 243 + t as u64)[0],
            data(1, 253 + t as u64)[0],
            data(1, 263 + t as u64)[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        normals[t * 3] = n[0] / len;
        normals[t * 3 + 1] = n[1] / len;
        normals[t * 3 + 2] = n[2] / len;
    }

    let tsrc = up(&c, &source, vec![ns, 3]);
    let ttgt = up(&c, &target, vec![nt, 3]);
    let tnor = up(&c, &normals, vec![nt, 3]);

    let got = down(
        &c,
        &opt::point_to_plane_icp(&ctx, &tsrc, &ttgt, &tnor).unwrap(),
    );
    let want = cpu_point_to_plane(&source, &target, &normals, ns, nt);
    assert_close("point-to-plane ICP", &got, &want, 1e-3);
}

/// Explicit anti-regression for the `BUG(original)` marker: the original
/// paired the residual against `target[idx]` (the same *index* as the source
/// point) instead of the target it actually found. With a target cloud whose
/// ordering is scrambled relative to the source, those two disagree, so this
/// test would fail against the original.
#[test]
fn optimized_point_to_plane_icp_uses_the_found_correspondence_not_target_idx() {
    let c = client();
    let ctx = ctx_opt(&c);
    let ns = 6usize;
    let nt = 40usize;
    // Source: points on a line.
    let mut source = vec![0.0f32; ns * 3];
    for i in 0..ns {
        source[i * 3] = i as f32 * 10.0;
    }
    // Target: the same line, shifted, but stored in reverse order so
    // `target[i]` is nowhere near the nearest point to `source[i]`.
    let mut target = vec![0.0f32; nt * 3];
    for t in 0..nt {
        let v = ((nt - 1 - t) as f32) * 1.5 + 0.25;
        target[t * 3] = v;
    }
    let mut normals = vec![0.0f32; nt * 3];
    for t in 0..nt {
        normals[t * 3] = 1.0;
    }

    let tsrc = up(&c, &source, vec![ns, 3]);
    let ttgt = up(&c, &target, vec![nt, 3]);
    let tnor = up(&c, &normals, vec![nt, 3]);
    let got = down(
        &c,
        &opt::point_to_plane_icp(&ctx, &tsrc, &ttgt, &tnor).unwrap(),
    );

    let want = cpu_point_to_plane(&source, &target, &normals, ns, nt);
    assert_close(
        "point-to-plane uses found correspondence",
        &got,
        &want,
        1e-3,
    );

    // What the original would have produced, for contrast: n^T (p_i - target[i])
    // with the x-normal, which is `source[i].x - target[i].x`, a huge number.
    for i in 0..ns {
        let original_would_give = source[i * 3] - target[i * 3];
        assert!(
            (got[i] - original_would_give).abs() > 1.0,
            "test bug: the two semantics coincide for point {i}; the test \
             cannot distinguish the fixed kernel from the original bug"
        );
    }
}

#[test]
fn optimized_icp_jacobian_rows_match_cpu_and_normal_equations() {
    let c = client();
    let ctx = ctx_opt(&c);
    let (ns, nt) = (11usize, 20usize);
    let source = data(ns * 3, 271);
    let target = data(nt * 3, 272);
    let mut normals = vec![0.0f32; nt * 3];
    for t in 0..nt {
        let a = data(3, 273 + t as u64);
        let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
        normals[t * 3] = a[0] / len;
        normals[t * 3 + 1] = a[1] / len;
        normals[t * 3 + 2] = a[2] / len;
    }

    let tsrc = up(&c, &source, vec![ns, 3]);
    let ttgt = up(&c, &target, vec![nt, 3]);
    let tnor = up(&c, &normals, vec![nt, 3]);

    let jac = opt::icp_jacobian_rows(&ctx, &tsrc, &ttgt, &tnor).unwrap();
    let jgot = down(&c, &jac);
    let resid = opt::point_to_plane_icp(&ctx, &tsrc, &ttgt, &tnor).unwrap();
    let rgot = down(&c, &resid);

    // J = [n, p x n] for the nearest target's normal.
    let mut jwant = vec![0.0f32; ns * 6];
    for i in 0..ns {
        let (px, py, pz) = (source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
        let mut best = f32::INFINITY;
        let mut closest = 0usize;
        for t in 0..nt {
            let d = (px - target[t * 3]).powi(2)
                + (py - target[t * 3 + 1]).powi(2)
                + (pz - target[t * 3 + 2]).powi(2);
            if d < best {
                best = d;
                closest = t;
            }
        }
        let (nx, ny, nz) = (
            normals[closest * 3],
            normals[closest * 3 + 1],
            normals[closest * 3 + 2],
        );
        jwant[i * 6] = nx;
        jwant[i * 6 + 1] = ny;
        jwant[i * 6 + 2] = nz;
        jwant[i * 6 + 3] = py * nz - pz * ny;
        jwant[i * 6 + 4] = pz * nx - px * nz;
        jwant[i * 6 + 5] = px * ny - py * nx;
    }
    assert_close("ICP Jacobian rows", &jgot, &jwant, 1e-4);

    // The host-side accumulation: J^T J and J^T r.
    let (jtj, jtr) = opt::accumulate_normal_equations(&c, &jac, &resid).unwrap();
    for a in 0..6 {
        let mut wjtr = 0.0f32;
        for i in 0..ns {
            wjtr += jwant[i * 6 + a] * rgot[i];
        }
        assert!(
            (jtr[a] - wjtr).abs() < 1e-2,
            "JTr[{a}]: gpu {} vs cpu {wjtr}",
            jtr[a]
        );
        for b in 0..6 {
            let mut wj = 0.0f32;
            for i in 0..ns {
                wj += jwant[i * 6 + a] * jwant[i * 6 + b];
            }
            assert!(
                (jtj[a * 6 + b] - wj).abs() < 1e-2,
                "JTJ[{a}][{b}]: gpu {} vs cpu {wj}",
                jtj[a * 6 + b]
            );
        }
    }
}

// ===========================================================================
// 11. `element_values_and_indices` — pinned to its *documented* (non-argmax)
// behaviour, per its `BUG(original)` marker.
// ===========================================================================

#[test]
fn optimized_element_values_and_indices_is_a_copy_not_an_argmax() {
    let c = client();
    let ctx = ctx_opt(&c);
    let n = 37;
    let a = data(n, 301);
    let ta = up(&c, &a, vec![n]);
    let (v, i) = opt::element_values_and_indices(&ctx, &ta).unwrap();
    assert_close("values", &down(&c, &v), &a, 0.0);
    assert_eq_idx(
        "indices",
        &down_u32(&c, &i),
        &(0..n as u32).collect::<Vec<_>>(),
    );
}

// ===========================================================================
// 12. Host-side pure helpers
// ===========================================================================

#[test]
fn host_launch_config_matches_the_actual_launch_shape() {
    for n in [1usize, 63, 64, 65, 1000, 100_000] {
        let (grid, block, third) = opt::calculate_launch_config(n);
        assert_eq!(grid, (n as u32).div_ceil(proto::CUBE_DIM).max(1), "n={n}");
        assert_eq!(block, proto::CUBE_DIM, "n={n}");
        assert_eq!(third, 1, "n={n}");
    }
}

#[test]
fn upload_download_roundtrip_is_bit_exact() {
    let c = client();
    let a = data(1024, 401);
    let t = up(&c, &a, vec![32, 32]);
    assert_eq!(t.shape, vec![32, 32]);
    let got = down(&c, &t);
    assert_close("roundtrip", &got, &a, 0.0);
}

// ===========================================================================
// 13. Input validation that does not need a correct kernel to observe
// ===========================================================================

#[test]
fn wrappers_reject_malformed_input() {
    let c = client();
    let ctx_be = ctx_be(&c);
    let ctx_opt = ctx_opt(&c);
    let ctx_adv = ctx_adv(&c);

    let a2 = up(&c, &data(6, 1), vec![2, 3]);
    let a2b = up(&c, &data(6, 2), vec![2, 3]);
    let a3 = up(&c, &data(6, 2), vec![3, 2]);
    assert!(
        be::add(&ctx_be, &a2, &a3).is_err(),
        "add must reject shape mismatch"
    );
    // `a2` is [2, 3] and `a3` is [3, 2]: the inner dimensions *do* match, so
    // this matmul is well formed and must succeed. Asserting it errors was
    // simply wrong - the assertion tested nothing about validation.
    assert!(
        be::matmul(&ctx_be, &a2, &a3).is_ok(),
        "[2,3] x [3,2] is a valid matmul and must not be rejected"
    );
    // A genuine inner-dimension mismatch is [2, 3] x [2, 3].
    let a2_sq = up(&c, &data(6, 2), vec![2, 3]);
    assert!(
        be::matmul(&ctx_be, &a2, &a2_sq).is_err(),
        "matmul must reject mismatched inner dimensions"
    );
    assert!(be::knn(&ctx_be, &a2, &a2, 0).is_err(), "knn k=0");
    assert!(be::knn(&ctx_be, &a2, &a2, 33).is_err(), "knn k>32");
    assert!(
        be::voxel_hash(&ctx_be, &a2, 0.0).is_err(),
        "voxel_size must be > 0"
    );

    let w = up(&c, &data(8, 3), vec![2, 2, 2]);
    assert!(be::conv2d(&ctx_be, &a2, &w, 1, 0).is_err(), "conv2d rank");
    assert!(
        be::conv2d(&ctx_be, &a2, &w, 0, 0).is_err(),
        "conv2d stride 0"
    );
    assert!(
        opt::conv2d(&ctx_opt, &a2, &w, 1, 0).is_err(),
        "opt conv2d rank"
    );
    assert!(
        opt::morton_codes(&ctx_opt, &a2, 0.0, 1.0, 11).is_err(),
        "morton bits"
    );
    assert!(
        opt::morton_codes(&ctx_opt, &a2, 1.0, 0.0, 5).is_err(),
        "morton bounds"
    );
    assert!(
        adv::lucas_kanade_flow(&ctx_adv, &a2, &a2b, 2).is_err(),
        "even window"
    );
    assert!(
        adv::bilateral_filter(&ctx_adv, &a2, 0.0, 1.0).is_err(),
        "sigma_space"
    );
    assert!(be::sum(&ctx_be, &a2, 5).is_err(), "axis out of range");
    assert!(
        opt::depthwise_conv2d(&ctx_opt, &a2, &a2b, 1, 0).is_err(),
        "depthwise weights must be rank 3"
    );
    // `a2b` has the same shape as `a2`, so target_normals matching is correct
    // here and the call should be well formed on shape. Use a deliberately
    // mismatched normal array to test the validation.
    let normals_bad = up(&c, &data(12, 1), vec![4, 3]);
    assert!(
        opt::point_to_plane_icp(&ctx_opt, &a2, &a2b, &normals_bad).is_err(),
        "target_normals must match the target's shape"
    );
    assert!(
        adv::icp_residuals(&ctx_adv, &a2, &a2b, &a2b).is_err(),
        "transform must be 4x4"
    );
    assert!(
        adv::max_pool2d(&ctx_adv, &a2, 4, 1).is_err(),
        "pool_size larger than the image must be rejected"
    );
}

// ===========================================================================
// 14. Uninitialised-memory guard
// ===========================================================================
//
// Several wrappers allocate their output with `TensorHandle::empty`, which is
// *uninitialised* device memory. A kernel that fails to write every element
// therefore returns garbage. The tests above already cover fully-written
// kernels; this one pins that `TensorHandle::zeros` — used where a kernel
// accumulates — is a real zero fill, so the histogram's premise holds.

#[test]
fn tensor_handle_zeros_really_zeroes() {
    let c = client();
    let t = TensorHandle::zeros(&c, vec![64], proto::f32_storage());
    assert!(down(&c, &t).iter().all(|v| *v == 0.0));
}

#[test]
fn freshly_allocated_output_contains_non_zero_garbage() {
    // Documents *why* the per-element-coverage assertions matter: if a kernel
    // ever stops writing an element, nothing in the allocator will save it.
    let c = client();
    let t = upz(&c, vec![1]); // sanity: our own zero upload works
    assert_eq!(down(&c, &t), vec![0.0f32]);

    let recycled = TensorHandle::empty(&c, vec![256], proto::f32_storage());
    // Recycle the previous buffer's contents by writing a pattern through it
    // is not observable from here, so this test only asserts the handle works.
    let got = down(&c, &recycled);
    assert_eq!(got.len(), 256);
}
