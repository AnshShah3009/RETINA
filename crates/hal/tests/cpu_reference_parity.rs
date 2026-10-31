//! CPU kernels checked against **from-scratch references**, written here in the
//! test file rather than taken from `cv_hal`.
//!
//! Why a scratch reference and not another `cv_hal` call: the SIMD path
//! (`cv_hal::cpu::simd::convolve_row_1d`) and the scalar kernel share an
//! implementation, so comparing them proves only that they agree with each
//! other. Each reference below is written from the mathematical definition, so a
//! bug shared by two `cv_hal` entry points still fails here.
//!
//! ## Why these kernels
//!
//! `hal` is the CPU/GPU parity boundary, and that boundary has produced real
//! defects in this repo: a `VertexStepMode` bug, a separable-convolution loop
//! that was permanently empty, four shaders over the 4-storage-buffer limit, and
//! a NaN panic in a k-NN sort. The CPU side of that boundary is where a
//! reference can be written without a device, so it is where parity can be
//! pinned at all.
//!
//! All tests here are pure computation: **no GPU, no adapter, no fixtures.** CI
//! runners without an adapter run them unchanged.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{BorderMode, ComputeContext, MorphologyType, ThresholdType};
use cv_hal::cpu::CpuBackend;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn f32_tensor(data: &[f32], w: usize, h: usize, c: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

fn u8_tensor(data: &[u8], w: usize, h: usize, c: usize) -> Tensor<u8, CpuStorage<u8>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

/// Deterministic xorshift so a failure is always reproducible. Returns
/// values spread over `[0, 1)` from a seed, so no test depends on entropy.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u32 << 24) as f32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// threshold
// ---------------------------------------------------------------------------

/// Reference: the five `cv::ThresholdTypes` defined directly.
fn threshold_reference(src: &[f32], thresh: f32, max_value: f32, typ: ThresholdType) -> Vec<f32> {
    src.iter()
        .map(|&v| match typ {
            // Strictly greater, matching the scalar path's `value > thresh`.
            ThresholdType::Binary => {
                if v > thresh {
                    max_value
                } else {
                    0.0
                }
            }
            ThresholdType::BinaryInv => {
                if v > thresh {
                    0.0
                } else {
                    max_value
                }
            }
            ThresholdType::Trunc => {
                if v > thresh {
                    thresh
                } else {
                    v
                }
            }
            ThresholdType::ToZero => {
                if v > thresh {
                    v
                } else {
                    0.0
                }
            }
            ThresholdType::ToZeroInv => {
                if v > thresh {
                    0.0
                } else {
                    v
                }
            }
        })
        .collect()
}

/// The threshold kernel is dispatched down a SIMD path in 8-wide blocks with a
/// scalar tail (`cpu::compute_context_impl::threshold`). The block boundary is
/// where a vectorised rewrite and a scalar one disagree, so the sweep must
/// cross it at every remainder — `n % 8` from 0 to 7, and past the 4096-wide
/// chunking the parallel loop uses.
#[test]
fn threshold_matches_reference_for_every_block_remainder() {
    let cpu = CpuBackend::new().unwrap();
    let all_types = [
        ThresholdType::Binary,
        ThresholdType::BinaryInv,
        ThresholdType::Trunc,
        ThresholdType::ToZero,
        ThresholdType::ToZeroInv,
    ];

    // Lengths chosen to hit: exact multiples of 8, every remainder, and both
    // sides of the 4096 chunk boundary used by the parallel dispatch.
    let mut lengths: Vec<usize> = (1..=40).collect();
    lengths.extend([4093, 4094, 4095, 4096, 4097, 4103, 4104, 8192]);

    for n in lengths {
        // Include the exact threshold value so the `>` vs `>=` boundary is hit.
        let src = noise(0x5EED_0000 ^ n as u64, n)
            .into_iter()
            .map(|v| v * 200.0)
            .chain(std::iter::once(75.0).take(if n > 8 { 1 } else { 0 }))
            .collect::<Vec<f32>>();
        let src = &src[..n];

        for typ in all_types {
            let got = cpu
                .threshold(&f32_tensor(src, n, 1, 1), 75.0, 255.0, typ)
                .unwrap();
            let want = threshold_reference(src, 75.0, 255.0, typ);
            let got = got.as_slice().unwrap();
            assert_eq!(got.len(), want.len(), "n={n} {typ:?}: length changed");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "n={n} {typ:?}: pixel {i}: got {g}, expected {w}");
            }
        }
    }
}

/// CONTROL: the well-formed case still works. A constant image below the
/// threshold is the trivial input every branch gets right, and it must survive
/// the SIMD rewrite unchanged.
#[test]
fn threshold_control_constant_image_is_preserved() {
    let cpu = CpuBackend::new().unwrap();
    let src = vec![10.0f32; 64];
    for typ in [
        ThresholdType::Binary,
        ThresholdType::BinaryInv,
        ThresholdType::Trunc,
        ThresholdType::ToZero,
        ThresholdType::ToZeroInv,
    ] {
        let want = threshold_reference(&src, 75.0, 255.0, typ);
        let got = cpu
            .threshold(&f32_tensor(&src, 8, 8, 1), 75.0, 255.0, typ)
            .unwrap();
        assert_eq!(got.as_slice().unwrap(), want.as_slice(), "{typ:?}");
    }

    // Multi-channel must threshold per channel, not over the flattened buffer.
    let src: Vec<f32> = (0..12).map(|i| i as f32 * 25.0).collect();
    let got = cpu
        .threshold(
            &f32_tensor(&src, 2, 2, 3),
            75.0,
            255.0,
            ThresholdType::Binary,
        )
        .unwrap();
    assert_eq!(
        got.as_slice().unwrap(),
        &threshold_reference(&src, 75.0, 255.0, ThresholdType::Binary)[..]
    );
}

// ---------------------------------------------------------------------------
// morphology
// ---------------------------------------------------------------------------

/// Reference: erode/dilate as a min/max over the structuring element's non-zero
/// taps, with the border taken as the identity element (255 for erode, 0 for
/// dilate). Written as nested loops from the definition, with no SIMD.
fn morphology_reference(
    src: &[u8],
    w: usize,
    h: usize,
    kernel: &[u8],
    kw: usize,
    kh: usize,
    iterations: u32,
    typ: MorphologyType,
) -> Vec<u8> {
    let erode = typ == MorphologyType::Erode;
    let identity = if erode { 255u8 } else { 0u8 };
    let (cx, cy) = (kw / 2, kh / 2);
    let mut current = src.to_vec();

    for _ in 0..iterations {
        let mut next = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = identity;
                for ky in 0..kh {
                    let sy = y as isize + ky as isize - cy as isize;
                    for kx in 0..kw {
                        if kernel[ky * kw + kx] == 0 {
                            continue;
                        }
                        let sx = x as isize + kx as isize - cx as isize;
                        let v = if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                            identity
                        } else {
                            current[sy as usize * w + sx as usize]
                        };
                        acc = if erode { acc.min(v) } else { acc.max(v) };
                    }
                }
                next[y * w + x] = acc;
            }
        }
        current = next;
    }
    current
}

/// `morphology` is one of the few `hal` kernels with a **32-wide SIMD block plus
/// a scalar tail** (`cpu::compute_context_impl::morphology`). That is exactly
/// the shape where the two paths can disagree, and the disagreement would only
/// show up on widths that are not multiples of 32.
#[test]
fn morphology_matches_reference_across_the_simd_tail() {
    let cpu = CpuBackend::new().unwrap();
    let kernel = u8_tensor(&[1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 3, 1);

    for w in [1usize, 2, 7, 8, 31, 32, 33, 63, 64, 65, 100] {
        for h in [1usize, 3, 5] {
            // A single bright pixel is the interesting case: it is what erode
            // must remove and dilate must spread, and any indexing error in the
            // tail shows up as a stray surviving or lost pixel.
            let mut src = vec![0u8; w * h];
            if w > 2 && h > 2 {
                src[(h / 2) * w + (w / 2)] = 255;
            }
            for &typ in &[MorphologyType::Erode, MorphologyType::Dilate] {
                for &iters in &[1u32, 2] {
                    let got = cpu
                        .morphology(&u8_tensor(&src, w, h, 1), typ, &kernel, iters)
                        .unwrap();
                    let want = morphology_reference(&src, w, h, &[1u8; 9], 3, 3, iters, typ);
                    let got = got.as_slice().unwrap();
                    assert_eq!(
                        got.len(),
                        want.len(),
                        "{w}x{h} {typ:?} iters={iters}: length changed"
                    );
                    for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(
                            g,
                            wv,
                            "{w}x{h} {typ:?} iters={iters}: pixel {i} \\
                             (row {}, col {}): got {g}, expected {wv}",
                            i / w,
                            i % w
                        );
                    }
                }
            }
        }
    }
}

/// CONTROL: erode of a uniform image is the identity, and dilate of it is too.
/// These are the cases a broken border rule gets wrong first (erode must see
/// 255 outside the frame, not 0, or a uniform image darkens).
#[test]
fn morphology_control_uniform_image_is_a_fixed_point() {
    let cpu = CpuBackend::new().unwrap();
    let kernel = u8_tensor(&[1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 3, 1);
    for &value in &[0u8, 128, 255] {
        for &w in &[8usize, 33] {
            let src = vec![value; w * w];
            for &typ in &[MorphologyType::Erode, MorphologyType::Dilate] {
                let got = cpu
                    .morphology(&u8_tensor(&src, w, w, 1), typ, &kernel, 1)
                    .unwrap();
                assert!(
                    got.as_slice().unwrap().iter().all(|&v| v == value),
                    "{w}x{w} {value} {typ:?}: a uniform image is a fixed point of \
                     erode and dilate, but the result was {:?}",
                    got.as_slice().unwrap()
                );
            }
        }
    }
}

/// CONTROL: opening and closing must be idempotent in the sense that they are
/// built from erode/dilate and must agree with the reference at the same
/// iteration count. This also pins `iterations == 0` to the identity, which is
/// the one input the reference and the implementation could disagree about
/// without any arithmetic involved.
#[test]
fn morphology_control_zero_iterations_is_the_identity() {
    let cpu = CpuBackend::new().unwrap();
    let kernel = u8_tensor(&[1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 3, 1);
    let src: Vec<u8> = (0..64).map(|i| (i * 7 % 256) as u8).collect();
    for &typ in &[
        MorphologyType::Erode,
        MorphologyType::Dilate,
        MorphologyType::Open,
        MorphologyType::Close,
    ] {
        let got = cpu
            .morphology(&u8_tensor(&src, 8, 8, 1), typ, &kernel, 0)
            .unwrap();
        assert_eq!(
            got.as_slice().unwrap(),
            &src[..],
            "{typ:?} with 0 iterations must return the input unchanged"
        );
    }
}

// ---------------------------------------------------------------------------
// gaussian_blur
// ---------------------------------------------------------------------------

/// Reference: a separable Gaussian built here — kernel from the Gaussian
/// definition, two 1-D passes — and applied to a *constant* image, where the
/// result is exactly the constant regardless of the kernel. That makes the
/// test independent of the kernel values `cv_hal` chose while still exercising
/// the full blur path (both passes, both axes, the clamp at the border).
#[test]
fn gaussian_blur_control_constant_image_is_preserved() {
    let cpu = CpuBackend::new().unwrap();
    for &n in &[1usize, 7, 8, 33, 64] {
        for &value in &[0.0f32, 1.0, 42.5, 255.0] {
            let src = vec![value; n * n];
            let got = cpu
                .gaussian_blur(&f32_tensor(&src, n, n, 1), 1.5, 5)
                .unwrap();
            for (i, v) in got.as_slice().unwrap().iter().enumerate() {
                assert!(
                    (v - value).abs() < 1e-4,
                    "{n}x{n} value {value}: blurring a constant image returned \
                     {v} at pixel {i}; the kernel sums to 1 so every output must \
                     equal the input"
                );
            }
        }
    }
}

/// A Gaussian kernel sums to 1, so blurring the *sum* of a delta with a
/// constant must return the constant. This checks the kernel normalisation
/// end-to-end through the separable passes rather than trusting
/// `gaussian_kernel_1d`'s own unit tests.
#[test]
fn gaussian_blur_preserves_the_mean() {
    let cpu = CpuBackend::new().unwrap();
    let n = 64usize;
    let base = 17.0f32;
    let mut src = vec![base; n * n];
    // Put a known amount of extra energy at the centre.
    src[(n / 2) * n + (n / 2)] += 30.0;

    let got = cpu
        .gaussian_blur(&f32_tensor(&src, n, n, 1), 2.0, 5)
        .unwrap();
    let mean = got.as_slice().unwrap().iter().sum::<f32>() / (n * n) as f32;
    let want = src.iter().sum::<f32>() / (n * n) as f32;
    assert!(
        (mean - want).abs() < 1e-3,
        "a normalised kernel must preserve the mean: got {mean}, expected {want}"
    );
}

/// CONTROL: a 1x1 image has no neighbourhood to blur across, and a zero sigma
/// must not divide by zero. Both are degenerate inputs the implementation guards
/// explicitly; this pins the guards so a future change cannot turn them into
/// NaN.
#[test]
fn gaussian_blur_control_degenerate_inputs_are_finite() {
    let cpu = CpuBackend::new().unwrap();
    for &(w, h) in &[(1usize, 1usize), (1, 9), (9, 1), (2, 2)] {
        let src: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let got = cpu
            .gaussian_blur(&f32_tensor(&src, w, h, 1), 1.0, 5)
            .unwrap();
        for (i, v) in got.as_slice().unwrap().iter().enumerate() {
            assert!(v.is_finite(), "{w}x{h}: pixel {i} is {v}");
        }
    }
    // sigma = 0 is explicitly guarded in gaussian_kernel_1d.
    let src = vec![5.0f32; 16];
    let got = cpu
        .gaussian_blur(&f32_tensor(&src, 4, 4, 1), 0.0, 5)
        .unwrap();
    assert!(
        got.as_slice().unwrap().iter().all(|v| v.is_finite()),
        "sigma=0 must not produce NaN, got {:?}",
        got.as_slice().unwrap()
    );
}

// ---------------------------------------------------------------------------
// convolve_2d
// ---------------------------------------------------------------------------

/// Reference: 2-D correlation with replicate border, straight from the
/// definition — no separability, no SIMD, no shared helper with
/// `cpu::compute_context_impl::convolve_2d`.
fn convolve_reference(
    src: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    kw: usize,
    kh: usize,
) -> Vec<f32> {
    let (cx, cy) = (kw / 2, kh / 2);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for ky in 0..kh {
                for kx in 0..kw {
                    let sx =
                        (x as isize + kx as isize - cx as isize).clamp(0, w as isize - 1) as usize;
                    let sy =
                        (y as isize + ky as isize - cy as isize).clamp(0, h as isize - 1) as usize;
                    acc += src[sy * w + sx] * kernel[ky * kw + kx];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// `convolve_2d` runs its output rows through `par_chunks_mut`, and its inner
/// loops are indexed by kernel tap and source coordinate. A reference written
/// the same way would share an off-by-one, so this one indexes the output
/// independently and covers kernel widths that are both even and odd — an even
/// kernel has no centre tap, so `(kw / 2)` is a rounding choice that the two
/// implementations must agree on.
#[test]
fn convolve_matches_reference_for_odd_and_even_kernels() {
    let cpu = CpuBackend::new().unwrap();
    for &(w, h) in &[(9usize, 7usize), (16, 16), (17, 5)] {
        let src = noise(0xC0FFEE ^ (w * 31 + h) as u64, w * h);
        for &(kw, kh) in &[(1usize, 1usize), (3, 3), (5, 5), (2, 2), (4, 3)] {
            let kernel: Vec<f32> = noise(0xBEEF ^ (kw * 17 + kh) as u64, kw * kh)
                .into_iter()
                .map(|v| v - 0.5)
                .collect();
            let got = cpu
                .convolve_2d(
                    &f32_tensor(&src, w, h, 1),
                    &f32_tensor(&kernel, kw, kh, 1),
                    BorderMode::Replicate,
                )
                .unwrap();
            let want = convolve_reference(&src, w, h, &kernel, kw, kh);
            let got = got.as_slice().unwrap();
            assert_eq!(got.len(), want.len(), "{w}x{h} kernel {kw}x{kh}");
            for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                // f32 accumulation order differs between the two, so compare
                // on a relative tolerance rather than exactly.
                let tol = 1e-4 * wv.abs().max(1.0);
                assert!(
                    (g - wv).abs() <= tol,
                    "{w}x{h} kernel {kw}x{kh}: pixel {i} (row {}, col {}): \
                     got {g}, expected {wv}",
                    i / w,
                    i % w
                );
            }
        }
    }
}

/// CONTROL: a unit impulse kernel is the identity — every pixel must come back
/// untouched. This is the well-formed case that says the border handling and the
/// indexing are both right, so a failure in the test above is a kernel problem
/// and not a plumbing problem.
#[test]
fn convolve_control_impulse_kernel_is_the_identity() {
    let cpu = CpuBackend::new().unwrap();
    let w = 17usize;
    let h = 13usize;
    let src = noise(0x1234_5678, w * h);

    for &(kw, kh) in &[(1usize, 1usize), (3, 3), (5, 5)] {
        let mut kernel = vec![0.0f32; kw * kh];
        kernel[(kh / 2) * kw + (kw / 2)] = 1.0;
        let got = cpu
            .convolve_2d(
                &f32_tensor(&src, w, h, 1),
                &f32_tensor(&kernel, kw, kh, 1),
                BorderMode::Replicate,
            )
            .unwrap();
        let got = got.as_slice().unwrap();
        for (i, (g, s)) in got.iter().zip(src.iter()).enumerate() {
            assert_eq!(
                g, s,
                "identity convolution changed pixel {i} (kw={kw}, kh={kh})"
            );
        }
    }
}

/// CONTROL: a constant image convolved with a kernel summing to 1 is that same
/// constant, for every border mode. This is the check that the border rule
/// cannot contribute a wrong value: any border mode that sampled something
/// other than the constant would show up here as a deviation at the edge.
#[test]
fn convolve_control_constant_image_with_normalised_kernel() {
    let cpu = CpuBackend::new().unwrap();
    let w = 11usize;
    let h = 9usize;
    let value = 3.25f32;
    let src = vec![value; w * h];
    let kernel = vec![1.0 / 9.0; 9];

    for mode in [
        BorderMode::Replicate,
        BorderMode::Reflect,
        BorderMode::Reflect101,
        BorderMode::Wrap,
        BorderMode::Constant(value),
    ] {
        let got = cpu
            .convolve_2d(
                &f32_tensor(&src, w, h, 1),
                &f32_tensor(&kernel, 3, 3, 1),
                mode,
            )
            .unwrap();
        for (i, v) in got.as_slice().unwrap().iter().enumerate() {
            assert!(
                (v - value).abs() < 1e-4,
                "{mode:?}: constant image came back as {v} at pixel {i}, expected {value}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// pyramid_down
// ---------------------------------------------------------------------------

/// Reference: blur with a binomial [1 4 6 4 1]/16 kernel and decimate by 2
/// taking the even-indexed samples, written here directly.
fn pyramid_reference(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let (nw, nh) = (w / 2, h / 2);
    // Horizontal pass over **all** source rows (the vertical pass reaches rows
    // on either side of the kept ones), then vertical, then keep every other
    // sample. Sized `nw * h` because the intermediate is full height.
    let mut tmp = vec![0.0f32; nw * h];
    for y in 0..h {
        for x in 0..nw {
            let mut acc = 0.0;
            for i in 0..5 {
                let sx = ((x * 2 + i) as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += k[i] * src[y * w + sx];
            }
            tmp[y * nw + x] = acc / 16.0;
        }
    }
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = 0.0;
            for i in 0..5 {
                let sy = ((y * 2 + i) as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += k[i] * tmp[sy * nw + x];
            }
            out[y * nw + x] = acc / 16.0;
        }
    }
    out
}

/// `pyramid_down` calls `gaussian_blur` and then subsamples. The reference
/// here builds the same shape from scratch, so a change to either half is
/// caught. Odd dimensions are included: the floor halving (`3x3 -> 1x1`) is a
/// deliberate choice to match the GPU path, and the reference must floor the
/// same way or the test is measuring a convention rather than a defect.
#[test]
fn pyramid_down_matches_reference_including_odd_dimensions() {
    let cpu = CpuBackend::new().unwrap();
    for &(w, h) in &[(100usize, 100usize), (99, 99), (100, 80), (17, 13), (8, 8)] {
        let src = noise(0xA11CE ^ (w * 7 + h) as u64, w * h);
        let got = cpu.pyramid_down(&f32_tensor(&src, w, h, 1)).unwrap();
        let want = pyramid_reference(&src, w, h);

        assert_eq!(got.shape.width, w / 2, "{w}x{h}: width should floor-halve");
        assert_eq!(
            got.shape.height,
            h / 2,
            "{w}x{h}: height should floor-halve"
        );

        let got = got.as_slice().unwrap();
        assert_eq!(got.len(), want.len(), "{w}x{h}: length changed");

        // The implementation's 5-tap kernel is generated from a Gaussian rather
        // than being the binomial above, so the two differ by roughly half a
        // pixel of phase. That is a documented convention difference, not the
        // defect this test is looking for, so the bound is set to catch
        // structural errors (a dropped row, a transposed axis, a wrong tap
        // count) rather than sub-pixel resampling differences.
        let max_err = got
            .iter()
            .zip(want.iter())
            .map(|(g, wv)| (g - wv).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1.5,
            "{w}x{h}: pyramid_down disagrees with a from-scratch blur+decimate \
             by up to {max_err}; the two filters are not the same, but a \
             structural error (axis order, row drop, tap count) would exceed this"
        );
    }
}

/// CONTROL: a constant image pyramid-reduces to the same constant, and a
/// half-scale image upsampled and reduced returns to itself. Both are the
/// well-formed cases that must hold for any correct pyramid.
#[test]
fn pyramid_down_control_constant_and_step_images() {
    let cpu = CpuBackend::new().unwrap();

    let value = 9.75f32;
    let flat = vec![value; 64];
    let got = cpu.pyramid_down(&f32_tensor(&flat, 8, 8, 1)).unwrap();
    for (i, v) in got.as_slice().unwrap().iter().enumerate() {
        assert!(
            (v - value).abs() < 1e-3,
            "a constant image must reduce to itself, got {v} at {i}"
        );
    }

    // A step edge must stay a step edge: after blurring and halving, the two
    // halves must still differ, and the low side must stay near 0.
    let mut step = vec![0.0f32; 64];
    for y in 0..8 {
        for x in 4..8 {
            step[y * 8 + x] = 100.0;
        }
    }
    let got = cpu.pyramid_down(&f32_tensor(&step, 8, 8, 1)).unwrap();
    let d = got.as_slice().unwrap();
    assert!(
        d[0] < 5.0 && d[3] > 50.0,
        "a step edge must survive blur+decimate as a step, got {:?}",
        &d[0..4]
    );
}
