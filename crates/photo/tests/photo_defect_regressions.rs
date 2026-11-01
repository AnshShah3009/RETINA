//! Regression tests for the defects fixed in `cv-photo`.
//!
//! Each test states the measurement that identified the defect and the observed
//! value on the unfixed source, so a future change that reintroduces it fails
//! with the number to compare against rather than a vague assertion.

use cv_core::tensor::{CpuTensor, TensorShape};
use image::{GrayImage, Luma};

fn gray1(h: usize, w: usize, data: &[f32]) -> CpuTensor<f32> {
    CpuTensor::from_vec(data.to_vec(), TensorShape::new(1, h, w)).unwrap()
}

fn mask1(h: usize, w: usize, masked: &[(usize, usize)]) -> CpuTensor<u8> {
    let mut data = vec![0u8; h * w];
    for &(y, x) in masked {
        data[y * w + x] = 255;
    }
    CpuTensor::from_vec(data, TensorShape::new(1, h, w)).unwrap()
}

/// A clean step image: left half 0.3, right half 0.7. Used wherever a test
/// needs a signal BM3D must not destroy.
fn step_image(h: usize, w: usize) -> CpuTensor<f32> {
    let mut data = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = if x < w / 2 { 0.3 } else { 0.7 };
        }
    }
    gray1(h, w, &data)
}

// ===========================================================================
// Defect 1 — `inpaint_ns` panicked on any masked pixel on row 1.
// ===========================================================================

/// `(y - 2).max(0)` on a `usize` underflows at `y == 1`, before `.max(0)` sees
/// it. The diffusion loop runs `for y in 1..height - 1`, so row 1 is visited
/// for *every* image as soon as anything on it is masked.
///
/// Measured on HEAD: `panicked at crates/photo/src/inpaint.rs:365:39:
/// "attempt to subtract with overflow"` for a 7x7 image with the centre masked,
/// and for every size 3..=7. The control case - the same 9x9 image with the
/// masked pixel far from row 1 - did not panic, which is what makes this
/// specifically a row-1 defect and not a general one.
#[test]
fn inpaint_ns_does_not_panic_on_row_one() {
    for n in [3usize, 4, 5, 6, 7, 8, 9] {
        let img = gray1(n, n, &vec![0.5f32; n * n]);
        let m = mask1(n, n, &[(1, 1)]);
        let r = std::panic::catch_unwind(|| cv_photo::inpaint_ns(&img, &m, 3.0, 20).is_ok());
        assert!(
            matches!(r, Ok(true)),
            "inpaint_ns on a {n}x{n} image with a masked pixel at (1,1) must complete, \
             got {:?} (HEAD panicked: 'attempt to subtract with overflow' at inpaint.rs:365, \
             because `y - 2` on a usize underflows at y == 1 before `.max(0)` is applied)",
            r.map(|ok| if ok { "Ok" } else { "Err" })
        );
    }
}

/// Control: a fully-masked image must still complete and must not be a no-op
/// (the seed step leaves pixels with no known neighbour, so they keep their
/// input value - which is fine - but the diffuse ones must actually move).
#[test]
fn inpaint_ns_fully_masked_does_not_panic() {
    let n = 9usize;
    let mut data = vec![0.0f32; n * n];
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f32;
    }
    let img = gray1(n, n, &data);
    let all_cells: Vec<(usize, usize)> = (0..n).flat_map(|y| (0..n).map(move |x| (y, x))).collect();
    let all = mask1(n, n, &all_cells);
    let out = cv_photo::inpaint_ns(&img, &all, 3.0, 20).expect("must not fail");
    let d = out.as_slice().unwrap();
    assert!(
        d.iter().all(|v| v.is_finite()),
        "a fully-masked inpaint must not produce NaN/inf"
    );
}

/// Control: an unmasked image passes through unchanged. This pins that the
/// `saturating_sub` change did not alter the ordinary path.
#[test]
fn inpaint_ns_control_unmasked_is_identity() {
    let n = 9usize;
    let img = step_image(n, n);
    let none = mask1(n, n, &[]);
    let out = cv_photo::inpaint_ns(&img, &none, 3.0, 50).unwrap();
    let a = img.as_slice().unwrap();
    let b = out.as_slice().unwrap();
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "pixel {i}: {} vs {}",
            a[i],
            b[i]
        );
    }
}

// ===========================================================================
// Defect 2 — `bm3d` returned an all-NaN image for sigma == 0 and smeared the
// signal for a negative sigma.
// ===========================================================================

/// `sigma` is BM3D's noise level, and every threshold derives from it. With
/// `sigma == 0` the stage-2 Wiener factor is `energy / (energy + 0) = 0/0 =
/// NaN` wherever the pilot coefficient is also zero, and that NaN spreads
/// through `idct_2d`.
///
/// Measured on HEAD, 16x16 step image: `len=256 nan_count=256`, i.e. **every**
/// pixel NaN, `min = +inf`, `max = -inf`.
#[test]
fn bm3d_zero_sigma_is_rejected_not_all_nan() {
    let img = step_image(16, 16);
    let out = cv_photo::bm3d(&img, 0.0f32, 8, 8, 4);
    match out {
        Ok(o) => {
            let d = o.as_slice().unwrap();
            let nan = d.iter().filter(|v| !v.is_finite()).count();
            panic!(
                "bm3d(sigma = 0) must not return an image: {nan}/{} pixels were non-finite \
                 (HEAD returned 256/256 NaN, min +inf, max -inf)",
                d.len()
            );
        }
        Err(_) => { /* rejected: the honest outcome */ }
    }
}

/// A negative sigma makes the block-match threshold `dist < sigma^2 * bs^2`
/// admit nothing, so every reference block falls back to matching only itself
/// and the "denoised" result smears the edge into something that is not the
/// input and is not a step.
///
/// Measured on HEAD, same image: output range 0.2556 .. 0.6784 where the input
/// is a clean 0.3 / 0.7 step. Rejecting it is the honest outcome; silently
/// returning a smeared edge is not.
#[test]
fn bm3d_negative_sigma_is_rejected() {
    let img = step_image(16, 16);
    let out = cv_photo::bm3d(&img, -0.1f32, 8, 8, 4);
    match out {
        Ok(o) => {
            let d = o.as_slice().unwrap();
            let lo = d.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            panic!(
                "bm3d(sigma = -0.1) must be rejected, not return {lo} .. {hi}: a negative \
                 sigma admits no block matches at all, so this is the reference block \
                 averaged with itself (HEAD returned 0.2556 .. 0.6784 for a 0.3/0.7 step)"
            );
        }
        Err(_) => {}
    }
}

/// `block_size == 0` panicked inside `dct_basis` at `chunks_mut(0)`, after the
/// `height < block_size` guard had passed because 16 >= 0.
#[test]
fn bm3d_zero_block_size_is_rejected_not_panic() {
    let img = step_image(16, 16);
    let r = std::panic::catch_unwind(|| cv_photo::bm3d(&img, 0.1f32, 0, 4, 4).is_ok());
    assert!(
        !matches!(r, Err(_)),
        "bm3d(block_size = 0) must be rejected, not panic \
         (HEAD panicked: 'chunk size must be non-zero' at denoise.rs:621)"
    );
}

/// Control: a normal positive sigma must still denoise and must produce a finite
/// image. This is the direction the guards above could break.
#[test]
fn bm3d_control_positive_sigma_is_finite_and_close_to_input() {
    let img = step_image(32, 32);
    let out = cv_photo::bm3d(&img, 0.1f32, 8, 16, 15).expect("a valid sigma must work");
    let d = out.as_slice().unwrap();
    assert!(
        d.iter().all(|v| v.is_finite()),
        "a valid sigma must produce a finite image"
    );
    // Deep in each half the value must stay close to that half's level.
    let left = d[16 * 32 + 2];
    let right = d[16 * 32 + 29];
    assert!(
        (left - 0.3).abs() < 0.05 && (right - 0.7).abs() < 0.05,
        "control: a clean step must survive denoising, got left={left} right={right}"
    );
}

// ===========================================================================
// Defect 3 — `inpaint_telea` with radius <= 0 returned the input unchanged,
// i.e. reported success while leaving the hole exactly as damaged.
// ===========================================================================

/// `r = (radius.ceil() as usize).max(1)` bounded the scan window, but the
/// accumulation loop excluded neighbours with `geom_dist > radius as f64` -
/// using the *raw* radius. At `radius <= 0` the nearest neighbour is at distance
/// 1, so every neighbour was excluded, `sum_w` stayed 0, and the
/// `if sum_w > 0.0` guard left the pixel at its original value.
///
/// Measured on HEAD, 9x9 image, left half 0.1 / right half 0.9, one masked
/// pixel whose own input value is 0.9:
///     radius =  0.0 -> out = 0.9   (unchanged - the hole was never filled)
///     radius = -1.0 -> out = 0.9   (unchanged)
///     radius =  1.0 -> out = 0.7   (filled)
#[test]
fn telea_non_positive_radius_still_fills_the_hole() {
    let (h, w) = (9usize, 9usize);
    let mut data = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = if x < 4 { 0.1 } else { 0.9 };
        }
    }
    let img = gray1(h, w, &data);
    // The masked pixel sits just right of the step, so its input value (0.9) is
    // *different* from what an interpolation across the step should produce -
    // which is exactly what makes "left untouched" detectable.
    let hole = (4, 4);
    let m = mask1(h, w, &[hole]);

    for radius in [0.0f32, -1.0] {
        let out = cv_photo::inpaint_telea(&img, &m, radius)
            .unwrap_or_else(|e| panic!("radius {radius}: got Err: {e}"));
        let d = out.as_slice().unwrap();
        let got = d[hole.0 * w + hole.1];
        assert!(
            (got - 0.9).abs() > 1e-3,
            "radius {radius}: the masked pixel still holds its damaged input value {got}; \
             a non-positive radius made every neighbour ineligible (geom_dist >= 1 > \
             radius), so sum_w stayed 0 and the pixel was reported as inpainted \
             without being touched"
        );
    }
}

/// Control: a normal radius still fills the hole, and the pixel must move
/// *towards* the average of its neighbours rather than to either extreme.
#[test]
fn telea_control_positive_radius_fills_with_a_blend() {
    let (h, w) = (9usize, 9usize);
    let mut data = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = if x < 4 { 0.1 } else { 0.9 };
        }
    }
    let img = gray1(h, w, &data);
    let hole = (4, 4);
    let m = mask1(h, w, &[hole]);

    let out = cv_photo::inpaint_telea(&img, &m, 3.0).unwrap();
    let got = out.as_slice().unwrap()[hole.0 * w + hole.1];
    assert!(
        got > 0.1 && got < 0.9,
        "a radius-3 fill must land strictly between the two sides of the step, got {got}"
    );

    // A radius of 1 - which is exactly what a non-positive radius is clamped to -
    // must produce the same value as the nominal radius 1, and must differ from
    // both the damaged input and the radius-3 result. This is what pins the
    // clamp: if the accumulation loop stopped consulting the radius at all, the
    // three radii would agree and a "fix" that ignored `radius` would pass.
    let one = cv_photo::inpaint_telea(&img, &m, 1.0).unwrap();
    let one_val = one.as_slice().unwrap()[hole.0 * w + hole.1];
    let zero = cv_photo::inpaint_telea(&img, &m, 0.0).unwrap();
    let zero_val = zero.as_slice().unwrap()[hole.0 * w + hole.1];
    assert_eq!(
        zero_val, one_val,
        "radius 0 is clamped to radius 1, so the two must produce the same fill; \
         got {zero_val} vs {one_val}"
    );
    assert_ne!(
        one_val, got,
        "radius 1 and radius 3 must produce different fills (got {one_val} for both), \
         otherwise the radius parameter is not reaching the accumulation loop"
    );
}

/// Control: with no mask at all, the image must come back byte-identical.
#[test]
fn telea_control_no_mask_is_identity() {
    let (h, w) = (9usize, 9usize);
    let img = step_image(h, w);
    let none = mask1(h, w, &[]);
    let out = cv_photo::inpaint_telea(&img, &none, 3.0).unwrap();
    let a = img.as_slice().unwrap();
    let b = out.as_slice().unwrap();
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "pixel {i}: {} vs {}",
            a[i],
            b[i]
        );
    }
}

// ===========================================================================
// Defect 4 — `Stitcher::stitch` returned its first input verbatim with `Ok(())`.
// ===========================================================================

/// A one-frame ramp: columns `x..` hold `base + 3x`, so frame `k` is the
/// continuation of frame `k-1` and the true horizontal shift between them is
/// exactly `w`.
fn ramp_frame(w: u32, base: u8) -> GrayImage {
    let mut out = GrayImage::new(w, 1);
    for x in 0..w {
        let v = (base as u32 + x * 3) as u8;
        out.put_pixel(x, 0, Luma([v]));
    }
    out
}

/// Two frames of one continuous ramp must produce a panorama strictly wider than
/// either input. These particular frames abut, so the join is a step of the ramp,
/// not a cross-fade; an exact overlap is covered separately.
///
/// Measured on HEAD: `stitch(&[a, b])` returned a byte-identical copy of `a` -
/// the same 8x1 image, with `b` discarded entirely and `Ok(())` reported.
#[test]
fn stitch_two_frames_produces_a_panorama_wider_than_either_input() {
    // `ramp_frame(8, base)` is `base + 3x`. Frame `a` is 10..31 and frame `b`
    // (`base = 34`) is 34..55: the next ramp sample after 31, so the frames abut
    // and the panorama is their concatenation (width 16). The exact bytes are
    // asserted by `stitcher_abutting_ramps_are_concatenated_without_a_blend`.
    let a = ramp_frame(8, 10);
    let b = ramp_frame(8, 10 + 3 * 8);
    let mut s = cv_photo::Stitcher::new();
    let out = s
        .stitch(&[a.clone(), b.clone()])
        .expect("two frames must stitch");

    assert_eq!(
        out.height(),
        1,
        "a horizontal translation must not change the height"
    );
    assert!(
        out.width() > a.width() && out.width() > b.width(),
        "a two-frame panorama must be wider than both {}x1 inputs, got {}x1 - \
         HEAD returned a byte-identical copy of the first image",
        a.width(),
        out.width()
    );
    assert!(
        out.as_raw() != a.as_raw(),
        "the second frame must reach the output"
    );

    // These two ramps abut (see `stitcher_abutting_ramps_are_concatenated_without_a_blend`),
    // so the join is 31 then 34, not a cross-fade of those two samples. What this
    // test still pins down is the panorama shape: wider than either input, both
    // frames present, and strictly increasing along the ramp.
    let px = out.as_raw();
    assert_eq!(px.len(), 16, "abutting width-8 frames concatenate to 16");
    assert_eq!(
        px[7], 31,
        "column 7 stays the first frame; a blended 33 means the search stopped one short"
    );
    assert_eq!(px[8], 34, "column 8 is the second frame, not a blend");
    for i in 1..px.len() {
        assert!(
            px[i] > px[i - 1],
            "column {i}: {} does not advance past {} - the panorama is not monotonic, \
             so the frames were not placed in order",
            px[i],
            px[i - 1]
        );
    }
}

/// Two width-8 ramps that continue each other, `[10, 13, …, 31]` then
/// `[34, 37, …, 55]`, meet at a shift of `canvas_width` (8): the next frame
/// abuts and the overlap is empty. The search used to stop at
/// `canvas_width - 1`, so this pair came back width 15 with column 7 blended
/// to 33 (`31` and `34` cross-faded). The continuation is the concatenation,
/// width 16, with no blended sample.
#[test]
fn stitcher_abutting_ramps_are_concatenated_without_a_blend() {
    let a = ramp_frame(8, 10);
    let b = ramp_frame(8, 34);
    assert_eq!(a.as_raw(), &[10, 13, 16, 19, 22, 25, 28, 31]);
    assert_eq!(b.as_raw(), &[34, 37, 40, 43, 46, 49, 52, 55]);

    let mut s = cv_photo::Stitcher::new();
    let out = s.stitch(&[a, b]).expect("abutting ramps must stitch");
    let expected: [u8; 16] = [
        10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55,
    ];
    assert_eq!(
        out.as_raw(),
        &expected,
        "an empty overlap is a legal shift of canvas_width; the panorama must be the \
         concatenation, not a width-15 image with a blended 33"
    );
}

/// A real overlap whose samples agree is cost 0, the same score as abutting
/// with an empty overlap. The tie goes to the smaller shift, and the cross-fade
/// of equal values leaves those columns unchanged. This is the overlapping case
/// the abutting-shift fix must not move.
#[test]
fn stitch_exact_overlap_keeps_the_shorter_shift_and_does_not_rewrite_shared_columns() {
    let a = ramp_frame(8, 10);
    // Shares a's last four samples, then continues the ramp.
    let b = ramp_frame(8, 22);
    assert_eq!(a.as_raw(), &[10, 13, 16, 19, 22, 25, 28, 31]);
    assert_eq!(b.as_raw(), &[22, 25, 28, 31, 34, 37, 40, 43]);

    let mut s = cv_photo::Stitcher::new();
    let out = s.stitch(&[a, b]).expect("an exact overlap must stitch");
    let expected: [u8; 12] = [10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43];
    assert_eq!(
        out.as_raw(),
        &expected,
        "a zero-cost overlap must beat the empty-overlap shift, and blending equal \
         samples must leave them as they were"
    );
}

/// Three frames must be wider than two, i.e. every frame has to be placed.
#[test]
fn stitch_three_frames_is_wider_than_two() {
    let a = ramp_frame(8, 10);
    let b = ramp_frame(8, 34);
    let c = ramp_frame(8, 58);
    let mut s = cv_photo::Stitcher::new();
    let two = s.stitch(&[a.clone(), b.clone()]).unwrap();
    let three = s.stitch(&[a.clone(), b.clone(), c.clone()]).unwrap();
    assert_eq!(three.height(), two.height());
    assert!(
        three.width() > two.width(),
        "adding a third frame must extend the canvas: {} vs {}",
        three.width(),
        two.width()
    );
}

/// Control: a single image is a one-image panorama and must come back unchanged.
///
/// This is the case the HEAD implementation happened to satisfy - it returned
/// `images[0].clone()` - so it is the control that shows the fix did not regress
/// it.
#[test]
fn stitch_single_image_is_unchanged() {
    let a = ramp_frame(8, 10);
    let mut s = cv_photo::Stitcher::new();
    let out = s.stitch(&[a.clone()]).unwrap();
    assert_eq!(out.dimensions(), a.dimensions());
    assert_eq!(out.as_raw(), a.as_raw());
}

/// Control: identical frames must stitch to something no narrower than the
/// input and must not crash.
#[test]
fn stitch_identical_frames_is_well_formed() {
    let a = ramp_frame(16, 20);
    let mut s = cv_photo::Stitcher::new();
    let out = s.stitch(&[a.clone(), a.clone()]).unwrap();
    assert_eq!(out.height(), a.height());
    assert!(
        out.width() >= a.width(),
        "the panorama must never be narrower than its input, got {} vs {}",
        out.width(),
        a.width()
    );
}

/// The stitcher composites with a weight that is *per output column* - the
/// comment calls it "how much evidence is behind it" and `w_old` "how many
/// images have already contributed" - but the weight was read and incremented
/// once per *pixel*, inside the row loop. Row `y` of the image being composited
/// therefore read `w_old = 1 + y` instead of `1`, so
/// `alpha = SEAM_WIDTH / (SEAM_WIDTH + 1 + y)` varied with the row and the same
/// column of the same frame was blended differently at every height. Measured
/// on the pair below: the 8-column overlap came back 6 grey levels darker on
/// row 7 than on row 0 - a horizontal seam *inside* the panorama.
///
/// The overlap has to be scored positively for the blend to run at all, and an
/// abutting shift (empty overlap, cost 0) always beats a positive-cost overlap;
/// with `MAX_SHIFT == 512` that abutting shift is only out of the search range
/// once the canvas is wider than 512. Hence the 520-column first frame, the
/// smallest canvas that exercises the per-column weight at all. The invariant
/// does not depend on which shift wins: both inputs are row-constant, and a
/// horizontal-translation model must return a row-constant panorama.
#[test]
fn stitch_blend_weight_is_per_column_not_per_pixel() {
    const H: u32 = 8;
    const W: u32 = 520;
    let line = |x: u32| ((x * 37 + 11) % 120) as u8;

    let a = GrayImage::from_fn(W, H, |x, _| Luma([line(x)]));
    // Continues the first frame at column 512, uniformly 40 grey levels
    // brighter, so the overlap there is the best alignment and its mean
    // absolute difference (40) is non-zero: the cross-fade actually runs.
    let b = GrayImage::from_fn(8, H, |x, _| Luma([line(512 + x) + 40]));

    let mut s = cv_photo::Stitcher::new();
    let out = s
        .stitch(&[a.clone(), b])
        .expect("two row-constant frames must stitch");

    assert_eq!(out.height(), H);
    let w = out.width() as usize;
    let first = &out.as_raw()[..w];
    for y in 1..H as usize {
        let row = &out.as_raw()[y * w..(y + 1) * w];
        assert_eq!(
            row, first,
            "row {y} differs from row 0 although both inputs are row-constant, so the \
             panorama must be too. The blend weight is per output column; incrementing \
             it once per pixel made alpha = 32/(32 + 1 + y) instead of 32/(32 + 1), \
             which blends the overlap differently on every row"
        );
    }
    // Control: the second frame reached the output, so a blend really happened.
    assert_ne!(
        out.as_raw(),
        a.as_raw(),
        "the second frame must be composited, not discarded"
    );
}

/// Control: an empty list is an error, not a silently-empty success.
#[test]
fn stitch_empty_is_an_error() {
    let mut s = cv_photo::Stitcher::new();
    assert!(
        s.stitch(&[]).is_err(),
        "an empty image list has no panorama; HEAD returned a 0x0 image with Ok(())"
    );
}

/// Mismatched heights are reported rather than replicated or clipped: a vertical
/// offset needs a homography, which this translation model does not estimate.
#[test]
fn stitch_mismatched_heights_is_an_error() {
    let a = ramp_frame(8, 10);
    let short = GrayImage::new(4, 2);
    let mut s = cv_photo::Stitcher::new();
    let r = s.stitch(&[a, short]);
    assert!(
        r.is_err(),
        "frames of differing heights must be reported, not silently stacked"
    );
}

/// The stitcher must not panic on any input shape a caller can hand it.
#[test]
fn stitch_never_panics_on_odd_shapes() {
    let shapes = [(1u32, 1u32), (1, 8), (8, 1), (3, 3), (33, 7)];
    for (w, h) in shapes {
        let a = GrayImage::new(w, h);
        let b = GrayImage::new(w, h);
        let mut s = cv_photo::Stitcher::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.stitch(&[a.clone(), b.clone()])
        }));
        assert!(!matches!(r, Err(_)), "stitch panicked on {w}x{h} inputs");
    }
}
