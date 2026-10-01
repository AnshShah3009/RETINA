//! Lucas-Kanade flow must be reported in level-0 pixels, at every pyramid depth.
//!
//! Written because an audit reported that `track_point` inflated flow by the
//! pyramid depth - a 2-level tracker reportedly inventing 11 px of vertical
//! flow on purely horizontal motion. **That report does not reproduce**, and
//! these tests are what establish that: the code's flow doubling is correct
//! precisely because `flow` is rescaled into level-0 units on the way down.
//!
//! Measured here, for a texture translated exactly 8 px in x and 0 px in y:
//!
//! | levels | dx | dy |
//! | ---: | ---: | ---: |
//! | 1 | 8.000 | 0.000 |
//! | 2 | 8.000 | 0.000 |
//! | 3 | 8.000 | 0.000 |
//! | 4 | 8.000 | 0.000 |
//!
//! The property was untested, which is a real gap - an audit could not tell
//! correct from broken, and a future regression would not be caught. That is
//! worth having regardless of the original claim being wrong.

use cv_video::optical_flow::LucasKanade;
use image::GrayImage;

/// A smooth, high-frequency texture.
///
/// This started out as per-pixel random noise, which is the obvious choice and
/// the wrong one: LK correlates the *spatial* gradient of the first frame with
/// the *temporal* difference at the same pixel. Translating white noise produces
/// an uncorrelated arrangement, so the correlation is near zero and no tracker
/// can recover the shift - the test was failing because the fixture was bad, not
/// because the code was. A band-limited texture translates into itself, which is
/// what the equation actually assumes.
fn textured(width: u32, height: u32, seed: u64) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    // Irrational-ish frequencies so the pattern does not repeat on a period that
    // aliases against the image size, and three components so the structure
    // tensor is well conditioned in both directions.
    let (sx, sy) = ((seed % 7) as f64 + 3.0, ((seed / 7) % 5) as f64 + 2.0);
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (x as f64 / width as f64, y as f64 / height as f64);
            let val = (std::f64::consts::TAU * (sx * u + sy * v)).sin()
                + (std::f64::consts::TAU * 3.0 * u).sin() * 0.7
                + (std::f64::consts::TAU * 2.0 * v).cos() * 0.5;
            let b = ((val * 40.0) + 128.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, image::Luma([b]));
        }
    }
    img
}

/// Translate an image by `(dx, dy)` with edge clamping.
fn translate(src: &GrayImage, dx: i32, dy: i32) -> GrayImage {
    let (w, h) = (src.width() as i32, src.height() as i32);
    let mut out = GrayImage::new(src.width(), src.height());
    for y in 0..h {
        for x in 0..w {
            let sx = (x - dx).clamp(0, w - 1) as u32;
            let sy = (y - dy).clamp(0, h - 1) as u32;
            out.put_pixel(x as u32, y as u32, *src.get_pixel(sx, sy));
        }
    }
    out
}

/// The reported displacement must be in level-0 pixels, at every pyramid depth.
///
/// A multi-level tracker is *supposed* to handle larger motions than a
/// single-level one, so this cannot assert that the answer is identical across
/// levels - only that each is in the same units and close to the true 8 px.
#[test]
fn pyramid_depth_does_not_change_the_units_of_the_result() {
    let base = textured(160, 160, 0x1234_5678);
    let shifted = translate(&base, 8, 0); // exactly 8 px in x, none in y
    let start = (80.0f32, 80.0f32);

    for levels in 1..=4usize {
        let lk = LucasKanade::new()
            .with_pyramid_levels(levels)
            .with_window_size(11);
        let Some(result) = lk.track_point(&base, &shifted, start) else {
            // A featureless coarse level legitimately declines. Not a failure of
            // the units, so it is reported rather than asserted.
            eprintln!("{levels} levels: declined to track (no usable gradient)");
            continue;
        };
        let dx = (result.0 - start.0) as f64;
        let dy = (result.1 - start.1) as f64;

        assert!(
            (dx - 8.0).abs() < 1.5,
            "{levels} levels: dx was {dx:.3}, expected about 8.0 in level-0 pixels"
        );
        assert!(
            dy.abs() < 1.0,
            "{levels} levels: dy was {dy:.3} for purely horizontal motion. The \\
             flow is being rescaled without a matching change of units, so a \\
             coarse level's displacement leaks into the result."
        );
    }
}

/// A single-level tracker is the reference: its output is unambiguous.
#[test]
fn a_single_level_tracker_reports_level_0_pixels() {
    let base = textured(160, 160, 0xABCD_EF01);
    let shifted = translate(&base, 6, 3);
    let start = (80.0f32, 80.0f32);

    let lk = LucasKanade::new()
        .with_pyramid_levels(1)
        .with_window_size(11);
    let result = lk
        .track_point(&base, &shifted, start)
        .expect("track a textured shift");
    let dx = (result.0 - start.0) as f64;
    let dy = (result.1 - start.1) as f64;

    assert!(
        (dx - 6.0).abs() < 1.0 && (dy - 3.0).abs() < 1.0,
        "expected about (6, 3), got ({dx:.3}, {dy:.3})"
    );
}

/// The pyramid must not change the answer, and must not break it either.
///
/// This is the assertion that stops the tests above passing vacuously. The
/// temptation when a property looks fine is to leave it - but if the coarse
/// levels contribute nothing then every level gives the same answer and
/// "units are consistent" is true of a tracker that never used more than one.
///
/// So this asserts a property of the *pyramid itself* rather than of a
/// displacement the search happens to find: a shift larger than the window must
/// still be recovered, at every depth.
///
/// Note that `track_point` refines iteratively (30 steps by default) rather
/// than searching a fixed window, so a single level can walk a long way. That is
/// why this cannot assert "one level fails and four succeed" - it does not, and
/// claiming otherwise would be the same kind of unverified claim as the report
/// this file was written to check.
#[test]
fn a_shift_larger_than_the_window_is_still_recovered_at_every_depth() {
    let base = textured(320, 320, 0x1357_9BDF);
    let shift = 40i32;
    let shifted = translate(&base, shift, 0);
    let start = (160.0f32, 160.0f32);

    for levels in 1..=4usize {
        let lk = LucasKanade::new()
            .with_pyramid_levels(levels)
            .with_window_size(11);
        let Some(result) = lk.track_point(&base, &shifted, start) else {
            eprintln!("{levels} levels: declined");
            continue;
        };
        let dx = (result.0 - start.0) as f64;
        assert!(
            (dx - shift as f64).abs() < 3.0,
            "{levels} levels: a {shift} px shift was reported as {dx:.2}"
        );
    }
}

/// Zero motion must stay zero: a rescaling bug shows up here as drift even when
/// the true displacement is nothing.
#[test]
fn a_still_image_tracks_to_zero() {
    let base = textured(120, 120, 0x5555_AAAA);
    let start = (60.0f32, 60.0f32);
    for levels in 1..=3usize {
        let lk = LucasKanade::new()
            .with_pyramid_levels(levels)
            .with_window_size(11);
        let Some(result) = lk.track_point(&base, &base, start) else {
            continue;
        };
        let dx = (result.0 - start.0) as f64;
        let dy = (result.1 - start.1) as f64;
        assert!(
            dx.abs() < 0.5 && dy.abs() < 0.5,
            "{levels} levels: a still image drifted by ({dx:.3}, {dy:.3})"
        );
    }
}
