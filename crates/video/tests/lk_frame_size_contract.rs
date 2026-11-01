//! `LucasKanade` must not consume two frames of different sizes.
//!
//! A capture is a stream at one resolution, but `cv-videoio`'s PNG-sequence
//! backend reads whatever files a directory holds: point it at a directory with
//! two recordings at different resolutions (a re-encoded tail, a second export,
//! a re-scaled chunk) and it hands the caller a sequence whose frame size
//! changes mid-stream. `Farneback::compute` rejects such a pair with
//! `DimensionMismatch`.
//!
//! `LucasKanade` did not check at all, and the answer it gave was not merely
//! imprecise - it was fabricated. The pyramid is built per frame, so level `L`
//! of the previous frame is `prev / 2^L` and level `L` of the next frame is
//! `next / 2^L`: for different sizes those are two different scalings of the
//! scene, and a coordinate that names one point in the previous frame names
//! another point in the next. The bilinear sampler clamps to the smaller array,
//! so nothing panics and nothing reports a problem: on a completely static
//! scene, captured at 160x120 and then at 120x90, the tracker reported ~20 px
//! of motion that does not exist.
//!
//! `track_point_single_pass` was worse: it indexed the next frame with the
//! previous frame's bounds and panicked with `Image index (135, 95) out of
//! bounds (120, 90)`.

use cv_video::optical_flow::LucasKanade;
use image::{GrayImage, Luma};

/// A smooth, band-limited texture that translates into itself.
fn textured(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        let (u, v) = (x as f64 / width as f64, y as f64 / height as f64);
        let val = (std::f64::consts::TAU * (3.0 * u + 2.0 * v)).sin()
            + (std::f64::consts::TAU * 3.0 * u).sin() * 0.7
            + (std::f64::consts::TAU * 2.0 * v).cos() * 0.5;
        Luma([((val * 40.0) + 128.0).clamp(0.0, 255.0) as u8])
    })
}

/// The still scene at the two resolutions a mixed directory would deliver.
fn still_pair() -> (GrayImage, GrayImage) {
    let full = textured(160, 120);
    // The same picture, resampled: no motion anywhere in the scene.
    let small = image::imageops::resize(&full, 120, 90, image::imageops::FilterType::Triangle);
    (full, small)
}

#[test]
fn pyramidal_tracking_declines_frames_of_different_sizes() {
    let (prev, next) = still_pair();
    assert_ne!(prev.dimensions(), next.dimensions());

    let lk = LucasKanade::new()
        .with_pyramid_levels(3)
        .with_window_size(11);

    // The scene is static, so any displacement is invented. Before the fix this
    // returned `Some((47.27, 62.11))` for the point below - 20.6 px of motion
    // that is not in the input.
    for point in [(60.0f32, 45.0f32), (40.0, 30.0), (80.0, 60.0)] {
        assert_eq!(
            lk.track_point(&prev, &next, point),
            None,
            "a {point:?} feature cannot be tracked between a {}x{} frame and a {}x{} frame: \
             level L of each pyramid is a different scaling of the scene, so the same \
             coordinate names two different points",
            prev.width(),
            prev.height(),
            next.width(),
            next.height()
        );
    }

    // The batch entry point must agree with the single-point one.
    let points = [(60.0f32, 45.0f32), (40.0, 30.0)];
    assert_eq!(lk.track_points(&prev, &next, &points), vec![None, None]);
}

#[test]
fn single_pass_tracking_declines_instead_of_panicking() {
    let (prev, next) = still_pair();
    let lk = LucasKanade::new().with_window_size(11);

    // A window that fits inside the previous frame but not inside the next one
    // used to index the smaller image out of bounds and panic.
    for point in [(140.0f32, 100.0f32), (60.0, 45.0)] {
        assert_eq!(
            lk.track_point_single_pass(&prev, &next, point),
            None,
            "{point:?} is not trackable across frames of different sizes"
        );
    }
}

/// Control: the guard must not refuse frames that do match, or the two tests
/// above would pass on a tracker that never tracks anything.
#[test]
fn tracking_of_equal_sized_frames_is_unchanged() {
    let prev = textured(160, 120);
    let next = GrayImage::from_fn(160, 120, |x, y| {
        *prev.get_pixel((x as i32 - 8).clamp(0, 159) as u32, y)
    });
    let lk = LucasKanade::new()
        .with_pyramid_levels(3)
        .with_window_size(11);
    let start = (80.0f32, 60.0f32);
    let (x, y) = lk
        .track_point(&prev, &next, start)
        .expect("an 8 px shift between equal-sized frames is trackable");
    assert!(
        ((x - start.0) - 8.0).abs() < 1.5 && (y - start.1).abs() < 1.0,
        "expected about (+8, 0), got ({:.3}, {:.3})",
        x - start.0,
        y - start.1
    );
    assert!(lk.track_point_single_pass(&prev, &next, start).is_some());
}
