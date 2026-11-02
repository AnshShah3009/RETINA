//! `TemplateTracker::update` never reports a box it did not search for.
//!
//! The search window is the last position +- `search_radius`, clipped to the
//! positions where the template fits in the frame. When that intersection was
//! empty (the frame shrank, or the last position lies further than the radius
//! from any valid position) the loops did not run and `update` returned the
//! stale last position as a successful track. Measured: initialised at
//! (80, 80, 10, 10) in a 100 x 100 frame with radius 5, an update on a 50 x 50
//! frame returned `Ok((80, 80, 10, 10))`, a box entirely outside the frame.
//! It now reports the target lost.

use cv_video::tracking::{TemplateTracker, Tracker};
use image::{GrayImage, Luma};

/// A frame with a bright 10 x 10 square at (x, y) on a dark background.
fn frame(w: u32, h: u32, x: u32, y: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |px, py| {
        let inside = px >= x && px < x + 10 && py >= y && py < y + 10;
        Luma([if inside {
            220
        } else {
            20 + ((px * 7 + py * 3) % 11) as u8
        }])
    })
}

#[test]
fn an_empty_search_window_reports_the_target_lost() {
    let mut tracker = TemplateTracker::new(5);
    tracker
        .init(&frame(100, 100, 80, 80), (80, 80, 10, 10))
        .unwrap();

    let result = tracker.update(&frame(50, 50, 20, 20));
    assert!(
        result.is_err(),
        "no position within 5 px of (80, 80) fits a 10 x 10 template in a 50 x 50 frame, \
         but update returned {result:?}"
    );
}

#[test]
fn a_window_clipped_by_the_frame_still_tracks() {
    // Control: near the border the window is clipped, not empty.
    let mut tracker = TemplateTracker::new(5);
    tracker
        .init(&frame(100, 100, 88, 88), (88, 88, 10, 10))
        .unwrap();
    let (x, y, w, h) = tracker.update(&frame(100, 100, 90, 90)).unwrap();
    assert_eq!((x, y, w, h), (90, 90, 10, 10));
    assert!(x + w <= 100 && y + h <= 100);
}
