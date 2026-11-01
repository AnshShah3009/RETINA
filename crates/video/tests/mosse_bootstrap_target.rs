//! MOSSE must recognise the frame it was trained on.
//!
//! `MosseTracker::init` bootstraps its filter from the initial patch plus seven
//! slightly displaced copies. Each displaced copy is extracted from a window
//! recentred by `(offset_x, offset_y)`, so the object moves inside the patch and
//! the ideal correlation peak moves with it. The regression target, however,
//! stayed pinned at `(0, 0)` for every copy.
//!
//! A correlation filter solves `H* = Σ G_i X_i* / Σ X_i X_i*`. Pairing a shifted
//! spectrum `X_i` with an unshifted `G` makes the samples cancel instead of
//! reinforce, so the filter is destroyed: the response for the training frame was
//! ~2.5e-5 (peak at a random index) instead of the target (peak ~1 at index 0),
//! and `update` reported the target lost (`None`) on the exact frame passed to
//! `init` - the one input on which a correctly learned filter has no excuse.
//!
//! The target must be shifted by the same displacement as the patch content.

use cv_core::{CpuTensor, TensorShape};
use cv_video::tracking::{BoundingBox, MosseTracker};

/// A deterministic, strongly textured 40x40 frame (any real image has texture;
/// a constant patch carries no information and is not what this tests).
fn textured_frame(width: usize, height: usize) -> CpuTensor<f32> {
    let data: Vec<f32> = (0..width * height)
        .map(|i| ((i * 37) % 255) as f32)
        .collect();
    CpuTensor::<f32>::from_vec(data, TensorShape::new(1, height, width)).unwrap()
}

#[test]
fn a_self_trained_filter_recognizes_the_frame_it_was_initialised_on() {
    let frame = textured_frame(40, 40);
    let bbox = BoundingBox::new(10.0, 10.0, 20.0, 20.0);

    let mut tracker = MosseTracker::new(0.125);
    tracker.init(&frame, bbox);

    // The identical frame must correlate with the learned filter at zero
    // displacement, so the tracker keeps the target rather than declaring loss.
    let tracked = tracker.update(&frame).expect(
        "MOSSE reported the target lost on the frame it was initialised from; \
         the bootstrap paired displaced patches with an unshifted regression target",
    );

    assert!(
        (tracked.x - bbox.x).abs() < 1e-9 && (tracked.y - bbox.y).abs() < 1e-9,
        "an identical frame must not move the box: ({}, {}) -> ({}, {})",
        bbox.x,
        bbox.y,
        tracked.x,
        tracked.y
    );
    assert_eq!((tracked.width, tracked.height), (bbox.width, bbox.height));
}
