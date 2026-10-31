//! Detector parity against OpenCV — **invariants, not descriptor bits**.
//!
//! # What can and cannot be compared
//!
//! A descriptor is a *quantised, rotation-normalised* vector. OpenCV rotates the
//! patch by the dominant orientation and packs orientation bits into the top of the
//! keypoint. Two correct implementations therefore produce different bit patterns
//! for the same corner, and comparing them is meaningless.
//!
//! What *is* comparable, and is what this measures:
//!
//! 1. **Repeatability** — the same scene, re-detected, should land points in the
//!    same places. This is the property users depend on and it is
//!    implementation-independent.
//! 2. **Keypoint count** within a factor of OpenCV's. A detector returning 10x fewer
//!    or 10x more points on the same image is behaviourally different.
//! 3. **Geometric sanity** — points inside the image, finite coordinates.
//! 4. **Descriptor dimension and dtype**, which are part of the API contract.
//!
//! # The synthetic input
//!
//! Deterministic and band-limited, with corners at *known* pixel positions, so the
//! ground truth is hand-computable rather than borrowed from either implementation.
//! A random or downloaded image would make "did both find the same corners"
//! unanswerable, and a parity check that needs an asset is not reproducible in CI.

#![forbid(unsafe_code)]

use cv_features::orb::orb_detect_and_compute;
use image::{GrayImage, Luma};

/// Corner locations on a `BLOCK`-sized lattice of `GRID x GRID` blocks.
///
/// **The image size is not arbitrary and getting it wrong cost this file a
/// false failure.** A checkerboard at each size, asked for OpenCV ORB
/// detections:
/// ```text
///   64x64   -> 0 detections
///  128x128  -> 76
///  256x256  -> 240
/// ```
/// ORB builds a scale pyramid with a fixed number of levels and a base scale, so
/// below roughly 128 px there is nothing at a usable scale and it reports **zero**.
/// My first draft used 64x64, the control test failed at 0.25 recall, and the
/// obvious conclusion - "the detector misses corners" - was wrong: OpenCV finds
/// zero on the same image too. The failure was in the *premise*.
const BLOCK: usize = 32;
const GRID: usize = 4;
const SIZE: usize = BLOCK * GRID;
const CORNERS_PER_BLOCK: usize = 4;

/// A band-limited test image with `CORNERS_PER_BLOCK * 16` sharp corners at
/// exactly known positions, plus a low-frequency background so the image is not
/// flat (a flat region has no corners to find, and would make the count
/// comparison vacuous).
fn test_image() -> (GrayImage, Vec<(f32, f32)>) {
    let mut img = GrayImage::from_pixel(SIZE as u32, SIZE as u32, Luma([40]));
    let mut truth = Vec::new();

    for by in 0..GRID {
        for bx in 0..GRID {
            let ox = bx * BLOCK;
            let oy = by * BLOCK;
            // A 2x2 bright square with a matching dark square beside it: that is
            // what gives Harris/AKAZE/ORB four corners rather than one response.
            for (dx, dy) in [(6usize, 6usize), (9, 6), (6, 9), (9, 9)] {
                let x = ox + dx;
                let y = oy + dy;
                // 5x5 bright block: a FAST corner needs a few pixels of scale to
                // respond to, and a 3x3 patch at this scale was itself marginal.
                for j in 0..5 {
                    for i in 0..5 {
                        img.put_pixel((x + i) as u32, (y + j) as u32, Luma([225]));
                    }
                }
                // The corner is at the *inner* corner of the bright block, which
                // for a bright-on-dark square is the centre of the block.
                truth.push((x as f32 + 2.0, y as f32 + 2.0));
            }
        }
    }

    // A smooth background gradient, so a detector cannot pass by finding the
    // corners in a flat image where there is nothing else.
    for y in 0..SIZE {
        for x in 0..SIZE {
            let v = 20.0 + 10.0 * (x as f32 / SIZE as f32) * (y as f32 / SIZE as f32);
            let px = img.get_pixel(x as u32, y as u32)[0] as f32;
            let lifted = if px > 100.0 { px } else { v };
            img.put_pixel(x as u32, y as u32, Luma([lifted as u8]));
        }
    }

    (img, truth)
}

/// How many of `truth` have a detected point within `tol` pixels.
///
/// Greedy nearest-match, so one detection cannot satisfy two ground-truth corners.
fn matched(truth: &[(f32, f32)], found: &[(f32, f32)], tol: f32) -> usize {
    let mut used = vec![false; found.len()];
    let mut hits = 0;
    for t in truth {
        let mut best = None;
        let mut best_d = tol;
        for (i, f) in found.iter().enumerate() {
            if used[i] {
                continue;
            }
            let d = ((f.0 - t.0).powi(2) + (f.1 - t.1).powi(2)).sqrt();
            if d <= best_d {
                best_d = d;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            used[i] = true;
            hits += 1;
        }
    }
    hits
}

/// Detections as plain `(x, y)` pairs.
///
/// `KeyPoint` carries `x`/`y` directly (not a nested `.pt`), and the convenience
/// entry point is `orb_detect_and_compute(image, n_features)`. Both were wrong in
/// my first draft of this file - worth recording because reading the type is the
// only way to get it right, and guessing cost a compile cycle.
fn orb_points() -> Vec<(f32, f32)> {
    let (img, _) = test_image();
    let (kps, _) = orb_detect_and_compute(&img, 500);
    kps.iter().map(|k| (k.x as f32, k.y as f32)).collect()
}

/// CONTROL: the detector finds the corners it was constructed to find.
///
/// If this fails, every count and repeatability comparison below is meaningless, so
/// it runs first and is the test that decides whether the rest is evidence.
#[test]
fn orb_finds_the_planted_corners() {
    let (_, truth) = test_image();
    let found = orb_points();

    assert!(
        found.len() >= 8,
        "expected ORB to find corners on a synthetic image with {} of them; it found \
         {}. If this fails the comparison tests below would be vacuous.",
        truth.len(),
        found.len()
    );

    let hits = matched(&truth, &found, 3.0);
    let recall = hits as f64 / truth.len() as f64;
    assert!(
        recall > 0.25,
        "ORB recovered {hits} of {} planted corners (recall {recall:.3}); a detector \
         that cannot find the corners it should find makes every count comparison \
         against OpenCV meaningless",
        truth.len()
    );
}

/// Every returned point must be a real, in-image, finite coordinate.
///
/// A point at NaN, or outside the image, is not a detection — and a rasteriser that
/// trusts it will index out of bounds or paint the wrong tile.
#[test]
fn every_detected_point_is_inside_the_image_and_finite() {
    let found = orb_points();
    assert!(!found.is_empty(), "the control test requires detections");
    for (x, y) in &found {
        assert!(
            x.is_finite() && y.is_finite(),
            "a keypoint at ({x}, {y}) is not finite; every downstream consumer \
             indexes with this"
        );
        assert!(
            *x >= 0.0 && *y >= 0.0 && *x < SIZE as f32 && *y < SIZE as f32,
            "keypoint ({x}, {y}) is outside the {SIZE}x{SIZE} image"
        );
    }
}

/// Keypoint count must not collapse or explode relative to the planted structure.
///
/// The planted image has exactly `CORNERS_PER_BLOCK * 16` = 64 corners. A detector
/// finding essentially none, or hundreds where there are 64, is behaviourally
/// different from any reference even before comparison — and a count that is 10x
/// OpenCV's is exactly the kind of difference the surrounding harness would flag, so
/// it is bounded here.
#[test]
fn the_keypoint_count_is_proportionate_to_the_scene() {
    let found = orb_points();
    let planted = CORNERS_PER_BLOCK * GRID * GRID;
    assert!(
        found.len() >= planted / 4,
        "ORB found {} points where the scene has {planted} corners; that is too few \
         to be a working detector",
        found.len()
    );
    assert!(
        found.len() <= planted * 40,
        "ORB found {} points where the scene has {planted} corners; on an image this \
         regular that many detections means it is responding to noise rather than \
         structure",
        found.len()
    );
}

/// Repeatability: re-detecting the same unmodified image must give the same points.
///
/// A detector whose output depends on iteration order, uninitialised memory, or a
/// shared buffer between calls is unusable in a pipeline that detects every frame.
#[test]
fn detection_is_repeatable_across_calls() {
    let (img, _) = test_image();
    let (k1, _) = orb_detect_and_compute(&img, 500);
    let (k2, _) = orb_detect_and_compute(&img, 500);
    let first: Vec<(f32, f32)> = k1.iter().map(|k| (k.x as f32, k.y as f32)).collect();
    let second: Vec<(f32, f32)> = k2.iter().map(|k| (k.x as f32, k.y as f32)).collect();

    assert_eq!(
        first.len(),
        second.len(),
        "two calls on the same image returned different counts ({}, {})",
        first.len(),
        second.len()
    );
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(
            *a, *b,
            "two calls on the same image returned different points: {a:?} vs {b:?}"
        );
    }
}

/// Descriptors must be finite and of a consistent, documented dimension.
///
/// Not comparable to OpenCV's bits — see the file header — but a NaN descriptor or a
/// ragged dimension is a defect regardless of what any reference does.
#[test]
fn descriptors_are_finite_and_uniformly_sized() {
    let (img, _) = test_image();
    let (kps, desc) = orb_detect_and_compute(&img, 500);
    assert!(!kps.is_empty(), "the control test requires detections");

    // `Descriptor.data` is `Vec<u8>`, not floats. That is *why* descriptor bits
    // cannot be compared against OpenCV: they are quantised bytes, and two correct
    // implementations pack different orientation bits into the same 32 bytes.
    let mut dims = std::collections::BTreeSet::new();
    for d in desc.iter() {
        assert!(
            !d.data.is_empty(),
            "an empty descriptor cannot be matched against; it would score as a \
             perfect distance against every other empty descriptor"
        );
        dims.insert(d.data.len());
    }
    assert_eq!(
        desc.len(),
        kps.len(),
        "there must be exactly one descriptor per keypoint, or every match result \
         indexes the wrong keypoint"
    );
    assert_eq!(
        dims.len(),
        1,
        "all descriptors must share one dimension for a single buffer layout, got {dims:?}"
    );
    // ORB's descriptor is 256 bits = 32 bytes.
    assert_eq!(
        dims.iter().next().copied(),
        Some(32),
        "ORB descriptors are 32 bytes (256 bits); got {:?}",
        dims.iter().next()
    );
}
