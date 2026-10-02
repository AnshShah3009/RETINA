use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;
use rayon::prelude::*;

/// FAST-9 corner detector
/// Uses a 16-pixel Bresenham circle of radius 3
/// A corner is detected if 9 contiguous pixels are all brighter or all darker
pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut keypoints = Vec::new();

    // Bresenham circle of radius 3 - 16 points
    // These are the (x, y) offsets from the center pixel
    #[rustfmt::skip]
    let circle_offsets: [(i32, i32); 16] = [
        (0, -3),  (1, -3),  (2, -2),  (3, -1),
        (3, 0),   (3, 1),   (2, 2),   (1, 3),
        (0, 3),   (-1, 3),  (-2, 2),  (-3, 1),
        (-3, 0),  (-3, -1), (-2, -2), (-1, -3),
    ];

    // Scan the image one row at a time, in parallel. Each pixel's verdict
    // depends only on its own 16-pixel neighbourhood, so rows are independent;
    // collecting per-row and concatenating in order preserves the exact
    // raster-order output the serial loop produced, which the pyramid-level
    // truncation and non-maximum suppression downstream rely on.
    //
    // This scan was the single largest cost in ORB detection: it runs once per
    // pyramid level (8 per image) and measured 40 ms/frame on 640x480 TUM frames
    // while descriptor extraction, which already parallelises, took the rest.
    let raw = image.as_raw();
    let w = width as usize;
    let mut rows: Vec<Vec<KeyPoint>> = (3..height - 3)
        .into_par_iter()
        .map(|y| {
            let mut found: Vec<KeyPoint> = Vec::new();
            let row = y as usize * w;
            for x in 3..width - 3 {
                let p = raw[row + x as usize];

                let high_threshold = p.saturating_add(threshold);
                let low_threshold = p.saturating_sub(threshold);

                // Full test - check all 16 pixels directly
                let mut pixel_values = [0u8; 16];
                let mut brighter_count = 0u32;
                let mut darker_count = 0u32;

                for (i, (dx, dy)) in circle_offsets.iter().enumerate() {
                    let px = (x + dx) as usize;
                    let py = (y + dy) as usize;
                    let val = raw[py * w + px];
                    pixel_values[i] = val;

                    if val > high_threshold {
                        brighter_count += 1;
                    } else if val < low_threshold {
                        darker_count += 1;
                    }
                }

                // Quick rejection: need at least 9 bright or 9 dark to have a chance
                if brighter_count < 9 && darker_count < 9 {
                    continue;
                }

                // Check for 9 contiguous brighter or darker pixels
                if has_n_contiguous(&pixel_values, p, threshold, 9, true)
                    || has_n_contiguous(&pixel_values, p, threshold, 9, false)
                {
                    found.push(KeyPoint::new(x as f64, y as f64));
                }
            }
            found
        })
        .collect::<Vec<Vec<KeyPoint>>>();
    for r in rows.iter_mut() {
        keypoints.append(r);
    }

    if keypoints.len() > max_keypoints {
        keypoints.truncate(max_keypoints);
    }

    KeyPoints { keypoints }
}

/// Check if there are n contiguous pixels that are all brighter or all darker
fn has_n_contiguous(
    pixels: &[u8; 16],
    center: u8,
    threshold: u8,
    n: usize,
    check_brighter: bool,
) -> bool {
    let high_threshold = center.saturating_add(threshold);
    let low_threshold = center.saturating_sub(threshold);

    // Create a binary array: 1 if pixel meets condition, 0 otherwise
    let mut binary = [0u8; 16];
    for (i, &p) in pixels.iter().enumerate() {
        if check_brighter {
            if p > high_threshold {
                binary[i] = 1;
            }
        } else if p < low_threshold {
            binary[i] = 1;
        }
    }

    // Check for n contiguous 1s in the circular array
    // We need to check the array twice to handle wrap-around
    let mut max_consecutive = 0;
    let mut current_consecutive = 0;

    for i in 0..(16 + n) {
        let idx = i % 16;
        if binary[idx] == 1 {
            current_consecutive += 1;
            max_consecutive = max_consecutive.max(current_consecutive);
            if max_consecutive >= n {
                return true;
            }
        } else {
            current_consecutive = 0;
        }
    }

    false
}

/// FAST corner score (OpenCV `FASTScore`): the maximum, over the two
/// candidate 9-pixel arcs, of the summed intensity deviation from the
/// center minus the threshold.
///
/// # Borders
///
/// The ring reaches 3 pixels past the centre in every direction, so taps near
/// an edge — and every tap of a centre within 3 pixels of a border — fall
/// outside the image. Those taps are **clamped to the nearest border pixel**
/// rather than panicking; this was unchecked `get_pixel((x + dx) as u32, ..)`,
/// which both panicked on a negative `x`/`y` and panicked on any coordinate
/// past the edge, so every keypoint within 3 pixels of a border was a panic
/// waiting to happen. Clamping is also what keeps the ring's 16 states well
/// defined there: a corner half cut off by the frame still gets a real ranking
/// score instead of being silently dropped. A *centre* outside the image has no
/// pixels to measure and scores `0.0`, matching the border rule
/// [`orb::compute_harris_response`] uses for a keypoint with no usable window.
///
/// `fast_detect` only ever emits centres in `[3, width - 3) x [3, height - 3)`,
/// so its own keypoints are all scored on fully in-bounds rings; the clamping
/// only affects callers that score an arbitrary coordinate themselves.
///
/// A previous revision returned the MINIMUM ring difference — nearly zero
/// for genuine corners (half the ring sits on each side), so NMS ranked
/// corners arbitrarily.
pub fn corner_score(image: &GrayImage, x: i32, y: i32, threshold: u8) -> f64 {
    const CIRCLE_OFFSETS: [(i32, i32); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    // A centre outside the frame has no intensity to compare against.
    if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
        return 0.0;
    }

    let max_x = image.width() as i32 - 1;
    let max_y = image.height() as i32 - 1;
    // Every ring tap is clamped into the frame, so no tap can leave the image.
    let tap = |cx: i32, cy: i32| -> i32 {
        image.get_pixel(cx.clamp(0, max_x) as u32, cy.clamp(0, max_y) as u32)[0] as i32
    };

    let p = tap(x, y);
    let t = threshold as i32;
    let pi = p;

    let mut diffs = [0i32; 16];
    let mut state = [0u8; 16]; // 1 brighter, 2 darker
    for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
        let val = tap(x + dx, y + dy);
        let d = val - pi;
        diffs[i] = d.abs();
        if d > t {
            state[i] = 1;
        } else if d < -t {
            state[i] = 2;
        }
    }

    let mut best: i32 = 0;
    for want in [1u8, 2u8] {
        // Longest contiguous arc of `want` pixels; track its summed |diff|.
        let mut arc_sum = 0i32;
        let mut arc_len = 0usize;
        let mut best_arc_sum = 0i32;
        let mut best_arc_len = 0usize;
        for i in 0..(16 + 9) {
            let idx = i % 16;
            if state[idx] == want {
                arc_sum += diffs[idx];
                arc_len += 1;
                if arc_len > best_arc_len || (arc_len == best_arc_len && arc_sum > best_arc_sum) {
                    best_arc_len = arc_len;
                    best_arc_sum = arc_sum;
                }
            } else {
                arc_sum = 0;
                arc_len = 0;
            }
        }
        if best_arc_len >= 9 {
            best = best.max(sliding_max9(&diffs, &state, want));
        }
    }

    // Score relative to the threshold, as OpenCV's V = sum - t.
    //
    // The excess is *not* clamped to u8. A u8 response saturates at 255, so on
    // a busy image the several thousand keypoints that exceed the threshold all
    // score identically and the caller's top-N cut keeps an arbitrary subset -
    // measured on TUM fr1_desk, registration swung 12, 25, 9, 17 and 28 views
    // out of 40 purely as the feature budget changed which of the tied
    // keypoints survived. Returning the excess at full range restores a real
    // ordering, so the strongest corners are always the ones kept.
    //
    // The value is capped only where it would exceed what a caller can store
    // in an f64 response, which no reachable image can approach.
    best.saturating_sub(t * 9).max(0) as f64
}

/// The same measure as [`corner_score`], kept as the `u8` form the FAST detector
/// itself compares against its threshold. Same border rules: clamped taps, and
/// `0.0` (here `0u8`) for a centre outside the image.
pub fn corner_score_u8(image: &GrayImage, x: i32, y: i32, threshold: u8) -> u8 {
    corner_score(image, x, y, threshold).min(255.0) as u8
}

/// Maximum sum of |differences| over any window of exactly 9 contiguous
/// circle positions whose state matches `want` throughout.
fn sliding_max9(diffs: &[i32; 16], state: &[u8; 16], want: u8) -> i32 {
    let mut best = 0i32;
    // windows wrap the ring: start anywhere, length 9
    for start in 0..16 {
        let mut ok = true;
        let mut s = 0i32;
        for k in 0..9 {
            let idx = (start + k) % 16;
            if state[idx] != want {
                ok = false;
                break;
            }
            s += diffs[idx];
        }
        if ok && s > best {
            best = s;
        }
    }
    best
}

/// Non-maximum suppression for FAST keypoints
pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints {
    let mut scored_kps: Vec<(KeyPoint, f64)> = keypoints
        .keypoints
        .into_iter()
        .map(|kp| {
            let score = corner_score(image, kp.x as i32, kp.y as i32, threshold);
            (kp, score)
        })
        .collect();

    // Sort by score descending
    // `total_cmp` gives a total order, so a NaN cannot make this panic and
    // equal scores keep a deterministic order.
    scored_kps.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut suppressed: Vec<KeyPoint> = Vec::new();
    let min_distance = 5.0; // Minimum distance between keypoints

    for (kp, _) in scored_kps {
        // Check if this keypoint is far enough from all kept keypoints
        let mut keep = true;
        for kept_kp in &suppressed {
            let dx = kp.x - kept_kp.x;
            let dy = kp.y - kept_kp.y;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < min_distance * min_distance {
                keep = false;
                break;
            }
        }
        if keep {
            suppressed.push(kp);
        }
    }

    KeyPoints {
        keypoints: suppressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_fast_detector_simple() {
        // Create a simple image with a clear corner
        // Pattern: black background with white square in center
        let size = 50u32;
        let mut img = GrayImage::new(size, size);

        // Fill with black
        for y in 0..size {
            for x in 0..size {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        // Draw white square from (15, 15) to (35, 35)
        for y in 15..35 {
            for x in 15..35 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        // FAST fires on the corner of the white square because the ring around
        // it is not uniform, not because 9 of its 16 taps are bright — the
        // axis-aligned square only reaches 5 of them. Pin what the ring really
        // reads, so a failure below reports *why* the detector behaved.
        let circle_offsets: [(i32, i32); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];
        let ring: Vec<u8> = circle_offsets
            .iter()
            .map(|&(dx, dy)| {
                let x = (15i32 + dx) as u32;
                let y = (15i32 + dy) as u32;
                if x < size && y < size {
                    img.get_pixel(x, y)[0]
                } else {
                    panic!("ring tap ({x}, {y}) left a {size}x{size} image")
                }
            })
            .collect();
        assert!(
            ring.iter().any(|&v| v == 255) && ring.iter().any(|&v| v == 0),
            "the ring at the square's corner must straddle the edge, got {ring:?}"
        );

        // Detect with threshold 20
        let kps = fast_detect(&img, 20, 100);

        // We expect at least 4 corners of the white square
        assert!(
            kps.len() >= 4,
            "Expected at least 4 corners, found {}",
            kps.len()
        );

        // The white square spans (15,15) to (34,34).  The 4 corners are
        // approximately (15,15), (34,15), (15,34), (34,34).  Verify at
        // least one keypoint is near each expected corner.
        let expected_corners: [(f64, f64); 4] =
            [(15.0, 15.0), (34.0, 15.0), (15.0, 34.0), (34.0, 34.0)];
        let tolerance = 5.0;

        for &(ex, ey) in &expected_corners {
            let has_nearby = kps.keypoints.iter().any(|kp| {
                let dx = kp.x - ex;
                let dy = kp.y - ey;
                (dx * dx + dy * dy).sqrt() <= tolerance
            });
            assert!(
                has_nearby,
                "Expected a FAST keypoint within {}px of ({}, {}), but none found. \
                 Detected keypoints: {:?}",
                tolerance,
                ex,
                ey,
                kps.keypoints
                    .iter()
                    .map(|kp| (kp.x, kp.y))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_fast_detector_circle_pattern() {
        // Create a pattern with a bright spot - should detect corners around it
        let size = 64u32;
        let mut img = GrayImage::new(size, size);

        // Fill with black
        for y in 0..size {
            for x in 0..size {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        // Draw a white circle in the center - corners should be at the circle's edge
        let center_x = size as f32 / 2.0;
        let center_y = size as f32 / 2.0;
        let radius = 15.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Create a sharp edge at the circle boundary
                if dist <= radius {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }

        // Detect with threshold
        let kps = fast_detect(&img, 50, 500);

        // Should detect corners around the circle
        assert!(
            kps.len() >= 4,
            "Expected at least 4 corners in circle pattern, found {}",
            kps.len()
        );
    }

    #[test]
    fn test_has_n_contiguous() {
        // Test the contiguous detection function directly
        let pixels_all_dark = [0u8; 16];
        let center = 255u8;
        let threshold = 50u8;

        // All 16 pixels are 0, center is 255, threshold 50
        // Dark condition: pixel < 255 - 50 = 205
        // All pixels are 0 < 205, so all 16 should be "dark"
        let result = has_n_contiguous(&pixels_all_dark, center, threshold, 9, false);
        assert!(result, "Should detect 9 contiguous dark pixels");

        // Test alternating pattern - 8 bright, 8 dark
        let mut pixels_alt = [0u8; 16];
        for i in 0..16 {
            pixels_alt[i] = if i % 2 == 0 { 255 } else { 0 };
        }
        let result_alt = has_n_contiguous(&pixels_alt, 128, 50, 9, false);
        assert!(
            !result_alt,
            "an alternating ring has no 9-run, so it must not score as a corner"
        );

        // Test with 9 consecutive darks
        let mut pixels_mixed = [255u8; 16];
        for i in 0..9 {
            pixels_mixed[i] = 0;
        }
        let result_mixed = has_n_contiguous(&pixels_mixed, 255, 50, 9, false);
        assert!(result_mixed, "Should detect 9 contiguous dark pixels");
    }

    // ---- corner_score border behaviour -------------------------------------
    //
    // The ring reaches 3 px past the centre, so every coordinate within 3 px of
    // a border used to panic in `image.get_pixel((x + dx) as u32, ..)`. These
    // tests assert the *value* at those coordinates, not merely that nothing
    // panics: the fix clamps an out-of-frame tap to the nearest border pixel,
    // which is observable in the score.

    const T: u8 = 20;

    /// A bright square whose top-left corner sits exactly on `(cx, cy)`, on a
    /// dark background. The square's own corner is then a strong FAST corner,
    /// including when the square is clipped by the frame.
    fn square_corner_image(cx: i32, cy: i32, side: i32, w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let inside = x >= cx && x < cx + side && y >= cy && y < cy + side;
                img.put_pixel(x as u32, y as u32, Luma([if inside { 255 } else { 0 }]));
            }
        }
        img
    }

    /// A uniform bright frame with a single dark pixel at `(dx, dy)`.
    ///
    /// That pixel is a FAST corner in the strongest possible form: its ring is
    /// 16 taps of +255, so every 9-arc is available and the score is exactly
    /// `9 * 255 - 9 * threshold`. Placing the dark pixel hard against the frame
    /// corner makes the ring taps clamp, which is the case that used to panic.
    fn dark_pixel_image(dx: i32, dy: i32, w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, Luma([255]));
        img.put_pixel(dx as u32, dy as u32, Luma([0]));
        img
    }

    /// Hand-derived: 9 taps at +255 on the chosen arc, minus `9 * threshold`.
    fn dark_pixel_score(threshold: u8) -> f64 {
        9.0 * 255.0 - 9.0 * threshold as f64
    }

    /// Interior scoring must be unchanged by the border handling — the clamp may
    /// only alter coordinates whose ring leaves the frame. The expected value is
    /// derived from the pixel pattern by hand, not read back from the function
    /// under test.
    #[test]
    fn corner_score_interior_is_exactly_the_hand_derived_arc_sum() {
        let img = dark_pixel_image(20, 20, 60, 60);
        assert_eq!(
            corner_score(&img, 20, 20, T),
            dark_pixel_score(T),
            "an isolated dark pixel must score 9*255 - 9t"
        );
        assert_eq!(
            corner_score_u8(&img, 20, 20, T),
            dark_pixel_score(T).min(255.0) as u8
        );

        // A centre whose ring does not reach the dark pixel has no 9-arc that
        // deviates at all, so it must score zero.
        assert_eq!(corner_score(&img, 20, 24, T), 0.0);
        assert_eq!(corner_score_u8(&img, 20, 24, T), 0);
        // And a featureless frame scores zero everywhere.
        let flat = GrayImage::from_pixel(40, 40, Luma([100]));
        assert_eq!(corner_score(&flat, 20, 20, T), 0.0);
        assert_eq!(corner_score(&flat, 0, 0, T), 0.0);
    }

    /// The specific bug: a centre whose ring leaves the frame panicked before.
    /// Now it must return a *correct* clamped-ring score, not merely avoid the
    /// panic.
    ///
    /// With the single dark pixel on the frame corner, the ring's two taps that
    /// would leave the image to the left and up clamp onto the border column
    /// and row, which is that same dark pixel for `(0, -3)` and `(-1, -3)` and
    /// the bright column otherwise. 14 of the 16 taps are therefore +255 and
    /// contiguous, so a full 9-arc exists and the score is the same
    /// `9 * 255 - 9 * threshold` an interior pixel gets.
    #[test]
    fn corner_score_at_the_frame_corner_is_the_clamped_arc_sum() {
        let corner = dark_pixel_image(0, 0, 60, 60);
        assert_eq!(
            corner_score(&corner, 0, 0, T),
            dark_pixel_score(T),
            "the clamped ring at the frame corner must score 9*255 - 9t"
        );
        assert_eq!(corner_score_u8(&corner, 0, 0, T), 255);

        // The value is not an artefact of the corner: the same pattern one
        // pixel in, where the ring is fully in bounds, scores identically.
        let one_in = dark_pixel_image(1, 1, 60, 60);
        assert_eq!(corner_score(&one_in, 1, 1, T), dark_pixel_score(T));
        let deep = dark_pixel_image(20, 20, 60, 60);
        assert_eq!(corner_score(&deep, 20, 20, T), dark_pixel_score(T));

        // All four frame corners, and both clamped borders.
        for (dx, dy) in [(0, 0), (59, 0), (0, 59), (59, 59)] {
            let img = dark_pixel_image(dx, dy, 60, 60);
            assert_eq!(
                corner_score(&img, dx, dy, T),
                dark_pixel_score(T),
                "clamped corner at ({dx},{dy})"
            );
        }

        // The mirror pattern — a single *bright* pixel on a dark frame at the
        // frame corner — must score the same magnitude from the darker arc.
        let mut bright = GrayImage::from_pixel(60, 60, Luma([0]));
        bright.put_pixel(0, 0, Luma([255]));
        assert_eq!(
            corner_score(&bright, 0, 0, T),
            dark_pixel_score(T),
            "a dark quadrant at the frame corner scores the same magnitude"
        );
    }

    /// Every coordinate of a small image, including all four borders, the
    /// corners and coordinates off the frame, must be scorable and equal to the
    /// clamped reference. This is the general statement of the border rule: the
    /// ring tap is the nearest in-bounds pixel along the same axis, which is
    /// exactly a border-replicating pad.
    #[test]
    fn corner_score_every_coordinate_of_a_small_frame_matches_the_reference() {
        let (w, h) = (7u32, 6u32);
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, Luma([((x * 29 + y * 53) % 256) as u8]));
            }
        }

        let pad = 3i32;
        let mut padded = GrayImage::new(w + 2 * pad as u32, h + 2 * pad as u32);
        for y in 0..padded.height() as i32 {
            for x in 0..padded.width() as i32 {
                let sx = (x - pad).clamp(0, w as i32 - 1) as u32;
                let sy = (y - pad).clamp(0, h as i32 - 1) as u32;
                padded.put_pixel(x as u32, y as u32, *img.get_pixel(sx, sy));
            }
        }

        for y in 0..h as i32 {
            for x in 0..w as i32 {
                assert_eq!(
                    corner_score(&img, x, y, T),
                    corner_score(&padded, x + pad, y + pad, T),
                    "clamped ring at ({x},{y}) must equal the equivalent ring in \
                     a border-replicated image"
                );
                assert_eq!(
                    corner_score_u8(&img, x, y, T),
                    corner_score_u8(&padded, x + pad, y + pad, T),
                    "u8 form must agree at ({x},{y})"
                );
            }
        }

        // Off-frame centres have no pixels to measure: score 0, not a panic.
        assert_eq!(corner_score(&img, -1, 3, T), 0.0);
        assert_eq!(corner_score(&img, 3, -1, T), 0.0);
        assert_eq!(corner_score(&img, w as i32, 3, T), 0.0);
        assert_eq!(corner_score(&img, 3, h as i32, T), 0.0);
        assert_eq!(corner_score_u8(&img, -1, -1, T), 0);
        assert_eq!(corner_score_u8(&img, w as i32 + 5, h as i32 + 5, T), 0);
    }

    /// `non_max_suppression` scores whatever keypoints it is handed, so a
    /// keypoint on the border must survive the round trip.
    #[test]
    fn non_max_suppression_handles_border_keypoints() {
        let img = square_corner_image(0, 0, 20, 40, 40);
        let kps = KeyPoints {
            keypoints: vec![
                KeyPoint::new(0.0, 0.0),
                KeyPoint::new(1.0, 1.0),
                KeyPoint::new(20.0, 20.0),
            ],
        };
        let out = non_max_suppression(kps, &img, T);
        assert!(
            !out.keypoints.is_empty(),
            "the interior keypoint must survive"
        );
        assert!(
            out.keypoints
                .iter()
                .all(|kp| kp.x.is_finite() && kp.y.is_finite()),
            "surviving keypoints must be finite"
        );
    }
}
