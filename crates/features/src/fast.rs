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
    let circle_offsets = CIRCLE_OFFSETS;

    // Scan the image one row at a time, in parallel. Each pixel's verdict
    // depends only on its own 16-pixel neighbourhood, so rows are independent;
    // collecting per-row and concatenating in order preserves the exact
    // raster-order output the serial loop produced, which the pyramid-level
    // truncation and non-maximum suppression downstream rely on.
    //
    // This scan was the single largest cost in ORB detection: it runs once per
    // pyramid level (8 per image) and measured 40 ms/frame on 640x480 TUM frames
    // while descriptor extraction, which already parallelises, took the rest.
    //
    // Each pixel is first screened with OpenCV's high-speed test before the
    // full ring is read: a run of 9 contiguous taps always contains tap 0 or
    // tap 8 (they are 8 apart, so 7 taps lie strictly between them on either
    // side) and likewise tap 4 or tap 12. A pixel failing that for both the
    // brighter and the darker class cannot have a 9-run of either, so skipping
    // it changes nothing. The survivors get the full 16-tap test as two 16-bit
    // masks whose 9-runs are found with shifts of the ring doubled into 32
    // bits; that is exactly "9 contiguous taps on the circular ring", the test
    // `has_n_contiguous` performs.
    let raw = image.as_raw();
    let w = width as usize;
    let ring: [isize; 16] = circle_offsets.map(|(dx, dy)| dy as isize * w as isize + dx as isize);
    let mut rows: Vec<Vec<KeyPoint>> = (3..height - 3)
        .into_par_iter()
        .map(|y| {
            let mut found: Vec<KeyPoint> = Vec::new();
            let row = y as usize * w;
            for x in 3..width - 3 {
                let center = row + x as usize;
                let p = raw[center];
                let tap = |i: usize| raw[(center as isize + ring[i]) as usize];

                let high_threshold = p.saturating_add(threshold);
                let low_threshold = p.saturating_sub(threshold);
                let class =
                    |v: u8| ((v > high_threshold) as u32) | (((v < low_threshold) as u32) << 1);

                // Bit 0: brighter, bit 1: darker.
                let pair_08 = class(tap(0)) | class(tap(8));
                let pair_412 = class(tap(4)) | class(tap(12));
                if pair_08 & pair_412 == 0 {
                    continue;
                }

                let mut bright = 0u32;
                let mut dark = 0u32;
                for (i, &off) in ring.iter().enumerate() {
                    let v = raw[(center as isize + off) as usize];
                    bright |= ((v > high_threshold) as u32) << i;
                    dark |= ((v < low_threshold) as u32) << i;
                }

                if has_9_contiguous_bits(bright) || has_9_contiguous_bits(dark) {
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

/// Whether the 16-tap ring mask (bit `i` set when tap `i` passes) holds 9
/// contiguous set bits, wrapping from tap 15 back to tap 0.
///
/// The ring is doubled into 32 bits so every circular run appears as a linear
/// one; `r` then keeps a bit only where it starts a run of 2, 4, 8 and finally
/// 9 set bits. The shifts fill with zeros, so no run is invented past bit 31.
#[inline]
fn has_9_contiguous_bits(mask: u32) -> bool {
    let m = mask | (mask << 16);
    let r = m & (m >> 1);
    let r = r & (r >> 2);
    let r = r & (r >> 4);
    r & (m >> 8) != 0
}

/// Check if there are n contiguous pixels that are all brighter or all darker
#[cfg(test)]
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

/// FAST corner score, OpenCV's `cornerScore<16>` (`fast_score.cpp`): the
/// **largest threshold for which `(x, y)` is still a FAST-9 corner**, i.e. the
/// maximum over all 9-pixel arcs of the ring of the smallest
/// `|ring - centre|` on that arc, minus one, where the arc is all brighter or
/// all darker than the centre. `0` when `(x, y)` is not a corner at
/// `threshold` (the value OpenCV's suppression grid holds for non-corners).
///
/// A corner at `threshold` therefore always scores `>= threshold`, and
/// re-running the detector with the score as the threshold still finds it
/// while `score + 1` does not. This is the `response` OpenCV's
/// `FastFeatureDetector` reports with non-maximum suppression on, and the
/// value its 3x3 suppression compares (see [`fast_detect_opencv`]).
///
/// This used to be Rosten's score - the arc's summed `|difference|` minus
/// `9 * threshold` - which differs from OpenCV's on every corner (372 against
/// OpenCV's 44 on the first keypoint of `tests/fast_opencv_parity.rs`) and
/// orders corners differently.
///
/// # Borders
///
/// The ring reaches 3 pixels past the centre in every direction, so taps near
/// an edge - and every tap of a centre within 3 pixels of a border - fall
/// outside the image. Those taps are **clamped to the nearest border pixel**
/// rather than panicking, so a corner half cut off by the frame still gets a
/// real score. A *centre* outside the image has no pixels to measure and
/// scores `0.0`, matching the border rule [`orb::compute_harris_response`]
/// uses for a keypoint with no usable window. OpenCV never scores within 3
/// pixels of the border; `fast_detect` and [`fast_detect_opencv`] only emit
/// centres in `[3, width - 3) x [3, height - 3)`, so the clamping only affects
/// callers that score an arbitrary coordinate themselves.
pub fn corner_score(image: &GrayImage, x: i32, y: i32, threshold: u8) -> f64 {
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

    let centre = tap(x, y);
    let mut ring = [0i32; 16];
    for (r, &(dx, dy)) in ring.iter_mut().zip(CIRCLE_OFFSETS.iter()) {
        *r = tap(x + dx, y + dy);
    }
    let score = opencv_corner_score(centre, &ring, threshold as i32);
    if score >= threshold as i32 {
        score as f64
    } else {
        0.0
    }
}

/// OpenCV's `cornerScore<16>` on a centre value and its 16 ring values (in
/// ring order, either direction, any starting tap).
///
/// Returns `max(threshold - 1, s)` where `s` is the largest threshold for
/// which the centre is a corner, exactly as OpenCV's scalar code does: `d[k]
/// = centre - ring[k]` extended to 25 entries, `a0` tracks the best "centre
/// brighter" arc minimum, `b0` the best "centre darker" arc maximum, and the
/// result is `-b0 - 1`.
fn opencv_corner_score(centre: i32, ring: &[i32; 16], threshold: i32) -> i32 {
    const K: usize = 8;
    const N: usize = K * 3 + 1;
    let mut d = [0i32; N];
    for (k, dk) in d.iter_mut().enumerate() {
        *dk = centre - ring[k % 16];
    }

    let mut a0 = threshold;
    for k in (0..16).step_by(2) {
        let mut a = d[k + 1].min(d[k + 2]);
        a = a.min(d[k + 3]);
        if a <= a0 {
            continue;
        }
        a = a.min(d[k + 4]);
        a = a.min(d[k + 5]);
        a = a.min(d[k + 6]);
        a = a.min(d[k + 7]);
        a = a.min(d[k + 8]);
        a0 = a0.max(a.min(d[k]));
        a0 = a0.max(a.min(d[k + 9]));
    }

    let mut b0 = -a0;
    for k in (0..16).step_by(2) {
        let mut b = d[k + 1].max(d[k + 2]);
        b = b.max(d[k + 3]);
        b = b.max(d[k + 4]);
        b = b.max(d[k + 5]);
        if b >= b0 {
            continue;
        }
        b = b.max(d[k + 6]);
        b = b.max(d[k + 7]);
        b = b.max(d[k + 8]);
        b0 = b0.min(b.max(d[k]));
        b0 = b0.min(b.max(d[k + 9]));
    }

    -b0 - 1
}

/// [`corner_score`] as a `u8`, the form OpenCV keeps in its suppression grid.
/// The score never exceeds 254, so nothing saturates. Same border rules:
/// clamped taps, and `0` for a centre outside the image.
pub fn corner_score_u8(image: &GrayImage, x: i32, y: i32, threshold: u8) -> u8 {
    corner_score(image, x, y, threshold).min(255.0) as u8
}

/// The 16-pixel Bresenham circle of radius 3, as `(dx, dy)`.
#[rustfmt::skip]
const CIRCLE_OFFSETS: [(i32, i32); 16] = [
    (0, -3),  (1, -3),  (2, -2),  (3, -1),
    (3, 0),   (3, 1),   (2, 2),   (1, 3),
    (0, 3),   (-1, 3),  (-2, 2),  (-3, 1),
    (-3, 0),  (-3, -1), (-2, -2), (-1, -3),
];

/// `cv::FAST(image, keypoints, threshold, nonmax_suppression,
/// FastFeatureDetector::TYPE_9_16)`, keypoint for keypoint.
///
/// * Every pixel in `[3, width - 3) x [3, height - 3)` with 9 contiguous ring
///   pixels all `> centre + threshold` or all `< centre - threshold` is a
///   corner (the same test as [`fast_detect`]).
/// * With `nonmax_suppression`, a corner is kept only if its
///   [`corner_score`] is **strictly greater** than the score of each of its 8
///   neighbours (non-corners score 0), so two adjacent corners with equal
///   scores are both dropped, as in OpenCV.
/// * Keypoints come out in raster order with `size = 7`, `angle = -1`,
///   `octave = 0` and `response` = the score with suppression on, `0` with it
///   off - what OpenCV reports.
///
/// This is not the greedy 5-pixel-radius [`non_max_suppression`] that
/// [`crate::orb`] uses; see `tests/fast_opencv_parity.rs` for the comparison
/// with `cv2.FastFeatureDetector`.
pub fn fast_detect_opencv(image: &GrayImage, threshold: u8, nonmax_suppression: bool) -> KeyPoints {
    let corners = fast_detect(image, threshold, usize::MAX).keypoints;
    let keypoint = |x: usize, y: usize, response: f64| {
        KeyPoint::new(x as f64, y as f64)
            .with_size(7.0)
            .with_response(response)
    };
    if !nonmax_suppression {
        return KeyPoints {
            keypoints: corners
                .iter()
                .map(|k| keypoint(k.x as usize, k.y as usize, 0.0))
                .collect(),
        };
    }

    let w = image.width() as usize;
    let h = image.height() as usize;
    // Score grid: 0 everywhere but at corners, like OpenCV's row buffers.
    let mut score = vec![0u8; w * h];
    for k in &corners {
        let (x, y) = (k.x as usize, k.y as usize);
        score[y * w + x] = corner_score_u8(image, x as i32, y as i32, threshold);
    }
    let keypoints = corners
        .iter()
        .filter_map(|k| {
            let (x, y) = (k.x as usize, k.y as usize);
            // Corners lie in [3, w - 3) x [3, h - 3), so all 8 neighbours
            // are inside the grid.
            let s = score[y * w + x];
            let beats_all = (y - 1..=y + 1)
                .flat_map(|ny| (x - 1..=x + 1).map(move |nx| (nx, ny)))
                .filter(|&(nx, ny)| (nx, ny) != (x, y))
                .all(|(nx, ny)| s > score[ny * w + nx]);
            beats_all.then(|| keypoint(x, y, s as f64))
        })
        .collect();
    KeyPoints { keypoints }
}

/// Non-maximum suppression for FAST keypoints
///
/// Greedy: candidates are visited strongest first by OpenCV's
/// [`corner_score`] (stable on ties) and one is kept unless a keypoint already
/// kept lies closer than 5 pixels.
///
/// This is *not* OpenCV's FAST suppression, which keeps a corner only if its
/// score beats its 8 neighbours; that is [`fast_detect_opencv`]. ORB's
/// detector uses this one.
pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints {
    // Each score depends only on its own keypoint, so scoring in parallel and
    // collecting in order gives exactly the serial list.
    let mut scored_kps: Vec<(KeyPoint, f64)> = keypoints
        .keypoints
        .into_par_iter()
        .map(|kp| {
            let score = corner_score(image, kp.x as i32, kp.y as i32, threshold);
            (kp, score)
        })
        .collect();

    // Sort by score descending
    // `total_cmp` gives a total order, so a NaN cannot make this panic and
    // equal scores keep a deterministic order.
    scored_kps.sort_by(|a, b| b.1.total_cmp(&a.1));

    let suppressed = match KeptGrid::new(&scored_kps) {
        // The grid only narrows which kept keypoints are compared against; the
        // comparison is the same `nms_too_close` test, so the kept list is
        // identical to the all-pairs scan below. Comparing every kept keypoint
        // was quadratic: 24k candidates on a 1080p frame took ~85 ms.
        Some(mut grid) => {
            let mut kept: Vec<KeyPoint> = Vec::new();
            for (kp, _) in scored_kps {
                if !grid.any_close(&kp, &kept) {
                    grid.insert(&kp, kept.len());
                    kept.push(kp);
                }
            }
            kept
        }
        // Non-finite or far-flung coordinates: compare against every kept
        // keypoint.
        None => {
            let mut kept: Vec<KeyPoint> = Vec::new();
            for (kp, _) in scored_kps {
                if !kept.iter().any(|kept_kp| nms_too_close(&kp, kept_kp)) {
                    kept.push(kp);
                }
            }
            kept
        }
    };

    KeyPoints {
        keypoints: suppressed,
    }
}

/// The suppression test: closer than 5 pixels (squared distance below 25).
#[inline]
fn nms_too_close(kp: &KeyPoint, kept_kp: &KeyPoint) -> bool {
    let min_distance = 5.0; // Minimum distance between keypoints
    let dx = kp.x - kept_kp.x;
    let dy = kp.y - kept_kp.y;
    dx * dx + dy * dy < min_distance * min_distance
}

/// Bucket grid over the candidates' bounding box for [`non_max_suppression`].
///
/// Cells are 8 pixels on a side, wider than the 5-pixel suppression radius, so
/// a kept keypoint within that radius of a candidate is always in the
/// candidate's cell or one of its eight neighbours, with three pixels of slack
/// for floating-point rounding. Each cell is a singly linked list of indices
/// into the kept list (`head` per cell, `next` per kept keypoint).
struct KeptGrid {
    min_x: f64,
    min_y: f64,
    cols: usize,
    rows: usize,
    head: Vec<u32>,
    next: Vec<u32>,
}

impl KeptGrid {
    const CELL: f64 = 8.0;
    /// Upper bound on cells, so a sparse spread of far-apart points falls back
    /// to the all-pairs scan instead of allocating a huge grid.
    const MAX_CELLS: f64 = 16.0 * 1024.0 * 1024.0;

    fn new(candidates: &[(KeyPoint, f64)]) -> Option<Self> {
        if candidates.is_empty() || candidates.len() >= u32::MAX as usize {
            return None;
        }
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (kp, _) in candidates {
            if !kp.x.is_finite() || !kp.y.is_finite() {
                return None;
            }
            min_x = min_x.min(kp.x);
            min_y = min_y.min(kp.y);
            max_x = max_x.max(kp.x);
            max_y = max_y.max(kp.y);
        }
        let cols = ((max_x - min_x) / Self::CELL).floor() + 1.0;
        let rows = ((max_y - min_y) / Self::CELL).floor() + 1.0;
        // Also rejects a NaN product from an overflowing span.
        if !(cols * rows <= Self::MAX_CELLS) {
            return None;
        }
        let (cols, rows) = (cols as usize, rows as usize);
        Some(Self {
            min_x,
            min_y,
            cols,
            rows,
            head: vec![u32::MAX; cols * rows],
            next: Vec::new(),
        })
    }

    fn cell_of(&self, kp: &KeyPoint) -> (usize, usize) {
        let cx = ((kp.x - self.min_x) / Self::CELL) as usize;
        let cy = ((kp.y - self.min_y) / Self::CELL) as usize;
        (cx.min(self.cols - 1), cy.min(self.rows - 1))
    }

    /// Whether any kept keypoint in the 3x3 cell neighbourhood is too close.
    fn any_close(&self, kp: &KeyPoint, kept: &[KeyPoint]) -> bool {
        let (cx, cy) = self.cell_of(kp);
        for ny in cy.saturating_sub(1)..=(cy + 1).min(self.rows - 1) {
            for nx in cx.saturating_sub(1)..=(cx + 1).min(self.cols - 1) {
                let mut i = self.head[ny * self.cols + nx];
                while i != u32::MAX {
                    if nms_too_close(kp, &kept[i as usize]) {
                        return true;
                    }
                    i = self.next[i as usize];
                }
            }
        }
        false
    }

    fn insert(&mut self, kp: &KeyPoint, index: usize) {
        let (cx, cy) = self.cell_of(kp);
        let cell = cy * self.cols + cx;
        self.next.push(self.head[cell]);
        self.head[cell] = index as u32;
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
            ring.contains(&255) && ring.contains(&0),
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
    /// 16 taps of +255, so every 9-arc is available and it stays a corner for
    /// every threshold below 255. Placing the dark pixel hard against the frame
    /// corner makes the ring taps clamp, which is the case that used to panic.
    fn dark_pixel_image(dx: i32, dy: i32, w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, Luma([255]));
        img.put_pixel(dx as u32, dy as u32, Luma([0]));
        img
    }

    /// Hand-derived: every arc's smallest |difference| is 255, so the corner
    /// test `255 > t` holds for every `t <= 254` and the score (the largest
    /// such `t`) is 254, whatever threshold it was detected at.
    fn dark_pixel_score(_threshold: u8) -> f64 {
        254.0
    }

    /// Interior scoring must be unchanged by the border handling — the clamp may
    /// only alter coordinates whose ring leaves the frame. The expected value is
    /// derived from the pixel pattern by hand, not read back from the function
    /// under test.
    #[test]
    fn corner_score_interior_is_exactly_the_hand_derived_score() {
        let img = dark_pixel_image(20, 20, 60, 60);
        assert_eq!(
            corner_score(&img, 20, 20, T),
            dark_pixel_score(T),
            "an isolated dark pixel must score 254"
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
    /// contiguous, so a full 9-arc exists and the score is the same 254 an
    /// interior pixel gets.
    #[test]
    fn corner_score_at_the_frame_corner_is_the_clamped_score() {
        let corner = dark_pixel_image(0, 0, 60, 60);
        assert_eq!(
            corner_score(&corner, 0, 0, T),
            dark_pixel_score(T),
            "the clamped ring at the frame corner must score 254"
        );
        assert_eq!(corner_score_u8(&corner, 0, 0, T), 254);

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

    /// The defining property of OpenCV's score, checked against the detector
    /// rather than a copy of the formula: a corner found at threshold `t`
    /// scores `s >= t`, is still detected at threshold `s`, and is gone at
    /// `s + 1`.
    #[test]
    fn corner_score_is_the_largest_threshold_that_keeps_the_corner() {
        let mut s = 7u32;
        let img = GrayImage::from_fn(40, 30, |_, _| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            Luma([(s >> 24) as u8])
        });
        let found_at = |t: u8, x: f64, y: f64| {
            fast_detect(&img, t, usize::MAX)
                .keypoints
                .iter()
                .any(|k| k.x == x && k.y == y)
        };
        let mut checked = 0;
        for t in [5u8, 30, 70] {
            for kp in fast_detect(&img, t, usize::MAX).keypoints {
                let score = corner_score(&img, kp.x as i32, kp.y as i32, t);
                assert!(
                    score >= t as f64,
                    "({}, {}) t={t} score={score}",
                    kp.x,
                    kp.y
                );
                let score = score as u8;
                assert!(found_at(score, kp.x, kp.y), "lost at its own score {score}");
                if score < 255 {
                    assert!(
                        !found_at(score + 1, kp.x, kp.y),
                        "({}, {}) still a corner above its score {score}",
                        kp.x,
                        kp.y
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 50, "only {checked} corners checked");
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

    /// The all-pairs greedy suppression the grid version replaced.
    fn reference_nms(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> Vec<KeyPoint> {
        let mut scored: Vec<(KeyPoint, f64)> = keypoints
            .keypoints
            .into_iter()
            .map(|kp| {
                let s = corner_score(image, kp.x as i32, kp.y as i32, threshold);
                (kp, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut kept: Vec<KeyPoint> = Vec::new();
        for (kp, _) in scored {
            let close = kept.iter().any(|k| {
                let dx = kp.x - k.x;
                let dy = kp.y - k.y;
                dx * dx + dy * dy < 25.0
            });
            if !close {
                kept.push(kp);
            }
        }
        kept
    }

    fn same_points(a: &[KeyPoint], b: &[KeyPoint]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(p, q)| p.x.to_bits() == q.x.to_bits() && p.y.to_bits() == q.y.to_bits())
    }

    /// The bucketed suppression keeps exactly the keypoints, in exactly the
    /// order, that comparing against every kept keypoint does: on real FAST
    /// output, on fractional coordinates sitting right at the 5-pixel radius,
    /// and on input with a non-finite coordinate (all-pairs fallback).
    #[test]
    fn non_max_suppression_matches_all_pairs_scan() {
        let (w, h) = (160u32, 120u32);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let img = GrayImage::from_fn(w, h, |x, y| {
            Luma([(((x / 7) * 37 + (y / 5) * 91) % 251) as u8 ^ (next() % 9) as u8])
        });
        let fast = fast_detect(&img, 10, usize::MAX);
        assert!(fast.len() > 100, "test image must give many candidates");
        let expect = reference_nms(fast.clone(), &img, 10);
        let got = non_max_suppression(fast, &img, 10);
        assert!(same_points(&got.keypoints, &expect));

        let mut pts = Vec::new();
        for i in 0..2000 {
            let x = (next() % 1600) as f64 / 10.0;
            let y = (next() % 1200) as f64 / 10.0;
            pts.push(KeyPoint::new(x, y));
            // A partner exactly on, just inside and just outside the radius.
            let d = [5.0, 4.999_999_999, 5.000_000_001][i % 3];
            pts.push(KeyPoint::new(x + d, y));
            pts.push(KeyPoint::new(x, y - d));
        }
        let kps = KeyPoints { keypoints: pts };
        let expect = reference_nms(kps.clone(), &img, 10);
        let got = non_max_suppression(kps.clone(), &img, 10);
        assert!(same_points(&got.keypoints, &expect));

        let mut with_nan = kps;
        with_nan.keypoints.push(KeyPoint::new(f64::NAN, 3.0));
        let expect = reference_nms(with_nan.clone(), &img, 10);
        let got = non_max_suppression(with_nan, &img, 10);
        assert_eq!(got.keypoints.len(), expect.len());
        assert!(
            got.keypoints
                .iter()
                .zip(&expect)
                .all(|(p, q)| p.x.to_bits() == q.x.to_bits() && p.y.to_bits() == q.y.to_bits())
        );
    }

    /// The ring in `fast_detect`'s tap order, for the reference scan below.
    const REF_RING: [(i32, i32); 16] = [
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

    /// The detector as it was before the high-speed pre-test and the bit-mask
    /// run test: read all 16 taps, then `has_n_contiguous` both ways.
    fn reference_fast_detect(image: &GrayImage, threshold: u8) -> Vec<(u32, u32)> {
        let (w, h) = image.dimensions();
        let mut out = Vec::new();
        for y in 3..h.saturating_sub(3) {
            for x in 3..w.saturating_sub(3) {
                let p = image.get_pixel(x, y)[0];
                let mut ring = [0u8; 16];
                for (i, &(dx, dy)) in REF_RING.iter().enumerate() {
                    ring[i] = image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0];
                }
                if has_n_contiguous(&ring, p, threshold, 9, true)
                    || has_n_contiguous(&ring, p, threshold, 9, false)
                {
                    out.push((x, y));
                }
            }
        }
        out
    }

    #[test]
    fn nine_run_bit_test_matches_has_n_contiguous_for_every_ring() {
        // Every brighter pattern of the ring: bit i of `mask` makes tap i
        // bright (255) against a centre of 128; the rest equal the centre.
        for mask in 0u32..(1 << 16) {
            let ring: [u8; 16] =
                std::array::from_fn(|i| if mask >> i & 1 == 1 { 255 } else { 128 });
            assert_eq!(
                has_9_contiguous_bits(mask),
                has_n_contiguous(&ring, 128, 20, 9, true),
                "mask {mask:#06x}"
            );
        }
    }

    #[test]
    fn fast_detect_matches_the_full_ring_reference_scan() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut images = Vec::new();
        // Blocky texture with noise: many genuine corners.
        images.push(GrayImage::from_fn(97, 61, |x, y| {
            Luma([(((x / 6) * 53 + (y / 4) * 29) % 256) as u8 ^ (next() % 13) as u8])
        }));
        // Pure noise over the full range, so the saturating thresholds at 0
        // and 255 are exercised.
        images.push(GrayImage::from_fn(64, 48, |_, _| Luma([next() as u8])));
        // Few grey levels: many taps exactly on a threshold.
        images.push(GrayImage::from_fn(50, 40, |_, _| {
            Luma([[0u8, 20, 40, 128, 215, 235, 255][(next() % 7) as usize]])
        }));
        // Smaller than the ring and exactly ring-sized.
        images.push(GrayImage::from_fn(6, 6, |_, _| Luma([next() as u8])));
        images.push(GrayImage::from_fn(7, 7, |_, _| Luma([next() as u8])));
        let mut corners = 0;
        for img in &images {
            for t in [0u8, 1, 10, 20, 50, 127, 200, 255] {
                let expect = reference_fast_detect(img, t);
                let got: Vec<(u32, u32)> = fast_detect(img, t, usize::MAX)
                    .keypoints
                    .iter()
                    .map(|kp| (kp.x as u32, kp.y as u32))
                    .collect();
                assert_eq!(got, expect, "{}x{} t={t}", img.width(), img.height());
                corners += got.len();
            }
        }
        assert!(
            corners > 1000,
            "the images must produce corners, got {corners}"
        );
    }
}
