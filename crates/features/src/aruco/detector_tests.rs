//! Detector tests on rendered markers: known homographies, in-plane
//! rotation, strong tilt, blur, noise, clutter and damaged payloads.

use super::*;
use nalgebra::{Matrix3, Rotation3, Vector3};

/// Deterministic xorshift generator for noise and clutter.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately normal, unit variance (sum of 12 uniforms).
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const BLACK: f64 = 30.0;
const WHITE: f64 = 220.0;

/// One marker on a white quiet zone, seen by a pinhole camera.
struct MarkerScene {
    dictionary: ArucoDictionary,
    id: usize,
    /// Payload cells `(x, y)` drawn with the wrong colour.
    flipped: Vec<(usize, usize)>,
    /// Image from marker plane, the plane in cell units with the marker
    /// spanning `[0, cells]^2`.
    image_from_plane: Matrix3<f64>,
    cells: usize,
}

impl MarkerScene {
    /// Marker `id` centred in front of the camera, `side_px` pixels wide when
    /// seen head-on, turned `roll` degrees in its own plane (clockwise in the
    /// image), then tilted `tilt_x` / `tilt_y` degrees about the image axes,
    /// with its centre shifted by `offset` pixels.
    fn new(
        dictionary: ArucoDictionary,
        id: usize,
        roll: f64,
        tilt_x: f64,
        tilt_y: f64,
        side_px: f64,
        offset: (f64, f64),
    ) -> Self {
        let cells = dictionary.marker_size() + 2;
        let f = 700.0;
        let distance = 10.0;
        let k = Matrix3::new(
            f,
            0.0,
            WIDTH as f64 / 2.0 + offset.0,
            0.0,
            f,
            HEIGHT as f64 / 2.0 + offset.1,
            0.0,
            0.0,
            1.0,
        );
        let r = (Rotation3::from_axis_angle(&Vector3::x_axis(), tilt_x.to_radians())
            * Rotation3::from_axis_angle(&Vector3::y_axis(), tilt_y.to_radians())
            * Rotation3::from_axis_angle(&Vector3::z_axis(), roll.to_radians()))
        .into_inner();
        // Plane (u, v) in cells -> marker frame, centred, in world units.
        let scale = side_px / f * distance / cells as f64;
        let centre = Matrix3::new(
            scale,
            0.0,
            -scale * cells as f64 / 2.0,
            0.0,
            scale,
            -scale * cells as f64 / 2.0,
            0.0,
            0.0,
            1.0,
        );
        let rt = Matrix3::from_columns(&[
            r.column(0).into_owned(),
            r.column(1).into_owned(),
            Vector3::new(0.0, 0.0, distance),
        ]);
        Self {
            dictionary,
            id,
            flipped: Vec::new(),
            image_from_plane: k * rt * centre,
            cells,
        }
    }

    fn project(&self, u: f64, v: f64) -> (f64, f64) {
        let q = self.image_from_plane * Vector3::new(u, v, 1.0);
        (q.x / q.z, q.y / q.z)
    }

    /// The marker's own top-left, top-right, bottom-right and bottom-left
    /// corners in the image.
    fn truth(&self) -> [(f64, f64); 4] {
        let c = self.cells as f64;
        [(0.0, 0.0), (c, 0.0), (c, c), (0.0, c)].map(|(u, v)| self.project(u, v))
    }

    /// Grey level of plane point `(u, v)` (cells), `None` outside the quiet
    /// zone of two cells.
    fn plane_value(&self, bits: &[Vec<u8>], u: f64, v: f64) -> Option<f64> {
        let c = self.cells as f64;
        if !(-2.0..c + 2.0).contains(&u) || !(-2.0..c + 2.0).contains(&v) {
            return None;
        }
        if !(0.0..c).contains(&u) || !(0.0..c).contains(&v) {
            return Some(WHITE);
        }
        let (gx, gy) = (u as usize, v as usize);
        if gx == 0 || gy == 0 || gx == self.cells - 1 || gy == self.cells - 1 {
            return Some(BLACK);
        }
        let mut white = bits[gy - 1][gx - 1] == 1;
        if self.flipped.contains(&(gx - 1, gy - 1)) {
            white = !white;
        }
        Some(if white { WHITE } else { BLACK })
    }

    /// Render over `background` (row-major, `WIDTH x HEIGHT`), 4x4
    /// supersampled.
    fn render_into(&self, background: &mut [f64]) {
        const SS: usize = 4;
        let bits = self.dictionary.get_marker(self.id).unwrap();
        let plane_from_image = self.image_from_plane.try_inverse().unwrap();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut acc = 0.0;
                let mut hit = false;
                for sy in 0..SS {
                    for sx in 0..SS {
                        let px = x as f64 + (sx as f64 + 0.5) / SS as f64 - 0.5;
                        let py = y as f64 + (sy as f64 + 0.5) / SS as f64 - 0.5;
                        let q = plane_from_image * Vector3::new(px, py, 1.0);
                        match self.plane_value(&bits, q.x / q.z, q.y / q.z) {
                            Some(v) => {
                                acc += v;
                                hit = true;
                            }
                            None => acc += background[y * WIDTH + x],
                        }
                    }
                }
                if hit {
                    background[y * WIDTH + x] = acc / (SS * SS) as f64;
                }
            }
        }
    }
}

fn blank(level: f64) -> Vec<f64> {
    vec![level; WIDTH * HEIGHT]
}

/// Separable Gaussian blur, replicated borders.
fn blur(img: &[f64], sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-r..=r)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = kernel.iter().sum();
    let pass = |src: &[f64], horizontal: bool| -> Vec<f64> {
        let mut out = vec![0.0; src.len()];
        for y in 0..HEIGHT as isize {
            for x in 0..WIDTH as isize {
                let mut acc = 0.0;
                for (k, wgt) in kernel.iter().enumerate() {
                    let d = k as isize - r;
                    let (sx, sy) = if horizontal { (x + d, y) } else { (x, y + d) };
                    let sx = sx.clamp(0, WIDTH as isize - 1) as usize;
                    let sy = sy.clamp(0, HEIGHT as isize - 1) as usize;
                    acc += wgt * src[sy * WIDTH + sx];
                }
                out[y as usize * WIDTH + x as usize] = acc / norm;
            }
        }
        out
    };
    pass(&pass(img, true), false)
}

fn to_tensor(img: &[f64], noise: f64, seed: u64) -> CpuTensor<u8> {
    let mut rng = Rng(seed);
    let px = img
        .iter()
        .map(|&v| (v + noise * rng.normal()).round().clamp(0.0, 255.0) as u8)
        .collect();
    CpuTensor::from_vec(px, TensorShape::new(1, HEIGHT, WIDTH)).unwrap()
}

fn detector(dictionary: ArucoDictionary, method: CornerRefineMethod) -> ArucoDetector {
    ArucoDetector::with_parameters(
        dictionary,
        DetectorParameters {
            corner_refinement_method: method,
            ..Default::default()
        },
    )
}

/// Largest distance between corner `k` of the detection and of the truth.
fn corner_errors(found: &[(f64, f64); 4], truth: &[(f64, f64); 4]) -> [f64; 4] {
    [0, 1, 2, 3].map(|k| (found[k].0 - truth[k].0).hypot(found[k].1 - truth[k].1))
}

/// Markers of the three dictionary sizes, turned in-plane by 0/90/180/270
/// degrees (and an odd angle), tilted up to 60 degrees, blurred and noisy:
/// each is found exactly once with its id, its corners in the marker's own
/// order. Without refinement the corners are the polygon vertices (border
/// pixels of the dark region, so up to about a pixel inside the true
/// corner; OpenCV 4.13 returns the same vertices on these renders, mean
/// 0.82 px, max 1.67 px), within 2 px; `cornerSubPix` brings them within
/// 0.5 px and at least halves the mean.
#[test]
fn finds_rotated_and_tilted_markers_with_ordered_corners() {
    let views = [
        // (roll, tilt_x, tilt_y)
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (180.0, 0.0, 0.0),
        (270.0, 0.0, 0.0),
        (33.0, 0.0, 0.0),
        (0.0, 30.0, 0.0),
        (90.0, 0.0, 45.0),
        (180.0, 50.0, 20.0),
        (270.0, -35.0, 40.0),
        (15.0, 0.0, 60.0),
        (200.0, 55.0, 0.0),
        (300.0, -40.0, -40.0),
    ];
    let dictionaries = [
        (ArucoDictionary::Dict4x4_50, 7),
        (ArucoDictionary::Dict5x5_100, 61),
        (ArucoDictionary::Dict6x6_250, 203),
    ];
    let mut stats = [(0.0f64, 0.0f64, 0usize); 2];
    for (n, &(roll, tx, ty)) in views.iter().enumerate() {
        for &(dict, id) in &dictionaries {
            let scene = MarkerScene::new(dict, id, roll, tx, ty, 150.0, (0.0, 0.0));
            let mut img = blank(WHITE);
            scene.render_into(&mut img);
            let image = to_tensor(&blur(&img, 0.8), 3.0, 1 + n as u64);
            let truth = scene.truth();
            for (slot, method) in [CornerRefineMethod::None, CornerRefineMethod::Subpix]
                .into_iter()
                .enumerate()
            {
                let found = detector(dict, method).detect_u8(&image).unwrap();
                let ids: Vec<usize> = found.iter().map(|m| m.id).collect();
                assert_eq!(ids, [id], "{dict:?} view {:?} {method:?}", views[n]);
                let errs = corner_errors(&found[0].corners, &truth);
                let max = errs.iter().cloned().fold(0.0, f64::max);
                let limit = if slot == 0 { 2.0 } else { 0.5 };
                assert!(
                    max < limit,
                    "{dict:?} view {:?} {method:?}: corner errors {errs:?}\n found {:?}\n truth {truth:?}",
                    views[n],
                    found[0].corners
                );
                let s = &mut stats[slot];
                s.0 += errs.iter().sum::<f64>();
                s.1 = s.1.max(max);
                s.2 += 4;
            }
        }
    }
    let (none_mean, subpix_mean) = (
        stats[0].0 / stats[0].2 as f64,
        stats[1].0 / stats[1].2 as f64,
    );
    eprintln!(
        "rendered markers, corner error: none mean {none_mean:.3} max {:.3} px; subpix mean \
         {subpix_mean:.3} max {:.3} px",
        stats[0].1, stats[1].1
    );
    assert!(
        subpix_mean < none_mean / 2.0,
        "{subpix_mean} vs {none_mean}"
    );
}

/// Several markers at different poses in one image: each id once, nothing
/// else - neither the inner contours of the markers' black frames nor the
/// contours of other threshold windows produce a second detection.
#[test]
fn several_markers_without_duplicates_or_strays() {
    let dict = ArucoDictionary::Dict4x4_50;
    let placed = [
        (3, 0.0, 0.0, 0.0, (-200.0, -120.0)),
        (11, 90.0, 35.0, 0.0, (0.0, -120.0)),
        (24, 180.0, 0.0, 40.0, (200.0, -120.0)),
        (36, 270.0, -30.0, 25.0, (-200.0, 110.0)),
        (42, 20.0, 20.0, -50.0, (0.0, 110.0)),
        (49, 135.0, 0.0, 0.0, (200.0, 110.0)),
    ];
    let mut img = blank(WHITE);
    let mut truths = Vec::new();
    for &(id, roll, tx, ty, offset) in &placed {
        let scene = MarkerScene::new(dict, id, roll, tx, ty, 95.0, offset);
        scene.render_into(&mut img);
        truths.push((id, scene.truth()));
    }
    let image = to_tensor(&blur(&img, 0.7), 2.0, 7);
    let found = ArucoDetector::new(dict).detect_u8(&image).unwrap();
    let mut ids: Vec<usize> = found.iter().map(|m| m.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, [3, 11, 24, 36, 42, 49]);
    for (id, truth) in truths {
        let m = found.iter().find(|m| m.id == id).unwrap();
        let errs = corner_errors(&m.corners, &truth);
        assert!(errs.iter().all(|&e| e < 1.6), "id {id}: {errs:?}");
    }
}

/// A clean background, and random clutter of filled rectangles, quads and
/// discs in random grey levels with noise: no marker is reported.
#[test]
fn no_markers_on_blank_or_cluttered_background() {
    for dict in [ArucoDictionary::Dict4x4_50, ArucoDictionary::Dict6x6_250] {
        for level in [0.0, 128.0, 255.0] {
            let image = to_tensor(&blank(level), 0.0, 1);
            assert!(
                ArucoDetector::new(dict)
                    .detect_u8(&image)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..4 {
        let mut img = blank(60.0 + 140.0 * rng.uniform());
        for _ in 0..60 {
            let level = 255.0 * rng.uniform();
            let (cx, cy) = (WIDTH as f64 * rng.uniform(), HEIGHT as f64 * rng.uniform());
            let size = 8.0 + 90.0 * rng.uniform();
            let angle = std::f64::consts::PI * rng.uniform();
            let aspect = 0.3 + 1.4 * rng.uniform();
            let shape = rng.next_u64() % 3;
            let (ca, sa) = (angle.cos(), angle.sin());
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                    let (u, v) = (dx * ca + dy * sa, (-dx * sa + dy * ca) / aspect);
                    let inside = match shape {
                        0 => u.abs() < size / 2.0 && v.abs() < size / 2.0,
                        1 => u.abs() + v.abs() < size / 2.0,
                        _ => u.hypot(v) < size / 2.0,
                    };
                    if inside {
                        img[y * WIDTH + x] = level;
                    }
                }
            }
        }
        let image = to_tensor(&blur(&img, 0.8), 4.0, 100 + round);
        for dict in [
            ArucoDictionary::Dict4x4_50,
            ArucoDictionary::Dict5x5_100,
            ArucoDictionary::Dict6x6_250,
        ] {
            let (found, rejected) = ArucoDetector::new(dict)
                .detect_gray_with_rejected(image.as_slice().unwrap(), WIDTH, HEIGHT)
                .unwrap();
            assert!(
                found.is_empty(),
                "clutter round {round}, {dict:?}: stray markers {:?}",
                found.iter().map(|m| m.id).collect::<Vec<_>>()
            );
            // The clutter does produce quadrilateral candidates.
            assert!(!rejected.is_empty());
        }
    }
}

/// Error correction: OpenCV accepts `int(maxCorrectionBits *
/// errorCorrectionRate)` wrong payload bits - 3 for 6x6_250 (5 * 0.6), 1 for
/// 5x5_100 (3 * 0.6) and none for 4x4_50 (1 * 0.6) at the default rate. A
/// marker beyond that is rejected, and its candidate (corners and payload in
/// the order `payload_distance_to_id` reads) is in the rejected list.
#[test]
fn error_correction_within_the_dictionary_capacity() {
    let flips = [(1, 1), (4, 2), (2, 4), (0, 3), (3, 0), (2, 2)];
    let cases = [
        (ArucoDictionary::Dict6x6_250, 113, 3),
        (ArucoDictionary::Dict5x5_100, 42, 1),
        (ArucoDictionary::Dict4x4_50, 9, 0),
    ];
    for (dict, id, capacity) in cases {
        let side = dict.marker_size();
        for n in 1..=capacity + 2 {
            let mut scene = MarkerScene::new(dict, id, 90.0, 20.0, -25.0, 140.0, (0.0, 0.0));
            scene.flipped = flips
                .iter()
                .copied()
                .filter(|&(x, y)| x < side && y < side)
                .take(n)
                .collect();
            assert_eq!(scene.flipped.len(), n);
            let mut img = blank(WHITE);
            scene.render_into(&mut img);
            let image = to_tensor(&blur(&img, 0.7), 2.0, n as u64);
            let (found, rejected) = ArucoDetector::new(dict)
                .detect_gray_with_rejected(image.as_slice().unwrap(), WIDTH, HEIGHT)
                .unwrap();
            if n <= capacity {
                assert_eq!(
                    found.iter().map(|m| m.id).collect::<Vec<_>>(),
                    [id],
                    "{dict:?}: {n} flipped bits must be corrected"
                );
                let errs = corner_errors(&found[0].corners, &scene.truth());
                assert!(errs.iter().all(|&e| e < 1.6), "{errs:?}");
            } else {
                assert!(
                    found.is_empty(),
                    "{dict:?}: {n} flipped bits must be rejected, found {:?}",
                    found.iter().map(|m| m.id).collect::<Vec<_>>()
                );
                // The marker is among the rejected candidates: corner `k`
                // whose payload reads `n` bits off the true code when its
                // top-left corner is the one nearest the marker's.
                let truth = scene.truth();
                let hit = rejected.iter().find_map(|r| {
                    let top_left = (0..4).find(|&k| {
                        (r.corners[k].0 - truth[0].0).hypot(r.corners[k].1 - truth[0].1) < 2.0
                    })?;
                    payload_distance_to_id(dict, r.payload, id, top_left)
                });
                assert_eq!(hit, Some(n as u32), "{dict:?}: rejected {rejected:?}");
            }
        }
    }

    // A higher correction rate corrects the 4x4 marker's single bit.
    let mut scene = MarkerScene::new(
        ArucoDictionary::Dict4x4_50,
        9,
        0.0,
        0.0,
        30.0,
        140.0,
        (0.0, 0.0),
    );
    scene.flipped = vec![(1, 2)];
    let mut img = blank(WHITE);
    scene.render_into(&mut img);
    let image = to_tensor(&img, 0.0, 1);
    let strict = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
    assert!(strict.detect_u8(&image).unwrap().is_empty());
    let lenient = ArucoDetector::with_parameters(
        ArucoDictionary::Dict4x4_50,
        DetectorParameters {
            error_correction_rate: 1.0,
            ..Default::default()
        },
    );
    let found = lenient.detect_u8(&image).unwrap();
    assert_eq!(found.iter().map(|m| m.id).collect::<Vec<_>>(), [9]);
}

/// A marker drawn by `draw_marker` (identical to OpenCV's
/// `generateImageMarker`) on a white canvas, turned by quarter turns: corners
/// and the number of rejected candidates are exactly OpenCV 4.13's
/// `detectMarkers` output on the same images.
#[test]
fn quarter_turns_match_opencv_exactly() {
    let marker = draw_marker(ArucoDictionary::Dict4x4_50, 7, 60).unwrap();
    let m = marker.as_slice().unwrap();
    let mut canvas = vec![255u8; 100 * 100];
    for y in 0..60 {
        canvas[(y + 20) * 100 + 20..(y + 20) * 100 + 80].copy_from_slice(&m[y * 60..y * 60 + 60]);
    }
    // `np.rot90(canvas, k)`: k quarter turns counter-clockwise.
    let expected = [
        [(20.0, 20.0), (79.0, 20.0), (79.0, 79.0), (20.0, 79.0)],
        [(20.0, 79.0), (20.0, 20.0), (79.0, 20.0), (79.0, 79.0)],
        [(79.0, 79.0), (20.0, 79.0), (20.0, 20.0), (79.0, 20.0)],
        [(79.0, 20.0), (79.0, 79.0), (20.0, 79.0), (20.0, 20.0)],
    ];
    let det = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
    let mut img = canvas;
    for want in expected {
        let (found, rejected) = det.detect_gray_with_rejected(&img, 100, 100).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, 7);
        assert_eq!(found[0].corners, want);
        assert_eq!(rejected.len(), 4);
        // Turn counter-clockwise: new (x, y) shows old (99 - y, x).
        let mut next = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 0..100 {
                next[y * 100 + x] = img[x * 100 + (99 - y)];
            }
        }
        img = next;
    }
}

/// Payload grids against OpenCV's `getBitsFromByteList` (1 = white),
/// including the right-aligned last byte of the 5x5 and 6x6 codes.
#[test]
fn dictionary_bits_match_opencv() {
    let cases: [(ArucoDictionary, usize, &[&str]); 9] = [
        (
            ArucoDictionary::Dict4x4_50,
            0,
            &["1011", "0101", "0011", "0010"],
        ),
        (
            ArucoDictionary::Dict4x4_50,
            7,
            &["1100", "0100", "1111", "0010"],
        ),
        (
            ArucoDictionary::Dict4x4_50,
            49,
            &["0101", "0000", "0001", "0011"],
        ),
        (
            ArucoDictionary::Dict5x5_100,
            0,
            &["10100", "01011", "01100", "10101", "11100"],
        ),
        (
            ArucoDictionary::Dict5x5_100,
            7,
            &["01110", "00100", "00101", "00011", "01011"],
        ),
        (
            ArucoDictionary::Dict5x5_100,
            49,
            &["01111", "01101", "01101", "11110", "10111"],
        ),
        (
            ArucoDictionary::Dict6x6_250,
            0,
            &["000111", "100011", "110111", "011000", "001010", "100110"],
        ),
        (
            ArucoDictionary::Dict6x6_250,
            7,
            &["100010", "001010", "010100", "001111", "001010", "011010"],
        ),
        (
            ArucoDictionary::Dict6x6_250,
            49,
            &["100100", "111110", "101101", "111000", "101100", "010100"],
        ),
    ];
    for (dict, id, rows) in cases {
        let want: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| r.bytes().map(|b| b - b'0').collect())
            .collect();
        assert_eq!(dict.get_marker(id).unwrap(), want, "{dict:?} id {id}");

        // `draw_marker` paints 1 white and 0 black inside a black frame.
        let cells = dict.marker_size() + 2;
        let img = draw_marker(dict, id, cells * 10).unwrap();
        let px = img.as_slice().unwrap();
        for gy in 0..cells {
            for gx in 0..cells {
                let v = px[(gy * 10 + 5) * cells * 10 + gx * 10 + 5];
                let inner = gx > 0 && gy > 0 && gx < cells - 1 && gy < cells - 1;
                let white = inner && want[gy - 1][gx - 1] == 1;
                assert_eq!(
                    v,
                    if white { 255 } else { 0 },
                    "{dict:?} id {id} cell {gx},{gy}"
                );
            }
        }
    }
}

#[test]
fn invalid_parameters_are_reported() {
    let image = CpuTensor::from_vec(vec![255u8; 64 * 64], TensorShape::new(1, 64, 64)).unwrap();
    let bad = [
        DetectorParameters {
            adaptive_thresh_win_size_min: 2,
            ..Default::default()
        },
        DetectorParameters {
            adaptive_thresh_win_size_max: 1,
            ..Default::default()
        },
        DetectorParameters {
            adaptive_thresh_win_size_step: 0,
            ..Default::default()
        },
        DetectorParameters {
            min_marker_perimeter_rate: 0.0,
            ..Default::default()
        },
        DetectorParameters {
            perspective_remove_ignored_margin_per_cell: 0.5,
            ..Default::default()
        },
        DetectorParameters {
            marker_border_bits: 0,
            ..Default::default()
        },
        DetectorParameters {
            error_correction_rate: f64::NAN,
            ..Default::default()
        },
        DetectorParameters {
            corner_refinement_method: CornerRefineMethod::Subpix,
            corner_refinement_win_size: 0,
            ..Default::default()
        },
    ];
    for params in bad {
        let det = ArucoDetector::with_parameters(ArucoDictionary::Dict4x4_50, params.clone());
        assert!(det.detect_u8(&image).is_err(), "{params:?}");
    }
    let ok = ArucoDetector::with_parameters(ArucoDictionary::Dict4x4_50, Default::default());
    assert!(ok.detect_u8(&image).unwrap().is_empty());
}
