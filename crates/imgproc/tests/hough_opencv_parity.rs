//! `hough_lines` / `hough_lines_p` against `cv2.HoughLinesWithAccumulator` /
//! `cv2.HoughLinesP` (OpenCV 4.13).
//!
//! Both used to run `canny(src, 50, 150)` on their input, so an edge map fed
//! to them (what OpenCV expects) was reduced to the edges of its edges, and
//! rho was binned with `floor((rho + diag) / res)`: one line at y = 50 in a
//! 100x100 image came back as two lines at rho 48.58 and 50.58. They now
//! take a binary edge map (nonzero = edge) and use OpenCV's accumulator
//! layout, peak rule, ordering and (for the probabilistic transform) its RNG,
//! so the outputs below match cv2 exactly.

use cv_imgproc::hough::{hough_lines, hough_lines_p};
use image::GrayImage;

fn rand_img(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut s = seed;
    GrayImage::from_fn(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        image::Luma([(s >> 24) as u8])
    })
}

fn hline() -> GrayImage {
    GrayImage::from_fn(100, 100, |_, y| {
        image::Luma([if y == 50 { 255 } else { 0 }])
    })
}

fn multi() -> GrayImage {
    let n = rand_img(80, 60, 9);
    GrayImage::from_fn(80, 60, |x, y| {
        let on = x == y || x == 20 || y == (x + 20) / 2 || n.get_pixel(x, y)[0] > 250;
        image::Luma([if on { 255 } else { 0 }])
    })
}

fn dashed() -> GrayImage {
    GrayImage::from_fn(120, 90, |x, y| {
        let on = (y == 30 && x % 12 < 8) || x + y == 100;
        image::Luma([if on { 255 } else { 0 }])
    })
}

/// `np.pi / div` as the f32 OpenCV casts it to.
fn theta(div: f64) -> f32 {
    (std::f64::consts::PI / div) as f32
}

fn check_lines(name: &str, img: &GrayImage, rho: f32, div: f64, thr: u32, exp: &[(f32, f32, u32)]) {
    let got: Vec<(f32, f32, u32)> = hough_lines(img, rho, theta(div), thr)
        .unwrap()
        .iter()
        .map(|l| (l.rho, l.theta, l.score))
        .collect();
    assert_eq!(got, exp, "{name}: hough_lines differs from cv2");
}

fn check_segments(
    name: &str,
    img: &GrayImage,
    rho: f32,
    div: f64,
    thr: u32,
    min_len: f32,
    gap: f32,
    exp: &[(i32, i32, i32, i32)],
) {
    let got: Vec<(i32, i32, i32, i32)> = hough_lines_p(img, rho, theta(div), thr, min_len, gap)
        .unwrap()
        .iter()
        .map(|s| (s.x1 as i32, s.y1 as i32, s.x2 as i32, s.y2 as i32))
        .collect();
    assert_eq!(got, exp, "{name}: hough_lines_p differs from cv2");
}

#[test]
fn one_line_is_one_peak_at_integer_rho() {
    check_lines("hline", &hline(), 1.0, 180.0, 50, HLINE);
    // The strongest peak is the line itself: rho 50, theta pi/2, 100 votes.
    let best = hough_lines(&hline(), 1.0, theta(180.0), 60).unwrap();
    assert_eq!(best.len(), 1);
    assert_eq!((best[0].rho, best[0].score), (50.0, 100));
}

#[test]
fn lines_on_an_edge_map_match_opencv() {
    check_lines("multi", &multi(), 1.0, 180.0, 25, MULTI);
    check_lines("multi coarse", &multi(), 2.0, 90.0, 25, MULTI_COARSE);
    check_lines("dashed", &dashed(), 1.0, 180.0, 40, DASHED);
}

#[test]
fn segments_match_opencv() {
    check_segments("p hline", &hline(), 1.0, 180.0, 50, 30.0, 5.0, P_HLINE);
    check_segments("p multi", &multi(), 1.0, 180.0, 20, 15.0, 3.0, P_MULTI);
    check_segments("p dashed", &dashed(), 1.0, 180.0, 30, 20.0, 5.0, P_DASHED);
    check_segments(
        "p dashed nogap",
        &dashed(),
        1.0,
        180.0,
        5,
        5.0,
        2.0,
        P_DASHED_NOGAP,
    );
}

#[test]
fn empty_and_zero_sized_inputs_give_no_lines() {
    let blank = GrayImage::new(30, 20);
    assert!(hough_lines(&blank, 1.0, theta(180.0), 1)
        .unwrap()
        .is_empty());
    assert!(hough_lines_p(&blank, 1.0, theta(180.0), 1, 1.0, 1.0)
        .unwrap()
        .is_empty());
    for (w, h) in [(0, 0), (0, 5), (5, 0)] {
        let img = GrayImage::new(w, h);
        assert!(hough_lines(&img, 1.0, theta(180.0), 1).unwrap().is_empty());
        assert!(hough_lines_p(&img, 1.0, theta(180.0), 1, 1.0, 1.0)
            .unwrap()
            .is_empty());
    }
}

// HLINE: cv2.HoughLinesWithAccumulator(img, 1.0, pi/180, 50)
const HLINE: &[(f32, f32, u32)] = &[
    (50.0, 1.5707963705062866, 100),
    (51.0, 1.5533430576324463, 57),
    (49.0, 1.5882495641708374, 57),
];
// MULTI: cv2.HoughLinesWithAccumulator(img, 1.0, pi/180, 25)
const MULTI: &[(f32, f32, u32)] = &[
    (9.0, 2.0245819091796875, 66),
    (20.0, 0.0, 60),
    (0.0, 2.356194496154785, 60),
    (8.0, 2.0420351028442383, 58),
    (-1.0, 2.373647689819336, 44),
    (1.0, 2.3387410640716553, 43),
    (7.0, 2.059488534927368, 39),
    (10.0, 2.0071287155151367, 37),
    (21.0, 0.01745329238474369, 33),
    (-19.0, 3.1241393089294434, 32),
    (2.0, 2.3212878704071045, 27),
    (6.0, 2.076941728591919, 26),
    (-2.0, 2.391101121902466, 26),
];
// MULTI_COARSE: cv2.HoughLinesWithAccumulator(img, 2.0, pi/90, 25)
const MULTI_COARSE: &[(f32, f32, u32)] = &[
    (20.0, 0.0, 64),
    (10.0, 2.0245819091796875, 64),
    (8.0, 2.059488534927368, 63),
    (0.0, 2.3387410640716553, 49),
    (6.0, 2.094395160675049, 41),
    (22.0, 0.03490658476948738, 35),
    (-18.0, 3.1066861152648926, 34),
    (14.0, 1.9547687768936157, 33),
    (2.0, 2.3038346767425537, 33),
    (-2.0, 2.4085543155670166, 33),
    (2.0, 2.1293017864227295, 32),
    (2.0, 2.2340214252471924, 28),
    (6.0, 2.268928050994873, 27),
    (-16.0, 3.0368728637695312, 27),
    (-6.0, 2.4434609413146973, 26),
];
// DASHED: cv2.HoughLinesWithAccumulator(img, 1.0, pi/180, 40)
const DASHED: &[(f32, f32, u32)] = &[
    (71.0, 0.7853981852531433, 90),
    (30.0, 1.5707963705062866, 81),
];
// P_HLINE: cv2.HoughLinesP(img, 1.0, pi/180, 50, minLineLength=30, maxLineGap=5)
const P_HLINE: &[(i32, i32, i32, i32)] = &[(0, 50, 99, 50)];
// P_MULTI: cv2.HoughLinesP(img, 1.0, pi/180, 20, minLineLength=15, maxLineGap=3)
const P_MULTI: &[(i32, i32, i32, i32)] = &[
    (20, 59, 20, 0),
    (52, 35, 79, 49),
    (0, 0, 59, 59),
    (0, 10, 51, 35),
];
// P_DASHED: cv2.HoughLinesP(img, 1.0, pi/180, 30, minLineLength=20, maxLineGap=5)
const P_DASHED: &[(i32, i32, i32, i32)] = &[(0, 30, 115, 30), (11, 89, 100, 0)];
// P_DASHED_NOGAP: cv2.HoughLinesP(img, 1.0, pi/180, 5, minLineLength=5, maxLineGap=2)
const P_DASHED_NOGAP: &[(i32, i32, i32, i32)] = &[
    (12, 30, 19, 30),
    (11, 89, 100, 0),
    (36, 30, 43, 30),
    (48, 30, 55, 30),
    (72, 30, 79, 30),
    (96, 30, 103, 30),
    (108, 30, 113, 30),
    (61, 30, 67, 30),
    (0, 30, 7, 30),
    (24, 30, 29, 30),
];
