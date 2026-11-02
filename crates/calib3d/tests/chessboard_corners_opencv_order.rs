//! `find_chessboard_corners` must return the inner corners, in OpenCV's order.
//!
//! The detector used to cluster the strongest Harris responses with 1-D
//! k-means along their PCA axes. Harris fires as strongly on the board's
//! outer L-corners (square against margin) as on its inner X-corners, so the
//! clusters mixed the two lattices. On an ordinary 9x6 board with a white
//! margin - axis-aligned or rotated - every case came back `Ok` with corners
//! up to 52 px from any inner corner, the same corner returned twice, and an
//! order unrelated to the board (first corner mapped to inner corner 45, 0,
//! 8 or 53 depending on the rotation). Calibration then fitted homographies
//! to scrambled correspondences.
//!
//! The order checked here is `cv2.findChessboardCorners`' (4.13), measured on
//! the same renders: row-major, `cols` per row, the first corner the one
//! whose inner diagonal square is black. On the eight rotations below cv2's
//! corners and this detector's agree to 0.06 px, in the same order, and an
//! inverted-colour board starts at the opposite corner in both.

use cv_calib3d::find_chessboard_corners;
use image::{GrayImage, Luma};
use nalgebra::{Matrix3, Point2, Vector3};

/// Render a board of `(cols + 1) x (rows + 1)` squares of side `sq` (board
/// units = pixels before `h`), with a one-square white margin, through the
/// board-to-image homography `h`; 4x4 supersampled, pixel centres at integer
/// coordinates. Returns the image and the inner corners in board row-major
/// order, mapped through `h`.
fn render(
    pattern: (usize, usize),
    sq: f64,
    h: &Matrix3<f64>,
    size: (u32, u32),
    inverted: bool,
) -> (GrayImage, Vec<Point2<f64>>) {
    let (cols, rows) = pattern;
    let h_inv = h.try_inverse().unwrap();
    let colour = |bx: f64, by: f64| -> f64 {
        // Board squares occupy [sq, sq + (cols + 1) sq) x [sq, sq + (rows + 1) sq).
        let sx = ((bx - sq) / sq).floor();
        let sy = ((by - sq) / sq).floor();
        if sx < 0.0 || sy < 0.0 || sx > cols as f64 || sy > rows as f64 {
            return 255.0;
        }
        let black = (sx as i64 + sy as i64) % 2 == if inverted { 1 } else { 0 };
        if black { 0.0 } else { 255.0 }
    };
    let mut img = GrayImage::new(size.0, size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let mut acc = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    let x = px as f64 + (i as f64 + 0.5) / 4.0 - 0.5;
                    let y = py as f64 + (j as f64 + 0.5) / 4.0 - 0.5;
                    let b = h_inv * Vector3::new(x, y, 1.0);
                    acc += colour(b.x / b.z, b.y / b.z);
                }
            }
            img.put_pixel(px, py, Luma([(acc / 16.0).round() as u8]));
        }
    }
    let corners = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (c, r)))
        .map(|(c, r)| {
            let b = Vector3::new(sq * (c as f64 + 2.0), sq * (r as f64 + 2.0), 1.0);
            let p = h * b;
            Point2::new(p.x / p.z, p.y / p.z)
        })
        .collect();
    (img, corners)
}

/// Board-to-image homography: rotate by `deg` and scale about the board
/// centre, place it at the image centre, then a mild perspective.
fn pose(pattern: (usize, usize), sq: f64, deg: f64, scale: f64, persp: (f64, f64)) -> Matrix3<f64> {
    let (cols, rows) = pattern;
    let bc = (
        sq * (cols as f64 + 3.0) / 2.0,
        sq * (rows as f64 + 3.0) / 2.0,
    );
    let (s, c) = deg.to_radians().sin_cos();
    let a = Matrix3::new(
        scale * c,
        -scale * s,
        320.0 - scale * (c * bc.0 - s * bc.1),
        scale * s,
        scale * c,
        240.0 - scale * (s * bc.0 + c * bc.1),
        0.0,
        0.0,
        1.0,
    );
    let p = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, persp.0, persp.1, 1.0);
    p * a
}

fn assert_same_order(found: &[Point2<f64>], truth: &[Point2<f64>], what: &str) {
    assert_eq!(found.len(), truth.len(), "{what}: corner count");
    let worst = found
        .iter()
        .zip(truth)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        worst < 0.3,
        "{what}: corner i is {worst:.2} px from inner corner i (first corner {:?}, expected {:?})",
        found[0],
        truth[0]
    );
}

#[test]
fn inner_corners_in_opencv_order_through_a_full_turn() {
    let pattern = (9, 6);
    for (i, deg) in [0.0, 15.0, -20.0, 35.0, 80.0, 100.0, 170.0, -100.0]
        .into_iter()
        .enumerate()
    {
        let persp = (2e-4 * (i % 3) as f64, -1e-4 * (i % 2) as f64);
        let h = pose(pattern, 30.0, deg, 1.2, persp);
        let (img, truth) = render(pattern, 30.0, &h, (640, 480), false);
        let found = find_chessboard_corners(&img, pattern)
            .unwrap_or_else(|e| panic!("rotation {deg}: {e}"));
        assert_same_order(&found, &truth, &format!("rotation {deg}"));
    }
}

#[test]
fn inverted_colours_start_at_the_opposite_corner() {
    let pattern = (9, 6);
    let h = pose(pattern, 30.0, 0.0, 1.0, (0.0, 0.0));
    let (img, truth) = render(pattern, 30.0, &h, (640, 480), true);
    let found = find_chessboard_corners(&img, pattern).unwrap();
    let reversed: Vec<Point2<f64>> = truth.iter().rev().copied().collect();
    assert_same_order(&found, &reversed, "inverted board");
}

#[test]
fn smaller_low_contrast_board() {
    // 7x6 inner corners, 20 px squares and 30/220 grey levels, as the older
    // unit test draws them.
    let pattern = (7, 6);
    let h = pose(pattern, 20.0, 10.0, 1.0, (1e-4, 0.0));
    let (mut img, truth) = render(pattern, 20.0, &h, (640, 480), false);
    for p in img.pixels_mut() {
        p.0[0] = (30.0 + (p.0[0] as f64 / 255.0) * 190.0).round() as u8;
    }
    let found = find_chessboard_corners(&img, pattern).unwrap();
    assert_same_order(&found, &truth, "7x6 board");
}

#[test]
fn an_image_without_the_board_is_an_error() {
    let pattern = (9, 6);
    let h = pose((5, 4), 30.0, 0.0, 1.0, (0.0, 0.0));
    // A 5x4 board cannot be reported as a 9x6 one.
    let (img, _) = render((5, 4), 30.0, &h, (640, 480), false);
    assert!(find_chessboard_corners(&img, pattern).is_err());
}
