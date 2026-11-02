//! `get_rotation_matrix_2d` is `cv2.getRotationMatrix2D(center, angle, scale)`:
//! angle in **degrees**, positive counter-clockwise on screen (y down), and the
//! same `[[α, β, (1-α)cx - β cy], [-β, α, β cx + (1-α) cy]]` layout in f64.
//!
//! `get_rotation_matrix` keeps its existing contract — angle in **radians** —
//! and has the same sign convention, so it equals `cv2.getRotationMatrix2D`
//! with the angle converted (to f32 precision).
//!
//! Reference values: OpenCV 4.13, `cv2.getRotationMatrix2D(c, a, s).tolist()`.

use cv_imgproc::{get_rotation_matrix, get_rotation_matrix_2d};
use nalgebra::Point2;

type Case = ((f32, f32), f64, f64, [[f64; 3]; 2]);

const CASES: [Case; 4] = [
    (
        (10.5, 7.25),
        30.0,
        1.0,
        [
            [0.8660254037844387, 0.49999999999999994, -2.218266739736606],
            [-0.49999999999999994, 0.8660254037844387, 6.2213158225628185],
        ],
    ),
    (
        (0.0, 0.0),
        90.0,
        2.0,
        [
            [1.2246467991473532e-16, 2.0, 0.0],
            [-2.0, 1.2246467991473532e-16, 0.0],
        ],
    ),
    (
        (3.0, 4.0),
        -45.0,
        0.5,
        [
            [0.3535533905932738, -0.35355339059327373, 3.353553390593274],
            [0.35355339059327373, 0.3535533905932738, 1.5251262658470839],
        ],
    ),
    (
        (100.0, 50.0),
        180.0,
        1.0,
        [
            [-1.0, 1.2246467991473532e-16, 200.0],
            [-1.2246467991473532e-16, -1.0, 100.00000000000001],
        ],
    ),
];

#[test]
fn degrees_variant_matches_opencv() {
    for ((cx, cy), angle, scale, want) in CASES {
        let got = get_rotation_matrix_2d(Point2::new(cx, cy), angle, scale);
        for r in 0..2 {
            for c in 0..3 {
                // Same formula and the same libm `cos`/`sin` as OpenCV; allow
                // a few ulp for a platform whose libm rounds differently.
                let tol = 4.0 * f64::EPSILON * want[r][c].abs().max(1.0);
                assert!(
                    (got[r][c] - want[r][c]).abs() <= tol,
                    "center ({cx}, {cy}) angle {angle} scale {scale}: M[{r}][{c}] = {} \
                     but cv2 gives {}",
                    got[r][c],
                    want[r][c]
                );
            }
        }
    }
}

/// The radians variant: same sign convention, so feeding it the angle in
/// radians reproduces OpenCV to f32 precision. Fed the *degrees* value it
/// does not, which is what made the unit difference worth a second function.
#[test]
fn radians_variant_has_opencvs_sign_convention() {
    for ((cx, cy), angle, scale, want) in CASES {
        let m = get_rotation_matrix(Point2::new(cx, cy), angle.to_radians() as f32, scale as f32);
        assert_eq!(m[(2, 0)], 0.0);
        assert_eq!(m[(2, 1)], 0.0);
        assert_eq!(m[(2, 2)], 1.0);
        for r in 0..2 {
            for c in 0..3 {
                let tol = 1e-5 * want[r][c].abs().max(1.0);
                assert!(
                    (m[(r, c)] as f64 - want[r][c]).abs() <= tol,
                    "angle {angle} deg as radians: M[{r}][{c}] = {} but cv2 gives {}",
                    m[(r, c)],
                    want[r][c]
                );
            }
        }
    }
    // 30 passed as if it were degrees is 30 rad, a different rotation.
    let m = get_rotation_matrix(Point2::new(10.5, 7.25), 30.0, 1.0);
    assert!((m[(0, 0)] as f64 - CASES[0].3[0][0]).abs() > 0.1);
}
