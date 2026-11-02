//! `calibrate_camera_planar` on views where Zhang's closed form has no camera.
//!
//! Zhang's method solves for the image of the absolute conic `B` and needs it
//! to be definite; with few or nearly fronto-parallel views it
//! often is not, `sqrt(lambda / B11)` is NaN, and calibration was refused:
//!
//!     AlgorithmError("calibrate_camera_planar produced non-finite or
//!     degenerate calibration")
//!
//! `cv2.calibrateCamera` calibrates the scene below, because its
//! initialisation (`initIntrinsicParams2D`) fixes the principal point at the
//! image centre and only solves the focal lengths in closed form.

use cv_calib3d::{calibrate_camera_planar, generate_chessboard_object_points};
use nalgebra::{Matrix3, Point2, Vector3};

/// The five fully visible boards of `lib_tests.rs`'
/// `calibrate_camera_from_chessboard_files_reports_usage`: a fronto-parallel
/// 7x6 board (16 px squares at (70, 50)) warped by 2-D homographies. The
/// closed-form `B` for these views has eigenvalues
/// `[-1.5e-5, -1.1e-5, 1.0]` - indefinite. `cv2.calibrateCamera` on the same
/// corners: rms 0.8146 px, fx 454.3, fy 426.5.
#[test]
fn calibrates_the_warped_board_views_cv2_calibrates() {
    let transforms = [
        Matrix3::new(1.00, 0.00, 0.0, 0.00, 1.00, 0.0, 0.0000, 0.0000, 1.0),
        Matrix3::new(0.92, -0.03, 14.0, 0.04, 1.00, -9.0, 0.0024, -0.0012, 1.0),
        Matrix3::new(1.06, 0.06, -20.0, -0.03, 0.94, 13.0, -0.0020, 0.0015, 1.0),
        Matrix3::new(0.89, 0.02, 18.0, -0.05, 1.08, 8.0, 0.0016, 0.0021, 1.0),
        Matrix3::new(0.95, 0.05, 10.0, 0.02, 1.04, 6.0, 0.0012, -0.0020, 1.0),
    ];
    let obj = generate_chessboard_object_points((7, 6), 0.04);
    let mut objs = Vec::new();
    let mut imgs = Vec::new();
    for m in &transforms {
        objs.push(obj.clone());
        imgs.push(
            (0..42)
                .map(|k| {
                    let p = m * Vector3::new(
                        70.0 + 16.0 * ((k % 7) as f64 + 1.0) - 0.5,
                        50.0 + 16.0 * ((k / 7) as f64 + 1.0) - 0.5,
                        1.0,
                    );
                    Point2::new(p.x / p.z, p.y / p.z)
                })
                .collect::<Vec<_>>(),
        );
    }
    let calib = calibrate_camera_planar(&objs, &imgs, (320, 240))
        .expect("cv2.calibrateCamera calibrates these views");
    assert!(
        calib.rms_reprojection_error < 0.9,
        "rms {} (cv2: 0.8146)",
        calib.rms_reprojection_error
    );
}
