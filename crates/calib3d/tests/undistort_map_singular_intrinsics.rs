//! Undistortion maps must be rejected when the new intrinsics are singular.
//!
//! `init_undistort_rectify_map` and the stereo rectifier both did
//! `new_intrinsics.matrix().try_inverse().unwrap_or(Matrix3::identity())`. The
//! identity means "the new camera has no intrinsics", so a destination pixel is
//! treated as already being in normalised coordinates. Measured on an 8x6 frame:
//!
//! ```text
//! valid:  map_x row 0 = [0,1,2,3,4,5,6,7]   map_y row 0 = [0; 8]
//! zero fx: map_x row 0 = [4,4,4,4,4,4,4,4]   map_y row 0 = [3; 8]
//! ```
//!
//! Every destination collapsed onto the principal point — undistorting with that
//! map reproduces a single pixel across the entire frame. Worse, all 48
//! destinations still mapped *inside* the source image, so the result passed any
//! bounds check and looked perfectly valid.

use cv_calib3d::{init_undistort_rectify_map, CameraCalibrationOptions};
use cv_core::{CameraIntrinsics, Distortion};
use nalgebra::Matrix3;

fn good() -> CameraIntrinsics {
    CameraIntrinsics::new(100.0, 100.0, 4.0, 3.0, 8, 6)
}

fn degenerate() -> CameraIntrinsics {
    CameraIntrinsics::new(0.0, 0.0, 4.0, 3.0, 8, 6)
}

/// The control: with valid intrinsics the map is the identity remap, every
/// destination mapping to its own coordinates.
#[test]
fn a_valid_map_is_the_identity_remap() {
    let k = good();
    let r = Matrix3::identity();
    let (mx, my) = init_undistort_rectify_map((8, 6), &k, &Distortion::none(), &r, &k)
        .expect("valid intrinsics must produce a map");

    assert_eq!(mx.len(), 48);
    assert_eq!(
        &mx[..8],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "an identity rectification with identical intrinsics is the identity remap"
    );
    assert!(
        my[..8].iter().all(|v| *v == 0.0),
        "row 0 must map to source row 0, got {:?}",
        &my[..8]
    );
}

#[test]
fn a_singular_new_intrinsic_matrix_is_rejected() {
    let err = init_undistort_rectify_map(
        (8, 6),
        &good(),
        &Distortion::none(),
        &Matrix3::identity(),
        &degenerate(),
    )
    .expect_err("singular new intrinsics must be reported, not replaced by identity");
    assert!(format!("{err}").contains("singular"), "got: {err}");
}

#[test]
fn a_singular_rectification_rotation_is_rejected() {
    let k = good();
    // A rank-deficient "rotation": all rows equal.
    let singular = Matrix3::new(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
    let err = init_undistort_rectify_map((8, 6), &k, &Distortion::none(), &singular, &k)
        .expect_err("a singular rectification must be reported");
    assert!(format!("{err}").contains("singular"), "got: {err}");
}

/// A map must never be produced when the intrinsics cannot be inverted - the
/// failure mode was a map that looked entirely well formed.
#[test]
fn no_map_is_returned_for_degenerate_intrinsics() {
    match init_undistort_rectify_map(
        (8, 6),
        &good(),
        &Distortion::none(),
        &Matrix3::identity(),
        &degenerate(),
    ) {
        Ok((mx, my)) => panic!(
            "expected an error, got a {}x{} map whose first row is {:?}",
            mx.len(),
            my.len(),
            &mx[..8]
        ),
        Err(_) => {}
    }
}

/// The fisheye sibling already did this correctly, with the same shape of check.
/// Asserted here so the pinhole fix is pinned against the version that was
/// right all along, rather than the two drifting apart again.
#[test]
fn the_fisheye_map_builder_is_guarded() {
    use cv_calib3d::fisheye_init_undistort_rectify_map;
    use cv_core::FisheyeDistortion;
    let k = good();
    let r = Matrix3::identity();

    assert!(
        fisheye_init_undistort_rectify_map((8, 6), &k, &FisheyeDistortion::default(), &r, &k)
            .is_ok(),
        "control: valid intrinsics must still produce a fisheye map"
    );

    let err = fisheye_init_undistort_rectify_map(
        (8, 6),
        &k,
        &FisheyeDistortion::default(),
        &r,
        &degenerate(),
    )
    .expect_err("the fisheye builder already rejected singular intrinsics");
    assert!(format!("{err}").contains("invertible"), "got: {err}");
}
