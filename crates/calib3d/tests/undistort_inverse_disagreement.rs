#![forbid(unsafe_code)]
//! `cv-core` and `cv-calib3d` must agree on whether a pixel can be un-distorted.
//!
//! Both crates invert the same Brown-Conrady model through the same
//! `Distortion::remove_checked`, and both are reached from the same pipeline.
//! `cv-calib3d::undistort_points` treats "the model has no inverse here" as an
//! error. `cv-core`'s `PinholeModel::try_unproject` used to treat it as success:
//! it called `Distortion::remove`, which reports non-convergence by returning its
//! input unchanged, so a pixel that cannot be un-distorted came back as a ray
//! built from the *distorted* coordinate - a confident, finite, wrong answer.
//!
//! The disagreement is not cosmetic: the ray cv-core returned does not even
//! reproject to the pixel it was asked about, so no caller could detect it by
//! checking the result.

use cv_calib3d::undistort_points;
use cv_core::{
    CameraIntrinsics, CameraIntrinsicsF32, Distortion, DistortionF32, PinholeModel, PinholeModelF32,
};
use nalgebra::Point2;

/// A barrel model whose forward map is not onto the whole plane.
///
/// `r -> r(1 - 0.5 r^2)` peaks at `r = 1/sqrt(3) ~= 0.5774`, where it reaches
/// `0.5443`; past that the map turns over. So a distorted radius of `0.6` is
/// simply not in the image of the model - `remove_checked` correctly returns
/// `None` for it. `k1 = -0.5` is an ordinary wide-angle value, not a made-up
/// pathological one.
const K1: f64 = -0.5;

fn distortion() -> Distortion {
    Distortion::new(K1, 0.0, 0.0, 0.0, 0.0)
}

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
}

#[test]
fn core_and_calib3d_agree_when_the_distortion_model_has_no_inverse() {
    let distortion = distortion();
    let intrinsics = intrinsics();
    // Normalised coordinates: ((620 - 320) / 500, (240 - 240) / 500) = (0.6, 0).
    let pixel = Point2::new(620.0, 240.0);

    // The premise: this radius has no preimage under the model.
    assert_eq!(
        distortion.remove_checked(0.6, 0.0),
        None,
        "r = 0.6 must be outside the image of this barrel model"
    );

    let calib = undistort_points(&[pixel], &intrinsics, &distortion);
    let core = PinholeModel::new(intrinsics, distortion).try_unproject(&pixel, 1.0);

    assert!(
        calib.is_err(),
        "cv-calib3d is expected to refuse a pixel with no inverse, got {calib:?}"
    );
    assert!(
        core.is_err(),
        "cv-core returned {core:?} for pixel {pixel:?}, whose distortion model has no \
         inverse, while cv-calib3d reports {:?} for the same pixel, intrinsics and \
         coefficients",
        calib.unwrap_err()
    );

    // And the ray cv-core used to return was not even self-consistent: inverting
    // (0.6, 0) by handing back (0.6, 0) and then projecting it forward gives
    // 0.6 * (1 - 0.5 * 0.36) = 0.492, not 0.6.
    assert!(
        (distortion.apply(0.6, 0.0).0 - 0.6).abs() > 0.1,
        "the old fallback ray does not reproject to the pixel it came from"
    );
}

#[test]
fn f32_core_agrees_with_calib3d_on_the_same_pixel() {
    let distortion = DistortionF32::new(K1 as f32, 0.0, 0.0, 0.0, 0.0);
    let intrinsics = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let pixel = Point2::new(620.0f32, 240.0);
    assert_eq!(distortion.remove_checked(0.6, 0.0), None);

    let core = PinholeModelF32::new(intrinsics, distortion).try_unproject(&pixel, 1.0);
    assert!(
        core.is_err(),
        "the f32 twin of try_unproject returned {core:?} where the distortion model has \
         no inverse"
    );
}
