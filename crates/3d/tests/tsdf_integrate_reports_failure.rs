//! `TSDFVolume::integrate` had two silent failure paths, both of which made the
//! call report success while doing nothing or something wrong.
//!
//! 1. `if let Ok(runner) = cv_runtime::best_runner()` swallowed the runner error,
//!    so a machine where not even the CPU device registry could be initialized got
//!    a **silent no-op**: the frame was never integrated and the volume never
//!    changed, but the call returned as though it had. A dropped frame is
//!    indistinguishable from a frame that legitimately contributed nothing.
//!
//! 2. `extrinsics.try_inverse().unwrap_or_else(Matrix4::identity)` replaced a
//!    singular pose with the identity. `extrinsics_inv` brings each *world* voxel
//!    into camera space, so the identity means "every world point is already a
//!    camera point" - every projective distance, and therefore every TSDF value,
//!    would be computed at the wrong pose, with the call reporting success.
//!
//! Both are now `Err`. Neither test needs a GPU: the checks happen before any
//! dispatch, so a caller with no adapter still reaches them.

use cv_3d::tsdf::TSDFVolume;
use cv_core::CameraIntrinsicsF32;
use nalgebra::{Matrix4, Vector3};

const W: usize = 32;
const H: usize = 24;

fn intrinsics() -> CameraIntrinsicsF32 {
    CameraIntrinsicsF32::new(
        180.0,
        180.0,
        (W / 2) as f32,
        (H / 2) as f32,
        W as u32,
        H as u32,
    )
}

fn volume() -> TSDFVolume {
    TSDFVolume::new(0.01, 0.015)
}

/// A frame with a plausible depth image and a valid pose.
fn frame() -> (Vec<f32>, Vec<Vector3<u8>>) {
    let mut depth = vec![0.0f32; W * H];
    let mut colors = vec![Vector3::new(0u8, 0, 0); W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            depth[i] = 2000.0; // millimetres
            colors[i] = Vector3::new(((x * 255) / (W - 1)) as u8, 40, 200);
        }
    }
    (depth, colors)
}

/// A singular pose must be reported, not silently replaced by the identity.
///
/// `Matrix4::zeros()` is the clearest singular input. It is reachable because
/// `extrinsics` is caller-supplied with no validation up the stack - a rotation
/// built from a degenerate basis is the realistic route.
#[test]
fn a_singular_extrinsics_matrix_is_reported_not_replaced_by_the_identity() {
    let (depth, colors) = frame();
    let mut vol = volume();

    let err = vol
        .integrate(
            &depth,
            Some(&colors),
            &intrinsics(),
            &Matrix4::<f32>::from_element(0.0),
            W,
            H,
        )
        .expect_err("a singular pose has no camera frame and must be reported");

    let msg = format!("{err}");
    assert!(
        msg.contains("singular") || msg.contains("extrinsics"),
        "the error must name the offending input; got: {msg}"
    );

    // And nothing may have been integrated: a singular pose must not leave a
    // partially-updated volume that looks like a successful integration.
    assert!(
        vol.extract_mesh().is_empty(),
        "a rejected frame must leave the volume empty, not partially integrated"
    );
}

/// CONTROL: a valid pose still integrates, so the check cannot pass by rejecting
/// everything.
#[test]
fn a_valid_pose_still_integrates() {
    let (depth, colors) = frame();
    let mut vol = volume();

    // Row-major, matching the sibling test's convention. A tilt plus a translation,
    // which is a rigid transform and therefore invertible.
    #[rustfmt::skip]
    let extrinsics = Matrix4::from_row_slice(&[
        0.939_372_7, 0.0,  0.342_897_8, 0.0, //
        0.0,         1.0,  0.0,         0.0, //
       -0.342_897_8, 0.0,  0.939_372_7, 0.0, //
        0.0,         0.0,  0.0,         1.0,
    ]);

    vol.integrate(&depth, Some(&colors), &intrinsics(), &extrinsics, W, H)
        .expect("a rigid transform is invertible, so integration must succeed");

    assert!(
        !vol.extract_mesh().is_empty(),
        "a valid frame must populate the volume - otherwise the test above is \
         vacuous because integration never works"
    );
}

/// A rotation-only pose (no translation) must also be accepted: the check is
/// invertibility, not "has a translation".
#[test]
fn a_rotation_only_pose_is_invertible_and_accepted() {
    let (depth, colors) = frame();
    let mut vol = volume();

    // 90 degrees about y, row-major.
    #[rustfmt::skip]
    let extrinsics = Matrix4::from_row_slice(&[
         0.0, 0.0, 1.0, 0.0, //
         0.0, 1.0, 0.0, 0.0, //
        -1.0, 0.0, 0.0, 0.0, //
         0.0, 0.0, 0.0, 1.0,
    ]);

    vol.integrate(&depth, Some(&colors), &intrinsics(), &extrinsics, W, H)
        .expect("a pure rotation is invertible");
}

/// A truncated depth buffer must be reported rather than integrating the frames it
/// does have. This is the boundary the sibling test file `length_checks.rs`
/// documents; it is asserted here through the public entry point as well, because
/// the `Result` return is new and the length check must still run.
#[test]
fn a_short_depth_buffer_is_reported() {
    let (depth, colors) = frame();
    let mut vol = volume();

    #[rustfmt::skip]
    let extrinsics = Matrix4::from_row_slice(&[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0f32,
    ]);
    // One sample short of `W * H`.
    let result = vol.integrate(
        &depth[..W * H - 1],
        Some(&colors),
        &intrinsics(),
        &extrinsics,
        W,
        H,
    );

    assert!(
        result.is_err(),
        "a depth buffer shorter than width * height must be an error, not a \
         silently partial integration"
    );
}
