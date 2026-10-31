//! Value-level validation of the KITTI pose and COLMAP intrinsics readers.
//!
//! Both of these readers used to *trust* a number they had read from disk:
//! KITTI accepted any 3x3 block with a non-zero determinant and handed it to
//! [`cv_core::Pose::new`], which documents that it is unchecked (a scaled or
//! reflected block is silently re-orthonormalised into a different rotation),
//! and COLMAP's `cameras.txt` reader count-checked the `SIMPLE_PINHOLE` /
//! `PINHOLE` parameters without looking at their values, so a negative focal
//! length became a `CameraIntrinsics` with no inverse.
//!
//! Both bugs are invisible to `assert!(read_x(p).is_ok())`: the parse *succeeds*
//! and every downstream metric is silently wrong. So each test here runs the
//! reader on **real files on disk** and asserts the outcome explicitly, and each
//! one carries a CONTROL assertion - a well-formed file that must still parse -
//! so it cannot pass vacuously. The controls deliberately use rotations that are
//! *not* the identity, so a check that rejected every non-identity matrix would
//! fail them.
//!
//! Temp paths are unique per test (process id + atomic counter) because the test
//! binary runs tests concurrently in a single process.

use cv_core::Error;
use nalgebra::{Matrix3, Vector3};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use cv_io::datasets::{colmap, kitti};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A self-deleting temporary directory with a name unique to (process, call).
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "cv_io_pose_valid_{tag}_{}_{seq}_{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let p = self.path.join(name);
        fs::write(&p, contents).expect("write temp file");
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A 90 degree rotation about +Z: a valid rotation that is *not* the identity.
const ROT90: [f64; 9] = [
    0.0, -1.0, 0.0, // row 0
    1.0, 0.0, 0.0, // row 1
    0.0, 0.0, 1.0, // row 2
];

/// Render `r` (row-major, 9 entries) and `t` as a KITTI `poses.txt` line.
fn kitti_line(r: &[f64; 9], t: [f64; 3]) -> String {
    format!(
        "{} {} {} {} {} {} {} {} {} {} {} {}",
        r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2]
    )
}

/// Rotation by 120 deg about the (1, 1, 1) axis, in exact integers:
/// `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`, `det = +1`, not the identity.
const ROT120: [f64; 9] = [
    0.0, 0.0, 1.0, //
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
];

fn matrix_of(r: &[f64; 9]) -> Matrix3<f64> {
    Matrix3::new(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
}

// ===========================================================================
// KITTI: the 3x4 block must be a proper rotation, not merely non-singular
// ===========================================================================

/// CONTROL: two well-formed poses whose rotations are both non-identity (one a
/// +90 deg yaw, one a +120 deg yaw about an arbitrary axis) parse correctly and
/// round-trip through the quaternion conversion unchanged.
///
/// This is what makes the rejection tests below meaningful: the check has to
/// discriminate *valid* rotations from invalid ones, not simply reject anything
/// that is not the identity.
#[test]
fn kitti_control_non_identity_rotations_are_accepted() {
    let dir = TempDir::new("kitti_control");
    let path = dir.write(
        "poses.txt",
        &format!(
            "{}\n{}\n",
            kitti_line(&ROT90, [1.5, -2.5, 3.5]),
            kitti_line(&ROT120, [0.0, 0.0, 0.0])
        ),
    );

    let poses = kitti::read_poses(&path).expect("CONTROL: valid KITTI poses must parse");
    assert_eq!(poses.len(), 2, "CONTROL: both poses must be read");
    assert_eq!(poses[0].translation, Vector3::new(1.5, -2.5, 3.5));

    let got = poses[0].rotation_matrix();
    assert!(
        (got - matrix_of(&ROT90)).norm() < 1e-12,
        "CONTROL: +90 deg yaw must survive the read unchanged, got {got:?}"
    );
    assert!(
        (poses[1].rotation_matrix() - matrix_of(&ROT120)).norm() < 1e-12,
        "CONTROL: the 120 deg rotation must survive the read unchanged"
    );
    // And the identity, which every real trajectory starts from, is still fine.
    let dir = TempDir::new("kitti_control_identity");
    let ident = kitti::read_poses(dir.write(
        "poses.txt",
        &kitti_line(&[1., 0., 0., 0., 1., 0., 0., 0., 1.], [0.; 3]),
    ))
    .expect("CONTROL: the identity rotation is a rotation");
    assert_eq!(ident.len(), 1);
}

/// DEFECT 1: a uniform scale factor of 2 is not a rotation. The old code only
/// rejected a *singular* block, so `2 * I` was accepted and `Pose::new`
/// re-orthonormalised it into `diag(1.75, 1.75, 1.75)` - a rotation about a
/// completely different axis, with no error anywhere.
#[test]
fn kitti_scaled_block_is_rejected() {
    let dir = TempDir::new("kitti_scale");
    let path = dir.write(
        "poses.txt",
        &kitti_line(
            &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            [1.0, 2.0, 3.0],
        ),
    );

    let err = kitti::read_poses(&path).expect_err("a scale-2 block is not a rotation");
    assert!(
        matches!(err, Error::ParseError(_)),
        "expected a ParseError, got {err}"
    );
}

/// DEFECT 1: a reflection (`det = -1`) is not a rotation either. The old code
/// passed it straight through and got `diag(0.5, 0.5, 0.5)` back.
#[test]
fn kitti_reflection_is_rejected() {
    let dir = TempDir::new("kitti_reflect");
    let path = dir.write(
        "poses.txt",
        &kitti_line(
            &[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
    );

    let err = kitti::read_poses(&path).expect_err("a reflection is not a rotation");
    assert!(
        matches!(err, Error::ParseError(_)),
        "expected a ParseError, got {err}"
    );
}

/// DEFECT 1: a huge but perfectly invertible block has a determinant of 1e18,
/// which is neither singular nor `+1`. Columns that are not unit length are not
/// a rotation, so this must be rejected too - this is the case a determinant
/// check alone lets through.
#[test]
fn kitti_large_scale_block_is_rejected() {
    let dir = TempDir::new("kitti_large");
    let path = dir.write(
        "poses.txt",
        &kitti_line(
            &[1e6, 0.0, 0.0, 0.0, 1e6, 0.0, 0.0, 0.0, 1e6],
            [0.0, 0.0, 0.0],
        ),
    );

    let err = kitti::read_poses(&path).expect_err("a 1e6-scaled block is not a rotation");
    assert!(
        matches!(err, Error::ParseError(_)),
        "expected a ParseError, got {err}"
    );
}

/// DEFECT 1: shear. The rows stay unit length and the determinant stays at +1,
/// so neither the singular check nor a bare determinant check catches it - only
/// an orthonormality check does.
#[test]
fn kitti_sheared_block_is_rejected() {
    let dir = TempDir::new("kitti_shear");
    let path = dir.write(
        "poses.txt",
        &kitti_line(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
    );

    let err = kitti::read_poses(&path).expect_err("a sheared block is not a rotation");
    assert!(
        matches!(err, Error::ParseError(_)),
        "expected a ParseError, got {err}"
    );
}

/// The previously-singular case keeps its own error message: an all-zero block
/// has `det = 0`, which is a different failure from "invertible but not a
/// rotation", and the message should say so.
#[test]
fn kitti_singular_block_is_rejected() {
    let dir = TempDir::new("kitti_singular");
    let path = dir.write("poses.txt", &kitti_line(&[0.0; 9], [1.0, 2.0, 3.0]));

    let err = kitti::read_poses(&path).expect_err("an all-zero block is not a rotation");
    assert!(
        matches!(err, Error::ParseError(_)),
        "expected a ParseError, got {err}"
    );
}

/// A valid first pose must not license a broken later one: the error has to name
/// the offending line, so a 2000-pose trajectory does not fail anonymously at
/// index 1999.
#[test]
fn kitti_bad_rotation_names_the_line() {
    let dir = TempDir::new("kitti_line_no");
    let path = dir.write(
        "poses.txt",
        &format!(
            "{}\n{}\n",
            kitti_line(&ROT90, [0.0, 0.0, 0.0]),
            kitti_line(
                &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0]
            )
        ),
    );

    let err = kitti::read_poses(&path).expect_err("line 2 is not a rotation");
    let msg = err.to_string();
    assert!(
        msg.contains("line 2"),
        "the error must locate the bad line, got: {msg}"
    );
}

// ===========================================================================
// COLMAP: camera intrinsics are values, not just a count of values
// ===========================================================================

/// CONTROL: both pinhole models parse, with the parameters mapped to
/// `CameraIntrinsics` in the documented order.
#[test]
fn colmap_control_pinhole_intrinsics_are_accepted() {
    let dir = TempDir::new("colmap_control");
    let path = dir.write(
        "cameras.txt",
        concat!(
            "# Camera list with one line of data per camera:\n",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
            "1 SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n",
            "2 PINHOLE 1280 720 800.0 810.0 640.0 360.0\n",
        ),
    );

    let cameras = colmap::read_cameras_text(&path).expect("CONTROL: valid cameras.txt");
    assert_eq!(cameras.len(), 2, "CONTROL: both cameras must be read");

    let simple = cameras[0]
        .intrinsics
        .expect("CONTROL: SIMPLE_PINHOLE intrinsics");
    assert_eq!(
        (simple.fx, simple.fy, simple.cx, simple.cy),
        (500.0, 500.0, 320.0, 240.0),
        "CONTROL: f must be applied to both axes"
    );

    let pinhole = cameras[1].intrinsics.expect("CONTROL: PINHOLE intrinsics");
    assert_eq!(
        (pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy),
        (800.0, 810.0, 640.0, 360.0)
    );
}

/// CONTROL: a model this reader does not map to intrinsics is still read, and
/// its raw parameters are preserved untouched. The value check must not leak
/// into the models that are stored verbatim.
#[test]
fn colmap_control_other_models_are_unaffected() {
    let dir = TempDir::new("colmap_other_model");
    let path = dir.write(
        "cameras.txt",
        "3 OPENCV 640 480 500.0 500.0 320.0 240.0 0.1 -0.2 0.001 0.002\n",
    );

    let cameras = colmap::read_cameras_text(&path).expect("CONTROL: OPENCV camera");
    assert_eq!(cameras[0].model, "OPENCV");
    assert_eq!(cameras[0].params.len(), 8);
    assert!(cameras[0].intrinsics.is_none());
}

/// DEFECT 2: the reported repro - `PINHOLE` with a negative focal length and a
/// principal point at a third of a pixel from the origin used to return `Ok`
/// with `intrinsics = Some((-500, -500, -1, -1))`. A negative focal length has
/// no inverse, and `CameraIntrinsics::inverse_matrix` then returned the
/// identity, so every 3-D point became its own pixel coordinate and
/// `solve_pnp_dlt` returned a confident pose for a camera with no focal length.
#[test]
fn colmap_negative_focal_length_is_rejected() {
    let dir = TempDir::new("colmap_negative_f");
    let path = dir.write("cameras.txt", "1 PINHOLE 640 480 -500 -500 -1 -1\n");

    let err = colmap::read_cameras_text(&path).expect_err("a negative focal length has no inverse");
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected an InvalidInput, got {err}"
    );
}

/// DEFECT 2: zero is the other half of the same defect - the intrinsic matrix
/// is exactly singular.
#[test]
fn colmap_zero_focal_length_is_rejected() {
    let dir = TempDir::new("colmap_zero_f");
    let path = dir.write("cameras.txt", "1 PINHOLE 640 480 0 0 320 240\n");

    let err = colmap::read_cameras_text(&path).expect_err("a zero focal length is singular");
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected an InvalidInput, got {err}"
    );
}

/// DEFECT 2: `SIMPLE_PINHOLE` carries a single focal length, so it needs the
/// same check.
#[test]
fn colmap_negative_focal_length_simple_pinhole_is_rejected() {
    let dir = TempDir::new("colmap_negative_f_simple");
    let path = dir.write("cameras.txt", "1 SIMPLE_PINHOLE 640 480 -500 320 240\n");

    let err = colmap::read_cameras_text(&path).expect_err("a negative focal length has no inverse");
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected an InvalidInput, got {err}"
    );
}

/// DECISION: an out-of-image principal point is **accepted**, it is only required
/// to be finite. Calibration can legitimately place the principal point outside
/// the sensor (a cropped or rescaled image, a partially masked camera), and a
/// centre point a little outside the frame still yields a perfectly invertible
/// intrinsic matrix. Rejecting it would refuse files that are mathematically
/// fine; what must not survive is a *non-finite* one, which produces NaN pixel
/// coordinates in every projection.
#[test]
fn colmap_out_of_image_principal_point_is_accepted() {
    let dir = TempDir::new("colmap_offimage_pp");
    let path = dir.write("cameras.txt", "1 PINHOLE 640 480 500 500 -100 -80\n");

    let cameras =
        colmap::read_cameras_text(&path).expect("DECISION: an off-image principal point is legal");
    let intr = cameras[0]
        .intrinsics
        .expect("DECISION: intrinsics must still be produced");
    assert_eq!(
        (intr.fx, intr.fy, intr.cx, intr.cy),
        (500.0, 500.0, -100.0, -80.0)
    );
}

/// ...but a non-finite principal point is not. `parse_f64` already refuses a
/// literal `NaN`/`inf` token, so the only forms that reach the value check are
/// literals that overflow `f64` and parse to infinity. The assertion names the
/// offending parameter, so the test also pins down *which* check fired: a
/// rejection that only says "cannot parse as f64" is the lexer, not this.
#[test]
fn colmap_non_finite_principal_point_is_rejected() {
    for (tag, line, param) in [
        (
            "cx_overflow",
            "1 PINHOLE 640 480 500 500 -1e400 240\n",
            "cx",
        ),
        ("cy_overflow", "1 PINHOLE 640 480 500 500 320 1e400\n", "cy"),
        (
            "simple_cx_overflow",
            "1 SIMPLE_PINHOLE 640 480 500 -1e400 240\n",
            "cx",
        ),
    ] {
        let dir = TempDir::new(tag);
        let path = dir.write("cameras.txt", line);
        let err = colmap::read_cameras_text(&path)
            .expect_err("a non-finite principal point must be rejected")
            .to_string();
        assert!(
            err.contains(param) && err.contains("finite"),
            "a non-finite principal point must be rejected by the value check \
             naming {param}, got: {line} -> {err}"
        );
    }
}

/// DECISION: the positive-focal check must discriminate, so a focal length that
/// is valid but *not* the width (as in every real calibration) is accepted while
/// only the sign/zero cases are refused. This is the mirror control for the
/// rejection tests above.
#[test]
fn colmap_small_positive_focal_length_is_accepted() {
    let dir = TempDir::new("colmap_small_f");
    // fx = 1.0 is tiny but strictly positive and invertible.
    let path = dir.write("cameras.txt", "1 PINHOLE 640 480 1 1 320 240\n");

    let cameras = colmap::read_cameras_text(&path).expect("a small positive focal is legal");
    let intr = cameras[0].intrinsics.expect("intrinsics");
    assert_eq!((intr.fx, intr.fy), (1.0, 1.0));
}
