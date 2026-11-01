//! `database_image_from_colmap` must convert COLMAP's pose convention.
//!
//! `cv_io::datasets::colmap::Image::pose` is **camera-to-world** (`translation`
//! is the camera centre), because the COLMAP reader inverts the world-to-camera
//! matrix stored in `images.txt`. `DatabaseImage::pose`, however, is documented
//! as **world-to-camera** (`x_cam = R * x_world + t`), which is what every other
//! producer in this crate stores (`benchmark.rs` uses `pose_cw`,
//! `synthetic.rs` uses the camera's world-to-camera pose).
//!
//! The bridge copied the COLMAP pose through unchanged. That is invisible for
//! the identity pose (which the existing bridge test uses), but for any real
//! rotation and translation the stored pose is the exact opposite of what the
//! field promises: its `translation` is the camera centre instead of `-R * C`,
//! and its rotation maps camera axes into world instead of the other way round.

use cv_core::Pose;
use cv_io::datasets::colmap::{Image, Point2D};
use cv_localization::database_image_from_colmap;
use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::collections::HashMap;

fn assert_pose_close(got: &Pose, want: &Pose, what: &str) {
    let rotation_diff = (got.rotation_matrix() - want.rotation_matrix())
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        rotation_diff < 1e-12,
        "{what}: rotation off by {:.3e}\n got {:?}\nwant {:?}",
        rotation_diff,
        got.rotation_matrix(),
        want.rotation_matrix()
    );
    let translation_diff = (got.translation - want.translation).norm();
    assert!(
        translation_diff < 1e-12,
        "{what}: translation off by {:.3e} ({} vs {})",
        translation_diff,
        got.translation,
        want.translation
    );
}

/// A COLMAP image posed at a real (non-identity) transform must end up in the
/// `DatabaseImage` with the *inverse* pose, so that multiplying its rotation and
/// adding its translation takes a world point into the camera.
#[test]
fn colmap_pose_is_converted_from_camera_to_world() {
    let camera_to_world = Pose::from_quat_translation(
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.2),
        Vector3::new(1.0, -2.0, 3.0),
    );
    let colmap_image = Image {
        id: 4,
        pose: camera_to_world,
        camera_id: 1,
        name: "frame.png".to_string(),
        points2d: vec![Point2D {
            x: 10.0,
            y: 20.0,
            point3d_id: 7,
        }],
    };

    let image = database_image_from_colmap(&colmap_image, &HashMap::new());
    let stored = image
        .pose
        .expect("a COLMAP image carries a registered pose");

    // The documented convention of `DatabaseImage::pose`.
    let world_to_camera = camera_to_world.inverse();
    assert_pose_close(&stored, &world_to_camera, "COLMAP bridge pose");

    // Independent check: the stored pose must send a world point to the same
    // camera coordinates as the inverse of COLMAP's camera-to-world pose.
    let world_point = Point3::new(4.0, -5.0, 6.0);
    let via_stored = stored.rotation * world_point.coords + stored.translation;
    let via_inverse = world_to_camera.rotation * world_point.coords + world_to_camera.translation;
    assert!(
        (via_stored - via_inverse).norm() < 1e-12,
        "world point maps to {via_stored} instead of {via_inverse}"
    );

    // `translation` in world-to-camera coordinates is `-R * C`, not the camera
    // centre `C` that COLMAP stores.
    let centre = camera_to_world.translation;
    assert!(
        (stored.translation - centre).norm() > 1e-3,
        "the stored translation is still the camera centre {centre}"
    );

    // Observations are independent of the pose convention and must survive.
    assert_eq!(image.landmarks, vec![None]);
    assert!((image.keypoints[0].x - 10.0).abs() < 1e-12);
}

/// Control: an identity COLMAP pose is its own inverse, so it must still come
/// through unchanged (this is what the pre-existing bridge test happened to
/// cover, and why the convention error was invisible).
#[test]
fn identity_colmap_pose_is_unchanged() {
    let colmap_image = Image {
        id: 1,
        pose: Pose::identity(),
        camera_id: 1,
        name: "frame.png".to_string(),
        points2d: Vec::new(),
    };
    let image = database_image_from_colmap(&colmap_image, &HashMap::new());
    assert_pose_close(&image.pose.unwrap(), &Pose::identity(), "identity pose");
}
