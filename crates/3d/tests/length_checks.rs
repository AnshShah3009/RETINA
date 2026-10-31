//! Optional and parallel arrays must be length-checked.
//!
//! Four defects, all found by auditing untested code, all with the same shape: a
//! function indexes a slice with an index derived from a *different* array and
//! panics on a length mismatch.
//!
//! - `gpu::tsdf::integrate_depth` checked the volume and weights lengths but not
//!   the depth image's, so `&[1.0]` with a 2x2 image indexed element 3.
//! - `TSDFVolume::integrate_ctx` indexed `depth_image[v * width + u]` and
//!   `color_image[idx]` with no length check at all.
//! - `odometry::downsample_depth` clamped its index to `width * height` - which
//!   says nothing about the slice's length.
//! - `filters::voxel_downsample` indexed `normals[i]` and `colors[i]` where `i`
//!   is a point index. Every other `filters` function validates its optional
//!   inputs; this one did not.
//!
//! A panic on a short input is not a crash the caller can handle - it takes the
//! process down - and each of these is a public API.

use nalgebra::{Point3, Vector3};

// `filters::voxel_downsample` works in f64, unlike the other functions here.
type P = Point3<f64>;
type V = Vector3<f64>;

/// Every one of these panicked before the fix.
#[test]
fn gpu_integrate_depth_rejects_a_short_depth_image() {
    let mut vol = vec![0.0f32; 8]; // 2x2x2
    let mut weights = vec![0.0f32; 8];
    let i = [100.0f32, 100.0, 1.0, 1.0];
    let p = nalgebra::Matrix4::identity();

    let r = cv_3d::gpu::tsdf::integrate_depth(
        &[1.0], // one element, 2x2 declared
        2,
        2,
        &p,
        &i,
        &mut vol,
        &mut weights,
        0.1,
        0.5,
    );
    let msg = r.expect_err("a depth image shorter than 2x2 must be rejected, not                           indexed past the end");
    assert!(
        msg.contains("depth image"),
        "the error should name the depth image, got: {msg}"
    );
}

#[test]
fn gpu_integrate_depth_accepts_a_correctly_sized_image() {
    let mut vol = vec![0.0f32; 8];
    let mut weights = vec![0.0f32; 8];
    let i = [100.0f32, 100.0, 1.0, 1.0];
    let p = nalgebra::Matrix4::identity();
    let depth = vec![1.0f32; 4]; // exactly 2x2

    assert!(
        cv_3d::gpu::tsdf::integrate_depth(&depth, 2, 2, &p, &i, &mut vol, &mut weights, 0.1, 0.5)
            .is_ok(),
        "a correctly sized image must still be accepted"
    );
}

#[test]
fn odometry_does_not_panic_on_a_short_depth_slice() {
    // 64x64 declared, ten elements supplied. Previously indexed 2048 and
    // panicked.
    let short = vec![1.0f32; 10];
    let k = cv_3d::tsdf::CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let result = std::panic::catch_unwind(move || {
        cv_3d::odometry::compute_rgbd_odometry(
            &short,
            &short,
            None,
            None,
            &k,
            64,
            64,
            cv_3d::odometry::OdometryMethod::PointToPlane,
        )
    });
    assert!(
        result.is_ok(),
        "compute_rgbd_odometry panicked on a depth slice shorter than \
         width*height"
    );
}

#[test]
fn voxel_downsample_tolerates_fewer_normals_than_points() {
    let points = vec![P::new(0.0, 0.0, 0.0), P::new(0.1, 0.0, 0.0)];
    let one_normal = vec![V::new(0.0, 0.0, 1.0)];

    let result = std::panic::catch_unwind(move || {
        cv_3d::filters::voxel_downsample(&points, Some(&one_normal), None, 2.0)
    });
    assert!(
        result.is_ok(),
        "voxel_downsample panicked indexing one normal with two point indices"
    );
}

#[test]
fn voxel_downsample_tolerates_fewer_colours_than_points() {
    let points = vec![P::new(0.0, 0.0, 0.0), P::new(0.1, 0.0, 0.0)];
    let one_color = vec![V::new(1.0, 0.0, 0.0)];

    let result = std::panic::catch_unwind(move || {
        cv_3d::filters::voxel_downsample(&points, None, Some(&one_color), 2.0)
    });
    assert!(
        result.is_ok(),
        "voxel_downsample panicked indexing one colour with two point indices"
    );
}

#[test]
fn voxel_downsample_still_works_with_matching_lengths() {
    let points = vec![P::new(0.0, 0.0, 0.0), P::new(0.1, 0.0, 0.0)];
    let normals = vec![V::new(0.0, 0.0, 1.0); 2];
    let colors = vec![V::new(1.0, 0.5, 0.25); 2];

    let r = cv_3d::filters::voxel_downsample(&points, Some(&normals), Some(&colors), 2.0);
    assert_eq!(
        r.points.len(),
        1,
        "both points are in one voxel of size 2.0"
    );
    assert!(r.normals.is_some(), "normals should be carried through");
    assert!(r.colors.is_some(), "colours should be carried through");
}
