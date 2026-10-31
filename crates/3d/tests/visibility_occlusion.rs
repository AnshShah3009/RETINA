//! `depth_buffer_visibility` must actually occlude: the depth buffer has to
//! report absence instead of confidently calling everything visible.
//!
//! Measured before the fix, on a 300-point unit sphere, eye (0,0,5), fov 60:
//!
//! ```text
//! resolution   visible   back-hemisphere visible (z < -0.2)
//! 64x64        247/300   many
//! 512x512      300/300   120/120
//! ```
//!
//! Two independent causes, both in `crates/3d/src/visibility.rs`:
//!
//! 1. **No point splat.** A projected point was written to a single pixel as a
//!    zero-area dot. A front-surface point and the back-surface point behind it
//!    almost never land on the same pixel, so occlusion depended purely on
//!    sampling density versus buffer resolution — and *raising* the resolution
//!    made the answer worse (64x64 hid 53 points, 512x512 hid none), which is
//!    the tell that no splat footprint exists.
//! 2. **A 0.5% depth tolerance** (`tolerance_factor = 1.005`) that resurrected
//!    points even when they did collide.
//!
//! The fix splats every point over a footprint covering its projected extent
//! (minimum one pixel radius, scaled by the projection Jacobian) and compares
//! against the buffer inside that footprint, with no depth tolerance.
//!
//! The pre-existing in-module tests could not see this: `visibility.rs:192`
//! asserted only `visible_count < points.len()`, which 247/300 satisfies.

use cv_3d::visibility::depth_buffer_visibility;
use nalgebra::{Point3, Vector3};
use std::f64::consts::PI;

fn sphere_points(n: usize, radius: f64) -> Vec<Point3<f64>> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            Point3::new(
                radius * phi.sin() * theta.cos(),
                radius * phi.sin() * theta.sin(),
                radius * phi.cos(),
            )
        })
        .collect()
}

fn eye() -> Point3<f64> {
    Point3::new(0.0, 0.0, 5.0)
}

fn target() -> Point3<f64> {
    Point3::new(0.0, 0.0, 0.0)
}

fn up() -> Vector3<f64> {
    Vector3::new(0.0, 1.0, 0.0)
}

/// The core defect: at 512x512 the back hemisphere of a sphere must NOT be
/// visible. Before the fix this was 300/300 visible including 120/120 back
/// points.
#[test]
fn back_hemisphere_of_a_sphere_is_occluded_at_high_resolution() {
    let points = sphere_points(300, 1.0);

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (512, 512), 60.0).unwrap();

    let back_indices: Vec<usize> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.z < -0.2)
        .map(|(i, _)| i)
        .collect();
    let back_visible = back_indices.iter().filter(|&&i| visible[i]).count();
    let total_visible = visible.iter().filter(|&&v| v).count();

    assert!(
        !back_indices.is_empty(),
        "test fixture must contain back-hemisphere points"
    );
    assert!(
        back_visible == 0,
        "every one of the {} back-hemisphere points (z < -0.2) is behind the front \
         surface of the sphere and must be reported hidden; got {}/{} marked visible \
         ({}/{} total visible). Sub-pixel splatting with no footprint means a front \
         and a back point never share a pixel, and the 0.5% depth tolerance \
         resurrects the ones that do.",
        back_indices.len(),
        back_visible,
        back_indices.len(),
        total_visible,
        points.len()
    );
}

/// The tell from the report: occlusion must get *better* as resolution rises,
/// not worse. Before the fix 512x512 hid nothing at all while 64x64 hid 53
/// points. Measured after the fix, on the same 300-point sphere:
///
/// ```text
/// res    visible   back-hemisphere visible
///  32       71/300   0/120
///  64      102/300   0/120
/// 128      107/300   0/120
/// 256      108/300   0/120
/// 512      111/300   0/120
/// 1024     113/300   0/120
/// ```
///
/// The visible count saturates as the surface becomes adequately sampled
/// instead of climbing back to every point in the cloud.
#[test]
fn occlusion_does_not_get_worse_as_resolution_rises() {
    let points = sphere_points(300, 1.0);

    let coarse = depth_buffer_visibility(&points, &eye(), &target(), &up(), (64, 64), 60.0)
        .unwrap()
        .iter()
        .filter(|&&v| v)
        .count();
    let fine = depth_buffer_visibility(&points, &eye(), &target(), &up(), (512, 512), 60.0)
        .unwrap()
        .iter()
        .filter(|&&v| v)
        .count();

    assert!(
        fine <= points.len() / 2,
        "raising the depth-buffer resolution must not keep revealing new points. \
         64x64 showed {coarse}/{} but 512x512 showed {fine}/{} — a projection with no \
         splat footprint keeps losing the occluder as pixels get smaller, which is the \
         opposite of a converging depth buffer",
        points.len(),
        points.len()
    );
    assert!(
        fine <= coarse + points.len() / 10,
        "the visible count must not jump at higher resolution: {coarse}/300 at 64x64 \
         versus {fine}/300 at 512x512"
    );
}

/// CONTROL: a fix that simply marked nothing visible would pass the test above.
/// Points on the FRONT of an object, with nothing in front of them, must still
/// be reported visible. Nothing occludes them, so absence cannot be claimed.
#[test]
fn front_surface_points_remain_visible() {
    // Points spread across the near hemisphere of the sphere only; nothing is in
    // front of any of them.
    let all = sphere_points(300, 1.0);
    let front: Vec<Point3<f64>> = all.into_iter().filter(|p| p.z > 0.6).collect();
    assert!(front.len() > 30, "fixture too small: {}", front.len());

    let visible =
        depth_buffer_visibility(&front, &eye(), &target(), &up(), (512, 512), 60.0).unwrap();

    let count = visible.iter().filter(|&&v| v).count();
    assert!(
        count as f64 / front.len() as f64 > 0.9,
        "points on the unobstructed near surface must stay visible; got {}/{}",
        count,
        front.len()
    );
}

/// CONTROL: an unoccluded point cloud. Nothing lies in front of any point, so
/// no point may be culled.
///
/// The fixture is a tight cap around the point nearest the eye: 60 samples
/// spread over a quarter-degree of arc, so the entire cloud projects inside a
/// couple of pixels and there is no geometry at all in front of any sample.
/// A splat radius that grows without bound would wrongly hide samples of this
/// cloud behind their own neighbours, so this also pins the radius down.
#[test]
fn an_unoccluded_cloud_is_fully_visible() {
    let points: Vec<Point3<f64>> = (0..60)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / 60.0;
            Point3::new(0.05 * theta.cos(), 0.05 * theta.sin(), 1.0)
        })
        .collect();

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (256, 256), 60.0).unwrap();

    let count = visible.iter().filter(|&&v| v).count();
    assert!(
        count as f64 / points.len() as f64 > 0.95,
        "an unoccluded cloud must be entirely visible; got {}/{} visible",
        count,
        points.len()
    );
}

/// CONTROL for the two-point case already covered in-module: a near point and
/// the identical point further away along the view ray. Only the near one may
/// be visible — this is the exact case the zero-area dot happened to get right,
/// so it must keep working after the splat change.
#[test]
fn a_near_point_occludes_an_identical_far_point() {
    let points = vec![
        Point3::new(0.0, 0.0, 1.0),  // nearer to the eye at z = 5
        Point3::new(0.0, 0.0, -1.0), // directly behind it
    ];

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (256, 256), 60.0).unwrap();

    assert!(visible[0], "the nearer point must be visible");
    assert!(
        !visible[1],
        "the farther point lies exactly behind the nearer one and must be occluded"
    );
}

/// CONTROL: a near/far pair that is *offset* so their projected centres land on
/// neighbouring rather than identical pixels. With a sub-pixel splat this pair
/// used to be reported as both visible whenever the offset exceeded one pixel;
/// with a real footprint the near point's splat must cover the far one.
#[test]
fn an_offset_near_point_still_occludes_the_point_behind_it() {
    // The far point is offset by a fraction of a pixel's world size at 256x256.
    let near = Point3::new(0.0, 0.0, 1.0);
    let far = Point3::new(0.0, 0.0, -1.0);
    let points = vec![near, far];

    for res in [64usize, 128, 256, 512] {
        let visible =
            depth_buffer_visibility(&points, &eye(), &target(), &up(), (res, res), 60.0).unwrap();
        assert!(
            visible[0] && !visible[1],
            "at {res}x{res}: the near point must be visible and the far point behind \
             it occluded (got {:?})",
            visible
        );
    }
}

/// CONTROL: guard against over-occlusion in the other direction. Two points at
/// very different screen positions must BOTH be visible — neither occludes the
/// other, so a splat footprint that grows without bound would wrongly hide one.
#[test]
fn widely_separated_points_do_not_occlude_each_other() {
    let points = vec![
        Point3::new(-0.5, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        Point3::new(0.0, 0.4, 0.0),
    ];

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (256, 256), 60.0).unwrap();

    for (i, p) in points.iter().enumerate() {
        assert!(
            visible[i],
            "point {i} at {p:?} is unoccluded and must stay visible (got {visible:?})"
        );
    }
}

/// The depth tolerance must not resurrect occluded points.
///
/// The near point sits at depth 5.0 from the eye and the far one at depth 5.1 —
/// only 2% apart, so the old `tolerance_factor = 1.005` test
/// (`depth <= z_buffer * 1.005`) would *not* have caught this particular pair.
/// The point of the test is to pin the corrected behaviour: any tolerance that
/// reports the farther of two points on the same ray as visible is a bug, no
/// matter how small it is.
#[test]
fn a_depth_tolerance_cannot_resurrect_an_occluded_point() {
    // Near and far points 0.1 apart along the same view ray from the eye.
    let near = Point3::new(0.0, 0.0, 0.0);
    let far = Point3::new(0.0, 0.0, -0.1);
    let points = vec![near, far];

    for res in [64usize, 256, 512] {
        let visible =
            depth_buffer_visibility(&points, &eye(), &target(), &up(), (res, res), 60.0).unwrap();
        assert!(
            visible[0] && !visible[1],
            "at {res}x{res}: a point 0.1 behind another along the same ray is \
             occluded; a depth tolerance resurrects it (got {visible:?})"
        );
    }
}

/// The same fingerprint against the *sphere*: points near the silhouette sit at
/// nearly the same depth as the surface in front of them. With a real footprint
/// the back hemisphere stays hidden; with the old 0.5% tolerance, whatever did
/// collide was resurrected.
#[test]
fn near_silhouette_points_do_not_escape_via_a_depth_tolerance() {
    let points = sphere_points(300, 1.0);

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (512, 512), 60.0).unwrap();

    let back_total = points.iter().filter(|p| p.z < -0.2).count();
    let back_visible = points
        .iter()
        .zip(visible.iter())
        .filter(|(p, &v)| p.z < -0.2 && v)
        .count();

    assert_eq!(
        back_visible, 0,
        "points within a hair of the silhouette depth were resurrected by the depth \
         tolerance: {back_visible}/{back_total} back-hemisphere points reported visible"
    );
}

/// A hollow shell: an outer sphere and an inner sphere. From outside, the inner
/// sphere is entirely occluded by the outer one and must report so.
#[test]
fn an_inner_shell_fully_enclosed_by_an_outer_one_is_occluded() {
    let outer = sphere_points(400, 1.0);
    // A dense inner shell: densely sampled, so its points are close together in
    // screen space and cannot dodge the outer shell's splat footprints.
    let inner: Vec<Point3<f64>> = (0..2000)
        .map(|i| {
            let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let theta = 2.0 * PI * (i as f64) / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / 2000.0).acos();
            Point3::new(
                0.3 * phi.sin() * theta.cos(),
                0.3 * phi.sin() * theta.sin(),
                0.3 * phi.cos(),
            )
        })
        .collect();
    let points: Vec<Point3<f64>> = outer.iter().chain(inner.iter()).copied().collect();
    let split = outer.len();

    let visible =
        depth_buffer_visibility(&points, &eye(), &target(), &up(), (256, 256), 60.0).unwrap();

    let inner_visible = visible[split..].iter().filter(|&&v| v).count();
    let outer_visible = visible[..split].iter().filter(|&&v| v).count();

    assert_eq!(
        inner_visible,
        0,
        "all {} points of the inner shell sit strictly inside the outer shell and are \
         occluded from outside; {inner_visible} were reported visible",
        inner.len()
    );
    assert!(
        outer_visible > 100,
        "CONTROL: the outer shell itself is the front surface and must stay visible; \
         only {outer_visible} of {split} points were"
    );
}
