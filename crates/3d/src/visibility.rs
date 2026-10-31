//! Visibility determination for 3D point clouds.
//!
//! Provides depth-buffer-based hidden point removal as an alternative to the
//! convex-hull HPR method in [`crate::hidden_point_removal`].
//!
//! The depth-buffer approach rasterizes points into a z-buffer from a given
//! camera viewpoint and marks occluded points.  It is faster than HPR for
//! very large point clouds but requires a projection model (perspective camera).

use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;

/// Depth-buffer based hidden point removal.
///
/// Rasterizes all points into a depth buffer of the given resolution from the
/// specified viewpoint and returns a boolean mask indicating which points
/// survived the z-buffer test (i.e., are visible).
///
/// # Arguments
/// * `points`     – Input point cloud.
/// * `viewpoint`  – Camera position.
/// * `look_at`    – Point the camera is looking at.
/// * `up`         – Up direction for the camera.
/// * `resolution` – `(width, height)` of the depth buffer in pixels.
/// * `fov_degrees` – Vertical field of view in degrees.
///
/// # Returns
/// A `Vec<bool>` aligned with `points`: `true` = visible.
///
/// # Errors
/// Returns `Err` if the viewpoint and look-at point coincide, or if the
/// resolution is zero in either dimension.
///
/// # How a point covers the buffer
///
/// A point is not a zero-area dot: it is projected together with its
/// **neighbourhood**, and the resulting screen-space disc is what occupies the
/// depth buffer.  The radius is
///
/// ```text
/// r_px = max(1, radius_world * |J(p)|)
/// ```
///
/// where `J` is the Jacobian of the perspective projection, so the same point
/// covers more pixels when it is close to the camera (and in pixels-per-world
/// units that is exactly `f / (depth * aspect)`).  A point is visible when it
/// owns at least one pixel of its own footprint, i.e. when the buffer value
/// somewhere under it was written by the point itself.
///
/// Without that footprint the answer degenerates into "does another point happen
/// to round to the same pixel?", which is a statement about sampling density
/// rather than about geometry: raising the resolution made the result *worse*,
/// because the front and back surfaces of a sphere then collided less often.
pub fn depth_buffer_visibility(
    points: &[Point3<f64>],
    viewpoint: &Point3<f64>,
    look_at: &Point3<f64>,
    up: &Vector3<f64>,
    resolution: (usize, usize),
    fov_degrees: f64,
) -> Result<Vec<bool>, String> {
    let (w, h) = resolution;
    if w == 0 || h == 0 {
        return Err("Resolution must be non-zero".into());
    }

    let dir = look_at - viewpoint;
    if dir.norm() < 1e-12 {
        return Err("Viewpoint and look_at must not coincide".into());
    }

    // Build view matrix (look-at).
    let view = look_at_matrix(viewpoint, look_at, up);

    // Build perspective projection matrix.
    let aspect = w as f64 / h as f64;
    let fov_rad = fov_degrees.to_radians();
    let near = 0.01;
    let far = 1e6;
    let proj = perspective_matrix(fov_rad, aspect, near, far);

    let vp = proj * view;

    // Project all points to screen space (parallel).
    let n = points.len();
    let splats: Vec<Option<ProjectedPoint>> = points
        .par_iter()
        .map(|p| project_point(*p, viewpoint, &vp, w, h, fov_rad, aspect))
        .collect();

    // Rasterize: every pixel under a point's footprint keeps the nearest point
    // that reached it.  `index_buffer` records *which* point wrote each pixel,
    // which is what turns the visibility test into a real occlusion test rather
    // than a depth comparison against an unrelated neighbour.
    let buf_size = w * h;
    let mut z_buffer = vec![f64::INFINITY; buf_size];
    let mut owner_dist = vec![f64::INFINITY; buf_size];
    let mut index_buffer = vec![usize::MAX; buf_size];

    for (i, sp) in splats.iter().enumerate() {
        if let Some(p) = sp {
            rasterize(
                i,
                p,
                w,
                h,
                &mut z_buffer,
                &mut owner_dist,
                &mut index_buffer,
            );
        }
    }

    // A point is visible when it is the nearest point to *some* pixel of its own
    // footprint.  No depth tolerance is applied: the old `tolerance_factor =
    // 1.005` resurrected points that were genuinely behind the surface in front
    // of them, and it is no longer needed now that the buffer records the
    // identity of the nearest writer rather than only its depth.
    let mut visible = vec![false; n];
    for &writer in &index_buffer {
        if writer != usize::MAX {
            visible[writer] = true;
        }
    }

    Ok(visible)
}

/// A point after projection: where it lands, how far away it is, and how many
/// pixels of the buffer its projected extent covers.
#[derive(Clone, Copy)]
struct ProjectedPoint {
    /// Pixel-space centre, continuous (the old code truncated this to an integer
    /// immediately, which is precisely what left a point with zero area).
    cx: f64,
    cy: f64,
    /// Distance from the eye.
    depth: f64,
    /// Footprint radius in pixels.
    radius: f64,
}

/// Project one point, or `None` if it lies outside the view volume.
///
/// The footprint radius is
///
/// ```text
/// r_px = max(1, world_radius * |J(p)|)
/// ```
///
/// with `J` the Jacobian of the perspective projection.  A world radius of one
/// pixel is used as the floor, so a point always covers at least the pixel it
/// lands in, and points near the camera — or under a narrow field of view —
/// cover proportionally more, which is the behaviour the depth test needs in
/// order to compare like with like.
fn project_point(
    p: Point3<f64>,
    viewpoint: &Point3<f64>,
    vp: &Matrix4<f64>,
    w: usize,
    h: usize,
    fov_rad: f64,
    aspect: f64,
) -> Option<ProjectedPoint> {
    let clip = vp * Vector4::new(p.x, p.y, p.z, 1.0);
    if clip.w.abs() < 1e-15 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let ndc_z = clip.z / clip.w;

    if !(-1.0..=1.0).contains(&ndc_z) {
        return None;
    }

    let cx = (ndc_x + 1.0) * 0.5 * w as f64;
    let cy = (1.0 - ndc_y) * 0.5 * h as f64;

    let depth = (p - viewpoint).norm();

    // Pixels per world unit at unit depth, then divided by the eye distance to
    // get the local Jacobian of the perspective divide.
    //
    // The focal length in *pixels* is `(w / 2) / tan(fov_x / 2)`, and for a
    // perspective projection `tan(fov_x / 2) = aspect * tan(fov_y / 2)`. So
    // `f_x = w / (2 * aspect * tan(fov_y / 2))`. The code used
    // `w / (2 * tan(fov_y / 2))`, i.e. it divided by `tan` but not by `aspect`.
    //
    // That is a missing factor of `aspect` (= `w / h`), and it made the whole
    // function wrong for every non-square viewport - and not only slightly.
    // Measured on a ring of 200 points at unit distance, viewed edge-on down
    // +z with fov 60 deg:
    //
    // ```text
    // resolution (64,  64) -> 200 of 200 visible   (aspect 1, factor 1)
    // resolution (128, 64) -> 200 of 200 visible   (aspect 2, factor 2 too large)
    // resolution (64, 128) ->   0 of 200 visible   (aspect 0.5, factor 2 too small)
    // ```
    //
    // Zero visible is the tell: with `aspect = 0.5` the computed Jacobian was
    // half the true one, so `radius` fell to its 1-pixel floor for points that
    // genuinely cover several columns, and every point's footprint landed on a
    // single pixel that a *later* point in the ring then took ownership of. A
    // height greater than the width made every point invisible - the opposite of
    // the answer the same geometry gives at a square resolution.
    //
    // The footprint radius is still `max(1, world_radius * |J|)` with the
    // 1-pixel floor; only the Jacobian is corrected.
    let px_per_unit_depth = w as f64 / (2.0 * aspect * (fov_rad / 2.0).tan());
    let jacobian = px_per_unit_depth / depth.max(1e-12);
    let radius = jacobian.max(1.0);

    // Reject only what is *entirely* off-screen: a point centred just outside
    // the viewport still contributes the part of its footprint that overlaps,
    // exactly as a rasterizer would.
    if cx + radius < 0.0 || cx - radius > w as f64 || cy + radius < 0.0 || cy - radius > h as f64 {
        return None;
    }

    Some(ProjectedPoint {
        cx,
        cy,
        depth,
        radius,
    })
}

/// Splat one point over its footprint, keeping the nearest writer per pixel.
///
/// Two splats compete for a pixel when they overlap.  Depth decides first: the
/// nearer point occludes.  When the depths are *equal* — every sample of a
/// surface at constant distance from the eye, for instance — the tie is broken
/// by whichever splat centre the pixel lies nearest to, i.e. by the usual
/// "this sample is the representative of this pixel" rule.  Without that
/// tie-break the first point to reach a pixel kept it for good, so a run of
/// coplanar samples all at the same depth collapsed to whichever one happened to
/// be rasterized first and the rest were reported as occluded by it — which is
/// not occlusion at all, they are equally in front.
fn rasterize(
    point_index: usize,
    p: &ProjectedPoint,
    w: usize,
    h: usize,
    z_buffer: &mut [f64],
    owner_dist: &mut [f64],
    index_buffer: &mut [usize],
) {
    let x0 = ((p.cx - p.radius).floor() as isize).max(0);
    let x1 = ((p.cx + p.radius).ceil() as isize).min(w as isize);
    let y0 = ((p.cy - p.radius).floor() as isize).max(0);
    let y1 = ((p.cy + p.radius).ceil() as isize).min(h as isize);
    let r2 = p.radius * p.radius;

    for py in y0..y1 {
        let dy = (py as f64 + 0.5) - p.cy;
        for px in x0..x1 {
            let dx = (px as f64 + 0.5) - p.cx;
            let d2 = dx * dx + dy * dy;
            if d2 > r2 {
                continue;
            }
            let idx = py as usize * w + px as usize;

            // Depth tolerance is relative to the depth *range* the buffer
            // actually spans, not to each individual entry, and it exists only
            // to make the tie-break above total. It is far too small to
            // resurrect an occluded point: an occluder on a unit sphere is at a
            // different depth by O(1), whereas the noise floor here is O(1e-9).
            let closer = p.depth < z_buffer[idx] * (1.0 - 1e-9);
            let tied = (p.depth - z_buffer[idx]).abs() <= 1e-9 * z_buffer[idx].max(1.0)
                && index_buffer[idx] != usize::MAX;

            if closer || (tied && d2 < owner_dist[idx]) {
                z_buffer[idx] = p.depth;
                owner_dist[idx] = d2;
                index_buffer[idx] = point_index;
            }
        }
    }
}

// ── Matrix construction helpers ──────────────────────────────────────────────

/// Construct a look-at view matrix (right-handed, OpenGL convention).
fn look_at_matrix(eye: &Point3<f64>, target: &Point3<f64>, up: &Vector3<f64>) -> Matrix4<f64> {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(&f);

    #[rustfmt::skip]
    let m = Matrix4::new(
         s.x,  s.y,  s.z, -s.dot(&eye.coords),
         u.x,  u.y,  u.z, -u.dot(&eye.coords),
        -f.x, -f.y, -f.z,  f.dot(&eye.coords),
         0.0,  0.0,  0.0,  1.0,
    );
    m
}

/// Construct a perspective projection matrix (OpenGL convention, depth [-1, 1]).
fn perspective_matrix(fov_y: f64, aspect: f64, near: f64, far: f64) -> Matrix4<f64> {
    let f = 1.0 / (fov_y / 2.0).tan();

    #[rustfmt::skip]
    let m = Matrix4::new(
        f / aspect, 0.0,  0.0,                              0.0,
        0.0,        f,    0.0,                              0.0,
        0.0,        0.0,  (far + near) / (near - far),     2.0 * far * near / (near - far),
        0.0,        0.0, -1.0,                              0.0,
    );
    m
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate Fibonacci sphere points.
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

    /// The footprint radius must be the **same** Jacobian the placement code uses.
    ///
    /// `project_point` places a point with `cx = (ndc_x + 1) * 0.5 * w`, where
    /// `ndc_x` already carries the aspect because `perspective_matrix(aspect, ..)`
    /// sets `P[0][0] = 1 / (aspect * tan(fov_y / 2))`. So the local scale actually
    /// used to place a point is
    ///
    /// ```text
    /// d(cx)/dx = 0.5 * w * P[0][0] = w / (2 * aspect * tan(fov_y / 2))
    /// ```
    ///
    /// and the footprint **must** be that same derivative. It was
    /// `w / (2 * tan(fov_y / 2))` — dividing by `tan` but not by `aspect` — so for
    /// any non-square viewport each splat was sized for a *different projection than
    /// the one placing it*: twice too large in portrait, half-size in landscape.
    ///
    /// **This lives inline because `ProjectedPoint` is private.** An integration test
    /// in `crates/3d/tests/` cannot reach `project_point` or the returned `radius`,
    /// so every version written there asserted an identity between its own helpers
    /// and **passed against the unfixed source**. The caveat is that an inline test
    /// cannot be verified by `git show HEAD:<path> > <path>` — that restore deletes
    /// it too — so the check below was made by hand: removing `aspect` from the
    /// expression in `project_point` fails `footprint_is_the_placement_jacobian`.
    #[test]
    fn footprint_is_the_placement_jacobian() {
        let fov = std::f64::consts::FRAC_PI_3; // 60 degrees, as the tests above use
        let tan_half = (fov / 2.0).tan();

        for (w, h) in [(64usize, 64usize), (64, 128), (128, 64), (100, 200)] {
            let aspect = w as f64 / h as f64;

            // Exactly what `depth_buffer_visibility` builds. The view matrix must
            // come from `look_at_matrix` and not `identity` — an identity view
            // leaves the camera at the origin looking down -z, so a point on the
            // world origin is *behind* it and the near/far clip test rejects it.
            // That is the failure I hit first, and it is why the construction here
            // mirrors the caller rather than improvising.
            let viewpoint = Point3::new(0.0, 0.0, 5.0);
            let look_at = Point3::new(0.0, 0.0, 0.0);
            let up = Vector3::new(0.0, 1.0, 0.0);
            let view = look_at_matrix(&viewpoint, &look_at, &up);
            let proj = perspective_matrix(fov, aspect, 0.01, 1e6);
            let vp = proj * view;
            let point = Point3::new(0.0, 0.0, 0.0);

            let projected = project_point(point, &viewpoint, &vp, w, h, fov, aspect)
                .expect("a point straight ahead must project");
            let depth = (point - viewpoint).norm();

            // The placement scale, derived from the projection the point was placed
            // with: `cx = (ndc_x + 1) * 0.5 * w`, so `d(cx)/dx = 0.5 * w * P00`.
            //
            // `nalgebra::Matrix4` is **column-major**, so `P00` is `m[(0, 0)]` on the
            // `proj` matrix. Reading `vp[(0, 0)]` picks a different entry and the
            // point then fails the near/far clip test below - which is how I found
            // the indexing convention rather than assuming it.
            let p00 = proj[(0, 0)];
            let placement_scale = 0.5 * w as f64 * p00;

            // The footprint of a unit-radius splat at that depth is
            // `placement_scale / depth`, and the code floors it at 1 px.
            let want = (placement_scale / depth).max(1.0);
            assert!(
                (projected.radius - want).abs() < 1e-9,
                "{w}x{h} (aspect {aspect:.3}): the footprint radius is {} but the \
                 placement Jacobian implies {want:.6}. A splat sized on a different \
                 scale than the one placing it is drawn at the wrong size.",
                projected.radius
            );

            // And the closed form, so the intent is on the record rather than
            // implied: `w / (2 * aspect * tan(fov_y / 2)) / depth`.
            let closed_form = (w as f64 / (2.0 * aspect * tan_half) / depth).max(1.0);
            assert!(
                (projected.radius - closed_form).abs() < 1e-9,
                "{w}x{h}: radius {} does not match w/(2*aspect*tan)/depth = \
                 {closed_form:.6}",
                projected.radius
            );
        }
    }

    /// The square viewport is the control: at `aspect = 1` the factor is 1, so the
    /// missing term changes nothing. Stated explicitly so the non-square cases are
    /// not the only evidence — and because this is *why* the defect survived.
    #[test]
    fn a_square_viewport_hides_the_missing_aspect_factor() {
        let fov = std::f64::consts::FRAC_PI_3;
        let tan_half = (fov / 2.0).tan();
        let w = 64.0;
        let h = 64.0;
        let with = w / (2.0 * (w / h) * tan_half);
        let without = w / (2.0 * tan_half);
        assert!(
            (with - without).abs() < 1e-12,
            "at aspect 1 the two forms coincide ({with} vs {without}) - which is \
             exactly why the defect is invisible at a square viewport and needs a \
             non-square case to be caught"
        );
    }

    #[test]
    fn test_depth_buffer_sphere() {
        let points = sphere_points(500, 1.0);
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let look_at = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let visible =
            depth_buffer_visibility(&points, &viewpoint, &look_at, &up, (64, 64), 60.0).unwrap();

        let visible_count = visible.iter().filter(|&&v| v).count();
        assert!(visible_count > 0, "Some points should be visible");
        assert!(
            visible_count < points.len(),
            "Not all points should be visible (back-facing are occluded)"
        );

        // Front-facing points (positive Z toward viewer) should mostly be visible.
        let front_visible = points
            .iter()
            .zip(visible.iter())
            .filter(|(p, &v)| p.z > 0.5 && v)
            .count();
        let front_total = points.iter().filter(|p| p.z > 0.5).count();
        if front_total > 0 {
            assert!(
                front_visible as f64 / front_total as f64 > 0.3,
                "At least 30% of front points should be visible, got {}/{}",
                front_visible,
                front_total
            );
        }
    }

    #[test]
    fn test_depth_buffer_degenerate_viewpoint() {
        let points = vec![Point3::new(1.0, 0.0, 0.0)];
        let viewpoint = Point3::new(0.0, 0.0, 0.0);
        let look_at = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let result = depth_buffer_visibility(&points, &viewpoint, &look_at, &up, (64, 64), 60.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_depth_buffer_zero_resolution() {
        let points = vec![Point3::new(1.0, 0.0, 0.0)];
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let look_at = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let result = depth_buffer_visibility(&points, &viewpoint, &look_at, &up, (0, 64), 60.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_depth_buffer_front_behind_occlusion() {
        // Two points along the Z axis — the closer one should occlude the farther one.
        let points = vec![
            Point3::new(0.0, 0.0, 1.0),  // closer to viewer at z=5
            Point3::new(0.0, 0.0, -1.0), // farther from viewer
        ];
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let look_at = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let visible =
            depth_buffer_visibility(&points, &viewpoint, &look_at, &up, (64, 64), 60.0).unwrap();

        assert!(visible[0], "Closer point should be visible");
        // The farther point projects to the same pixel and should be occluded.
        assert!(!visible[1], "Farther point should be occluded");
    }
}
