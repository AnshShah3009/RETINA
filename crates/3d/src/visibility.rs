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
    let screen_coords: Vec<Option<(usize, usize, f64)>> = points
        .par_iter()
        .map(|p| {
            let clip = vp * Vector4::new(p.x, p.y, p.z, 1.0);
            if clip.w.abs() < 1e-15 {
                return None;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let ndc_z = clip.z / clip.w;

            let sx = ((ndc_x + 1.0) * 0.5 * w as f64) as isize;
            let sy = ((1.0 - ndc_y) * 0.5 * h as f64) as isize;

            if sx < 0
                || sx >= w as isize
                || sy < 0
                || sy >= h as isize
                || !(-1.0..=1.0).contains(&ndc_z)
            {
                return None;
            }

            let depth = (p - viewpoint).norm();
            Some((sx as usize, sy as usize, depth))
        })
        .collect();

    // Z-buffer: for each pixel, track the closest point index.
    let buf_size = w * h;
    let mut z_buffer = vec![f64::MAX; buf_size];
    let mut index_buffer = vec![usize::MAX; buf_size];

    for (i, sc) in screen_coords.iter().enumerate() {
        if let Some((sx, sy, depth)) = sc {
            let idx = sy * w + sx;
            if *depth < z_buffer[idx] {
                z_buffer[idx] = *depth;
                index_buffer[idx] = i;
            }
        }
    }

    // Mark visible points: those that are the closest at their pixel,
    // or within a small tolerance of the closest.
    let tolerance_factor = 1.005; // 0.5% depth tolerance
    let mut visible = vec![false; n];

    for (i, sc) in screen_coords.iter().enumerate() {
        if let Some((sx, sy, depth)) = sc {
            let idx = sy * w + sx;
            if *depth <= z_buffer[idx] * tolerance_factor {
                visible[i] = true;
            }
        }
    }

    Ok(visible)
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
