//! A Gaussian splat with no invertible covariance must be skipped, not painted.
//!
//! `inverse_covariance` and `ProjectedGaussian::inv_cov_2d` both returned
//! `try_inverse().unwrap_or(Matrix3::zeros())`. The zero matrix is not a neutral
//! fallback - it is read as "this splat has no extent", and the rasterizer
//! evaluates
//!
//! ```text
//! mahalanobis = a·dx² + 2b·dx·dy + c·dy²
//! alpha       = exp(-0.5 · mahalanobis) · opacity
//! ```
//!
//! from it. Every coefficient of a zero matrix is 0, so `mahalanobis` was 0 at
//! **every** pixel of the splat's tile, and `alpha` came out at full opacity:
//! measured 0.8 at pixel offsets of (0,0), (10,10), (100,100) and (1000,1000)
//! alike. A splat with no defined footprint painted its entire tile - the
//! opposite of vanishing, and invisible as a bug.
//!
//! `Gaussian::new` clamps scale to `>= 1e-4`, so this only arises from a splat
//! built outside that constructor: a deserialised file, or a field assigned
//! directly.

use cv_rendering::gaussian_splatting::{Gaussian, ProjectedGaussian};
use nalgebra::{Matrix3, Point3, Vector2, Vector3, Vector4};

fn splat_with_scale(s: Vector3<f32>) -> Gaussian {
    Gaussian::new(
        Point3::new(0.0, 0.0, 2.0),
        s,
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
    )
}

fn projected(covariance: Matrix3<f32>, center: Vector2<f32>) -> ProjectedGaussian {
    ProjectedGaussian {
        center,
        covariance,
        depth: 2.0,
        opacity: 0.8,
        color: Vector3::new(1.0, 0.0, 0.0),
        rotation: Vector4::new(0.0, 0.0, 0.0, 1.0),
        scale: Vector3::new(1.0, 1.0, 1.0),
        features: Vector3::zeros(),
    }
}

/// The control: a well-formed splat inverts, and the conic decays with distance.
///
/// Measured on the *screen-space* covariance, because that is where pixels are
/// the unit. The 3-D inverse of a scale-0.1 splat has diagonal `1/0.1² = 100`, so
/// its alpha falls to zero within a single pixel - too sharp to say anything
/// useful about decay, and not what the rasterizer integrates.
#[test]
fn a_well_formed_splat_inverts_and_decays() {
    // A screen-space covariance with a ~20 px radius: inverse diagonal 1/400.
    let cov = Matrix3::new(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 1.0);
    let pg = projected(cov, Vector2::new(160.0, 120.0));
    let inv = pg
        .inv_cov_2d()
        .expect("a non-degenerate covariance must invert");

    let centre = pg_alpha(&inv, 0.0, 0.0);
    let near = pg_alpha(&inv, 10.0, 0.0);
    let far = pg_alpha(&inv, 60.0, 0.0);
    assert!(
        centre > near && near > far,
        "alpha must decay away from the centre: centre={centre} near={near} far={far}"
    );
    assert!(
        centre > 0.79 && centre <= 0.8,
        "alpha at the centre should be the full opacity, got {centre}"
    );
    assert!(
        far < 0.01,
        "alpha 60 px out should be negligible for a 20 px radius, got {far}"
    );
}

/// The 3-D inverse still inverts for a non-degenerate splat.
#[test]
fn a_well_formed_splats_3d_covariance_inverts() {
    let g = splat_with_scale(Vector3::new(0.1, 0.1, 0.1));
    assert!(
        g.inverse_covariance().is_some(),
        "a non-degenerate covariance must invert"
    );
}

fn pg_alpha(inv: &Matrix3<f32>, dx: f32, dy: f32) -> f32 {
    let maha = inv[(0, 0)] * dx * dx + 2.0 * inv[(0, 1)] * dx * dy + inv[(1, 1)] * dy * dy;
    ((-0.5 * maha).exp() * 0.8).min(0.99)
}

/// A covariance flattened on one axis is singular and must report absence.
#[test]
fn a_flattened_splat_reports_that_it_cannot_invert() {
    let mut g = splat_with_scale(Vector3::new(0.1, 0.1, 0.1));
    // Reached by a deserialiser or a direct assignment, bypassing the
    // constructor's clamp.
    g.scale = Vector3::new(0.1, 0.0, 0.1);

    assert!(
        g.inverse_covariance().is_none(),
        "a singular covariance has no inverse, and must say so rather than \
         returning a zero matrix that reads as 'no extent'"
    );
}

#[test]
fn a_zero_scale_splat_reports_that_it_cannot_invert() {
    let mut g = splat_with_scale(Vector3::new(0.1, 0.1, 0.1));
    g.scale = Vector3::new(0.0, 0.0, 0.0);
    assert!(g.inverse_covariance().is_none());
}

/// The rasterizer's input. A zero covariance block has no footprint, so the
/// splat must report absence and be skipped.
#[test]
fn a_singular_screen_space_covariance_reports_absence() {
    let pg = projected(Matrix3::zeros(), Vector2::new(320.0, 240.0));
    assert!(
        pg.inv_cov_2d().is_none(),
        "a zero 2-D covariance must not hand back a zero inverse"
    );
}

/// The measured pre-fix behaviour, pinned so the reason for the change is
/// explicit rather than historical.
#[test]
fn a_zero_inverse_would_have_painted_the_whole_tile() {
    // This is what the old fallback produced, and why returning it was wrong.
    let zero = Matrix3::zeros();
    for (dx, dy) in [
        (0.0f32, 0.0f32),
        (10.0, 10.0),
        (100.0, 100.0),
        (1000.0, 1000.0),
    ] {
        assert_eq!(
            pg_alpha(&zero, dx, dy),
            0.8,
            "a zero inverse yields the same alpha at ({dx},{dy}) as at the centre \
             - which is exactly the whole-tile paint this change removes"
        );
    }
}
