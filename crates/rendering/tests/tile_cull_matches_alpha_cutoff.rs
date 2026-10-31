//! The rendered image must not depend on the tile size.
//!
//! ## The defect
//!
//! `GaussianRasterizer::compute_tile_bounds` sized its tile range from a fixed
//! **3-sigma** radius, while the inner loop drops any pixel whose alpha is below
//! a fixed cutoff:
//!
//! ```text
//! alpha_i = exp(-sigma^2 / 2) * opacity
//! ```
//!
//! The cutoff only bites once
//!
//! ```text
//! sigma_c = sqrt(2 * ln(opacity / ALPHA_CUTOFF))
//! ```
//!
//! For `opacity = 0.8` and a cutoff of `1e-4` that is **4.2396 sigma**. Every
//! pixel between 3 sigma and 4.2396 sigma is therefore *kept by the cutoff* but
//! *outside the tile range the splat was written into*. Whether such a pixel got
//! drawn depended entirely on which tile it fell in: drawn if that tile was in
//! the splat's range, black if it was not. The image was a function of the tile
//! size, which is supposed to be a memory-management knob.
//!
//! ## Measurement
//!
//! 128x128 viewport, `focal_length` 300, camera at the origin looking down +z.
//! One isotropic splat of world scale 0.05 at `(0.1, -0.05, 3.0)`, which
//! projects to centre `(74.0, 59.0)` px with screen sigma `5.0038` px - so a
//! 3-sigma radius of 15.008 px against a cutoff radius of 21.210 px.
//!
//! The reference is an exhaustive **untiled** per-pixel evaluation of the same
//! conic: the same inverse screen-space covariance, the same Gaussian, the same
//! cutoff, over every pixel. That is the image the tiling is supposed to
//! reproduce, so "differs from untiled" is a defect and not a definition.
//!
//! ```text
//!   tile   max |d alpha| vs untiled   alpha sum   pixels drawn but black
//!   4x4           2.474940e-3         125.63933        208
//!   8x8           5.901516e-4         125.72057         64
//!   16x16         1.184884e-4         125.73455          5
//!   32x32         9.984352e-5         125.73512          0
//!   64x64         9.984352e-5         125.73512          0
//! ```
//!
//! The lost pixels were the outer arc of the disc, where the splat's tail crosses
//! into a neighbouring tile. After the fix every tile size in that table lands on
//! the 32x32 row.
//!
//! ## What this test is *not*
//!
//! The `9.98e-5` residual is not a tiling artefact: it is the `alpha_i < 1e-4`
//! cutoff itself, which the untiled reference applies as well but at a pixel that
//! may land just under it. The assertion below is therefore stated as *no pixel
//! that the cutoff keeps may be missing*, not as bit-equality with the reference.
//! Asserting the exact figure would be asserting on the cutoff, which is a
//! separate decision.

use cv_rendering::gaussian_splatting::{
    Camera, Gaussian, GaussianCloud, GaussianRasterizer, SphericalHarmonics,
};
use nalgebra::{Point3, Vector3, Vector4};

const N: usize = 128;
const FOCAL: f32 = 300.0;
const CUTOFF: f32 = 1e-4;

fn camera() -> Camera {
    Camera::new(
        Point3::origin(),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        FOCAL,
        N as u32,
        N as u32,
    )
}

/// The splat from the measurement above: small enough that its 3-sigma radius
/// (15 px) is much smaller than the viewport, so the bound is interior and the
/// defect is not hidden by viewport clamping.
fn cloud() -> GaussianCloud {
    let mut cloud = GaussianCloud::new();
    cloud.push(Gaussian {
        position: Point3::new(0.1, -0.05, 3.0),
        scale: Vector3::new(0.05, 0.05, 0.05),
        rotation: Vector4::new(0.0, 0.0, 0.0, 1.0),
        opacity: 0.8,
        spherical_harmonics: SphericalHarmonics::from_dc(Vector3::new(1.0, 0.0, 0.0)),
        features: Vector3::zeros(),
    });
    cloud
}

/// Per-pixel sigma from the splat's own inverse screen-space covariance, so a
/// pixel's distance is measured in the units the cutoff is stated in.
fn sigma_table() -> (Vec<f32>, [f32; 4]) {
    let cam = camera();
    let pg = GaussianRasterizer::new(cam, 16, 16).project_gaussians(&cloud())[0].clone();
    let inv = pg.inv_cov_2d().expect("a non-degenerate splat inverts");
    let (cx, cy) = (pg.center.x, pg.center.y);
    let opacity = pg.opacity;
    let mut table = vec![0.0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let maha = inv[(0, 0)] * dx * dx + 2.0 * inv[(0, 1)] * dx * dy + inv[(1, 1)] * dy * dy;
            table[y * N + x] = maha.max(0.0).sqrt();
        }
    }
    let s = pg.covariance[(0, 0)].sqrt();
    (table, [cx, cy, s, opacity])
}

/// CONTROL: the untiled reference must itself be non-trivial, or "no pixel is
/// missing" would be satisfied by an empty image.
#[test]
fn the_untiled_reference_covers_a_substantial_disc() {
    let (sigma, [_, _, s_px, opacity]) = sigma_table();
    // The radius at which the cutoff starts biting, derived the same way the
    // fixed cull radius is, so the two cannot drift apart in the test.
    let sigma_c = (2.0 * (opacity / CUTOFF).ln()).sqrt();
    let kept = sigma.iter().filter(|s| **s <= 3.0).count();
    let annulus = sigma.iter().filter(|s| **s > 3.0 && **s <= sigma_c).count();
    println!(
        "untiled: {kept} px within 3 sigma, {annulus} px in the 3-to-{sigma_c:.4} sigma \
         annulus the defect lost (sigma = {s_px:.4} px, opacity = {opacity})"
    );
    assert!(
        kept > 500,
        "control: the reference disc must be substantial, got {kept} px within 3 sigma"
    );
    assert!(
        annulus > 200,
        "control: the 3-to-{sigma_c:.4} sigma annulus must be large enough for the defect \
         to bite, got {annulus} px - a splat this small would not exercise it"
    );
}

/// The defect: the tile grid decides which pixels of the splat's tail get drawn.
///
/// Asserted as "no pixel whose alpha exceeds the cutoff is missing", because
/// that is the property the two constants must agree on. The reference alpha is
/// recomputed here rather than taken from the rasterizer so the assertion cannot
/// pass by both sides changing together.
#[test]
fn the_rendered_alpha_does_not_depend_on_the_tile_size() {
    let (sigma, [_, _, _, opacity]) = sigma_table();
    let sigma_c = (2.0 * (opacity / CUTOFF).ln()).sqrt();
    let reference: Vec<f32> = sigma
        .iter()
        .map(|s| (-0.5 * s * s).exp() * opacity)
        .collect();

    for tile in [4u32, 8, 16, 32, 64] {
        let cam = camera();
        let out = GaussianRasterizer::new(cam, tile, tile).rasterize(&cloud());

        // The annulus between the old 3-sigma cull and the cutoff: exactly the
        // pixels the two constants disagreed about.
        let annulus = |i: usize| sigma[i] > 3.0 && sigma[i] <= sigma_c;

        let mut missing_in_annulus: Vec<(usize, usize, f32, f32)> = Vec::new();
        let mut extra = 0usize;
        for i in 0..N * N {
            let kept = reference[i] >= CUTOFF;
            let drawn = out.alpha[i] >= CUTOFF;
            if kept && !drawn && annulus(i) {
                missing_in_annulus.push((i % N, i / N, sigma[i], reference[i]));
            } else if !kept && drawn {
                extra += 1;
            }
        }

        let max_err = reference
            .iter()
            .zip(out.alpha.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let sum_ref: f32 = reference.iter().sum();
        let sum_out: f32 = out.alpha.iter().sum();

        println!(
            "tile {tile:3}: max |d alpha| {max_err:.6e}, sum {sum_ref:.5} -> {sum_out:.5}, \
             annulus pixels lost {}, spurious {extra}",
            missing_in_annulus.len()
        );

        assert!(
            missing_in_annulus.is_empty(),
            "tile {tile}x{tile} dropped {} pixels of the splat's tail that the alpha cutoff \
             keeps. The cull radius and the cutoff must be one threshold: at opacity \
             {opacity} the cutoff only bites past {sigma_c:.4} sigma, so a 3-sigma cull \
             loses everything between the two depending on which tile a pixel landed in. \
             First few: {:?}",
            missing_in_annulus.len(),
            &missing_in_annulus
                .iter()
                .take(6)
                .map(|(x, y, s, a)| (*x, *y, *s, *a))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            extra, 0,
            "tile {tile}x{tile} drew {extra} pixels the cutoff rejects"
        );
        assert!(
            (sum_ref - sum_out).abs() < 1e-3 * sum_ref,
            "tile {tile}x{tile}: alpha mass {sum_out} against the untiled {sum_ref}"
        );
    }
}

/// CONTROL: a splat entirely inside one tile still renders, and its centre is
/// at full opacity. Without this, the test above could be satisfied by a cull
/// that drops everything.
#[test]
fn a_well_formed_splat_still_renders_at_every_tile_size() {
    let cam = camera();
    let mut baseline: Option<Vec<f32>> = None;
    for tile in [8u32, 16, 32] {
        let out = GaussianRasterizer::new(cam.clone(), tile, tile).rasterize(&cloud());
        let centre = 59 * N + 74; // (74, 59)
        assert!(
            out.alpha[centre] > 0.7,
            "tile {tile}: the splat centre must be near full opacity, got {}",
            out.alpha[centre]
        );
        assert!(
            out.alpha[centre] <= 0.8 + 1e-6,
            "tile {tile}: alpha cannot exceed the splat's own opacity 0.8, got {}",
            out.alpha[centre]
        );
        match &baseline {
            None => baseline = Some(out.alpha.clone()),
            Some(b) => {
                let m = b
                    .iter()
                    .zip(out.alpha.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    m < 1e-6,
                    "tile {tile} differs from tile 8 by {m:e}; a splat fully inside a tile \
                     must be bit-identical whatever the grid"
                );
            }
        }
    }
}

/// CONTROL: a tile grid coarser than the viewport must not change the image
/// either - `num_tiles` already rounds up, and a bound that indexed past it
/// would panic here rather than silently diverge.
#[test]
fn a_tile_larger_than_the_viewport_still_renders() {
    let cam = camera();
    let fine = GaussianRasterizer::new(cam.clone(), 16, 16).rasterize(&cloud());
    let coarse = GaussianRasterizer::new(cam, 512, 512).rasterize(&cloud());
    let m = fine
        .alpha
        .iter()
        .zip(coarse.alpha.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("512x512 tile (one tile for the whole viewport): max |d alpha| {m:e}");
    assert!(
        m < 1e-6,
        "a single-tile grid produced a different image: {m:e}"
    );
}
