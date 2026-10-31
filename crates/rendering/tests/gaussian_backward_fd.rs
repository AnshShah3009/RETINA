//! `DifferentiableRasterizer::backward` must return the gradient of
//! `compute_loss(render(cloud), target)`.
//!
//! It used to return all-zero gradients, so `GaussianTrainer::train` never
//! moved a single parameter. Every analytic gradient here is checked against
//! a central finite difference of the full forward pass (projection, tile
//! culling, alpha compositing over a background, and the masked MSE loss).

use cv_rendering::gaussian_splatting::{Camera, DifferentiableRasterizer, Gaussian, GaussianCloud};
use nalgebra::{Point3, Vector3, Vector4};

const W: u32 = 40;
const H: u32 = 36;

/// A camera that is neither at the origin nor axis-aligned, so the view
/// rotation and translation both enter the chain rule.
fn camera() -> Camera {
    let q = Vector4::new(0.03, -0.05, 0.02, 1.0).normalize();
    Camera::new(Point3::new(0.1, -0.05, -0.2), q, 70.0, W, H)
}

/// Three overlapping, anisotropic, rotated splats of different colours, so the
/// compositing order and the transmittance terms matter.
fn scene() -> GaussianCloud {
    let mut cloud = GaussianCloud::new();
    cloud.push(
        Gaussian::new(
            Point3::new(0.0, 0.0, 4.0),
            Vector3::new(0.35, 0.18, 0.2),
            Vector4::new(0.1, 0.2, 0.3, 0.9),
            Vector3::new(0.9, 0.2, 0.1),
        )
        .with_opacity(0.6),
    );
    cloud.push(
        Gaussian::new(
            Point3::new(0.25, 0.1, 4.6),
            Vector3::new(0.2, 0.3, 0.15),
            Vector4::new(-0.3, 0.1, 0.4, 0.8),
            Vector3::new(0.1, 0.8, 0.3),
        )
        .with_opacity(0.7),
    );
    cloud.push(
        Gaussian::new(
            Point3::new(-0.2, 0.15, 5.2),
            Vector3::new(0.3, 0.25, 0.3),
            Vector4::new(0.0, -0.4, 0.1, 0.9),
            Vector3::new(0.2, 0.3, 0.9),
        )
        .with_opacity(0.5),
    );
    cloud
}

fn target() -> Vec<Vector3<f32>> {
    (0..W * H)
        .map(|i| {
            let x = (i % W) as f32;
            let y = (i / W) as f32;
            Vector3::new(
                0.5 + 0.4 * (0.21 * x).sin(),
                0.5 + 0.4 * (0.17 * y + 1.0).cos(),
                0.5 + 0.3 * (0.13 * (x + y)).sin(),
            )
        })
        .collect()
}

fn loss(r: &DifferentiableRasterizer, cloud: &GaussianCloud, t: &[Vector3<f32>]) -> f32 {
    r.compute_loss(&r.render(cloud), t)
}

/// Central difference of the loss with respect to one scalar parameter.
fn numeric<F>(
    r: &DifferentiableRasterizer,
    cloud: &GaussianCloud,
    t: &[Vector3<f32>],
    idx: usize,
    eps: f32,
    perturb: F,
) -> f32
where
    F: Fn(&mut Gaussian, f32),
{
    let mut plus = GaussianCloud::from_gaussians(cloud.gaussians.clone());
    perturb(&mut plus.gaussians[idx], eps);
    let mut minus = GaussianCloud::from_gaussians(cloud.gaussians.clone());
    perturb(&mut minus.gaussians[idx], -eps);
    (loss(r, &plus, t) - loss(r, &minus, t)) / (2.0 * eps)
}

fn check(name: &str, analytic: f32, numeric: f32, scale: f32) {
    let err = (analytic - numeric).abs();
    let tol = 0.03 * analytic.abs().max(numeric.abs()) + 2e-3 * scale;
    println!("{name:>14}: analytic {analytic:+.6e} numeric {numeric:+.6e} err {err:.2e}");
    assert!(
        err <= tol,
        "{name}: analytic {analytic} vs numeric {numeric} (err {err}, tol {tol})"
    );
}

fn rasterizer() -> DifferentiableRasterizer {
    DifferentiableRasterizer::new(camera(), Vector3::new(0.3, 0.25, 0.2))
}

/// Largest finite-difference magnitude over all parameters of all splats, used
/// as the absolute-tolerance scale so near-zero components are not judged
/// relatively.
fn scale_of(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[test]
fn colour_and_opacity_gradients_match_finite_differences() {
    let r = rasterizer();
    let cloud = scene();
    let t = target();
    let grads = r.backward(&cloud, &r.render(&cloud), &t);
    assert_eq!(grads.len(), cloud.num_gaussians());

    let mut pairs = Vec::new();
    for i in 0..cloud.num_gaussians() {
        for c in 0..3 {
            let n = numeric(&r, &cloud, &t, i, 1e-2, |g, e| {
                g.spherical_harmonics.coeffs[c] += e
            });
            pairs.push((format!("g{i} colour[{c}]"), grads[i].color_grad[c], n));
        }
        let n = numeric(&r, &cloud, &t, i, 1e-3, |g, e| g.opacity += e);
        pairs.push((format!("g{i} opacity"), grads[i].opacity_grad, n));
    }
    let s = scale_of(&pairs.iter().map(|p| p.2).collect::<Vec<_>>());
    assert!(s > 1e-3, "scene gives no gradient signal: {s}");
    for (name, a, n) in pairs {
        check(&name, a, n, s);
    }
}

#[test]
fn position_gradients_match_finite_differences() {
    let r = rasterizer();
    let cloud = scene();
    let t = target();
    let grads = r.backward(&cloud, &r.render(&cloud), &t);

    let mut pairs = Vec::new();
    for i in 0..cloud.num_gaussians() {
        for k in 0..3 {
            let n = numeric(&r, &cloud, &t, i, 1e-2, |g, e| g.position[k] += e);
            pairs.push((format!("g{i} pos[{k}]"), grads[i].position_grad[k], n));
        }
    }
    let s = scale_of(&pairs.iter().map(|p| p.2).collect::<Vec<_>>());
    assert!(s > 1e-3, "scene gives no gradient signal: {s}");
    for (name, a, n) in pairs {
        check(&name, a, n, s);
    }
}

#[test]
fn scale_and_rotation_gradients_match_finite_differences() {
    let r = rasterizer();
    let cloud = scene();
    let t = target();
    let grads = r.backward(&cloud, &r.render(&cloud), &t);

    let mut pairs = Vec::new();
    for i in 0..cloud.num_gaussians() {
        for k in 0..3 {
            let n = numeric(&r, &cloud, &t, i, 5e-3, |g, e| g.scale[k] += e);
            pairs.push((format!("g{i} scale[{k}]"), grads[i].scale_grad[k], n));
        }
        for k in 0..4 {
            let n = numeric(&r, &cloud, &t, i, 1e-2, |g, e| g.rotation[k] += e);
            pairs.push((format!("g{i} rot[{k}]"), grads[i].rotation_grad[k], n));
        }
    }
    let s = scale_of(&pairs.iter().map(|p| p.2).collect::<Vec<_>>());
    assert!(s > 1e-3, "scene gives no gradient signal: {s}");
    for (name, a, n) in pairs {
        check(&name, a, n, s);
    }
}

#[test]
fn background_is_composited_behind_the_splats() {
    // The background shows through the final transmittance:
    //     C = sum_i c_i a_i T_i + T_N * bg,   T_N = 1 - accumulated alpha.
    let bg = Vector3::new(0.3, 0.25, 0.2);
    let r = rasterizer();
    let with_bg = r.render(&scene());
    let without = r.rasterizer.rasterize(&scene());
    let mut max_err = 0.0f32;
    let mut uncovered = 0;
    for i in 0..with_bg.color.len() {
        let expected = without.color[i] + bg * (1.0 - without.alpha[i]);
        max_err = max_err.max((with_bg.color[i] - expected).norm());
        if without.alpha[i] < 0.5 {
            uncovered += 1;
        }
    }
    assert!(uncovered > 0, "scene must leave some background visible");
    assert!(max_err < 1e-5, "background compositing off by {max_err}");
    // An empty cloud renders exactly the background.
    let empty = r.render(&GaussianCloud::new());
    assert!(empty.color.iter().all(|c| (c - bg).norm() < 1e-6));
}
