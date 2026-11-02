//! A `Pose2` heading is an angle: `π - 0.01` and `-π + 0.01` are 0.02 rad
//! apart, not 6.26.
//!
//! `BetweenFactor` already wraps its heading residual to `[-π, π]`, and so does
//! GTSAM's `PriorFactor<Pose2>` (its error is `Pose2::Logmap(prior⁻¹ · x)`).
//! `PriorFactor` used `Variable::local`, which for `Pose2` was a plain vector
//! difference, so a prior that the value satisfies to 0.02 rad reported a
//! heading error of `-6.2632` - with `σ = 0.01` a whitened cost of 392 289
//! instead of 4. Values leave `[-π, π]` routinely, because `Pose2::retract`
//! adds the increment without wrapping, so a robot that turns past π and then
//! receives a compass prior hits this.
//!
//! `Variable::local` is also iSAM2's relinearisation test, which therefore saw
//! a 2π "change" in a variable that had moved 0.02 rad.

use cv_optimize::factor_graph::{Factor, FactorGraph, Key, NoiseModel, Values, Variable};
use cv_optimize::factors::PriorFactor;
use nalgebra::Vector3;
use std::f64::consts::PI;

const SIGMA: f64 = 0.01;

fn pose2(x: f64, y: f64, theta: f64) -> Variable {
    Variable::Pose2(Vector3::new(x, y, theta))
}

#[test]
fn pose2_prior_heading_error_is_wrapped() {
    let key = Key::symbol('x', 0);
    let prior = PriorFactor::new(
        key,
        pose2(1.0, 2.0, PI - 0.01),
        NoiseModel::Isotropic(SIGMA, 3),
    );
    let mut values = Values::new();
    values.insert(key, pose2(1.0, 2.0, -PI + 0.01));

    let e = prior.error(&values);
    assert!(e[0].abs() < 1e-12 && e[1].abs() < 1e-12, "{e:?}");
    assert!(
        (e[2] - 0.02).abs() < 1e-12,
        "heading error {} for headings 0.02 rad apart",
        e[2]
    );

    let mut graph = FactorGraph::new();
    graph.add(prior);
    let cost = graph.total_error(&values);
    let want = (0.02 / SIGMA).powi(2);
    assert!(
        (cost - want).abs() < 1e-6,
        "total_error {cost}, expected {want}"
    );
}

#[test]
fn pose2_local_wraps_the_heading() {
    let a = pose2(0.0, 0.0, 3.2);
    let b = pose2(0.5, -0.5, -3.0);
    let d = a.local(&b);
    assert!((d[0] - 0.5).abs() < 1e-12 && (d[1] + 0.5).abs() < 1e-12);
    // -3.0 - 3.2 = -6.2  ->  wrapped: 2π - 6.2 = 0.0831853...
    assert!(
        (d[2] - (2.0 * PI - 6.2)).abs() < 1e-12,
        "heading difference {}",
        d[2]
    );
}
