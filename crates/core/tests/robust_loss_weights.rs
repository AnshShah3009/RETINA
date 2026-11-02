//! `RobustLoss::weight` is the IRLS weight `rho'(r) / (r rho''(0))`, i.e.
//! `rho'(r) / r` scaled so the weight of a zero residual is 1, and it is
//! continuous in `r`.
//!
//! Five of the six losses already followed that convention (Welsch's
//! `exp(-r^2/mu)` and TLS's 1 are `rho'/(2r)` for losses with
//! `rho ~ r^2` at 0; Huber's, Cauchy's and Tukey's are `rho'/r` for losses
//! with `rho ~ r^2/2`), and `weight` returns 1 for `|r| < 1e-6`. Geman-McClure
//! (`rho = mu r^2 / (mu + r^2)`, `rho ~ r^2` at 0) returned the unscaled
//! `rho'/r = 2 mu^2 / (mu + r^2)^2`, so its weight jumped from 1 below 1e-6 to
//! 2 just above it. Its weight is now `(mu / (mu + r^2))^2`, the GNC weight
//! of Yang et al. (2020) for this loss.

use cv_core::RobustLoss;

fn all() -> Vec<RobustLoss> {
    vec![
        RobustLoss::GemanMcClure { mu: 0.7 },
        RobustLoss::Welsch { mu: 0.7 },
        RobustLoss::Huber { mu: 0.7 },
        RobustLoss::TruncatedLeastSquares { c: 0.7 },
        RobustLoss::Cauchy { mu: 0.7 },
        RobustLoss::Tukey { mu: 0.7 },
    ]
}

#[test]
fn weights_are_continuous_at_the_small_residual_cutoff() {
    for loss in all() {
        let below = loss.weight(0.999e-6);
        let above = loss.weight(1.001e-6);
        assert!(
            (below - 1.0).abs() < 1e-6 && (above - 1.0).abs() < 1e-6,
            "{loss:?}: weight {below} just below 1e-6, {above} just above"
        );
    }
}

#[test]
fn weights_are_the_normalised_derivative_of_the_loss() {
    // rho''(0): 2 for the losses that are r^2 near 0, 1 for those that are r^2/2.
    let curvature_at_zero = |loss: &RobustLoss| match loss {
        RobustLoss::GemanMcClure { .. }
        | RobustLoss::Welsch { .. }
        | RobustLoss::TruncatedLeastSquares { .. } => 2.0f64,
        _ => 1.0,
    };
    for loss in all() {
        // Central differences in f32: below r ~ 0.1, `evaluate` itself loses
        // too many digits (Welsch's 1 - exp(-r^2/mu)) to differentiate.
        for r in [0.1f32, 0.3, 0.5, 0.65, 0.9, 1.5, 3.0] {
            let h = 1e-2f32 * r;
            let drho =
                (loss.evaluate(r + h) as f64 - loss.evaluate(r - h) as f64) / (2.0 * h as f64);
            let want = drho / (r as f64 * curvature_at_zero(&loss));
            let got = loss.weight(r) as f64;
            assert!(
                // 1%: the difference quotient's own error near Tukey's knee is
                // 0.4%; Geman-McClure was off by a factor of 2.
                (got - want).abs() < 1e-2 * want.abs() + 1e-6,
                "{loss:?} r={r}: weight {got}, rho'(r)/(r rho''(0)) = {want}"
            );
        }
    }
}
