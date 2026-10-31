//! Brent's method for root finding.

/// Find a root of `f` in the bracket `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs. Iteration stops when the
/// root is bracketed to within `tol + 4·ε·|x|` in **x** (SciPy's `xtol` with its
/// default `rtol`), or when `f` is exactly zero. This is a line-for-line port
/// of SciPy's `brentq.c` (Charles Harris's implementation, BSD-licensed).
///
/// `tol` used to gate `|f|` as well. That is a statement in the units of `f`,
/// so a shallow function stopped anywhere: `brentq(|x| 1e-13 * (x - 1.5), 1, 2,
/// 1e-12, 100)` returned the bracket end `2.0`. The old loop also froze the
/// contrapoint's value (`let fc = fa`, never updated), so inverse quadratic
/// interpolation ran on a stale point: `x^3 - 2x - 5` on `[2, 3]` took 81
/// evaluations where SciPy takes 8.
///
/// # Errors
/// * `f(a)` and `f(b)` have the same sign (no bracket), or
/// * `max_iter` iterations are exhausted before the bracket is narrower than
///   the tolerance. Running out of iterations is not a root find — returning
///   the last iterate as `Ok` would report `1.4141414141` as a root of
///   `x^2 - 2` at `tol = 1e-12`, where `|f| = 2.0e-4`.
pub fn brentq(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let rtol = 4.0 * f64::EPSILON;
    let (mut xpre, mut xcur) = (a, b);
    let (mut xblk, mut fblk) = (0.0, 0.0);
    let (mut spre, mut scur) = (0.0f64, 0.0f64);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);

    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err("f(a) and f(b) must have opposite signs".into());
    }

    let mut width = (b - a).abs();
    for _ in 0..max_iter {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (tol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        width = 2.0 * sbis.abs();
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // Secant (interpolation).
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // Inverse quadratic (extrapolation).
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // Good short step.
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            // Bisect.
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }

    Err(format!(
        "Brent's method did not converge in {max_iter} iterations: bracket width {width:.3e} > \
         tol {tol:e} (|f| = {:.3e})",
        fcur.abs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brentq_sqrt2() {
        // Root of x^2 - 2 = 0 in [1, 2] => sqrt(2)
        let root = brentq(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!(
            (root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root ≈ √2, got {}",
            root
        );
    }

    #[test]
    fn brentq_cubic() {
        // Root of x^3 - x - 2 = 0 near x ≈ 1.5214
        let root = brentq(|x| x.powi(3) - x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root.powi(3) - root - 2.0).abs() < 1e-10);
    }

    /// An exhausted iteration budget is not a root. Measured on the old code:
    /// `brentq(x^2 - 2, 1, 2, 1e-12, 5)` returned `Ok(1.4141414141)` where the
    /// residual is `2.0e-4` — eight orders of magnitude above the tolerance the
    /// caller asked for, and indistinguishable from a converged result.
    #[test]
    fn brentq_iteration_cap_is_not_reported_as_a_root() {
        let f = |x: f64| x * x - 2.0;
        for max_iter in [0usize, 1, 2, 3, 5] {
            match brentq(f, 1.0, 2.0, 1e-12, max_iter) {
                Ok(x) => panic!(
                    "max_iter={max_iter}: returned Ok({x}) with |f(x)| = {:.3e}",
                    f(x).abs()
                ),
                Err(e) => assert!(
                    e.contains("did not converge"),
                    "max_iter={max_iter}: unexpected error: {e}"
                ),
            }
        }
    }

    /// Control: with an adequate budget the same problem must converge to sqrt(2)
    /// — so the test above cannot pass by rejecting everything.
    #[test]
    fn brentq_converges_with_an_adequate_budget() {
        let root = brentq(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10, "{root}");
    }

    /// Control: a *steep* function converges on the bracket-width criterion, where
    /// `|f|` cannot be pushed below `tol` in f-units. That legitimate outcome must
    /// still be `Ok`.
    #[test]
    fn brentq_accepts_a_converged_bracket_on_a_steep_function() {
        let f = |x: f64| 1e9 * (x - 1.5);
        let root = brentq(f, 1.0, 2.0, 1e-12, 100).expect("bracket narrows below tol");
        assert!((root - 1.5).abs() <= 1e-12, "{root}");
    }
}
