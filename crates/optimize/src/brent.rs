//! Brent's method for root finding.

/// Find a root of `f` in the bracket `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs. Iteration stops when the
/// bracket is narrower than `tol` or `|f| < tol` at an endpoint.
///
/// # Errors
/// * `f(a)` and `f(b)` have the same sign (no bracket), or
/// * `max_iter` iterations are exhausted while the bracket is still wider than
///   `tol` and `|f|` is still above it. Running out of iterations is not a root
///   find — returning the last iterate as `Ok` would report `1.4141414141` as a
///   root of `x^2 - 2` at `tol = 1e-12`, where `|f| = 2.0e-4`.
pub fn brentq(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err("f(a) and f(b) must have opposite signs".into());
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let fc = fa;
    let mut mflag = true;
    let mut d = 0.0; // will be set before used

    for _ in 0..max_iter {
        if fb.abs() < tol {
            return Ok(b);
        }
        if fa.abs() < tol {
            return Ok(a);
        }
        if (b - a).abs() < tol {
            return Ok(b);
        }

        let s = if (fa - fc).abs() > 1e-30 && (fb - fc).abs() > 1e-30 {
            // Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant method
            b - fb * (b - a) / (fb - fa)
        };

        let cond1 = {
            let lo = (3.0 * a + b) / 4.0;
            let (min_ab, max_ab) = if lo < b { (lo, b) } else { (b, lo) };
            s < min_ab || s > max_ab
        };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            // Bisection
            let s_new = (a + b) / 2.0;
            mflag = true;
            d = c; // safe: d used only when mflag is false on next iteration, and we set c below
            c = b;
            let fs = f(s_new);
            if fa * fs < 0.0 {
                b = s_new;
                fb = fs;
            } else {
                a = s_new;
                fa = fs;
            }
        } else {
            mflag = false;
            d = c;
            c = b;
            let fs = f(s);
            if fa * fs < 0.0 {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    // Iterations exhausted. Only accept `b` if it is a root by one of the two
    // criteria the loop itself uses; otherwise say so instead of returning a
    // number that merely looks like a root.
    let fb = f(b);
    if fb.abs() < tol || (b - a).abs() < tol {
        Ok(b)
    } else {
        Err(format!(
            "Brent's method did not converge in {max_iter} iterations: bracket width {:.3e}, \
             |f(b)| = {:.3e} > tol {tol:e}",
            (b - a).abs(),
            fb.abs()
        ))
    }
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
