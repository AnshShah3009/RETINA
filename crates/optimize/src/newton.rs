//! Newton's method for root finding.

/// Find a root of `f` using Newton's method.
///
/// # Arguments
/// * `f` - Function whose root is sought.
/// * `fprime` - Derivative of `f`.
/// * `x0` - Initial guess.
/// * `tol` - Convergence tolerance (on `|f(x)|`).
/// * `max_iter` - Maximum iterations.
///
/// # Errors
/// `Err` when the derivative vanishes, and — importantly — when `max_iter`
/// iterations are used up without `|f(x)| < tol`. An iteration cap is not
/// convergence: returning the iterate as `Ok` would hand the caller a point
/// that is not a root (Newton oscillates between 0 and 1 forever on
/// `x^3 - 2x + 2` from `x0 = 0`, for example) with no way to tell.
pub fn newton(
    f: impl Fn(f64) -> f64,
    fprime: impl Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return Ok(x);
        }
        let fp = fprime(x);
        if fp.abs() < 1e-30 {
            return Err("Derivative is zero; Newton's method cannot continue".into());
        }
        let x_new = x - fx / fp;
        if (x_new - x).abs() < tol {
            // The step stopped moving. That is only a root if the residual
            // agrees, so verify rather than assume.
            return if f(x_new).abs() < tol {
                Ok(x_new)
            } else {
                Err(format!(
                    "Newton's method stalled at x = {x_new}: step size {} < tol {tol} but |f(x)| = {}",
                    (x_new - x).abs(),
                    f(x_new).abs()
                ))
            };
        }
        x = x_new;
    }

    let residual = f(x).abs();
    if residual < tol {
        Ok(x)
    } else {
        Err(format!(
            "Newton's method did not converge in {max_iter} iterations: |f({x})| = {residual:e} > tol {tol:e}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_cos_x_minus_x() {
        // Root of cos(x) - x = 0 (Dottie number ≈ 0.7390851332)
        let root = newton(|x| x.cos() - x, |x| -x.sin() - 1.0, 0.5, 1e-12, 100).unwrap();
        assert!(
            (root.cos() - root).abs() < 1e-10,
            "cos(root) should equal root, got {}",
            root
        );
        assert!((root - 0.7390851332).abs() < 1e-8);
    }

    #[test]
    fn newton_square_root() {
        // Root of x^2 - 5 = 0 => sqrt(5)
        let root = newton(|x| x * x - 5.0, |x| 2.0 * x, 2.0, 1e-12, 100).unwrap();
        assert!((root - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    /// A 2-cycle is not a root. `x^3 - 2x + 2` maps 0 -> 1 -> 0 forever, so
    /// Newton never converges however many iterations it is allowed; the old
    /// code returned `Ok` at the iteration cap, and the callers could not tell
    /// the returned `x` was not a root (`|f(x)| = 2.0`, not `tol = 1e-12`).
    #[test]
    fn newton_two_cycle_reports_failure_instead_of_a_non_root() {
        let f = |x: f64| x.powi(3) - 2.0 * x + 2.0;
        let fp = |x: f64| 3.0 * x * x - 2.0;

        for max_iter in [1usize, 2, 5, 100] {
            match newton(f, fp, 0.0, 1e-12, max_iter) {
                Ok(x) => panic!(
                    "max_iter={max_iter}: two-cycle returned Ok({x}) with |f(x)| = {}",
                    f(x).abs()
                ),
                Err(e) => assert!(
                    e.contains("did not converge"),
                    "max_iter={max_iter}: unexpected error: {e}"
                ),
            }
        }
    }

    /// Control: a genuine root must still be returned, and the real root of that
    /// cubic (≈ -1.769292) must be reachable from a starting point that is not
    /// on the cycle.
    #[test]
    fn newton_still_finds_the_real_cubic_root() {
        let f = |x: f64| x.powi(3) - 2.0 * x + 2.0;
        let fp = |x: f64| 3.0 * x * x - 2.0;
        let root = newton(f, fp, -2.0, 1e-12, 100).expect("must converge from -2");
        assert!(
            (root + 1.769_292_354_238_631_4).abs() < 1e-9,
            "root = {root}"
        );
        assert!(f(root).abs() < 1e-12);
    }

    /// Control: a stalled step is not a root either. `f = 1` has no root, and the
    /// huge derivative makes every Newton step `-1e-13`: below `tol`, while `|f|`
    /// stays at 1. The old code returned `Ok(-1e-13)` from this path.
    #[test]
    fn newton_rejects_a_non_root_when_the_step_does_not_move() {
        let f = |_x: f64| 1.0;
        let fp = |_x: f64| 1e13;
        match newton(f, fp, 0.0, 1e-12, 100) {
            Ok(x) => panic!("stalled step returned Ok({x}) with |f(x)| = {}", f(x).abs()),
            Err(e) => assert!(
                e.contains("stalled") || e.contains("did not converge"),
                "{e}"
            ),
        }
    }
}
