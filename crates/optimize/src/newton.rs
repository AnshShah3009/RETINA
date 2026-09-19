//! Newton's method for root finding.

/// Find a root of `f` using Newton's method.
///
/// # Arguments
/// * `f` - Function whose root is sought.
/// * `fprime` - Derivative of `f`.
/// * `x0` - Initial guess.
/// * `tol` - Convergence tolerance.
/// * `max_iter` - Maximum iterations.
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
            return Ok(x_new);
        }
        x = x_new;
    }
    Ok(x)
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
}
