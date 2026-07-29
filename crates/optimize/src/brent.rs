//! Brent's method for root finding.

/// Find a root of `f` in the bracket `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs.
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

    Ok(b)
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
}
