//! Interpolation functions.
//!
//! Provides 1D linear, cubic spline, and Akima interpolation,
//! plus 2D bilinear interpolation on regular grids.

/// 1D linear interpolation.
///
/// `x` and `y` are the known data points (must be sorted by `x`, same length, at least 2 points).
/// `x_new` are the query points. Values outside the range are clamped to the boundary values.
pub fn interp1d_linear(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, String> {
    validate_inputs(x, y)?;

    Ok(x_new.iter().map(|&xi| linear_at(x, y, xi)).collect())
}

fn validate_inputs(x: &[f64], y: &[f64]) -> Result<(), String> {
    if x.len() != y.len() {
        return Err(format!(
            "x and y must have the same length, got {} and {}",
            x.len(),
            y.len()
        ));
    }
    if x.len() < 2 {
        return Err("Need at least 2 data points".to_string());
    }
    for i in 1..x.len() {
        if x[i] <= x[i - 1] {
            return Err("x values must be strictly increasing".to_string());
        }
    }
    Ok(())
}

fn find_interval(x: &[f64], xi: f64) -> usize {
    match x.binary_search_by(|v| v.partial_cmp(&xi).unwrap()) {
        Ok(i) => i.min(x.len() - 2),
        Err(i) => {
            if i == 0 {
                0
            } else {
                (i - 1).min(x.len() - 2)
            }
        }
    }
}

fn linear_at(x: &[f64], y: &[f64], xi: f64) -> f64 {
    if xi <= x[0] {
        return y[0];
    }
    if xi >= x[x.len() - 1] {
        return y[y.len() - 1];
    }
    let i = find_interval(x, xi);
    let t = (xi - x[i]) / (x[i + 1] - x[i]);
    y[i] + t * (y[i + 1] - y[i])
}

/// 1D natural cubic spline interpolation.
///
/// Uses the Thomas algorithm to solve the tridiagonal system for spline coefficients.
/// Natural boundary conditions: second derivative = 0 at endpoints.
pub fn interp1d_cubic(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, String> {
    validate_inputs(x, y)?;

    let n = x.len();
    let nm1 = n - 1;

    // Compute intervals and slopes
    let h: Vec<f64> = (0..nm1).map(|i| x[i + 1] - x[i]).collect();

    // Build tridiagonal system for second derivatives (natural spline: M[0]=M[n-1]=0)
    // Interior equations: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*(d[i]-d[i-1])
    let n_interior = if n >= 3 { n - 2 } else { 0 };

    let mut m = vec![0.0; n]; // second derivatives

    if n_interior > 0 {
        let d: Vec<f64> = (0..nm1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        // Thomas algorithm
        let mut a = vec![0.0; n_interior]; // sub-diagonal
        let mut b = vec![0.0; n_interior]; // main diagonal
        let mut c = vec![0.0; n_interior]; // super-diagonal
        let mut rhs = vec![0.0; n_interior];

        for i in 0..n_interior {
            let gi = i + 1; // global index
            if i > 0 {
                a[i] = h[gi - 1];
            }
            b[i] = 2.0 * (h[gi - 1] + h[gi]);
            if i < n_interior - 1 {
                c[i] = h[gi];
            }
            rhs[i] = 6.0 * (d[gi] - d[gi - 1]);
        }

        // Forward elimination
        for i in 1..n_interior {
            let factor = a[i] / b[i - 1];
            b[i] -= factor * c[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }

        // Back substitution
        let last = n_interior - 1;
        m[last + 1] = rhs[last] / b[last];
        for i in (0..last).rev() {
            m[i + 1] = (rhs[i] - c[i] * m[i + 2]) / b[i];
        }
    }

    // Evaluate spline at query points
    let result = x_new
        .iter()
        .map(|&xi| {
            if xi <= x[0] {
                return y[0];
            }
            if xi >= x[nm1] {
                return y[nm1];
            }
            let i = find_interval(x, xi);
            let dx = xi - x[i];
            let hi = h[i];
            // Cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
            let ai = y[i];
            let ci = m[i] / 2.0;
            let di = (m[i + 1] - m[i]) / (6.0 * hi);
            let bi = (y[i + 1] - y[i]) / hi - hi * (2.0 * m[i] + m[i + 1]) / 6.0;
            ai + dx * (bi + dx * (ci + dx * di))
        })
        .collect();

    Ok(result)
}

/// 2D bilinear interpolation on a regular grid.
///
/// `x` and `y` define the grid axes (sorted, at least 2 elements each).
/// `z` contains values in row-major order: `z[iy * x.len() + ix]`.
/// `xi` and `yi` are query points (must have the same length).
pub fn interp2d_linear(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xi: &[f64],
    yi: &[f64],
) -> Result<Vec<f64>, String> {
    if x.len() < 2 || y.len() < 2 {
        return Err("Grid axes must have at least 2 points".to_string());
    }
    if z.len() != y.len() * x.len() {
        return Err(format!(
            "z length {} does not match y.len() * x.len() = {}",
            z.len(),
            y.len() * x.len()
        ));
    }
    if xi.len() != yi.len() {
        return Err("xi and yi must have the same length".to_string());
    }

    let nx = x.len();

    let result = xi
        .iter()
        .zip(yi.iter())
        .map(|(&xq, &yq)| {
            let xq = xq.clamp(x[0], x[x.len() - 1]);
            let yq = yq.clamp(y[0], y[y.len() - 1]);

            let ix = find_interval(x, xq);
            let iy = find_interval(y, yq);

            let tx = if (x[ix + 1] - x[ix]).abs() < 1e-15 {
                0.0
            } else {
                (xq - x[ix]) / (x[ix + 1] - x[ix])
            };
            let ty = if (y[iy + 1] - y[iy]).abs() < 1e-15 {
                0.0
            } else {
                (yq - y[iy]) / (y[iy + 1] - y[iy])
            };

            let z00 = z[iy * nx + ix];
            let z10 = z[iy * nx + ix + 1];
            let z01 = z[(iy + 1) * nx + ix];
            let z11 = z[(iy + 1) * nx + ix + 1];

            let v0 = z00 + tx * (z10 - z00);
            let v1 = z01 + tx * (z11 - z01);
            v0 + ty * (v1 - v0)
        })
        .collect();

    Ok(result)
}

/// Akima interpolation (robust to outliers).
///
/// Uses Akima's method for computing slopes, which avoids overshooting
/// near outliers by using a weighted average of adjacent slopes.
pub fn interp1d_akima(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, String> {
    validate_inputs(x, y)?;
    let n = x.len();
    if n < 3 {
        // Fall back to linear for very short data
        return interp1d_linear(x, y, x_new);
    }

    let nm1 = n - 1;

    // Compute slopes between consecutive points
    let mut m = Vec::with_capacity(nm1 + 4);
    let slopes: Vec<f64> = (0..nm1)
        .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        .collect();

    // Extend slopes at boundaries (Akima's method needs m[-2], m[-1] and m[n-1], m[n])
    // Use parabolic extrapolation at boundaries
    if slopes.len() >= 2 {
        m.push(3.0 * slopes[0] - 2.0 * slopes[1]);
        m.push(2.0 * slopes[0] - slopes[1]);
    } else {
        m.push(slopes[0]);
        m.push(slopes[0]);
    }
    m.extend_from_slice(&slopes);
    if slopes.len() >= 2 {
        m.push(2.0 * slopes[nm1 - 1] - slopes[nm1 - 2]);
        m.push(3.0 * slopes[nm1 - 1] - 2.0 * slopes[nm1 - 2]);
    } else {
        m.push(slopes[nm1 - 1]);
        m.push(slopes[nm1 - 1]);
    }

    // Compute Akima weights and derivative at each knot
    let mut t = vec![0.0; n];
    for i in 0..n {
        let mi = i + 2; // offset into extended m array
        let w1 = (m[mi + 1] - m[mi]).abs();
        let w2 = (m[mi - 1] - m[mi - 2]).abs();
        let wsum = w1 + w2;
        if wsum < 1e-30 {
            t[i] = 0.5 * (m[mi - 1] + m[mi]);
        } else {
            t[i] = (w1 * m[mi - 1] + w2 * m[mi]) / wsum;
        }
    }

    // Evaluate using Hermite basis
    let result = x_new
        .iter()
        .map(|&xi| {
            if xi <= x[0] {
                return y[0];
            }
            if xi >= x[nm1] {
                return y[nm1];
            }
            let i = find_interval(x, xi);
            let hi = x[i + 1] - x[i];
            let u = (xi - x[i]) / hi;
            let u2 = u * u;
            let u3 = u2 * u;

            let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            let h10 = u3 - 2.0 * u2 + u;
            let h01 = -2.0 * u3 + 3.0 * u2;
            let h11 = u3 - u2;

            h00 * y[i] + h10 * hi * t[i] + h01 * y[i + 1] + h11 * hi * t[i + 1]
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_exact() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0]; // y = 2x
        let x_new = vec![0.5, 1.5, 2.5];
        let result = interp1d_linear(&x, &y, &x_new).unwrap();
        for (r, expected) in result.iter().zip([1.0, 3.0, 5.0].iter()) {
            assert!(
                (r - expected).abs() < 1e-10,
                "Expected {}, got {}",
                expected,
                r
            );
        }
    }

    #[test]
    fn test_cubic_smooth() {
        // Natural cubic spline on a dense grid should closely approximate y = x^2
        let n = 20;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let x_new = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let result = interp1d_cubic(&x, &y, &x_new).unwrap();
        for (&r, &xn) in result.iter().zip(x_new.iter()) {
            let expected = xn * xn;
            assert!(
                (r - expected).abs() < 0.05,
                "At x={}: expected {}, got {}",
                xn,
                expected,
                r
            );
        }

        // Cubic spline should pass through the original data points exactly
        let x_orig = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_orig: Vec<f64> = x_orig.iter().map(|&xi| xi * xi).collect();
        let result_at_knots = interp1d_cubic(&x_orig, &y_orig, &x_orig).unwrap();
        for (&r, &expected) in result_at_knots.iter().zip(y_orig.iter()) {
            assert!(
                (r - expected).abs() < 1e-10,
                "Knot mismatch: expected {}, got {}",
                expected,
                r
            );
        }
    }

    #[test]
    fn test_interp2d_linear_grid() {
        // z = x + 2*y on a 3x3 grid
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let z: Vec<f64> = y
            .iter()
            .flat_map(|&yi| x.iter().map(move |&xi| xi + 2.0 * yi))
            .collect();

        let xi = vec![0.5, 1.5];
        let yi = vec![0.5, 1.5];
        let result = interp2d_linear(&x, &y, &z, &xi, &yi).unwrap();
        // At (0.5, 0.5): 0.5 + 1.0 = 1.5
        assert!((result[0] - 1.5).abs() < 1e-10);
        // At (1.5, 1.5): 1.5 + 3.0 = 4.5
        assert!((result[1] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_akima_interpolation() {
        // Akima should handle data without wild oscillations
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2
        let x_new = vec![0.5, 2.5, 4.5];
        let result = interp1d_akima(&x, &y, &x_new).unwrap();

        // Akima won't be exact for x^2 but should be reasonable
        for (&r, &xn) in result.iter().zip(x_new.iter()) {
            let expected = xn * xn;
            assert!(
                (r - expected).abs() < 1.0,
                "At x={}: expected ~{}, got {}",
                xn,
                expected,
                r
            );
        }
    }

    #[test]
    fn test_validation_errors() {
        assert!(interp1d_linear(&[1.0], &[1.0], &[1.0]).is_err());
        assert!(interp1d_linear(&[1.0, 2.0], &[1.0], &[1.0]).is_err());
        assert!(interp1d_linear(&[2.0, 1.0], &[1.0, 2.0], &[1.5]).is_err());
    }
}
