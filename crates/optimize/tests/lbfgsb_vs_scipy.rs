//! `minimize_lbfgsb` against `scipy.optimize.minimize(method='L-BFGS-B')`.
//!
//! The old solver ran the unconstrained two-loop direction, projected the trial
//! point, took the Armijo slope from the unprojected direction and accepted a
//! failed line search even if `f` went up. On
//! `f = (x+1)² + 10(y-1)² + 3xy`, `x ≥ 0`, from `(2, 5)` it stopped after 200
//! iterations at `f = 1.01345`, not converged; SciPy reaches `(0, 1)`, `f = 1`
//! in 5 iterations.
//!
//! Reference values from SciPy 1.x, default options (`m = 10`,
//! `pgtol = 1e-5`, `factr·eps = 2.22e-9`).

use cv_optimize::general::{minimize_lbfgsb, BfgsConfig, Bounds};

fn b(lo: Option<f64>, hi: Option<f64>) -> Bounds {
    Bounds {
        lower: lo,
        upper: hi,
    }
}

fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }
    g
}

fn assert_close(name: &str, got: &[f64], want: &[f64], tol: f64) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{name}: x[{i}] = {g}, SciPy {w}");
    }
}

#[test]
fn review_problem_reaches_the_bound() {
    let f = |v: &[f64]| (v[0] + 1.0).powi(2) + 10.0 * (v[1] - 1.0).powi(2) + 3.0 * v[0] * v[1];
    let g = |v: &[f64]| {
        vec![
            2.0 * (v[0] + 1.0) + 3.0 * v[1],
            20.0 * (v[1] - 1.0) + 3.0 * v[0],
        ]
    };
    let bounds = [b(Some(0.0), None), b(None, None)];
    let r = minimize_lbfgsb(f, g, &[2.0, 5.0], &bounds, 10, &BfgsConfig::default());
    // SciPy: x = [0.0, 1.0], fun = 1.0, nit = 5, success.
    assert!(r.converged, "not converged after {} its", r.iterations);
    assert_close("review", &r.x, &[0.0, 1.0], 1e-6);
    assert!((r.fun - 1.0).abs() < 1e-10, "f = {}", r.fun);
    assert!(r.iterations <= 20, "{} iterations", r.iterations);
}

#[test]
fn rosenbrock_unbounded() {
    let bounds = [b(None, None), b(None, None)];
    let r = minimize_lbfgsb(
        rosen,
        rosen_der,
        &[-1.2, 1.0],
        &bounds,
        10,
        &BfgsConfig::default(),
    );
    // SciPy: x = [0.99999895, 0.99999802], fun = 2.8e-12, nit = 36.
    assert!(r.converged);
    assert_close("rosen2", &r.x, &[1.0, 1.0], 1e-4);
    assert!(r.fun < 1e-9, "f = {}", r.fun);
}

#[test]
fn rosenbrock_4d_with_an_active_upper_bound() {
    let bounds: Vec<Bounds> = (0..4).map(|_| b(Some(-2.0), Some(0.8))).collect();
    let r = minimize_lbfgsb(
        rosen,
        rosen_der,
        &[-1.0, 1.5, -0.5, 1.2],
        &bounds,
        10,
        &BfgsConfig::default(),
    );
    // SciPy: x = [0.8, 0.650919070655266, 0.42940162414707667,
    //             0.18438575346870636], fun = 0.5026184416097589, nit = 26.
    assert!(r.converged);
    assert_close(
        "rosen4",
        &r.x,
        &[
            0.8,
            0.650919070655266,
            0.42940162414707667,
            0.18438575346870636,
        ],
        1e-4,
    );
    assert!((r.fun - 0.5026184416097589).abs() < 1e-8, "f = {}", r.fun);
    assert_eq!(r.x[0], 0.8, "the active bound is hit exactly");
}

#[test]
fn rosenbrock_2d_minimum_on_the_box_corner_side() {
    let bounds = [b(Some(-1.5), Some(0.5)), b(Some(-1.5), Some(2.0))];
    let r = minimize_lbfgsb(
        rosen,
        rosen_der,
        &[-1.2, 1.0],
        &bounds,
        10,
        &BfgsConfig::default(),
    );
    // SciPy: x = [0.5, 0.25], fun = 0.25, nit = 20.
    assert!(r.converged);
    assert_close("rosen2 box", &r.x, &[0.5, 0.25], 1e-5);
    assert!((r.fun - 0.25).abs() < 1e-9, "f = {}", r.fun);
}

#[test]
fn quadratic_from_an_infeasible_start() {
    let a = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.5], [0.5, -0.5, 2.0]];
    let bv = [1.0, -8.0, 3.0];
    let f = move |v: &[f64]| {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += 0.5 * v[i] * a[i][j] * v[j];
            }
            s -= bv[i] * v[i];
        }
        s
    };
    let g = move |v: &[f64]| {
        (0..3)
            .map(|i| (0..3).map(|j| a[i][j] * v[j]).sum::<f64>() - bv[i])
            .collect::<Vec<f64>>()
    };
    let bounds: Vec<Bounds> = (0..3).map(|_| b(Some(-1.0), Some(1.0))).collect();
    let r = minimize_lbfgsb(f, g, &[5.0, 5.0, 5.0], &bounds, 10, &BfgsConfig::default());
    // SciPy: x = [0.375, -1.0, 1.0], fun = -8.28125, nit = 3.
    assert!(r.converged);
    assert_close("quad3", &r.x, &[0.375, -1.0, 1.0], 1e-7);
    assert!((r.fun + 8.28125).abs() < 1e-10, "f = {}", r.fun);
}

#[test]
fn objective_never_increases() {
    // Whatever happens inside, the returned point must be no worse than the
    // (projected) start.
    let f = |v: &[f64]| (v[0] + 1.0).powi(2) + 10.0 * (v[1] - 1.0).powi(2) + 3.0 * v[0] * v[1];
    let g = |v: &[f64]| {
        vec![
            2.0 * (v[0] + 1.0) + 3.0 * v[1],
            20.0 * (v[1] - 1.0) + 3.0 * v[0],
        ]
    };
    let bounds = [b(Some(0.0), None), b(None, None)];
    for max_iters in 1..8 {
        let cfg = BfgsConfig {
            max_iters,
            ..BfgsConfig::default()
        };
        let r = minimize_lbfgsb(f, g, &[2.0, 5.0], &bounds, 10, &cfg);
        assert!(r.fun <= f(&[2.0, 5.0]));
        assert!(r.x[0] >= 0.0);
        assert!((r.fun - f(&r.x)).abs() == 0.0, "fun is f(x)");
    }
}
