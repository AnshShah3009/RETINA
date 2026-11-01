//! A new factor that only touches variables that already exist must still be
//! optimised by `Isam2Solver::update`.
//!
//! `update` built its affected set from the keys in `new_values` alone (plus the
//! periodic relinearization scan). A factor added in an update that introduces
//! no new variable therefore left that set empty, hit the `affected.is_empty()`
//! early return, and was never linearised — even though it was pushed onto
//! `self.factors`. That is the ordinary case in incremental SLAM: a
//! loop-closure factor between two poses that already exist.
//!
//! Measured on the input below (two 1-D variables, a prior and two between
//! factors, the second added on its own):
//!
//! ```text
//!                                     before fix          after fix
//!   estimate of x1                    1.000000            2.980198
//!   total_error() reported            4.245e-52           4.245e-52
//!   actual cost of the graph          4.000e+04           4.245e-52
//! ```
//!
//! The reported cost and the estimate both described a graph the solver had not
//! seen: the factor was silently dropped.

use cv_optimize::factor_graph::{Key, NoiseModel, Values, Variable};
use cv_optimize::factors::{BetweenFactor, PriorFactor};
use cv_optimize::isam2::{GNParams, Isam2Config, Isam2Solver};
use nalgebra::DVector;

const SIGMA_PRIOR: f64 = 0.01;
const SIGMA_WEAK: f64 = 0.1;
const SIGMA_STRONG: f64 = 0.01;

fn scalar(v: f64) -> Variable {
    Variable::Vector(DVector::from_vec(vec![v]))
}

fn estimate(solver: &Isam2Solver, key: Key) -> f64 {
    match solver.estimate(&key) {
        Some(Variable::Vector(v)) if v.len() == 1 => v[0],
        other => panic!("expected a 1-D vector variable at {key}, got {other:?}"),
    }
}

/// Weighted cost of the three factors of the test graph, evaluated directly from
/// the two estimates.
fn cost(x0: f64, x1: f64) -> f64 {
    let d = x1 - x0;
    (x0 / SIGMA_PRIOR).powi(2)
        + ((d - 1.0) / SIGMA_WEAK).powi(2)
        + ((d - 3.0) / SIGMA_STRONG).powi(2)
}

fn solver() -> Isam2Solver {
    Isam2Solver::new(Isam2Config {
        gauss_newton_params: GNParams {
            max_iters: 30,
            tolerance: 1e-10,
        },
        ..Default::default()
    })
}

fn weak_between(k0: Key, k1: Key) -> Box<BetweenFactor> {
    Box::new(BetweenFactor::new(
        k0,
        k1,
        scalar(1.0),
        NoiseModel::Isotropic(SIGMA_WEAK, 1),
    ))
}

fn strong_between(k0: Key, k1: Key) -> Box<BetweenFactor> {
    Box::new(BetweenFactor::new(
        k0,
        k1,
        scalar(3.0),
        NoiseModel::Isotropic(SIGMA_STRONG, 1),
    ))
}

fn prior(k0: Key) -> Box<PriorFactor> {
    Box::new(PriorFactor::new(
        k0,
        scalar(0.0),
        NoiseModel::Isotropic(SIGMA_PRIOR, 1),
    ))
}

fn initial_values(k0: Key, k1: Key) -> Values {
    let mut vals = Values::new();
    vals.insert(k0, scalar(0.5));
    vals.insert(k1, scalar(2.0));
    vals
}

/// The closed-form minimiser of
/// `(x0/σp)² + ((x1-x0-1)/σw)² + ((x1-x0-3)/σs)²`:
/// the second normal equation scores the offset `d = x1 - x0` to
/// `(σw²·3 + σs²·1)/(σw² + σs²) = 30100/10100`, and the first one then fixes
/// `x0 = 0`.
fn expected_x1() -> f64 {
    let ws = SIGMA_WEAK.powi(2);
    let ss = SIGMA_STRONG.powi(2);
    (ws * 3.0 + ss * 1.0) / (ws + ss)
}

/// Adding a loop-closure factor between two existing variables must re-optimise
/// them and must leave `total_error()` describing the graph that now exists —
/// including that factor.
#[test]
fn a_new_factor_between_existing_variables_is_optimised() {
    let k0 = Key::symbol('x', 0);
    let k1 = Key::symbol('x', 1);

    let mut solver = solver();
    solver
        .update(
            vec![prior(k0), weak_between(k0, k1)],
            initial_values(k0, k1),
        )
        .expect("first update");

    // Control: the first update reached the two-factor optimum (x0 = 0, d = 1),
    // and its reported cost agrees with the estimates it returned.
    let (x0, x1) = (estimate(&solver, k0), estimate(&solver, k1));
    assert!((x1 - 1.0).abs() < 1e-6, "first update: x1 = {x1}");
    let two_factor_cost =
        |x0: f64, x1: f64| (x0 / SIGMA_PRIOR).powi(2) + (((x1 - x0) - 1.0) / SIGMA_WEAK).powi(2);
    assert!(
        (solver.total_error() - two_factor_cost(x0, x1)).abs()
            <= 1e-9 * two_factor_cost(x0, x1).max(1.0),
        "first update: reported cost {} vs computed {}",
        solver.total_error(),
        two_factor_cost(x0, x1)
    );

    // The loop-closure factor: same two variables, no new values.
    solver
        .update(vec![strong_between(k0, k1)], Values::new())
        .expect("second update");

    let (x0, x1) = (estimate(&solver, k0), estimate(&solver, k1));
    let actual = cost(x0, x1);
    assert!(
        (solver.total_error() - actual).abs() <= 1e-9 * actual.max(1.0),
        "total_error() reports {} but the factors of the graph cost {actual:e} at the \
         returned estimates (x0 = {x0}, x1 = {x1}); the new factor was never linearised",
        solver.total_error()
    );
    assert!(
        (x1 - expected_x1()).abs() < 1e-6,
        "x1 = {x1}, but the three-factor optimum is {} (it stayed at the old \
         two-factor solution 1.0)",
        expected_x1()
    );
    assert!(x0.abs() < 1e-6, "x0 = {x0}, the three-factor optimum is 0");
}

/// Control: feeding the same three factors to a single `update` must reach the
/// same point, so the test above cannot pass by relaxing the problem.
#[test]
fn the_same_three_factors_in_one_update_reach_the_same_point() {
    let k0 = Key::symbol('x', 0);
    let k1 = Key::symbol('x', 1);

    let mut solver = solver();
    solver
        .update(
            vec![prior(k0), weak_between(k0, k1), strong_between(k0, k1)],
            initial_values(k0, k1),
        )
        .expect("update");

    let (x0, x1) = (estimate(&solver, k0), estimate(&solver, k1));
    assert!((x1 - expected_x1()).abs() < 1e-6, "x1 = {x1}");
    assert!(x0.abs() < 1e-6, "x0 = {x0}");
    let actual = cost(x0, x1);
    assert!(
        (solver.total_error() - actual).abs() <= 1e-9 * actual.max(1.0),
        "reported cost {} vs computed {actual:e}",
        solver.total_error()
    );
}
