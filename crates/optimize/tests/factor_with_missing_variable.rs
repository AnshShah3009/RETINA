//! A factor may name a variable that has no value yet; that is not a panic.
//!
//! `FactorGraph::linearize` states the contract in its own words — "A factor may
//! legally reference a variable that has no value; skip it so the `offsets`
//! lookups below cannot panic on a missing key" — and `Isam2Solver::update`
//! repeats it: "skip such keys so `build_ordering`/`apply_delta` cannot panic on
//! a missing variable". Both exist because a graph is built incrementally: a
//! pose arrives with the relative constraint that reaches the *next* pose, and
//! that pose's value is delivered in a later update. Every other crate in this
//! workspace builds graphs in exactly that order.
//!
//! The contract only held for the linear system. `FactorGraph::total_error` and
//! `Isam2Solver::compute_total_error` evaluated *every* factor, and the ready-made
//! factors (`PriorFactor`, `BetweenFactor`, `RangeFactor`, `BearingFactor`,
//! `ProjectionFactor`) reach their keys through `Values::get(..).expect(..)`. So
//! the same graph that Gauss–Newton optimised happily panicked on the LM path,
//! on `total_error`, and inside `Isam2Solver::update`:
//!
//! ```text
//! optimize_gn  -> Ok, x0 = 0.0        (linearize skipped the factor)
//! optimize_lm  -> panic "key2 not found: crates/optimize/src/factors.rs"
//! total_error  -> panic "key2 not found"
//! isam2 update -> panic "key2 not found"
//! ```
//!
//! Skipping the factor is the only answer consistent with the linear system the
//! solver actually builds: a factor that contributes nothing to `H`/`b` must
//! contribute nothing to the cost those quantities are minimising.

use cv_optimize::factor_graph::{
    FactorGraph, GNConfig, Key, LMParams, NoiseModel, Values, Variable,
};
use cv_optimize::factors::{BetweenFactor, PriorFactor};
use cv_optimize::isam2::{GNParams, Isam2Config, Isam2Solver};
use nalgebra::DVector;

const SIGMA_PRIOR: f64 = 0.01;
const SIGMA_BETWEEN: f64 = 0.1;

fn scalar(v: f64) -> Variable {
    Variable::Vector(DVector::from_vec(vec![v]))
}

fn value_of(values: &Values, key: Key) -> f64 {
    match values.get(&key) {
        Some(Variable::Vector(v)) if v.len() == 1 => v[0],
        other => panic!("expected a 1-D vector variable at {key}, got {other:?}"),
    }
}

/// Prior `x0 = 0`; between `x0 --1--> x1`. Only `x0` has a value.
fn graph_and_values() -> (FactorGraph, Values, Key, Key) {
    let x0 = Key::symbol('x', 0);
    let x1 = Key::symbol('x', 1);
    let mut graph = FactorGraph::new();
    graph.add(PriorFactor::new(
        x0,
        scalar(0.0),
        NoiseModel::Isotropic(SIGMA_PRIOR, 1),
    ));
    graph.add(BetweenFactor::new(
        x0,
        x1,
        scalar(1.0),
        NoiseModel::Isotropic(SIGMA_BETWEEN, 1),
    ));
    let mut values = Values::new();
    values.insert(x0, scalar(0.5));
    (graph, values, x0, x1)
}

/// Weighted squared error of the one factor whose variables all have values.
fn prior_only_cost(x0: f64) -> f64 {
    (x0 / SIGMA_PRIOR).powi(2)
}

/// `total_error` must report the cost of the factors it can evaluate, not panic
/// on the one it cannot.
#[test]
fn total_error_skips_a_factor_whose_variable_has_no_value() {
    let (graph, values, x0, _x1) = graph_and_values();

    let reported = graph.total_error(&values);
    let expected = prior_only_cost(value_of(&values, x0));
    assert!(
        (reported - expected).abs() <= 1e-9 * expected,
        "total_error returned {reported:e}, but the factors that can be evaluated cost \
         {expected:e} (the between factor names a variable with no value and used to \
         panic here instead)"
    );
    assert!(
        (reported - 2500.0).abs() < 1e-6,
        "sanity: (0.5 / 0.01)^2 = 2500, got {reported}"
    );
}

/// The LM path reported on the same graph must not panic, and must optimise the
/// factors it can see — the same ones Gauss–Newton already optimises.
#[test]
fn optimize_lm_skips_a_factor_whose_variable_has_no_value() {
    let (graph, values, x0, x1) = graph_and_values();

    let solution = graph
        .optimize_lm(&values, &LMParams::default())
        .expect("LM must not fail on a factor whose variable has no value");

    assert!(
        !solution.contains(&x1),
        "the factor's second variable had no value, so LM cannot have invented one"
    );
    assert!(
        value_of(&solution, x0).abs() < 1e-6,
        "x0 = {}, the prior x0 = 0 alone puts it at 0",
        value_of(&solution, x0)
    );

    // The returned values now describe a graph the solver can fully evaluate:
    // the between factor is still unsolvable, so the cost is the prior's alone.
    assert!(
        graph.total_error(&solution) < 1e-12,
        "cost after LM: {:e}",
        graph.total_error(&solution)
    );
}

/// Control: with every variable present the very same graph must report *both*
/// factors, so the tests above cannot pass by having dropped the between factor
/// from the normal case.
#[test]
fn total_error_still_counts_every_factor_when_all_values_are_present() {
    let (graph, values, x0, x1) = graph_and_values();
    let mut complete = values.clone();
    complete.insert(x1, scalar(1.0));

    let expected = prior_only_cost(value_of(&complete, x0))
        + ((value_of(&complete, x1) - value_of(&complete, x0) - 1.0) / SIGMA_BETWEEN).powi(2);
    let reported = graph.total_error(&complete);
    assert!(
        (reported - expected).abs() <= 1e-9 * expected,
        "total_error returned {reported:e}, the two factors cost {expected:e}"
    );
    // And the between factor is genuinely part of that number.
    assert!(
        expected > prior_only_cost(value_of(&complete, x0)),
        "sanity: the control must include the between factor's contribution"
    );

    // Gauss-Newton reaches the same answer on the complete graph, so the
    // skipping path and the normal path agree when nothing is missing.
    let solution = graph
        .optimize_gn(&complete, &GNConfig::default())
        .expect("GN on the complete graph");
    assert!(value_of(&solution, x0).abs() < 1e-6);
    assert!((value_of(&solution, x1) - 1.0).abs() < 1e-6);
}

/// The incremental solver has the same contract, and it is the one that hits it
/// in practice: pose 0 arrives with the constraint that reaches pose 1, and pose
/// 1's value is delivered on the next update.
#[test]
fn isam2_update_accepts_a_factor_whose_variable_arrives_later() {
    let x0 = Key::symbol('x', 0);
    let x1 = Key::symbol('x', 1);

    let mut solver = Isam2Solver::new(Isam2Config {
        gauss_newton_params: GNParams {
            max_iters: 30,
            tolerance: 1e-10,
        },
        ..Default::default()
    });

    let mut first = Values::new();
    first.insert(x0, scalar(0.5));
    solver
        .update(
            vec![
                Box::new(PriorFactor::new(
                    x0,
                    scalar(0.0),
                    NoiseModel::Isotropic(SIGMA_PRIOR, 1),
                )),
                Box::new(BetweenFactor::new(
                    x0,
                    x1,
                    scalar(1.0),
                    NoiseModel::Isotropic(SIGMA_BETWEEN, 1),
                )),
            ],
            first,
        )
        .expect("a factor whose second variable has no value yet must not panic");

    assert_eq!(solver.num_variables(), 1, "only x0 has a value yet");
    assert_eq!(solver.num_factors(), 2, "both factors are held for later");
    assert!(
        solver.total_error() < 1e-12,
        "the prior x0 = 0 is the only evaluable constraint: error {:e}",
        solver.total_error()
    );
    let x0_estimate = match solver.estimate(&x0) {
        Some(Variable::Vector(v)) if v.len() == 1 => v[0],
        other => panic!("expected a 1-D vector at x0, got {other:?}"),
    };
    assert!(
        x0_estimate.abs() < 1e-6,
        "x0 = {x0_estimate}, the prior x0 = 0 puts it at 0"
    );

    // Now pose 1 arrives: the between factor becomes evaluable and must move x1.
    let mut second = Values::new();
    second.insert(x1, scalar(1.4));
    solver
        .update(vec![], second)
        .expect("second update with the missing value");

    let x1_estimate = match solver.estimate(&x1) {
        Some(Variable::Vector(v)) if v.len() == 1 => v[0],
        other => panic!("expected a 1-D vector at x1, got {other:?}"),
    };
    assert!(
        (x1_estimate - 1.0).abs() < 1e-6,
        "x1 = {x1_estimate}, the between factor wants x1 - x0 = 1 with x0 = 0"
    );
    assert!(
        solver.total_error() < 1e-12,
        "both factors are now evaluable and satisfied: error {:e}",
        solver.total_error()
    );
}
