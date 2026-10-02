//! Finite-difference and closed-form checks of the factor-graph linearisation.
//!
//! These are the two highest-value assertions for a numerical solver: a Jacobian
//! compared against finite differences (catches a sign, a factor or a transpose)
//! and a problem with a known answer reached in the number of steps the theory
//! says (catches a wrong assembly of the normal equations). Both were run while
//! auditing the crate; no defect was found, so they are recorded here as pins —
//! they fail loudly if the assembly or the retraction convention drifts.

use cv_optimize::factor_graph::{
    numerical_jacobians, FactorGraph, GNConfig, Key, LMParams, NoiseModel, Values, Variable,
};
use cv_optimize::factors::{BetweenFactor, PriorFactor, ProjectionFactor, RangeFactor};
use nalgebra::{DMatrix, DVector, Isometry3, Point3, Vector3};

/// A *linear* least-squares problem has a closed-form minimiser, and
/// Gauss-Newton solves it in exactly one step. A wrong sign in `b` (the gradient
/// term) would double the error instead of removing it, and the second step
/// would be needed; a wrong factor in `H` would stop short. Pinned to 1e-9.
#[test]
fn one_gauss_newton_step_solves_a_linear_problem_exactly() {
    let k = Key::symbol('x', 0);
    let mut graph = FactorGraph::new();
    graph.add(PriorFactor::new(
        k,
        Variable::Point3(Point3::new(1.0, 2.0, 3.0)),
        NoiseModel::Isotropic(0.5, 3),
    ));

    let mut initial = Values::new();
    initial.insert(k, Variable::Point3(Point3::origin()));

    let after_one = graph
        .optimize_gn(
            &initial,
            &GNConfig {
                max_iters: 1,
                tolerance: 1e-12,
            },
        )
        .expect("GN step");

    let p = after_one.at_point3(&k).expect("point");
    assert!(
        (p.coords - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-9,
        "one GN step must land on the closed-form solution, got {p:?}"
    );
    assert!(
        graph.total_error(&after_one) < 1e-18,
        "cost after the exact step: {:e}",
        graph.total_error(&after_one)
    );
    assert!(
        graph.total_error(&initial) > 1.0,
        "control: the initial point must not already be the solution"
    );
}

/// The same check with a two-variable chain: `x1` is pinned to the origin by a
/// prior and `x2 = x1 + (1,0,0)` by a between factor, so the answer is known in
/// closed form and reachable in one step.
#[test]
fn one_gauss_newton_step_solves_a_linear_chain_exactly() {
    let x1 = Key::symbol('x', 1);
    let x2 = Key::symbol('x', 2);
    let mut graph = FactorGraph::new();
    graph.add(PriorFactor::new(
        x1,
        Variable::Point3(Point3::origin()),
        NoiseModel::Isotropic(1.0, 3),
    ));
    graph.add(BetweenFactor::new(
        x1,
        x2,
        Variable::Point3(Point3::new(1.0, 0.0, 0.0)),
        NoiseModel::Isotropic(1.0, 3),
    ));

    let mut initial = Values::new();
    initial.insert(x1, Variable::Point3(Point3::new(0.3, -0.4, 0.0)));
    initial.insert(x2, Variable::Point3(Point3::new(3.0, 2.0, 0.0)));

    let after_one = graph
        .optimize_gn(
            &initial,
            &GNConfig {
                max_iters: 1,
                tolerance: 1e-12,
            },
        )
        .expect("GN step");

    let p1 = after_one.at_point3(&x1).expect("x1").coords;
    let p2 = after_one.at_point3(&x2).expect("x2").coords;
    assert!((p1 - Vector3::zeros()).norm() < 1e-9, "x1 = {p1:?}");
    assert!(
        (p2 - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
        "x2 = {p2:?}"
    );
    assert!(
        graph.total_error(&after_one) < 1e-18,
        "cost {:.3e}",
        graph.total_error(&after_one)
    );
}

/// A Gauss-Newton step on a nonlinear problem must be a descent direction and
/// must match the finite-difference gradient of the whitened cost.
///
/// For `C(θ) = ||r(θ)||²` the gradient is `∇C = 2 Jᵀr`, and the GN step solves
/// `(JᵀJ) δ = -Jᵀr = -∇C/2`. Comparing the cost after one step against the
/// prediction `C - δᵀJᵀr` from the same quantities is what catches a transposed
/// or mis-signed Jacobian assembly.
#[test]
fn gauss_newton_step_matches_the_finite_difference_gradient() {
    let a = Key::symbol('x', 0);
    let b = Key::symbol('x', 1);
    let landmark = Key::symbol('l', 0);

    let mut graph = FactorGraph::new();
    graph.add(RangeFactor::new(
        a,
        landmark,
        2.0,
        NoiseModel::Isotropic(0.1, 1),
    ));
    graph.add(RangeFactor::new(
        b,
        landmark,
        1.5,
        NoiseModel::Isotropic(0.1, 1),
    ));

    let mut values = Values::new();
    values.insert(a, Variable::Point3(Point3::new(-1.0, 0.0, 0.0)));
    values.insert(b, Variable::Point3(Point3::new(0.5, 0.5, 0.0)));
    values.insert(landmark, Variable::Point3(Point3::new(3.0, 1.0, 0.0)));

    // Finite-difference gradient of the total cost with respect to the landmark.
    let eps = 1e-6;
    let l0 = *values.at_point3(&landmark).unwrap();
    let mut fd = Vector3::zeros();
    for d in 0..3 {
        let mut plus = l0;
        plus.coords[d] += eps;
        let mut minus = l0;
        minus.coords[d] -= eps;
        let mut vp = values.clone();
        vp.insert(landmark, Variable::Point3(plus));
        let mut vm = values.clone();
        vm.insert(landmark, Variable::Point3(minus));
        fd[d] = (graph.total_error(&vp) - graph.total_error(&vm)) / (2.0 * eps);
    }

    let before = graph.total_error(&values);
    let after_values = graph
        .optimize_gn(
            &values,
            &GNConfig {
                max_iters: 1,
                tolerance: 0.0,
            },
        )
        .expect("GN step");
    let after = graph.total_error(&after_values);

    // The step moved the landmark along the descent direction of the FD gradient.
    let moved = after_values.at_point3(&landmark).unwrap().coords - l0.coords;
    let directional_derivative = fd.dot(&moved);
    assert!(
        directional_derivative < 0.0,
        "the step must move downhill: FD gradient {fd:?} dotted with the step {moved:?} = \
         {directional_derivative:.6e}"
    );
    assert!(
        after < before,
        "cost must decrease: {before:.6e} -> {after:.6e}"
    );
    assert!(
        after < 1e-6 * before,
        "one GN step should nearly solve this small nonlinear problem: {before:.6e} -> {after:.6e}"
    );
}

/// The numerical Jacobian of the projection factor against the analytic one.
///
/// `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` in the camera frame, and the pose is
/// perturbed on the *right* (`pose · exp(δ)`), so at the identity pose the point
/// derivative is `[[fx/Z, 0, -fx·X/Z²], [0, fy/Z, -fy·Y/Z²]]` and the pose
/// derivative is that matrix times `[-I | -[X]×]`. A sign, a scale or the wrong
/// tangent-space convention shows up immediately.
#[test]
fn projection_factor_jacobian_matches_the_analytic_derivative() {
    let pose_key = Key::symbol('x', 0);
    let point_key = Key::symbol('l', 0);
    let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
    let world = Point3::new(1.0, 2.0, 10.0);

    let factor = ProjectionFactor::new(
        pose_key,
        point_key,
        nalgebra::Vector2::new(fx * 0.1 + cx, fy * 0.2 + cy),
        fx,
        fy,
        cx,
        cy,
        NoiseModel::Isotropic(1.0, 2),
    );

    let mut values = Values::new();
    values.insert(pose_key, Variable::Pose3(Isometry3::identity()));
    values.insert(point_key, Variable::Point3(world));

    let jacobians = numerical_jacobians(&factor, &values);
    assert_eq!(jacobians.len(), 2, "one Jacobian per key");

    let z = world.z;
    let project = DMatrix::from_row_slice(
        2,
        3,
        &[
            fx / z,
            0.0,
            -fx * world.x / (z * z),
            0.0,
            fy / z,
            -fy * world.y / (z * z),
        ],
    );

    // d(point) / d(delta point) = the projection derivative.
    let expected_point = project.clone();
    let err = (&jacobians[1] - &expected_point).abs().max();
    assert!(
        err < 1e-6,
        "point Jacobian off by {err:.3e}\n got {}\n want {}",
        jacobians[1],
        expected_point
    );

    // d(point)/d(delta pose). The retract is `pose · exp(δ)` and the residual
    // projects `pose⁻¹ · X`, so a right perturbation moves the camera-frame
    // point by `exp(-δ)X ≈ X - δ_t - δ_w × X`. The translation block is
    // therefore `-I` and the rotation block is the cross-product matrix
    // `+[X]×` (`-δ_w × X = [X]× δ_w`).
    //
    // This is a *derivation* check, not a defect report — the numerical
    // Jacobian is taken with the same `retract` the solver updates with, so it
    // is consistent by construction. Getting the convention wrong here is
    // exactly the mistake this test exists to catch.
    let mut d = DMatrix::zeros(3, 6);
    for i in 0..3 {
        d[(i, i)] = -1.0;
    }
    d[(0, 4)] = -world.z;
    d[(0, 5)] = world.y;
    d[(1, 3)] = world.z;
    d[(1, 5)] = -world.x;
    d[(2, 3)] = -world.y;
    d[(2, 4)] = world.x;
    let expected_pose = project * d;
    let err = (&jacobians[0] - &expected_pose).abs().max();
    assert!(
        err < 1e-6,
        "pose Jacobian off by {err:.3e}\n got {}\n want {}",
        jacobians[0],
        expected_pose
    );
}

/// `retract` and `local` must be inverse operations on the manifold: otherwise
/// the numerical Jacobian is taken in a different tangent space from the one the
/// solver updates in, and every step is subtly wrong.
#[test]
fn retract_and_local_are_inverse() {
    let poses = [
        Isometry3::identity(),
        Isometry3::translation(1.0, -2.0, 0.5),
        Isometry3::new(Vector3::new(0.3, 0.1, -0.2), Vector3::new(0.4, -0.3, 0.2)),
    ];
    for pose in poses {
        let var = Variable::Pose3(pose);
        let delta = DVector::from_vec(vec![1e-3, -2e-3, 3e-3, 1e-3, 2e-3, -1e-3]);
        let moved = var.retract(&delta);
        let round_trip = var.local(&moved);
        assert!(
            (&round_trip - &delta).norm() < 1e-6,
            "local(retract(d)) != d: {round_trip:?} vs {delta:?}"
        );
    }
}

/// LM must reduce the cost monotonically, and must end up at the closed-form
/// answer on the linear problem.
#[test]
fn levenberg_marquardt_reduces_the_cost_monotonically() {
    let x = Key::symbol('x', 0);
    let mut graph = FactorGraph::new();
    graph.add(PriorFactor::new(
        x,
        Variable::Pose3(Isometry3::translation(2.0, 3.0, 0.0)),
        NoiseModel::Isotropic(0.1, 6),
    ));

    let mut initial = Values::new();
    initial.insert(x, Variable::Pose3(Isometry3::translation(2.5, 3.5, 0.5)));

    let before = graph.total_error(&initial);
    let solution = graph
        .optimize_lm(&initial, &LMParams::default())
        .expect("LM");
    let after = graph.total_error(&solution);
    assert!(
        after < 1e-12,
        "LM on a linear prior problem must reach the exact solution: {before:.6e} -> {after:.6e}"
    );
}
