//! Regression tests for defects found by auditing the numerical solvers.
//!
//! Each test here failed on the code that shipped before the corresponding fix;
//! the failure text is recorded in the module docs of the fixed functions. They
//! are integration tests so that they exercise the public API only — the same
//! surface a caller sees.

use cv_optimize::general::{brentq, curve_fit, newton};
use cv_optimize::gpu_solver::GpuCgSolver;
use cv_optimize::pose_graph::PoseGraph;
use cv_optimize::sparse::{CgSolver, LinearSolver, SparseMatrix, Triplet};
use nalgebra::{DVector, Isometry3, Matrix6, Vector3};

fn cpu_device() -> cv_hal::cpu::CpuBackend {
    cv_hal::cpu::CpuBackend::new().expect("CPU backend")
}

// ── newton ──────────────────────────────────────────────────────────────────

/// `x^3 - 2x + 2` maps `0 -> 1 -> 0` forever, so Newton never converges from
/// `x0 = 0`. Before the fix it returned `Ok(0.0)` after 100 iterations, with
/// `|f(0)| = 2.0` — a "root" that is off by 2, with no error flag.
#[test]
fn newton_iteration_cap_is_not_a_root() {
    let f = |x: f64| x.powi(3) - 2.0 * x + 2.0;
    let fp = |x: f64| 3.0 * x * x - 2.0;
    for max_iter in [1usize, 2, 5, 100] {
        match newton(f, fp, 0.0, 1e-12, max_iter) {
            Ok(x) => panic!(
                "max_iter={max_iter}: returned Ok({x}) with |f(x)| = {}",
                f(x).abs()
            ),
            Err(e) => assert!(e.contains("did not converge"), "{e}"),
        }
    }
}

/// Control: the same cubic does have a real root at ≈ -1.7692923542, reachable
/// from a starting point that is not on the 2-cycle.
#[test]
fn newton_control_finds_the_real_root() {
    let f = |x: f64| x.powi(3) - 2.0 * x + 2.0;
    let fp = |x: f64| 3.0 * x * x - 2.0;
    let root = newton(f, fp, -2.0, 1e-12, 100).expect("converges from -2");
    assert!(
        (root + 1.769_292_354_238_631_4).abs() < 1e-9,
        "root = {root}"
    );
}

// ── brentq ──────────────────────────────────────────────────────────────────

/// Before the fix: `brentq(x^2 - 2, 1, 2, 1e-12, 5)` returned `Ok(1.41414141)`
/// with `|f| = 2.04e-4`, eight orders of magnitude above the requested
/// tolerance, indistinguishable from a converged result.
#[test]
fn brentq_iteration_cap_is_not_a_root() {
    let f = |x: f64| x * x - 2.0;
    for max_iter in [0usize, 1, 2, 3, 5] {
        match brentq(f, 1.0, 2.0, 1e-12, max_iter) {
            Ok(x) => panic!(
                "max_iter={max_iter}: returned Ok({x}) with |f(x)| = {:.3e}",
                f(x).abs()
            ),
            Err(e) => assert!(e.contains("did not converge"), "{e}"),
        }
    }
}

/// Control: the same problem with an adequate budget converges to sqrt(2).
#[test]
fn brentq_control_converges() {
    let root = brentq(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100).expect("converges");
    assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10, "{root}");
}

/// Control: on a steep function the bracket-width criterion is the honest
/// stopping rule and must still be reported as success.
#[test]
fn brentq_control_accepts_a_narrow_bracket() {
    let root = brentq(|x| 1e9 * (x - 1.5), 1.0, 2.0, 1e-12, 100).expect("bracket narrows");
    assert!((root - 1.5).abs() <= 1e-12, "{root}");
}

// ── conjugate gradient ──────────────────────────────────────────────────────

fn laplacian(n: usize) -> SparseMatrix {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: 2.0,
        });
        if i > 0 {
            triplets.push(Triplet {
                row: i,
                col: i - 1,
                val: -1.0,
            });
            triplets.push(Triplet {
                row: i - 1,
                col: i,
                val: -1.0,
            });
        }
    }
    SparseMatrix::from_triplets(n, n, &triplets)
}

/// Before the fix: a 200x200 Laplacian solved with `max_iters = 10,
/// tolerance = 1e-10` returned `Ok` with `||Ax - b|| = 1.28e2`.
#[test]
fn cg_iteration_cap_is_not_a_solution() {
    let a = laplacian(200);
    let b = DVector::from_element(200, 1.0);
    let cpu = cpu_device();
    let device = cv_hal::compute::ComputeDevice::Cpu(&cpu);
    let solver = CgSolver {
        max_iters: 10,
        tolerance: 1e-10,
    };
    match solver.solve(&device, &a, &b) {
        Ok(x) => {
            let residual = (&a.spmv_native(&x).unwrap() - &b).norm();
            panic!("unconverged solve returned Ok with ||Ax - b|| = {residual:.3e}");
        }
        Err(e) => assert!(e.contains("did not converge"), "{e}"),
    }
}

/// Control: with a realistic budget the returned vector meets its own bound.
#[test]
fn cg_control_converges() {
    let a = laplacian(200);
    let b = DVector::from_element(200, 1.0);
    let cpu = cpu_device();
    let device = cv_hal::compute::ComputeDevice::Cpu(&cpu);
    let solver = CgSolver {
        max_iters: 2000,
        tolerance: 1e-10,
    };
    let x = solver.solve(&device, &a, &b).expect("converges");
    assert!((&a.spmv_native(&x).unwrap() - &b).norm() < 1e-10);
}

/// The same defect in the GPU solver's CPU fallback (its own doc comment said
/// "Return best solution even if not fully converged").
#[test]
fn gpu_cg_iteration_cap_is_not_a_solution() {
    let a = laplacian(200);
    let b = DVector::from_element(200, 1.0);
    let ctx = cv_hal::compute::get_device().expect("device");
    let solver = GpuCgSolver::new()
        .with_tolerance(1e-10)
        .with_max_iterations(10);
    match solver.solve(&ctx, &a, &b) {
        Ok(x) => {
            let ax = a.spmv_native(&x).unwrap();
            panic!(
                "unconverged solve returned Ok with ||Ax - b|| = {:.3e}",
                (&ax - &b).norm()
            );
        }
        Err(e) => assert!(e.contains("did not converge"), "{e}"),
    }
}

/// Control: the GPU solver with an adequate budget converges.
#[test]
fn gpu_cg_control_converges() {
    let a = laplacian(200);
    let b = DVector::from_element(200, 1.0);
    let ctx = cv_hal::compute::get_device().expect("device");
    let solver = GpuCgSolver::new()
        .with_tolerance(1e-10)
        .with_max_iterations(2000);
    let x = solver.solve(&ctx, &a, &b).expect("converges");
    assert!((&a.spmv_native(&x).unwrap() - &b).norm() < 1e-10);
}

// ── pose graph ──────────────────────────────────────────────────────────────

/// Weighted sum of squared edge residuals, recomputed from the public node and
/// edge state (the same quantity `PoseGraph::optimize` reports).
fn pose_graph_cost(graph: &PoseGraph) -> f64 {
    let mut total = 0.0;
    for edge in &graph.edges {
        let error_se3 = edge.measurement.inverse()
            * (graph.nodes[&edge.from].inverse() * graph.nodes[&edge.to]);
        let e = nalgebra::Vector6::new(
            error_se3.translation.vector.x,
            error_se3.translation.vector.y,
            error_se3.translation.vector.z,
            error_se3.rotation.scaled_axis().x,
            error_se3.rotation.scaled_axis().y,
            error_se3.rotation.scaled_axis().z,
        );
        total += e.dot(&(edge.information * e));
    }
    total
}

fn chain_graph() -> PoseGraph {
    let mut graph = PoseGraph::new();
    graph.add_node(0, Isometry3::identity());
    graph.set_fixed(0);
    graph.add_node(1, Isometry3::translation(1.6, 0.4, 0.0));
    graph.add_node(2, Isometry3::translation(3.4, -0.2, 0.0));
    graph.add_edge(
        0,
        1,
        Isometry3::translation(1.0, 0.0, 0.0),
        Matrix6::identity(),
    );
    graph.add_edge(
        1,
        2,
        Isometry3::translation(1.0, 0.0, 0.0),
        Matrix6::identity(),
    );
    graph
}

/// The returned cost must describe the poses that are returned. Before the fix
/// a single iteration returned `1.52e0` (the cost *before* the update) while the
/// poses it handed back cost `7.15e-12`.
#[test]
fn pose_graph_returned_cost_matches_the_returned_poses() {
    for iterations in [1usize, 2, 3, 30] {
        let mut graph = chain_graph();
        let before = pose_graph_cost(&graph);
        let returned = graph.optimize(iterations).expect("optimizes");
        let actual = pose_graph_cost(&graph);
        assert!(
            (returned - actual).abs() <= 1e-12 * actual.max(1.0),
            "iterations={iterations}: returned {returned:.3e}, poses cost {actual:.3e}"
        );
        assert!(
            actual < before,
            "iterations={iterations}: no progress ({before:.3e} -> {actual:.3e})"
        );
    }
}

/// Control: the solver still reaches the closed-form chain solution.
#[test]
fn pose_graph_control_converges() {
    let mut graph = chain_graph();
    let error = graph.optimize(30).expect("optimizes");
    assert!(error < 1e-12, "final cost {error:.3e}");
    assert!((graph.nodes[&1].translation.vector - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((graph.nodes[&2].translation.vector - Vector3::new(2.0, 0.0, 0.0)).norm() < 1e-6);
}

// ── curve_fit ───────────────────────────────────────────────────────────────

/// `y = a·b·x` identifies the two parameters only up to a common scale, so
/// `JᵀJ` is singular. Before the fix the covariance came back exactly zero —
/// "both parameters known exactly" — in the one case where they are not
/// determined at all.
#[test]
fn curve_fit_singular_normal_matrix_does_not_report_zero_covariance() {
    let x_data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let noise = |i: usize| ((i * 29 % 7) as f64 - 3.0) * 0.05;
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &x)| 6.0 * x + noise(i))
        .collect();
    let model = |x: f64, p: &[f64]| p[0] * p[1] * x;

    let res = curve_fit(model, &x_data, &y_data, &[2.0, 3.0], 100).expect("fits");
    assert!(
        (res.params[0] * res.params[1] - 6.0).abs() < 0.1,
        "fit is still right: {:?}",
        res.params
    );
    assert!(
        res.covariance[0][0] > 0.0 && res.covariance[1][1] > 0.0,
        "unidentifiable parameters reported zero variance: {:?}",
        res.covariance
    );
}

/// Control: for a well-conditioned fit the covariance equals the closed form
/// `(JᵀJ)⁻¹·RSS/(m-np)` — so the test above cannot pass by reporting *any*
/// non-zero matrix.
#[test]
fn curve_fit_covariance_control_matches_the_closed_form() {
    let x_data: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let noise = |i: usize| ((i * 37 % 11) as f64 - 5.0) * 0.01;
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &x)| 2.0 * x + 1.0 + noise(i))
        .collect();
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let res = curve_fit(model, &x_data, &y_data, &[0.0, 0.0], 100).expect("fits");

    let m = x_data.len() as f64;
    let sxx: f64 = x_data.iter().map(|x| x * x).sum();
    let sx: f64 = x_data.iter().sum();
    let det = sxx * m - sx * sx;
    let rss: f64 = res.residuals.iter().map(|r| r * r).sum();
    let s2 = rss / (m - 2.0);
    let expected = [
        [m / det * s2, -sx / det * s2],
        [-sx / det * s2, sxx / det * s2],
    ];
    let scale = expected[0][0].abs() + expected[1][1].abs();
    assert!(scale > 0.0);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (res.covariance[i][j] - expected[i][j]).abs() <= 1e-5 * scale,
                "cov[{i}][{j}] = {} vs closed form {}",
                res.covariance[i][j],
                expected[i][j]
            );
        }
    }
}
