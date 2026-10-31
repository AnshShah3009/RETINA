//! Regression tests for the CG **breakdown criterion**, as distinct from its
//! convergence criterion.
//!
//! `CgSolver::run` and `SparseLMSolver::solve_lm_step` both carried
//! `if p_ap.abs() < 1e-10 { break; }`. That looks like a guard against dividing
//! by zero; it is not one. `pᵀAp` is quadratic in the units of the residual, so
//! a fixed absolute threshold fires *while the system is still perfectly
//! solvable* whenever the problem is expressed in small units — and it fires at
//! a different iteration for every overall scale, so the answer depended on the
//! units the residual happened to be measured in.
//!
//! The only genuine CG breakdown is `pᵀAp <= 0`: `A` has no energy along `p`,
//! and `alpha = rsold/pap` is undefined. A small-but-positive `pᵀAp` is the
//! opposite situation — CG is near convergence and needs a *large* step to
//! finish.
//!
//! Baseline for every "before" number below: commit `4029e46`.

use cv_optimize::sparse::{CgSolver, LinearSolver, SparseMatrix, Triplet};
use cv_optimize::CostFunction;
use nalgebra::DVector;

fn cpu_device() -> cv_hal::compute::ComputeDevice<'static> {
    let cpu = Box::leak(Box::new(
        cv_hal::cpu::CpuBackend::new().expect("CPU backend"),
    ));
    cv_hal::compute::ComputeDevice::Cpu(cpu)
}

/// `A = diag(1..n) * scale`: SPD, well conditioned, and `x = 1` is an exact
/// solution of `A x = b` for `b = A·1`. `|x - 1|` is therefore the entire error.
fn diagonal_spd(n: usize, scale: f64) -> (SparseMatrix, DVector<f64>) {
    let mut triplets = Vec::with_capacity(n);
    let mut b = DVector::zeros(n);
    for i in 0..n {
        let v = (1.0 + i as f64) * scale;
        triplets.push(Triplet::new(i, i, v));
        b[i] = v; // A · ones
    }
    (SparseMatrix::from_triplets(n, n, &triplets), b)
}

/// **The defect.** An SPD system with a known exact solution must be solved to
/// that solution regardless of the units the matrix is expressed in. Scaling
/// `A` and `b` together by `1/s` leaves the answer *mathematically identical*;
/// before the fix the answer degraded by eight orders of magnitude across the
/// scale range and ended up worse than the zero step.
///
/// Measured against `4029e46` (`|x - 1|`, lower is better):
///
/// | scale | before | after |
/// | ---: | ---: | ---: |
/// | `1e0` | `5.6e-08` | `7.7e-17` |
/// | `1e-3` | `1.2e-02` | `3.9e-11` |
/// | `1e-5` | `3.2e+00` | `3.9e-06` |
/// | `1e-6` | `6.3e+00` | (finite, small) |
#[test]
fn cg_solves_an_spd_system_regardless_of_the_units_it_is_given() {
    let device = cpu_device();
    let solver = CgSolver {
        max_iters: 1000,
        tolerance: 1e-12,
    };
    let truth = DVector::from_element(40, 1.0);

    // Every one of these is the *same* linear problem written in different
    // units, so every one of them has the same correct answer.
    for scale in [1.0f64, 1e-3, 1e-5, 1e-6, 1e-7] {
        let (a, b) = diagonal_spd(40, scale);
        let (x, converged) = solver
            .solve_relaxed(&device, &a, &b)
            .expect("a device/storage failure, not non-convergence");

        assert!(
            x.iter().all(|v| v.is_finite()),
            "scale={scale:.0e}: returned a non-finite iterate: {:?}",
            x.as_slice()
        );
        // Scale *invariant* bound: the error must be judged against the size of
        // the system, not against a unit-dependent constant.
        let relative = (&x - &truth).norm() / truth.norm();
        assert!(
            relative < 1e-4,
            "scale={scale:.0e}: |x-1|/|1| = {relative:.3e}. The answer changed when only \
             the units changed, so the breakdown test is not scale-free \
             (converged={converged})."
        );
    }
}

/// **The control.** The same solver, at the same scales, on the system it was
/// always good at. If the test above could pass by rejecting everything, this
/// fails — and it also pins the case where the old threshold *happened* to be
/// fine, so a future "fix" that simply loosens the check for everyone cannot
/// hide here.
#[test]
fn cg_control_still_meets_its_tolerance_on_an_unscaled_system() {
    let device = cpu_device();
    let solver = CgSolver {
        max_iters: 2000,
        tolerance: 1e-12,
    };
    for scale in [1.0f64, 1e-3] {
        let (a, b) = diagonal_spd(40, scale);
        let x = solver
            .solve(&device, &a, &b)
            .unwrap_or_else(|e| panic!("scale={scale:.0e} failed to converge: {e}"));
        let relative = (&x - &DVector::from_element(40, 1.0)).norm();
        assert!(relative < 1e-6, "scale={scale:.0e}: |x-1| = {relative:.3e}");
    }
}

/// **The control on the other side.** `A = 0` is the true breakdown case: there
/// is no energy along any direction, `pᵀAp == 0`, and no step exists. The
/// solver must report it rather than divide by zero or invent an answer — and
/// `b = 0` is the one instance where `x = 0` really is the exact solution.
#[test]
fn cg_control_reports_a_true_breakdown_instead_of_guessing() {
    let device = cpu_device();
    let solver = CgSolver {
        max_iters: 100,
        tolerance: 1e-12,
    };

    // A = 0, b = 0 -> the answer is exactly zero and is attained at iteration 0.
    let a_zero = SparseMatrix::from_triplets(3, 3, &[]);
    let x = solver
        .solve(&device, &a_zero, &DVector::zeros(3))
        .expect("x = 0 solves A x = 0 for any A, including A = 0");
    assert_eq!(x.as_slice(), &[0.0, 0.0, 0.0]);

    // A = 0, b != 0 -> genuinely unsolvable, must not be reported as a solution.
    match solver.solve(&device, &a_zero, &DVector::from_element(3, 1.0)) {
        Ok(x) => {
            let residual =
                (&a_zero.spmv_native(&x).unwrap() - &DVector::from_element(3, 1.0)).norm();
            panic!(
                "A = 0, b = 1 returned Ok({:?}) with ||Ax-b|| = {residual:.3e}",
                x.as_slice()
            );
        }
        Err(e) => assert!(e.contains("did not converge"), "{e}"),
    }
}

/// The LM step has its own copy of the loop, carrying *two* independent
/// units-dependences, and both had to go.
///
/// 1. `if pᵀAp.abs() < 1e-10 { break; }` returned the **zero vector** it
///    started from. A zero step is what a *rejected* step looks like, so
///    `SparseLMSolver::minimize` filed it under the rejection branch, multiplied
///    lambda by 10 twelve times, and gave up — returning the initial guess while
///    reporting nothing.
/// 2. The stopping rule `||residual|| < self.config.tolerance` was *absolute*
///    while the right-hand side it solves, `rhs = -Jᵀr`, is in the caller's
///    residual units. Once the residuals got small the very first iteration
///    returned, having solved none of the system — so the step was CG's initial
///    guess `delta = -rhs`, a nearly *fixed* vector, and LM oscillated with it
///    forever inside a residual it had declared "converged".
///
/// Measured against `4029e46`, fitting `a·x + b = y` (closed form `a = 2,
/// b = 1`) while scaling only the residuals by `s` — the same fit:
///
/// | scale | before `a` | before `b` | after `a` | after `b` |
/// | ---: | ---: | ---: | ---: | ---: |
/// | `1e0` | `2.000` | `1.000` | `2.000` | `1.000` |
/// | `1e-3` | `2.060` | `0.152` | `2.000` | `1.000` |
/// | `1e-4` | `0.021` | `0.0016` | `2.000` | `1.000` |
/// | `1e-5` | **`0.000`** | **`0.000`** | `2.000` | `1.000` |
/// | `1e-6` | **`0.000`** | **`0.000`** | `2.000` | `1.000` |
///
/// With only the breakdown test fixed, `1e-5` and below reached
/// `(2.052, 0.294)` — no longer frozen, but still 0.7 wrong in `b`, which is
/// what exposed the second cause.
#[test]
fn sparse_lm_fits_a_closed_form_problem_regardless_of_residual_units() {
    struct LinearFit {
        y: DVector<f64>,
        scale: f64,
    }
    impl cv_optimize::CostFunction for LinearFit {
        fn dimensions(&self) -> (usize, usize) {
            (self.y.len(), 2)
        }
        fn residuals(&self, p: &DVector<f64>) -> DVector<f64> {
            let (a, b) = (p[0], p[1]);
            DVector::from_iterator(
                self.y.len(),
                self.y
                    .iter()
                    .enumerate()
                    .map(|(i, &yi)| self.scale * (a * (i as f64 + 1.0) + b - yi)),
            )
        }
        fn jacobian(&self, _p: &DVector<f64>) -> SparseMatrix {
            let mut t = Vec::new();
            for i in 0..self.y.len() {
                t.push(Triplet::new(i, 0, self.scale * (i as f64 + 1.0)));
                t.push(Triplet::new(i, 1, self.scale));
            }
            SparseMatrix::from_triplets(self.y.len(), 2, &t)
        }
    }

    let device = cpu_device();
    // y = 2x + 1 on x = 1..20, so the closed-form answer is (a, b) = (2, 1).
    let y = DVector::from_iterator(20, (1..=20).map(|i| 2.0 * i as f64 + 1.0));

    for scale in [1.0f64, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8] {
        let fit = LinearFit {
            y: y.clone(),
            scale,
        };
        let solver = cv_optimize::SparseLMSolver::new(&device);
        let x = solver
            .minimize(&fit, DVector::from_vec(vec![0.0, 0.0]))
            .unwrap_or_else(|e| panic!("scale={scale:.0e}: {e}"));
        let (a, b) = (x[0], x[1]);
        assert!(
            (a - 2.0).abs() < 1e-3 && (b - 1.0).abs() < 1e-3,
            "scale={scale:.0e}: LM returned (a, b) = ({a:.6}, {b:.6}); the closed-form \
             answer is (2, 1) and the residual scaling does not change the problem"
        );
    }
}

/// Control for the LM test above: the unscaled fit was already correct, and it
/// must stay correct. It also checks the result against the *cost*, not just
/// the parameters — a returned `x` whose residual is worse than the initial
/// guess would be a second, independent failure this test would otherwise miss.
#[test]
fn sparse_lm_control_returns_a_point_that_actually_reduces_the_cost() {
    struct LinearFit {
        y: DVector<f64>,
    }
    impl cv_optimize::CostFunction for LinearFit {
        fn dimensions(&self) -> (usize, usize) {
            (self.y.len(), 2)
        }
        fn residuals(&self, p: &DVector<f64>) -> DVector<f64> {
            let (a, b) = (p[0], p[1]);
            DVector::from_iterator(
                self.y.len(),
                self.y
                    .iter()
                    .enumerate()
                    .map(|(i, &yi)| a * (i as f64 + 1.0) + b - yi),
            )
        }
        fn jacobian(&self, _p: &DVector<f64>) -> SparseMatrix {
            let mut t = Vec::new();
            for i in 0..self.y.len() {
                t.push(Triplet::new(i, 0, i as f64 + 1.0));
                t.push(Triplet::new(i, 1, 1.0));
            }
            SparseMatrix::from_triplets(self.y.len(), 2, &t)
        }
    }
    let fit = LinearFit {
        y: DVector::from_iterator(20, (1..=20).map(|i| 2.0 * i as f64 + 1.0)),
    };
    let device = cpu_device();
    let start = DVector::from_vec(vec![0.0, 0.0]);
    let solver = cv_optimize::SparseLMSolver::new(&device);
    let x = solver.minimize(&fit, start.clone()).expect("minimize");

    let before = fit.residuals(&start).norm_squared();
    let after = fit.residuals(&x).norm_squared();
    assert!(
        after < before * 1e-10,
        "LM returned a point whose cost is {after:.3e}; the start was {before:.3e}"
    );
    assert!((x[0] - 2.0).abs() < 1e-4, "a = {}, expected 2", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-4, "b = {}, expected 1", x[1]);
}
