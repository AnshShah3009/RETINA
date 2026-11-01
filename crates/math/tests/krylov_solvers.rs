//! `cg_solve` / `gmres_solve`: lucky breakdown, scale-free thresholds, and no
//! `Ok` for an unconverged solve.
//!
//! Before the fix:
//! * GMRES left the Arnoldi loop on a lucky breakdown *before* applying the
//!   earlier Givens rotations to column `j`, so the triangular solve used an
//!   unrotated column: `diag(1, 2)`, `b = [1, 1]`, one cycle of 2 returned a
//!   relative residual of 0.107 (Krylov dimension 2 solves it exactly).
//! * Breakdown / pivot tests were absolute (`1e-15`, `1e-30`): `1e-16 ·
//!   diag(1, 2, 3)` gave `Ok([0, 0, 0])`, relative residual 1.0.
//! * Both returned `Ok` after the iteration cap whatever the residual:
//!   `cg_solve` with `max_iters = 1` on `[[4, 1], [1, 3]]` gave `Ok` at relative
//!   residual 0.71.
//!
//! References are exact solutions (`numpy.linalg.solve`).

use cv_math::sparse::{CsrMatrix, cg_solve, gmres_solve};
use nalgebra::DVector;

fn diag(d: &[f64]) -> CsrMatrix {
    let t: Vec<(usize, usize, f64)> = d.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
    CsrMatrix::from_triplets(d.len(), d.len(), &t)
}

fn rel_residual(a: &CsrMatrix, x: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (b - a.spmv(x)).norm() / b.norm()
}

#[test]
fn gmres_lucky_breakdown_applies_the_earlier_rotations() {
    let a = diag(&[1.0, 2.0]);
    let b = DVector::from_vec(vec![1.0, 1.0]);
    let x = gmres_solve(&a, &b, 2, 2, 1e-12).expect("Krylov dimension 2 is exact");
    assert!(
        (x[0] - 1.0).abs() < 1e-14 && (x[1] - 0.5).abs() < 1e-14,
        "{x}"
    );
}

#[test]
fn tiny_scale_systems_are_solved_not_zeroed() {
    let a = diag(&[1e-16, 2e-16, 3e-16]);
    let b = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    let want = [1e16, 5e15, 1e16 / 3.0];
    let xg = gmres_solve(&a, &b, 3, 3, 1e-12).expect("gmres");
    let xc = cg_solve(&a, &b, 3, 1e-12).expect("cg");
    for x in [&xg, &xc] {
        for i in 0..3 {
            assert!(((x[i] - want[i]) / want[i]).abs() < 1e-12, "{x}");
        }
        assert!(rel_residual(&a, x, &b) < 1e-12);
    }
    // And the same at the other end of the range.
    let a = diag(&[1e16, 2e16, 3e16]);
    let xg = gmres_solve(&a, &b, 3, 3, 1e-12).expect("gmres");
    let xc = cg_solve(&a, &b, 3, 1e-12).expect("cg");
    assert!(rel_residual(&a, &xg, &b) < 1e-12);
    assert!(rel_residual(&a, &xc, &b) < 1e-12);
}

#[test]
fn cg_iteration_cap_is_an_error_with_the_residual() {
    let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let e = cg_solve(&a, &b, 1, 1e-10).expect_err("one CG step cannot solve a 2x2");
    assert!(
        e.contains("did not converge") && e.contains("residual"),
        "{e}"
    );
    // Control: two steps solve it (x = [1/11, 7/11]).
    let x = cg_solve(&a, &b, 2, 1e-10).expect("two CG steps are exact");
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-12 && (x[1] - 7.0 / 11.0).abs() < 1e-12);
}

#[test]
fn gmres_iteration_cap_is_an_error_with_the_residual() {
    // Non-symmetric 5x5; restart 2 and 2 iterations cannot reach 1e-12.
    let t = [
        (0, 0, 4.0),
        (0, 1, 1.0),
        (1, 0, -1.0),
        (1, 1, 3.0),
        (1, 2, 0.5),
        (2, 2, 5.0),
        (2, 3, 2.0),
        (3, 1, 1.0),
        (3, 3, 2.0),
        (3, 4, -1.0),
        (4, 0, 0.5),
        (4, 4, 3.0),
    ];
    let a = CsrMatrix::from_triplets(5, 5, &t);
    let b = DVector::from_vec(vec![1.0, -2.0, 3.0, 0.5, 1.0]);
    let e = gmres_solve(&a, &b, 2, 2, 1e-12).expect_err("2 iterations are not enough");
    assert!(
        e.contains("did not converge") && e.contains("residual"),
        "{e}"
    );
    // Control: restarted GMRES(2) converges with budget; numpy.linalg.solve:
    let want = [
        0.3972602739726028,
        -0.589041095890411,
        0.3287671232876712,
        0.678082191780822,
        0.2671232876712329,
    ];
    let x = gmres_solve(&a, &b, 2, 200, 1e-12).expect("converges");
    for i in 0..5 {
        assert!((x[i] - want[i]).abs() < 1e-10, "x[{i}] = {}", x[i]);
    }
}

#[test]
fn cg_rejects_an_indefinite_matrix() {
    let a = diag(&[1.0, -1.0]);
    let b = DVector::from_vec(vec![1.0, 1.0]);
    assert!(cg_solve(&a, &b, 10, 1e-10).is_err());
}

#[test]
fn zero_rhs_is_the_zero_solution() {
    let a = diag(&[1.0, 2.0]);
    let b = DVector::zeros(2);
    assert_eq!(cg_solve(&a, &b, 5, 1e-10).unwrap(), DVector::zeros(2));
    assert_eq!(gmres_solve(&a, &b, 2, 5, 1e-10).unwrap(), DVector::zeros(2));
}
