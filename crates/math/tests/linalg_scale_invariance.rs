//! Regression tests for the linalg defects:
//!
//! * 8. `qr_solve` used an ABSOLUTE rank threshold (`|R[i,i]| < 1e-14`). R's
//!    diagonal scales with the input, so a perfectly well-conditioned system
//!    scaled down by 1e-14 was wrongly declared rank-deficient.
//! * 9. `cond` used an ABSOLUTE singularity cutoff (`min_sv < 1e-15`), so
//!    `diag(1, 1e-16)` reported INFINITY where the true condition number is
//!    1e16 - a value f64 represents exactly.
//!
//! Each test carries a CONTROL assertion.

use cv_math::linalg::{cond, qr_solve};
use nalgebra::{DMatrix, DVector};

fn rel_close(got: f64, want: f64, tol: f64) -> bool {
    (got / want - 1.0).abs() <= tol
}

// ---------------------------------------------------------------------------
// Defect 8: qr_solve relative rank threshold
// ---------------------------------------------------------------------------

#[test]
fn qr_solve_scaled_well_conditioned_system() {
    // A = [[2,-1],[-1,2]] * 1e-14, b = [1,2] * 1e-14.
    // The exact solution is x = [4/3, 5/3] and the conditioning is unchanged
    // by the scaling. Pre-fix: Err("QR solve failed: rank-deficient matrix")
    // because |R[0,0]| = 2e-14 ... and R[1,1] = 1e-14 both sat under the
    // absolute 1e-14 cutoff.
    let s = 1e-14;
    let a = DMatrix::from_row_slice(2, 2, &[2.0 * s, -s, -s, 2.0 * s]);
    let b = DVector::from_column_slice(&[s, 2.0 * s]);
    let x = qr_solve(&a, &b).unwrap_or_else(|e| {
        panic!("qr_solve on a well-conditioned system scaled by 1e-14 failed: {e}")
    });
    assert!((x[0] - 4.0 / 3.0).abs() < 1e-9, "x[0] = {}, want 4/3", x[0]);
    assert!((x[1] - 5.0 / 3.0).abs() < 1e-9, "x[1] = {}, want 5/3", x[1]);

    // Same at 1e-15 and 1e-16 - still perfectly solvable.
    for s in [1e-15_f64, 1e-16, 1e-20] {
        let a = DMatrix::from_row_slice(2, 2, &[2.0 * s, -s, -s, 2.0 * s]);
        let b = DVector::from_column_slice(&[s, 2.0 * s]);
        let x = qr_solve(&a, &b).unwrap_or_else(|e| panic!("s = {s:e}: {e}"));
        assert!(
            (x[0] - 4.0 / 3.0).abs() < 1e-9,
            "s = {s:e}: x[0] = {}",
            x[0]
        );
        assert!(
            (x[1] - 5.0 / 3.0).abs() < 1e-9,
            "s = {s:e}: x[1] = {}",
            x[1]
        );
    }

    // Upscaling in the other direction must keep working too.
    let s = 1e14_f64;
    let a = DMatrix::from_row_slice(2, 2, &[2.0 * s, -s, -s, 2.0 * s]);
    let b = DVector::from_column_slice(&[s, 2.0 * s]);
    let x = qr_solve(&a, &b).unwrap_or_else(|e| panic!("s = {s:e}: {e}"));
    assert!((x[0] - 4.0 / 3.0).abs() < 1e-9);

    // CONTROL: the unscaled system, and a genuinely rank-deficient one.
    let a = DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
    let b = DVector::from_column_slice(&[1.0, 2.0]);
    let x = qr_solve(&a, &b).unwrap();
    assert!((x[0] - 4.0 / 3.0).abs() < 1e-12);
    assert!((x[1] - 5.0 / 3.0).abs() < 1e-12);

    let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let b = DVector::from_column_slice(&[1.0, 2.0]);
    assert!(
        qr_solve(&singular, &b).is_err(),
        "CONTROL: a truly rank-deficient matrix must still be rejected"
    );
}

// ---------------------------------------------------------------------------
// Defect 9: cond relative singularity cutoff
// ---------------------------------------------------------------------------

#[test]
fn cond_is_relative_to_matrix_scale() {
    // diag(1, 1e-16): true condition number 1e16, representable exactly in f64.
    // Pre-fix: INFINITY, because min_sv = 1e-16 < 1e-15 in absolute terms.
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 1e-16]));
    let c = cond(&a);
    assert!(
        c.is_finite(),
        "cond(diag(1,1e-16)) = {c}, want a finite 1e16"
    );
    assert!(
        rel_close(c, 1e16, 1e-9),
        "cond(diag(1,1e-16)) = {c}, want 1e16"
    );

    // The same matrix scaled up by 1e8 has the SAME condition number, and
    // pre-fix it was also reported as INFINITY.
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e8, 1e-8]));
    let c = cond(&a);
    assert!(
        c.is_finite(),
        "cond(1e8*diag(1,1e-16)) = {c}, want finite 1e16"
    );
    assert!(rel_close(c, 1e16, 1e-9));

    // And scaled down by 1e-8.
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e-8, 1e-24]));
    let c = cond(&a);
    assert!(
        c.is_finite(),
        "cond(1e-8*diag(1,1e-16)) = {c}, want finite 1e16"
    );
    assert!(rel_close(c, 1e16, 1e-6));

    // CONTROL: well-conditioned matrices give exactly the same answer as before.
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 2.0]));
    assert!(rel_close(cond(&a), 2.0, 1e-12));
    let a = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 3.0]);
    // singular values 4 and 2
    assert!(rel_close(cond(&a), 2.0, 1e-12));
    // A large-norm but perfectly conditioned matrix.
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e10, 2e10]));
    assert!(rel_close(cond(&a), 2.0, 1e-12));

    // CONTROL: truly singular matrices must still report INFINITY.
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(cond(&a).is_infinite(), "rank-1 matrix must be INFINITY");
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(cond(&a).is_infinite(), "duplicate rows must be INFINITY");
    assert!(cond(&DMatrix::zeros(3, 3)).is_infinite());
}
