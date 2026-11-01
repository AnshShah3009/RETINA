//! Regressions for confirmed `cv-math` defects. Each test fails on the
//! pre-fix behavior and keeps a control for the case that was already correct.

use cv_math::Interp1d;
use cv_math::integrate::{simpson, trapezoid};
use cv_math::interpolate::interp1d_linear;
use cv_math::linalg::{eig, lu_solve, qr_solve};
use cv_math::polygon_iou;
use cv_math::sparse::{CsrMatrix, eigs_inverse_power};
use cv_math::spatial::{DistanceMetric, KDTree, cdist, pdist};
use cv_math::special::{bessel_y0, bessel_y1, bessel_yn};
use geo::{LineString, Polygon};
use nalgebra::{DMatrix, DVector};

fn square(x: f64, y: f64, size: f64) -> Polygon<f64> {
    Polygon::new(
        LineString::from(vec![
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y),
        ]),
        vec![],
    )
}

fn eigen_residual(a: &DMatrix<f64>, vals: &[(f64, f64)], vecs: &DMatrix<f64>) -> f64 {
    use nalgebra::Complex;
    let n = a.nrows();
    let ac = a.map(|x| Complex::new(x, 0.0));
    let mut worst = 0.0f64;
    let mut j = 0;
    while j < n {
        let (re, im) = vals[j];
        let v = if im != 0.0 && j + 1 < n {
            DVector::from_fn(n, |k, _| Complex::new(vecs[(k, j)], vecs[(k, j + 1)]))
        } else {
            DVector::from_fn(n, |k, _| Complex::new(vecs[(k, j)], 0.0))
        };
        worst = worst.max((&ac * &v - &v * Complex::new(re, im)).norm());
        j += if im != 0.0 && j + 1 < n { 2 } else { 1 };
    }
    worst
}

// ---------------------------------------------------------------------------
// 1. cdist / pdist must not zip-truncate unequal coordinates.
//    cdist([[0, 0, 5]], [[0, 0]]) was 0; that dropped axis has length 5.
// ---------------------------------------------------------------------------

fn panic_payload(payload: &Box<dyn std::any::Any + Send>) -> String {
    // The argument must stay a `&Box`. Coercing it to `&(dyn Any + Send)`
    // wraps the box itself, and `downcast_ref::<String>()` then misses the
    // assertion message (`debug_assert_eq` formats one).
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "non-string panic".to_string()
    }
}

fn mismatch_is_not_a_truncated_zero(got: f64, what: &str) {
    assert!(
        got.is_nan(),
        "{what} = {got}; zip-dropping the trailing 5 returned 0, \
         but a distance is only defined when the dimensions match"
    );
    assert_ne!(got, 0.0);
}

#[test]
fn cdist_does_not_drop_a_coordinate_of_length_five() {
    // Debug builds also `debug_assert` the equal-length precondition. That
    // refusal is not the old answer 0. Release builds return NaN.
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cdist(
            &[vec![0.0, 0.0, 5.0]],
            &[vec![0.0, 0.0]],
            DistanceMetric::Euclidean,
        )
    }));
    match ran {
        Ok(d) => mismatch_is_not_a_truncated_zero(d[(0, 0)], "cdist([[0, 0, 5]], [[0, 0]])"),
        Err(payload) => {
            let msg = panic_payload(&payload);
            assert!(
                msg.contains("refusing to drop") || msg.contains("lengths"),
                "panic was not the dimension check: {msg}"
            );
        }
    }

    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdist(
            &[vec![0.0, 0.0, 5.0], vec![0.0, 0.0]],
            DistanceMetric::Euclidean,
        )
    }));
    match ran {
        Ok(condensed) => mismatch_is_not_a_truncated_zero(condensed[0], "pdist"),
        Err(payload) => {
            let msg = panic_payload(&payload);
            assert!(
                msg.contains("refusing to drop") || msg.contains("lengths"),
                "panic was not the dimension check: {msg}"
            );
        }
    }

    // Equal lengths are unchanged, including a 3-vector whose last coordinate
    // really is a distance of 5 from the origin.
    let same = cdist(
        &[vec![0.0, 0.0, 5.0]],
        &[vec![0.0, 0.0, 0.0]],
        DistanceMetric::Euclidean,
    );
    assert!(
        (same[(0, 0)] - 5.0).abs() < 1e-12,
        "equal-dimension distance = {}",
        same[(0, 0)]
    );
    let plane = cdist(
        &[vec![0.0, 0.0], vec![1.0, 0.0]],
        &[vec![0.0, 1.0]],
        DistanceMetric::Euclidean,
    );
    assert!((plane[(0, 0)] - 1.0).abs() < 1e-12);
    assert!((plane[(1, 0)] - 2.0_f64.sqrt()).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 2. A NaN coordinate must not enter the k-NN heap.
// ---------------------------------------------------------------------------

#[test]
fn kdtree_query_skips_non_finite_and_returns_finite_neighbor() {
    let points = vec![vec![f64::NAN, 0.0], vec![0.0, 0.0], vec![4.0, 0.0]];
    let tree = KDTree::new(&points).unwrap();
    let neighbors = tree.query(&[0.2, 0.0], 1);
    assert!(!neighbors.is_empty(), "finite query returned no neighbor");
    assert_eq!(
        neighbors[0].0, 1,
        "nearest index {}, want the finite origin (NaN point must not win)",
        neighbors[0].0
    );
    assert!(
        neighbors[0].1.is_finite(),
        "distance {} is not finite",
        neighbors[0].1
    );
    assert!((neighbors[0].1 - 0.2).abs() < 1e-12);

    // k > 1 still lists only finite distances, closest first.
    let neighbors = tree.query(&[0.2, 0.0], 3);
    assert!(neighbors.iter().all(|(_, d)| d.is_finite()));
    assert_eq!(neighbors[0].0, 1);

    // Finite queries are unchanged.
    let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let tree = KDTree::new(&points).unwrap();
    let neighbors = tree.query(&[0.1, 0.1], 2);
    assert_eq!(neighbors[0].0, 0);
    assert!(neighbors[0].1.is_finite());
    assert!(neighbors[1].1.is_finite());
    assert!(neighbors[0].1 < neighbors[1].1);
}

// ---------------------------------------------------------------------------
// 3. polygon IoU of two tiny identical squares.
// ---------------------------------------------------------------------------

#[test]
fn polygon_iou_of_tiny_identical_squares_is_one() {
    let tiny = 1e-5;
    let a = square(0.0, 0.0, tiny);
    let b = square(0.0, 0.0, tiny);
    let iou = polygon_iou(&a, &b);
    assert!(
        (iou - 1.0).abs() < 1e-9,
        "IoU of identical squares of side 1e-5 = {iou}, want 1 (intersection area 1e-10 was clamped to 0)"
    );

    // Side 1 is unchanged.
    let a = square(0.0, 0.0, 1.0);
    let b = square(0.0, 0.0, 1.0);
    let iou = polygon_iou(&a, &b);
    assert!((iou - 1.0).abs() < 1e-12, "side-1 IoU = {iou}");

    // A genuinely empty intersection stays 0, at both scales.
    let disjoint = polygon_iou(&square(0.0, 0.0, 1.0), &square(5.0, 5.0, 1.0));
    assert_eq!(disjoint, 0.0);
    let disjoint_tiny = polygon_iou(&square(0.0, 0.0, tiny), &square(1.0, 1.0, tiny));
    assert_eq!(disjoint_tiny, 0.0);
}

// ---------------------------------------------------------------------------
// 4. Bessel Y is NaN for negative arguments; the positive branch is unchanged.
// ---------------------------------------------------------------------------

#[test]
fn bessel_y0_of_negative_is_nan_and_y0_of_one_is_unchanged() {
    assert!(
        bessel_y0(-1.0).is_nan(),
        "bessel_y0(-1) = {}, want NaN (SciPy y0(-1) is NaN)",
        bessel_y0(-1.0)
    );
    assert!(
        bessel_y1(-1.0).is_nan(),
        "bessel_y1(-1) = {}, want NaN",
        bessel_y1(-1.0)
    );

    // x == 0 stays the singularity.
    assert!(bessel_y0(0.0).is_infinite() && bessel_y0(0.0).is_sign_negative());
    assert!(bessel_y1(0.0).is_infinite() && bessel_y1(0.0).is_sign_negative());

    let y0 = bessel_y0(1.0);
    assert!(y0.is_finite(), "bessel_y0(1) = {y0}");
    // Numerical Recipes value at x = 1 (the log term vanishes). SciPy is
    // 0.08825696421567697; the fit must stay on the value the positive branch
    // already produced, which the yn(2, 1) regression also pins.
    assert!(
        (y0 - 0.08825696421567697).abs() < 1e-6,
        "bessel_y0(1) = {y0}"
    );
    let y1 = bessel_y1(1.0);
    assert!(y1.is_finite(), "bessel_y1(1) = {y1}");
    assert!(
        (bessel_yn(2, 1.0) - -1.6506826068162554).abs() < 1e-7,
        "bessel_yn(2, 1) moved: {}",
        bessel_yn(2, 1.0)
    );
}

// ---------------------------------------------------------------------------
// 5. Inverse power iteration must not report success when the budget ends
//    with a large residual. max_iters = 0 used to return λ = 0.
// ---------------------------------------------------------------------------

#[test]
fn eigs_inverse_power_errors_when_the_budget_does_not_converge() {
    let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 4.0), (1, 1, 9.0)]);
    let exhausted = eigs_inverse_power(&a, 0.0, 0, 1e-8);
    assert!(
        exhausted.is_err(),
        "max_iters = 0 returned Ok({})",
        exhausted.unwrap().0
    );

    let one = eigs_inverse_power(&a, 0.0, 1, 1e-8);
    assert!(
        one.is_err(),
        "max_iters = 1 returned Ok({}) (about 4.825, not an eigenvalue)",
        one.unwrap().0
    );

    // A converged call still returns the eigenvalue closest to the shift.
    let (val, vec) = eigs_inverse_power(&a, 0.0, 50, 1e-10)
        .unwrap_or_else(|e| panic!("converged inverse iteration failed: {e}"));
    assert!((val - 4.0).abs() < 1e-6, "converged λ = {val}, want 4");
    let residual = (&a.spmv(&vec) - &vec * val).norm();
    assert!(residual < 1e-6, "residual {residual}");
}

// ---------------------------------------------------------------------------
// 6. Interp1d must use the real linear blend on tiny knot spacing.
// ---------------------------------------------------------------------------

#[test]
fn interp1d_tiny_spacing_uses_the_real_blend() {
    let ip = Interp1d::new(vec![0.0, 1e-12], vec![0.0, 100.0]).unwrap();
    let y = ip.call(1e-15);
    assert!(
        (y - 0.1).abs() < 1e-9,
        "Interp1d(1e-15) = {y}, want 0.1 (the 1e-10 cutoff forced t = 0.5 and returned 50)"
    );
    let reference = interp1d_linear(&[0.0, 1e-12], &[0.0, 100.0], &[1e-15]).unwrap();
    assert!((y - reference[0]).abs() < 1e-12);

    // Ordinary spacing is unchanged.
    let ip = Interp1d::new(vec![0.0, 2.0], vec![0.0, 100.0]).unwrap();
    assert!((ip.call(0.5) - 25.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 7. A Jordan block is defective: do not return Ok with residual 1.
// ---------------------------------------------------------------------------

#[test]
fn eig_rejects_a_jordan_block_with_a_large_residual() {
    let jordan = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
    if let Ok((vals, vecs)) = eig(&jordan) {
        let residual = eigen_residual(&jordan, &vals, &vecs);
        panic!(
            "Jordan block returned Ok with residual {residual} (the second vector's residual is 1)"
        );
    }

    let diagonal = DMatrix::from_diagonal(&DVector::from_column_slice(&[3.0, 7.0]));
    let (vals, vecs) = eig(&diagonal).unwrap_or_else(|e| panic!("diagonal eig failed: {e}"));
    let residual = eigen_residual(&diagonal, &vals, &vecs);
    assert!(residual < 1e-8, "diagonal residual {residual}");
    let mut reals: Vec<f64> = vals.iter().map(|v| v.0).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((reals[0] - 3.0).abs() < 1e-8);
    assert!((reals[1] - 7.0).abs() < 1e-8);
}

// ---------------------------------------------------------------------------
// 8. qr_solve must accept diag(1e16, 1), and still reject a zero pivot.
// ---------------------------------------------------------------------------

#[test]
fn qr_solve_accepts_a_large_but_nonzero_pivot() {
    let a = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e16, 1.0]));
    let b = DVector::from_column_slice(&[1e16, 3.0]);
    let x = qr_solve(&a, &b).unwrap_or_else(|e| panic!("qr_solve refused a solvable system: {e}"));
    assert!((x[0] - 1.0).abs() < 1e-6, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-6, "x[1] = {}", x[1]);

    let lu = lu_solve(&a, &b).unwrap();
    assert!((x[0] - lu[0]).abs() < 1e-6);
    assert!((x[1] - lu[1]).abs() < 1e-6);

    // A truly zero pivot is still rank-deficient.
    let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(
        qr_solve(&singular, &DVector::from_column_slice(&[1.0, 2.0])).is_err(),
        "exactly singular matrix must still be rejected"
    );
    let zero_pivot = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e16, 0.0]));
    assert!(
        qr_solve(&zero_pivot, &DVector::from_column_slice(&[1.0, 1.0])).is_err(),
        "an exactly zero pivot must still be rejected"
    );
}

// ---------------------------------------------------------------------------
// 9. n = 0 must not divide by zero. The integral of x on [0, 1] is 1/2.
// ---------------------------------------------------------------------------

#[test]
fn trapezoid_and_simpson_with_zero_panels_integrate() {
    let trap = trapezoid(|x| x, 0.0, 1.0, 0);
    assert!(
        (trap - 0.5).abs() < 1e-12,
        "trapezoid(n=0) = {trap}, want 0.5"
    );
    let simp = simpson(|x| x, 0.0, 1.0, 0);
    assert!(
        (simp - 0.5).abs() < 1e-12,
        "simpson(n=0) = {simp}, want 0.5"
    );

    // n = 1 stays valid (Simpson already promotes an odd count to even).
    let trap = trapezoid(|x| x, 0.0, 1.0, 1);
    assert!((trap - 0.5).abs() < 1e-12, "trapezoid(n=1) = {trap}");
    let simp = simpson(|x| x, 0.0, 1.0, 1);
    assert!((simp - 0.5).abs() < 1e-12, "simpson(n=1) = {simp}");

    // A previously-valid panel count is unchanged in exactness for a linear
    // integrand, and Simpson on a cubic stays exact.
    let trap = trapezoid(|x| x, 0.0, 1.0, 8);
    assert!((trap - 0.5).abs() < 1e-12);
    let simp = simpson(|x| x * x * x, 0.0, 1.0, 100);
    assert!((simp - 0.25).abs() < 1e-12, "simpson cubic = {simp}");
}
