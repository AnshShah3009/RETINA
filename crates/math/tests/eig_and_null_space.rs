//! `eig` must return eigenvectors, and `null_space` must see a wide matrix's
//! whole null space.
//!
//! `eig` used to return the real-Schur vectors `Q`, which are eigenvectors
//! only for normal matrices: for `[[1, 1], [0, 2]]` the column for `λ = 2`
//! was `[0, 1]` with residual 1. `null_space` read a thin SVD, so for an
//! `m x n` matrix with `m < n` the `n - m` directions without a singular value
//! were never returned: `null_space([[1, 0, 0]])` was 3x0.

use cv_math::linalg::{eig, null_space};
use nalgebra::{Complex, DMatrix, DVector};

fn residual(a: &DMatrix<f64>, vals: &[(f64, f64)], vecs: &DMatrix<f64>) -> f64 {
    let n = a.nrows();
    let ac = a.map(|x| Complex::new(x, 0.0));
    let mut worst = 0.0f64;
    let mut j = 0;
    while j < n {
        let (re, im) = vals[j];
        let v: DVector<Complex<f64>> = if im != 0.0 {
            DVector::from_fn(n, |k, _| Complex::new(vecs[(k, j)], vecs[(k, j + 1)]))
        } else {
            DVector::from_fn(n, |k, _| Complex::new(vecs[(k, j)], 0.0))
        };
        assert!((v.norm() - 1.0).abs() < 1e-9, "eigenvector {j} not unit");
        let r = (&ac * &v - &v * Complex::new(re, im)).norm();
        worst = worst.max(r);
        j += if im != 0.0 { 2 } else { 1 };
    }
    worst
}

#[test]
fn eig_returns_eigenvectors_of_a_non_normal_matrix() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 2.0]);
    let (vals, vecs) = eig(&a).unwrap();
    assert!(residual(&a, &vals, &vecs) < 1e-10);
}

#[test]
fn eig_packs_complex_pairs_like_dgeev() {
    // Rotation-and-scale block plus a real eigenvalue; not normal.
    let a = DMatrix::from_row_slice(3, 3, &[0.0, -2.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 5.0]);
    let (vals, vecs) = eig(&a).unwrap();
    assert!(vals.iter().any(|v| v.1 != 0.0), "expected a complex pair");
    assert!(residual(&a, &vals, &vecs) < 1e-9);
}

#[test]
fn eig_gives_independent_vectors_for_a_repeated_eigenvalue() {
    let a = DMatrix::<f64>::identity(3, 3) * 2.0;
    let (vals, vecs) = eig(&a).unwrap();
    assert!(residual(&a, &vals, &vecs) < 1e-12);
    assert!(
        vecs.determinant().abs() > 0.5,
        "repeated eigenvalue got dependent vectors"
    );
}

#[test]
fn null_space_of_a_wide_matrix_has_full_dimension() {
    let a = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
    let ns = null_space(&a, 1e-10);
    assert_eq!(ns.shape(), (3, 2));
    assert!((&a * &ns).norm() < 1e-12);
    assert!((ns.transpose() * &ns - DMatrix::identity(2, 2)).norm() < 1e-12);

    let b = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);
    let ns = null_space(&b, 1e-10);
    assert_eq!(ns.ncols(), 3, "rank-1 2x4 has a 3-dimensional null space");
    assert!((&b * &ns).norm() < 1e-10);
}
