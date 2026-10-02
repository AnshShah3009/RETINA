//! Regression tests for the sparse eigensolver defects:
//!
//! * 5. `eigs_power` used an all-ones start vector. That vector is an EXACT
//!    eigenvector of the discrete-Laplacian / Poisson SPD matrices the routine
//!    exists to solve, so the iteration was a fixed point and it returned an
//!    arbitrary eigenvalue (1 instead of 3 for [[2,-1],[-1,2]]) while
//!    reporting success.
//! * 6. `eigsh` had the same start vector, so `A*q0` was proportional to `q0`,
//!    `b_0 = 0`, the loop broke after j = 0 and it returned ONE eigenvalue as
//!    `Ok` when `k = 2` was requested.
//!
//! Each test carries a CONTROL assertion.

use cv_math::sparse::{eigs_power, eigsh, CsrMatrix, EigWhich};

fn laplacian2() -> CsrMatrix {
    // [[2,-1],[-1,2]]: eigenvalues 1 and 3. Its all-ones vector is an
    // eigenvector with eigenvalue 1 - the degenerate case.
    CsrMatrix::from_triplets(
        2,
        2,
        &[(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)],
    )
}

fn laplacian4() -> CsrMatrix {
    // 1D Dirichlet Laplacian on 4 nodes: diag 2, off-diag -1.
    CsrMatrix::from_triplets(
        4,
        4,
        &[
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 2.0),
        ],
    )
}

#[test]
fn eigs_power_finds_dominant_eigenvalue_of_laplacian() {
    // Pre-fix: Ok((1.0000000000000002, ...)) - the SMALLEST eigenvalue,
    // reported as success, because the all-ones start vector is exactly the
    // eigenvector for eigenvalue 1 and the iteration never moved.
    // The power method converges at the rate (l2/l1)^(2k) = (1/3)^(2k), so
    // tol = 1e-10 is only reached after ~18 iterations and the eigenvector is
    // accurate to ~1e-9, not 1e-15.
    let a = laplacian2();
    let (val, vec) = eigs_power(&a, 1000, 1e-10)
        .unwrap_or_else(|e| panic!("eigs_power on a 2x2 Laplacian failed: {e}"));
    assert!(
        (val - 3.0).abs() < 1e-8,
        "eigs_power([[2,-1],[-1,2]]) = {val}, want the dominant eigenvalue 3.0 (not 1.0)"
    );
    // The eigenvector must be aligned with [1,-1]/sqrt(2).
    assert!(
        vec[0] * vec[1] < 0.0,
        "dominant eigenvector [{}, {}] should be anti-symmetric",
        vec[0],
        vec[1]
    );
    let residual = (&a.spmv(&vec) - &vec * val).norm();
    assert!(residual < 1e-7, "residual {residual} is too large");

    // Same for the 4-node Laplacian: eigenvalues are 2 - 2*cos(k*pi/5),
    // k = 1..4, i.e. 0.381966, 1.381966, 2.618034, 3.618034.
    let a = laplacian4();
    let (val, vec) = eigs_power(&a, 2000, 1e-10)
        .unwrap_or_else(|e| panic!("eigs_power on a 4x4 Laplacian failed: {e}"));
    let dominant = 2.0 - 2.0 * (4.0 * std::f64::consts::PI / 5.0).cos();
    assert!(
        (val - dominant).abs() < 1e-8,
        "eigs_power(4-node Laplacian) = {val}, want {dominant}"
    );
    let rel_res = (&a.spmv(&vec) - &vec * val).norm() / val.abs();
    assert!(rel_res < 1e-8, "relative residual {rel_res} is too large");

    // CONTROL: a matrix whose all-ones start is not an eigenvector still
    // converges to the dominant eigenvalue.
    let d = CsrMatrix::from_triplets(3, 3, &[(0, 0, 1.0), (1, 1, 5.0), (2, 2, 10.0)]);
    let (val, vec) = eigs_power(&d, 1000, 1e-10)
        .unwrap_or_else(|e| panic!("CONTROL: eigs_power on a diagonal matrix failed: {e}"));
    assert!((val - 10.0).abs() < 1e-6, "CONTROL val = {val}, want 10");
    assert!(
        vec[2].abs() > 0.99,
        "CONTROL: eigenvector should align with e3"
    );
}

#[test]
fn eigsh_returns_k_eigenvalues_of_laplacian() {
    // Pre-fix: Ok([0.9999999999999998]) — ONE value, reported as success,
    // when k = 2 was requested (b_0 = 0, loop broke at j = 0).
    let a = laplacian2();
    let (vals, vecs) = eigsh(&a, 2, EigWhich::Largest, 50, 1e-10)
        .unwrap_or_else(|e| panic!("eigsh(k=2) on a 2x2 Laplacian failed: {e}"));
    assert_eq!(vals.len(), 2, "eigsh must return exactly k = 2 eigenvalues");
    assert!(
        (vals[0] - 3.0).abs() < 1e-8,
        "vals[0] = {}, want 3.0",
        vals[0]
    );
    assert!(
        (vals[1] - 1.0).abs() < 1e-8,
        "vals[1] = {}, want 1.0",
        vals[1]
    );
    assert_eq!(vecs.ncols(), 2);
    for c in 0..2 {
        assert!((vecs.column(c).norm() - 1.0).abs() < 1e-10);
        let v = vecs.column(c).clone_owned();
        let r = (&a.spmv(&v) - &v * vals[c]).norm();
        assert!(r < 1e-8, "Ritz residual {r} for column {c} is too large");
    }

    // Smallest-first ordering on the 4-node Laplacian.
    let a = laplacian4();
    let (vals, _) = eigsh(&a, 2, EigWhich::Smallest, 100, 1e-10)
        .unwrap_or_else(|e| panic!("eigsh on a 4x4 Laplacian failed: {e}"));
    assert_eq!(vals.len(), 2);
    let exact = [
        2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos(),
        2.0 - 2.0 * (2.0 * std::f64::consts::PI / 5.0).cos(),
    ];
    assert!(
        (vals[0] - exact[0]).abs() < 1e-8,
        "vals[0] = {}, want {}",
        vals[0],
        exact[0]
    );
    assert!(
        (vals[1] - exact[1]).abs() < 1e-8,
        "vals[1] = {}, want {}",
        vals[1],
        exact[1]
    );

    // CONTROL: a diagonal matrix with a non-degenerate start — all three
    // request shapes already worked and must keep working.
    let t: Vec<_> = (0..5).map(|i| (i, i, (i + 1) as f64)).collect();
    let a = CsrMatrix::from_triplets(5, 5, &t);
    let (vals, _) = eigsh(&a, 3, EigWhich::Largest, 100, 1e-10).unwrap();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 5.0).abs() < 1e-6, "vals[0] = {}", vals[0]);
    assert!((vals[1] - 4.0).abs() < 1e-6, "vals[1] = {}", vals[1]);
    assert!((vals[2] - 3.0).abs() < 1e-6, "vals[2] = {}", vals[2]);

    let (vals, _) = eigsh(&a, 2, EigWhich::Smallest, 100, 1e-10).unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6, "vals[0] = {}", vals[0]);
    assert!((vals[1] - 2.0).abs() < 1e-6, "vals[1] = {}", vals[1]);

    let (vals, _) = eigsh(&a, 2, EigWhich::NearSigma(6.0), 100, 1e-10).unwrap();
    let mut sorted: Vec<f64> = vals.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Eigenvalues are 1..5, so the two nearest 6.0 are 5 and 4 (ascending).
    assert!((sorted[0] - 4.0).abs() < 1e-6, "sorted[0] = {}", sorted[0]);
    assert!((sorted[1] - 5.0).abs() < 1e-6, "sorted[1] = {}", sorted[1]);
}
