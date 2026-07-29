//! General-purpose linear algebra module.
//!
//! Provides a unified, friendly interface over nalgebra's decompositions
//! plus additional utilities like pseudo-inverse, null space, condition number,
//! and a basic compressed sparse row (CSR) matrix type.
//!
//! # Example
//!
//! ```rust
//! use nalgebra::{DMatrix, DVector};
//! use cv_scientific::linalg;
//!
//! let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 5.0, 3.0]);
//! let b = DVector::from_column_slice(&[4.0, 7.0]);
//! let x = linalg::solve(&a, &b).unwrap();
//! assert!((x[0] - 5.0).abs() < 1e-10);
//! assert!((x[1] + 6.0).abs() < 1e-10);
//! ```

use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Matrix decompositions
// ---------------------------------------------------------------------------

/// LU decomposition with partial pivoting.
///
/// Returns `(L, U, pivot_indices)` where `P * A = L * U`.
/// `pivot_indices[i]` is the row that row `i` was swapped with during
/// factorization.
#[allow(clippy::type_complexity)]
pub fn lu_decompose(a: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>, Vec<usize>), String> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err("Matrix must be non-empty".into());
    }

    let lu = a.clone().lu();

    // Extract L and U via the LU struct.
    // nalgebra LU gives us P*A = L*U. We reconstruct P by comparing L*U rows to A rows.
    let l = lu.l();
    let u = lu.u();

    // Recover pivot indices: row i of P*A = row pivots[i] of A.
    // L*U = P*A, so row i of L*U equals row pivots[i] of A.
    let lu_product = &l * &u;
    let mut pivots = vec![0usize; m];
    for i in 0..m {
        for j in 0..m {
            let diff: f64 = (0..n).map(|k| (lu_product[(i, k)] - a[(j, k)]).abs()).sum();
            if diff < 1e-10 {
                pivots[i] = j;
                break;
            }
        }
    }

    Ok((l, u, pivots))
}

/// Solve `A x = b` using LU decomposition.
pub fn lu_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }
    if m != b.nrows() {
        return Err("Dimension mismatch between A and b".into());
    }

    let lu = a.clone().lu();
    lu.solve(b)
        .ok_or_else(|| "LU solve failed (singular matrix)".into())
}

/// QR decomposition via Householder reflections.
///
/// Returns `(Q, R)` where `A = Q * R`, Q is orthogonal and R is upper
/// triangular.
pub fn qr_decompose(a: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err("Matrix must be non-empty".into());
    }

    let qr = a.clone().qr();
    let q = qr.q();
    let r = qr.r();
    Ok((q, r))
}

/// Solve the least-squares problem min ||Ax - b||^2 via QR decomposition.
pub fn qr_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, String> {
    if a.nrows() != b.nrows() {
        return Err("Dimension mismatch between A and b".into());
    }

    let (q, r) = qr_decompose(a)?;
    let qt_b = q.transpose() * b;

    // Back-substitute: R x = Q^T b (use only the first n rows)
    let n = a.ncols();
    let r_top = r.rows(0, n).clone_owned();
    let qt_b_top = qt_b.rows(0, n).clone_owned();

    // Check for zero diagonal (rank-deficient)
    for i in 0..n {
        if r_top[(i, i)].abs() < 1e-14 {
            return Err("QR solve failed: rank-deficient matrix".into());
        }
    }

    // Back-substitution
    let mut x = DVector::zeros(n);
    for i in (0..n).rev() {
        let mut s = qt_b_top[i];
        for j in (i + 1)..n {
            s -= r_top[(i, j)] * x[j];
        }
        x[i] = s / r_top[(i, i)];
    }

    Ok(x)
}

/// Singular Value Decomposition.
///
/// Returns `(U, sigma, Vt)` where `A = U * diag(sigma) * Vt`.
#[allow(clippy::type_complexity)]
pub fn svd(a: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>), String> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err("Matrix must be non-empty".into());
    }

    let decomp = a.clone().svd(true, true);
    let u = decomp.u.ok_or("SVD failed to compute U")?;
    let vt = decomp.v_t.ok_or("SVD failed to compute V^T")?;
    Ok((u, decomp.singular_values, vt))
}

/// Eigendecomposition for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by ascending eigenvalue.
/// Each column of the eigenvector matrix is an eigenvector.
pub fn eigh(a: &DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>), String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }

    let eigen = a.clone().symmetric_eigen();
    let vals = eigen.eigenvalues;
    let vecs = eigen.eigenvectors;

    // Sort by ascending eigenvalue
    let mut indices: Vec<usize> = (0..vals.len()).collect();
    indices.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap());

    let sorted_vals = DVector::from_fn(vals.len(), |i, _| vals[indices[i]]);
    let sorted_vecs = DMatrix::from_fn(m, n, |r, c| vecs[(r, indices[c])]);

    Ok((sorted_vals, sorted_vecs))
}

/// General eigendecomposition (eigenvalues may be complex).
///
/// Returns `(eigenvalue_pairs, eigenvector_matrix)` where each eigenvalue pair
/// is `(real_part, imaginary_part)`. Uses the real Schur decomposition.
#[allow(clippy::type_complexity)]
pub fn eig(a: &DMatrix<f64>) -> Result<(Vec<(f64, f64)>, DMatrix<f64>), String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }

    let schur = a.clone().schur();
    let (q_schur, t) = schur.clone().unpack();

    // Extract eigenvalues from the quasi-upper-triangular T.
    // 1x1 diagonal blocks are real eigenvalues; 2x2 blocks give complex pairs.
    let mut eigenvalues = Vec::new();
    let mut i = 0;
    while i < m {
        if i + 1 < m && t[(i + 1, i)].abs() > 1e-14 {
            // 2x2 block
            let a11 = t[(i, i)];
            let a12 = t[(i, i + 1)];
            let a21 = t[(i + 1, i)];
            let a22 = t[(i + 1, i + 1)];
            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det;
            if disc < 0.0 {
                let real = trace / 2.0;
                let imag = (-disc).sqrt() / 2.0;
                eigenvalues.push((real, imag));
                eigenvalues.push((real, -imag));
            } else {
                let sqrt_disc = disc.sqrt();
                eigenvalues.push(((trace + sqrt_disc) / 2.0, 0.0));
                eigenvalues.push(((trace - sqrt_disc) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigenvalues.push((t[(i, i)], 0.0));
            i += 1;
        }
    }

    Ok((eigenvalues, q_schur))
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

/// Matrix determinant.
pub fn det(a: &DMatrix<f64>) -> f64 {
    a.clone().determinant()
}

/// Matrix inverse.
pub fn inv(a: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }

    a.clone()
        .try_inverse()
        .ok_or_else(|| "Matrix is singular".into())
}

/// Matrix rank via SVD.
///
/// Singular values smaller than `tol` are treated as zero.
pub fn rank(a: &DMatrix<f64>, tol: f64) -> usize {
    let decomp = a.clone().svd(false, false);
    decomp.singular_values.iter().filter(|&&s| s > tol).count()
}

/// Condition number (ratio of largest to smallest singular value).
///
/// Returns `f64::INFINITY` for singular matrices.
pub fn cond(a: &DMatrix<f64>) -> f64 {
    let decomp = a.clone().svd(false, false);
    let sv = &decomp.singular_values;
    if sv.is_empty() {
        return f64::NAN;
    }
    let max_sv = sv.iter().cloned().fold(0.0_f64, f64::max);
    let min_sv = sv.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_sv.abs() < 1e-15 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

/// Which matrix norm to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNorm {
    /// Frobenius norm: sqrt(sum of squared elements).
    Frobenius,
    /// Maximum absolute column sum.
    One,
    /// Maximum absolute row sum.
    Inf,
    /// Spectral norm: largest singular value.
    Spectral,
}

/// Compute a matrix norm.
pub fn norm(a: &DMatrix<f64>, norm_type: MatrixNorm) -> f64 {
    let (m, n) = a.shape();
    match norm_type {
        MatrixNorm::Frobenius => {
            let mut s = 0.0;
            for i in 0..m {
                for j in 0..n {
                    s += a[(i, j)] * a[(i, j)];
                }
            }
            s.sqrt()
        }
        MatrixNorm::One => {
            let mut max_col = 0.0_f64;
            for j in 0..n {
                let mut col_sum = 0.0;
                for i in 0..m {
                    col_sum += a[(i, j)].abs();
                }
                max_col = max_col.max(col_sum);
            }
            max_col
        }
        MatrixNorm::Inf => {
            let mut max_row = 0.0_f64;
            for i in 0..m {
                let mut row_sum = 0.0;
                for j in 0..n {
                    row_sum += a[(i, j)].abs();
                }
                max_row = max_row.max(row_sum);
            }
            max_row
        }
        MatrixNorm::Spectral => {
            let decomp = a.clone().svd(false, false);
            decomp
                .singular_values
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max)
        }
    }
}

/// Moore-Penrose pseudo-inverse via SVD.
///
/// Singular values smaller than `tol` are treated as zero.
pub fn pinv(a: &DMatrix<f64>, tol: f64) -> Result<DMatrix<f64>, String> {
    let (u, sigma, vt) = svd(a)?;
    let n_sv = sigma.len();

    // pinv(A) = V * Sigma_pinv * U^T
    // For A (m x n): U is (m x m), Vt is (n x n).
    // V = Vt^T is (n x n), U^T is (m x m).
    // Sigma_pinv must be (n x m) so product is (n x n) * (n x m) * (m x m) = (n x m).
    let v = vt.transpose();
    let ut = u.transpose();
    let rows_sp = v.ncols(); // n (to match V columns)
    let cols_sp = ut.nrows(); // m (to match U^T rows)
    let mut sigma_pinv = DMatrix::zeros(rows_sp, cols_sp);
    for i in 0..n_sv {
        if sigma[i] > tol {
            sigma_pinv[(i, i)] = 1.0 / sigma[i];
        }
    }

    Ok(v * sigma_pinv * ut)
}

/// Null space basis vectors.
///
/// Returns a matrix whose columns form an orthonormal basis for the null
/// space of `a`. Singular values <= `tol` are treated as zero.
pub fn null_space(a: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let decomp = a.clone().svd(false, true);
    let vt = decomp.v_t.unwrap();
    let sv = &decomp.singular_values;

    let mut null_cols = Vec::new();
    for i in 0..sv.len() {
        if sv[i] <= tol {
            null_cols.push(vt.row(i).transpose());
        }
    }
    // Also include rows of Vt beyond the number of singular values
    // (for wide matrices where n > m)
    for i in sv.len()..vt.nrows() {
        null_cols.push(vt.row(i).transpose());
    }

    if null_cols.is_empty() {
        DMatrix::zeros(a.ncols(), 0)
    } else {
        let n = a.ncols();
        let k = null_cols.len();
        let mut result = DMatrix::zeros(n, k);
        for (c, col) in null_cols.iter().enumerate() {
            for r in 0..n {
                result[(r, c)] = col[r];
            }
        }
        result
    }
}

/// Solve `A x = b` using the most appropriate method.
///
/// - Square matrix: LU decomposition
/// - Rectangular (over-determined): QR least squares
pub fn solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, String> {
    let (m, n) = a.shape();
    if m != b.nrows() {
        return Err("Dimension mismatch between A and b".into());
    }
    if m == n {
        lu_solve(a, b)
    } else {
        qr_solve(a, b)
    }
}

/// Solve `A X = B` (multiple right-hand sides).
pub fn solve_multi(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let (m, _n) = a.shape();
    if m != b.nrows() {
        return Err("Dimension mismatch between A and B".into());
    }

    let ncols_b = b.ncols();
    let mut result = DMatrix::zeros(a.ncols(), ncols_b);
    for j in 0..ncols_b {
        let col = b.column(j).clone_owned();
        let x = solve(a, &col)?;
        result.set_column(j, &x);
    }

    Ok(result)
}

/// Cholesky decomposition for positive-definite matrices.
///
/// Returns the lower-triangular factor `L` such that `A = L L^T`.
pub fn cholesky(a: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }

    let chol = nalgebra::linalg::Cholesky::new(a.clone())
        .ok_or("Cholesky failed (matrix not positive definite)")?;
    Ok(chol.l())
}

/// Solve `A x = b` using Cholesky decomposition (A must be positive-definite).
pub fn cholesky_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, String> {
    let (m, n) = a.shape();
    if m != n {
        return Err(format!("Matrix must be square, got {}x{}", m, n));
    }
    if m != b.nrows() {
        return Err("Dimension mismatch between A and b".into());
    }

    let chol = nalgebra::linalg::Cholesky::new(a.clone())
        .ok_or("Cholesky failed (matrix not positive definite)")?;
    Ok(chol.solve(b))
}

// ---------------------------------------------------------------------------
// Sparse matrix support
// ---------------------------------------------------------------------------

/// Compressed Sparse Row (CSR) matrix.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointer array (length `nrows + 1`).
    pub row_ptr: Vec<usize>,
    /// Column index for each non-zero entry.
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Build a CSR matrix from `(row, col, value)` triplets.
    ///
    /// Duplicate entries at the same position are summed.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        // Group by row
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nrows];
        for &(r, c, v) in triplets {
            assert!(r < nrows, "row index {} out of bounds (nrows={})", r, nrows);
            assert!(c < ncols, "col index {} out of bounds (ncols={})", c, ncols);
            rows[r].push((c, v));
        }

        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for row in &mut rows {
            // Sort by column, then merge duplicates
            row.sort_by_key(|&(c, _)| c);

            let mut prev_col: Option<usize> = None;
            for &(c, v) in row.iter() {
                if prev_col == Some(c) {
                    // Sum duplicate
                    *values.last_mut().unwrap() += v;
                } else {
                    col_idx.push(c);
                    values.push(v);
                    prev_col = Some(c);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Convert to a dense nalgebra matrix.
    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.nrows, self.ncols);
        for r in 0..self.nrows {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                m[(r, self.col_idx[idx])] += self.values[idx];
            }
        }
        m
    }

    /// Sparse matrix-vector multiply: `y = A * x`.
    pub fn spmv(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            x.nrows(),
            self.ncols,
            "Vector length {} does not match ncols {}",
            x.nrows(),
            self.ncols
        );

        let mut y = DVector::zeros(self.nrows);
        for r in 0..self.nrows {
            let mut sum = 0.0;
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[r] = sum;
        }
        y
    }

    /// Transpose the matrix (returns a new CSR matrix).
    pub fn transpose(&self) -> CsrMatrix {
        let mut triplets = Vec::with_capacity(self.values.len());
        for r in 0..self.nrows {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                triplets.push((self.col_idx[idx], r, self.values[idx]));
            }
        }
        CsrMatrix::from_triplets(self.ncols, self.nrows, &triplets)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn mat_approx_eq(a: &DMatrix<f64>, b: &DMatrix<f64>, eps: f64) -> bool {
        assert_eq!(a.shape(), b.shape());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if !approx_eq(a[(i, j)], b[(i, j)], eps) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_lu_decompose_and_solve() {
        // 3x3 system: A = [[2,1,1],[4,3,3],[8,7,9]], b = [1,1,1]
        let a = DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0]);
        let b = DVector::from_column_slice(&[1.0, 1.0, 1.0]);

        // Decompose
        let (l, u, pivots) = lu_decompose(&a).unwrap();
        assert_eq!(pivots.len(), 3);

        // L should be lower triangular (with ones on diagonal)
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    l[(i, j)].abs() < 1e-12,
                    "L[{},{}] = {} not zero",
                    i,
                    j,
                    l[(i, j)]
                );
            }
        }
        // U should be upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    u[(i, j)].abs() < 1e-12,
                    "U[{},{}] = {} not zero",
                    i,
                    j,
                    u[(i, j)]
                );
            }
        }

        // Solve
        let x = lu_solve(&a, &b).unwrap();
        let residual = &a * &x - &b;
        for i in 0..3 {
            assert!(
                residual[i].abs() < 1e-10,
                "residual[{}] = {}",
                i,
                residual[i]
            );
        }
    }

    #[test]
    fn test_qr_decompose() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (q, r) = qr_decompose(&a).unwrap();

        // Q^T Q should be identity (orthogonality)
        let qtq = q.transpose() * &q;
        let id = DMatrix::identity(q.ncols(), q.ncols());
        assert!(mat_approx_eq(&qtq, &id, 1e-10), "Q is not orthogonal");

        // R should be upper triangular
        for i in 0..r.nrows() {
            for j in 0..i.min(r.ncols()) {
                assert!(
                    r[(i, j)].abs() < 1e-12,
                    "R[{},{}] = {} not zero",
                    i,
                    j,
                    r[(i, j)]
                );
            }
        }

        // Q * R should reconstruct A
        let qr = &q * &r;
        assert!(mat_approx_eq(&qr, &a, 1e-10), "QR != A");
    }

    #[test]
    fn test_qr_solve_least_squares() {
        // Over-determined system: 3 equations, 2 unknowns
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_column_slice(&[1.0, 2.0, 2.0]);
        let x = qr_solve(&a, &b).unwrap();

        // Check normal equations: A^T A x = A^T b
        let ata = a.transpose() * &a;
        let atb = a.transpose() * &b;
        let residual = &ata * &x - &atb;
        for i in 0..2 {
            assert!(
                residual[i].abs() < 1e-10,
                "Normal eq residual[{}] = {}",
                i,
                residual[i]
            );
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, sigma, vt) = svd(&a).unwrap();

        // Reconstruct: A = U * diag(sigma) * Vt
        let min_dim = sigma.len();
        let mut sigma_mat = DMatrix::zeros(u.ncols(), vt.nrows());
        for i in 0..min_dim {
            sigma_mat[(i, i)] = sigma[i];
        }
        let reconstructed = &u * sigma_mat * &vt;
        assert!(
            mat_approx_eq(&reconstructed, &a, 1e-10),
            "SVD reconstruction failed"
        );
    }

    #[test]
    fn test_eigh_symmetric() {
        // Symmetric matrix with known eigenvalues: [[2, 1], [1, 2]]
        // eigenvalues: 1, 3
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (vals, vecs) = eigh(&a).unwrap();

        assert!(approx_eq(vals[0], 1.0, 1e-10), "eigenvalue 0 = {}", vals[0]);
        assert!(approx_eq(vals[1], 3.0, 1e-10), "eigenvalue 1 = {}", vals[1]);

        // Verify A * v = lambda * v for each eigenpair
        for i in 0..2 {
            let v = vecs.column(i).clone_owned();
            let av = &a * &v;
            let lv = &v * vals[i];
            for r in 0..2 {
                assert!(
                    approx_eq(av[r], lv[r], 1e-10),
                    "Eigenvector {} check failed",
                    i
                );
            }
        }
    }

    #[test]
    fn test_det_inv_rank() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        // det = 1*4 - 2*3 = -2
        assert!(approx_eq(det(&a), -2.0, 1e-10));

        // inv
        let a_inv = inv(&a).unwrap();
        let prod = &a * &a_inv;
        let id = DMatrix::identity(2, 2);
        assert!(mat_approx_eq(&prod, &id, 1e-10), "A * A^-1 != I");

        // rank = 2
        assert_eq!(rank(&a, 1e-10), 2);

        // Rank-deficient matrix
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(rank(&b, 1e-10), 1);
    }

    #[test]
    fn test_cond_and_norms() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);

        // cond = 2/1 = 2
        assert!(approx_eq(cond(&a), 2.0, 1e-10));

        // Frobenius = sqrt(1+4) = sqrt(5)
        assert!(approx_eq(
            norm(&a, MatrixNorm::Frobenius),
            5.0_f64.sqrt(),
            1e-10
        ));

        // 1-norm = max col sum = 2
        assert!(approx_eq(norm(&a, MatrixNorm::One), 2.0, 1e-10));

        // inf-norm = max row sum = 2
        assert!(approx_eq(norm(&a, MatrixNorm::Inf), 2.0, 1e-10));

        // spectral = largest singular value = 2
        assert!(approx_eq(norm(&a, MatrixNorm::Spectral), 2.0, 1e-10));
    }

    #[test]
    fn test_pinv_rectangular() {
        // 3x2 matrix
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let a_pinv = pinv(&a, 1e-10).unwrap();

        // A * pinv(A) * A should equal A
        let apa = &a * &a_pinv * &a;
        assert!(
            mat_approx_eq(&apa, &a, 1e-10),
            "pinv property A*pinv(A)*A = A failed"
        );

        // pinv(A) * A * pinv(A) should equal pinv(A)
        let pap = &a_pinv * &a * &a_pinv;
        assert!(
            mat_approx_eq(&pap, &a_pinv, 1e-10),
            "pinv property pinv(A)*A*pinv(A) = pinv(A) failed"
        );
    }

    #[test]
    fn test_cholesky_decompose_and_solve() {
        // Positive definite: [[4, 2], [2, 3]]
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let l = cholesky(&a).unwrap();

        // L should be lower triangular
        assert!(l[(0, 1)].abs() < 1e-12);

        // L * L^T should equal A
        let llt = &l * l.transpose();
        assert!(mat_approx_eq(&llt, &a, 1e-10), "L * L^T != A");

        // Solve
        let b = DVector::from_column_slice(&[1.0, 2.0]);
        let x = cholesky_solve(&a, &b).unwrap();
        let residual = &a * &x - &b;
        for i in 0..2 {
            assert!(
                residual[i].abs() < 1e-10,
                "Cholesky solve residual[{}] = {}",
                i,
                residual[i]
            );
        }
    }

    #[test]
    fn test_csr_matrix() {
        // Build a 3x3 sparse matrix: [[1,0,2],[0,3,0],[4,0,5]]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 5.0),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);

        // to_dense round-trip
        let dense = csr.to_dense();
        let expected =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]);
        assert!(mat_approx_eq(&dense, &expected, 1e-15));

        // spmv
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let y = csr.spmv(&x);
        // [1*1+0*2+2*3, 0*1+3*2+0*3, 4*1+0*2+5*3] = [7, 6, 19]
        assert!(approx_eq(y[0], 7.0, 1e-15));
        assert!(approx_eq(y[1], 6.0, 1e-15));
        assert!(approx_eq(y[2], 19.0, 1e-15));

        // transpose
        let csrt = csr.transpose();
        let dense_t = csrt.to_dense();
        assert!(mat_approx_eq(&dense_t, &expected.transpose(), 1e-15));
    }

    #[test]
    fn test_csr_duplicate_entries() {
        // Duplicate entries should be summed
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);
        let dense = csr.to_dense();
        let expected = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 3.0]);
        assert!(mat_approx_eq(&dense, &expected, 1e-15));
    }

    #[test]
    fn test_null_space() {
        // Rank-1 matrix: [[1,2],[2,4]] has 1D null space
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let ns = null_space(&a, 1e-10);
        assert_eq!(ns.ncols(), 1, "Expected 1D null space");

        // A * null_vector should be zero
        let nv = ns.column(0).clone_owned();
        let zero = &a * &nv;
        for i in 0..2 {
            assert!(zero[i].abs() < 1e-10, "A * null_vec != 0 at {}", i);
        }
    }

    #[test]
    fn test_solve_dispatch() {
        // Square: uses LU
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 5.0, 3.0]);
        let b = DVector::from_column_slice(&[4.0, 7.0]);
        let x = solve(&a, &b).unwrap();
        assert!(approx_eq(x[0], 5.0, 1e-10));
        assert!(approx_eq(x[1], -6.0, 1e-10));

        // Rectangular: uses QR
        let a2 = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let b2 = DVector::from_column_slice(&[1.0, 1.0, 2.0]);
        let x2 = solve(&a2, &b2).unwrap();
        // Check normal equations hold
        let residual = a2.transpose() * (&a2 * &x2 - &b2);
        for i in 0..2 {
            assert!(residual[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_eig_general() {
        // Real eigenvalues case: [[2, 1], [0, 3]] has eigenvalues 2 and 3
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 0.0, 3.0]);
        let (eigenvalues, _q) = eig(&a).unwrap();
        assert_eq!(eigenvalues.len(), 2);

        let mut reals: Vec<f64> = eigenvalues.iter().map(|e| e.0).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(reals[0], 2.0, 1e-10));
        assert!(approx_eq(reals[1], 3.0, 1e-10));

        // All imaginary parts should be zero
        for (_, imag) in &eigenvalues {
            assert!(imag.abs() < 1e-10, "Unexpected imaginary part: {}", imag);
        }
    }
}
