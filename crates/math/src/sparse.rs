//! Sparse matrix utilities and eigensolvers.
//!
//! Provides a feature-rich CSR (Compressed Sparse Row) matrix, iterative
//! eigensolvers (power iteration, inverse power iteration, Lanczos), and
//! iterative linear solvers (Conjugate Gradient, GMRES).
//!
//! # Example
//!
//! ```rust
//! use cv_scientific::sparse::{CsrMatrix, eigsh, EigWhich};
//!
//! // 3x3 diagonal matrix with eigenvalues 1, 2, 3
//! let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
//! let a = CsrMatrix::from_triplets(3, 3, &triplets);
//!
//! // Find the 2 largest eigenvalues
//! let (vals, _vecs) = eigsh(&a, 2, EigWhich::Largest, 100, 1e-10).unwrap();
//! assert!((vals[0] - 3.0).abs() < 1e-6);
//! assert!((vals[1] - 2.0).abs() < 1e-6);
//! ```

use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// CSR Matrix
// ---------------------------------------------------------------------------

/// Compressed Sparse Row (CSR) matrix.
///
/// This is a more full-featured implementation than the one in [`crate::linalg`],
/// adding `from_dense`, `spmm`, `diagonal`, `scale`, `add_identity`, `nnz`,
/// and `density`.
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
            row.sort_by_key(|&(c, _)| c);
            let mut prev_col: Option<usize> = None;
            for &(c, v) in row.iter() {
                if prev_col == Some(c) {
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

    /// Build a CSR matrix from a dense `DMatrix`, dropping entries with
    /// absolute value below `tol`.
    pub fn from_dense(mat: &DMatrix<f64>, tol: f64) -> Self {
        let (nrows, ncols) = mat.shape();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for r in 0..nrows {
            for c in 0..ncols {
                let v = mat[(r, c)];
                if v.abs() > tol {
                    col_idx.push(c);
                    values.push(v);
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

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Fraction of non-zero entries.
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
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

    /// Sparse matrix-matrix multiply: `C = self * other` (both CSR).
    pub fn spmm(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(
            self.ncols, other.nrows,
            "Dimension mismatch: {}x{} * {}x{}",
            self.nrows, self.ncols, other.nrows, other.ncols
        );

        let mut triplets = Vec::new();
        for r in 0..self.nrows {
            // Accumulate row r of C using a dense accumulator
            let mut acc = vec![0.0f64; other.ncols];
            let mut touched = Vec::new();
            for idx_a in self.row_ptr[r]..self.row_ptr[r + 1] {
                let k = self.col_idx[idx_a];
                let a_val = self.values[idx_a];
                for idx_b in other.row_ptr[k]..other.row_ptr[k + 1] {
                    let c = other.col_idx[idx_b];
                    if acc[c] == 0.0 {
                        touched.push(c);
                    }
                    acc[c] += a_val * other.values[idx_b];
                }
            }
            for &c in &touched {
                if acc[c].abs() > 0.0 {
                    triplets.push((r, c, acc[c]));
                }
                acc[c] = 0.0; // reset for next row
            }
        }

        CsrMatrix::from_triplets(self.nrows, other.ncols, &triplets)
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

    /// Extract the diagonal as a vector.
    ///
    /// Returns a vector of length `min(nrows, ncols)`.
    #[allow(clippy::needless_range_loop)]
    pub fn diagonal(&self) -> DVector<f64> {
        let n = self.nrows.min(self.ncols);
        let mut diag = DVector::zeros(n);
        for r in 0..n {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                if self.col_idx[idx] == r {
                    diag[r] = self.values[idx];
                    break;
                }
            }
        }
        diag
    }

    /// Scale all entries: `A = alpha * A`.
    pub fn scale(&mut self, alpha: f64) {
        for v in &mut self.values {
            *v *= alpha;
        }
    }

    /// Add a scaled identity: `A = A + sigma * I`.
    ///
    /// Only modifies the diagonal. If a diagonal entry does not already exist
    /// in the sparsity pattern, the matrix is rebuilt with that entry added.
    #[allow(clippy::needless_range_loop)]
    pub fn add_identity(&mut self, sigma: f64) {
        let n = self.nrows.min(self.ncols);
        // Try to modify existing diagonal entries first
        let mut found = vec![false; n];
        for r in 0..n {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                if self.col_idx[idx] == r {
                    self.values[idx] += sigma;
                    found[r] = true;
                    break;
                }
            }
        }
        // If any diagonal entries were missing, rebuild
        if found.iter().any(|&f| !f) {
            let mut triplets = Vec::with_capacity(self.values.len() + n);
            for r in 0..self.nrows {
                for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                    triplets.push((r, self.col_idx[idx], self.values[idx]));
                }
            }
            for r in 0..n {
                if !found[r] {
                    triplets.push((r, r, sigma));
                }
            }
            let rebuilt = CsrMatrix::from_triplets(self.nrows, self.ncols, &triplets);
            self.row_ptr = rebuilt.row_ptr;
            self.col_idx = rebuilt.col_idx;
            self.values = rebuilt.values;
        }
    }
}

// ---------------------------------------------------------------------------
// Eigensolvers
// ---------------------------------------------------------------------------

/// Which eigenvalues to compute.
#[derive(Debug, Clone, Copy)]
pub enum EigWhich {
    /// Largest magnitude eigenvalues.
    Largest,
    /// Smallest magnitude eigenvalues.
    Smallest,
    /// Eigenvalues closest to a given shift.
    NearSigma(f64),
}

/// Power iteration for finding the largest eigenvalue/eigenvector of a
/// symmetric matrix.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn eigs_power(
    a: &CsrMatrix,
    max_iters: usize,
    tol: f64,
) -> Result<(f64, DVector<f64>), String> {
    assert_eq!(a.nrows, a.ncols, "Matrix must be square");
    let n = a.nrows;
    if n == 0 {
        return Err("Matrix must be non-empty".into());
    }

    // Start with a deterministic vector (1, 1, ..., 1) normalized
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    let mut eigenvalue = 0.0;

    for _ in 0..max_iters {
        let w = a.spmv(&v);
        let new_eigenvalue = v.dot(&w);
        let norm = w.norm();
        if norm < 1e-15 {
            return Err("Power iteration: zero vector encountered".into());
        }
        v = w / norm;
        if (new_eigenvalue - eigenvalue).abs() < tol {
            return Ok((new_eigenvalue, v));
        }
        eigenvalue = new_eigenvalue;
    }

    Ok((eigenvalue, v))
}

/// Inverse power iteration for finding the eigenvalue closest to `sigma`.
///
/// Internally solves `(A - sigma*I) * v = v_old` using CG at each step.
/// Returns `(eigenvalue, eigenvector)`.
pub fn eigs_inverse_power(
    a: &CsrMatrix,
    sigma: f64,
    max_iters: usize,
    tol: f64,
) -> Result<(f64, DVector<f64>), String> {
    assert_eq!(a.nrows, a.ncols, "Matrix must be square");
    let n = a.nrows;
    if n == 0 {
        return Err("Matrix must be non-empty".into());
    }

    // Build shifted matrix: B = A - sigma * I
    let mut b = a.clone();
    b.add_identity(-sigma);

    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    let mut eigenvalue = 0.0;

    for _ in 0..max_iters {
        // Solve B * w = v using CG (or GMRES for non-SPD)
        let w = gmres_solve(&b, &v, 30, 200, 1e-12)?;
        let norm = w.norm();
        if norm < 1e-15 {
            return Err("Inverse power iteration: zero vector encountered".into());
        }
        v = &w / norm;
        // Rayleigh quotient with original matrix
        let av = a.spmv(&v);
        let new_eigenvalue = v.dot(&av);
        if (new_eigenvalue - eigenvalue).abs() < tol {
            return Ok((new_eigenvalue, v));
        }
        eigenvalue = new_eigenvalue;
    }

    Ok((eigenvalue, v))
}

/// Lanczos algorithm for computing `k` eigenvalues (and eigenvectors) of a
/// symmetric sparse matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues is a vector of
/// length `k` and eigenvectors is an `n x k` matrix. Eigenvalues are sorted
/// by the criterion specified in `which` (descending for `Largest`, ascending
/// for `Smallest`).
pub fn eigsh(
    a: &CsrMatrix,
    k: usize,
    which: EigWhich,
    max_iters: usize,
    tol: f64,
) -> Result<(DVector<f64>, DMatrix<f64>), String> {
    assert_eq!(a.nrows, a.ncols, "Matrix must be square");
    let n = a.nrows;
    if k == 0 || k > n {
        return Err(format!("k={} must be in [1, n={}]", k, n));
    }

    // Number of Lanczos vectors to compute (use more than k for better convergence)
    let m = max_iters.min(n).max(k + 1);

    // Lanczos vectors (stored as columns)
    let mut q_vecs: Vec<DVector<f64>> = Vec::with_capacity(m + 1);
    // Tridiagonal entries
    let mut alpha = Vec::with_capacity(m); // diagonal
    let mut beta = Vec::with_capacity(m); // sub-diagonal

    // Start with a normalized vector
    let q0 = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    q_vecs.push(q0);

    let mut w;
    for j in 0..m {
        w = a.spmv(&q_vecs[j]);
        let a_j = q_vecs[j].dot(&w);
        alpha.push(a_j);

        // w = w - alpha_j * q_j
        w -= &q_vecs[j] * a_j;
        if j > 0 {
            // w = w - beta_{j-1} * q_{j-1}
            w -= &q_vecs[j - 1] * beta[j - 1];
        }

        // Re-orthogonalize (full, for numerical stability)
        for qi in &q_vecs {
            let proj = qi.dot(&w);
            w -= qi * proj;
        }

        let b_j = w.norm();
        if b_j < tol {
            // Invariant subspace found
            break;
        }
        beta.push(b_j);
        q_vecs.push(w / b_j);
    }

    let m_actual = alpha.len();
    if m_actual == 0 {
        return Err("Lanczos produced no vectors".into());
    }

    // Build tridiagonal matrix T and solve dense eigenproblem
    let mut t = DMatrix::zeros(m_actual, m_actual);
    for i in 0..m_actual {
        t[(i, i)] = alpha[i];
        if i + 1 < m_actual && i < beta.len() {
            t[(i, i + 1)] = beta[i];
            t[(i + 1, i)] = beta[i];
        }
    }

    let eigen = t.clone().symmetric_eigen();
    let vals = eigen.eigenvalues;
    let vecs_t = eigen.eigenvectors;

    // Sort eigenvalues according to `which`
    let mut indices: Vec<usize> = (0..vals.len()).collect();
    match which {
        EigWhich::Largest => {
            indices.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
        }
        EigWhich::Smallest => {
            indices.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap());
        }
        EigWhich::NearSigma(sigma) => {
            indices.sort_by(|&i, &j| {
                let di = (vals[i] - sigma).abs();
                let dj = (vals[j] - sigma).abs();
                di.partial_cmp(&dj).unwrap()
            });
        }
    }

    let k_actual = k.min(m_actual);
    let mut result_vals = DVector::zeros(k_actual);
    let mut result_vecs = DMatrix::zeros(n, k_actual);

    // Map Ritz vectors back to original space: v_i = Q * z_i
    let q_count = q_vecs.len();
    for (out_col, &idx) in indices.iter().take(k_actual).enumerate() {
        result_vals[out_col] = vals[idx];
        for j in 0..m_actual.min(q_count) {
            let coeff = vecs_t[(j, idx)];
            result_vecs.column_mut(out_col).axpy(coeff, &q_vecs[j], 1.0);
        }
        // Normalize
        let norm = result_vecs.column(out_col).norm();
        if norm > 1e-15 {
            result_vecs.column_mut(out_col).scale_mut(1.0 / norm);
        }
    }

    Ok((result_vals, result_vecs))
}

// ---------------------------------------------------------------------------
// Iterative linear solvers
// ---------------------------------------------------------------------------

/// Conjugate Gradient solver for symmetric positive-definite systems `A x = b`.
///
/// Returns the solution vector `x`.
pub fn cg_solve(
    a: &CsrMatrix,
    b: &DVector<f64>,
    max_iters: usize,
    tol: f64,
) -> Result<DVector<f64>, String> {
    assert_eq!(a.nrows, a.ncols, "Matrix must be square for CG");
    assert_eq!(a.nrows, b.nrows(), "Dimension mismatch");

    let n = a.nrows;
    let mut x = DVector::zeros(n);
    let mut r = b.clone(); // r = b - A*x, but x=0 so r=b
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    let b_norm = b.norm();
    if b_norm < 1e-15 {
        return Ok(x);
    }

    for _ in 0..max_iters {
        let ap = a.spmv(&p);
        let p_ap = p.dot(&ap);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rs_old / p_ap;
        x += &p * alpha;
        r -= &ap * alpha;
        let rs_new = r.dot(&r);
        if rs_new.sqrt() / b_norm < tol {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        p = &r + &p * beta;
        rs_old = rs_new;
    }

    Ok(x)
}

/// GMRES(restart) solver for general sparse systems `A x = b`.
///
/// Uses the restarted Generalized Minimal Residual method. The `restart`
/// parameter controls how many Arnoldi vectors are kept before restarting
/// (typical value: 30).
pub fn gmres_solve(
    a: &CsrMatrix,
    b: &DVector<f64>,
    restart: usize,
    max_iters: usize,
    tol: f64,
) -> Result<DVector<f64>, String> {
    assert_eq!(a.nrows, a.ncols, "Matrix must be square for GMRES");
    assert_eq!(a.nrows, b.nrows(), "Dimension mismatch");

    let n = a.nrows;
    let b_norm = b.norm();
    if b_norm < 1e-15 {
        return Ok(DVector::zeros(n));
    }

    let mut x = DVector::zeros(n);

    for _outer in 0..(max_iters / restart.max(1) + 1) {
        let r = b - &a.spmv(&x);
        let r_norm = r.norm();
        if r_norm / b_norm < tol {
            return Ok(x);
        }

        let m = restart.min(n);
        // Arnoldi basis vectors
        let mut v: Vec<DVector<f64>> = Vec::with_capacity(m + 1);
        v.push(&r / r_norm);

        // Upper Hessenberg matrix H (stored as (m+1) x m)
        let mut h = DMatrix::zeros(m + 1, m);
        // Transformed RHS for least squares
        let mut g = DVector::zeros(m + 1);
        g[0] = r_norm;

        // Givens rotation coefficients
        let mut cs: Vec<f64> = Vec::with_capacity(m);
        let mut sn: Vec<f64> = Vec::with_capacity(m);

        let mut j = 0;
        while j < m {
            let w = a.spmv(&v[j]);
            let mut wj = w;

            // Arnoldi: orthogonalize against previous basis vectors
            for i in 0..=j {
                h[(i, j)] = v[i].dot(&wj);
                wj -= &v[i] * h[(i, j)];
            }
            h[(j + 1, j)] = wj.norm();

            if h[(j + 1, j)].abs() < 1e-15 {
                // Lucky breakdown
                j += 1;
                break;
            }
            v.push(wj / h[(j + 1, j)]);

            // Apply previous Givens rotations to column j of H
            for i in 0..j {
                let temp = cs[i] * h[(i, j)] + sn[i] * h[(i + 1, j)];
                h[(i + 1, j)] = -sn[i] * h[(i, j)] + cs[i] * h[(i + 1, j)];
                h[(i, j)] = temp;
            }

            // Compute new Givens rotation for row j
            let a_jj = h[(j, j)];
            let a_j1j = h[(j + 1, j)];
            let denom = (a_jj * a_jj + a_j1j * a_j1j).sqrt();
            if denom < 1e-15 {
                cs.push(1.0);
                sn.push(0.0);
            } else {
                cs.push(a_jj / denom);
                sn.push(a_j1j / denom);
            }

            h[(j, j)] = cs[j] * a_jj + sn[j] * a_j1j;
            h[(j + 1, j)] = 0.0;

            // Apply rotation to g
            let temp = cs[j] * g[j] + sn[j] * g[j + 1];
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
            g[j] = temp;

            if g[j + 1].abs() / b_norm < tol {
                j += 1;
                break;
            }
            j += 1;
        }

        // Back-substitute to find y: H(0..j, 0..j) * y = g(0..j)
        let mut y = DVector::zeros(j);
        for i in (0..j).rev() {
            let mut s = g[i];
            for k in (i + 1)..j {
                s -= h[(i, k)] * y[k];
            }
            if h[(i, i)].abs() < 1e-15 {
                y[i] = 0.0;
            } else {
                y[i] = s / h[(i, i)];
            }
        }

        // Update x = x + V * y
        for i in 0..j {
            x += &v[i] * y[i];
        }

        // Check convergence
        let final_r = b - &a.spmv(&x);
        if final_r.norm() / b_norm < tol {
            return Ok(x);
        }
    }

    Ok(x)
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

    #[test]
    fn test_csr_from_triplets_to_dense_roundtrip() {
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 5.0),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        let dense = csr.to_dense();
        let expected =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]);
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_csr_from_dense_roundtrip() {
        let mat = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]);
        let csr = CsrMatrix::from_dense(&mat, 1e-15);
        let back = csr.to_dense();
        assert_eq!(mat, back);
        assert_eq!(csr.nnz(), 5); // 5 non-zero entries
    }

    #[test]
    fn test_spmv_known_result() {
        // A = [[2, 0, 1], [0, 3, 0], [1, 0, 4]]
        let triplets = vec![
            (0, 0, 2.0),
            (0, 2, 1.0),
            (1, 1, 3.0),
            (2, 0, 1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let y = a.spmv(&x);
        // [2*1+0*2+1*3, 0*1+3*2+0*3, 1*1+0*2+4*3] = [5, 6, 13]
        assert!(approx_eq(y[0], 5.0, 1e-12));
        assert!(approx_eq(y[1], 6.0, 1e-12));
        assert!(approx_eq(y[2], 13.0, 1e-12));
    }

    #[test]
    fn test_spmm() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]
        let a =
            CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);
        let b =
            CsrMatrix::from_triplets(2, 2, &[(0, 0, 5.0), (0, 1, 6.0), (1, 0, 7.0), (1, 1, 8.0)]);
        let c = a.spmm(&b);
        let dense_c = c.to_dense();
        let expected = DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(dense_c, expected);
    }

    #[test]
    fn test_diagonal_and_density() {
        let triplets = vec![(0, 0, 10.0), (0, 1, 1.0), (1, 1, 20.0), (2, 2, 30.0)];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        let diag = csr.diagonal();
        assert!(approx_eq(diag[0], 10.0, 1e-15));
        assert!(approx_eq(diag[1], 20.0, 1e-15));
        assert!(approx_eq(diag[2], 30.0, 1e-15));
        // 4 nnz out of 9
        assert!(approx_eq(csr.density(), 4.0 / 9.0, 1e-12));
    }

    #[test]
    fn test_scale_and_add_identity() {
        let triplets = vec![(0, 0, 2.0), (1, 1, 3.0)];
        let mut csr = CsrMatrix::from_triplets(2, 2, &triplets);
        csr.scale(2.0);
        let d = csr.to_dense();
        assert!(approx_eq(d[(0, 0)], 4.0, 1e-15));
        assert!(approx_eq(d[(1, 1)], 6.0, 1e-15));

        csr.add_identity(1.0);
        let d2 = csr.to_dense();
        assert!(approx_eq(d2[(0, 0)], 5.0, 1e-15));
        assert!(approx_eq(d2[(1, 1)], 7.0, 1e-15));
    }

    #[test]
    fn test_transpose() {
        let triplets = vec![(0, 1, 5.0), (1, 0, 3.0), (2, 2, 7.0)];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let at = a.transpose();
        let dense_at = at.to_dense();
        assert!(approx_eq(dense_at[(1, 0)], 5.0, 1e-15));
        assert!(approx_eq(dense_at[(0, 1)], 3.0, 1e-15));
        assert!(approx_eq(dense_at[(2, 2)], 7.0, 1e-15));
    }

    #[test]
    fn test_power_iteration_dominant_eigenvalue() {
        // Diagonal matrix with eigenvalues 1, 2, 5 => dominant = 5
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 5.0)];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let (val, vec) = eigs_power(&a, 1000, 1e-10).unwrap();
        assert!(approx_eq(val, 5.0, 1e-6));
        // Eigenvector should be aligned with e3
        assert!(vec[2].abs() > 0.99);
    }

    #[test]
    fn test_eigsh_largest_eigenvalues() {
        // Diagonal matrix with eigenvalues 1, 2, 3, 4, 5
        let triplets: Vec<_> = (0..5).map(|i| (i, i, (i + 1) as f64)).collect();
        let a = CsrMatrix::from_triplets(5, 5, &triplets);
        let (vals, vecs) = eigsh(&a, 3, EigWhich::Largest, 100, 1e-10).unwrap();
        assert_eq!(vals.len(), 3);
        // Should be 5, 4, 3 (descending)
        assert!(approx_eq(vals[0], 5.0, 1e-6));
        assert!(approx_eq(vals[1], 4.0, 1e-6));
        assert!(approx_eq(vals[2], 3.0, 1e-6));
        // Each eigenvector column should have unit norm
        for c in 0..3 {
            assert!(approx_eq(vecs.column(c).norm(), 1.0, 1e-10));
        }
    }

    #[test]
    fn test_eigsh_smallest_eigenvalues() {
        let triplets: Vec<_> = (0..5).map(|i| (i, i, (i + 1) as f64)).collect();
        let a = CsrMatrix::from_triplets(5, 5, &triplets);
        let (vals, _) = eigsh(&a, 2, EigWhich::Smallest, 100, 1e-10).unwrap();
        assert!(approx_eq(vals[0], 1.0, 1e-6));
        assert!(approx_eq(vals[1], 2.0, 1e-6));
    }

    #[test]
    fn test_cg_solve_spd_system() {
        // SPD matrix: A = [[4, 1], [1, 3]], b = [1, 2]
        // Solution: x = [1/11, 7/11]
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, &triplets);
        let b = DVector::from_column_slice(&[1.0, 2.0]);
        let x = cg_solve(&a, &b, 100, 1e-10).unwrap();
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-8));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-8));
    }

    #[test]
    fn test_gmres_solve_general_system() {
        // Non-symmetric system: A = [[2, 1], [0, 3]], b = [5, 9]
        // Solution: x = [1, 3]
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, &triplets);
        let b = DVector::from_column_slice(&[5.0, 9.0]);
        let x = gmres_solve(&a, &b, 10, 100, 1e-10).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-8));
        assert!(approx_eq(x[1], 3.0, 1e-8));
    }

    #[test]
    fn test_inverse_power_iteration() {
        // Diagonal matrix with eigenvalues 1, 5, 10
        // Shift sigma=0.5 should find eigenvalue closest to 0.5 => 1.0
        let triplets = vec![(0, 0, 1.0), (1, 1, 5.0), (2, 2, 10.0)];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let (val, _) = eigs_inverse_power(&a, 0.5, 200, 1e-8).unwrap();
        assert!(approx_eq(val, 1.0, 1e-4));
    }

    #[test]
    fn test_eigsh_near_sigma() {
        // Eigenvalues 1, 3, 5, 7, 9 — find 2 nearest to 6.0 => 5 and 7
        let triplets: Vec<_> = (0..5).map(|i| (i, i, (2 * i + 1) as f64)).collect();
        let a = CsrMatrix::from_triplets(5, 5, &triplets);
        let (vals, _) = eigsh(&a, 2, EigWhich::NearSigma(6.0), 100, 1e-10).unwrap();
        let mut sorted: Vec<f64> = vals.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(sorted[0], 5.0, 1e-6));
        assert!(approx_eq(sorted[1], 7.0, 1e-6));
    }
}
