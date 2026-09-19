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
//! use cv_math::linalg;
//!
//! let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 5.0, 3.0]);
//! let b = DVector::from_column_slice(&[4.0, 7.0]);
//! let x = linalg::solve(&a, &b).unwrap();
//! assert!((x[0] - 5.0).abs() < 1e-10);
//! assert!((x[1] + 6.0).abs() < 1e-10);
//! ```

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use num_traits::Float;

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
    // nalgebra LU gives us P*A = L*U, with the permutation available from `lu.p()`.
    let l = lu.l();
    let u = lu.u();

    // Recover the permutation from nalgebra's own pivot sequence rather than
    // matching rows of L*U against A: the previous row-matching used a fixed
    // 1e-10 tolerance, so for large-magnitude matrices no row matched and every
    // pivot silently stayed 0.
    let perm = lu.p();
    let mut perm_mat = DMatrix::<f64>::identity(m, m);
    perm.permute_rows(&mut perm_mat);

    // Row i of P*A equals row `pivots[i]` of A, i.e. the single 1 in row i of
    // the permutation matrix sits in column `pivots[i]`.
    let mut pivots = vec![0usize; m];
    for i in 0..m {
        for j in 0..m {
            if perm_mat[(i, j)] != 0.0 {
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
    if a.nrows() < n {
        return Err(format!(
            "under-determined system ({}x{}): QR least-squares path requires m >= n",
            a.nrows(),
            n
        ));
    }
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
    indices.sort_by(|&i, &j| vals[i].total_cmp(&vals[j]));

    let sorted_vals = DVector::from_fn(vals.len(), |i, _| vals[indices[i]]);
    let sorted_vecs = DMatrix::from_fn(m, n, |r, c| vecs[(r, indices[c])]);

    Ok((sorted_vals, sorted_vecs))
}

/// Smallest eigenvector of a symmetric 3x3 matrix, in closed form.
///
/// Matches Open3D `PointCloudImpl.h` / Geometric Tools
/// `RobustEigenSymmetric3x3`: the eigenvalue is found with the trigonometric
/// (Cardano) method and the eigenvector is the largest cross-product of the
/// rows of `M - lambda_min * I`. No iteration, exact result in ~50 scalar ops.
///
/// Generic over the scalar type so the f32 GPU/point-cloud paths and the f64
/// CPU normal-estimation path share one implementation.
///
/// Degenerate inputs (zero or isotropic matrices) return the unit vector
/// `(0, 0, 1)` — the same fallback used by every previous copy.
pub fn min_eigenvector_3x3<T>(m: &Matrix3<T>) -> Vector3<T>
where
    T: nalgebra::Scalar + Float,
{
    fn cross3<T: Float>(a: [T; 3], b: [T; 3]) -> [T; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn dot3<T: Float>(a: [T; 3], b: [T; 3]) -> T {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    let two = T::from(2.0f64).unwrap();
    let three = T::from(3.0f64).unwrap();
    let six = T::from(6.0f64).unwrap();
    let fallback = || Vector3::new(T::zero(), T::zero(), T::one());

    // Normalize to prevent numerical overflow.
    let mut max_c = T::zero();
    for i in 0..3 {
        for j in 0..3 {
            let v = m[(i, j)].abs();
            if v > max_c {
                max_c = v;
            }
        }
    }
    if max_c < T::from(1e-30f64).unwrap() {
        return fallback();
    }
    let s = T::one() / max_c;
    let a00 = m[(0, 0)] * s;
    let a01 = m[(0, 1)] * s;
    let a02 = m[(0, 2)] * s;
    let a11 = m[(1, 1)] * s;
    let a12 = m[(1, 2)] * s;
    let a22 = m[(2, 2)] * s;

    let norm = a01 * a01 + a02 * a02 + a12 * a12;
    let q = (a00 + a11 + a22) / three;
    let b00 = a00 - q;
    let b11 = a11 - q;
    let b22 = a22 - q;
    let p = ((b00 * b00 + b11 * b11 + b22 * b22 + two * norm) / six).sqrt();
    if p < T::from(1e-10f64).unwrap() {
        return fallback();
    }

    // Determinant of (A - q*I) / p.
    let c00 = b11 * b22 - a12 * a12;
    let c01 = a01 * b22 - a12 * a02;
    let c02 = a01 * a12 - b11 * a02;
    let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
    let half_det = {
        let v = det * T::from(0.5f64).unwrap();
        if v < -T::one() {
            -T::one()
        } else if v > T::one() {
            T::one()
        } else {
            v
        }
    };
    let angle = half_det.acos() / three;

    // Minimum eigenvalue: q + p * cos(angle + 2*pi/3) * 2.
    let two_thirds_pi = T::from(2.094_395_1f64).unwrap();
    let eval_min = q + p * (angle + two_thirds_pi).cos() * two;

    // Eigenvector: best cross-product of rows of (A - eval_min * I).
    let r0 = [a00 - eval_min, a01, a02];
    let r1 = [a01, a11 - eval_min, a12];
    let r2 = [a02, a12, a22 - eval_min];

    let r0xr1 = cross3(r0, r1);
    let r0xr2 = cross3(r0, r2);
    let r1xr2 = cross3(r1, r2);

    let d0 = dot3(r0xr1, r0xr1);
    let d1 = dot3(r0xr2, r0xr2);
    let d2 = dot3(r1xr2, r1xr2);

    let best = if d0 >= d1 && d0 >= d2 {
        r0xr1
    } else if d1 >= d2 {
        r0xr2
    } else {
        r1xr2
    };

    let len = dot3(best, best).sqrt();
    if len < T::from(1e-10f64).unwrap() {
        return fallback();
    }
    Vector3::new(best[0] / len, best[1] / len, best[2] / len)
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
    fn test_lu_decompose_large_magnitude_pivots() {
        // Regression: the pivot rows were recovered by matching rows of L*U
        // against A with a fixed 1e-10 tolerance, so for a large-magnitude
        // matrix no row matched and every pivot silently stayed 0.
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                2.0e8, 1.0e8, 1.0e8, //
                8.0e8, 3.0e8, 3.0e8, //
                2.0e8, 7.0e8, 9.0e8,
            ],
        );
        let (l, u, pivots) = lu_decompose(&a).unwrap();

        // pivots must be a genuine permutation of 0..3.
        let mut sorted = pivots.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![0, 1, 2],
            "pivots not a permutation: {pivots:?}"
        );

        // Verify P*A = L*U with a relative tolerance.
        let lu = &l * &u;
        let mut max_rel = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let pa = a[(pivots[i], j)];
                max_rel = max_rel.max((pa - lu[(i, j)]).abs() / pa.abs().max(1.0));
            }
        }
        assert!(max_rel < 1e-10, "P*A != L*U, max relative error {max_rel}");
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
    fn test_eigh_nan_does_not_panic() {
        // Regression: the eigenvalue sort used partial_cmp(..).unwrap(), which
        // panicked as soon as an eigenvalue was NaN.
        let a = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
        let result = eigh(&a);
        assert!(result.is_ok());
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

    /// Verbatim copy of the previous `min_eigenvector_3x3` (the Open3D /
    /// Geometric Tools algorithm which lived in `cv-3d` and `cv-pointcloud`),
    /// used to pin the shared helper's behaviour.
    fn min_eigenvector_3x3_reference(m: &Matrix3<f64>) -> Vector3<f64> {
        let max_c = m.abs().max();
        if max_c < 1e-30 {
            return Vector3::z();
        }
        let s = 1.0 / max_c;
        let (a00, a01, a02) = (m[(0, 0)] * s, m[(0, 1)] * s, m[(0, 2)] * s);
        let (a11, a12, a22) = (m[(1, 1)] * s, m[(1, 2)] * s, m[(2, 2)] * s);

        let norm = a01 * a01 + a02 * a02 + a12 * a12;
        let q = (a00 + a11 + a22) / 3.0;
        let b00 = a00 - q;
        let b11 = a11 - q;
        let b22 = a22 - q;
        let p = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0).sqrt();
        if p < 1e-10 {
            return Vector3::z();
        }

        let c00 = b11 * b22 - a12 * a12;
        let c01 = a01 * b22 - a12 * a02;
        let c02 = a01 * a12 - b11 * a02;
        let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
        let half_det = (det * 0.5).clamp(-1.0, 1.0);
        let angle = half_det.acos() / 3.0;

        const TWO_THIRDS_PI: f64 = 2.094_395_1;
        let eval_min = q + p * (angle + TWO_THIRDS_PI).cos() * 2.0;

        let r0 = Vector3::new(a00 - eval_min, a01, a02);
        let r1 = Vector3::new(a01, a11 - eval_min, a12);
        let r2 = Vector3::new(a02, a12, a22 - eval_min);

        let r0xr1 = r0.cross(&r1);
        let r0xr2 = r0.cross(&r2);
        let r1xr2 = r1.cross(&r2);
        let d0 = r0xr1.norm_squared();
        let d1 = r0xr2.norm_squared();
        let d2 = r1xr2.norm_squared();
        let best = if d0 >= d1 && d0 >= d2 {
            r0xr1
        } else if d1 >= d2 {
            r0xr2
        } else {
            r1xr2
        };
        let len = best.norm();
        if len < 1e-10 {
            return Vector3::z();
        }
        best / len
    }

    #[test]
    fn test_min_eigenvector_3x3_matches_previous_implementation() {
        // Hand-built symmetric matrices: general SPD, a plane covariance,
        // a diagonal with a zero eigenvalue, the isotropic (p == 0) case and
        // the all-zero degenerate matrix.
        let cases = [
            Matrix3::new(4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0),
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            Matrix3::from_diagonal(&Vector3::new(2.0, 1.0, 0.0)),
            Matrix3::identity() * 3.0,
            Matrix3::zeros(),
        ];

        for (i, m) in cases.iter().enumerate() {
            let shared = min_eigenvector_3x3(m);
            let reference = min_eigenvector_3x3_reference(m);
            // Same direction (same sign convention): a close 1:1 match, or a
            // sign flip on the degenerate/isotropic cases where any direction
            // is an eigenvector.
            let direct = (shared - reference).norm();
            let flipped = (shared + reference).norm();
            assert!(
                direct < 1e-9 || flipped < 1e-9,
                "case {i}: shared {:?} vs reference {:?}",
                shared,
                reference
            );
            assert!(
                (shared.norm() - 1.0).abs() < 1e-9,
                "case {i}: not unit length: {:?}",
                shared
            );
        }

        // Degenerate zero matrix must fall back to the unit +Z vector,
        // exactly as every previous copy did.
        let zero = min_eigenvector_3x3(&Matrix3::<f64>::zeros());
        assert!((zero - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn test_min_eigenvector_3x3_f32() {
        // The f32 call sites (GPU / point-cloud) must agree with the f64 path.
        let m32 = Matrix3::new(1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let n = min_eigenvector_3x3(&m32);
        assert!(n.z.abs() > 0.99, "expected z-eigenvector, got {:?}", n);
        let degenerate: Vector3<f32> = min_eigenvector_3x3(&Matrix3::<f32>::zeros());
        assert!((degenerate - Vector3::new(0.0f32, 0.0, 1.0)).norm() < 1e-12);
    }
}
