use crate::sparse::{LinearSolver, SparseMatrix};
use cv_hal::compute::ComputeDevice;
use nalgebra::DVector;

/// Maximum iterations before declaring non-convergence.
const MAX_ITERATIONS: usize = 1000;

/// Default convergence tolerance.
const DEFAULT_TOLERANCE: f64 = 1e-10;

/// GPU-accelerated Conjugate Gradient solver.
pub struct GpuCgSolver {
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for GpuCgSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuCgSolver {
    pub fn new() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            max_iterations: MAX_ITERATIONS,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// CPU fallback implementation of CG solver.
    /// Used when GPU is not available or for small systems.
    fn solve_cpu(&self, a: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String> {
        let n = b.len();

        // x₀ = 0
        let mut x = DVector::zeros(n);

        // r₀ = b - A*x₀ = b (since x₀ = 0)
        let mut r = b.clone();

        // p₀ = r₀
        let mut p = r.clone();

        // rr = r·r
        let mut rr = r.dot(&r);

        for _k in 0..self.max_iterations {
            // Compute Ap = A * p (sparse matrix-vector multiply)
            let ap = spmv_cpu(a, &p);

            // α = rr / (p·Ap)
            let p_ap = p.dot(&ap);
            if p_ap.abs() < 1e-30 {
                return Err("CG: breakdown, p·Ap ≈ 0".to_string());
            }
            let alpha = rr / p_ap;

            // x = x + α*p
            x += alpha * &p;

            // r = r - α*Ap
            r -= alpha * &ap;

            // Check convergence
            let rr_new = r.dot(&r);
            if rr_new.sqrt() < self.tolerance {
                return Ok(x);
            }

            // β = rr_new / rr
            let beta = rr_new / rr;

            // p = r + β*p
            p = &r + beta * &p;

            rr = rr_new;
        }

        // Return best solution even if not fully converged
        Ok(x)
    }
}

/// CPU sparse matrix-vector multiply: y = A * x (CSR format)
fn spmv_cpu(a: &SparseMatrix, x: &DVector<f64>) -> DVector<f64> {
    let mut y = DVector::zeros(a.rows);
    for i in 0..a.rows {
        let row_start = a.row_ptr[i] as usize;
        let row_end = a.row_ptr[i + 1] as usize;
        let mut sum = 0.0;
        for j in row_start..row_end {
            sum += a.values[j] * x[a.col_indices[j] as usize];
        }
        y[i] = sum;
    }
    y
}

impl LinearSolver for GpuCgSolver {
    fn solve(
        &self,
        _ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, String> {
        // For now, use CPU fallback
        // GPU implementation will use the SpMV and vector_ops WGSL shaders
        // TODO: Implement GPU dispatch when GpuContext is available
        self.solve_cpu(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::Triplet;

    #[test]
    fn test_cg_solver_diagonal() {
        // Solve: [2 0 0; 0 3 0; 0 0 4] * x = [4; 9; 16]
        // Expected: x = [2; 3; 4]
        let triplets = vec![
            Triplet {
                row: 0,
                col: 0,
                val: 2.0,
            },
            Triplet {
                row: 1,
                col: 1,
                val: 3.0,
            },
            Triplet {
                row: 2,
                col: 2,
                val: 4.0,
            },
        ];
        let a = SparseMatrix::from_triplets(3, 3, &triplets);
        let b = DVector::from_vec(vec![4.0, 9.0, 16.0]);

        let solver = GpuCgSolver::new();
        let ctx = cv_hal::compute::get_device().unwrap();
        let x = solver.solve(&ctx, &a, &b).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}, expected 2.0", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "x[1] = {}, expected 3.0", x[1]);
        assert!((x[2] - 4.0).abs() < 1e-6, "x[2] = {}, expected 4.0", x[2]);
    }

    #[test]
    fn test_cg_solver_spd() {
        // Solve: [4 1; 1 3] * x = [1; 2]
        // This is SPD (eigenvalues > 0)
        let triplets = vec![
            Triplet {
                row: 0,
                col: 0,
                val: 4.0,
            },
            Triplet {
                row: 0,
                col: 1,
                val: 1.0,
            },
            Triplet {
                row: 1,
                col: 0,
                val: 1.0,
            },
            Triplet {
                row: 1,
                col: 1,
                val: 3.0,
            },
        ];
        let a = SparseMatrix::from_triplets(2, 2, &triplets);
        let b = DVector::from_vec(vec![1.0, 2.0]);

        let solver = GpuCgSolver::new();
        let ctx = cv_hal::compute::get_device().unwrap();
        let x = solver.solve(&ctx, &a, &b).unwrap();

        // Verify A*x ≈ b
        let ax = spmv_cpu(&a, &x);
        for i in 0..b.len() {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6,
                "A*x[{}] = {}, expected {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_cg_solver_larger() {
        // 5x5 tridiagonal SPD matrix
        let n = 5;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet {
                row: i,
                col: i,
                val: 4.0,
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
        let a = SparseMatrix::from_triplets(n, n, &triplets);
        let b = DVector::from_vec(vec![1.0; n]);

        let solver = GpuCgSolver::new();
        let ctx = cv_hal::compute::get_device().unwrap();
        let x = solver.solve(&ctx, &a, &b).unwrap();

        // Verify A*x ≈ b
        let ax = spmv_cpu(&a, &x);
        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6,
                "A*x[{}] = {}, expected {}",
                i,
                ax[i],
                b[i]
            );
        }
    }
}
