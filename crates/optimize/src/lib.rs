pub mod factor_graph;
pub mod factors;
pub mod gpu_solver;
pub mod isam2;
pub mod pose_graph;
pub mod sparse;

// Note: isam2 has its own inline factor graph types that predate the
// canonical types in factor_graph.rs / factors.rs.  To avoid name
// collisions at the crate root, import from the modules directly:
//   use cv_optimize::factor_graph::{FactorGraph, Key, Values, ...};
//   use cv_optimize::factors::{PriorFactor, BetweenFactor, ...};

use cv_hal::compute::ComputeDevice;
use nalgebra::DVector;
use sparse::{CgSolver, LinearSolver, SparseMatrix};

pub trait CostFunction {
    fn dimensions(&self) -> (usize, usize); // (residuals, parameters)
    fn residuals(&self, params: &DVector<f64>) -> DVector<f64>;
    fn jacobian(&self, params: &DVector<f64>) -> SparseMatrix;
}

pub struct SparseLMSolver<'a> {
    pub ctx: &'a ComputeDevice<'a>,
    pub config: LMConfig,
}

pub struct LMConfig {
    pub max_iters: usize,
    pub lambda: f64,
    pub tolerance: f64,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            lambda: 0.001,
            tolerance: 1e-6,
        }
    }
}

impl<'a> SparseLMSolver<'a> {
    pub fn new(ctx: &'a ComputeDevice<'a>) -> Self {
        Self {
            ctx,
            config: LMConfig::default(),
        }
    }

    pub fn minimize(
        &self,
        cost_fn: &dyn CostFunction,
        initial_params: DVector<f64>,
    ) -> Result<DVector<f64>, String> {
        let mut x = initial_params;
        let mut lambda = self.config.lambda;

        let mut r = cost_fn.residuals(&x);
        let mut current_err = r.norm_squared();

        for _ in 0..self.config.max_iters {
            let j = cost_fn.jacobian(&x);

            // J^T * J * delta = -J^T * r
            // This part is tricky with SparseMatrix.
            // For now, let's assume we solve it via CG or a direct solver.
            // We need J^T * J.

            // Simplified: solve J * delta = -r in least squares sense
            // delta = (J^T J + lambda*I)^-1 * (-J^T r)

            // For now, let's keep the solve method signature and implement the logic there.
            let delta = self.solve_lm_step(&j, &r, lambda)?;

            let next_x = &x + &delta;
            let next_r = cost_fn.residuals(&next_x);
            let next_err = next_r.norm_squared();

            if next_err < current_err {
                current_err = next_err;
                x = next_x;
                r = next_r;
                lambda /= 10.0;
                if delta.norm() < self.config.tolerance {
                    break;
                }
            } else {
                lambda *= 10.0;
            }
        }

        Ok(x)
    }

    fn solve_lm_step(
        &self,
        j: &SparseMatrix,
        r: &DVector<f64>,
        lambda: f64,
    ) -> Result<DVector<f64>, String> {
        // Solve (J^T J + lambda * diag(J^T J)) delta = -J^T r using CG
        let n = j.cols;
        let mut x = DVector::zeros(n);
        let rhs = -j.transpose_spmv_ctx(self.ctx, r)?;

        // Cache diagonal of J^T J for Marquardt damping
        let diag_jtj = self.compute_jtj_diagonal(j);

        let mut residual = rhs.clone();
        let mut p = residual.clone();
        let mut rsold = residual.dot(&residual);

        for _ in 0..100 {
            let jp = j.spmv_ctx(self.ctx, &p)?;
            let j_tj_p = j.transpose_spmv_ctx(self.ctx, &jp)?;
            let v = j_tj_p + lambda * diag_jtj.component_mul(&p);

            let pap = p.dot(&v);
            if pap.abs() < 1e-10 {
                break;
            }
            let alpha = rsold / pap;
            x += alpha * &p;
            residual -= alpha * &v;

            let rsnew = residual.dot(&residual);
            if rsnew.sqrt() < self.config.tolerance {
                break;
            }
            p = &residual + (rsnew / rsold) * &p;
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Compute diagonal of J^T J for Marquardt damping
    fn compute_jtj_diagonal(&self, j: &SparseMatrix) -> DVector<f64> {
        let mut diag = DVector::zeros(j.cols);
        for r in 0..j.rows {
            let start = j.row_ptr[r] as usize;
            let end = j.row_ptr[r + 1] as usize;
            for i in start..end {
                let c = j.col_indices[i] as usize;
                let v = j.values[i];
                diag[c] += v * v;
            }
        }
        diag
    }

    /// Solves the normal equations SparseMatrix * x = b
    pub fn solve(&self, a: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String> {
        let solver = CgSolver {
            max_iters: 1000,
            tolerance: self.config.tolerance,
        };
        solver.solve(self.ctx, a, b)
    }
}
