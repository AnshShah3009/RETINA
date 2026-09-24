use cv_core::Tensor;
use cv_hal::compute::ComputeDevice;
pub use faer::sparse::Triplet;
use nalgebra::DVector;

/// Sparse Matrix representation in CSR format for GPU optimization
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[Triplet<usize, usize, f64>],
    ) -> Self {
        // Convert COO (triplets) to CSR. Sort first so that duplicate
        // (row, col) entries are adjacent and get SUMMED — e.g. two loop
        // closures contributing to the same Hessian block must accumulate,
        // not overwrite.
        let mut sorted: Vec<Triplet<usize, usize, f64>> = triplets.to_vec();
        sorted.sort_by_key(|t| (t.row, t.col));

        let mut merged: Vec<(usize, usize, f64)> = Vec::with_capacity(sorted.len());
        for t in sorted {
            match merged.last_mut() {
                Some(last) if last.0 == t.row && last.1 == t.col => last.2 += t.val,
                _ => merged.push((t.row, t.col, t.val)),
            }
        }

        let mut row_counts = vec![0; rows];
        for &(r, _, _) in &merged {
            row_counts[r] += 1;
        }

        let mut row_ptr = vec![0; rows + 1];
        for i in 0..rows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        let mut current_row_pos = vec![0; rows];
        let mut col_indices = vec![0u32; merged.len()];
        let mut values = vec![0.0; merged.len()];

        for (r, c, v) in merged {
            let pos = row_ptr[r] as usize + current_row_pos[r];
            col_indices[pos] = c as u32;
            values[pos] = v;
            current_row_pos[r] += 1;
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    pub fn spmv_ctx(&self, ctx: &ComputeDevice, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        match ctx {
            ComputeDevice::Gpu(gpu) => {
                // The SpMV is dispatched through a type-erased `ctx`, so the
                // compiler cannot prove it is a large, well-conditioned,
                // compute-bound product; and the round trip is actively harmful
                // here. The LM solve calls this twice per CG iteration, up to
                // 200 times per iteration and ~10 iterations per bundle
                // adjustment, so a device whose queues are already busy (or
                // whose wait is bounded) stalls the whole optimization. Use the
                // GPU only when the system is large enough to pay for it.
                const GPU_MIN_NNZ: usize = 1 << 20;
                if self.values.len() < GPU_MIN_NNZ {
                    return self.spmv_native(x);
                }

                // GPU kernels operate in f32; convert once for upload.
                let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
                let values_f32: Vec<f32> = self.values.iter().map(|&v| v as f32).collect();

                use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
                let x_tensor: cv_core::CpuTensor<f32> =
                    Tensor::from_vec(x_f32, cv_core::TensorShape::new(1, x.len(), 1))
                        .map_err(|e| format!("SpMV input tensor creation failed: {}", e))?;
                let x_gpu = x_tensor
                    .to_gpu_ctx(gpu)
                    .map_err(|e| format!("Upload to GPU failed: {}", e))?;
                let res_gpu = ctx
                    .spmv(&self.row_ptr, &self.col_indices, &values_f32, &x_gpu)
                    .map_err(|e| format!("GPU SpMV failed: {}", e))?;
                let res_cpu = res_gpu
                    .to_cpu_ctx(gpu)
                    .map_err(|e| format!("Download from GPU failed: {}", e))?;
                Ok(DVector::from_vec(
                    res_cpu
                        .as_slice()
                        .map_err(|e| format!("Data not on CPU: {}", e))?
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                ))
            }
            ComputeDevice::Cpu(_cpu) => self.spmv_native(x),
            ComputeDevice::Mlx(_) => Err("MLX SpMV not implemented yet. Use CPU backend.".into()),
        }
    }

    /// Native f64 sparse mat-vec. Used for the CPU backend and for small
    /// systems on other backends: downcasting to f32 for a GPU round trip
    /// loses ~7 significant digits (which stalls CG convergence on the LM
    /// normal equations) and costs more in transfers than the multiply itself
    /// for anything but a very large matrix.
    pub fn spmv_native(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        if x.len() != self.cols {
            return Err(format!(
                "spmv dimension mismatch: vector has {} entries, matrix has {} columns",
                x.len(),
                self.cols
            ));
        }
        let mut res = DVector::zeros(self.rows);
        for r in 0..self.rows {
            let start = self.row_ptr[r] as usize;
            let end = self.row_ptr[r + 1] as usize;
            let mut sum = 0.0f64;
            for i in start..end {
                sum += self.values[i] * x[self.col_indices[i] as usize];
            }
            res[r] = sum;
        }
        Ok(res)
    }

    /// Normal equations (J^T J, J^T r) as a *sparse* J^T J.
    ///
    /// J^T J is formed from the nonzeros of J directly, so the system stays
    /// sparse end to end. That matters for bundle adjustment: the Jacobian has
    /// one block per observation, so J^T J has O(observations) nonzeros while
    /// the dense form is O(parameters^2) and dominates the iteration.
    pub fn normal_equations_sparse(&self, r: &DVector<f64>) -> (SparseMatrix, DVector<f64>) {
        if r.len() != self.rows {
            return (SparseMatrix::from_triplets(0, 0, &[]), DVector::zeros(0));
        }
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut jtr = DVector::zeros(self.cols);
        for row in 0..self.rows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let rv = r[row];
            for a in start..end {
                let ca = self.col_indices[a] as usize;
                let va = self.values[a];
                jtr[ca] += va * rv;
                for b in start..end {
                    let cb = self.col_indices[b] as usize;
                    triplets.push(Triplet::new(ca, cb, va * self.values[b]));
                }
            }
        }
        (
            SparseMatrix::from_triplets(self.cols, self.cols, &triplets),
            jtr,
        )
    }

    /// Scale the diagonal of a sparse matrix in place (Marquardt damping).
    pub fn scale_diagonal(&mut self, scale: f64) {
        for r in 0..self.rows {
            let start = self.row_ptr[r] as usize;
            let end = self.row_ptr[r + 1] as usize;
            for i in start..end {
                if self.col_indices[i] as usize == r {
                    self.values[i] *= scale;
                }
            }
        }
    }

    /// Diagonal of J^T J, used for Marquardt damping.
    pub fn jtj_diagonal(&self) -> DVector<f64> {
        let mut diag = DVector::zeros(self.cols);
        for r in 0..self.rows {
            let start = self.row_ptr[r] as usize;
            let end = self.row_ptr[r + 1] as usize;
            for i in start..end {
                let c = self.col_indices[i] as usize;
                let v = self.values[i];
                diag[c] += v * v;
            }
        }
        diag
    }

    /// Normal equations (J^T J, J^T r) for J = self, built from the sparse
    /// structure. Returns a dense J^T J (the system is much smaller than J once
    /// the observations outnumber the parameters) and an exact J^T r.
    pub fn normal_equations(&self, r: &DVector<f64>) -> (nalgebra::DMatrix<f64>, DVector<f64>) {
        if r.len() != self.rows {
            return (
                nalgebra::DMatrix::zeros(self.cols, self.cols),
                DVector::zeros(self.cols),
            );
        }
        let mut jtj = nalgebra::DMatrix::zeros(self.cols, self.cols);
        let mut jtr = DVector::zeros(self.cols);
        for row in 0..self.rows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let rv = r[row];
            for a in start..end {
                let ca = self.col_indices[a] as usize;
                let va = self.values[a];
                jtr[ca] += va * rv;
                for b in start..end {
                    let cb = self.col_indices[b] as usize;
                    jtj[(ca, cb)] += va * self.values[b];
                }
            }
        }
        (jtj, jtr)
    }

    /// Materialise this CSR matrix as a dense `DMatrix`.
    ///
    /// Intended for the small/medium systems where a dense normal-equation solve
    /// is still the right algorithm; the caller decides the size threshold.
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut m = nalgebra::DMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let start = self.row_ptr[r] as usize;
            let end = self.row_ptr[r + 1] as usize;
            for i in start..end {
                m[(r, self.col_indices[i] as usize)] = self.values[i];
            }
        }
        m
    }

    pub fn transpose_spmv_ctx(
        &self,
        ctx: &ComputeDevice,
        y: &DVector<f64>,
    ) -> Result<DVector<f64>, String> {
        match ctx {
            ComputeDevice::Cpu(_) | ComputeDevice::Gpu(_) => {
                let mut res = DVector::zeros(self.cols);
                for r in 0..self.rows {
                    let start = self.row_ptr[r] as usize;
                    let end = self.row_ptr[r + 1] as usize;
                    for i in start..end {
                        let c = self.col_indices[i] as usize;
                        res[c] += self.values[i] * y[r];
                    }
                }
                Ok(res)
            }
            ComputeDevice::Mlx(_) => {
                Err("MLX transpose SpMV not implemented yet. Use CPU backend.".into())
            }
        }
    }
}

pub trait LinearSolver {
    fn solve(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, String>;
}

/// Conjugate Gradient solver on GPU/CPU
pub struct CgSolver {
    pub max_iters: usize,
    pub tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_solves_simple_system() {
        // 3x3 SPD tridiagonal matrix: [2,-1,0; -1,2,-1; 0,-1,2]
        // Solve Ax = b with b = [1, 0, 1]
        // Known solution: x = [1, 1, 1] (verify: A*[1,1,1] = [2-1, -1+2-1, -1+2] = [1,0,1])
        let triplets = vec![
            Triplet {
                row: 0,
                col: 0,
                val: 2.0,
            },
            Triplet {
                row: 0,
                col: 1,
                val: -1.0,
            },
            Triplet {
                row: 1,
                col: 0,
                val: -1.0,
            },
            Triplet {
                row: 1,
                col: 1,
                val: 2.0,
            },
            Triplet {
                row: 1,
                col: 2,
                val: -1.0,
            },
            Triplet {
                row: 2,
                col: 1,
                val: -1.0,
            },
            Triplet {
                row: 2,
                col: 2,
                val: 2.0,
            },
        ];

        let a = SparseMatrix::from_triplets(3, 3, &triplets);
        let b = DVector::from_vec(vec![1.0, 0.0, 1.0]);

        let cpu = cv_hal::cpu::CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let solver = CgSolver {
            max_iters: 100,
            tolerance: 1e-10,
        };

        let x = solver
            .solve(&device, &a, &b)
            .expect("CG solver should converge");

        let expected = vec![1.0, 1.0, 1.0];
        for i in 0..3 {
            assert!(
                (x[i] - expected[i]).abs() < 1e-6,
                "x[{}] = {}, expected {}",
                i,
                x[i],
                expected[i]
            );
        }
    }
}

impl LinearSolver for CgSolver {
    fn solve(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, String> {
        let mut x = DVector::zeros(a.cols);
        let mut residual = b - a.spmv_ctx(ctx, &x)?;
        let mut p = residual.clone();
        let mut rsold = residual.dot(&residual);

        for _ in 0..self.max_iters {
            let ap = a.spmv_ctx(ctx, &p)?;
            let pap = p.dot(&ap);
            if pap.abs() < 1e-10 {
                break;
            }
            let alpha = rsold / pap;
            x += alpha * &p;
            residual -= alpha * &ap;

            let rsnew = residual.dot(&residual);
            if rsnew.sqrt() < self.tolerance {
                break;
            }
            p = &residual + (rsnew / rsold) * &p;
            rsold = rsnew;
        }

        Ok(x)
    }
}
