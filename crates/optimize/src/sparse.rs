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
            ComputeDevice::Cpu(_cpu) => {
                // Native f64 SpMV: downcasting matrix and vector to f32 here
                // loses ~7 significant digits and stalls CG convergence when
                // solving LM normal equations.
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
            ComputeDevice::Mlx(_) => Err("MLX SpMV not implemented yet. Use CPU backend.".into()),
        }
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
