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
        // Convert COO (triplets) to CSR
        let mut row_counts = vec![0; rows];
        for t in triplets {
            row_counts[t.row] += 1;
        }

        let mut row_ptr = vec![0; rows + 1];
        for i in 0..rows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        let mut current_row_pos = vec![0; rows];
        let mut col_indices = vec![0; triplets.len()];
        let mut values = vec![0.0; triplets.len()];

        for t in triplets {
            let pos = row_ptr[t.row] as usize + current_row_pos[t.row];
            col_indices[pos] = t.col as u32;
            values[pos] = t.val;
            current_row_pos[t.row] += 1;
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
        // Convert f64 to f32 for GPU SpMV
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let values_f32: Vec<f32> = self.values.iter().map(|&v| v as f32).collect();

        let x_tensor: cv_core::CpuTensor<f32> =
            Tensor::from_vec(x_f32, cv_core::TensorShape::new(1, x.len(), 1))
                .map_err(|e| format!("SpMV input tensor creation failed: {}", e))?;

        // SpMV always returns a result tensor on the same device as input x
        match ctx {
            ComputeDevice::Gpu(gpu) => {
                use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
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
                let res_cpu = ctx
                    .spmv(&self.row_ptr, &self.col_indices, &values_f32, &x_tensor)
                    .map_err(|e| format!("CPU SpMV failed: {}", e))?;
                Ok(DVector::from_vec(
                    res_cpu
                        .as_slice()
                        .map_err(|e| format!("Data not on CPU: {}", e))?
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                ))
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
