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
    /// Solve `A x = b` to this solver's tolerance.
    ///
    /// Returns `Err` when the tolerance was **not** met. That is deliberate: an
    /// unconverged `x` is not a solution to `A x = b`, and a caller that needs a
    /// solution must not receive one that is quietly wrong.
    fn solve(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, String>;

    /// Solve `A x = b`, returning the iterate actually reached and whether it met
    /// the tolerance.
    ///
    /// This exists because `solve`'s contract is wrong for one caller. An
    /// optimizer does not need a *solution* - it needs a usable step, and it judges
    /// steps by whether the cost decreases, which is a stronger test than the
    /// linear residual. An approximate iterate from conjugate gradient on a
    /// symmetric positive-definite system is still a descent direction.
    ///
    /// Without this, that caller has exactly two options when `solve` returns `Err`:
    /// abandon an optimisation that is working, or substitute something
    /// fabricated. `sfm`'s Levenberg-Marquardt step chose the latter and
    /// substituted a **zero step** - which is also precisely what a *rejected* step
    /// looks like, so a solver failure and an unproductive step became
    /// indistinguishable, and the loop reported "stalled" when the truth was that
    /// its linear solver had failed. Measured on a real system the residual was
    /// `1.301e-7` against a `1e-10` tolerance: a perfectly usable step, thrown away.
    ///
    /// The default implementation delegates to `solve`, so a solver that cannot
    /// report a partial result keeps its previous behaviour.
    fn solve_relaxed(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<(DVector<f64>, bool), String> {
        self.solve(ctx, a, b).map(|x| (x, true))
    }
}

/// Conjugate Gradient solver on GPU/CPU
pub struct CgSolver {
    pub max_iters: usize,
    pub tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tridiagonal SPD system of size `n` (a 1-D Laplacian), which is what a
    /// bundle-adjustment normal-equations matrix looks like at low rank.
    fn laplacian(n: usize) -> SparseMatrix {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet {
                row: i,
                col: i,
                val: 2.0,
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
        SparseMatrix::from_triplets(n, n, &triplets)
    }

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

    /// An iteration cap is not convergence. Measured on the old code: a 200x200
    /// Laplacian with `max_iters = 10, tolerance = 1e-10` returned `Ok` with
    /// `||Ax - b|| = 1.28e2` — twelve orders of magnitude above the tolerance it
    /// was asked for, and the LM step that consumes it gets a silently wrong
    /// direction.
    #[test]
    /// The iterate reached must be available to a caller that can use it.
    ///
    /// `solve` rejecting a non-converged result is correct for a caller that needs
    /// a *solution* to `A x = b`. An optimizer needs a *step*, and it judges steps
    /// by whether the cost decreases - a stronger test than the linear residual.
    /// Before `solve_relaxed` existed, the caller's only options were to abandon a
    /// working optimisation or fabricate, and `sfm`'s Levenberg-Marquardt step
    /// fabricated a zero step.
    ///
    /// **Measured on the system LM actually solves**, `J^T J + lambda*diag(J^T J)`
    /// with `lambda = 10`, 200x200, and only ONE allowed iteration:
    ///
    /// ```text
    ///  k   damped iterate ||Ax-b||   the zero step ||A*0-b||
    ///  1             7.03e-2                      1.41e1
    ///  2             3.20e-3                      1.41e1
    ///  5             3.02e-7                      1.41e1
    /// ```
    ///
    /// A single iteration is already 200x better than the zero step that replaced
    /// it. That is the case this fix is about.
    ///
    /// An earlier version of this test used an *undamped* Laplacian and asserted the
    /// same property, which is **false**: on that system CG's residual 2-norm grows
    /// from `1.41e1` to `1.28e2` over ten iterations, so the iterate is worse than
    /// zero by that criterion. Measuring it also showed the implementation matches
    /// an independently written dense reference CG to the last digit at every
    /// iteration, so that is CG's nature and not a bug - and it is precisely why the
    /// linear residual is the wrong gate for an optimizer step. Damping is what
    /// makes the system well-conditioned; without it the iteration is being asked to
    /// do something it was never going to do in ten steps.
    #[test]
    fn cg_relaxed_solve_hands_back_the_iterate_it_reached() {
        let n = 200usize;
        let mut a = laplacian(n);
        a.scale_diagonal(1.0 + 10.0);
        let b = DVector::from_element(n, 1.0);
        let cpu = cv_hal::cpu::CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let solver = CgSolver {
            max_iters: 1,
            tolerance: 1e-12,
        };

        // One iteration cannot reach 1e-12, so `solve` must refuse it...
        assert!(
            solver.solve(&device, &a, &b).is_err(),
            "a single iteration cannot meet 1e-12 on this system"
        );

        // ...while the relaxed form hands back what it did reach, flagged as not
        // converged.
        let (x, converged) = solver
            .solve_relaxed(&device, &a, &b)
            .expect("the relaxation must not turn a usable iterate into an error");
        assert!(
            !converged,
            "the same budget cannot have converged, or the two entry points disagree"
        );
        assert!(
            x.iter().all(|v| v.is_finite()),
            "the iterate must be finite; a partial solve is still a step"
        );

        // THE POINT OF THE FIX: this iterate is worth far more than the zero step
        // substituted for it. If it were not, discarding it would have cost nothing.
        let residual_of = |v: &DVector<f64>| (&a.spmv_native(v).unwrap() - &b).norm();
        let reached = residual_of(&x);
        let zero = residual_of(&DVector::zeros(x.len()));
        assert!(
            reached < zero * 0.1,
            "one damped CG iteration must be far better than the zero step that used \
             to replace it: ||Ax-b|| {reached:.6e} vs ||A*0-b|| {zero:.6e}"
        );
    }

    #[test]
    fn cg_iteration_cap_is_not_reported_as_a_solution() {
        let a = laplacian(200);
        let b = DVector::from_element(200, 1.0);
        let cpu = cv_hal::cpu::CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let solver = CgSolver {
            max_iters: 10,
            tolerance: 1e-10,
        };
        match solver.solve(&device, &a, &b) {
            Ok(x) => {
                let residual = (&a.spmv_native(&x).unwrap() - &b).norm();
                panic!("unconverged solve returned Ok with ||Ax - b|| = {residual:.3e}");
            }
            Err(e) => assert!(e.contains("did not converge"), "unexpected error: {e}"),
        }
    }

    /// Control: the same system with a realistic budget must still solve, and the
    /// returned vector must satisfy the residual bound it was given.
    #[test]
    fn cg_converges_with_an_adequate_budget() {
        let a = laplacian(200);
        let b = DVector::from_element(200, 1.0);
        let cpu = cv_hal::cpu::CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let solver = CgSolver {
            max_iters: 2000,
            tolerance: 1e-10,
        };
        let x = solver.solve(&device, &a, &b).expect("must converge");
        let residual = (&a.spmv_native(&x).unwrap() - &b).norm();
        assert!(
            residual < 1e-10,
            "returned solution violates its own tolerance: {residual:.3e}"
        );
    }
}

impl CgSolver {
    /// One implementation of conjugate gradient, shared by both entry points.
    ///
    /// Returns the iterate reached and its residual norm, so `solve` can decide
    /// whether to call it a solution and `solve_relaxed` can hand it back
    /// regardless. Duplicating the loop would let the two drift apart.
    fn run(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<(DVector<f64>, f64), String> {
        let mut x = DVector::zeros(a.cols);
        let mut residual = b - a.spmv_ctx(ctx, &x)?;
        let mut p = residual.clone();
        let mut rsold = residual.dot(&residual);

        for _ in 0..self.max_iters {
            let ap = a.spmv_ctx(ctx, &p)?;
            let pap = p.dot(&ap);
            // `p·Ap <= 0` means `A` has no energy along `p`: CG cannot continue.
            // This is the *only* breakdown criterion, and it is scale-free.
            //
            // The loop used to carry an extra `pap.abs() < 1e-10` test, which
            // looked like a guard against dividing by ~0 but was not one. `p·Ap`
            // is quadratic in the problem's units, so a fixed threshold fires
            // long before the system is actually exhausted whenever the matrix
            // is small - and it fires at a different *iteration* for every
            // overall scale, so the answer depended on the units the residual
            // happened to be measured in.
            //
            // Measured on `A = diag(1..40) * s`, `b = A·1`, i.e. an SPD system
            // whose exact solution is `x = 1` (|x-1| is the whole error):
            //
            // ```text
            //     s     broken out at   ||x - 1||    never breaks out: |x - 1|
            //   1e0       iter 33          5.6e-08        iter 39, then diverges
            //   1e-3      iter  1          1.2e-02        3.9e-11
            //   1e-5      iter  1          3.2e+00        3.9e-06
            //   1e-6      iter  1          6.3e+00        diverges
            // ```
            //
            // At `s = 1e-5` the very first iteration has `p·Ap = 6.7e-10`, the
            // "breakdown" test fires, and a matrix that is `10^5` times further
            // from singular than the one it handled is abandoned at
            // |x-1| = 3.2 - a *worse* answer than the zero step, which is what
            // the caller was told was the thing being avoided. `A·p = 0` and a
            // vanishing-but-positive `p·Ap` are entirely different situations
            // and only the first one is breakdown; dividing by the second gives
            // a large step in a direction `A` has not yet cancelled, which is
            // what the next iteration needs in order to finish.
            //
            // If the step length or the iterate itself does become non-finite,
            // stop there rather than propagating it: the final residual check
            // decides whether the result counts as a solution.
            if pap <= 0.0 {
                // Breakdown: `A` has no energy along `p`. This is not
                // convergence, so let the final residual check decide.
                break;
            }
            let alpha = rsold / pap;
            x += alpha * &p;
            residual -= alpha * &ap;

            let rsnew = residual.dot(&residual);
            if !rsnew.is_finite() {
                break;
            }
            if rsnew.sqrt() < self.tolerance {
                return Ok((x, rsnew.sqrt()));
            }
            p = &residual + (rsnew / rsold) * &p;
            rsold = rsnew;
        }

        let final_residual = residual.norm();
        Ok((x, final_residual))
    }
}

impl LinearSolver for CgSolver {
    fn solve(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, String> {
        let (x, residual) = self.run(ctx, a, b)?;
        if residual < self.tolerance {
            return Ok(x);
        }
        // The iteration cap is not convergence, and a caller that needs a
        // solution must not be handed one that is quietly wrong. Callers that can
        // use an approximate step ask for it via `solve_relaxed`.
        Err(format!(
            "conjugate gradient did not converge in {} iterations (tolerance {:.3e}): \
             ||A x - b|| = {:.3e}",
            self.max_iters, self.tolerance, residual
        ))
    }

    fn solve_relaxed(
        &self,
        ctx: &ComputeDevice,
        a: &SparseMatrix,
        b: &DVector<f64>,
    ) -> Result<(DVector<f64>, bool), String> {
        let (x, residual) = self.run(ctx, a, b)?;
        Ok((x, residual < self.tolerance))
    }
}
