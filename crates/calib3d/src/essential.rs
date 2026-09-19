use nalgebra::{Matrix3, Vector3, SVD};

/// Essential Matrix solver using Nistér's 5-point Algorithm.
pub struct EssentialSolver;

impl EssentialSolver {
    /// Estimate Essential Matrix E from 5 point correspondences.
    /// Points must be in normalized camera coordinates (K-inv * pixels).
    pub fn estimate_5point(
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> crate::Result<Vec<Matrix3<f64>>> {
        if pts1.len() < 5 || pts1.len() != pts2.len() {
            return Err(cv_core::Error::InvalidInput(
                "Exactly 5 points required for 5-point algorithm".into(),
            ));
        }

        // 1. Form 5x9 matrix A
        let mut a = nalgebra::DMatrix::<f64>::zeros(pts1.len(), 9);
        for i in 0..pts1.len() {
            let u1 = pts1[i][0];
            let v1 = pts1[i][1];
            let u2 = pts2[i][0];
            let v2 = pts2[i][1];

            a[(i, 0)] = u2 * u1;
            a[(i, 1)] = u2 * v1;
            a[(i, 2)] = u2;
            a[(i, 3)] = v2 * u1;
            a[(i, 4)] = v2 * v1;
            a[(i, 5)] = v2;
            a[(i, 6)] = u1;
            a[(i, 7)] = v1;
            a[(i, 8)] = 1.0;
        }

        // 2. Find nullspace (4 basis matrices E1, E2, E3, E4)
        let svd = SVD::new(a, false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed to compute V_t".into()))?;
        let rows = v_t.nrows();

        // The nullspace is spanned by the last 4 rows of V^T (columns of V)
        // Since we likely have 5 points, rows=9. The nullspace is dim 4 (9-5).
        // The last 4 rows correspond to the smallest singular values.
        let get_e = |row_idx: usize| {
            let r = v_t.row(row_idx);
            Matrix3::new(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
        };

        let e1 = get_e(rows - 1);
        let e2 = get_e(rows - 2);
        let e3 = get_e(rows - 3);
        let e4 = get_e(rows - 4);

        // 3. Build 10x20 constraint matrix M
        let mut m = nalgebra::DMatrix::<f64>::zeros(10, 20);
        Self::build_constraint_matrix(&mut m, &e1, &e2, &e3, &e4);

        // 4. Gauss-Jordan elimination
        Self::gauss_jordan(&mut m);

        // 5. Construct action matrix for variable z and solve
        let roots = Self::solve_polynomial_system(&m);

        let mut results = Vec::new();
        for z in roots {
            // Recover x, y for each z
            if let Some(e) = Self::recover_e(&m, z, &e1, &e2, &e3, &e4) {
                results.push(e);
            }
        }

        // Fallback for planar scenes or degenerate cases
        if results.is_empty() && pts1.len() >= 8 {
            if let Ok(f) = crate::FundamentalSolver::estimate(pts1, pts2) {
                // Ensure E is valid Essential Matrix (singular values [s, s, 0])
                let svd_f = f.svd(true, true);
                let u = svd_f.u.unwrap_or(Matrix3::identity());
                let vt = svd_f.v_t.unwrap_or(Matrix3::identity());
                let s = (svd_f.singular_values[0] + svd_f.singular_values[1]) / 2.0;
                let e_proj = u * Matrix3::from_diagonal(&Vector3::new(s, s, 0.0)) * vt;
                return Ok(vec![e_proj]);
            }
        }

        Ok(results)
    }

    fn build_constraint_matrix(
        m: &mut nalgebra::DMatrix<f64>,
        e1: &Matrix3<f64>,
        e2: &Matrix3<f64>,
        e3: &Matrix3<f64>,
        e4: &Matrix3<f64>,
    ) {
        let basis = [e1, e2, e3, e4];

        let get_det_coeff = |i: usize, j: usize, k: usize| {
            let mut val = 0.0;
            let a = basis[i];
            let b = basis[j];
            let c = basis[k];
            for p in 0..3 {
                for q in 0..3 {
                    for r in 0..3 {
                        if (p != q) && (q != r) && (p != r) {
                            let sgn = if (q as i32 - p as i32)
                                * (r as i32 - q as i32)
                                * (p as i32 - r as i32)
                                > 0
                            {
                                1.0
                            } else {
                                -1.0
                            };
                            val += sgn * a[(0, p)] * b[(1, q)] * c[(2, r)];
                        }
                    }
                }
            }
            val
        };

        // Monomials: x=0, y=1, z=2, 1=3
        // Order: x^3, y^3, x^2y, xy^2, x^2z, xyz, y^2z, xz^2, yz^2, z^3, x^2, xy, y^2, xz, yz, z^2, x, y, z, 1
        let monomials = [
            (0, 0, 0),
            (1, 1, 1),
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 1, 2),
            (0, 2, 2),
            (1, 2, 2),
            (2, 2, 2), // Eliminated (0-9)
            (0, 0, 3),
            (0, 1, 3),
            (1, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (2, 2, 3),
            (0, 3, 3),
            (1, 3, 3),
            (2, 3, 3),
            (3, 3, 3), // Basis (10-19)
        ];

        // Row 0: det(E) = 0
        for (idx, &(i, j, k)) in monomials.iter().enumerate() {
            let sum_coeff = if i == j && j == k {
                get_det_coeff(i, j, k)
            } else if i == j || j == k || i == k {
                get_det_coeff(i, j, k) + get_det_coeff(j, k, i) + get_det_coeff(k, i, j)
            } else {
                get_det_coeff(0, 1, 2)
                    + get_det_coeff(0, 2, 1)
                    + get_det_coeff(1, 0, 2)
                    + get_det_coeff(1, 2, 0)
                    + get_det_coeff(2, 0, 1)
                    + get_det_coeff(2, 1, 0)
            };
            m[(0, idx)] = sum_coeff;
        }

        let get_trace_coeff = |i: usize, j: usize, k: usize, r: usize, c: usize| {
            let term1 = 2.0 * (basis[i] * basis[j].transpose() * basis[k])[(r, c)];
            let term2 = (basis[i] * basis[j].transpose()).trace() * basis[k][(r, c)];
            term1 - term2
        };

        // Rows 1-9: 2EE^T E - trace(EE^T)E = 0
        for r in 0..3 {
            for c in 0..3 {
                let row_idx = 1 + r * 3 + c;
                if row_idx >= 10 {
                    break;
                }

                for (idx, &(i, j, k)) in monomials.iter().enumerate() {
                    let mut sum_coeff = 0.0;
                    if i == j && j == k {
                        sum_coeff = get_trace_coeff(i, j, k, r, c);
                    } else if i == j || j == k || i == k {
                        sum_coeff = get_trace_coeff(i, j, k, r, c)
                            + get_trace_coeff(j, k, i, r, c)
                            + get_trace_coeff(k, i, j, r, c);
                    } else {
                        let perms = [
                            (i, j, k),
                            (i, k, j),
                            (j, i, k),
                            (j, k, i),
                            (k, i, j),
                            (k, j, i),
                        ];
                        for p in perms {
                            sum_coeff += get_trace_coeff(p.0, p.1, p.2, r, c);
                        }
                    }
                    m[(row_idx, idx)] = sum_coeff;
                }
            }
        }
    }

    fn gauss_jordan(m: &mut nalgebra::DMatrix<f64>) {
        let (rows, cols) = m.shape();
        let mut pivot_row = 0;
        for j in 0..cols {
            if pivot_row >= rows {
                break;
            }
            let mut best_i = pivot_row;
            for i in pivot_row + 1..rows {
                if m[(i, j)].abs() > m[(best_i, j)].abs() {
                    best_i = i;
                }
            }
            if m[(best_i, j)].abs() < 1e-12 {
                continue;
            }
            m.swap_rows(pivot_row, best_i);

            let factor = 1.0 / m[(pivot_row, j)];
            for col in j..cols {
                m[(pivot_row, col)] *= factor;
            }

            for i in 0..rows {
                if i != pivot_row {
                    let f = m[(i, j)];
                    for col in j..cols {
                        m[(i, col)] -= f * m[(pivot_row, col)];
                    }
                }
            }
            pivot_row += 1;
        }
    }

    fn solve_polynomial_system(m: &nalgebra::DMatrix<f64>) -> Vec<f64> {
        // Construct Action Matrix B for variable z.
        // B maps basis vector v to z*v.
        // Basis V: [x^2, xy, y^2, xz, yz, z^2, x, y, z, 1]
        // Indices in M columns 10-19:
        // 10:x^2, 11:xy, 12:y^2, 13:xz, 14:yz, 15:z^2, 16:x, 17:y, 18:z, 19:1

        // We need z * basis element expressed in basis elements.
        // z * x^2 = x^2z -> Index 4 in elim monomials.
        // z * xy  = xyz  -> Index 5 in elim monomials.
        // z * y^2 = y^2z -> Index 6 in elim monomials.
        // z * xz  = xz^2 -> Index 7 in elim monomials.
        // z * yz  = yz^2 -> Index 8 in elim monomials.
        // z * z^2 = z^3  -> Index 9 in elim monomials.
        // z * x   = xz   -> Basis Index 3 (col 13)
        // z * y   = yz   -> Basis Index 4 (col 14)
        // z * z   = z^2  -> Basis Index 5 (col 15)
        // z * 1   = z    -> Basis Index 8 (col 18)

        // The first 6 involve eliminated monomials. We use the reduced M to substitute.
        // M has form [I | B_coefs].  Eliminated = -B_coefs * Basis.
        // So x^2z = -sum(m[4, 10+k] * basis[k])

        let mut action = nalgebra::DMatrix::<f64>::zeros(10, 10);

        // Map from Eliminated monomial index (0-9) to Row in M (0-9)
        // Since we did full Gauss-Jordan, row i corresponds to eliminated monomial i.

        // 1. z * x^2 -> x^2z (Index 4)
        for k in 0..10 {
            action[(0, k)] = -m[(4, 10 + k)];
        }
        // 2. z * xy -> xyz (Index 5)
        for k in 0..10 {
            action[(1, k)] = -m[(5, 10 + k)];
        }
        // 3. z * y^2 -> y^2z (Index 6)
        for k in 0..10 {
            action[(2, k)] = -m[(6, 10 + k)];
        }
        // 4. z * xz -> xz^2 (Index 7)
        for k in 0..10 {
            action[(3, k)] = -m[(7, 10 + k)];
        }
        // 5. z * yz -> yz^2 (Index 8)
        for k in 0..10 {
            action[(4, k)] = -m[(8, 10 + k)];
        }
        // 6. z * z^2 -> z^3 (Index 9)
        for k in 0..10 {
            action[(5, k)] = -m[(9, 10 + k)];
        }

        // 7. z * x -> xz (Basis 3)
        action[(6, 3)] = 1.0;
        // 8. z * y -> yz (Basis 4)
        action[(7, 4)] = 1.0;
        // 9. z * z -> z^2 (Basis 5)
        action[(8, 5)] = 1.0;
        // 10. z * 1 -> z (Basis 8)
        action[(9, 8)] = 1.0;

        let decomp = action.complex_eigenvalues();
        let mut roots = Vec::new();
        for val in decomp.iter() {
            if val.im.abs() < 1e-6 {
                roots.push(val.re);
            }
        }
        roots
    }

    fn recover_e(
        m: &nalgebra::DMatrix<f64>,
        z: f64,
        e1: &Matrix3<f64>,
        e2: &Matrix3<f64>,
        e3: &Matrix3<f64>,
        e4: &Matrix3<f64>,
    ) -> Option<Matrix3<f64>> {
        // We have z. To find x and y, we can solve the linear system from the basis relations.
        // B * V = z * V  => (B - zI) * V = 0.
        // V is the eigenvector. But we know the last element of V is 1.

        // Alternatively, use the polynomial relations directly from rows of M.
        // We need x and y.
        // Basis V: [x^2, xy, y^2, xz, yz, z^2, x, y, z, 1]
        // We know z. So xz, yz, z^2, z, 1 are known.
        // We can find x and y from xz/z or yz/z? Unstable if z ~ 0.

        // Better: Solve the linear system for the nullspace of (B - zI).
        let mut action = nalgebra::DMatrix::<f64>::zeros(10, 10);
        // Reconstruct Action Matrix (same as above)
        for k in 0..10 {
            action[(0, k)] = -m[(4, 10 + k)];
        }
        for k in 0..10 {
            action[(1, k)] = -m[(5, 10 + k)];
        }
        for k in 0..10 {
            action[(2, k)] = -m[(6, 10 + k)];
        }
        for k in 0..10 {
            action[(3, k)] = -m[(7, 10 + k)];
        }
        for k in 0..10 {
            action[(4, k)] = -m[(8, 10 + k)];
        }
        for k in 0..10 {
            action[(5, k)] = -m[(9, 10 + k)];
        }
        action[(6, 3)] = 1.0;
        action[(7, 4)] = 1.0;
        action[(8, 5)] = 1.0;
        action[(9, 8)] = 1.0;

        for i in 0..10 {
            action[(i, i)] -= z;
        }

        let svd = action.svd(false, true);
        if let Some(v_t) = svd.v_t {
            // Null vector is the last row of V^T
            let null_vec = v_t.row(9);
            // Basis V: [x^2, xy, y^2, xz, yz, z^2, x, y, z, 1]
            // Scale such that last element is 1
            if null_vec[9].abs() > 1e-8 {
                let scale = 1.0 / null_vec[9];
                let x = null_vec[6] * scale;
                let y = null_vec[7] * scale;
                return Some(x * e1 + y * e2 + z * e3 + e4);
            }
        }

        None
    }
}
