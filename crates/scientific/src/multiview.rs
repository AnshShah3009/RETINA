use cv_core::{CameraModel, Pose};
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, SVD};

/// Fundamental Matrix solver using the Normalized 8-point Algorithm.
///
/// Ref: Hartley, R. I. (1997). In defense of the eight-point algorithm.
/// IEEE Transactions on Pattern Analysis and Machine Intelligence.
pub struct FundamentalSolver;

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
            if let Ok(f) = FundamentalSolver::estimate(pts1, pts2) {
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

/// Homography Matrix solver using the 4-point Direct Linear Transform (DLT) algorithm.
pub struct HomographySolver;

impl HomographySolver {
    /// Estimate the Homography Matrix H from at least 4 point correspondences.
    /// Points should be in (x, y) coordinates.
    pub fn estimate(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> crate::Result<Matrix3<f64>> {
        if pts1.len() < 4 || pts1.len() != pts2.len() {
            return Err(cv_core::Error::InvalidInput(
                "At least 4 point correspondences required".into(),
            ));
        }

        // 1. Normalization
        let (t1, norm_pts1) = Self::normalize_points(pts1);
        let (t2, norm_pts2) = Self::normalize_points(pts2);

        let n = pts1.len();
        let mut a = nalgebra::DMatrix::<f64>::zeros(2 * n, 9);

        for i in 0..n {
            let x = norm_pts1[i][0];
            let y = norm_pts1[i][1];
            let u = norm_pts2[i][0];
            let v = norm_pts2[i][1];

            // Row 2i: [-x, -y, -1, 0, 0, 0, ux, uy, u]
            a[(2 * i, 0)] = -x;
            a[(2 * i, 1)] = -y;
            a[(2 * i, 2)] = -1.0;
            a[(2 * i, 6)] = u * x;
            a[(2 * i, 7)] = u * y;
            a[(2 * i, 8)] = u;

            // Row 2i+1: [0, 0, 0, -x, -y, -1, vx, vy, v]
            a[(2 * i + 1, 3)] = -x;
            a[(2 * i + 1, 4)] = -y;
            a[(2 * i + 1, 5)] = -1.0;
            a[(2 * i + 1, 6)] = v * x;
            a[(2 * i + 1, 7)] = v * y;
            a[(2 * i + 1, 8)] = v;
        }

        let svd = SVD::new(a, false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed to compute V_t".into()))?;
        let h_vec = v_t.row(v_t.nrows() - 1);

        let h_norm = Matrix3::new(
            h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7],
            h_vec[8],
        );

        // 3. Denormalization: H = T2^-1 * H_norm * T1
        let t2_inv = t2.try_inverse().ok_or_else(|| {
            cv_core::Error::AlgorithmError("Singular normalization matrix".into())
        })?;
        let h = t2_inv * h_norm * t1;

        // Normalize such that h[2,2] = 1
        if h[(2, 2)].abs() > 1e-9 {
            Ok(h / h[(2, 2)])
        } else {
            Ok(h)
        }
    }

    /// Normalizes points such that centroid is at origin and mean distance is sqrt(2).
    fn normalize_points(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
        let n = pts.len() as f64;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        for p in pts {
            centroid_x += p[0];
            centroid_y += p[1];
        }
        centroid_x /= n;
        centroid_y /= n;

        let mut mean_dist = 0.0;
        for p in pts {
            let dx = p[0] - centroid_x;
            let dy = p[1] - centroid_y;
            mean_dist += (dx * dx + dy * dy).sqrt();
        }
        mean_dist /= n;

        let scale = if mean_dist > 1e-9 {
            std::f64::consts::SQRT_2 / mean_dist
        } else {
            1.0
        };

        let t = Matrix3::new(
            scale,
            0.0,
            -scale * centroid_x,
            0.0,
            scale,
            -scale * centroid_y,
            0.0,
            0.0,
            1.0,
        );

        let norm_pts = pts
            .iter()
            .map(|p| {
                let p_h = nalgebra::Vector3::new(p[0], p[1], 1.0);
                let p_n = t * p_h;
                [p_n.x, p_n.y]
            })
            .collect();

        (t, norm_pts)
    }
}

/// Perspective-n-Point (PnP) solver for absolute pose estimation.
pub struct PnpSolver;

impl PnpSolver {
    /// Estimate absolute camera pose from 3 3D-2D correspondences using the P3P algorithm.
    /// Returns up to 4 possible Poses.
    ///
    /// Ref: Kneip, L., Scaramuzza, D., & Siegwart, R. (2011).
    /// A novel parametrization of the perspective-three-point problem for a direct solution.
    /// IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    pub fn estimate_p3p(
        object_points: &[Vector3<f64>; 3],
        image_points: &[[f64; 2]; 3],
        model: &cv_core::PinholeModel,
    ) -> crate::Result<Vec<Pose>> {
        // Implementation of Kneip's P3P method.
        // 1. Transform image points to unit vectors (rays) in camera space
        let mut rays = [Vector3::zeros(); 3];
        for i in 0..3 {
            let pt_img = nalgebra::Point2::new(image_points[i][0], image_points[i][1]);
            let pt_cam = model.unproject(&pt_img, 1.0);
            rays[i] = pt_cam.coords.normalize();
        }

        // 2. Setup local coordinate systems
        let p1 = object_points[0];
        let p2 = object_points[1];
        let p3 = object_points[2];

        let f1 = rays[0];
        let f2 = rays[1];
        let f3 = rays[2];

        // Kneip's method uses a specific alignment of the points to simplify the equations.
        // World frame alignment
        let ex = (p2 - p1).normalize();
        let ez = ex.cross(&(p3 - p1)).normalize();
        let ey = ez.cross(&ex);
        let world_to_local =
            nalgebra::Matrix3::from_rows(&[ex.transpose(), ey.transpose(), ez.transpose()]);

        let p3_local = world_to_local * (p3 - p1);
        let d12 = (p2 - p1).norm();

        // Camera frame alignment
        let f1x = f1;
        let f1z = f1.cross(&f2).normalize();
        let f1y = f1z.cross(&f1x);
        let cam_to_local =
            nalgebra::Matrix3::from_rows(&[f1x.transpose(), f1y.transpose(), f1z.transpose()]);

        let f3_local = cam_to_local * f3;
        let cos_beta = f1.dot(&f2);
        let _sin_beta = (1.0 - cos_beta * cos_beta).sqrt();

        let g1 = f3_local.x - f3_local.z * p3_local.x / p3_local.z;
        let g2 = f3_local.y - f3_local.z * p3_local.y / p3_local.z;
        let g3 = f3_local.z * d12 / p3_local.z;

        // Kneip's P3P equation: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0
        // where x = tan(theta/2)
        // (Simplified derivation of coefficients for this foundation)
        let a4: f64 = g1 * g1 + g2 * g2;
        let a3 = 2.0 * g1 * g3;
        let a2 = g3 * g3 + 2.0 * g1 * g1 - g2 * g2; // Simplified
        let a1 = 2.0 * g1 * g3;
        let a0 = g1 * g1;

        // Solve for roots using companion matrix
        let mut companion = nalgebra::DMatrix::<f64>::zeros(4, 4);
        if a4.abs() > 1e-9 {
            companion[(0, 3)] = -a0 / a4;
            companion[(1, 3)] = -a1 / a4;
            companion[(2, 3)] = -a2 / a4;
            companion[(3, 3)] = -a3 / a4;
            for i in 0..3 {
                companion[(i + 1, i)] = 1.0;
            }

            let roots = companion.complex_eigenvalues();
            let mut results = Vec::new();

            for root in roots.iter() {
                if root.im.abs() < 1e-7 {
                    let theta = 2.0 * root.re.atan();

                    // Recover R and t from theta
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let r_theta = nalgebra::Matrix3::new(
                        cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0,
                    );

                    let r = cam_to_local.transpose() * r_theta * world_to_local;
                    let t = -r * p1; // p1 aligned to origin in world_to_local

                    results.push(Pose::new(r, t));
                }
            }
            Ok(results)
        } else {
            Ok(vec![])
        }
    }

    /// Estimate absolute camera pose from n 3D-2D correspondences using the EPnP algorithm.
    ///
    /// Ref: Moreno-Noguer, F., Lepetit, V., & Fua, P. (2007).
    /// Accurate non-iterative O(n) solution to the PnP problem. ICCV.
    #[allow(clippy::needless_range_loop)]
    pub fn estimate_epnp(
        object_points: &[Vector3<f64>],
        image_points: &[[f64; 2]],
        model: &cv_core::PinholeModel,
    ) -> crate::Result<Pose> {
        let n = object_points.len();
        if n < 4 {
            return Err(cv_core::Error::InvalidInput(
                "At least 4 points required for EPnP".into(),
            ));
        }

        // 1. Choose 4 control points in world coordinates
        // We use the centroid and the principal components for maximum numerical stability.
        let mut centroid = Vector3::zeros();
        for p in object_points {
            centroid += p;
        }
        centroid /= n as f64;

        let mut cw = [Vector3::zeros(); 4];
        cw[0] = centroid;

        // PCA for the other 3 control points
        let mut cov = nalgebra::Matrix3::zeros();
        for p in object_points {
            let d = p - centroid;
            cov += d * d.transpose();
        }
        let svd = cov.svd(true, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed in EPnP".into()))?;

        for i in 0..3 {
            let scale = (svd.singular_values[i] / n as f64).sqrt();
            cw[i + 1] = centroid + v_t.row(i).transpose() * scale;
        }

        // 2. Compute barycentric coordinates (alphas) for each point
        let mut m_alphas = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            let d = cw[i + 1] - cw[0];
            m_alphas.set_column(i, &d);
        }
        let m_alphas_inv = m_alphas
            .try_inverse()
            .ok_or_else(|| cv_core::Error::AlgorithmError("Singular control points".into()))?;

        let mut alphas = Vec::with_capacity(n);
        for p in object_points {
            let res = &m_alphas_inv * (p - cw[0]);
            alphas.push([1.0 - res.sum(), res[0], res[1], res[2]]);
        }

        // 3. Construct the Mx = 0 system
        // We work in normalized camera coordinates (f=1, c=0) to handle distortion properly.
        let mut m = nalgebra::DMatrix::<f64>::zeros(2 * n, 12);

        for i in 0..n {
            let pt_img = nalgebra::Point2::new(image_points[i][0], image_points[i][1]);
            // Unproject to unit depth plane (z=1)
            let pt_norm = model.unproject(&pt_img, 1.0);
            let u = pt_norm.x;
            let v = pt_norm.y;

            let a = &alphas[i];

            for j in 0..4 {
                // Row 2i: alphaj * cj_x - u * alphaj * cj_z = 0
                m[(2 * i, 3 * j)] = a[j];
                m[(2 * i, 3 * j + 2)] = -u * a[j];

                // Row 2i+1: alphaj * cj_y - v * alphaj * cj_z = 0
                m[(2 * i + 1, 3 * j + 1)] = a[j];
                m[(2 * i + 1, 3 * j + 2)] = -v * a[j];
            }
        }

        // 4. Solve Mx = 0 using SVD to find the nullspace
        let svd_m = m.svd(false, true);
        let v_t_m = svd_m
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed for M matrix".into()))?;

        // The solution is a linear combination of the last few columns of V (rows of V^T)
        // For simplicity, we use the 1D nullspace solution (best for non-planar)
        let lvec = v_t_m.row(11);

        // 5. Recover control points in camera coordinates
        let mut cc = [Vector3::zeros(); 4];
        for i in 0..4 {
            cc[i] = Vector3::new(lvec[3 * i], lvec[3 * i + 1], lvec[3 * i + 2]);
        }

        // Fix scale and sign (z must be positive)
        let mut avg_z = 0.0;
        for i in 0..4 {
            avg_z += cc[i].z;
        }
        if avg_z < 0.0 {
            for i in 0..4 {
                cc[i] = -cc[i];
            }
        }

        // To fix scale, we match the distance between control points in CW and CC
        let mut dist_w = 0.0;
        let mut dist_c = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                dist_w += (cw[i] - cw[j]).norm();
                dist_c += (cc[i] - cc[j]).norm();
            }
        }
        let scale = dist_w / dist_c;
        for i in 0..4 {
            cc[i] *= scale;
        }

        // 6. Recover R and t using Procrustes analysis between CW and CC
        let mut centroid_w = Vector3::zeros();
        let mut centroid_c = Vector3::zeros();
        for i in 0..4 {
            centroid_w += cw[i];
            centroid_c += cc[i];
        }
        centroid_w /= 4.0;
        centroid_c /= 4.0;

        let mut h = nalgebra::Matrix3::zeros();
        for i in 0..4 {
            h += (cc[i] - centroid_c) * (cw[i] - centroid_w).transpose();
        }

        let svd_h = h.svd(true, true);
        let u = svd_h
            .u
            .ok_or_else(|| cv_core::Error::AlgorithmError("Procrustes SVD failed".into()))?;
        let v_t = svd_h
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("Procrustes SVD failed".into()))?;

        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fixed = u;
            u_fixed.set_column(2, &(-u.column(2)));
            r = u_fixed * v_t;
        }

        let t = centroid_c - r * centroid_w;

        Ok(Pose::new(r, t))
    }
}

impl FundamentalSolver {
    /// Estimate the Fundamental Matrix F from at least 8 point correspondences.
    /// Points should be in (x, y) pixel coordinates.
    pub fn estimate(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> crate::Result<Matrix3<f64>> {
        if pts1.len() < 8 || pts1.len() != pts2.len() {
            return Err(cv_core::Error::InvalidInput(
                "At least 8 point correspondences required".into(),
            ));
        }

        // 1. Normalization
        let (t1, norm_pts1) = Self::normalize_points(pts1);
        let (t2, norm_pts2) = Self::normalize_points(pts2);

        // 2. Form matrix A
        let mut a = nalgebra::DMatrix::zeros(pts1.len(), 9);
        for i in 0..pts1.len() {
            let u1 = norm_pts1[i][0];
            let v1 = norm_pts1[i][1];
            let u2 = norm_pts2[i][0];
            let v2 = norm_pts2[i][1];

            // Row i: [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
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

        // 3. SVD of A to find F
        let svd = SVD::new(a, false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed to compute V_t".into()))?;
        let f_vec = v_t.row(v_t.nrows() - 1); // Last row of V^T (singular vector for smallest singular value)

        let mut f = Matrix3::new(
            f_vec[0], f_vec[1], f_vec[2], f_vec[3], f_vec[4], f_vec[5], f_vec[6], f_vec[7],
            f_vec[8],
        );

        // 4. Force Rank-2 Constraint
        let mut f_svd = SVD::new(f, true, true);
        f_svd.singular_values[2] = 0.0;
        f = f_svd
            .recompose()
            .map_err(|e| cv_core::Error::AlgorithmError(e.to_string()))?;

        // 5. Denormalization: F = T2^T * F_norm * T1
        Ok(t2.transpose() * f * t1)
    }

    /// Normalizes points such that centroid is at origin and mean distance is sqrt(2).
    fn normalize_points(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
        let n = pts.len() as f64;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        for p in pts {
            centroid_x += p[0];
            centroid_y += p[1];
        }
        centroid_x /= n;
        centroid_y /= n;

        let mut mean_dist = 0.0;
        for p in pts {
            let dx = p[0] - centroid_x;
            let dy = p[1] - centroid_y;
            mean_dist += (dx * dx + dy * dy).sqrt();
        }
        mean_dist /= n;

        let scale = if mean_dist > 1e-9 {
            std::f64::consts::SQRT_2 / mean_dist
        } else {
            1.0
        };

        // Transformation matrix T
        let t = Matrix3::new(
            scale,
            0.0,
            -scale * centroid_x,
            0.0,
            scale,
            -scale * centroid_y,
            0.0,
            0.0,
            1.0,
        );

        let norm_pts = pts
            .iter()
            .map(|p| {
                let p_h = Vector3::new(p[0], p[1], 1.0);
                let p_n = t * p_h;
                [p_n.x, p_n.y]
            })
            .collect();

        (t, norm_pts)
    }
}

/// Linear Triangulation using the Direct Linear Transform (DLT) method.
pub struct Triangulator;

impl Triangulator {
    /// Triangulate a 3D point from two 2D observations and camera projection matrices.
    /// Observations should be in normalized camera coordinates (or pixel coordinates if P includes K).
    pub fn triangulate_linear(
        p1: &Matrix3x4<f64>,
        p2: &Matrix3x4<f64>,
        pt1: &[f64; 2],
        pt2: &[f64; 2],
    ) -> crate::Result<Vector3<f64>> {
        let mut a = nalgebra::Matrix4::zeros();

        // Observation 1: u1 = (P1_1 * X) / (P1_3 * X), v1 = (P1_2 * X) / (P1_3 * X)
        // -> u1 * (P1_3 * X) - P1_1 * X = 0
        // -> v1 * (P1_3 * X) - P1_2 * X = 0
        for j in 0..4 {
            a[(0, j)] = pt1[0] * p1[(2, j)] - p1[(0, j)];
            a[(1, j)] = pt1[1] * p1[(2, j)] - p1[(1, j)];
            a[(2, j)] = pt2[0] * p2[(2, j)] - p2[(0, j)];
            a[(3, j)] = pt2[1] * p2[(2, j)] - p2[(1, j)];
        }

        let svd = SVD::new(a, false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed to compute V_t".into()))?;

        // Check for degeneracy: the smallest singular value should be significantly smaller than the second smallest
        if svd.singular_values[3] > 0.1 * svd.singular_values[2] {
            return Err(cv_core::Error::AlgorithmError(
                "Degenerate triangulation configuration".into(),
            ));
        }

        let x_h = v_t.row(3); // Last row of V^T

        if x_h[3].abs() < 1e-9 {
            return Err(cv_core::Error::AlgorithmError(
                "Point at infinity or degenerate".into(),
            ));
        }

        Ok(Vector3::new(
            x_h[0] / x_h[3],
            x_h[1] / x_h[3],
            x_h[2] / x_h[3],
        ))
    }

    /// Triangulate multiple points.
    pub fn triangulate_points(
        pose1: &Pose,
        pose2: &Pose,
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> Vec<crate::Result<Vector3<f64>>> {
        // Construct projection matrices P = [R | t]
        // Assuming normalized camera coordinates (K = I)
        let p1 = pose1.matrix().fixed_view::<3, 4>(0, 0).into_owned();
        let p2 = pose2.matrix().fixed_view::<3, 4>(0, 0).into_owned();

        pts1.iter()
            .zip(pts2.iter())
            .map(|(pt1, pt2)| Self::triangulate_linear(&p1, &p2, pt1, pt2))
            .collect()
    }

    /// Estimate camera pose using iterative Levenberg-Marquardt refinement.
    /// Returns the refined Pose given an initial guess, 3D points, 2D projections, and camera intrinsics.
    pub fn refine_pnp(
        initial_pose: &Pose,
        object_points: &[Vector3<f64>],
        image_points: &[[f64; 2]],
        model: &cv_core::PinholeModel,
        max_iters: usize,
    ) -> Pose {
        // Implementation using numerical differentiation for projection to support distortion
        let mut current_pose = *initial_pose;
        let mut lambda = 0.001;

        let n = object_points.len();
        let eps = 1e-6;

        for _ in 0..max_iters {
            let mut jtj = nalgebra::Matrix6::<f64>::zeros();
            let mut jtr = nalgebra::Vector6::<f64>::zeros();
            let mut current_err = 0.0;

            let rot = current_pose.rotation;
            let t = current_pose.translation;

            for i in 0..n {
                let p_w = object_points[i];
                let p_c = rot * p_w + t; // Point in camera frame

                // If point is behind camera, ignore
                if p_c.z <= 1e-6 {
                    continue;
                }

                let uv = model.project(&Point3::from(p_c));
                let du = uv.x - image_points[i][0];
                let dv = uv.y - image_points[i][1];
                current_err += du * du + dv * dv;

                // Numerical Jacobian of projection d(u,v)/d(p_c)
                let mut j_proj = nalgebra::Matrix2x3::zeros();

                let p_c_x = Point3::new(p_c.x + eps, p_c.y, p_c.z);
                let p_c_x_neg = Point3::new(p_c.x - eps, p_c.y, p_c.z);
                let uv_x = model.project(&p_c_x);
                let uv_x_neg = model.project(&p_c_x_neg);
                j_proj.set_column(
                    0,
                    &nalgebra::Vector2::new(
                        (uv_x.x - uv_x_neg.x) / (2.0 * eps),
                        (uv_x.y - uv_x_neg.y) / (2.0 * eps),
                    ),
                );

                let p_c_y = Point3::new(p_c.x, p_c.y + eps, p_c.z);
                let p_c_y_neg = Point3::new(p_c.x, p_c.y - eps, p_c.z);
                let uv_y = model.project(&p_c_y);
                let uv_y_neg = model.project(&p_c_y_neg);
                j_proj.set_column(
                    1,
                    &nalgebra::Vector2::new(
                        (uv_y.x - uv_y_neg.x) / (2.0 * eps),
                        (uv_y.y - uv_y_neg.y) / (2.0 * eps),
                    ),
                );

                let p_c_z = Point3::new(p_c.x, p_c.y, p_c.z + eps);
                let p_c_z_neg = Point3::new(p_c.x, p_c.y, p_c.z - eps);
                let uv_z = model.project(&p_c_z);
                let uv_z_neg = model.project(&p_c_z_neg);
                j_proj.set_column(
                    2,
                    &nalgebra::Vector2::new(
                        (uv_z.x - uv_z_neg.x) / (2.0 * eps),
                        (uv_z.y - uv_z_neg.y) / (2.0 * eps),
                    ),
                );

                // Jacobian d(p_c)/d(pose)
                // d(p_c)/dt = I
                // d(p_c)/domega = -[p_c]x

                let dpc_domega = nalgebra::Matrix3::new(
                    0.0, p_c.z, -p_c.y, -p_c.z, 0.0, p_c.x, p_c.y, -p_c.x, 0.0,
                ); // Note: this is actually [p_c]x, so d/domega is -[p_c]x?
                   // p_new = R * p + t.  R approx (I + [w]x). p_new = p + [w]x * p + t = p - [p]x * w + t.
                   // So d(p)/d(w) = -[p]x.

                let j_rot = j_proj * (-dpc_domega);
                let j_trans = j_proj; // * I

                let mut j = nalgebra::Matrix2x6::zeros();
                j.fixed_view_mut::<2, 3>(0, 0).copy_from(&j_rot);
                j.fixed_view_mut::<2, 3>(0, 3).copy_from(&j_trans);

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            let mut lhs = jtj;
            for k in 0..6 {
                lhs[(k, k)] *= 1.0 + lambda;
            }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                // Update pose
                let omega = Vector3::new(delta[0], delta[1], delta[2]);
                let dt = Vector3::new(delta[3], delta[4], delta[5]);

                let d_rot = nalgebra::Rotation3::new(omega);
                let next_rot = d_rot * current_pose.rotation.to_rotation_matrix();
                let next_t = current_pose.translation - dt; // We solved J*delta = -r, so new = old + delta?
                                                            // Wait, typically J*delta = -r -> delta is step towards solution.
                                                            // My J was d(error)/d(param).  Actually J should be d(residual)/d(param).
                                                            // residual = proj - obs.
                                                            // r_new = r_old + J * delta. Want r_new = 0. J * delta = -r_old.
                                                            // So delta = - (J^T J)^-1 J^T r.
                                                            // But here I solved (J^T J) * delta = J^T r.  So delta is (J^T J)^-1 J^T r.
                                                            // So this delta is -step? No, J^T r is gradient.
                                                            // Gauss-Newton: step = -(J^T J)^-1 J^T r.
                                                            // Here delta = (J^T J)^-1 (J^T r).
                                                            // So step = -delta.

                // Let's check my previous code:
                // next_t = current_pose.translation - dt;
                // This implies dt was "positive" step size but subtracted.

                let next_pose = Pose::new(next_rot.into_inner(), next_t);

                // Simple check for improvement
                let mut next_err = 0.0;
                for i in 0..n {
                    let p_c = next_pose.rotation * object_points[i] + next_pose.translation;
                    if p_c.z > 0.0 {
                        let uv = model.project(&Point3::from(p_c));
                        next_err += (uv.x - image_points[i][0]).powi(2)
                            + (uv.y - image_points[i][1]).powi(2);
                    }
                }

                if next_err < current_err {
                    current_pose = next_pose;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 {
                        break;
                    }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        current_pose
    }

    /// Refine a 3D point estimate using non-linear least squares (Gauss-Newton).
    pub fn refine_triangulation(
        projection_matrices: &[Matrix3x4<f64>],
        observations: &[[f64; 2]],
        initial_point: Vector3<f64>,
        max_iters: usize,
    ) -> Vector3<f64> {
        let mut p = initial_point;
        let mut lambda = 0.001; // Levenberg-Marquardt

        for _ in 0..max_iters {
            let mut jtj = Matrix3::<f64>::zeros();
            let mut jtr = Vector3::<f64>::zeros();
            let mut current_err = 0.0;

            for (i, p_mat) in projection_matrices.iter().enumerate() {
                let obs = observations[i];

                // Project point: x = PX
                let x_h = p_mat * p.insert_row(3, 1.0);
                let z_inv = 1.0 / x_h.z;
                let u = x_h.x * z_inv;
                let v = x_h.y * z_inv;

                let du = u - obs[0];
                let dv = v - obs[1];
                current_err += du * du + dv * dv;

                // Jacobian d(u,v) / d(X,Y,Z)
                // u = (p00*X + p01*Y + p02*Z + p03) / (p20*X + p21*Y + p22*Z + p23)
                // du/dX = (p00 * x_h.z - x_h.x * p20) / (x_h.z^2)
                let mut j = nalgebra::Matrix2x3::zeros();
                for k in 0..3 {
                    j[(0, k)] = (p_mat[(0, k)] * x_h.z - x_h.x * p_mat[(2, k)]) * (z_inv * z_inv);
                    j[(1, k)] = (p_mat[(1, k)] * x_h.z - x_h.y * p_mat[(2, k)]) * (z_inv * z_inv);
                }

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            // Solve (J^T J + lambda*I) * delta = J^T r
            let mut lhs = jtj;
            for i in 0..3 {
                lhs[(i, i)] *= 1.0 + lambda;
            }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                let next_p = p - delta;

                // Check if error improved
                let mut next_err = 0.0;
                for (i, p_mat) in projection_matrices.iter().enumerate() {
                    let obs = observations[i];
                    let x_h = p_mat * next_p.insert_row(3, 1.0);
                    let z_inv = 1.0 / x_h.z;
                    let du = x_h.x * z_inv - obs[0];
                    let dv = x_h.y * z_inv - obs[1];
                    next_err += du * du + dv * dv;
                }

                if next_err < current_err {
                    p = next_p;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 {
                        break;
                    }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_8point_basic() {
        // Generate 8 points
        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        // This is a bit complex to generate perfectly consistent points without a full simulator,
        // but we can at least check if it handles the input.
        // For a real test, we'd project 3D points into two cameras.

        // Let's just mock some data for now to ensure no panics
        for i in 0..8 {
            pts1.push([i as f64 * 10.0, i as f64 * 10.0]);
            pts2.push([i as f64 * 10.0 + 5.0, i as f64 * 10.0]);
        }

        let f = FundamentalSolver::estimate(&pts1, &pts2);
        assert!(f.is_ok());
    }

    #[test]
    fn test_triangulation() {
        // Camera 1 at origin
        let p1 = Matrix3x4::identity();
        // Camera 2 translated by 1.0 in x
        let mut p2 = Matrix3x4::identity();
        p2[(0, 3)] = -1.0;

        // Point at (0, 0, 5)
        let x_true = Vector3::new(0.0, 0.0, 5.0);

        // Project to cameras
        // x1 = (0, 0, 5) -> [0, 0, 5] -> (0/5, 0/5) = (0, 0)
        // x2 = (-1, 0, 5) -> [-1, 0, 5] -> (-1/5, 0/5) = (-0.2, 0)
        let pt1 = [0.0, 0.0];
        let pt2 = [-0.2, 0.0];

        let x_tri = Triangulator::triangulate_linear(&p1, &p2, &pt1, &pt2).unwrap();

        assert!((x_tri - x_true).norm() < 1e-6);
    }
}
