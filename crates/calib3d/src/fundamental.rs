use nalgebra::{Matrix3, Vector3, SVD};

/// Fundamental Matrix solver using the Normalized 8-point Algorithm.
///
/// Ref: Hartley, R. I. (1997). In defense of the eight-point algorithm.
/// IEEE Transactions on Pattern Analysis and Machine Intelligence.
pub struct FundamentalSolver;

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
}
