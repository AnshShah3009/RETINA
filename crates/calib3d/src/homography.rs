use nalgebra::{Matrix3, SVD};

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
