use nalgebra::Matrix3;

/// Homography Matrix solver using the 4-point Direct Linear Transform (DLT) algorithm.
pub struct HomographySolver;

impl HomographySolver {
    /// Estimate the Homography Matrix H from at least 4 point correspondences.
    /// Points should be in (x, y) coordinates.
    ///
    /// Delegates to [`crate::dlt::solve_dlt_homography`], the single normalised
    /// DLT implementation in the workspace.
    pub fn estimate(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> crate::Result<Matrix3<f64>> {
        if pts1.len() < 4 || pts1.len() != pts2.len() {
            return Err(cv_core::Error::InvalidInput(
                "At least 4 point correspondences required".into(),
            ));
        }

        crate::dlt::solve_dlt_homography(pts1, pts2).ok_or_else(|| {
            cv_core::Error::AlgorithmError(
                "Homography DLT failed (degenerate configuration)".into(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn apply(h: &Matrix3<f64>, p: [f64; 2]) -> [f64; 2] {
        let v = h * Vector3::new(p[0], p[1], 1.0);
        [v[0] / v[2], v[1] / v[2]]
    }

    #[test]
    fn estimate_recovers_known_homography() {
        let h = Matrix3::new(
            1.1, 0.05, 120.0, //
            -0.02, 0.95, 90.0, //
            0.0001, 0.0002, 1.0,
        );
        let src: Vec<[f64; 2]> = (0..8)
            .map(|i| [30.0 + (i % 4) as f64 * 150.0, 20.0 + (i / 4) as f64 * 180.0])
            .collect();
        let dst: Vec<[f64; 2]> = src.iter().map(|p| apply(&h, *p)).collect();

        let est = HomographySolver::estimate(&src, &dst).expect("homography");
        let est = est / est[(2, 2)];
        assert!((est - h / h[(2, 2)]).norm() < 1e-9);
    }

    /// Exactly 4 correspondences — the minimal sample. This used to return a
    /// wrong matrix because nalgebra's thin SVD dropped the null vector.
    #[test]
    fn estimate_is_exact_for_four_correspondences() {
        let h = Matrix3::new(
            1.2, 0.1, 300.0, //
            -0.05, 0.9, 220.0, //
            0.0002, 0.0001, 1.0,
        );
        let src = [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]];
        let dst: Vec<[f64; 2]> = src.iter().map(|p| apply(&h, *p)).collect();

        let est = HomographySolver::estimate(&src, &dst).expect("homography");
        let est = est / est[(2, 2)];
        assert!((est - h / h[(2, 2)]).norm() < 1e-9);
    }

    #[test]
    fn estimate_rejects_degenerate_input() {
        assert!(HomographySolver::estimate(&[[0.0, 0.0]; 3], &[[0.0, 0.0]; 3]).is_err());
        // All points coincident: the normalisation is undefined.
        assert!(HomographySolver::estimate(&[[1.0, 1.0]; 4], &[[2.0, 2.0]; 4]).is_err());
    }
}
