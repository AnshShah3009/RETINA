use nalgebra::Matrix3;

/// Essential Matrix solver using Nistér's 5-point Algorithm.
pub struct EssentialSolver;

impl EssentialSolver {
    /// Estimate an essential matrix from five point correspondences.
    ///
    /// The points must be paired normalized camera coordinates.  A verified
    /// Nistér implementation is not currently available in this crate: the
    /// polynomial basis, coefficient expansion, and numerical root recovery
    /// must be supplied together and validated before returning matrices.
    /// Silently returning an unrelated eight-point estimate would be unsafe.
    pub fn estimate_5point(
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> crate::Result<Vec<Matrix3<f64>>> {
        if pts1.len() != 5 || pts2.len() != 5 {
            return Err(cv_core::Error::InvalidInput(
                "Exactly 5 point pairs required for the 5-point algorithm".into(),
            ));
        }
        if pts1
            .iter()
            .chain(pts2.iter())
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(cv_core::Error::InvalidInput(
                "5-point correspondences must be finite".into(),
            ));
        }

        Err(cv_core::Error::AlgorithmError(
            "Nistér 5-point solver is not implemented: a verified implementation must \
             supply the consistent cubic basis, coefficient expansion, and real-root \
             recovery"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::EssentialSolver;

    #[test]
    fn test_estimate_5point_requires_exactly_five_points() {
        // The solver is a minimal five-point method.  Do not accept a larger
        // sample and do not fall back to a different estimator.
        for count in [0_usize, 1, 2, 3, 4, 6, 8, 20] {
            let points = vec![[0.1, 0.2]; count];
            assert!(EssentialSolver::estimate_5point(&points, &points).is_err());
        }

        let five = vec![[0.1, 0.2]; 5];
        let six = vec![[0.1, 0.2]; 6];
        assert!(EssentialSolver::estimate_5point(&five, &six).is_err());
        assert!(EssentialSolver::estimate_5point(&five, &[[f64::NAN, 0.0]; 5]).is_err());
    }

    #[test]
    fn test_estimate_5point_fails_explicitly_until_verified() {
        let points = vec![
            [0.1, 0.2],
            [0.4, -0.1],
            [-0.3, 0.5],
            [0.7, 0.3],
            [-0.6, -0.4],
        ];
        let result = EssentialSolver::estimate_5point(&points, &points);
        let error = result.expect_err("unimplemented solver must not return matrices");
        let message = error.to_string();
        assert!(message.contains("Nistér"));
        assert!(message.contains("not implemented"));
    }
}
