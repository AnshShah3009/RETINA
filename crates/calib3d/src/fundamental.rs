use nalgebra::Matrix3;

/// Fundamental Matrix solver using the Normalized 8-point Algorithm.
///
/// Ref: Hartley, R. I. (1997). In defense of the eight-point algorithm.
/// IEEE Transactions on Pattern Analysis and Machine Intelligence.
pub struct FundamentalSolver;

impl FundamentalSolver {
    /// Estimate the Fundamental Matrix F from at least 8 point correspondences.
    /// Points should be in (x, y) pixel coordinates.
    ///
    /// Delegates to [`crate::dlt::solve_dlt_fundamental`], the single normalised
    /// 8-point implementation in the workspace (also used by
    /// [`crate::find_fundamental_mat`] and `cv-features`' RANSAC estimator).
    pub fn estimate(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> crate::Result<Matrix3<f64>> {
        if pts1.len() < 8 || pts1.len() != pts2.len() {
            return Err(cv_core::Error::InvalidInput(
                "At least 8 point correspondences required".into(),
            ));
        }

        crate::dlt::solve_dlt_fundamental(pts1, pts2).ok_or_else(|| {
            cv_core::Error::AlgorithmError("Fundamental 8-point solve failed".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Vector3};

    /// Two pinhole cameras looking at deterministic random 3D points.
    fn synthetic_correspondences(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let k = Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0);
        let angle = 0.15f64;
        let r = Matrix3::new(
            angle.cos(),
            0.0,
            angle.sin(),
            0.0,
            1.0,
            0.0,
            -angle.sin(),
            0.0,
            angle.cos(),
        );
        let t = Vector3::new(-0.5, 0.0, 0.0);

        let mut s = 99u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        let mut i = 0usize;
        while pts1.len() < n && i < 10 * n {
            i += 1;
            let x = Vector3::new(next() * 4.0, next() * 4.0, 5.0 + next() * 3.0);
            let y = r * x + t;
            let u = k * x;
            let v = k * y;
            pts1.push([u[0] / u[2], u[1] / u[2]]);
            pts2.push([v[0] / v[2], v[1] / v[2]]);
        }
        (pts1, pts2)
    }

    fn worst_epipolar(f: &Matrix3<f64>, pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> f64 {
        let mut worst = 0.0f64;
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = Vector3::new(p1[0], p1[1], 1.0);
            let x2 = Vector3::new(p2[0], p2[1], 1.0);
            worst = worst.max(x2.dot(&(f * x1)).abs());
        }
        worst
    }

    #[test]
    fn estimate_satisfies_the_epipolar_constraint() {
        let (pts1, pts2) = synthetic_correspondences(12);
        let f = FundamentalSolver::estimate(&pts1, &pts2).expect("F");
        assert!(worst_epipolar(&f, &pts1, &pts2) < 1e-6);

        let sv = f.svd(false, false).singular_values;
        assert!(sv[2] < 1e-9 * sv[0], "F is not rank 2: {sv:?}");
    }

    /// Exactly 8 correspondences (the minimal sample) used to be solved with the
    /// wrong right singular vector, giving a visibly non-epipolar F.
    #[test]
    fn estimate_is_consistent_for_the_minimal_eight_point_sample() {
        let (pts1, pts2) = synthetic_correspondences(8);
        let f = FundamentalSolver::estimate(&pts1, &pts2).expect("F");
        assert!(
            worst_epipolar(&f, &pts1, &pts2) < 1e-6,
            "minimal-sample F is not epipolar"
        );
    }

    /// The three previously-separate solvers must agree up to sign and scale.
    #[test]
    fn estimate_agrees_with_find_fundamental_mat() {
        let (pts1, pts2) = synthetic_correspondences(12);

        let f1 = FundamentalSolver::estimate(&pts1, &pts2).expect("F");
        let p1: Vec<Point2<f64>> = pts1.iter().map(|p| Point2::new(p[0], p[1])).collect();
        let p2: Vec<Point2<f64>> = pts2.iter().map(|p| Point2::new(p[0], p[1])).collect();
        let f2 = crate::find_fundamental_mat(&p1, &p2).expect("F");

        let unit = |m: &Matrix3<f64>| m / m.norm();
        let direct = (unit(&f1) - unit(&f2)).norm();
        let flipped = (unit(&f1) + unit(&f2)).norm();
        assert!(
            direct < 1e-9 || flipped < 1e-9,
            "solvers disagree: {direct} / {flipped}"
        );
    }
}
