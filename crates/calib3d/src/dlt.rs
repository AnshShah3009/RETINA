//! Normalised Direct Linear Transform estimators.
//!
//! This module owns the workspace's single implementations of
//!
//! * **Hartley normalisation** — translate so the centroid is at the origin and
//!   scale so the mean point distance from it is `sqrt(2)`,
//! * the **normalised DLT for homographies**, and
//! * the **normalised 8-point algorithm** for fundamental matrices.
//!
//! Every other DLT/normalisation copy in the workspace delegates here:
//! [`crate::homography::HomographySolver`],
//! [`crate::fundamental::FundamentalSolver`],
//! [`crate::essential_fundamental::find_fundamental_mat`] (8-point part),
//! [`crate::calibration`]'s planar homographies and image-normalisation helper,
//! [`crate::pnp`]'s Hartley transform, and `cv-features`' RANSAC estimators.

use nalgebra::{DMatrix, DVector, Matrix3, SVD};

/// Hartley normalisation: translate so the centroid is at the origin, then
/// scale so the mean distance of the points from it is `sqrt(2)`.
///
/// Returns the normalising transform `T` (so that `p_norm = T · p_hom`) and the
/// normalised points, or `None` when the transform is undefined — an empty
/// input, or points that are all coincident (mean distance `<= 1e-12`).
pub fn hartley_normalize(pts: &[[f64; 2]]) -> Option<(Matrix3<f64>, Vec<[f64; 2]>)> {
    if pts.is_empty() {
        return None;
    }
    let n = pts.len() as f64;
    let mean_x = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let mean_dist = pts
        .iter()
        .map(|p| ((p[0] - mean_x).powi(2) + (p[1] - mean_y).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    if mean_dist <= 1e-12 {
        return None;
    }

    let scale = std::f64::consts::SQRT_2 / mean_dist;
    let t = Matrix3::new(
        scale,
        0.0,
        -scale * mean_x,
        0.0,
        scale,
        -scale * mean_y,
        0.0,
        0.0,
        1.0,
    );
    let normalized = pts
        .iter()
        .map(|p| [(p[0] - mean_x) * scale, (p[1] - mean_y) * scale])
        .collect();
    Some((t, normalized))
}

/// Right singular vector belonging to the smallest singular value of the
/// `m x 9` design matrix `a`.
///
/// nalgebra's thin SVD only returns `min(m, n)` right singular vectors, so for
/// the minimal-sample case (`m < 9`, i.e. exactly 4 homography or 8 fundamental
/// correspondences) the null vector the DLT needs is *not* among them and
/// `v_t.row(v_t.nrows() - 1)` returns a vector with a strictly positive
/// `‖A v‖`. Padding the system with (zero) rows up to `9 x 9` makes the null
/// vector part of `v_t`, which is what the pre-consolidation `cv-features`
/// RANSAC solver did and what every path here now does.
fn smallest_right_singular_vector(a: DMatrix<f64>) -> Option<DVector<f64>> {
    let a = if a.nrows() < 9 {
        let mut padded = DMatrix::<f64>::zeros(9, 9);
        padded.view_mut((0, 0), (a.nrows(), 9)).copy_from(&a);
        padded
    } else {
        a
    };
    let v_t = SVD::new(a, false, true).v_t?;
    Some(v_t.row(v_t.nrows() - 1).transpose().into_owned())
}

/// Estimate the homography `H` with `dst ~ H · src` from `>= 4`
/// correspondences using the normalised DLT.
///
/// Returns `None` for fewer than 4 pairs, for mismatched slice lengths, or when
/// normalisation/denormalisation is impossible.
pub fn solve_dlt_homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }

    // 1. Hartley normalisation of both point sets.
    let (t1, n1) = hartley_normalize(src)?;
    let (t2, n2) = hartley_normalize(dst)?;

    // 2. Design matrix A h = 0.
    let n = src.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let x = n1[i][0];
        let y = n1[i][1];
        let u = n2[i][0];
        let v = n2[i][1];
        let r0 = 2 * i;
        let r1 = r0 + 1;

        // Row 2i:   [-x, -y, -1, 0, 0, 0, ux, uy, u]
        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;

        // Row 2i+1: [0, 0, 0, -x, -y, -1, vx, vy, v]
        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }

    // 3. The homography is the null vector of A.
    let h = smallest_right_singular_vector(a)?;
    let h_norm = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);

    // 4. Denormalisation: H = T2^-1 · H_norm · T1.
    let t2_inv = t2.try_inverse()?;
    let h = t2_inv * h_norm * t1;

    if h[(2, 2)].abs() > 1e-12 {
        Some(h / h[(2, 2)])
    } else {
        Some(h)
    }
}

/// Enforce the rank-2 constraint on a 3x3 matrix by zeroing its smallest
/// singular value.
pub fn enforce_rank2(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let sigma = Matrix3::new(
        svd.singular_values[0],
        0.0,
        0.0,
        0.0,
        svd.singular_values[1],
        0.0,
        0.0,
        0.0,
        0.0,
    );
    Some(u * sigma * v_t)
}

/// Estimate the fundamental matrix from `>= 8` point correspondences with the
/// normalised 8-point algorithm, enforcing the rank-2 constraint.
///
/// Returns `None` for fewer than 8 pairs, for mismatched slice lengths, or when
/// normalisation/rank enforcement fails.
pub fn solve_dlt_fundamental(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    if pts1.len() < 8 || pts1.len() != pts2.len() {
        return None;
    }

    let (t1, n1) = hartley_normalize(pts1)?;
    let (t2, n2) = hartley_normalize(pts2)?;

    let n = pts1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = n1[i][0];
        let y1 = n1[i][1];
        let x2 = n2[i][0];
        let y2 = n2[i][1];
        // Row i: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }

    let f = smallest_right_singular_vector(a)?;
    let f0 = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
    let f_rank2 = enforce_rank2(&f0)?;

    // Denormalisation: F = T2^T · F_norm · T1.
    Some(t2.transpose() * f_rank2 * t1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// A known homography with a perspective term.
    fn known_homography() -> Matrix3<f64> {
        Matrix3::new(
            1.2, 0.1, 300.0, //
            -0.05, 0.9, 220.0, //
            0.0002, 0.0001, 1.0,
        )
    }

    fn apply(h: &Matrix3<f64>, p: [f64; 2]) -> [f64; 2] {
        let v = h * Vector3::new(p[0], p[1], 1.0);
        [v[0] / v[2], v[1] / v[2]]
    }

    /// Deterministic uniform noise in `[-0.5, 0.5]`.
    struct Noise(u64);
    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        }
    }

    /// The unified solver must reproduce a known homography from noisy
    /// correspondences, including the minimal (4 point) case.
    #[test]
    fn unified_dlt_reproduces_known_homography_from_noisy_correspondences() {
        let h = known_homography();

        // 12 correspondences spread over a VGA-sized image, ±0.5 px noise.
        let src: Vec<[f64; 2]> = (0..12)
            .map(|i| [40.0 + (i % 4) as f64 * 180.0, 30.0 + (i / 4) as f64 * 190.0])
            .collect();
        let mut noise = Noise(0x5eed);
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                let q = apply(&h, *p);
                [q[0] + noise.next(), q[1] + noise.next()]
            })
            .collect();

        let est = solve_dlt_homography(&src, &dst).expect("homography");
        let est = est / est[(2, 2)];
        let reference = h / h[(2, 2)];

        // Linear part within 1%, projective row within 1e-5, and the
        // translation within a pixel (it carries the ±0.5 px input noise).
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!(
                (est[(i, j)] - reference[(i, j)]).abs() < 1e-2,
                "entry ({i}, {j}): {} vs {}",
                est[(i, j)],
                reference[(i, j)]
            );
        }
        for (i, j) in [(2, 0), (2, 1)] {
            assert!(
                (est[(i, j)] - reference[(i, j)]).abs() < 1e-5,
                "projective entry ({i}, {j}): {} vs {}",
                est[(i, j)],
                reference[(i, j)]
            );
        }
        assert!(
            (est[(0, 2)] - reference[(0, 2)]).abs() < 1.0
                && (est[(1, 2)] - reference[(1, 2)]).abs() < 1.0,
            "translation drifted: {est} vs {reference}"
        );

        // The recovered homography transfers the (noise-free) source points to
        // sub-pixel accuracy, which is the accuracy RANSAC thresholds rely on.
        let mut worst = 0.0f64;
        for p in src.iter() {
            let q = apply(&h, *p);
            let e = apply(&est, *p);
            worst = worst.max(((q[0] - e[0]).powi(2) + (q[1] - e[1]).powi(2)).sqrt());
        }
        assert!(worst < 1.0, "transfer error {worst} px");
    }

    /// Exactly 4 correspondences: the minimal-sample case, which the thin SVD
    /// used to solve with the wrong right singular vector.
    #[test]
    fn unified_dlt_is_exact_for_the_minimal_four_point_sample() {
        let h = known_homography();
        let src = [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]];
        let dst: Vec<[f64; 2]> = src.iter().map(|p| apply(&h, *p)).collect();

        let est = solve_dlt_homography(&src, &dst).expect("homography");
        let est = est / est[(2, 2)];
        let reference = h / h[(2, 2)];
        assert!(
            (est - reference).norm() < 1e-9,
            "4-point DLT is not exact:\n{est}\nexpected\n{reference}"
        );
    }

    /// Norm-sensitive input: large pixel coordinates must not degrade the solve.
    #[test]
    fn unified_dlt_is_stable_for_large_coordinates() {
        let h = known_homography();
        let src: Vec<[f64; 2]> = (0..10)
            .map(|i| {
                [
                    2000.0 + (i % 5) as f64 * 1500.0,
                    1500.0 + (i / 5) as f64 * 1200.0,
                ]
            })
            .collect();
        let dst: Vec<[f64; 2]> = src.iter().map(|p| apply(&h, *p)).collect();

        let est = solve_dlt_homography(&src, &dst).expect("homography");
        let est = est / est[(2, 2)];
        assert!((est - h / h[(2, 2)]).norm() < 1e-8);
    }

    /// Hartley normalisation: centroid at the origin, mean distance sqrt(2),
    /// and `T` consistent with the returned points.
    #[test]
    fn hartley_normalize_centres_and_scales() {
        let pts = [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]];
        let (t, n) = hartley_normalize(&pts).expect("normalisation");
        let mean = n.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0], a[1] + p[1]]);
        assert!((mean[0] / 4.0).abs() < 1e-12 && (mean[1] / 4.0).abs() < 1e-12);
        let mean_dist = n
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
            .sum::<f64>()
            / 4.0;
        assert!((mean_dist - std::f64::consts::SQRT_2).abs() < 1e-12);

        for (p, q) in pts.iter().zip(n.iter()) {
            let v = t * Vector3::new(p[0], p[1], 1.0);
            assert!((v[0] - q[0]).abs() < 1e-12 && (v[1] - q[1]).abs() < 1e-12);
        }

        // Degenerate inputs report failure instead of an arbitrary transform.
        assert!(hartley_normalize(&[]).is_none());
        assert!(hartley_normalize(&[[1.0, 1.0], [1.0, 1.0]]).is_none());
    }

    /// The 8-point solver must produce a rank-2 matrix that satisfies the
    /// epipolar constraint, including for exactly 8 (the minimal sample).
    #[test]
    fn unified_eight_point_fundamental_is_rank_two_and_consistent() {
        let (pts1, pts2) = synthetic_correspondences(8);
        let f = solve_dlt_fundamental(&pts1, &pts2).expect("fundamental");

        let sv = f.svd(false, false).singular_values;
        assert!(sv[2] < 1e-9 * sv[0], "F is not rank 2: {sv:?}");

        let mut worst = 0.0f64;
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = Vector3::new(p1[0], p1[1], 1.0);
            let x2 = Vector3::new(p2[0], p2[1], 1.0);
            worst = worst.max(x2.dot(&(f * x1)).abs());
        }
        assert!(worst < 1e-6, "epipolar residual {worst}");
    }

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

        let mut noise = Noise(0xabcdef);
        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        let mut i = 0usize;
        while pts1.len() < n && i < 10 * n {
            i += 1;
            let x = Vector3::new(
                noise.next() * 4.0,
                noise.next() * 4.0,
                5.0 + noise.next() * 3.0,
            );
            let y = r * x + t;
            let u = k * x;
            let v = k * y;
            if u[2].abs() < 1e-9 || v[2].abs() < 1e-9 {
                continue;
            }
            pts1.push([u[0] / u[2], u[1] / u[2]]);
            pts2.push([v[0] / v[2], v[1] / v[2]]);
        }
        (pts1, pts2)
    }
}
