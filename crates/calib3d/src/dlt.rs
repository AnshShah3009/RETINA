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
    // `!mean_dist.is_finite()` is the load-bearing half.
    //
    // A NaN coordinate makes `mean_dist` NaN, and `NaN <= 1e-12` is **false** -
    // so the guard passes and an all-NaN normalisation matrix goes into LAPACK's
    // SVD, whose convergence test is a comparison that can never become true.
    // The call then does not return at all: `calibrate_camera_planar` with one
    // NaN observation was killed at 300 s and again at 120 s, and an 8x9 matrix
    // with a single NaN did not finish the SVD in 120 s on its own.
    //
    // A hang is worse than a wrong answer, because nothing reports it.
    if !mean_dist.is_finite() || mean_dist <= 1e-12 {
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
///
/// `pub(crate)` because `essential_fundamental` has the same `m x 9` design
/// matrix and the same defect: it read `v_t.row(v_t.nrows() - 1)` directly, which
/// is row 7 - not the null vector - when `m == 8`.
pub(crate) fn smallest_right_singular_vector(a: DMatrix<f64>) -> Option<DVector<f64>> {
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

/// Smallest ratio `sigma_k / sigma_1` that a normalised DLT design matrix must
/// meet for the 9-vector it solves for to be *uniquely* determined.
///
/// # Why the 9-vector, and why this form of the test
///
/// Both the homography design matrix (`2N x 9`) and the fundamental one
/// (`N x 9`) encode a homogeneous system `A h = 0` in **9 unknowns**. A
/// homogeneous system always has at least the trivial solution, so `A` is
/// *always* singular and its smallest singular value is *always* ~0 - that is
/// the null direction we are looking for, not a symptom of anything. What
/// decides whether the answer is meaningful is whether the null space is
/// **one-dimensional**: that requires `A` to have rank **8**, i.e. `sigma_8`
/// to be genuinely separated from zero.
///
/// So the criterion is `sigma_8 / sigma_1` of the design matrix, i.e. the 8th
/// largest singular value relative to the largest. Measured on this workspace's
/// own configurations:
///
/// | configuration                                | `sigma_8 / sigma_1` |
/// |----------------------------------------------|--------------------|
/// | 3 of 4 source points collinear               | `0.0`              |
/// | all 4 source points collinear                | `0.0`              |
/// | 4 of 5 source points collinear               | `0.0`              |
/// | 4-point trapezium with a 3 px sliver (valid)  | `1.3e-3`           |
/// | 4-point image-corner quad (valid)            | `2.4e-1`           |
/// | 12-point grid (valid)                        | `2.9e-1`           |
///
/// With 3 collinear points the design matrix drops to rank 7 (null space
/// 2-dimensional) and the homography is a 1-parameter family: the vanishing
/// point of the line is never observed, so *every* member fits the
/// correspondences equally well. The SVD still returns a unit-norm vector, so
/// the caller gets `Ok` with a plausible-looking matrix; in the measured
/// degenerate cases its `(2,2)` entry came out ~1e-14, which is exactly why the
/// `h[(2,2)].abs() > 1e-12` normalisation guard falls through and hands back the
/// unscaled rank-1 matrix that projects every input point to infinity.
///
/// # Why `sigma_8 / sigma_1` and not an absolute cutoff
///
/// A singular value scales linearly with `A`, so any absolute threshold is
/// really a test on the *units of the input*, not on its geometry. That is the
/// same mistake the earlier `qr_solve` defect made (see `cv_math::linalg`: an
/// absolute `1e-14` on `R`'s diagonal reported a perfectly well-conditioned
/// system rank-deficient merely for being scaled down by `1e-14`); the fix
/// there and here is the same shape - test **relative** to `sigma_1`. A
/// Hartley-normalised design matrix has `sigma_1 = O(1)` anyway, which is the
/// whole point of normalising, but the relative form keeps the predicate
/// correct if that ever changes. Measured scale sweep over the same geometry at
/// `1e-6`, `1e-3`, `1`, `1e3`, `1e6`: `sigma_8/sigma_1` moves by under 0.4 %,
/// i.e. it is scale-invariant to f64 noise as intended.
///
/// # Why `1e-9`
///
/// The null vector returned by the SVD is perturbed by roughly
/// `eps / (sigma_8 / sigma_1)`, so `1e-9` buys ~7 significant digits in the
/// homography - three orders of magnitude beyond pixel accuracy at any image
/// size, and eight orders of magnitude above the `0.0` that exact degeneracy
/// produces. It also sits ~6 orders of magnitude *below* the skinniest valid
/// configuration measured here (`1.3e-3`), so it does not reject real input
/// just for being awkward. `tests/degenerate_homography.rs` pins both sides of
/// the boundary.
pub const DLT_RANK_TOLERANCE: f64 = 1e-9;

/// [`smallest_right_singular_vector`], but `None` unless the design matrix's
/// null space is exactly one-dimensional.
///
/// This is the homography solver's degeneracy gate; see [`DLT_RANK_TOLERANCE`]
/// for why `sigma_8 / sigma_1` is the right quantity and why it is relative.
///
/// Kept separate from [`smallest_right_singular_vector`] rather than folded
/// into it because that helper is `pub(crate)` and shared with
/// `essential_fundamental`, whose rank requirement is *different* - see
/// [`solve_dlt_fundamental`] for the measurements behind that difference.
fn unique_null_vector(a: DMatrix<f64>) -> Option<DVector<f64>> {
    let a = if a.nrows() < 9 {
        let mut padded = DMatrix::<f64>::zeros(9, 9);
        padded.view_mut((0, 0), (a.nrows(), 9)).copy_from(&a);
        padded
    } else {
        a
    };
    let svd = SVD::new(a, false, true);
    let v_t = svd.v_t?;
    let singular_values = svd.singular_values;

    // nalgebra returns singular values in descending order, so `len - 2` is the
    // 8th largest of the (at most 9) values, and `len - 1` is the null
    // direction itself - which is always ~0 and carries no information.
    let n_sv = singular_values.len();
    let sigma_max = singular_values[0];
    let sigma_8 = if n_sv >= 2 {
        singular_values[n_sv - 2]
    } else {
        0.0
    };

    // `sigma_8 / sigma_1`: relative to the largest singular value, so the test
    // is invariant to the overall scale of the input.
    let ratio = if sigma_max > 0.0 {
        sigma_8 / sigma_max
    } else {
        0.0
    };
    // `!(x > t)` rather than `x <= t` so a NaN ratio also fails closed.
    if !(ratio > DLT_RANK_TOLERANCE) {
        return None;
    }

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

    // Reject non-finite input *before* the SVD.
    //
    // A design matrix built from NaN propagates into LAPACK's bidiagonalisation,
    // whose convergence test is a comparison - and every comparison against NaN
    // is false, so it never terminates. `calibrate_camera_planar` feeding this a
    // single NaN observation did not return at all; it hung the whole workspace
    // test suite for 360 s.
    //
    // Returning `None` is the honest answer: a homography through a point at
    // infinity does not exist, and the caller already treats `None` as "this
    // view cannot be used".
    if src.iter().flatten().any(|v| !v.is_finite()) || dst.iter().flatten().any(|v| !v.is_finite())
    {
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

    // 3. The homography is the null vector of A - but only if A has rank 8, so
    // that the null space is one-dimensional. Without this gate a collinear or
    // near-collinear set returns a unit-norm vector from a 2-dimensional null
    // space, i.e. an arbitrary member of a 1-parameter family of homographies
    // that all fit the correspondences equally well.
    let h = unique_null_vector(a)?;
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
///
/// # What it refuses, and why those are exactly the right refusals
///
/// The projection `U diag(s1, s2, 0) V^T` is well-defined for *any* 3x3 matrix,
/// so this function could "succeed" on everything. It reports `None` only for
/// the inputs on which it has nothing to do, or where success would be a lie:
///
/// * **non-finite entries.** LAPACK's bidiagonalisation decides convergence by
///   comparison, and every comparison against NaN is false, so the SVD does not
///   terminate. Checked *before* the decomposition - see
///   `cv_math::linalg::is_finite` for the recorded hang.
/// * **the zero matrix** (`s1 == 0`): there is nothing to project onto, and
///   returning an all-zero "rank-2 matrix" invites the caller to divide by it.
/// * **rank <= 1** (`s2 / s1` below tolerance). Here `s3` is already 0, so the
///   projection is a *no-op*: it returns the rank-1 input unchanged. A caller
///   checking `is_some()` would conclude it had been handed a rank-2 matrix.
///   Measured on `[[1,2,3],[2,4,6],[3,6,9]]`, which used to come back with
///   norm 14 and still rank 1.
///
///   This is load-bearing for `solve_dlt_fundamental`: it is what makes a
///   collinear-in-one-image sample return `None` instead of a rank-1 matrix
///   that "fits" it with exactly zero residual while encoding no epipolar
///   geometry. See that function's docs for the full argument.
///
/// # What it deliberately does *not* refuse
///
/// **An input that is already rank 2** - which is the normal case, and the one
/// that matters most. `solve_dlt_fundamental` finds a null vector of a design
/// matrix of rank 8, so the space of solutions is spanned by the fundamental
/// matrix, which is rank 2 by construction. Measured singular values of the
/// un-projected `F`: `[5.33e-2, 4.06e-4, 0.0e0]` - exactly rank 2, with
/// `s3 == 0` already. An earlier draft of this function rejected that, and the
/// entire 8-point solver returned `None` for every input; `s3` says nothing
/// about whether rank enforcement *worked*. The discriminating quantity is
/// `s2`: nonzero means there is a second direction to keep, zero means the input
/// had nothing to project onto.
///
/// The tolerance is **relative** (`s2 / s1 > 1e-9`) for the same reason as
/// [`DLT_RANK_TOLERANCE`]: an absolute cutoff tests the units of the input
/// rather than its geometry.
pub fn enforce_rank2(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let sv = &svd.singular_values;
    let sigma_max = sv[0];
    if !(sigma_max > 0.0) {
        // Zero matrix: there is nothing to project onto.
        return None;
    }
    // Rank <= 1: the projection would be a no-op, so `Some` would be a lie.
    if !(sv[1] > DLT_RANK_TOLERANCE * sigma_max) {
        return None;
    }
    let sigma = Matrix3::new(sv[0], 0.0, 0.0, 0.0, sv[1], 0.0, 0.0, 0.0, 0.0);
    Some(u * sigma * v_t)
}

/// Estimate the fundamental matrix from `>= 8` point correspondences with the
/// normalised 8-point algorithm, enforcing the rank-2 constraint.
///
/// Returns `None` for fewer than 8 pairs, for mismatched slice lengths, or when
/// normalisation/rank enforcement fails.
///
/// # The degeneracy here is *not* the homography one - do not copy the H rule
///
/// `solve_dlt_homography` gates on `sigma_8 / sigma_1` of its design matrix
/// because 4-point minimal samples are so thin that ordinary scenes walk into
/// rank deficiency. The 8-point fundamental solver has none of that problem -
/// with 8 correspondences the design matrix has 8 rows for 9 unknowns, so rank
/// 8 (null space 1-dimensional) is the *generic* case and needs no gate at all.
/// Measured rank of the design matrix `A` over this workspace's configurations:
///
/// | configuration                          | rank of `A` | `solve_dlt_fundamental` |
/// |----------------------------------------|-------------|--------------------------|
/// | general 8-point                        | 8           | `Some`, `sigma_8/sigma_1` = `0.0` |
/// | general 9- and 20-point                | 8           | `Some`                   |
/// | 3 collinear in image 1 + 5 general     | 8           | `Some`                   |
/// | **all 8 collinear in image 1**         | **6**       | `None`                   |
/// | **true 3D line, 8 and 20 points**      | **3-6**     | `None`                   |
/// | **all 8 on one epipolar line (img 2)** | **6**       | `None`                   |
/// | coplanar scene, 8 and 20 points         | 6           | `Some`                   |
///
/// Two things follow, and they are different answers to different questions.
///
/// **`sigma_8 / sigma_1` is vacuous here.** For a rank-8 `A` the 8th singular
/// value *is* the null direction, so the homography criterion reads `0.0` for
/// every healthy input. Copying the H gate would either reject everything or
/// nothing.
///
/// **The real F degeneracies are far worse than "collinear", and they are caught
/// by `enforce_rank2`, not by a rank test on `A`.** It is tempting to assume
/// collinear points in one image are fine for `F` - scene points on a 3D line
/// (a rail, a road edge) do project to a line in both images, and the epipolar
/// line is exactly what `F` predicts. Measurement says otherwise: for 8 points
/// collinear in image 1 the design matrix has rank 6, and the DLT's answer comes
/// out **rank 1**, which `enforce_rank2` now refuses. That refusal is correct,
/// and it is worth spelling out why a rank-1 `F = a b^T` is worthless here
/// rather than merely imprecise:
///
/// ```text
/// x2' F x1 = x2' a · (b' x1)
/// ```
///
/// Every `x1` lies on the line `b' x = 0`, so `b' x1 = 0` for all of them and
/// the residual is **exactly zero for any `a`, `b` whatsoever**. A rank-1 `F`
/// therefore "satisfies" a collinear-in-image-1 sample by construction, while
/// encoding no epipolar geometry at all - it merely restates that the points
/// are collinear. Before this guard the solver returned such a matrix with a
/// measured residual of `8.9e-16`, i.e. a perfect-looking score on information
/// the score cannot see. Returning `None` is the honest answer.
///
/// Note that this is *not* "collinearity is a degeneracy for F": a set with
/// **3** collinear points in image 1 still has rank 8 and is still solved (see
/// the table), because the 3-dof family the collinearity introduces is removed
/// again by the remaining general points and the rank-2 constraint.
///
/// **Coplanar scenes are deliberately still accepted.** Rank 6, yet a valid `F`
/// comes out: the correspondences do determine a fundamental matrix that
/// satisfies them, they just do not determine it *uniquely*. Refusing would
/// break `calibrate_stereo_rectify`-style paths and
/// `find_essential_mat_ransac`'s initialisation, which share this step, and it
/// would blank a solver that cannot tell a 20-point coplanar tracking sequence
/// from a 20-point general one. Disambiguating planarity needs a consumer that
/// can see more than one sample; see `essential_fundamental`, whose RANSAC loop
/// over the 5-point algorithm is the right place for it.
pub fn solve_dlt_fundamental(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    if pts1.len() < 8 || pts1.len() != pts2.len() {
        return None;
    }

    // Same exposure as the homography solver: a NaN point set reaches LAPACK's
    // bidiagonalisation, whose convergence test never becomes true, so the solve
    // does not return.
    if pts1.iter().flatten().any(|v| !v.is_finite())
        || pts2.iter().flatten().any(|v| !v.is_finite())
    {
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
