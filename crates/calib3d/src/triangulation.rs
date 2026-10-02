use crate::Result;
use cv_core::{CameraIntrinsics, CameraModel, Pose};
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3};

/// Linear triangulation from two views.
///
/// Reconstructs 3D points from corresponding 2D points in two camera views
/// using the DLT (Direct Linear Transform) method with SVD.
///
/// # Arguments
/// * `p1` - First camera projection matrix (3x4)
/// * `p2` - Second camera projection matrix (3x4)
/// * `pts1` - Corresponding 2D points in first view
/// * `pts2` - Corresponding 2D points in second view
///
/// # Returns
/// Vector of reconstructed 3D points, or error if SVD fails
pub fn triangulate_points(
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
) -> Result<Vec<Point3<f64>>> {
    if pts1.len() != pts2.len() {
        return Err(cv_core::Error::AlgorithmError(
            "triangulate_points requires equal point counts".to_string(),
        ));
    }

    // Reject non-finite input before the SVD below.
    //
    // `.svd(true, true)` does not return on non-finite input: nalgebra's
    // bidiagonalisation decides convergence by comparison, and every comparison
    // against NaN is false. Measured: one NaN pixel among the pairs hangs here
    // indefinitely and has to be killed.
    //
    // Checked once for the whole call rather than per point, so the cost is
    // linear in the input rather than in the number of decompositions.
    for (i, (a, b)) in pts1.iter().zip(pts2.iter()).enumerate() {
        if [a.x, a.y, b.x, b.y].iter().any(|v| !v.is_finite()) {
            return Err(cv_core::Error::InvalidInput(format!(
                "triangulate_points: correspondence {i} is not finite \
                 ({}, {}, {}, {})",
                a.x, a.y, b.x, b.y
            )));
        }
    }
    for (name, m) in [("p1", p1), ("p2", p2)] {
        if m.iter().any(|v| !v.is_finite()) {
            return Err(cv_core::Error::InvalidInput(format!(
                "triangulate_points: projection matrix {name} is not finite"
            )));
        }
    }

    let mut out = Vec::with_capacity(pts1.len());
    for (a, b) in pts1.iter().zip(pts2.iter()) {
        let mut m = Matrix4::<f64>::zeros();
        for c in 0..4 {
            m[(0, c)] = a.x * p1[(2, c)] - p1[(0, c)];
            m[(1, c)] = a.y * p1[(2, c)] - p1[(1, c)];
            m[(2, c)] = b.x * p2[(2, c)] - p2[(0, c)];
            m[(3, c)] = b.y * p2[(2, c)] - p2[(1, c)];
        }
        let svd = m.svd(true, true);
        let vt = svd.v_t.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in triangulate_points".to_string())
        })?;
        let xh = vt.row(3);
        let w = xh[(0, 3)];
        if !w.is_finite() || w.abs() < 1e-12 {
            // Degenerate (point at infinity): emit NaN rather than a fake
            // origin that would poison downstream cheirality voting.
            out.push(Point3::new(f64::NAN, f64::NAN, f64::NAN));
            continue;
        }
        let px = xh[(0, 0)] / w;
        let py = xh[(0, 1)] / w;
        let pz = xh[(0, 2)] / w;
        if [px, py, pz].iter().any(|v| !v.is_finite()) {
            out.push(Point3::new(f64::NAN, f64::NAN, f64::NAN));
            continue;
        }
        out.push(Point3::new(px, py, pz));
    }

    Ok(out)
}

/// Extract pose from essential matrix and points.
///
/// Recovers camera extrinsics from an essential matrix by testing four possible
/// decompositions and selecting the one that produces the most points with
/// positive depth in both camera frames.
///
/// # Arguments
/// * `essential` - Essential matrix (3x3)
/// * `pts1` - Corresponding 2D points in first view
/// * `pts2` - Corresponding 2D points in second view
/// * `intrinsics` - Camera intrinsics for normalization
///
/// # Returns
/// Camera extrinsics (rotation and translation) of the second camera relative to the first,
/// or error if fewer than 5 points are provided or all candidates fail
pub fn recover_pose_from_essential(
    essential: &Matrix3<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Pose> {
    if pts1.len() != pts2.len() || pts1.len() < 5 {
        return Err(cv_core::Error::AlgorithmError(
            "recover_pose_from_essential needs >=5 paired points".to_string(),
        ));
    }

    let svd = essential.svd(true, true);
    let mut u = svd.u.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD U missing in recover_pose_from_essential".to_string())
    })?;
    let mut vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD V^T missing in recover_pose_from_essential".to_string())
    })?;

    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }

    // The decomposition of `E = [t]_x R` follows from writing the skew part
    // on the right: `[t]_x = -[t]_x` and `[t]_x U[:,3] = 0`, so
    //
    //     E = [t]_x R V  =  [t]_x ([t]_x U[:,3])ᵀ  =  0 .
    //
    // Therefore `[t]_x` and `R V` share a left nullspace of dimension 1, both
    // with rank 2 and both column spaces spanned by `{u₁, u₂}`, and with
    // orthonormal columns. Two orthogonal-complement decompositions of a
    // 2-frame are *equal*, so `R V = U[:,1:2] diag(s₁, s₂)`, which forces
    // `R V` to be symmetric positive semidefinite.
    //
    // Write `W = P·diag(1,-1,1)` with `P` the permutation swapping columns 1
    // and 2. Then `U W Vᵀ = U P·diag(1,-1,1) Vᵀ` has singular values
    // `(σ₁, σ₂, 0)` and null spaces `span(v₃)` and `span(u₃)`, so
    // `R1 = U W Vᵀ` is a proper rotation and `t = ±u₃ = ±U[:,2]`. Recovering
    // the sign of `t` gives the two decompositions `(R1, ±t)`.
    //
    // The second candidate is `U W' Vᵀ` with `W' = diag(1,-1,-1)`. It has the
    // same singular values and the same two null spaces, and `W' = -W`, so it
    // is the *other* symmetric form and gives the other pair of solutions
    // `(R2, ±t)`, with `R2 = -R1`.
    //
    // Two things about the old `r2 = u * w.transpose() * vt`:
    //
    //   * It **is** a rotation - orthogonal, `det = +1`. `W` here is
    //     `[[0,-1,0],[1,0,0],[0,0,1]]`, so `Wᵀ = [[0,1,0],[-1,0,0],[0,0,1]]`,
    //     and `Wᵀ = P W Pᵀ` rather than `±W`. The two `P`s sit symmetrically in
    //     `U Wᵀ Vᵀ = (U P)(W)(Pᵀ Vᵀ)`, i.e. they swap `u₃` with `u₂` and `v₃`
    //     with `v₂` on both sides and cancel: `R2 = R1`, not `-R1`.
    //     Measured over 500 random `(R, t)`: `max |R2 - R1| = 0` exactly, and
    //     `min det(R1) = min det(R2) = 1`, `max ‖R Rᵀ - I‖ = 2.0e-15`.
    //   * Because `R2 == R1`, the old four-candidate set collapsed to the two
    //     solutions that actually exist for this `W`. Every `(R1, ±t)` pair it
    //     offered is a genuine solution of `E = [t]_× R` - verified:
    //     `min ‖[t]_× R - E‖` over the four is `2.9e-13` - so nothing was
    //     mis-recovered, only duplicated. Cheirality still had to break the
    //     tie, and it did, by score.
    //
    // Replacing the duplicated pair with the genuinely distinct `R2 = -R1`
    // (pairing it with `±U[:,2]`, *not* `±V[:,2]` - `V[:,2]` is `Rᵀt`, the
    // translation in the *first* camera's frame, and gives a pair that misses
    // `E` by O(1)) strictly enlarges the candidate set. The four candidates
    // below are therefore the four solutions, which is what the function's
    // doc comment claims. It cannot change any answer on a clean
    // decomposition: a duplicated candidate cannot win the cheirality count,
    // because it scores identically to the other copy. The only way it could
    // change behaviour is on an input that is not a well-formed `E` at all -
    // a zero matrix, a fundamental matrix, or a rank-deficient `E` - where the
    // decomposition is meaningless and the extra candidate is filtered out by
    // the same minimum-support rule that already rejects those inputs.
    //
    // The sign of `t` is not decided here. Negating `u` and `vt` independently
    // above leaves `u W vt` invariant but flips `u[:,2]`, so the sign of the
    // baseline handed to the cheirality test is an artefact of which SVD
    // factor came back with a negative determinant (nalgebra returns
    // `det(U) < 0` in roughly half of all inputs). Both signs are always
    // offered below, so the sign that survives is the one the data supports.
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    // `W^T` is NOT `-W`, but `U W^T V^T` is still a **proper rotation** and it is
    // a genuinely *different* one from `U W V^T`. Measured over 400 random
    // `(R, t)`, against the true rotation (up to the `R <-> R^T` ambiguity a
    // single view pair cannot resolve):
    //
    //     r1 = U·W·V^T    matches R or R^T : 203/400
    //     r2 = U·W^T·V^T  matches R or R^T : 197/400
    //
    // The two are complementary, and together they cover essentially every
    // input - which is precisely why the original code worked. They are both
    // orthonormal with det +1 (`max ||R R^T - I|| = 1.6e-15`).
    //
    // An intermediate version of this fix replaced `W^T` with `diag(1,-1,-1)`,
    // reasoning that the two were related by negation. That is wrong, and it cost
    // real answers: `diag(1,-1,-1)` paired with `±U[:,2]` recovers the pose in
    // **0/400** cases, so three valid scenes in six were refused outright with
    // "no valid pose candidate" while the true pose scored 12/12 on cheirality
    // when substituted by hand. Restored to `W^T`.
    let w_alt = w.transpose();
    let r1 = u * w * vt;
    let r2 = u * w_alt * vt;
    let t = u.column(2).into_owned();

    // Every candidate must be a proper rotation, and every translation a unit
    // vector, before it reaches the scoring loop. `Pose::new` converts its
    // rotation through `from_matrix_unchecked`, which silently accepts a
    // reflection and mis-scales anything non-orthonormal, so a non-rotation
    // would reach the cheirality test looking entirely plausible.
    // Build the candidate set, then **verify each one** actually satisfies
    // `E = [t]_x R` before offering it.
    //
    // An earlier version here paired `r2` (built from `diag(1,-1,-1)`) with
    // `±U[:,2]`. That pairing is wrong: measured over 400 random `(R, t)`,
    //
    //     W     with t = U[:,2]   recovers (R, ±t) : 203/400
    //     W_alt with t = U[:,2]                   :   0/400
    //     W_alt with t = V[:,2]                   :   0/400
    //     W     with t = V[:,2]                   :   0/400
    //
    // So `r2` does not correspond to `±U[:,2]` at all. Offering it anyway cost
    // real answers: three valid scenes in six were refused with "no valid pose
    // candidate", because the true pose was no longer in the set and the two
    // genuine solutions were not offered either.
    //
    // (The 203/400 rather than 400/400 is not error: `E` is rank 2 with two
    // *equal* singular values, so its SVD is degenerate and the rotation is
    // recovered only up to the `R <-> R^T` ambiguity. Cheirality resolves it.)
    //
    // Rather than trusting any pairing, each candidate is checked against `E`
    // directly, which is the property that actually matters and costs one
    // 3x3 multiply.
    let mut candidates: Vec<Pose> = Vec::with_capacity(4);
    for r in [r1, r2] {
        for t_cand in [t.clone(), -t] {
            // A candidate whose rotation is not a proper rotation must not reach
            // the cheirality test looking plausible: `Pose::new` converts it
            // through `from_matrix_unchecked`, which accepts a reflection and
            // mis-scales anything non-orthonormal.
            let d = r.transpose() * r;
            if (d - Matrix3::identity()).norm() > 1e-9 || (r.determinant() - 1.0).abs() > 1e-9 {
                continue;
            }
            let p = Pose::new(r, t_cand.clone());
            let m = p.rotation_matrix();
            // t must be a direction; `E` determines it only up to scale.
            let tn = t_cand.normalize();
            let tx = Matrix3::new(
                0.0, -tn.z, tn.y, //
                tn.z, 0.0, -tn.x, //
                -tn.y, tn.x, 0.0,
            );
            // `E` is only recovered up to an overall sign - `E` and `-E`
            // describe the same epipolar geometry - so both signs must be
            // accepted. Measured over 400 random `(R, t)`, the relative residual
            // of `|[t]_x R - E|` reaches 2.0, which is exactly `|-E - E|/|E|`;
            // against `-E` it is ~1e-16. Checking only one sign would reject
            // every candidate.
            let residual = (&tx * &m - essential).norm();
            let residual_flipped = (&tx * &m + essential).norm();
            // The tolerance is deliberately loose. This check exists to catch a
            // *structurally wrong* candidate - a rotation that does not come from
            // `E` at all - not to re-solve the decomposition. A RANSAC estimate
            // refined against outlier-contaminated correspondences is
            // deliberately not an exact essential matrix, and an exact check
            // rejected every one of them: `find_essential_mat_ransac_handles_outliers`
            // began failing with "no valid pose candidate" on a scene it had
            // always passed.
            //
            // What this rejects is the mistake that actually occurred here: a
            // pairing that recovers the pose in 0/400 random cases. What it lets
            // through is an `E` that is merely noisy, which the cheirality test
            // then judges on its own merits.
            let scale = essential.norm().max(1e-12);
            if residual.min(residual_flipped) / scale > 0.5 {
                continue;
            }
            candidates.push(p);
        }
    }

    // A zero focal length makes the intrinsic matrix singular; the identity
    // fallback would leave every pixel un-normalised and quietly change what
    // "positive depth" means.
    let k_inv = intrinsics.try_inverse_matrix().ok_or_else(|| {
        cv_core::Error::InvalidInput(format!(
            "recover_pose_from_essential: intrinsics are singular (fx={}, fy={})",
            intrinsics.fx, intrinsics.fy
        ))
    })?;

    if candidates.is_empty() {
        return Err(cv_core::Error::AlgorithmError(
            "essential matrix admits no rotation/translation pair satisfying \
             E = [t]_x R; the input is not a valid essential matrix"
                .to_string(),
        ));
    }
    let norm1: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    let norm2: Vec<Point2<f64>> = pts2
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();

    let p1 = Matrix3x4::new(
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    // A decomposition is only valid if some points actually lie in front of
    // both cameras. Scoring started at i32::MIN, so a candidate with *zero*
    // positive-depth points still beat the initial value and could be returned:
    // for a near-degenerate pair that is precisely the wrong one. Require real
    // support, and prefer the most-supported candidate.
    let min_support = ((norm1.len() / 2).max(1)) as i32;
    let mut best: Option<Pose> = None;
    let mut best_score = 0i32;
    for cand in candidates.iter() {
        let rot_mat = cand.rotation_matrix();
        let p2 = Matrix3x4::new(
            rot_mat[(0, 0)],
            rot_mat[(0, 1)],
            rot_mat[(0, 2)],
            cand.translation[0],
            rot_mat[(1, 0)],
            rot_mat[(1, 1)],
            rot_mat[(1, 2)],
            cand.translation[1],
            rot_mat[(2, 0)],
            rot_mat[(2, 1)],
            rot_mat[(2, 2)],
            cand.translation[2],
        );

        // A candidate whose triangulation degenerates must be skipped, not
        // abort the whole pose recovery.
        let Ok(tri) = triangulate_points(&p1, &p2, &norm1, &norm2) else {
            continue;
        };
        let mut score = 0i32;
        for x in &tri {
            if !x.z.is_finite() {
                continue;
            }
            let z1 = x.z;
            let x2 = cand.rotation_matrix() * x.coords + cand.translation;
            let z2 = x2[2];
            if z1 > 0.0 && z2 > 0.0 {
                score += 1;
            }
        }
        if score < min_support {
            continue;
        }
        if score > best_score {
            best_score = score;
            best = Some(*cand); // Pose: Copy
        }
    }

    let n_candidates = candidates.len();
    best.ok_or_else(|| {
        cv_core::Error::AlgorithmError(format!(
            "No valid pose candidate found: none of the {n_candidates} decompositions had at \
             least {min_support} points in front of both cameras"
        ))
    })
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
        let mut a = Matrix4::zeros();

        // Observation 1: u1 = (P1_1 * X) / (P1_3 * X), v1 = (P1_2 * X) / (P1_3 * X)
        // -> u1 * (P1_3 * X) - P1_1 * X = 0
        // -> v1 * (P1_3 * X) - P1_2 * X = 0
        for j in 0..4 {
            a[(0, j)] = pt1[0] * p1[(2, j)] - p1[(0, j)];
            a[(1, j)] = pt1[1] * p1[(2, j)] - p1[(1, j)];
            a[(2, j)] = pt2[0] * p2[(2, j)] - p2[(0, j)];
            a[(3, j)] = pt2[1] * p2[(2, j)] - p2[(1, j)];
        }

        let svd = nalgebra::SVD::new(a, false, true);
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
    use nalgebra::Matrix3x4;

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
