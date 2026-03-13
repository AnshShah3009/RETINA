use crate::Result;
use cv_core::{CameraIntrinsics, Pose};
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
        return Err(cv_core::Error::CalibrationError(
            "triangulate_points requires equal point counts".to_string(),
        ));
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
            cv_core::Error::CalibrationError("SVD failed in triangulate_points".to_string())
        })?;
        let xh = vt.row(3);
        let w = xh[(0, 3)];
        if w.abs() < 1e-12 {
            out.push(Point3::new(0.0, 0.0, 0.0));
            continue;
        }
        out.push(Point3::new(xh[(0, 0)] / w, xh[(0, 1)] / w, xh[(0, 2)] / w));
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
        return Err(cv_core::Error::CalibrationError(
            "recover_pose_from_essential needs >=5 paired points".to_string(),
        ));
    }

    let svd = essential.svd(true, true);
    let mut u = svd.u.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD U missing in recover_pose_from_essential".to_string())
    })?;
    let mut vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError(
            "SVD V^T missing in recover_pose_from_essential".to_string(),
        )
    })?;

    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;
    let t = u.column(2).into_owned();

    let candidates = [
        Pose::new(r1, t),
        Pose::new(r1, -t),
        Pose::new(r2, t),
        Pose::new(r2, -t),
    ];

    let k_inv = intrinsics.inverse_matrix();
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

    let mut best = None;
    let mut best_score = i32::MIN;
    for cand in candidates {
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

        let tri = triangulate_points(&p1, &p2, &norm1, &norm2)?;
        let mut score = 0i32;
        for x in &tri {
            let z1 = x.z;
            let x2 = cand.rotation * x.coords + cand.translation;
            let z2 = x2[2];
            if z1 > 0.0 && z2 > 0.0 {
                score += 1;
            }
        }
        if score > best_score {
            best_score = score;
            best = Some(cand);
        }
    }

    best.ok_or_else(|| {
        cv_core::Error::CalibrationError("No valid pose candidate found".to_string())
    })
}
