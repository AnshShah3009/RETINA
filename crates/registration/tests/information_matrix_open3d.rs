//! `get_information_matrix_from_point_clouds` must be Open3D's
//! `GetInformationMatrixFromPointClouds`:
//!
//! ```text
//! G_r = [ -[q]x | I3 ]      (3 x 6, q = target point of the correspondence)
//! H   = sum_r G_r^T G_r
//! ```
//!
//! over correspondences whose distance is within `max_correspondence_distance`.
//! The rotation-first twist ordering matches Open3D. Earlier versions placed the
//! source coordinates in the translation rows and multiplied the rotation rows
//! by the residual - so an exact alignment gave `diag [1043, 1043, 1043, 0, 0, 0]`
//! (a zero rotation block, the opposite of what a perfect alignment says).

use cv_core::point_cloud::PointCloud;
use cv_registration::get_information_matrix_from_point_clouds;
use nalgebra::{Matrix3, Matrix4, Matrix6, Point3, Vector3};

fn grid(offset: Vector3<f32>) -> Vec<Point3<f32>> {
    let mut v = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..3 {
                v.push(Point3::new(x as f32 * 0.2, y as f32 * 0.3, z as f32 * 0.25) + offset);
            }
        }
    }
    v
}

fn skew(q: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -q.z, q.y, q.z, 0.0, -q.x, -q.y, q.x, 0.0)
}

/// Hand-computed Sigma G^T G in f64.
fn reference(points: &[Point3<f32>]) -> Matrix6<f64> {
    let mut h = Matrix6::<f64>::zeros();
    for p in points {
        let q = p.coords.cast::<f64>();
        let mut g = nalgebra::Matrix3x6::<f64>::zeros();
        g.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew(&q)));
        g.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        h += g.transpose() * g;
    }
    h
}

fn assert_close(got: &Matrix6<f32>, want: &Matrix6<f64>, what: &str) {
    for i in 0..6 {
        for j in 0..6 {
            let g = f64::from(got[(i, j)]);
            let w = want[(i, j)];
            assert!(
                (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
                "{what}: H[{i},{j}] = {g}, expected {w}\ngot:\n{got}\nwant:\n{want}"
            );
        }
    }
}

#[test]
fn identity_alignment_matches_hand_computed_sum() {
    let pts = grid(Vector3::new(0.1, -0.2, 0.3));
    let cloud = PointCloud::new(pts.clone());
    let h = get_information_matrix_from_point_clouds(&cloud, &cloud, &Matrix4::identity(), 0.05);
    assert_close(&h, &reference(&pts), "identity alignment");
    // Translation block is the correspondence count times identity.
    assert!((h[(3, 3)] - pts.len() as f32).abs() < 1e-3);
    assert!(
        h[(0, 0)] > 0.0,
        "rotation block must not vanish at exact alignment"
    );
}

#[test]
fn uses_the_target_point_and_the_transform() {
    // Source sits 0.01 off in x; the transform brings it back. Correspondences
    // are found after the transform, and G is built from the *target* point.
    let tgt = grid(Vector3::zeros());
    let src: Vec<Point3<f32>> = tgt
        .iter()
        .map(|p| p - Vector3::new(0.01, 0.0, 0.0))
        .collect();
    let mut t = Matrix4::identity();
    t[(0, 3)] = 0.01;
    let h = get_information_matrix_from_point_clouds(
        &PointCloud::new(src),
        &PointCloud::new(tgt.clone()),
        &t,
        0.05,
    );
    assert_close(&h, &reference(&tgt), "transformed source");
}

#[test]
fn correspondences_beyond_the_distance_are_excluded() {
    let tgt = grid(Vector3::zeros());
    let src: Vec<Point3<f32>> = tgt
        .iter()
        .map(|p| p + Vector3::new(0.0, 0.0, 0.03))
        .collect();
    let s = PointCloud::new(src);
    let t = PointCloud::new(tgt.clone());
    // 0.03 offset: inside a 0.05 gate, outside a 0.02 gate.
    let inside = get_information_matrix_from_point_clouds(&s, &t, &Matrix4::identity(), 0.05);
    assert_close(&inside, &reference(&tgt), "0.05 gate");
    let outside = get_information_matrix_from_point_clouds(&s, &t, &Matrix4::identity(), 0.02);
    assert_eq!(outside, Matrix6::zeros(), "0.02 gate must admit nothing");
}

#[test]
fn rotation_translation_coupling_follows_the_origin_like_open3d() {
    // Open3D's H depends on where the origin is only through the cross term:
    // shifting every point by c leaves the translation block at N * I, while
    // the coupling block is -sum [q]x^T ... i.e. it moves with the centroid.
    // For a cloud centred on the origin the coupling block is zero.
    let base = grid(Vector3::zeros());
    let centroid = base.iter().fold(Vector3::zeros(), |a, p| a + p.coords) / base.len() as f32;
    let centred: Vec<Point3<f32>> = base.iter().map(|p| p - centroid).collect();
    let c = PointCloud::new(centred.clone());
    let h0 = get_information_matrix_from_point_clouds(&c, &c, &Matrix4::identity(), 0.05);
    for i in 0..3 {
        for j in 3..6 {
            assert!(
                h0[(i, j)].abs() < 1e-3 && h0[(j, i)].abs() < 1e-3,
                "centred cloud must have zero rotation-translation coupling, H =\n{h0}"
            );
        }
    }
    // Shifted far away: the translation block is unchanged (still N * I), and
    // the coupling equals N * [c]x (the G^T G cross term for q = q0 + c).
    let shift = Vector3::new(5.0f32, -3.0, 2.0);
    let moved: Vec<Point3<f32>> = centred.iter().map(|p| p + shift).collect();
    let m = PointCloud::new(moved.clone());
    let h1 = get_information_matrix_from_point_clouds(&m, &m, &Matrix4::identity(), 0.05);
    let n = moved.len() as f32;
    for i in 3..6 {
        for j in 3..6 {
            let want = if i == j { n } else { 0.0 };
            assert!((h1[(i, j)] - want).abs() < 1e-3);
        }
    }
    let coupling = h1.fixed_view::<3, 3>(0, 3).into_owned();
    let want = skew(&shift.cast::<f64>()).cast::<f32>() * n;
    assert!(
        (coupling - want).abs().max() < 1e-2,
        "coupling block {coupling} != N [c]x {want}"
    );
    assert_close(&h1, &reference(&moved), "shifted cloud");
}
