//! A point without a position (a NaN or infinite coordinate) gets the default
//! normal from `estimate_normals_knn` and does not change any other normal.
//!
//! It used to go into the KD-tree with the others: building the tree panicked
//! on the NaN, and a NaN neighbour would have made every covariance it joined
//! NaN. Non-finite points are now left out of the tree, so the neighbourhoods
//! of the other points are those of the clean cloud.

use cv_3d::filters::estimate_normals_knn;
use nalgebra::{Point3, Vector3};

/// A gently curved 8x8 patch.
fn surface() -> Vec<Point3<f64>> {
    let mut pts = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            let (x, y) = (i as f64 * 0.1, j as f64 * 0.1);
            pts.push(Point3::new(x, y, 0.2 * x * x - 0.1 * y * y + 0.05 * x * y));
        }
    }
    pts
}

#[test]
fn a_nan_point_gets_the_default_normal_and_leaves_neighbours_alone() {
    let clean = surface();
    let want = estimate_normals_knn(&clean, 8);
    // Control: a real, upward surface normal everywhere.
    assert!(
        want.iter()
            .all(|n| n.z > 0.8 && (n.norm() - 1.0).abs() < 1e-9)
    );

    for (insert_at, bad) in [
        (20usize, Point3::new(f64::NAN, 0.0, 0.0)),
        (0, Point3::new(0.3, f64::NAN, f64::NAN)),
        (64, Point3::new(0.0, 0.0, f64::INFINITY)),
    ] {
        let mut pts = clean.clone();
        pts.insert(insert_at, bad);
        let got = estimate_normals_knn(&pts, 8);
        assert_eq!(got.len(), pts.len());
        assert_eq!(
            got[insert_at],
            Vector3::new(0.0, 0.0, 1.0),
            "the non-finite point at {insert_at} must get the default normal"
        );
        let others: Vec<Vector3<f64>> = got
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != insert_at)
            .map(|(_, n)| *n)
            .collect();
        for (i, (g, w)) in others.iter().zip(&want).enumerate() {
            assert!(
                (g - w).amax() < 1e-12,
                "non-finite point at {insert_at} changed normal {i}: {g:?} vs {w:?}"
            );
        }
    }
}

#[test]
fn too_few_finite_points_give_the_default_normal() {
    // Three points, but only two have a position: no neighbourhood of three.
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(f64::NAN, 1.0, 0.0),
    ];
    let got = estimate_normals_knn(&pts, 3);
    assert!(got.iter().all(|n| *n == Vector3::new(0.0, 0.0, 1.0)));
}
