//! `estimate_normals_knn(points, k)` is Open3D's
//! `estimate_normals(KDTreeSearchParamKNN(k))`: the neighbourhood is the `k`
//! nearest points *including the query itself*, and fewer than three points
//! give the default `(0, 0, 1)`.
//!
//! It used the query plus `k` other points (`k + 1` in all). On the cloud
//! below with `k = 5` the smallest `|n . n_open3d|` over the 60 points was
//! 0.0069 (a normal at right angles to Open3D's); a numpy PCA over the `k`
//! nearest including the point reproduces Open3D to 6e-16, and over `k + 1`
//! does not.
//!
//! References: Open3D 0.20.0. Directions are compared up to sign: Open3D does
//! not orient normals without a viewpoint or existing normals, this function
//! orients them into `z >= 0`.

use cv_3d::filters::estimate_normals_knn;
use nalgebra::{Point3, Vector3};

fn cloud() -> Vec<Point3<f64>> {
    (0..60)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                (t * 0.37).sin() * 0.5 + (t * 0.011).cos() * 0.2,
                (t * 0.53).cos() * 0.4 + (t * 0.007).sin() * 0.3,
                (t * 0.29).sin() * (t * 0.13).cos() * 0.3,
            )
        })
        .collect()
}

fn assert_same_line(got: &Vector3<f64>, want: [f64; 3], what: &str) {
    let want = Vector3::from(want);
    let dot = got.dot(&want).abs();
    assert!(
        (dot - 1.0).abs() < 1e-9,
        "{what}: {got:?} vs Open3D {want:?} (|dot| = {dot})"
    );
}

#[test]
fn knn_neighbourhood_counts_the_point_itself() {
    const OPEN3D_K5: [(usize, [f64; 3]); 4] = [
        (0, [0.270886214581, -0.473839717543, -0.837912036451]),
        (1, [-0.290956088006, 0.883269218056, 0.367668387663]),
        (17, [0.004647087700, 0.275754360271, 0.961216904433]),
        (42, [0.274148038478, 0.960156640542, 0.054240931233]),
    ];
    let normals = estimate_normals_knn(&cloud(), 5);
    for (i, want) in OPEN3D_K5 {
        assert_same_line(&normals[i], want, &format!("k=5 point {i}"));
    }
}

#[test]
fn fewer_than_three_points_give_the_default_normal() {
    let pts: Vec<Point3<f64>> = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.2],
        [1.0, 1.0, 0.5],
        [0.5, 0.3, 2.0],
    ]
    .iter()
    .map(|p| Point3::from(*p))
    .collect();
    // Open3D: k = 1 and k = 2 give (0, 0, 1) for every point.
    for k in [1, 2] {
        for (i, n) in estimate_normals_knn(&pts, k).iter().enumerate() {
            assert_eq!(*n, Vector3::z(), "k={k} point {i}");
        }
    }
    // k = 3: the point and its two nearest. Open3D 0.20.0, rounded to 1e-6.
    let open3d_k3 = [
        [0.0, 0.196116, -0.980581],
        [0.0, 0.447214, -0.894427],
        [0.282216, 0.188144, -0.940721],
        [0.259161, 0.431934, -0.863868],
        [0.116369, -0.914327, -0.387896],
    ];
    for (i, (n, want)) in estimate_normals_knn(&pts, 3)
        .iter()
        .zip(open3d_k3)
        .enumerate()
    {
        let dot = n.dot(&Vector3::from(want)).abs();
        assert!((dot - 1.0).abs() < 1e-5, "k=3 point {i}: {n:?} vs {want:?}");
    }
    // k beyond the cloud size uses the whole cloud, as Open3D.
    for n in estimate_normals_knn(&pts, 10) {
        let dot = n.dot(&Vector3::new(0.990499, 0.10891, -0.083965)).abs();
        assert!((dot - 1.0).abs() < 1e-5, "k=10: {n:?}");
    }
}
