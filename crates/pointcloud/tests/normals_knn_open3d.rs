//! `estimate_normals(pc, k)` is Open3D's
//! `estimate_normals(KDTreeSearchParamKNN(k))`: the neighbourhood is the `k`
//! nearest points *including the query itself*, and fewer than three points
//! give the default `(0, 0, 1)`.
//!
//! It used the `k` nearest *other* points and left the query out of the
//! covariance. On the cloud below with `k = 5` point 0's normal was at
//! |dot| = 0.935 to Open3D's.
//!
//! References: Open3D 0.20.0 (the same cloud and values as
//! `cv-3d/tests/normals_knn_open3d.rs`). Compared up to sign.

use cv_core::PointCloud;
use cv_pointcloud::estimate_normals;
use nalgebra::{Point3, Vector3};

#[test]
fn knn_neighbourhood_counts_the_point_itself() {
    let points: Vec<Point3<f32>> = (0..60)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                ((t * 0.37).sin() * 0.5 + (t * 0.011).cos() * 0.2) as f32,
                ((t * 0.53).cos() * 0.4 + (t * 0.007).sin() * 0.3) as f32,
                ((t * 0.29).sin() * (t * 0.13).cos() * 0.3) as f32,
            )
        })
        .collect();
    let mut pc = PointCloud::new(points);
    estimate_normals(&mut pc, 5);
    let normals = pc.normals.clone().unwrap();
    const OPEN3D_K5: [(usize, [f64; 3]); 4] = [
        (0, [0.270886214581, -0.473839717543, -0.837912036451]),
        (1, [-0.290956088006, 0.883269218056, 0.367668387663]),
        (17, [0.004647087700, 0.275754360271, 0.961216904433]),
        (42, [0.274148038478, 0.960156640542, 0.054240931233]),
    ];
    for (i, want) in OPEN3D_K5 {
        let got = normals[i].cast::<f64>();
        let dot = got.dot(&Vector3::from(want)).abs();
        assert!(
            (dot - 1.0).abs() < 1e-4,
            "k=5 point {i}: {got:?} vs Open3D {want:?} (|dot| = {dot})"
        );
    }

    // k = 2: two points cannot define a plane; Open3D gives (0, 0, 1).
    let mut pc2 = pc.clone();
    pc2.normals = None;
    estimate_normals(&mut pc2, 2);
    assert!(pc2.normals.unwrap().iter().all(|n| *n == Vector3::z()));
}
