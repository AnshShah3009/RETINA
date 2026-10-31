//! DBSCAN labels with the each-point-queued-once expansion. The old loop
//! pushed every neighbour of every core point into the seed queue, so a dense
//! cluster of N mutually-close points queued O(N^2) entries; the labels must
//! be unchanged by bounding that.

use cv_core::PointCloud;
use cv_pointcloud::cluster_dbscan;
use nalgebra::Point3;

#[test]
fn two_dense_blobs_and_noise() {
    let mut points = Vec::new();
    for i in 0..400 {
        let a = i as f32 * 0.1;
        points.push(Point3::new(
            0.05 * a.sin(),
            0.05 * a.cos(),
            0.0005 * i as f32 % 0.05,
        ));
    }
    for i in 0..300 {
        let a = i as f32 * 0.1;
        points.push(Point3::new(5.0 + 0.05 * a.sin(), 0.05 * a.cos(), 0.0));
    }
    points.push(Point3::new(20.0, 20.0, 20.0)); // noise
    let pc = PointCloud {
        points,
        colors: None,
        normals: None,
    };
    let labels = cluster_dbscan(&pc, 0.2, 5);
    assert!(labels[..400].iter().all(|&l| l == labels[0]) && labels[0] >= 0);
    assert!(labels[400..700].iter().all(|&l| l == labels[400]) && labels[400] >= 0);
    assert_ne!(labels[0], labels[400]);
    assert_eq!(labels[700], -1);
}

/// A border point first visited as noise joins the cluster that reaches it.
#[test]
fn noise_reached_from_a_cluster_becomes_border() {
    let mut points: Vec<Point3<f32>> = (0..10)
        .map(|i| Point3::new(i as f32 * 0.1, 0.0, 0.0))
        .collect();
    // Visited first (index 0 below), too sparse to be core, but within eps of the line's end.
    points.insert(0, Point3::new(1.05, 0.0, 0.0));
    let pc = PointCloud {
        points,
        colors: None,
        normals: None,
    };
    let labels = cluster_dbscan(&pc, 0.16, 3);
    assert!(
        labels.iter().all(|&l| l == labels[1] && l >= 0),
        "{labels:?}"
    );
}
