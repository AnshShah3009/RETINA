//! `remove_radius_outliers` counts *neighbours*, so a point never counts itself.
//!
//! The doc contract is "points with fewer than `min_points` neighbours within
//! `radius` are removed", and Open3D's `remove_radius_outlier(nb_points,
//! radius)` (0.20, measured) keeps a point only when at least `nb_points`
//! *other* points lie within the radius. The query used to count the point
//! itself, so every threshold was one too lenient: with `min_points = 1` an
//! isolated point - zero neighbours - survived, and so did every other point of
//! the cloud below.

use cv_core::PointCloud;
use cv_pointcloud::point_cloud::remove_radius_outliers;
use nalgebra::Point3;

fn cloud(n: usize) -> PointCloud {
    PointCloud::new(
        (0..n)
            .map(|i| {
                let t = i as f64;
                Point3::new(
                    ((t * 0.37).sin() * 2.0 + (t * 0.011).cos()) as f32,
                    ((t * 0.53).cos() * 1.5 + (t * 0.007).sin()) as f32,
                    ((t * 0.29).sin() * (t * 0.13).cos()) as f32,
                )
            })
            .collect(),
    )
}

#[test]
fn an_isolated_point_has_no_neighbours() {
    let pc = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(5.0, 5.0, 5.0),
    ]);
    // Open3D 0.20: remove_radius_outlier(1, 0.2) -> [0, 1].
    let (_, kept) = remove_radius_outliers(&pc, 0.2, 1);
    assert_eq!(kept, vec![0, 1]);
    // Open3D 0.20: remove_radius_outlier(2, 0.2) -> []. Each point of the pair
    // has exactly one neighbour.
    let (filtered, kept) = remove_radius_outliers(&pc, 0.2, 2);
    assert!(kept.is_empty(), "kept {kept:?}");
    assert!(filtered.is_empty());
}

#[test]
fn matches_open3d_on_a_spread_cloud() {
    // Same 500 points as `open3d.geometry.PointCloud.remove_radius_outlier`
    // (Open3D 0.20.0), which keeps 87, 131 and 154 points respectively.
    let pc = cloud(500);
    for (nb, radius, expected) in [(3usize, 0.3f32, 87usize), (1, 0.2, 131), (8, 0.5, 154)] {
        let (_, kept) = remove_radius_outliers(&pc, radius, nb);
        assert_eq!(
            kept.len(),
            expected,
            "nb_points {nb}, radius {radius}: kept {} (Open3D {expected})",
            kept.len()
        );
    }
}
