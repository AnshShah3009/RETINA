//! `statistical_outlier_removal` keeps the same points as Open3D's
//! `remove_statistical_outlier(nb_neighbors, std_ratio)`.
//!
//! Open3D's neighbourhood is `KNN(nb_neighbors)`, which includes the query
//! point itself at distance 0, and the mean is taken over all of them; the
//! spread is the sample standard deviation and the threshold is strict. The
//! filter averaged over `nb_neighbors` *other* points with a population
//! standard deviation and `<=`, so the same parameters kept a different set.
//! The 3x3x3 lattice in `parity/parity_open3d.py` happened to agree; a less
//! regular cloud does not.
//!
//! Reference values: Open3D 0.20.0 on the closed-form cloud below
//! (`np.sin(i*0.37)*2 + np.cos(i*0.011)`, ... for `i in range(500)`).

use cv_3d::filters::statistical_outlier_removal;
use nalgebra::Point3;

fn cloud(n: usize) -> Vec<Point3<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                (t * 0.37).sin() * 2.0 + (t * 0.011).cos(),
                (t * 0.53).cos() * 1.5 + (t * 0.007).sin(),
                (t * 0.29).sin() * (t * 0.13).cos(),
            )
        })
        .collect()
}

#[test]
fn kept_counts_match_open3d() {
    let pts = cloud(500);
    for (nb, ratio, open3d) in [
        (5usize, 0.5f64, 364usize),
        (10, 1.0, 410),
        (20, 0.3, 340),
        (3, 2.0, 484),
    ] {
        let (kept, idx) = statistical_outlier_removal(&pts, nb, ratio);
        assert_eq!(kept.len(), idx.len());
        assert_eq!(
            idx.len(),
            open3d,
            "nb_neighbors {nb}, std_ratio {ratio}: kept {} (Open3D {open3d})",
            idx.len()
        );
    }
}

#[test]
fn the_pointcloud_crate_agrees() {
    let pts = cloud(500);
    let pc = cv_core::PointCloud::new(
        pts.iter()
            .map(|p| Point3::new(p.x as f32, p.y as f32, p.z as f32))
            .collect(),
    );
    for (nb, ratio, open3d) in [
        (5usize, 0.5f64, 364usize),
        (10, 1.0, 410),
        (20, 0.3, 340),
        (3, 2.0, 484),
    ] {
        let (_, idx) = cv_pointcloud::point_cloud::remove_statistical_outliers(&pc, nb, ratio);
        assert_eq!(
            idx.len(),
            open3d,
            "cv-pointcloud nb_neighbors {nb}, std_ratio {ratio}: kept {} (Open3D {open3d})",
            idx.len()
        );
    }
}
