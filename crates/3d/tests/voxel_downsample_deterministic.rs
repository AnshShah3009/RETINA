//! `cv_3d::gpu::voxel_downsample` is the same reduction as
//! `cv_3d::filters::voxel_downsample`: voxels keyed on the global grid
//! (`floor(p / voxel_size)`), one centroid per voxel, in sorted key order.
//!
//! It used to anchor its grid at `points[0]`, so the voxel boundaries moved
//! with whichever point came first, and it returned the voxels in `HashMap`
//! order, which differs between runs. It also had no guard for a zero or NaN
//! voxel size.

use nalgebra::Point3;

/// A spread cloud whose first point is not on the global grid of the sizes
/// tested below.
fn cloud() -> Vec<Point3<f32>> {
    (0..400)
        .map(|i| {
            let t = i as f32;
            Point3::new(
                0.137 + (t * 0.37).sin() * 2.0,
                0.291 + (t * 0.53).cos() * 1.5,
                0.053 + (t * 0.29).sin() * (t * 0.13).cos(),
            )
        })
        .collect()
}

fn via_filters(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
    let pts: Vec<Point3<f64>> = points.iter().map(|p| p.cast::<f64>()).collect();
    cv_3d::filters::voxel_downsample(&pts, None, None, voxel_size as f64)
        .points
        .iter()
        .map(|p| p.cast::<f32>())
        .collect()
}

#[test]
fn gpu_voxel_downsample_matches_the_filter_and_is_ordered() {
    let pts = cloud();
    for vs in [0.25f32, 0.5, 1.0] {
        let want = via_filters(&pts, vs);
        assert!(want.len() > 1, "voxel size {vs} must leave several voxels");

        let got = cv_3d::gpu::voxel_downsample(&pts, vs);
        assert_eq!(got, want, "gpu::voxel_downsample at size {vs}");

        let got = cv_3d::gpu::point_cloud::voxel_downsample(&pts, vs);
        assert_eq!(got, want, "gpu::point_cloud::voxel_downsample at size {vs}");

        // Sorted key order: each centroid lies in its voxel, and the voxel
        // keys of consecutive centroids are strictly increasing.
        let keys: Vec<(i64, i64, i64)> = want
            .iter()
            .map(|p| {
                let k = |v: f32| ((v as f64) / vs as f64).floor() as i64;
                (k(p.x), k(p.y), k(p.z))
            })
            .collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "size {vs}: {keys:?}");
    }
}

#[test]
fn gpu_voxel_downsample_rejects_invalid_sizes() {
    let pts = cloud();
    for vs in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            cv_3d::gpu::voxel_downsample(&pts, vs).is_empty(),
            "voxel size {vs}"
        );
        assert!(
            cv_3d::gpu::point_cloud::voxel_downsample(&pts, vs).is_empty(),
            "voxel size {vs}"
        );
    }
    assert!(cv_3d::gpu::voxel_downsample(&[], 1.0).is_empty());
}
