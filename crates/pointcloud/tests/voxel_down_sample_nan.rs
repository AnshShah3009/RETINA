//! A point without a finite position must not join a voxel.
//!
//! `(NaN / vs).floor() as i32` saturates to 0, so a NaN point was bucketed
//! into voxel (0, 0, 0) and turned the real point there into
//! `[NaN, 0.15, 0.15]`: one bad input point destroyed a good one.

use cv_core::PointCloud;
use cv_pointcloud::voxel_down_sample;
use nalgebra::Point3;

#[test]
fn nan_points_do_not_poison_their_voxel() {
    let pc = PointCloud {
        points: vec![
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(f32::NAN, 0.2, 0.2),
            Point3::new(5.0, 5.0, 5.0),
        ],
        colors: None,
        normals: None,
    };
    let out = voxel_down_sample(&pc, 1.0);
    let mut pts = out.points.clone();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    assert_eq!(
        pts,
        vec![Point3::new(0.1, 0.1, 0.1), Point3::new(5.0, 5.0, 5.0)]
    );
}
