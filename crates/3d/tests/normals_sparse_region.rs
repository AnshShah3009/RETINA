//! The CPU normal estimator's 27-voxel neighbourhood is sized from the whole
//! cloud. Points in a sparse region (a 25-point wall beside a dense 10k-point
//! floor) found fewer than 3 neighbours there and silently got (0, 0, 1);
//! they must get the PCA normal of their true nearest neighbours.

use cv_3d::gpu::point_cloud::compute_normals_cpu;
use nalgebra::Point3;

#[test]
fn sparse_wall_normals_are_computed_not_defaulted() {
    let mut points = Vec::new();
    for i in 0..100 {
        for j in 0..100 {
            points.push(Point3::new(i as f32 * 0.01, j as f32 * 0.01, 0.0));
        }
    }
    let wall_start = points.len();
    for i in 0..5 {
        for j in 0..5 {
            points.push(Point3::new(3.0, i as f32 * 0.2, 1.0 + j as f32 * 0.2));
        }
    }
    let normals = compute_normals_cpu(&points, 8, 0.0);
    for n in &normals[wall_start..] {
        assert!(n.x.abs() > 0.99, "wall normal {n:?} should be +-x");
    }
    for n in &normals[..100] {
        assert!(n.z.abs() > 0.99, "floor normal {n:?} should be +-z");
    }
}
