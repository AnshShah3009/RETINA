//! Alpha Shapes reconstruction
//!
//! Alpha Shapes are a generalization of the convex hull.
//! This implementation delegates to Ball Pivoting with alpha as the ball radius.

use super::ball_pivoting::ball_pivoting;
use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;

/// Alpha Shapes reconstruction
pub fn alpha_shapes(cloud: &PointCloud, alpha: f32) -> TriangleMesh {
    ball_pivoting(cloud, alpha)
}
