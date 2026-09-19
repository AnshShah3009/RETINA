//! Spatial Data Structures
//!
//! Implements:
//! - KDTree for fast nearest neighbor search
//! - Octree for spatial partitioning
//! - VoxelGrid for voxelization
//! - BVH for accelerated ray-triangle intersection
//! - HashGrid for O(1) neighbor queries on uniform distributions

pub mod bvh;
pub mod hash_grid;
pub mod kd_tree;
pub mod octree;
pub mod voxel_grid;

pub use bvh::Bvh;
pub use hash_grid::HashGrid;
pub use kd_tree::KDTree;
pub use octree::Octree;
pub use voxel_grid::VoxelGrid;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_kdtree_nearest_neighbor() {
        let mut tree = KDTree::<usize>::new();

        // Insert 5 known points with their index as data
        let points = [
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(-2.0, -2.0, -2.0),
        ];
        for (i, &p) in points.iter().enumerate() {
            tree.insert(p, i);
        }

        // Query a point closest to (0.9, 0.1, 0.1) -- should match points[0] = (1,0,0)
        let query = Point3::new(0.9, 0.1, 0.1);
        let result = tree.nearest_neighbor(&query);
        assert!(result.is_some(), "KDTree should find a nearest neighbor");

        let (nearest_point, data, dist_sq) = result.unwrap();
        assert_eq!(data, 0, "Nearest neighbor should be point index 0");
        assert!(
            (nearest_point.x - 1.0).abs() < 1e-6
                && (nearest_point.y - 0.0).abs() < 1e-6
                && (nearest_point.z - 0.0).abs() < 1e-6,
            "Nearest point should be (1, 0, 0), got {:?}",
            nearest_point
        );

        // Also verify distance is correct: ||(0.9,0.1,0.1) - (1,0,0)|| = sqrt(0.01+0.01+0.01)
        let expected_dist_sq = 0.03;
        assert!(
            (dist_sq - expected_dist_sq).abs() < 1e-5,
            "Expected squared distance ~{}, got {}",
            expected_dist_sq,
            dist_sq
        );

        // Query near (5,5,5) should match points[3]
        let query2 = Point3::new(4.9, 5.1, 5.0);
        let (_, data2, _) = tree.nearest_neighbor(&query2).unwrap();
        assert_eq!(data2, 3, "Nearest neighbor should be point index 3");
    }
}
