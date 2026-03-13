//! Pose Graph Optimization for SLAM
//!
//! Refines camera trajectory by optimizing a graph of camera poses and
//! relative constraints (loop closures).
//!
//! # Overview
//! This module provides pose graph optimization for incremental SLAM systems.
//! It maintains a graph of camera poses connected by relative pose measurements
//! from matching or loop closure detection, then optimizes to find the
//! maximum-likelihood trajectory.

use cv_core::Pose;
use cv_optimize::pose_graph::PoseGraph as OptimizePoseGraph;
use nalgebra::Isometry3;

/// Edge in the pose graph representing a relative pose measurement.
///
/// Connects two camera poses (keyframes) with a measured relative transformation
/// and measurement uncertainty (via the information matrix).
#[derive(Clone)]
pub struct PoseGraphEdge {
    /// ID of the source (from) pose
    pub from: usize,
    /// ID of the target (to) pose
    pub to: usize,
    /// Measured relative pose from `from` to `to` camera frame
    pub relative_pose: Pose,
    /// Information matrix (inverse covariance) representing measurement uncertainty.
    /// Higher values indicate more confident measurements.
    pub information: nalgebra::Matrix6<f64>,
}

/// Pose graph for incremental SLAM trajectory optimization.
///
/// Maintains a set of camera poses (keyframes) connected by relative pose
/// measurements from matching or loop closure detection. The graph can be
/// optimized to find the trajectory that best satisfies all constraints.
#[derive(Clone)]
pub struct PoseGraph {
    /// Camera poses indexed by keyframe ID
    pub poses: Vec<Pose>,
    /// Relative pose constraints between poses
    pub edges: Vec<PoseGraphEdge>,
}

impl Default for PoseGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PoseGraph {
    /// Create an empty pose graph.
    pub fn new() -> Self {
        Self {
            poses: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a camera pose (keyframe) to the graph.
    ///
    /// # Arguments
    /// * `pose` - The camera pose (extrinsic transformation)
    ///
    /// # Returns
    /// The keyframe ID (index) assigned to this pose, which can be used
    /// to create edges connecting this pose to others.
    pub fn add_pose(&mut self, pose: Pose) -> usize {
        let id = self.poses.len();
        self.poses.push(pose);
        id
    }

    /// Add a relative pose constraint between two poses.
    ///
    /// # Arguments
    /// * `from` - ID of the source (earlier) pose
    /// * `to` - ID of the target (later) pose
    /// * `rel` - Measured relative transformation from `from` to `to`
    /// * `info` - Information matrix (6×6, inverse covariance) weighting the edge.
    ///   Use `Matrix6::identity()` for unit weight.
    pub fn add_edge(&mut self, from: usize, to: usize, rel: Pose, info: nalgebra::Matrix6<f64>) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            relative_pose: rel,
            information: info,
        });
    }

    /// Optimize the pose graph using Gauss-Newton optimization.
    ///
    /// Refines all camera poses to minimize the sum of squared errors across
    /// all pose constraints (edges) while maintaining consistency of the trajectory.
    ///
    /// # Algorithm
    /// Uses Gauss-Newton optimization with:
    /// - Automatic differentiation via numerical derivatives
    /// - Regularization with small diagonal perturbation (1e-6)
    /// - Convergence when error is below 1e-9 or iteration limit reached
    ///
    /// # Constraints
    /// - First pose (index 0) is fixed as the reference frame
    /// - Edges enforce relative pose constraints between cameras
    /// - Information matrices (edge weights) scale residual influence
    ///
    /// # Arguments
    /// * `iterations` - Maximum number of optimization iterations to perform
    ///
    /// # Returns
    /// - `Ok(final_error)` with sum of weighted squared errors after optimization
    /// - `Err(msg)` if Cholesky decomposition fails (non-positive-definite Hessian)
    ///
    /// # Example
    /// ```ignore
    /// # use cv_optimize::pose_graph::PoseGraph;
    /// # use nalgebra::{Isometry3, Vector3, Matrix6};
    /// let mut graph = PoseGraph::new();
    /// let pose1 = Isometry3::identity();
    /// let pose2 = Isometry3::translation(0.1, 0.0, 0.0);
    ///
    /// graph.add_node(0, pose1);
    /// graph.add_node(1, pose2);
    /// graph.add_edge(id1, id2, pose2, Matrix6::identity());
    ///
    /// let result = graph.optimize(10);
    /// assert!(result.is_ok());
    /// ```
    pub fn optimize(&mut self, iterations: usize) -> Result<f64, String> {
        if self.poses.is_empty() {
            return Ok(0.0);
        }

        // Convert slam poses to cv_optimize's Isometry3 representation
        let mut opt_graph = OptimizePoseGraph::new();
        for (i, pose) in self.poses.iter().enumerate() {
            let iso = Isometry3::from(*pose);
            opt_graph.add_node(i, iso);
        }

        // Fix first pose as reference (common in SLAM)
        opt_graph.set_fixed(0);

        // Add edges from constraints
        for edge in &self.edges {
            let iso_measurement = Isometry3::from(edge.relative_pose);
            opt_graph.add_edge(edge.from, edge.to, iso_measurement, edge.information);
        }

        // Run optimization
        let final_error = opt_graph.optimize(iterations)?;

        // Convert optimized poses back to cv_core::Pose
        for (i, pose) in self.poses.iter_mut().enumerate() {
            if let Some(optimized_iso) = opt_graph.nodes.get(&i) {
                *pose = Pose::from(*optimized_iso);
            }
        }

        Ok(final_error)
    }

    /// Get the number of poses (keyframes) in the graph.
    pub fn num_poses(&self) -> usize {
        self.poses.len()
    }

    /// Get the number of edges (constraints) in the graph.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if the graph is empty (no poses).
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn create_identity_pose() -> Pose {
        Pose::identity()
    }

    fn create_translated_pose(x: f64, y: f64, z: f64) -> Pose {
        Pose::new(Matrix3::identity(), Vector3::new(x, y, z))
    }

    #[test]
    fn test_pose_graph_creation() {
        let graph = PoseGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.num_poses(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_add_single_pose() {
        let mut graph = PoseGraph::new();
        let pose = create_identity_pose();
        let id = graph.add_pose(pose);

        assert_eq!(id, 0);
        assert_eq!(graph.num_poses(), 1);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_add_multiple_poses() {
        let mut graph = PoseGraph::new();

        for i in 0..5 {
            let pose = create_translated_pose(i as f64, 0.0, 0.0);
            let id = graph.add_pose(pose);
            assert_eq!(id, i);
        }

        assert_eq!(graph.num_poses(), 5);
    }

    #[test]
    fn test_add_sequential_edge() {
        let mut graph = PoseGraph::new();

        // Add two poses
        graph.add_pose(create_identity_pose());
        graph.add_pose(create_translated_pose(1.0, 0.0, 0.0));

        // Add edge between them
        let relative_pose = create_translated_pose(1.0, 0.0, 0.0);
        let info = nalgebra::Matrix6::identity();
        graph.add_edge(0, 1, relative_pose, info);

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.edges[0].from, 0);
        assert_eq!(graph.edges[0].to, 1);
    }

    #[test]
    fn test_add_loop_closure_edge() {
        let mut graph = PoseGraph::new();

        // Create a sequence of poses
        for i in 0..4 {
            graph.add_pose(create_translated_pose(i as f64, 0.0, 0.0));
        }

        // Add sequential edges
        for i in 0..3 {
            let relative_pose = create_translated_pose(1.0, 0.0, 0.0);
            let info = nalgebra::Matrix6::identity();
            graph.add_edge(i, i + 1, relative_pose, info);
        }

        // Add loop closure edge (closing the loop from pose 3 back to pose 0)
        let closure_pose = create_translated_pose(-3.0, 0.0, 0.0);
        let closure_info = nalgebra::Matrix6::identity();
        graph.add_edge(3, 0, closure_pose, closure_info);

        assert_eq!(graph.num_edges(), 4);
        assert_eq!(graph.num_poses(), 4);
    }

    #[test]
    fn test_pose_edge_consistency() {
        let mut graph = PoseGraph::new();

        let pose1 = create_identity_pose();
        let pose2 = create_translated_pose(1.0, 1.0, 0.0);

        let id1 = graph.add_pose(pose1);
        let id2 = graph.add_pose(pose2);

        let relative_pose = Pose::new(Matrix3::identity(), Vector3::new(1.0, 1.0, 0.0));
        let info = nalgebra::Matrix6::identity();

        graph.add_edge(id1, id2, relative_pose, info);

        // Verify edge was added correctly
        assert_eq!(graph.edges[0].from, id1);
        assert_eq!(graph.edges[0].to, id2);
        assert!(
            (graph.edges[0].relative_pose.translation.x - 1.0).abs() < 1e-6
                && (graph.edges[0].relative_pose.translation.y - 1.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_multiple_edges_same_node() {
        let mut graph = PoseGraph::new();

        // Create central node with multiple connections
        graph.add_pose(create_identity_pose()); // Node 0
        graph.add_pose(create_translated_pose(1.0, 0.0, 0.0)); // Node 1
        graph.add_pose(create_translated_pose(0.0, 1.0, 0.0)); // Node 2
        graph.add_pose(create_translated_pose(-1.0, 0.0, 0.0)); // Node 3

        let info = nalgebra::Matrix6::identity();

        // Connect all to central node
        graph.add_edge(0, 1, create_translated_pose(1.0, 0.0, 0.0), info);
        graph.add_edge(0, 2, create_translated_pose(0.0, 1.0, 0.0), info);
        graph.add_edge(0, 3, create_translated_pose(-1.0, 0.0, 0.0), info);

        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.num_poses(), 4);
    }

    #[test]
    fn test_edge_information_matrix() {
        let mut graph = PoseGraph::new();

        graph.add_pose(create_identity_pose());
        graph.add_pose(create_translated_pose(1.0, 0.0, 0.0));

        let relative_pose = create_translated_pose(1.0, 0.0, 0.0);
        let mut info = nalgebra::Matrix6::identity();
        info[(0, 0)] = 100.0; // High weight on x-translation

        graph.add_edge(0, 1, relative_pose, info);

        assert!((graph.edges[0].information[(0, 0)] - 100.0).abs() < 1e-6);
        assert!((graph.edges[0].information[(1, 1)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_pose_graph() {
        let mut graph = PoseGraph::new();
        let n_poses = 100;

        // Add linear trajectory
        for i in 0..n_poses {
            let pose = create_translated_pose(i as f64 * 0.1, 0.0, 0.0);
            graph.add_pose(pose);
        }

        // Add sequential edges
        for i in 0..(n_poses - 1) {
            let relative_pose = create_translated_pose(0.1, 0.0, 0.0);
            let info = nalgebra::Matrix6::identity();
            graph.add_edge(i, i + 1, relative_pose, info);
        }

        assert_eq!(graph.num_poses(), n_poses);
        assert_eq!(graph.num_edges(), n_poses - 1);
    }

    #[test]
    fn test_dense_pose_graph() {
        let mut graph = PoseGraph::new();
        let n = 10;

        // Create fully connected graph
        for i in 0..n {
            let pose = create_translated_pose(i as f64, 0.0, 0.0);
            graph.add_pose(pose);
        }

        // Add edges between all pairs
        let info = nalgebra::Matrix6::identity();
        for i in 0..n {
            for j in (i + 1)..n {
                let relative_pose = create_translated_pose((j - i) as f64, 0.0, 0.0);
                graph.add_edge(i, j, relative_pose, info);
            }
        }

        let expected_edges = (n * (n - 1)) / 2;
        assert_eq!(graph.num_edges(), expected_edges);
    }

    #[test]
    fn test_pose_graph_edge_structure() {
        let mut graph = PoseGraph::new();

        graph.add_pose(create_identity_pose());
        graph.add_pose(create_translated_pose(1.0, 0.0, 0.0));

        let relative_pose = create_translated_pose(1.0, 0.0, 0.0);
        let info = nalgebra::Matrix6::identity();
        graph.add_edge(0, 1, relative_pose, info);

        let edge = &graph.edges[0];
        assert_eq!(edge.from, 0);
        assert_eq!(edge.to, 1);
        assert!((edge.relative_pose.translation.x - 1.0).abs() < 1e-6);
        assert!((edge.relative_pose.translation.y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_circular_trajectory() {
        let mut graph = PoseGraph::new();

        // Create four poses in a square
        graph.add_pose(create_translated_pose(0.0, 0.0, 0.0)); // (0,0)
        graph.add_pose(create_translated_pose(1.0, 0.0, 0.0)); // (1,0)
        graph.add_pose(create_translated_pose(1.0, 1.0, 0.0)); // (1,1)
        graph.add_pose(create_translated_pose(0.0, 1.0, 0.0)); // (0,1)

        let info = nalgebra::Matrix6::identity();

        // Connect in order
        graph.add_edge(0, 1, create_translated_pose(1.0, 0.0, 0.0), info);
        graph.add_edge(1, 2, create_translated_pose(0.0, 1.0, 0.0), info);
        graph.add_edge(2, 3, create_translated_pose(-1.0, 0.0, 0.0), info);
        graph.add_edge(3, 0, create_translated_pose(0.0, -1.0, 0.0), info); // Loop closure

        assert_eq!(graph.num_edges(), 4);
        assert_eq!(graph.num_poses(), 4);
    }
}
