use cv_optimize::pose_graph::{Edge, PoseGraph};
use nalgebra::{Isometry3, Matrix6, Vector3};

#[test]
fn test_pose_graph_simple_loop() {
    let mut graph = PoseGraph::new();

    // Node 0 at origin, fixed
    graph.add_node(0, Isometry3::identity());
    graph.set_fixed(0);

    // Node 1 at x=1.0
    graph.add_node(1, Isometry3::translation(1.1, 0.0, 0.0)); // Initial guess slightly off

    // Edge 0 -> 1: relative x=1.0
    let info = Matrix6::identity();
    graph.add_edge(0, 1, Isometry3::translation(1.0, 0.0, 0.0), info);

    // Optimize
    let error = graph.optimize(10).unwrap();
    println!("Final error: {}", error);

    let pose1 = graph.nodes[&1];
    assert!((pose1.translation.vector.x - 1.0).abs() < 1e-3);
    assert!(error < 1e-5);
}

#[test]
fn test_pose_graph_rotation() {
    let mut graph = PoseGraph::new();
    graph.add_node(0, Isometry3::identity());
    graph.set_fixed(0);

    // Node 1 moved and rotated 90 deg around Z
    // Initial guess is identity
    graph.add_node(1, Isometry3::identity());

    let measurement = Isometry3::new(Vector3::new(1.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.5708));
    graph.add_edge(0, 1, measurement, Matrix6::identity());

    graph.optimize(20).unwrap();

    let pose1 = graph.nodes[&1];
    assert!((pose1.translation.vector.x - 1.0).abs() < 1e-2);
    assert!((pose1.rotation.scaled_axis().z - 1.5708).abs() < 1e-2);
}
