#![forbid(unsafe_code)]
use cv_3d::mesh::TriangleMesh;
use cv_core::{Polygon, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use nalgebra::Point3;

#[test]
fn test_cpu_nms_boxes() {
    let device = CpuBackend::new().unwrap();

    // Create some overlapping boxes
    // [x1, y1, x2, y2, score]
    let data = vec![
        10.0, 10.0, 50.0, 50.0, 0.9, // Box 0
        15.0, 15.0, 55.0, 55.0, 0.8, // Box 1 (high overlap with 0)
        100.0, 100.0, 150.0, 150.0, 0.7, // Box 2 (no overlap)
    ];

    let shape = TensorShape::new(1, 3, 5);
    // Explicitly specify CpuStorage to avoid ambiguity
    let tensor: Tensor<f32, cv_core::storage::CpuStorage<f32>> =
        Tensor::from_vec(data, shape).unwrap();

    let kept = device.nms_boxes(&tensor, 0.5).unwrap();

    assert_eq!(kept.len(), 2);
    assert!(kept.contains(&0));
    assert!(kept.contains(&2));
    assert!(!kept.contains(&1));
}

#[test]
fn test_cpu_nms_rotated_boxes() {
    let device = CpuBackend::new().unwrap();

    // [cx, cy, w, h, angle, score]
    let data = vec![
        30.0, 30.0, 40.0, 40.0, 0.0, 0.9, // Box 0
        32.0, 32.0, 40.0, 40.0, 10.0, 0.8, // Box 1 (high overlap)
        120.0, 120.0, 40.0, 40.0, 45.0, 0.7, // Box 2
    ];

    let shape = TensorShape::new(1, 3, 6);
    let tensor: Tensor<f32, cv_core::storage::CpuStorage<f32>> =
        Tensor::from_vec(data, shape).unwrap();

    let kept = device.nms_rotated_boxes(&tensor, 0.5).unwrap();

    assert_eq!(kept.len(), 2);
    assert!(kept.contains(&0));
    assert!(kept.contains(&2));
}

#[test]
fn test_cpu_nms_polygons() {
    let device = CpuBackend::new().unwrap();

    let mut p1 = Polygon::new(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
    let mut p2 = Polygon::new(vec![[2.0, 2.0], [12.0, 2.0], [12.0, 12.0], [2.0, 12.0]]);
    let mut p3 = Polygon::new(vec![[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]]);

    p1.ensure_counter_clockwise();
    p2.ensure_counter_clockwise();
    p3.ensure_counter_clockwise();

    let polygons = vec![p1, p2, p3];
    let scores = vec![0.9, 0.8, 0.7];

    let kept = device.nms_polygons(&polygons, &scores, 0.4).unwrap();

    assert_eq!(kept.len(), 2);
    assert!(kept.contains(&0));
    assert!(kept.contains(&2));
}

#[test]
fn test_mesh_sampling() {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];
    let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

    let mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);
    let pc = mesh.sample_points(100);

    assert_eq!(pc.points.len(), 100);

    // Check bounds
    let (min, max) = mesh.bounds();
    for p in &pc.points {
        assert!(p.x >= min.x - 1e-6 && p.x <= max.x + 1e-6);
        assert!(p.y >= min.y - 1e-6 && p.y <= max.y + 1e-6);
        assert!(p.z >= min.z - 1e-6 && p.z <= max.z + 1e-6);
    }
}
