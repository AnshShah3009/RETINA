use cv_3d::mesh::processing::*;
use cv_3d::mesh::TriangleMesh;
use nalgebra::Point3;

#[test]
fn test_mesh_normals_and_area() {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];

    let mut mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);

    // Test surface area
    let area = mesh.surface_area();
    assert!((area - 0.5).abs() < 1e-6);

    // Test face normals
    let normals = mesh.compute_face_normals();
    assert_eq!(normals.len(), 1);
    assert!((normals[0].z - 1.0).abs() < 1e-6);

    // Test vertex normals
    mesh.compute_vertex_normals();
    assert!(mesh.normals.is_some());
    assert!((mesh.normals.as_ref().unwrap()[0].z - 1.0).abs() < 1e-6);
}

#[test]
fn test_laplacian_smooth() {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.3, 0.3, 0.5), // Point sticking out
    ];
    let faces = vec![[0, 1, 3], [1, 2, 3], [2, 0, 3]];

    let mut mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);
    let original_z = mesh.vertices[3].z;

    laplacian_smooth(&mut mesh, 5, 0.5);

    // Point 3 should have moved towards the others (z should decrease)
    assert!(mesh.vertices[3].z < original_z);
}

#[test]
fn test_odometry_point_to_plane_basic() {
    use cv_3d::odometry::*;
    use cv_3d::tsdf::CameraIntrinsics;

    let width = 32;
    let height = 32;
    let mut depth = vec![0.0f32; width * height];
    let mut target_depth = vec![0.0f32; width * height];

    let intrinsics = CameraIntrinsics {
        fx: 32.0,
        fy: 32.0,
        cx: 16.0,
        cy: 16.0,
    };

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            // A paraboloid shape: z = 1.0 + 0.2*xf^2 + 0.3*yf^2
            let xf = (x as f32 - 16.0) / 16.0;
            let yf = (y as f32 - 16.0) / 16.0;
            let z = 1.0 + 0.2 * xf * xf + 0.3 * yf * yf;
            depth[idx] = z;
            target_depth[idx] = z + 0.1; // Shifted by 0.1m
        }
    }

    let result = compute_rgbd_odometry(
        &depth,
        &target_depth,
        None,
        None,
        &intrinsics,
        width,
        height,
        OdometryMethod::PointToPlane,
    );

    assert!(result.is_some());
    let res = result.unwrap();
    // Translation in Z should be approx 0.1
    let tz = res.transformation.column(3)[2];
    assert!((tz - 0.1).abs() < 0.2);
}

#[test]
fn test_simplify_vertex_clustering() {
    let mut vertices = Vec::new();
    // Create two clusters of points
    for i in 0..10 {
        vertices.push(Point3::new(0.01 * i as f32, 0.0, 0.0));
        vertices.push(Point3::new(1.0 + 0.01 * i as f32, 0.0, 0.0));
    }
    let faces = vec![];
    let mut mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);

    // Clustering with voxel size 0.5 should reduce to 2 vertices
    simplify_vertex_clustering(&mut mesh, 0.5);

    assert_eq!(mesh.vertices.len(), 2);
}
