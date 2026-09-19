//! Roundtrip tests for new I/O formats (glTF, LAS).

#[cfg(feature = "gltf")]
mod gltf_tests {
    use cv_io::gltf_io::{gltf_to_triangle_mesh, read_gltf, write_glb};
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_glb_roundtrip() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        let normals: Vec<Vector3<f32>> = vertices
            .iter()
            .map(|v| Vector3::new(v.x, v.y, v.z).normalize())
            .collect();

        let tmp = std::env::temp_dir().join("test_retina_roundtrip.glb");
        write_glb(&tmp, &vertices, &faces, Some(&normals)).unwrap();

        let meshes = read_gltf(&tmp).unwrap();
        assert!(!meshes.is_empty(), "Should read at least one mesh");

        let mesh = &meshes[0];
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 4);

        // Verify vertices match (within tolerance)
        for (orig, loaded) in vertices.iter().zip(mesh.vertices.iter()) {
            assert!((orig.x - loaded.x).abs() < 1e-4);
            assert!((orig.y - loaded.y).abs() < 1e-4);
            assert!((orig.z - loaded.z).abs() < 1e-4);
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_gltf_to_triangle_mesh_conversion() {
        let gltf_mesh = cv_io::gltf_io::GltfMesh {
            name: "test".to_string(),
            vertices: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: None,
            faces: vec![[0, 1, 2]],
            tex_coords: None,
        };
        let mesh = gltf_to_triangle_mesh(&gltf_mesh);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.faces.len(), 1);
    }
}

#[cfg(feature = "las")]
mod las_tests {
    use cv_io::las_io::{
        filter_by_classification, las_to_point_cloud, point_cloud_to_las, read_las, write_las,
        LasData,
    };
    use nalgebra::Point3;

    fn sample_las_data() -> LasData {
        LasData {
            points: vec![
                Point3::new(1.0, 2.0, 3.0),
                Point3::new(4.0, 5.0, 6.0),
                Point3::new(7.0, 8.0, 9.0),
            ],
            colors: None,
            intensities: Some(vec![0.5, 0.8, 0.3]),
            classifications: Some(vec![2, 3, 2]), // ground, vegetation, ground
            return_numbers: None,
            number_of_returns: None,
            gps_times: None,
            bounds: (1.0, 2.0, 3.0, 7.0, 8.0, 9.0),
            num_points: 3,
        }
    }

    #[test]
    fn test_las_roundtrip() {
        let data = sample_las_data();
        let tmp = std::env::temp_dir().join("test_retina_roundtrip.las");

        write_las(&tmp, &data).unwrap();
        let loaded = read_las(&tmp).unwrap();

        assert_eq!(loaded.num_points, 3);
        for (orig, loaded_pt) in data.points.iter().zip(loaded.points.iter()) {
            assert!((orig.x - loaded_pt.x).abs() < 0.01);
            assert!((orig.y - loaded_pt.y).abs() < 0.01);
            assert!((orig.z - loaded_pt.z).abs() < 0.01);
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_las_to_point_cloud() {
        let data = sample_las_data();
        let cloud = las_to_point_cloud(&data);
        assert_eq!(cloud.points.len(), 3);
    }

    #[test]
    fn test_point_cloud_to_las() {
        let cloud =
            cv_core::PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
        let las = point_cloud_to_las(&cloud);
        assert_eq!(las.num_points, 2);
    }

    #[test]
    fn test_filter_by_classification() {
        let data = sample_las_data();
        let ground = filter_by_classification(&data, 2);
        assert_eq!(ground.num_points, 2); // two ground points
        let veg = filter_by_classification(&data, 3);
        assert_eq!(veg.num_points, 1); // one vegetation point
    }
}
