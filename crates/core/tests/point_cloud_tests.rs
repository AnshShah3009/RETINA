use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};

#[test]
fn test_point_cloud_result_handling() {
    let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)];
    let cloud = PointCloud::new(points);

    // 1. Valid colors
    let colors = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
    let cloud_with_colors = cloud.clone().with_colors(colors);
    assert!(cloud_with_colors.is_ok());

    // 2. Invalid colors (count mismatch)
    let bad_colors = vec![Point3::new(1.0, 0.0, 0.0)];
    let cloud_bad_colors = cloud.clone().with_colors(bad_colors);
    assert!(cloud_bad_colors.is_err());
    assert!(cloud_bad_colors
        .unwrap_err()
        .to_string()
        .contains("Color count"));

    // 3. Valid normals
    let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, 1.0)];
    let cloud_with_normals = cloud.clone().with_normals(normals);
    assert!(cloud_with_normals.is_ok());

    // 4. Invalid normals (count mismatch)
    let bad_normals = vec![Vector3::new(0.0, 0.0, 1.0)];
    let cloud_bad_normals = cloud.with_normals(bad_normals);
    assert!(cloud_bad_normals.is_err());
    assert!(cloud_bad_normals
        .unwrap_err()
        .to_string()
        .contains("Normal count"));
}
