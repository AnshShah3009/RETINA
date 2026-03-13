use cv_core::{PointCloud, Rect};
use cv_scientific::geometry::*;
use cv_scientific::point_cloud::*;
use nalgebra::Point3;

#[test]
fn test_vectorized_iou_and_polygon_iou() {
    let r1 = Rect::new(0.0, 0.0, 10.0, 10.0);
    let r2 = Rect::new(5.0, 5.0, 10.0, 10.0);

    let boxes1 = vec![r1];
    let boxes2 = vec![r2];

    let ious = vectorized_iou(&boxes1, &boxes2);
    // Intersection: 5x5 = 25. Union: 100 + 100 - 25 = 175. 25/175 = 1/7 approx 0.1428
    assert!((ious[[0, 0]] - 0.142857).abs() < 1e-5);

    use geo::LineString;
    use geo::Polygon;
    let p1 = Polygon::new(
        LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]),
        vec![],
    );
    let p2 = Polygon::new(
        LineString::from(vec![
            (5.0, 5.0),
            (15.0, 5.0),
            (15.0, 15.0),
            (5.0, 15.0),
            (5.0, 5.0),
        ]),
        vec![],
    );

    let polys1 = vec![p1];
    let polys2 = vec![p2];
    let poly_ious = vectorized_polygon_iou(&polys1, &polys2);
    assert!((poly_ious[[0, 0]] - 0.142857).abs() < 1e-5);
}

#[test]
fn test_point_cloud_normals_parallel() {
    let mut points = Vec::new();
    for x in 0..10 {
        for y in 0..10 {
            points.push(Point3::new(x as f32, y as f32, 0.0));
        }
    }
    let mut pc = PointCloud::new(points);
    estimate_normals(&mut pc, 10);

    assert!(pc.normals.is_some());
    let ns = pc.normals.as_ref().unwrap();
    // Normals should be vertical [0,0,1] or [0,0,-1]
    for n in ns {
        assert!(n.z.abs() > 0.9);
    }
}

#[test]
fn test_segment_plane_parallel() {
    let mut points = Vec::new();
    // Ground plane
    for x in 0..20 {
        for y in 0..20 {
            points.push(Point3::new(x as f32, y as f32, 0.0));
        }
    }
    // Outliers
    points.push(Point3::new(0.0, 0.0, 10.0));

    let pc = PointCloud::new(points);
    let (model, inliers) = segment_plane(&pc, 0.1, 3, 100);

    assert!(model.is_some());
    let [a, b, c, d] = model.unwrap();
    // Equation: z = 0 => 0x + 0y + 1z + 0 = 0
    assert!(a.abs() < 0.1);
    assert!(b.abs() < 0.1);
    assert!(c.abs() > 0.9);
    assert!(d.abs() < 0.1);
    assert_eq!(inliers.len(), 400);
}

#[test]
fn test_outlier_removal_parallel() {
    let mut points = Vec::new();
    // A tight cluster
    for _ in 0..10 {
        points.push(Point3::new(0.0, 0.0, 0.0));
    }
    // An outlier
    points.push(Point3::new(10.0, 10.0, 10.0));

    let pc = PointCloud::new(points);

    // Radius removal
    let (pc_filtered, inliers) = remove_radius_outliers(&pc, 1.0, 5);
    assert_eq!(pc_filtered.len(), 10);
    assert!(!inliers.contains(&10));

    // Statistical removal
    let (pc_filtered_stat, inliers_stat) = remove_statistical_outliers(&pc, 5, 1.0);
    assert_eq!(pc_filtered_stat.len(), 10);
    assert!(!inliers_stat.contains(&10));
}
