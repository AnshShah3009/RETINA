use cv_runtime::orchestrator::ResourceGroup;
use nalgebra::{Matrix3x4, Point2, Point3};
use rayon::prelude::*;

/// Error type for SfM operations
#[derive(Debug, thiserror::Error)]
pub enum SfmError {
    #[error("Triangulation failed: {0}")]
    TriangulationFailed(String),
}

pub type Result<T> = std::result::Result<T, SfmError>;

/// Triangulate a single 3D point from two views.
///
/// Uses the DLT (Direct Linear Transformation) algorithm.
/// p1, p2 are the 2D points in homogeneous coordinates (normalized or pixel).
/// P1, P2 are the 3x4 projection matrices for each view.
pub fn triangulate_point_dlt(
    p1: &Point2<f64>,
    p2: &Point2<f64>,
    proj1: &Matrix3x4<f64>,
    proj2: &Matrix3x4<f64>,
) -> Result<Point3<f64>> {
    let mut a = nalgebra::Matrix4::<f64>::zeros();

    // Row 1: x1 * P1[2] - P1[0]
    for j in 0..4 {
        a[(0, j)] = p1.x * proj1[(2, j)] - proj1[(0, j)];
    }
    // Row 2: y1 * P1[2] - P1[1]
    for j in 0..4 {
        a[(1, j)] = p1.y * proj1[(2, j)] - proj1[(1, j)];
    }
    // Row 3: x2 * P2[2] - P2[0]
    for j in 0..4 {
        a[(2, j)] = p2.x * proj2[(2, j)] - proj2[(0, j)];
    }
    // Row 4: y2 * P2[2] - P2[1]
    for j in 0..4 {
        a[(3, j)] = p2.y * proj2[(2, j)] - proj2[(1, j)];
    }

    // Solve using SVD: Ax = 0
    let svd = a.svd(true, true);
    let vt = svd
        .v_t
        .ok_or_else(|| SfmError::TriangulationFailed("SVD failed".to_string()))?;

    // The solution is the last row of V^T (last column of V)
    let last_row = vt.row(3);

    // Homogeneous to Euclidean
    let w = last_row[3];
    if w.abs() < 1e-12 {
        return Err(SfmError::TriangulationFailed(
            "Point at infinity".to_string(),
        ));
    }

    Ok(Point3::new(
        last_row[0] / w,
        last_row[1] / w,
        last_row[2] / w,
    ))
}

/// Triangulate multiple points from two views.
pub fn triangulate_points(
    points1: &[Point2<f64>],
    points2: &[Point2<f64>],
    proj1: &Matrix3x4<f64>,
    proj2: &Matrix3x4<f64>,
) -> Result<Vec<Point3<f64>>> {
    let s = cv_runtime::orchestrator::scheduler()
        .map_err(|e| SfmError::TriangulationFailed(e.to_string()))?;
    let group = s
        .get_default_group()
        .map_err(|e| SfmError::TriangulationFailed(e.to_string()))?;
    triangulate_points_ctx(points1, points2, proj1, proj2, &group)
}

/// Triangulate multiple points with context-aware parallelism
pub fn triangulate_points_ctx(
    points1: &[Point2<f64>],
    points2: &[Point2<f64>],
    proj1: &Matrix3x4<f64>,
    proj2: &Matrix3x4<f64>,
    group: &ResourceGroup,
) -> Result<Vec<Point3<f64>>> {
    if points1.len() != points2.len() {
        return Err(SfmError::TriangulationFailed(
            "Point counts must match".to_string(),
        ));
    }

    group.run(|| {
        points1
            .par_iter()
            .zip(points2.par_iter())
            .map(|(p1, p2)| triangulate_point_dlt(p1, p2, proj1, proj2))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulation_simple() {
        let p1 = Matrix3x4::<f64>::identity();
        let p2 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pt1 = Point2::new(0.0, 0.0);
        let pt2 = Point2::new(-0.2, 0.0);

        let triangulated = triangulate_point_dlt(&pt1, &pt2, &p1, &p2).unwrap();

        assert!((triangulated.x - 0.0).abs() < 1e-6);
        assert!((triangulated.y - 0.0).abs() < 1e-6);
        assert!((triangulated.z - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_triangulation_point_at_different_depth() {
        let p1 = Matrix3x4::<f64>::identity();
        let p2 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pt1 = Point2::new(0.0, 0.0);
        let pt2 = Point2::new(-0.1, 0.0);

        let triangulated = triangulate_point_dlt(&pt1, &pt2, &p1, &p2).unwrap();

        assert!(triangulated.z > 0.0);
    }

    #[test]
    fn test_triangulation_identical_cameras() {
        let p1 = Matrix3x4::<f64>::identity();
        let p2 = Matrix3x4::<f64>::identity();

        let pt1 = Point2::new(0.1, 0.2);
        let pt2 = Point2::new(0.1, 0.2);

        let result = triangulate_point_dlt(&pt1, &pt2, &p1, &p2);

        assert!(result.is_ok());
    }

    #[test]
    fn test_triangulation_error_display() {
        let err = SfmError::TriangulationFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_triangulation_off_axis_point() {
        let p1 = Matrix3x4::<f64>::identity();
        let p2 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pt1 = Point2::new(0.2, 0.3);
        let pt2 = Point2::new(-0.05, 0.3);

        let result = triangulate_point_dlt(&pt1, &pt2, &p1, &p2);
        assert!(result.is_ok());

        let triangulated = result.unwrap();
        assert!(triangulated.z > 0.0);
    }

    #[test]
    fn test_triangulation_large_baseline() {
        let p1 = Matrix3x4::<f64>::identity();
        let p2 =
            Matrix3x4::<f64>::new(1.0, 0.0, 0.0, -10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pt1 = Point2::new(0.0, 0.0);
        let pt2 = Point2::new(-0.1, 0.0);

        let result = triangulate_point_dlt(&pt1, &pt2, &p1, &p2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_triangulation_symmetric_setup() {
        let p1 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p2 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pt1 = Point2::new(0.1, 0.2);
        let pt2 = Point2::new(-0.1, 0.2);

        let result = triangulate_point_dlt(&pt1, &pt2, &p1, &p2);
        assert!(result.is_ok());
    }
}
