use nalgebra::{Matrix3, Point3, Vector3};

/// Oriented bounding box computed via PCA.
#[derive(Debug, Clone)]
pub struct OrientedBoundingBox {
    /// Center of the bounding box.
    pub center: Point3<f64>,
    /// Three principal axes (unit vectors), ordered by decreasing extent.
    pub axes: [Vector3<f64>; 3],
    /// Half-extents along each principal axis.
    pub extents: Vector3<f64>,
}

/// Compute the axis-aligned bounding box.
///
/// # Returns
/// `(min_corner, max_corner)`.
///
/// # Panics
/// Panics if `points` is empty.
pub fn compute_aabb(points: &[Point3<f64>]) -> (Point3<f64>, Point3<f64>) {
    assert!(!points.is_empty(), "compute_aabb: empty point cloud");
    let mut min = points[0];
    let mut max = points[0];
    for p in &points[1..] {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    (min, max)
}

/// Compute an oriented bounding box using PCA on the point cloud.
///
/// The three principal axes of the cloud define the OBB orientation; the
/// extents are the half-widths of the cloud projected onto each axis.
pub fn compute_obb(points: &[Point3<f64>]) -> OrientedBoundingBox {
    assert!(!points.is_empty(), "compute_obb: empty point cloud");

    let n = points.len() as f64;

    // Centroid.
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    centroid /= n;

    // 3x3 covariance matrix.
    let mut cov = Matrix3::<f64>::zeros();
    for p in points {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    cov /= n;

    // Eigen decomposition (nalgebra's symmetric eigen — fine for 3x3).
    let eig = cov.symmetric_eigen();

    // Sort eigenvalues descending (largest extent first).
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let axes = [
        eig.eigenvectors.column(order[0]).into_owned(),
        eig.eigenvectors.column(order[1]).into_owned(),
        eig.eigenvectors.column(order[2]).into_owned(),
    ];

    // Project points onto axes and find extents.
    let mut mins = [f64::MAX; 3];
    let mut maxs = [f64::MIN; 3];
    for p in points {
        let d = p.coords - centroid;
        for (k, ax) in axes.iter().enumerate() {
            let proj = d.dot(ax);
            mins[k] = mins[k].min(proj);
            maxs[k] = maxs[k].max(proj);
        }
    }

    let extents = Vector3::new(
        (maxs[0] - mins[0]) / 2.0,
        (maxs[1] - mins[1]) / 2.0,
        (maxs[2] - mins[2]) / 2.0,
    );

    // Adjust center to be the midpoint of the projected extents.
    let center_offset = Vector3::new(
        (maxs[0] + mins[0]) / 2.0,
        (maxs[1] + mins[1]) / 2.0,
        (maxs[2] + mins[2]) / 2.0,
    );
    let center = Point3::from(
        centroid
            + axes[0] * center_offset.x
            + axes[1] * center_offset.y
            + axes[2] * center_offset.z,
    );

    OrientedBoundingBox {
        center,
        axes,
        extents,
    }
}
