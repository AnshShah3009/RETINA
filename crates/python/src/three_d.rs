use crate::core::{PyPointCloud, PyTriangleMesh};
use crate::helpers::{
    ndarray_to_points, normals_to_ndarray, normals_to_py, panic_payload_to_string, pts_from_py,
};
use cv_3d::mesh::reconstruction;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct PyMeshReconstruction;

#[pymethods]
impl PyMeshReconstruction {
    #[staticmethod]
    pub fn create_sphere(center: (f32, f32, f32), radius: f32, num_points: usize) -> PyPointCloud {
        let pc = reconstruction::create_sphere_point_cloud(
            Point3::new(center.0, center.1, center.2),
            radius,
            num_points,
        );
        PyPointCloud {
            points: pc.points,
            normals: pc.normals,
        }
    }

    #[staticmethod]
    pub fn create_plane(
        origin: (f32, f32, f32),
        normal: (f32, f32, f32),
        size: f32,
        num_points: usize,
    ) -> PyPointCloud {
        let pc = reconstruction::create_plane_point_cloud(
            Point3::new(origin.0, origin.1, origin.2),
            Vector3::new(normal.0, normal.1, normal.2),
            size,
            num_points,
        );
        PyPointCloud {
            points: pc.points,
            normals: pc.normals,
        }
    }

    #[staticmethod]
    pub fn ball_pivoting(cloud: &PyPointCloud, ball_radius: f32) -> PyTriangleMesh {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        let mesh = reconstruction::ball_pivoting(&pc, ball_radius);
        PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        }
    }

    #[staticmethod]
    pub fn alpha_shapes(cloud: &PyPointCloud, alpha: f32) -> PyTriangleMesh {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        let mesh = reconstruction::alpha_shapes(&pc, alpha);
        PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        }
    }

    #[staticmethod]
    pub fn poisson(cloud: &PyPointCloud, depth: usize) -> Option<PyTriangleMesh> {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        reconstruction::poisson_reconstruction(&pc, depth, 1.0).map(|mesh| PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        })
    }
}

// ── Module-level normal estimation functions ──────────────────────────────────

/// Auto-select the fastest available backend (GPU -> CPU).
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_auto(points: Vec<(f32, f32, f32)>, k: usize) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_auto(&pts_from_py(&points), k))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

/// CPU-only: voxel-hash kNN + analytic eigensolver.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_cpu(points: Vec<(f32, f32, f32)>, k: usize) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_cpu(&pts_from_py(&points), k))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

/// GPU: Morton sort (CPU) + WebGPU PCA.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_gpu(points: Vec<(f32, f32, f32)>, k: usize) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_gpu(&pts_from_py(&points), k))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

/// Hybrid: CPU kNN + GPU batch eigenvectors.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_hybrid(
    points: Vec<(f32, f32, f32)>,
    k: usize,
) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_hybrid(&pts_from_py(&points), k))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

/// Fast approximate: ring cross-product average.
#[pyfunction]
fn estimate_normals_approx_integral(
    points: Vec<(f32, f32, f32)>,
) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_approx_integral(&pts_from_py(
            &points,
        )))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

/// O(n) normals from a structured depth image.
#[pyfunction]
fn estimate_normals_from_depth(
    depth: Vec<f32>,
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> PyResult<Vec<(f32, f32, f32)>> {
    std::panic::catch_unwind(|| {
        normals_to_py(cv_3d::estimate_normals_from_depth(
            &depth, width, height, fx, fy, cx, cy,
        ))
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(panic_payload_to_string(e)))
}

// ── NumPy-native normal estimation ───────────────────────────────────────────

/// Estimate normals -- numpy array in, numpy array out (fastest Python path).
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_3d::estimate_normals_auto(&pts, k))
}

/// CPU normals -- numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_cpu_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_3d::estimate_normals_cpu(&pts, k))
}

/// GPU normals -- numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_gpu_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_3d::estimate_normals_gpu(&pts, k))
}

/// Hybrid normals -- numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_hybrid_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_3d::estimate_normals_hybrid(&pts, k))
}

/// Depth image normals -- numpy array in/out (O(n), fastest path for RGBD).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn estimate_normals_from_depth_np<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray1<f32>,
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let d = depth.as_slice().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input depth array must be contiguous in memory",
        )
    })?;
    let normals = cv_3d::estimate_normals_from_depth(d, width, height, fx, fy, cx, cy);
    Ok(normals_to_ndarray(py, normals))
}

// ── Point cloud operations ───────────────────────────────────────────────────

/// Determine which points in a 3D point cloud are visible from a viewpoint.
#[pyfunction]
#[pyo3(signature = (points, viewpoint, radius=0.0))]
fn hidden_point_removal(
    points: Vec<(f64, f64, f64)>,
    viewpoint: (f64, f64, f64),
    radius: f64,
) -> PyResult<Vec<usize>> {
    let pts: Vec<nalgebra::Point3<f64>> = points
        .iter()
        .map(|&(x, y, z)| nalgebra::Point3::new(x, y, z))
        .collect();
    let vp = nalgebra::Point3::new(viewpoint.0, viewpoint.1, viewpoint.2);

    let result = cv_3d::hidden_point_removal::hidden_point_removal(&pts, &vp, radius)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    Ok(result.visible_indices)
}

/// Statistical outlier removal. Returns (inlier_points, inlier_indices).
#[pyfunction]
fn statistical_outlier_removal(
    points: Vec<(f64, f64, f64)>,
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<(f64, f64, f64)>, Vec<usize>) {
    let pts: Vec<nalgebra::Point3<f64>> = points
        .iter()
        .map(|&(x, y, z)| nalgebra::Point3::new(x, y, z))
        .collect();
    let (inliers, indices) =
        cv_3d::filters::statistical_outlier_removal(&pts, nb_neighbors, std_ratio);
    let out_pts = inliers.iter().map(|p| (p.x, p.y, p.z)).collect();
    (out_pts, indices)
}

/// Radius outlier removal. Returns (inlier_points, inlier_indices).
#[pyfunction]
fn radius_outlier_removal(
    points: Vec<(f64, f64, f64)>,
    radius: f64,
    min_neighbors: usize,
) -> (Vec<(f64, f64, f64)>, Vec<usize>) {
    let pts: Vec<nalgebra::Point3<f64>> = points
        .iter()
        .map(|&(x, y, z)| nalgebra::Point3::new(x, y, z))
        .collect();
    let (inliers, indices) = cv_3d::filters::radius_outlier_removal(&pts, radius, min_neighbors);
    let out_pts = inliers.iter().map(|p| (p.x, p.y, p.z)).collect();
    (out_pts, indices)
}

/// Voxel downsampling. Groups points into cubic voxels and returns centroids.
#[pyfunction]
fn voxel_downsample(points: Vec<(f64, f64, f64)>, voxel_size: f64) -> Vec<(f64, f64, f64)> {
    let pts: Vec<nalgebra::Point3<f64>> = points
        .iter()
        .map(|&(x, y, z)| nalgebra::Point3::new(x, y, z))
        .collect();
    let result = cv_3d::filters::voxel_downsample(&pts, None, None, voxel_size);
    result.points.iter().map(|p| (p.x, p.y, p.z)).collect()
}

// ── KD-tree ──────────────────────────────────────────────────────────────────

/// Query a KD-tree for the k nearest neighbours of a point.
#[pyfunction]
fn kdtree_query(points: Vec<Vec<f64>>, query: Vec<f64>, k: usize) -> PyResult<Vec<(usize, f64)>> {
    let tree = cv_scientific::spatial::KDTree::new(&points)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(tree.query(&query, k))
}

// ── 2D Geometry ──────────────────────────────────────────────────────────────

/// Compute the convex hull of a set of 2D points.
#[pyfunction]
fn convex_hull_2d(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    use cv_scientific::geometry2d::{convex_hull, Point2D};
    let pts: Vec<Point2D> = points.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    let hull = convex_hull(&pts);
    hull.exterior.iter().map(|p| (p.x, p.y)).collect()
}

/// Compute the Delaunay triangulation of a set of 2D points.
#[pyfunction]
fn delaunay(points: Vec<(f64, f64)>) -> Vec<(usize, usize, usize)> {
    use cv_scientific::geometry2d::{delaunay_triangulation, Point2D};
    let pts: Vec<Point2D> = points.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    delaunay_triangulation(&pts)
        .into_iter()
        .map(|[a, b, c]| (a, b, c))
        .collect()
}

/// Compute the area of a polygon defined by its vertices.
#[pyfunction]
fn polygon_area(vertices: Vec<(f64, f64)>) -> f64 {
    use cv_scientific::geometry2d::{Point2D, Polygon};
    let mut pts: Vec<Point2D> = vertices.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    if pts.len() >= 2
        && (pts.first().unwrap().x != pts.last().unwrap().x
            || pts.first().unwrap().y != pts.last().unwrap().y)
    {
        pts.push(pts[0].clone());
    }
    let poly = Polygon::new(pts, vec![]);
    poly.area()
}

/// Test whether a point lies inside a polygon.
#[pyfunction]
fn point_in_polygon(x: f64, y: f64, polygon: Vec<(f64, f64)>) -> bool {
    use cv_scientific::geometry2d::{point_in_polygon as pip, Point2D, Polygon};
    let mut pts: Vec<Point2D> = polygon
        .iter()
        .map(|&(px, py)| Point2D::new(px, py))
        .collect();
    if pts.len() >= 2
        && (pts.first().unwrap().x != pts.last().unwrap().x
            || pts.first().unwrap().y != pts.last().unwrap().y)
    {
        pts.push(pts[0].clone());
    }
    let poly = Polygon::new(pts, vec![]);
    let point = Point2D::new(x, y);
    pip(&point, &poly)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeshReconstruction>()?;
    // Normal estimation -- direct functions
    m.add_function(wrap_pyfunction!(estimate_normals_auto, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_approx_integral, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_from_depth, m)?)?;
    // NumPy-native versions
    m.add_function(wrap_pyfunction!(estimate_normals_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_cpu_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_gpu_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_hybrid_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_from_depth_np, m)?)?;
    // Point cloud operations
    m.add_function(wrap_pyfunction!(hidden_point_removal, m)?)?;
    m.add_function(wrap_pyfunction!(statistical_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(radius_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    // KD-tree
    m.add_function(wrap_pyfunction!(kdtree_query, m)?)?;
    // 2D geometry
    m.add_function(wrap_pyfunction!(convex_hull_2d, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_area, m)?)?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    Ok(())
}
