use cv_3d::mesh::reconstruction;
use cv_core::point_cloud::PointCloud;
use cv_core::tensor::{CpuTensor, TensorShape};
use cv_core::{CameraIntrinsics, KeyPoint};
use cv_dnn::DnnNet;
use cv_features::{fast, gftt, harris};
use cv_optimize::factor_graph::{
    FactorGraph, GNConfig, Key, LMParams, NoiseModel, Values, Variable,
};
use cv_optimize::factors::{BetweenFactor, PriorFactor, RangeFactor};
use cv_optimize::isam2::Isam2;
use cv_runtime::orchestrator::{scheduler, WorkloadHint};
use cv_slam::Slam as RustSlam;
use cv_videoio::{open_video, VideoCapture};
use nalgebra::{DVector, Point3, Vector3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// ── helpers ───────────────────────────────────────────────────────────────────

fn pts_from_py(points: &[(f32, f32, f32)]) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|&(x, y, z)| Point3::new(x, y, z))
        .collect()
}

fn normals_to_py(normals: Vec<Vector3<f32>>) -> Vec<(f32, f32, f32)> {
    normals.into_iter().map(|n| (n.x, n.y, n.z)).collect()
}

#[pyclass]
#[derive(Clone, Copy)]
pub enum PyWorkloadHint {
    Latency,
    Throughput,
    PowerSave,
    Default,
}

impl From<PyWorkloadHint> for WorkloadHint {
    fn from(hint: PyWorkloadHint) -> Self {
        match hint {
            PyWorkloadHint::Latency => WorkloadHint::Latency,
            PyWorkloadHint::Throughput => WorkloadHint::Throughput,
            PyWorkloadHint::PowerSave => WorkloadHint::PowerSave,
            PyWorkloadHint::Default => WorkloadHint::Default,
        }
    }
}

// ============== Video Bindings ==============

#[pyclass]
pub struct PyVideoCapture {
    inner: Box<dyn VideoCapture>,
}

#[pymethods]
impl PyVideoCapture {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let cap = open_video(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: cap })
    }

    pub fn read(&mut self) -> PyResult<Option<Vec<u8>>> {
        match self.inner.read() {
            Ok(img) => Ok(Some(img.into_vec())),
            Err(_) => Ok(None),
        }
    }

    pub fn is_opened(&self) -> bool {
        self.inner.is_opened()
    }
}

// ============== DNN Bindings ==============

#[pyclass]
pub struct PyDnnNet {
    inner: DnnNet,
    runner: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PyDnnNet {
    #[new]
    pub fn new(path: &str, hint: PyWorkloadHint) -> PyResult<Self> {
        let inner = DnnNet::load(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let runner = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner, runner })
    }

    pub fn forward(&self, image: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<Vec<f32>>> {
        let gray =
            image::GrayImage::from_raw(width as u32, height as u32, image).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image dimensions")
            })?;

        let dyn_img = image::DynamicImage::ImageLuma8(gray);

        let input_tensor = self
            .inner
            .preprocess(&dyn_img, &self.runner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let outputs = self
            .inner
            .forward(&input_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut py_outputs = Vec::new();
        for tensor in outputs {
            if let Ok(slice) = tensor.as_slice() {
                py_outputs.push(slice.to_vec());
            }
        }

        Ok(py_outputs)
    }
}

// ============== SLAM Bindings ==============

#[pyclass]
pub struct PySlam {
    inner: RustSlam,
    _group: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PySlam {
    #[new]
    pub fn new(hint: PyWorkloadHint) -> PyResult<Self> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let group = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let intrinsics = CameraIntrinsics::new_ideal(640, 480);
        Ok(Self {
            inner: RustSlam::new(group.clone(), intrinsics),
            _group: group,
        })
    }

    pub fn process_frame(
        &mut self,
        image: Vec<u8>,
        width: usize,
        height: usize,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let (pose, tracked) = self
            .inner
            .process_image(&gray)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        let mat = pose.matrix();
        let flat_mat: Vec<f32> = mat.iter().map(|&v| v as f32).collect();

        Ok((flat_mat, tracked))
    }
}

// ============== Feature Detection Bindings ==============

#[pyclass]
#[allow(clippy::new_without_default)]
pub struct PyKeyPoints {
    keypoints: Vec<KeyPoint>,
}

#[pymethods]
#[allow(clippy::new_without_default)]
impl PyKeyPoints {
    #[new]
    pub fn new() -> Self {
        Self {
            keypoints: Vec::new(),
        }
    }

    pub fn __len__(&self) -> usize {
        self.keypoints.len()
    }

    pub fn to_list(&self) -> Vec<(f64, f64)> {
        self.keypoints.iter().map(|kp| (kp.x, kp.y)).collect()
    }
}

#[pyclass]
pub struct PyFeatureDetector;

#[pymethods]
impl PyFeatureDetector {
    #[staticmethod]
    pub fn fast_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        threshold: u8,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = fast::fast_detect(&gray, threshold, 1000);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn harris_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        k: f64,
        threshold: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::harris_detect(&gray, 3, 3, k, threshold);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn gftt_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        max_corners: usize,
        quality_level: f64,
        min_distance: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = gftt::gftt_detect(&gray, max_corners, quality_level, min_distance);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn shi_tomasi_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        max_corners: usize,
        quality_level: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::shi_tomasi_detect(&gray, max_corners, quality_level, 1.0);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }
}

// ============== ISAM2 Bindings ==============

#[pyclass]
pub struct PyIsam2 {
    inner: Isam2,
}

#[pymethods]
impl PyIsam2 {
    #[new]
    pub fn new(optimize_on_update: bool, batch_optimize: bool) -> Self {
        Self {
            inner: Isam2::with_config(optimize_on_update, batch_optimize),
        }
    }

    pub fn add_pose(&self, id: usize, x: f64, y: f64, z: f64) {
        self.inner.add_pose(id, Vector3::new(x, y, z));
    }

    pub fn add_point(&self, id: usize, x: f64, y: f64, z: f64) {
        self.inner.add_point(id, Point3::new(x, y, z));
    }

    pub fn add_factor(&self, from: usize, to: usize, tx: f64, ty: f64, tz: f64, noise: f64) {
        use nalgebra::DVector;
        let measurement = DVector::from_vec(vec![tx, ty, tz]);
        self.inner.add_factor(from, to, measurement, noise);
    }

    pub fn update(&self) -> PyResult<()> {
        self.inner
            .update()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    pub fn optimize(&self) -> PyResult<()> {
        self.inner
            .optimize()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    pub fn get_pose(&self, id: usize) -> Option<(f64, f64, f64)> {
        self.inner.get_pose(id).map(|p| (p.x, p.y, p.z))
    }

    pub fn get_point(&self, id: usize) -> Option<(f64, f64, f64)> {
        self.inner.get_point(id).map(|p| (p.x, p.y, p.z))
    }

    pub fn get_all_poses(&self) -> Vec<(usize, f64, f64, f64)> {
        self.inner
            .get_all_poses()
            .into_iter()
            .map(|(id, p)| (id, p.x, p.y, p.z))
            .collect()
    }

    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    pub fn num_factors(&self) -> usize {
        self.inner.num_factors()
    }
}

// ============== Point Cloud & Mesh Bindings ==============

#[pyclass]
pub struct PyPointCloud {
    points: Vec<Point3<f32>>,
    normals: Option<Vec<Vector3<f32>>>,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    pub fn new(points: Vec<(f32, f32, f32)>) -> Self {
        Self {
            points: points
                .iter()
                .map(|(x, y, z)| Point3::new(*x, *y, *z))
                .collect(),
            normals: None,
        }
    }

    #[staticmethod]
    pub fn from_list(points: Vec<(f32, f32, f32)>) -> Self {
        Self {
            points: points
                .iter()
                .map(|(x, y, z)| Point3::new(*x, *y, *z))
                .collect(),
            normals: None,
        }
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.points.len() * 3);
        for p in &self.points {
            result.push(p.x);
            result.push(p.y);
            result.push(p.z);
        }
        result
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn set_normals(&mut self, normals: Vec<(f32, f32, f32)>) {
        self.normals = Some(
            normals
                .iter()
                .map(|(x, y, z)| Vector3::new(*x, *y, *z))
                .collect(),
        );
    }

    /// Get normals as a flat list: [nx0,ny0,nz0, nx1,ny1,nz1, ...].
    /// Returns `None` if normals have not been estimated yet.
    pub fn get_normals_flat(&self) -> Option<Vec<f32>> {
        self.normals.as_ref().map(|ns| {
            let mut out = Vec::with_capacity(ns.len() * 3);
            for n in ns {
                out.push(n.x);
                out.push(n.y);
                out.push(n.z);
            }
            out
        })
    }

    /// Get normals as list of (nx, ny, nz) tuples.
    pub fn get_normals(&self) -> Option<Vec<(f32, f32, f32)>> {
        self.normals
            .as_ref()
            .map(|ns| ns.iter().map(|n| (n.x, n.y, n.z)).collect())
    }

    /// Estimate normals in-place. `method` selects the algorithm:
    ///
    /// - `"auto"` (default) — GPU if available, else CPU
    /// - `"cpu"` — voxel-hash kNN + analytic eigensolver
    /// - `"gpu"` — Morton sort + WebGPU PCA (Metal on Apple Silicon)
    /// - `"hybrid"` — CPU kNN + GPU batch eigenvectors
    /// - `"approx_cross"` — fast 2-neighbour cross-product (~3× faster)
    /// - `"approx_integral"` — fast ring cross-product (~2.5× faster)
    #[pyo3(signature = (k=15, method="auto"))]
    pub fn estimate_normals(&mut self, k: usize, method: &str) {
        let normals = match method {
            "cpu" => cv_point_cloud::estimate_normals_cpu(&self.points, k),
            "gpu" => cv_point_cloud::estimate_normals_gpu(&self.points, k),
            "hybrid" => cv_point_cloud::estimate_normals_hybrid(&self.points, k),
            "approx_cross" => cv_point_cloud::estimate_normals_approx_cross(&self.points),
            "approx_integral" => cv_point_cloud::estimate_normals_approx_integral(&self.points),
            _ => cv_point_cloud::estimate_normals_auto(&self.points, k),
        };
        self.normals = Some(normals);
    }

    pub fn has_normals(&self) -> bool {
        self.normals.is_some()
    }

    pub fn points_to_list(&self) -> Vec<(f32, f32, f32)> {
        self.points.iter().map(|p| (p.x, p.y, p.z)).collect()
    }
}

#[pyclass]
#[allow(clippy::new_without_default)]
pub struct PyTriangleMesh {
    vertices: Vec<Point3<f32>>,
    faces: Vec<[usize; 3]>,
}

#[pymethods]
#[allow(clippy::new_without_default)]
impl PyTriangleMesh {
    #[new]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn add_vertex(&mut self, x: f32, y: f32, z: f32) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Point3::new(x, y, z));
        idx
    }

    pub fn add_face(&mut self, v0: usize, v1: usize, v2: usize) {
        self.faces.push([v0, v1, v2]);
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn to_obj(&self) -> String {
        let mut output = String::new();
        for v in &self.vertices {
            output.push_str(&format!("v {} {} {}\n", v.x, v.y, v.z));
        }
        for f in &self.faces {
            output.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        output
    }
}

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

// ============== Tensor Bindings ==============

#[pyclass]
pub struct PyTensor {
    data: Vec<f32>,
    shape: (usize, usize, usize),
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    #[staticmethod]
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    #[staticmethod]
    pub fn ones(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }
}

// ============== Runtime Bindings ==============

#[pyclass]
pub struct PyRuntime;

#[pymethods]
impl PyRuntime {
    #[staticmethod]
    pub fn get_global_load() -> PyResult<Vec<(u32, usize)>> {
        let _s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(vec![])
    }

    #[staticmethod]
    pub fn get_num_devices() -> usize {
        1
    }
}

// ── Module-level normal estimation functions ──────────────────────────────────
//
// These accept a list of (x,y,z) tuples and return a list of (nx,ny,nz) tuples,
// making them easy to use with numpy:
//   normals = np.array(cv_native.estimate_normals_auto(points.tolist(), k=15))

/// Auto-select the fastest available backend (GPU → CPU).
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_auto(points: Vec<(f32, f32, f32)>, k: usize) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_auto(
        &pts_from_py(&points),
        k,
    ))
}

/// CPU-only: voxel-hash kNN + analytic eigensolver. ~19 ms / 40k pts.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_cpu(points: Vec<(f32, f32, f32)>, k: usize) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_cpu(
        &pts_from_py(&points),
        k,
    ))
}

/// GPU: Morton sort (CPU) + WebGPU PCA. Uses Metal on Apple Silicon. ~17 ms / 40k pts.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_gpu(points: Vec<(f32, f32, f32)>, k: usize) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_gpu(
        &pts_from_py(&points),
        k,
    ))
}

/// Hybrid: CPU kNN + GPU batch eigenvectors. Best for large clouds + discrete GPU.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_hybrid(points: Vec<(f32, f32, f32)>, k: usize) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_hybrid(
        &pts_from_py(&points),
        k,
    ))
}

/// Fast approximate: 2-neighbour cross-product. ~3× faster, slightly less accurate.
#[pyfunction]
fn estimate_normals_approx_cross(points: Vec<(f32, f32, f32)>) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_approx_cross(&pts_from_py(
        &points,
    )))
}

/// Fast approximate: ring cross-product average. ~2.5× faster, smooth results.
#[pyfunction]
fn estimate_normals_approx_integral(points: Vec<(f32, f32, f32)>) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_approx_integral(
        &pts_from_py(&points),
    ))
}

/// O(n) normals from a structured depth image — ideal for RGBD / depth cameras.
///
/// Args:
///     depth: flat list of depth values, row-major, H×W elements (metres or mm).
///     width, height: image dimensions.
///     fx, fy: focal lengths in pixels.
///     cx, cy: principal point in pixels.
///
/// Returns list of (nx,ny,nz) tuples in camera space, one per pixel.
#[pyfunction]
fn estimate_normals_from_depth(
    depth: Vec<f32>,
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Vec<(f32, f32, f32)> {
    normals_to_py(cv_point_cloud::estimate_normals_from_depth(
        &depth, width, height, fx, fy, cx, cy,
    ))
}

// ── NumPy-native normal estimation ───────────────────────────────────────────
// These accept an (n,3) float32 ndarray and return an (n,3) float32 ndarray.
// No Python tuple/list allocation — minimal overhead between numpy and Rust.
//
// Usage:
//   normals = cv_native.estimate_normals_np(pts_np, k=15)   # pts_np: (n,3) float32

fn ndarray_to_points(pts: &PyReadonlyArray2<f32>) -> Vec<Point3<f32>> {
    let arr = pts.as_array();
    let n = arr.shape()[0];
    (0..n)
        .map(|i| Point3::new(arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
        .collect()
}

fn normals_to_ndarray<'py>(
    py: Python<'py>,
    normals: Vec<Vector3<f32>>,
) -> Bound<'py, PyArray2<f32>> {
    use numpy::ndarray::Array2;
    let n = normals.len();
    let mut arr = Array2::<f32>::zeros((n, 3));
    for (i, v) in normals.iter().enumerate() {
        arr[[i, 0]] = v.x;
        arr[[i, 1]] = v.y;
        arr[[i, 2]] = v.z;
    }
    arr.into_pyarray_bound(py)
}

/// Estimate normals — numpy array in, numpy array out (fastest Python path).
/// Auto-selects GPU if available, else CPU.
///
/// Args:
///     points: (n, 3) float32 ndarray.
///     k:      neighbour count (default 15).
/// Returns:
///     (n, 3) float32 ndarray of unit normals.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_point_cloud::estimate_normals_auto(&pts, k))
}

/// CPU normals — numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_cpu_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_point_cloud::estimate_normals_cpu(&pts, k))
}

/// GPU normals — numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_gpu_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_point_cloud::estimate_normals_gpu(&pts, k))
}

/// Hybrid normals — numpy array in/out.
#[pyfunction]
#[pyo3(signature = (points, k=15))]
fn estimate_normals_hybrid_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_point_cloud::estimate_normals_hybrid(&pts, k))
}

/// Fast approximate (cross-product) normals — numpy array in/out.
#[pyfunction]
fn estimate_normals_approx_cross_np<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let pts = ndarray_to_points(&points);
    normals_to_ndarray(py, cv_point_cloud::estimate_normals_approx_cross(&pts))
}

/// Depth image normals — numpy array in/out (O(n), fastest path for RGBD).
///
/// Args:
///     depth:  (H*W,) float32 ndarray, row-major, metric depth.
///     width, height: image dimensions.
///     fx, fy, cx, cy: pinhole camera intrinsics.
/// Returns:
///     (H*W, 3) float32 ndarray.
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
) -> Bound<'py, PyArray2<f32>> {
    let d = depth.as_slice().unwrap();
    let normals = cv_point_cloud::estimate_normals_from_depth(d, width, height, fx, fy, cx, cy);
    normals_to_ndarray(py, normals)
}

// ============== Factor Graph Bindings ==============

#[pyclass]
pub struct PyFactorGraph {
    inner: FactorGraph,
}

#[pymethods]
impl PyFactorGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: FactorGraph::new(),
        }
    }

    /// Add a prior factor anchoring a variable to a fixed value.
    ///
    /// Args:
    ///     key: Variable key (u64).
    ///     value: Prior value as a list of floats.
    ///     sigma: Isotropic noise standard deviation.
    fn add_prior(&mut self, key: u64, value: Vec<f64>, sigma: f64) {
        let var = vec_to_variable(&value);
        let dim = var.dim();
        self.inner.add(PriorFactor::new(
            Key(key),
            var,
            NoiseModel::Isotropic(sigma, dim),
        ));
    }

    /// Add a between factor measuring relative transform between two variables.
    ///
    /// Args:
    ///     key1, key2: Variable keys.
    ///     measurement: Relative measurement as a list of floats.
    ///     sigma: Isotropic noise standard deviation.
    fn add_between(&mut self, key1: u64, key2: u64, measurement: Vec<f64>, sigma: f64) {
        let var = vec_to_variable(&measurement);
        let dim = var.dim();
        self.inner.add(BetweenFactor::new(
            Key(key1),
            Key(key2),
            var,
            NoiseModel::Isotropic(sigma, dim),
        ));
    }

    /// Add a range factor measuring Euclidean distance between two variables.
    ///
    /// Args:
    ///     key1, key2: Variable keys.
    ///     distance: Measured distance.
    ///     sigma: Noise standard deviation.
    fn add_range(&mut self, key1: u64, key2: u64, distance: f64, sigma: f64) {
        self.inner.add(RangeFactor::new(
            Key(key1),
            Key(key2),
            distance,
            NoiseModel::Isotropic(sigma, 1),
        ));
    }

    /// Optimize using Gauss-Newton.
    ///
    /// Args:
    ///     initial: Dict mapping key (u64) to value (list of floats).
    ///     max_iters: Maximum number of iterations.
    ///
    /// Returns:
    ///     Dict mapping key to optimized value.
    #[pyo3(signature = (initial, max_iters=100))]
    fn optimize_gn(
        &self,
        initial: HashMap<u64, Vec<f64>>,
        max_iters: usize,
    ) -> PyResult<HashMap<u64, Vec<f64>>> {
        let values = hashmap_to_values(&initial);
        let config = GNConfig {
            max_iters,
            ..GNConfig::default()
        };
        let result = self
            .inner
            .optimize_gn(&values, &config)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(values_to_hashmap(&result))
    }

    /// Optimize using Levenberg-Marquardt.
    ///
    /// Args:
    ///     initial: Dict mapping key (u64) to value (list of floats).
    ///     max_iters: Maximum number of iterations.
    ///
    /// Returns:
    ///     Dict mapping key to optimized value.
    #[pyo3(signature = (initial, max_iters=100))]
    fn optimize_lm(
        &self,
        initial: HashMap<u64, Vec<f64>>,
        max_iters: usize,
    ) -> PyResult<HashMap<u64, Vec<f64>>> {
        let values = hashmap_to_values(&initial);
        let config = LMParams {
            max_iters,
            ..LMParams::default()
        };
        let result = self
            .inner
            .optimize_lm(&values, &config)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(values_to_hashmap(&result))
    }

    /// Compute total squared whitened error.
    fn total_error(&self, values: HashMap<u64, Vec<f64>>) -> f64 {
        let vals = hashmap_to_values(&values);
        self.inner.total_error(&vals)
    }

    /// Number of factors in the graph.
    fn num_factors(&self) -> usize {
        self.inner.len()
    }
}

/// Convert a flat Vec<f64> to the appropriate Variable type based on length.
fn vec_to_variable(v: &[f64]) -> Variable {
    match v.len() {
        1 => Variable::Scalar(v[0]),
        2 => Variable::Point2([v[0], v[1]]),
        3 => Variable::Point3(nalgebra::Point3::new(v[0], v[1], v[2])),
        _ => Variable::Vector(DVector::from_vec(v.to_vec())),
    }
}

fn hashmap_to_values(map: &HashMap<u64, Vec<f64>>) -> Values {
    let mut values = Values::new();
    for (&key, val) in map {
        values.insert(Key(key), vec_to_variable(val));
    }
    values
}

fn values_to_hashmap(values: &Values) -> HashMap<u64, Vec<f64>> {
    let mut map = HashMap::new();
    for (&key, var) in &values.values {
        map.insert(key.0, var.to_vector().as_slice().to_vec());
    }
    map
}

// ============== Hidden Point Removal Bindings ==============

/// Determine which points in a 3D point cloud are visible from a viewpoint.
///
/// Uses the Katz-Tal-Basri spherical flipping + convex hull algorithm.
///
/// Args:
///     points: List of (x, y, z) tuples.
///     viewpoint: (x, y, z) tuple for the camera/eye position.
///     radius: Flipping radius. Pass 0.0 for auto-computation.
///
/// Returns:
///     List of indices of visible points.
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

    let result = cv_point_cloud::hidden_point_removal::hidden_point_removal(&pts, &vp, radius)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    Ok(result.visible_indices)
}

// ============== Image Inpainting Bindings ==============

/// Inpaint masked regions using the Telea Fast Marching Method.
///
/// Args:
///     image: Flat list of f32 pixel values (CHW layout: channels * height * width).
///     channels: Number of image channels.
///     height: Image height.
///     width: Image width.
///     mask: Flat list of u8 mask values (H * W). Non-zero = inpaint region.
///     radius: Neighbourhood radius for weighted averaging.
///
/// Returns:
///     Flat list of f32 pixel values with masked region filled.
#[pyfunction]
fn inpaint_telea(
    image: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    mask: Vec<u8>,
    radius: f32,
) -> PyResult<Vec<f32>> {
    let img_tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mask_tensor = CpuTensor::<u8>::from_vec(mask, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = cv_photo::inpaint_telea(&img_tensor, &mask_tensor, radius)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Inpaint masked regions using Navier-Stokes PDE diffusion.
///
/// Args:
///     image: Flat list of f32 pixel values (CHW layout).
///     channels: Number of image channels.
///     height: Image height.
///     width: Image width.
///     mask: Flat list of u8 mask values (H * W). Non-zero = inpaint region.
///     radius: Smoothing extent parameter.
///     iterations: Number of diffusion iterations.
///
/// Returns:
///     Flat list of f32 pixel values with masked region filled.
#[pyfunction]
#[pyo3(signature = (image, channels, height, width, mask, radius=3.0, iterations=100))]
fn inpaint_ns(
    image: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    mask: Vec<u8>,
    radius: f32,
    iterations: u32,
) -> PyResult<Vec<f32>> {
    let img_tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mask_tensor = CpuTensor::<u8>::from_vec(mask, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = cv_photo::inpaint_ns(&img_tensor, &mask_tensor, radius, iterations)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============== Scientific Function Bindings ==============

/// Compute the 1D FFT of real-valued input.
///
/// Args:
///     data: List of real-valued samples.
///
/// Returns:
///     List of (real, imaginary) pairs representing the complex spectrum.
#[pyfunction]
fn fft(data: Vec<f64>) -> Vec<(f64, f64)> {
    cv_scientific::fft::fft(&data)
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Compute the 1D inverse FFT.
///
/// Args:
///     data: List of (real, imaginary) pairs.
///
/// Returns:
///     List of (real, imaginary) pairs (imaginary should be near zero for real signals).
#[pyfunction]
fn ifft(data: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let complex: Vec<rustfft::num_complex::Complex<f64>> = data
        .iter()
        .map(|&(re, im)| rustfft::num_complex::Complex::new(re, im))
        .collect();
    cv_scientific::fft::ifft(&complex)
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Minimize a scalar function using Nelder-Mead simplex method.
///
/// Args:
///     py: Python interpreter handle.
///     f: Callable taking a list of floats, returning a float.
///     x0: Initial guess as a list of floats.
///     max_iters: Maximum number of iterations (default 1000).
///
/// Returns:
///     Tuple of (optimal_x, optimal_value).
#[pyfunction]
#[pyo3(signature = (f, x0, max_iters=1000))]
fn minimize_nelder_mead(
    _py: Python,
    f: PyObject,
    x0: Vec<f64>,
    max_iters: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let config = cv_scientific::optimize::NelderMeadConfig {
        max_iters,
        ..cv_scientific::optimize::NelderMeadConfig::default()
    };
    let result = cv_scientific::optimize::minimize_nelder_mead(
        |x| {
            Python::with_gil(|py| {
                let args = (x.to_vec(),);
                f.call1(py, args)
                    .and_then(|r| r.extract::<f64>(py))
                    .unwrap_or(f64::MAX)
            })
        },
        &x0,
        &config,
    );
    Ok((result.x, result.fun))
}

/// Simple linear regression: y = slope * x + intercept.
///
/// Args:
///     x: Independent variable data.
///     y: Dependent variable data.
///
/// Returns:
///     Tuple of (slope, intercept, r_squared).
#[pyfunction]
fn linear_regression(x: Vec<f64>, y: Vec<f64>) -> PyResult<(f64, f64, f64)> {
    let result = cv_scientific::stats::linregress(&x, &y)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok((result.slope, result.intercept, result.r_squared))
}

// ============== 2D Geometry Bindings ==============

/// Compute the convex hull of a set of 2D points.
///
/// Args:
///     points: List of (x, y) tuples.
///
/// Returns:
///     List of (x, y) tuples forming the convex hull vertices (closed polygon).
#[pyfunction]
fn convex_hull_2d(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    use cv_scientific::geometry2d::{convex_hull, Point2D};
    let pts: Vec<Point2D> = points.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    let hull = convex_hull(&pts);
    hull.exterior.iter().map(|p| (p.x, p.y)).collect()
}

/// Compute the Delaunay triangulation of a set of 2D points.
///
/// Args:
///     points: List of (x, y) tuples.
///
/// Returns:
///     List of (i, j, k) index triples, each representing a triangle.
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
///
/// Args:
///     vertices: List of (x, y) tuples forming the polygon boundary.
///               The polygon is automatically closed if the last point differs from the first.
///
/// Returns:
///     Unsigned area of the polygon.
#[pyfunction]
fn polygon_area(vertices: Vec<(f64, f64)>) -> f64 {
    use cv_scientific::geometry2d::{Point2D, Polygon};
    let mut pts: Vec<Point2D> = vertices.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    // Close the polygon if not already closed.
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
///
/// Args:
///     x: Point x-coordinate.
///     y: Point y-coordinate.
///     polygon: List of (x, y) tuples forming the polygon boundary.
///
/// Returns:
///     True if the point is inside the polygon.
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

// ============== Point Cloud Filtering Bindings ==============

/// Statistical outlier removal. Returns (inlier_points, inlier_indices).
///
/// Removes points whose mean distance to `nb_neighbors` nearest neighbours
/// exceeds `mean + std_ratio * stddev`.
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
        cv_point_cloud::filters::statistical_outlier_removal(&pts, nb_neighbors, std_ratio);
    let out_pts = inliers.iter().map(|p| (p.x, p.y, p.z)).collect();
    (out_pts, indices)
}

/// Radius outlier removal. Returns (inlier_points, inlier_indices).
///
/// Removes points that have fewer than `min_neighbors` within the given `radius`.
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
    let (inliers, indices) =
        cv_point_cloud::filters::radius_outlier_removal(&pts, radius, min_neighbors);
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
    let result = cv_point_cloud::filters::voxel_downsample(&pts, None, None, voxel_size);
    result.points.iter().map(|p| (p.x, p.y, p.z)).collect()
}

// ============== HDR & Tone Mapping Bindings ==============

/// Reinhard tone mapping: maps HDR radiance to LDR [0,1].
///
/// Args:
///     hdr_data: Flat list of f64 pixel values (CHW layout).
///     channels: Number of channels.
///     height, width: Image dimensions.
///     gamma: Gamma correction exponent (e.g. 2.2).
///
/// Returns:
///     Flat list of f32 pixel values in [0, 1].
#[pyfunction]
fn tonemap_reinhard(
    hdr_data: Vec<f64>,
    channels: usize,
    height: usize,
    width: usize,
    gamma: f64,
) -> PyResult<Vec<f32>> {
    let tensor = CpuTensor::<f64>::from_vec(hdr_data, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cv_photo::tonemap_reinhard(&tensor, gamma)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Mertens exposure fusion: fuse multiple LDR images into one.
///
/// Args:
///     images: List of flat f32 image data (each CHW layout, values in [0,1]).
///     channels: Number of channels.
///     height, width: Image dimensions.
///
/// Returns:
///     Flat list of f32 pixel values — the fused image.
#[pyfunction]
fn merge_mertens(
    images: Vec<Vec<f32>>,
    channels: usize,
    height: usize,
    width: usize,
) -> PyResult<Vec<f32>> {
    let tensors: Result<Vec<CpuTensor<f32>>, _> = images
        .into_iter()
        .map(|data| CpuTensor::<f32>::from_vec(data, TensorShape::new(channels, height, width)))
        .collect();
    let tensors =
        tensors.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cv_photo::merge_mertens(&tensors)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============== Denoising Bindings ==============

/// Non-local means denoising for single-channel images.
///
/// Args:
///     image: Flat list of f32 pixel values (1, H, W layout).
///     height, width: Image dimensions.
///     h: Filter strength.
///     template_window: Size of comparison patch (odd number, e.g. 7).
///     search_window: Size of search area (odd number, e.g. 21).
///
/// Returns:
///     Flat list of f32 pixel values — the denoised image.
#[pyfunction]
fn fast_nl_means_denoising(
    image: Vec<f32>,
    height: usize,
    width: usize,
    h: f32,
    template_window: usize,
    search_window: usize,
) -> PyResult<Vec<f32>> {
    let tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cv_photo::fast_nl_means_denoising(&tensor, h, template_window, search_window)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============== Signal Processing Bindings ==============

/// Welch's method for power spectral density estimation.
///
/// Args:
///     data: Input signal samples.
///     nperseg: Segment length for FFT.
///     sample_rate: Sampling frequency in Hz.
///
/// Returns:
///     Tuple of (frequencies, psd) vectors.
#[pyfunction]
fn welch_psd(data: Vec<f64>, nperseg: usize, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    cv_scientific::signal::welch(&data, nperseg, None, sample_rate)
}

/// Design a Butterworth lowpass filter.
///
/// Args:
///     order: Filter order (>= 1).
///     cutoff: Cutoff frequency in Hz.
///     sample_rate: Sampling frequency in Hz.
///
/// Returns:
///     Tuple of (b, a) coefficient vectors (transfer function form).
#[pyfunction]
fn butter_lowpass(order: usize, cutoff: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    cv_scientific::signal::butter(order, cutoff, sample_rate)
}

/// Zero-phase forward-backward digital filtering.
///
/// Args:
///     b: Numerator coefficients.
///     a: Denominator coefficients.
///     x: Input signal.
///
/// Returns:
///     Filtered signal (same length as x).
#[pyfunction]
fn filtfilt(b: Vec<f64>, a: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    cv_scientific::signal::filtfilt(&b, &a, &x)
}

/// Find peaks (local maxima) in a 1D signal.
///
/// Args:
///     data: Input signal.
///     min_height: Optional minimum peak height.
///     min_distance: Optional minimum distance between peaks.
///
/// Returns:
///     List of peak indices.
#[pyfunction]
#[pyo3(signature = (data, min_height=None, min_distance=None))]
fn find_peaks(data: Vec<f64>, min_height: Option<f64>, min_distance: Option<usize>) -> Vec<usize> {
    cv_scientific::signal::find_peaks(&data, min_height, min_distance)
}

// ============== Distance Transform Bindings ==============

/// Compute the distance transform of a binary image.
///
/// Each output pixel holds the distance to the nearest zero-valued pixel.
///
/// Args:
///     binary: Flat list of f32 pixel values (H * W). Zero = background.
///     height, width: Image dimensions.
///     dist_type: Distance metric — "l1", "l2", or "chessboard".
///
/// Returns:
///     Flat list of f32 distance values.
#[pyfunction]
fn distance_transform(
    binary: Vec<f32>,
    height: usize,
    width: usize,
    dist_type: &str,
) -> PyResult<Vec<f32>> {
    let dt = match dist_type {
        "l1" => cv_imgproc::distance_transform::DistanceType::L1,
        "l2" => cv_imgproc::distance_transform::DistanceType::L2,
        "chessboard" => cv_imgproc::distance_transform::DistanceType::Chessboard,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown distance type '{}'. Use 'l1', 'l2', or 'chessboard'.",
                other
            )));
        }
    };
    let tensor = CpuTensor::<f32>::from_vec(binary, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cv_imgproc::distance_transform::distance_transform(&tensor, dt)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============== Sparse & Spatial Bindings ==============

/// Query a KD-tree for the k nearest neighbours of a point.
///
/// Args:
///     points: List of points (each a list of floats, all same dimensionality).
///     query: Query point (list of floats).
///     k: Number of neighbours to return.
///
/// Returns:
///     List of (index, distance) pairs sorted by ascending distance.
#[pyfunction]
fn kdtree_query(points: Vec<Vec<f64>>, query: Vec<f64>, k: usize) -> PyResult<Vec<(usize, f64)>> {
    let tree = cv_scientific::spatial::KDTree::new(&points)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(tree.query(&query, k))
}

/// Solve a sparse linear system Ax = b using conjugate gradient.
///
/// Args:
///     triplets: List of (row, col, value) entries defining A.
///     nrows: Number of rows of A.
///     ncols: Number of columns of A.
///     b: Right-hand side vector.
///
/// Returns:
///     Solution vector x.
#[pyfunction]
fn sparse_solve_cg(
    triplets: Vec<(usize, usize, f64)>,
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let mat = cv_scientific::sparse::CsrMatrix::from_triplets(nrows, ncols, &triplets);
    let b_vec = DVector::from_vec(b);
    let result = cv_scientific::sparse::cg_solve(&mat, &b_vec, 1000, 1e-10)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(result.as_slice().to_vec())
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkloadHint>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PyDnnNet>()?;
    m.add_class::<PyKeyPoints>()?;
    m.add_class::<PyFeatureDetector>()?;
    m.add_class::<PyIsam2>()?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyTriangleMesh>()?;
    m.add_class::<PyMeshReconstruction>()?;
    m.add_class::<PyTensor>()?;
    // Normal estimation — direct functions
    m.add_function(wrap_pyfunction!(estimate_normals_auto, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_approx_cross, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_approx_integral, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_from_depth, m)?)?;
    // NumPy-native versions (fastest — no Python list/tuple allocation)
    m.add_function(wrap_pyfunction!(estimate_normals_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_cpu_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_gpu_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_hybrid_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_approx_cross_np, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals_from_depth_np, m)?)?;
    // Factor graph
    m.add_class::<PyFactorGraph>()?;
    // Hidden point removal
    m.add_function(wrap_pyfunction!(hidden_point_removal, m)?)?;
    // Image inpainting
    m.add_function(wrap_pyfunction!(inpaint_telea, m)?)?;
    m.add_function(wrap_pyfunction!(inpaint_ns, m)?)?;
    // Scientific functions
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    // 2D geometry
    m.add_function(wrap_pyfunction!(convex_hull_2d, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_area, m)?)?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    // Point cloud filtering
    m.add_function(wrap_pyfunction!(statistical_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(radius_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    // HDR & tone mapping
    m.add_function(wrap_pyfunction!(tonemap_reinhard, m)?)?;
    m.add_function(wrap_pyfunction!(merge_mertens, m)?)?;
    // Denoising
    m.add_function(wrap_pyfunction!(fast_nl_means_denoising, m)?)?;
    // Signal processing
    m.add_function(wrap_pyfunction!(welch_psd, m)?)?;
    m.add_function(wrap_pyfunction!(butter_lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(filtfilt, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    // Distance transform
    m.add_function(wrap_pyfunction!(distance_transform, m)?)?;
    // Sparse & spatial
    m.add_function(wrap_pyfunction!(kdtree_query, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_solve_cg, m)?)?;
    Ok(())
}
