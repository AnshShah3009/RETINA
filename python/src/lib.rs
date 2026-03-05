use cv_3d::mesh::reconstruction;
use cv_core::point_cloud::PointCloud;
use cv_core::{CameraIntrinsics, KeyPoint};
use cv_dnn::DnnNet;
use cv_features::{fast, gftt, harris};
use cv_optimize::isam2::Isam2;
use cv_point_cloud;
use cv_runtime::orchestrator::{scheduler, WorkloadHint};
use cv_slam::Slam as RustSlam;
use cv_videoio::{open_video, VideoCapture};
use nalgebra::{Point3, Vector3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let mat = pose.matrix();
        let flat_mat: Vec<f32> = mat.iter().map(|&v| v as f32).collect();

        Ok((flat_mat, tracked))
    }
}

// ============== Feature Detection Bindings ==============

#[pyclass]
pub struct PyKeyPoints {
    keypoints: Vec<KeyPoint>,
}

#[pymethods]
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn optimize(&self) -> PyResult<()> {
        self.inner
            .optimize()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
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
pub struct PyTriangleMesh {
    vertices: Vec<Point3<f32>>,
    faces: Vec<[usize; 3]>,
}

#[pymethods]
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
    Ok(())
}
