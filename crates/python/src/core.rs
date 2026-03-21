use cv_core::KeyPoint;
use nalgebra::{Point3, Vector3};
use pyo3::prelude::*;

#[pyclass]
pub struct PyPointCloud {
    pub(crate) points: Vec<Point3<f32>>,
    pub(crate) normals: Option<Vec<Vector3<f32>>>,
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
    /// - `"auto"` (default) -- GPU if available, else CPU
    /// - `"cpu"` -- voxel-hash kNN + analytic eigensolver
    /// - `"gpu"` -- Morton sort + WebGPU PCA (Metal on Apple Silicon)
    /// - `"hybrid"` -- CPU kNN + GPU batch eigenvectors
    /// - `"approx_integral"` -- fast ring cross-product (~2.5x faster)
    #[pyo3(signature = (k=15, method="auto"))]
    pub fn estimate_normals(&mut self, k: usize, method: &str) {
        let normals = match method {
            "cpu" => cv_3d::estimate_normals_cpu(&self.points, k),
            "gpu" => cv_3d::estimate_normals_gpu(&self.points, k),
            "hybrid" => cv_3d::estimate_normals_hybrid(&self.points, k),
            "approx_integral" => cv_3d::estimate_normals_approx_integral(&self.points),
            _ => cv_3d::estimate_normals_auto(&self.points, k),
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
    pub(crate) vertices: Vec<Point3<f32>>,
    pub(crate) faces: Vec<[usize; 3]>,
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

#[pyclass]
#[allow(clippy::new_without_default)]
pub struct PyKeyPoints {
    pub(crate) keypoints: Vec<KeyPoint>,
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyTriangleMesh>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyKeyPoints>()?;
    Ok(())
}
