use cv_optimize::factor_graph::{
    FactorGraph, GNConfig, Key, LMParams, NoiseModel, Values, Variable,
};
use cv_optimize::factors::{BetweenFactor, PriorFactor, RangeFactor};
use cv_optimize::isam2::Isam2;
use nalgebra::{DVector, Point3, Vector3};
use pyo3::prelude::*;
use std::collections::HashMap;

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
    fn add_range(&mut self, key1: u64, key2: u64, distance: f64, sigma: f64) {
        self.inner.add(RangeFactor::new(
            Key(key1),
            Key(key2),
            distance,
            NoiseModel::Isotropic(sigma, 1),
        ));
    }

    /// Optimize using Gauss-Newton.
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

/// Minimize a scalar function using Nelder-Mead simplex method.
#[pyfunction]
#[pyo3(signature = (f, x0, max_iters=1000))]
fn minimize_nelder_mead(
    _py: Python,
    f: PyObject,
    x0: Vec<f64>,
    max_iters: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let config = cv_optimize::general::NelderMeadConfig {
        max_iters,
        ..cv_optimize::general::NelderMeadConfig::default()
    };
    let result = cv_optimize::general::minimize_nelder_mead(
        |x: &[f64]| {
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

/// Solve a sparse linear system Ax = b using conjugate gradient.
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIsam2>()?;
    m.add_class::<PyFactorGraph>()?;
    m.add_function(wrap_pyfunction!(minimize_nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_solve_cg, m)?)?;
    Ok(())
}
