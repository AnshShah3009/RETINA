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
    f: Py<PyAny>,
    x0: Vec<f64>,
    max_iters: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let config = cv_optimize::general::NelderMeadConfig {
        max_iters,
        ..cv_optimize::general::NelderMeadConfig::default()
    };
    // A failure inside the user's objective is a failure of *this call*, not a
    // large objective value. The `unwrap_or(f64::MAX)` that used to stand here
    // collapsed every Python exception - a raising callback, a callback that
    // returns a non-float, a callback that returns a numpy array - into "the
    // objective was astronomically bad here", so Nelder-Mead happily steered
    // away from the region that raised and converged on a point from a
    // successful (or unvisited) evaluation, then reported `fun = f64::MAX` with
    // a confidence a caller has no way to question.
    //
    // Measured: `minimize_nelder_mead(always_raises, [0.0])` returned
    // `([0.0], 1.7976931348623157e+308)` - the unvisited start point, labelled
    // as a converged result, with no Python exception raised.
    let failure: std::sync::Mutex<Option<PyErr>> = std::sync::Mutex::new(None);
    let result = cv_optimize::general::minimize_nelder_mead(
        |x: &[f64]| {
            Python::attach(|py| {
                // Propagate an earlier failure by refusing to call Python again:
                // re-entering the interpreter from inside an optimiser callback
                // that already holds the GIL would deadlock.
                if failure.lock().map(|g| g.is_some()).unwrap_or(true) {
                    return f64::MAX;
                }
                let args = (x.to_vec(),);
                match f.call1(py, args).and_then(|r| r.extract::<f64>(py)) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Ok(mut slot) = failure.lock() {
                            *slot = Some(e);
                        }
                        f64::MAX
                    }
                }
            })
        },
        &x0,
        &config,
    );
    if let Ok(slot) = failure.into_inner() {
        if let Some(e) = slot {
            return Err(e);
        }
    }
    Ok((result.x, result.fun))
}

/// Failure from [`cg_solve_checked`], split by whose fault it is.
#[derive(Debug, PartialEq)]
enum CgSolveError {
    /// The caller described no solvable system: a non-square shape, a
    /// right-hand side that does not match the row count, or an index out of
    /// bounds.
    InvalidSystem(String),
    /// The system was well formed but the solver did not converge.
    Solver(String),
}

/// Validate a CG system, then solve it, without entering `cv-math`'s panicking
/// entry points on bad input.
///
/// `cg_solve` asserts that the matrix is square and that `b` has one entry per
/// row, and `CsrMatrix::from_triplets` asserts that every row and column index
/// is in bounds. A binding that forwards unvalidated input turns a caller
/// error into a Rust panic (`PanicException`), which is neither the
/// `ValueError` this function's neighbours raise nor something the caller can
/// catch and correct.
///
/// The old guard compared `b.len()` against `ncols`. For Ax = b the right-hand
/// side has one entry per **row**, so a 1x2 system with two RHS entries slipped
/// through and panicked inside `cg_solve`:
/// `sparse_solve_cg([(0,0,1.0)], 1, 2, [1.0, 2.0])` -> `assertion 'left ==
/// right' failed: Matrix must be square for CG`.
fn cg_solve_checked(
    nrows: usize,
    ncols: usize,
    triplets: &[(usize, usize, f64)],
    b: Vec<f64>,
) -> Result<Vec<f64>, CgSolveError> {
    if nrows != ncols {
        return Err(CgSolveError::InvalidSystem(format!(
            "the matrix must be square, got {nrows} rows and {ncols} columns"
        )));
    }
    if b.len() != nrows {
        return Err(CgSolveError::InvalidSystem(format!(
            "the right-hand side has {} entries but the matrix has {nrows} rows",
            b.len()
        )));
    }
    if let Some(&(r, c, _)) = triplets.iter().find(|&&(r, c, _)| r >= nrows || c >= ncols) {
        return Err(CgSolveError::InvalidSystem(format!(
            "triplet ({r}, {c}) is out of bounds for a {nrows}x{ncols} matrix"
        )));
    }
    let mat = cv_scientific::sparse::CsrMatrix::from_triplets(nrows, ncols, triplets);
    let b_vec = DVector::from_vec(b);
    let result =
        cv_scientific::sparse::cg_solve(&mat, &b_vec, 1000, 1e-10).map_err(CgSolveError::Solver)?;
    Ok(result.as_slice().to_vec())
}

/// Solve a sparse linear system Ax = b using conjugate gradient.
#[pyfunction]
fn sparse_solve_cg(
    triplets: Vec<(usize, usize, f64)>,
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
) -> PyResult<Vec<f64>> {
    cg_solve_checked(nrows, ncols, &triplets, b).map_err(|e| match e {
        CgSolveError::InvalidSystem(msg) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("sparse_solve_cg: {msg}"))
        }
        CgSolveError::Solver(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the RHS guard comparing against `ncols` instead of
    /// `nrows`.
    ///
    /// A 1x2 system with two RHS entries matched `ncols`, so the binding waved
    /// it through and `cg_solve` asserted "Matrix must be square for CG" - a
    /// `PanicException` from Python. These inputs must now come back as
    /// `InvalidSystem`, not as a panic (and not as a solver failure).
    #[test]
    fn sparse_solve_cg_rejects_malformed_systems() {
        assert!(
            matches!(
                cg_solve_checked(1, 2, &[(0, 0, 1.0)], vec![1.0, 2.0]),
                Err(CgSolveError::InvalidSystem(_))
            ),
            "a 1x2 system is not solvable by CG"
        );
        assert!(
            matches!(
                cg_solve_checked(2, 2, &[(0, 0, 1.0)], vec![1.0]),
                Err(CgSolveError::InvalidSystem(_))
            ),
            "one RHS entry cannot solve a two-row system"
        );
        assert!(
            matches!(
                cg_solve_checked(1, 1, &[(5, 0, 1.0)], vec![1.0]),
                Err(CgSolveError::InvalidSystem(_))
            ),
            "row 5 is out of bounds for a 1x1 matrix"
        );
        assert!(
            matches!(
                cg_solve_checked(2, 2, &[(0, 3, 1.0)], vec![1.0, 1.0]),
                Err(CgSolveError::InvalidSystem(_))
            ),
            "column 3 is out of bounds for a 2x2 matrix"
        );

        // Control: the well-formed system still solves exactly.
        let solution = cg_solve_checked(1, 1, &[(0, 0, 2.0)], vec![4.0]).expect("2x = 4");
        assert!((solution[0] - 2.0).abs() < 1e-10, "got {solution:?}");
    }
}
