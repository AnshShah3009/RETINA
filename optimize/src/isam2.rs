//! iSAM2: Incremental Smoothing and Mapping
//!
//! Pure Rust implementation of the iSAM2 algorithm for incremental nonlinear optimization.
//! Based on "iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree"
//! by Kaess et al. (IJRR 2012).
//!
//! This module provides both incremental (`update`) and batch (`optimize_batch`) modes.
//! The incremental mode only re-linearizes factors touching variables whose estimates
//! have changed beyond a configurable threshold, giving O(affected) cost per update
//! rather than O(total).
//!
//! # Factor Graph Types
//!
//! The types `Key`, `Variable`, `NoiseModel`, and the `Factor` trait are defined
//! inline here. When `crate::factor_graph` becomes available, these can be replaced
//! with re-exports from that module.

use nalgebra::{DMatrix, DVector, Isometry3, Point3, UnitQuaternion, Vector3};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Factor graph primitives (inline until crate::factor_graph is ready)
// ---------------------------------------------------------------------------

/// Unique identifier for a variable in the factor graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key(pub u64);

impl Key {
    /// Create a pose key: `x(id)`.
    pub fn pose(id: u64) -> Self {
        Key(id)
    }
    /// Create a landmark key: `l(id)`, offset into a separate namespace.
    pub fn landmark(id: u64) -> Self {
        Key(1_000_000 + id)
    }
}

/// A variable value in the factor graph.
#[derive(Debug, Clone)]
pub enum Variable {
    /// A generic vector of dimension N.
    Vector(DVector<f64>),
    /// A 3D pose (translation + rotation).
    Pose3(Isometry3<f64>),
    /// A 3D point / landmark.
    Point3(Point3<f64>),
}

impl Variable {
    /// Dimension of the tangent space (degrees of freedom).
    pub fn dim(&self) -> usize {
        match self {
            Variable::Vector(v) => v.len(),
            Variable::Pose3(_) => 6,
            Variable::Point3(_) => 3,
        }
    }

    /// Retract (apply a tangent-space delta to produce an updated variable).
    pub fn retract(&self, delta: &DVector<f64>) -> Variable {
        match self {
            Variable::Vector(v) => Variable::Vector(v + delta),
            Variable::Pose3(iso) => {
                let t = Vector3::new(delta[0], delta[1], delta[2]);
                let r = Vector3::new(delta[3], delta[4], delta[5]);
                let d = Isometry3::new(t, r);
                Variable::Pose3(iso * d)
            }
            Variable::Point3(p) => {
                Variable::Point3(Point3::new(p.x + delta[0], p.y + delta[1], p.z + delta[2]))
            }
        }
    }

    /// Local coordinates: compute the tangent-space vector from `self` to `other`.
    pub fn local(&self, other: &Variable) -> DVector<f64> {
        match (self, other) {
            (Variable::Vector(a), Variable::Vector(b)) => b - a,
            (Variable::Pose3(a), Variable::Pose3(b)) => {
                let d = a.inverse() * b;
                let t = d.translation.vector;
                let r = d.rotation.scaled_axis();
                DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
            }
            (Variable::Point3(a), Variable::Point3(b)) => {
                DVector::from_vec(vec![b.x - a.x, b.y - a.y, b.z - a.z])
            }
            _ => panic!("local() called on mismatched Variable types"),
        }
    }
}

/// Noise model (diagonal or full covariance).
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Diagonal noise: vector of standard deviations.
    Diagonal(DVector<f64>),
    /// Full covariance matrix.
    Full(DMatrix<f64>),
}

impl NoiseModel {
    /// Return the information (inverse covariance) matrix.
    pub fn information(&self) -> DMatrix<f64> {
        match self {
            NoiseModel::Diagonal(sigmas) => {
                let n = sigmas.len();
                let mut info = DMatrix::zeros(n, n);
                for i in 0..n {
                    info[(i, i)] = 1.0 / (sigmas[i] * sigmas[i]);
                }
                info
            }
            NoiseModel::Full(cov) => {
                cov.clone()
                    .try_inverse()
                    .unwrap_or_else(|| DMatrix::identity(cov.nrows(), cov.ncols()))
            }
        }
    }

    /// Whitened error: Sigma^{-1/2} * e.
    pub fn whiten(&self, error: &DVector<f64>) -> DVector<f64> {
        match self {
            NoiseModel::Diagonal(sigmas) => {
                let mut w = error.clone();
                for i in 0..w.len() {
                    w[i] /= sigmas[i];
                }
                w
            }
            NoiseModel::Full(cov) => {
                // Use Cholesky: cov = L L^T, whitened = L^{-1} e
                if let Some(chol) = cov.clone().cholesky() {
                    chol.solve(error)
                } else {
                    error.clone()
                }
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            NoiseModel::Diagonal(s) => s.len(),
            NoiseModel::Full(m) => m.nrows(),
        }
    }
}

/// A factor in the factor graph.
pub trait Factor: Send + Sync {
    /// The keys (variables) this factor depends on.
    fn keys(&self) -> &[Key];

    /// Evaluate the unwhitened error given current variable estimates.
    fn error(&self, values: &HashMap<Key, Variable>) -> DVector<f64>;

    /// The noise model for this factor.
    fn noise_model(&self) -> &NoiseModel;

    /// Compute the whitened error (convenience).
    fn whitened_error(&self, values: &HashMap<Key, Variable>) -> DVector<f64> {
        self.noise_model().whiten(&self.error(values))
    }

    /// Compute the Jacobians by numerical differentiation.
    /// Returns a list of (key, Jacobian) pairs.
    fn jacobians(&self, values: &HashMap<Key, Variable>) -> Vec<(Key, DMatrix<f64>)> {
        let e0 = self.error(values);
        let m = e0.len();
        let eps = 1e-8;
        let mut result = Vec::new();
        for &k in self.keys() {
            let var = &values[&k];
            let n = var.dim();
            let mut jac = DMatrix::zeros(m, n);
            for j in 0..n {
                let mut delta = DVector::zeros(n);
                delta[j] = eps;
                let perturbed_var = var.retract(&delta);
                let mut perturbed_values = values.clone();
                perturbed_values.insert(k, perturbed_var);
                let ep = self.error(&perturbed_values);
                jac.set_column(j, &((ep - &e0) / eps));
            }
            result.push((k, jac));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Concrete factor types
// ---------------------------------------------------------------------------

/// Prior factor: constrains a single variable to a given value.
pub struct PriorFactor {
    key: Key,
    prior: Variable,
    noise: NoiseModel,
}

impl PriorFactor {
    pub fn new(key: Key, prior: Variable, noise: NoiseModel) -> Self {
        Self { key, prior, noise }
    }
}

impl Factor for PriorFactor {
    fn keys(&self) -> &[Key] {
        std::slice::from_ref(&self.key)
    }

    fn error(&self, values: &HashMap<Key, Variable>) -> DVector<f64> {
        let val = &values[&self.key];
        self.prior.local(val)
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

/// Between factor: constrains the relative transformation between two variables.
pub struct BetweenFactor {
    keys: [Key; 2],
    measurement: Variable,
    noise: NoiseModel,
}

impl BetweenFactor {
    pub fn new(key1: Key, key2: Key, measurement: Variable, noise: NoiseModel) -> Self {
        Self {
            keys: [key1, key2],
            measurement,
            noise,
        }
    }
}

impl Factor for BetweenFactor {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn error(&self, values: &HashMap<Key, Variable>) -> DVector<f64> {
        let v1 = &values[&self.keys[0]];
        let v2 = &values[&self.keys[1]];
        match (&self.measurement, v1, v2) {
            (Variable::Vector(m), Variable::Vector(a), Variable::Vector(b)) => {
                // error = (b - a) - m
                (b - a) - m
            }
            (Variable::Pose3(m), Variable::Pose3(a), Variable::Pose3(b)) => {
                // error = Log(m^{-1} * a^{-1} * b)
                let predicted = a.inverse() * b;
                let err_iso = m.inverse() * predicted;
                let t = err_iso.translation.vector;
                let r = err_iso.rotation.scaled_axis();
                DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
            }
            (Variable::Point3(m), Variable::Point3(a), Variable::Point3(b)) => {
                let dx = b.x - a.x - m.x;
                let dy = b.y - a.y - m.y;
                let dz = b.z - a.z - m.z;
                DVector::from_vec(vec![dx, dy, dz])
            }
            _ => panic!("BetweenFactor: mismatched variable types"),
        }
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

// ---------------------------------------------------------------------------
// iSAM2 configuration
// ---------------------------------------------------------------------------

/// Gauss-Newton parameters for the linear solve step.
pub struct GNParams {
    pub max_iters: usize,
    pub tolerance: f64,
}

impl Default for GNParams {
    fn default() -> Self {
        Self {
            max_iters: 10,
            tolerance: 1e-6,
        }
    }
}

/// iSAM2 configuration.
pub struct Isam2Config {
    /// Variables are re-linearized when their tangent-space delta exceeds this.
    pub relinearize_threshold: f64,
    /// Only check for relinearization every N updates.
    pub relinearize_skip: usize,
    /// Whether to enable partial (selective) relinearization.
    pub enable_partial_relinearization: bool,
    /// Gauss-Newton solver parameters.
    pub gauss_newton_params: GNParams,
}

impl Default for Isam2Config {
    fn default() -> Self {
        Self {
            relinearize_threshold: 0.1,
            relinearize_skip: 10,
            enable_partial_relinearization: true,
            gauss_newton_params: GNParams::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// iSAM2 solver
// ---------------------------------------------------------------------------

/// iSAM2 incremental nonlinear solver.
///
/// Maintains a factor graph and variable estimates, supporting both incremental
/// updates (only re-linearizing affected variables) and full batch optimization.
pub struct Isam2Solver {
    config: Isam2Config,

    /// Current variable estimates.
    theta: HashMap<Key, Variable>,

    /// All factors added so far.
    factors: Vec<Box<dyn Factor>>,

    /// Mapping from key to indices of factors that touch it.
    linear_factors: HashMap<Key, Vec<usize>>,

    /// Variable values at the time they were last linearized.
    last_linearized: HashMap<Key, Variable>,

    /// How many times `update()` has been called.
    update_count: usize,

    /// Cached total error from last optimization.
    last_error: f64,
}

impl Isam2Solver {
    /// Create a new solver with the given configuration.
    pub fn new(config: Isam2Config) -> Self {
        Self {
            config,
            theta: HashMap::new(),
            factors: Vec::new(),
            linear_factors: HashMap::new(),
            last_linearized: HashMap::new(),
            update_count: 0,
            last_error: f64::INFINITY,
        }
    }

    /// Incremental update: add new factors and initial values, then optimize
    /// only the affected sub-problem.
    pub fn update(
        &mut self,
        new_factors: Vec<Box<dyn Factor>>,
        new_values: HashMap<Key, Variable>,
    ) -> Result<(), String> {
        // 1. Insert new variable estimates
        for (k, v) in &new_values {
            self.theta.insert(*k, v.clone());
            self.last_linearized.insert(*k, v.clone());
        }

        // 2. Register new factors and build adjacency
        let base_idx = self.factors.len();
        for (i, factor) in new_factors.into_iter().enumerate() {
            let fi = base_idx + i;
            for &k in factor.keys() {
                self.linear_factors.entry(k).or_default().push(fi);
            }
            self.factors.push(factor);
        }

        self.update_count += 1;

        // 3. Determine affected variables
        let mut affected: HashSet<Key> = new_values.keys().copied().collect();

        // Check relinearization periodically
        if self.config.enable_partial_relinearization
            && self.update_count % self.config.relinearize_skip == 0
        {
            let relin = self.needs_relinearization();
            affected.extend(relin);
        }

        // Also include neighbours of affected variables (factors touching them
        // may also touch other variables that need updating).
        let mut expanded: HashSet<Key> = affected.clone();
        for &k in &affected {
            if let Some(fi_list) = self.linear_factors.get(&k) {
                for &fi in fi_list {
                    for &fk in self.factors[fi].keys() {
                        expanded.insert(fk);
                    }
                }
            }
        }
        let affected = expanded;

        if affected.is_empty() || self.factors.is_empty() {
            return Ok(());
        }

        // 4. Collect affected factor indices
        let mut affected_factor_set: BTreeSet<usize> = BTreeSet::new();
        for &k in &affected {
            if let Some(fi_list) = self.linear_factors.get(&k) {
                for &fi in fi_list {
                    affected_factor_set.insert(fi);
                }
            }
        }
        let affected_factors: Vec<usize> = affected_factor_set.into_iter().collect();

        // 5. Build ordering for affected variables
        let mut ordered_keys: Vec<Key> = affected.into_iter().collect();
        ordered_keys.sort();
        let key_to_col: HashMap<Key, usize> = ordered_keys
            .iter()
            .scan(0usize, |offset, k| {
                let col = *offset;
                *offset += self.theta[k].dim();
                Some((*k, col))
            })
            .collect();
        let total_dim: usize = ordered_keys.iter().map(|k| self.theta[k].dim()).sum();

        // 6. Gauss-Newton iterations on the affected sub-problem
        for _gn_iter in 0..self.config.gauss_newton_params.max_iters {
            let (h, b) = self.build_linear_system(&affected_factors, &key_to_col, total_dim);

            // Regularize
            let mut h_reg = h;
            for i in 0..total_dim {
                h_reg[(i, i)] += 1e-6;
            }

            let delta = match h_reg.clone().cholesky() {
                Some(chol) => chol.solve(&b),
                None => {
                    // Fallback: add stronger damping
                    for i in 0..total_dim {
                        h_reg[(i, i)] += 1e-3;
                    }
                    h_reg
                        .cholesky()
                        .ok_or("Cholesky decomposition failed")?
                        .solve(&b)
                }
            };

            // Check convergence
            let delta_norm = delta.norm();

            // Apply updates
            for &k in &ordered_keys {
                let col = key_to_col[&k];
                let dim = self.theta[&k].dim();
                let dk = delta.rows(col, dim).into_owned();
                let updated = self.theta[&k].retract(&dk);
                self.theta.insert(k, updated);
            }

            if delta_norm < self.config.gauss_newton_params.tolerance {
                break;
            }
        }

        // 7. Update linearization points
        for &k in &ordered_keys {
            self.last_linearized.insert(k, self.theta[&k].clone());
        }

        // 8. Cache error
        self.last_error = self.compute_total_error();

        Ok(())
    }

    /// Get current estimate for a variable.
    pub fn estimate(&self, key: &Key) -> Option<&Variable> {
        self.theta.get(key)
    }

    /// Get current estimate as an Isometry3.
    pub fn estimate_pose3(&self, key: &Key) -> Option<Isometry3<f64>> {
        match self.theta.get(key)? {
            Variable::Pose3(iso) => Some(*iso),
            _ => None,
        }
    }

    /// Get current estimate as a Point3.
    pub fn estimate_point3(&self, key: &Key) -> Option<Point3<f64>> {
        match self.theta.get(key)? {
            Variable::Point3(p) => Some(*p),
            _ => None,
        }
    }

    /// Get the total error of all factors at the current estimates.
    pub fn total_error(&self) -> f64 {
        self.last_error
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.theta.len()
    }

    /// Number of factors.
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Full batch optimization: re-linearize all factors from scratch.
    pub fn optimize_batch(&mut self) -> Result<f64, String> {
        if self.factors.is_empty() || self.theta.is_empty() {
            return Ok(0.0);
        }

        let all_factors: Vec<usize> = (0..self.factors.len()).collect();

        // Build ordering over all variables
        let mut ordered_keys: Vec<Key> = self.theta.keys().copied().collect();
        ordered_keys.sort();
        let key_to_col: HashMap<Key, usize> = ordered_keys
            .iter()
            .scan(0usize, |offset, k| {
                let col = *offset;
                *offset += self.theta[k].dim();
                Some((*k, col))
            })
            .collect();
        let total_dim: usize = ordered_keys.iter().map(|k| self.theta[k].dim()).sum();

        for _gn_iter in 0..self.config.gauss_newton_params.max_iters {
            let (h, b) = self.build_linear_system(&all_factors, &key_to_col, total_dim);

            let mut h_reg = h;
            for i in 0..total_dim {
                h_reg[(i, i)] += 1e-6;
            }

            let delta = match h_reg.clone().cholesky() {
                Some(chol) => chol.solve(&b),
                None => {
                    for i in 0..total_dim {
                        h_reg[(i, i)] += 1e-3;
                    }
                    h_reg
                        .cholesky()
                        .ok_or("Cholesky decomposition failed in batch mode")?
                        .solve(&b)
                }
            };

            let delta_norm = delta.norm();

            for &k in &ordered_keys {
                let col = key_to_col[&k];
                let dim = self.theta[&k].dim();
                let dk = delta.rows(col, dim).into_owned();
                let updated = self.theta[&k].retract(&dk);
                self.theta.insert(k, updated);
            }

            if delta_norm < self.config.gauss_newton_params.tolerance {
                break;
            }
        }

        // Update all linearization points
        for &k in &ordered_keys {
            self.last_linearized.insert(k, self.theta[&k].clone());
        }

        self.last_error = self.compute_total_error();
        Ok(self.last_error)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Identify variables whose estimates have drifted beyond the relinearization threshold.
    fn needs_relinearization(&self) -> HashSet<Key> {
        let mut result = HashSet::new();
        for (k, last_val) in &self.last_linearized {
            if let Some(cur_val) = self.theta.get(k) {
                let delta = last_val.local(cur_val);
                if delta.norm() > self.config.relinearize_threshold {
                    result.insert(*k);
                }
            }
        }
        result
    }

    /// Build the Gauss-Newton Hessian (H) and gradient (b) for the given factors.
    fn build_linear_system(
        &self,
        factor_indices: &[usize],
        key_to_col: &HashMap<Key, usize>,
        total_dim: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let mut h = DMatrix::zeros(total_dim, total_dim);
        let mut b = DVector::zeros(total_dim);

        for &fi in factor_indices {
            let factor = &self.factors[fi];
            let err = factor.error(&self.theta);
            let info = factor.noise_model().information();
            let jacs = factor.jacobians(&self.theta);

            // For each pair of Jacobians (J_i, J_j):
            //   H[i,j] += J_i^T * info * J_j
            //   b[i]   -= J_i^T * info * err
            for &(ki, ref ji) in &jacs {
                if let Some(&ci) = key_to_col.get(&ki) {
                    let di = ji.ncols();
                    let jt_info = ji.transpose() * &info;

                    // b[i] -= J_i^T * info * err
                    let bi = &jt_info * &err;
                    b.rows_mut(ci, di).sub_assign(&bi);

                    for &(kj, ref jj) in &jacs {
                        if let Some(&cj) = key_to_col.get(&kj) {
                            let dj = jj.ncols();
                            let hij = &jt_info * jj;
                            h.view_mut((ci, cj), (di, dj)).add_assign(&hij);
                        }
                    }
                }
            }
        }

        (h, b)
    }

    /// Compute total squared Mahalanobis error over all factors.
    fn compute_total_error(&self) -> f64 {
        let mut total = 0.0;
        for factor in &self.factors {
            let we = factor.whitened_error(&self.theta);
            total += we.dot(&we);
        }
        total
    }
}

impl Default for Isam2Solver {
    fn default() -> Self {
        Self::new(Isam2Config::default())
    }
}

use std::ops::{AddAssign, SubAssign};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a diagonal noise model with uniform sigma.
    fn diag_noise(dim: usize, sigma: f64) -> NoiseModel {
        NoiseModel::Diagonal(DVector::from_element(dim, sigma))
    }

    // -----------------------------------------------------------------------
    // Test 1: Simple 1D chain with prior + between factors
    // -----------------------------------------------------------------------
    #[test]
    fn test_1d_chain_converges() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 20,
                tolerance: 1e-8,
            },
            ..Default::default()
        });

        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);

        // Prior: x0 = 0
        let prior = PriorFactor::new(
            k0,
            Variable::Vector(DVector::from_element(1, 0.0)),
            diag_noise(1, 0.01),
        );

        // Between x0->x1: delta = 1.0
        let b01 = BetweenFactor::new(
            k0,
            k1,
            Variable::Vector(DVector::from_element(1, 1.0)),
            diag_noise(1, 0.1),
        );
        // Between x1->x2: delta = 2.0
        let b12 = BetweenFactor::new(
            k1,
            k2,
            Variable::Vector(DVector::from_element(1, 2.0)),
            diag_noise(1, 0.1),
        );

        let mut init = HashMap::new();
        init.insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
        init.insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
        init.insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));

        solver
            .update(vec![Box::new(prior), Box::new(b01), Box::new(b12)], init)
            .unwrap();

        // x0 should be near 0 (strong prior)
        let x0 = match solver.estimate(&k0).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!("wrong type"),
        };
        assert!(
            (x0 - 0.0).abs() < 0.05,
            "x0 = {} should be near 0.0",
            x0
        );

        // x1 should be near 1.0
        let x1 = match solver.estimate(&k1).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!("wrong type"),
        };
        assert!(
            (x1 - 1.0).abs() < 0.2,
            "x1 = {} should be near 1.0",
            x1
        );

        // x2 should be near 3.0
        let x2 = match solver.estimate(&k2).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!("wrong type"),
        };
        assert!(
            (x2 - 3.0).abs() < 0.2,
            "x2 = {} should be near 3.0",
            x2
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Incremental updates — add factors one at a time
    // -----------------------------------------------------------------------
    #[test]
    fn test_incremental_updates() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 20,
                tolerance: 1e-8,
            },
            ..Default::default()
        });

        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);

        // Step 1: add x0 with prior
        {
            let prior = PriorFactor::new(
                k0,
                Variable::Vector(DVector::from_element(1, 0.0)),
                diag_noise(1, 0.01),
            );
            let mut vals = HashMap::new();
            vals.insert(k0, Variable::Vector(DVector::from_element(1, 0.1)));
            solver.update(vec![Box::new(prior)], vals).unwrap();

            let x0 = match solver.estimate(&k0).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!(
                (x0 - 0.0).abs() < 0.05,
                "After prior, x0 = {} should be near 0",
                x0
            );
        }

        // Step 2: add x1 and between(x0, x1) = 1.0
        {
            let b01 = BetweenFactor::new(
                k0,
                k1,
                Variable::Vector(DVector::from_element(1, 1.0)),
                diag_noise(1, 0.1),
            );
            let mut vals = HashMap::new();
            vals.insert(k1, Variable::Vector(DVector::from_element(1, 0.5)));
            solver.update(vec![Box::new(b01)], vals).unwrap();

            let x1 = match solver.estimate(&k1).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!(
                (x1 - 1.0).abs() < 0.3,
                "After between, x1 = {} should be near 1.0",
                x1
            );
        }

        // Step 3: add x2 and between(x1, x2) = 2.0
        {
            let b12 = BetweenFactor::new(
                k1,
                k2,
                Variable::Vector(DVector::from_element(1, 2.0)),
                diag_noise(1, 0.1),
            );
            let mut vals = HashMap::new();
            vals.insert(k2, Variable::Vector(DVector::from_element(1, 2.0)));
            solver.update(vec![Box::new(b12)], vals).unwrap();

            let x2 = match solver.estimate(&k2).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!(
                (x2 - 3.0).abs() < 0.5,
                "After 2nd between, x2 = {} should be near 3.0",
                x2
            );
        }

        assert_eq!(solver.num_variables(), 3);
        assert_eq!(solver.num_factors(), 3);
    }

    // -----------------------------------------------------------------------
    // Test 3: Batch vs incremental produce similar results
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_vs_incremental() {
        let sigma = 0.1;

        // --- Incremental ---
        let mut inc = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-10,
            },
            ..Default::default()
        });

        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);

        // Add everything incrementally
        {
            let prior = PriorFactor::new(
                k0,
                Variable::Vector(DVector::from_element(1, 0.0)),
                diag_noise(1, 0.01),
            );
            let mut vals = HashMap::new();
            vals.insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
            inc.update(vec![Box::new(prior)], vals).unwrap();
        }
        {
            let b01 = BetweenFactor::new(
                k0,
                k1,
                Variable::Vector(DVector::from_element(1, 1.0)),
                diag_noise(1, sigma),
            );
            let mut vals = HashMap::new();
            vals.insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
            inc.update(vec![Box::new(b01)], vals).unwrap();
        }
        {
            let b12 = BetweenFactor::new(
                k1,
                k2,
                Variable::Vector(DVector::from_element(1, 2.0)),
                diag_noise(1, sigma),
            );
            let mut vals = HashMap::new();
            vals.insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));
            inc.update(vec![Box::new(b12)], vals).unwrap();
        }

        // --- Batch ---
        let mut batch = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-10,
            },
            ..Default::default()
        });

        // Add all factors and values at once, then batch optimize
        let prior = PriorFactor::new(
            k0,
            Variable::Vector(DVector::from_element(1, 0.0)),
            diag_noise(1, 0.01),
        );
        let b01 = BetweenFactor::new(
            k0,
            k1,
            Variable::Vector(DVector::from_element(1, 1.0)),
            diag_noise(1, sigma),
        );
        let b12 = BetweenFactor::new(
            k1,
            k2,
            Variable::Vector(DVector::from_element(1, 2.0)),
            diag_noise(1, sigma),
        );

        let mut vals = HashMap::new();
        vals.insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
        vals.insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
        vals.insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));

        // Insert values and factors without optimizing (just populate)
        for (k, v) in &vals {
            batch.theta.insert(*k, v.clone());
            batch.last_linearized.insert(*k, v.clone());
        }
        let factors: Vec<Box<dyn Factor>> =
            vec![Box::new(prior), Box::new(b01), Box::new(b12)];
        for (i, f) in factors.into_iter().enumerate() {
            for &k in f.keys() {
                batch.linear_factors.entry(k).or_default().push(i);
            }
            batch.factors.push(f);
        }

        batch.optimize_batch().unwrap();

        // Compare
        for k in [k0, k1, k2] {
            let vi = match inc.estimate(&k).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            let vb = match batch.estimate(&k).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!(
                (vi - vb).abs() < 0.15,
                "Key {:?}: incremental={} vs batch={} differ too much",
                k,
                vi,
                vb
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: 2D pose SLAM loop (using 3-element vectors for x, y, theta)
    // -----------------------------------------------------------------------
    #[test]
    fn test_2d_pose_slam_loop() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-8,
            },
            ..Default::default()
        });

        // 4 poses in a square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> back to (0,0)
        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);
        let k3 = Key(3);

        let sigma = 0.05;

        // Prior on x0 at origin
        let prior = PriorFactor::new(
            k0,
            Variable::Vector(DVector::from_vec(vec![0.0, 0.0, 0.0])),
            diag_noise(3, 0.01),
        );

        // Between factors (dx, dy, dtheta)
        let b01 = BetweenFactor::new(
            k0,
            k1,
            Variable::Vector(DVector::from_vec(vec![1.0, 0.0, 0.0])),
            diag_noise(3, sigma),
        );
        let b12 = BetweenFactor::new(
            k1,
            k2,
            Variable::Vector(DVector::from_vec(vec![0.0, 1.0, 0.0])),
            diag_noise(3, sigma),
        );
        let b23 = BetweenFactor::new(
            k2,
            k3,
            Variable::Vector(DVector::from_vec(vec![-1.0, 0.0, 0.0])),
            diag_noise(3, sigma),
        );
        // Loop closure: x3 -> x0 should be (0, -1, 0)
        let b30 = BetweenFactor::new(
            k3,
            k0,
            Variable::Vector(DVector::from_vec(vec![0.0, -1.0, 0.0])),
            diag_noise(3, sigma),
        );

        // Noisy initial estimates
        let mut vals = HashMap::new();
        vals.insert(
            k0,
            Variable::Vector(DVector::from_vec(vec![0.1, -0.1, 0.05])),
        );
        vals.insert(
            k1,
            Variable::Vector(DVector::from_vec(vec![1.1, 0.1, -0.05])),
        );
        vals.insert(
            k2,
            Variable::Vector(DVector::from_vec(vec![0.9, 1.1, 0.03])),
        );
        vals.insert(
            k3,
            Variable::Vector(DVector::from_vec(vec![-0.1, 0.9, -0.02])),
        );

        solver
            .update(
                vec![
                    Box::new(prior),
                    Box::new(b01),
                    Box::new(b12),
                    Box::new(b23),
                    Box::new(b30),
                ],
                vals,
            )
            .unwrap();

        // Verify estimates are close to ground truth
        let expected = [
            (k0, vec![0.0, 0.0, 0.0]),
            (k1, vec![1.0, 0.0, 0.0]),
            (k2, vec![1.0, 1.0, 0.0]),
            (k3, vec![0.0, 1.0, 0.0]),
        ];

        for (k, gt) in &expected {
            let est = match solver.estimate(k).unwrap() {
                Variable::Vector(v) => v.clone(),
                _ => panic!("wrong type"),
            };
            for i in 0..gt.len() {
                assert!(
                    (est[i] - gt[i]).abs() < 0.15,
                    "Key {:?} dim {}: est={} gt={}",
                    k,
                    i,
                    est[i],
                    gt[i]
                );
            }
        }

        // Error should be small
        assert!(
            solver.total_error() < 1.0,
            "Total error {} should be small",
            solver.total_error()
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: 3D Pose3 factors (Isometry3)
    // -----------------------------------------------------------------------
    #[test]
    fn test_pose3_between_factor() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-8,
            },
            ..Default::default()
        });

        let k0 = Key(0);
        let k1 = Key(1);

        let gt0 = Isometry3::identity();
        let gt1 = Isometry3::translation(1.0, 0.0, 0.0);
        let measured = gt0.inverse() * gt1;

        let prior = PriorFactor::new(
            k0,
            Variable::Pose3(gt0),
            diag_noise(6, 0.01),
        );

        let between = BetweenFactor::new(
            k0,
            k1,
            Variable::Pose3(measured),
            diag_noise(6, 0.05),
        );

        let mut vals = HashMap::new();
        vals.insert(k0, Variable::Pose3(Isometry3::translation(0.1, 0.0, 0.0)));
        vals.insert(k1, Variable::Pose3(Isometry3::translation(0.8, 0.1, 0.0)));

        solver
            .update(vec![Box::new(prior), Box::new(between)], vals)
            .unwrap();

        let est0 = solver.estimate_pose3(&k0).unwrap();
        let est1 = solver.estimate_pose3(&k1).unwrap();

        assert!(
            est0.translation.vector.norm() < 0.1,
            "Pose0 should be near origin, got {:?}",
            est0.translation.vector
        );
        assert!(
            (est1.translation.vector.x - 1.0).abs() < 0.15,
            "Pose1.x should be near 1.0, got {}",
            est1.translation.vector.x
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Point3 estimation
    // -----------------------------------------------------------------------
    #[test]
    fn test_point3_estimation() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 20,
                tolerance: 1e-8,
            },
            ..Default::default()
        });

        let k0 = Key::landmark(0);
        let k1 = Key::landmark(1);

        let prior = PriorFactor::new(
            k0,
            Variable::Point3(Point3::new(1.0, 2.0, 3.0)),
            diag_noise(3, 0.01),
        );

        let between = BetweenFactor::new(
            k0,
            k1,
            Variable::Point3(Point3::new(1.0, 0.0, 0.0)),
            diag_noise(3, 0.05),
        );

        let mut vals = HashMap::new();
        vals.insert(k0, Variable::Point3(Point3::new(1.2, 1.8, 3.1)));
        vals.insert(k1, Variable::Point3(Point3::new(2.0, 1.5, 3.0)));

        solver
            .update(vec![Box::new(prior), Box::new(between)], vals)
            .unwrap();

        let p0 = solver.estimate_point3(&k0).unwrap();
        assert!(
            (p0.x - 1.0).abs() < 0.1 && (p0.y - 2.0).abs() < 0.1 && (p0.z - 3.0).abs() < 0.1,
            "Point0 should be near (1,2,3), got {:?}",
            p0
        );

        let p1 = solver.estimate_point3(&k1).unwrap();
        assert!(
            (p1.x - 2.0).abs() < 0.15 && (p1.y - 2.0).abs() < 0.15 && (p1.z - 3.0).abs() < 0.15,
            "Point1 should be near (2,2,3), got {:?}",
            p1
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Default config
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_config() {
        let solver = Isam2Solver::default();
        assert_eq!(solver.num_variables(), 0);
        assert_eq!(solver.num_factors(), 0);
        assert!(solver.total_error().is_infinite());
    }

    // -----------------------------------------------------------------------
    // Test 8: Empty update does not crash
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_update() {
        let mut solver = Isam2Solver::default();
        solver.update(vec![], HashMap::new()).unwrap();
        assert_eq!(solver.num_variables(), 0);
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible Isam2 wrapper (used by Python bindings)
// ---------------------------------------------------------------------------

use std::sync::RwLock;

/// Legacy Isam2 interface for backward compatibility with Python bindings.
/// Wraps `Isam2Solver` with a simpler, thread-safe API.
pub struct Isam2 {
    solver: RwLock<Isam2Solver>,
    pending_factors: RwLock<Vec<Box<dyn Factor>>>,
    pending_values: RwLock<HashMap<Key, Variable>>,
    #[allow(dead_code)]
    optimize_on_update: bool,
}

impl Isam2 {
    pub fn new() -> Self {
        Self::with_config(true, false)
    }

    pub fn with_config(optimize_on_update: bool, _batch_optimize: bool) -> Self {
        Self {
            solver: RwLock::new(Isam2Solver::default()),
            pending_factors: RwLock::new(Vec::new()),
            pending_values: RwLock::new(HashMap::new()),
            optimize_on_update,
        }
    }

    pub fn add_pose(&self, id: usize, initial: Vector3<f64>) {
        if let Ok(mut vals) = self.pending_values.write() {
            vals.insert(Key(id as u64), Variable::Vector(DVector::from_column_slice(&[initial.x, initial.y, initial.z])));
        }
    }

    pub fn add_point(&self, id: usize, initial: Point3<f64>) {
        if let Ok(mut vals) = self.pending_values.write() {
            vals.insert(Key(1_000_000 + id as u64), Variable::Point3(initial));
        }
    }

    pub fn add_factor(&self, from: usize, to: usize, measurement: DVector<f64>, noise: f64) {
        let dim = measurement.len();
        let factor = BetweenFactor::new(
            Key(from as u64),
            Key(to as u64),
            Variable::Vector(measurement),
            NoiseModel::Diagonal(DVector::from_element(dim, noise)),
        );
        if let Ok(mut factors) = self.pending_factors.write() {
            factors.push(Box::new(factor));
        }
    }

    pub fn update(&self) -> Result<(), String> {
        let factors = std::mem::take(&mut *self.pending_factors.write().map_err(|e| e.to_string())?);
        let values = std::mem::take(&mut *self.pending_values.write().map_err(|e| e.to_string())?);
        let mut solver = self.solver.write().map_err(|e| e.to_string())?;
        solver.update(factors, values)
    }

    pub fn optimize(&self) -> Result<(), String> {
        let mut solver = self.solver.write().map_err(|e| e.to_string())?;
        solver.optimize_batch()?;
        Ok(())
    }

    pub fn get_pose(&self, id: usize) -> Option<Vector3<f64>> {
        let solver = self.solver.read().ok()?;
        match solver.estimate(&Key(id as u64))? {
            Variable::Vector(v) if v.len() >= 3 => Some(Vector3::new(v[0], v[1], v[2])),
            Variable::Pose3(iso) => Some(iso.translation.vector),
            _ => None,
        }
    }

    pub fn get_point(&self, id: usize) -> Option<Point3<f64>> {
        let solver = self.solver.read().ok()?;
        solver.estimate_point3(&Key(1_000_000 + id as u64))
    }

    pub fn get_all_poses(&self) -> Vec<(usize, Vector3<f64>)> {
        let solver = self.solver.read().unwrap();
        let mut result = Vec::new();
        for id in 0..1_000_000u64 {
            if let Some(var) = solver.estimate(&Key(id)) {
                match var {
                    Variable::Vector(v) if v.len() >= 3 => {
                        result.push((id as usize, Vector3::new(v[0], v[1], v[2])));
                    }
                    Variable::Pose3(iso) => {
                        result.push((id as usize, iso.translation.vector));
                    }
                    _ => {}
                }
            }
        }
        result
    }

    pub fn num_nodes(&self) -> usize {
        self.solver.read().map(|s| s.num_variables()).unwrap_or(0)
    }

    pub fn num_factors(&self) -> usize {
        self.solver.read().map(|s| s.num_factors()).unwrap_or(0)
    }
}

impl Default for Isam2 {
    fn default() -> Self {
        Self::new()
    }
}
