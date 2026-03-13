//! GTSAM-equivalent factor graph framework for nonlinear least-squares optimization.
//!
//! Provides a type-safe, extensible factor graph with Gauss-Newton and
//! Levenberg-Marquardt solvers. Variables live on manifolds (SE(3), R^n, etc.)
//! and are updated via the retract/local operations.

use nalgebra::{DMatrix, DVector, Isometry3, Point3, Vector3};
use std::collections::HashMap;

// ── Key ──────────────────────────────────────────────────────────────────────

/// Type-safe variable key (GTSAM-style symbol encoding).
///
/// The upper 8 bits store an ASCII character symbol, the lower 56 bits store
/// a numeric index.  For example `Key::symbol('x', 0)` produces a unique key
/// for the first pose variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key(pub u64);

impl Key {
    /// Create a key from a character symbol and index: `Key::symbol('x', 0)`.
    pub fn symbol(c: char, idx: u64) -> Self {
        Key((c as u64) << 56 | idx)
    }

    /// Extract the numeric index (lower 56 bits).
    pub fn index(&self) -> u64 {
        self.0 & 0x00FFFFFFFFFFFFFF
    }

    /// Extract the symbol character (upper 8 bits).
    pub fn symbol_char(&self) -> char {
        ((self.0 >> 56) & 0xFF) as u8 as char
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.symbol_char(), self.index())
    }
}

// ── Variable ─────────────────────────────────────────────────────────────────

/// Variable types that can live in a [`Values`] container.
#[derive(Debug, Clone)]
pub enum Variable {
    /// 2-D pose: (x, y, theta).
    Pose2(Vector3<f64>),
    /// 3-D rigid-body pose in SE(3).
    Pose3(Isometry3<f64>),
    /// 2-D Euclidean point.
    Point2([f64; 2]),
    /// 3-D Euclidean point.
    Point3(Point3<f64>),
    /// Scalar value.
    Scalar(f64),
    /// Arbitrary-dimension vector.
    Vector(DVector<f64>),
}

impl Variable {
    /// Tangent-space dimension of this variable.
    pub fn dim(&self) -> usize {
        match self {
            Variable::Pose2(_) => 3,
            Variable::Pose3(_) => 6,
            Variable::Point2(_) => 2,
            Variable::Point3(_) => 3,
            Variable::Scalar(_) => 1,
            Variable::Vector(v) => v.len(),
        }
    }

    /// Linearisation point as a flat vector (used internally by the solver).
    pub fn to_vector(&self) -> DVector<f64> {
        match self {
            Variable::Pose2(v) => DVector::from_column_slice(v.as_slice()),
            Variable::Pose3(iso) => {
                let t = iso.translation.vector;
                let r = iso.rotation.scaled_axis();
                DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
            }
            Variable::Point2(p) => DVector::from_vec(vec![p[0], p[1]]),
            Variable::Point3(p) => DVector::from_vec(vec![p.x, p.y, p.z]),
            Variable::Scalar(s) => DVector::from_vec(vec![*s]),
            Variable::Vector(v) => v.clone(),
        }
    }

    /// Manifold retraction: `self (+) delta`.
    ///
    /// For Euclidean variables this is plain addition.  For SE(3) the delta is
    /// applied as a right-side exponential map update.
    pub fn retract(&self, delta: &DVector<f64>) -> Variable {
        match self {
            Variable::Pose2(v) => Variable::Pose2(v + Vector3::new(delta[0], delta[1], delta[2])),
            Variable::Pose3(iso) => {
                let dt = Vector3::new(delta[0], delta[1], delta[2]);
                let dr = Vector3::new(delta[3], delta[4], delta[5]);
                let d_iso = Isometry3::new(dt, dr);
                Variable::Pose3(iso * d_iso)
            }
            Variable::Point2(p) => Variable::Point2([p[0] + delta[0], p[1] + delta[1]]),
            Variable::Point3(p) => {
                Variable::Point3(Point3::new(p.x + delta[0], p.y + delta[1], p.z + delta[2]))
            }
            Variable::Scalar(s) => Variable::Scalar(s + delta[0]),
            Variable::Vector(v) => Variable::Vector(v + delta),
        }
    }

    /// Local coordinates: tangent-space difference `other (-) self`.
    pub fn local(&self, other: &Variable) -> DVector<f64> {
        match (self, other) {
            (Variable::Pose3(a), Variable::Pose3(b)) => {
                let d = a.inverse() * b;
                let t = d.translation.vector;
                let r = d.rotation.scaled_axis();
                DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
            }
            _ => other.to_vector() - self.to_vector(),
        }
    }
}

// ── NoiseModel ───────────────────────────────────────────────────────────────

/// Gaussian noise model for a factor.
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Diagonal covariance with per-component standard deviations.
    Diagonal(DVector<f64>),
    /// Isotropic noise: single sigma repeated `dim` times.
    Isotropic(f64, usize),
    /// Unit covariance (identity information).
    Unit(usize),
    /// Full covariance matrix.
    Full(DMatrix<f64>),
}

impl NoiseModel {
    /// Residual dimension.
    pub fn dim(&self) -> usize {
        match self {
            NoiseModel::Diagonal(v) => v.len(),
            NoiseModel::Isotropic(_, d) => *d,
            NoiseModel::Unit(d) => *d,
            NoiseModel::Full(m) => m.nrows(),
        }
    }

    /// Square-root information matrix (whitening matrix).
    ///
    /// For a noise model with covariance Sigma the square-root information is
    /// Sigma^{-1/2} such that whitened_error = sqrt_info * error.
    pub fn sqrt_information(&self) -> DMatrix<f64> {
        match self {
            NoiseModel::Diagonal(sigmas) => {
                let n = sigmas.len();
                let mut m = DMatrix::zeros(n, n);
                for i in 0..n {
                    m[(i, i)] = 1.0 / sigmas[i];
                }
                m
            }
            NoiseModel::Isotropic(sigma, dim) => DMatrix::identity(*dim, *dim) * (1.0 / sigma),
            NoiseModel::Unit(dim) => DMatrix::identity(*dim, *dim),
            NoiseModel::Full(cov) => {
                // Sigma^{-1/2} via Cholesky of Sigma^{-1}
                let n = cov.nrows();
                match cov.clone().try_inverse() {
                    Some(info) => match info.cholesky() {
                        Some(chol) => chol.l().transpose(),
                        None => DMatrix::identity(n, n),
                    },
                    None => DMatrix::identity(n, n),
                }
            }
        }
    }

    /// Information matrix (Sigma^{-1}).
    pub fn information(&self) -> DMatrix<f64> {
        let sq = self.sqrt_information();
        sq.transpose() * &sq
    }
}

// ── Factor trait ─────────────────────────────────────────────────────────────

/// Core factor trait.  Every factor computes an unwhitened residual and
/// provides its noise model and the keys of the variables it connects.
pub trait Factor: Send + Sync {
    /// Variable keys this factor depends on.
    fn keys(&self) -> &[Key];

    /// Residual dimension (unwhitened).
    fn dim(&self) -> usize;

    /// Unwhitened residual vector.
    fn error(&self, values: &Values) -> DVector<f64>;

    /// Noise model for this factor.
    fn noise_model(&self) -> &NoiseModel;

    /// Whitened error: `sqrt_info * error`.
    fn whitened_error(&self, values: &Values) -> DVector<f64> {
        let sqrt_info = self.noise_model().sqrt_information();
        sqrt_info * self.error(values)
    }

    /// Jacobians w.r.t. each connected variable.
    ///
    /// The default implementation returns empty (the solver will use numerical
    /// differentiation).  Override to provide analytical Jacobians.
    fn jacobians(&self, _values: &Values) -> Option<Vec<DMatrix<f64>>> {
        None
    }
}

/// Numerical Jacobians via central differences.
pub fn numerical_jacobians(factor: &dyn Factor, values: &Values) -> Vec<DMatrix<f64>> {
    let eps = 1e-5;
    let keys = factor.keys().to_vec();
    let r0 = factor.error(values);
    let m = r0.len();

    let mut jacobians = Vec::with_capacity(keys.len());
    for key in &keys {
        let var = values.get(key).expect("variable not found for key");
        let n = var.dim();
        let mut jac = DMatrix::zeros(m, n);

        for j in 0..n {
            let mut delta_plus = DVector::zeros(n);
            let mut delta_minus = DVector::zeros(n);
            delta_plus[j] = eps;
            delta_minus[j] = -eps;

            let mut vals_plus = values.clone();
            vals_plus.insert(*key, var.retract(&delta_plus));
            let r_plus = factor.error(&vals_plus);

            let mut vals_minus = values.clone();
            vals_minus.insert(*key, var.retract(&delta_minus));
            let r_minus = factor.error(&vals_minus);

            let col = (r_plus - r_minus) / (2.0 * eps);
            jac.set_column(j, &col);
        }
        jacobians.push(jac);
    }
    jacobians
}

// ── Values ───────────────────────────────────────────────────────────────────

/// Container for variable estimates.
#[derive(Debug, Clone, Default)]
pub struct Values {
    pub values: HashMap<Key, Variable>,
}

impl Values {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: Key, value: Variable) {
        self.values.insert(key, value);
    }

    pub fn get(&self, key: &Key) -> Option<&Variable> {
        self.values.get(key)
    }

    pub fn contains(&self, key: &Key) -> bool {
        self.values.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        self.values.keys()
    }

    /// Convenience accessor for Pose3 variables.
    pub fn at_pose3(&self, key: &Key) -> Option<&Isometry3<f64>> {
        match self.values.get(key)? {
            Variable::Pose3(iso) => Some(iso),
            _ => None,
        }
    }

    /// Convenience accessor for Point3 variables.
    pub fn at_point3(&self, key: &Key) -> Option<&Point3<f64>> {
        match self.values.get(key)? {
            Variable::Point3(p) => Some(p),
            _ => None,
        }
    }

    /// Apply a stacked delta vector to all variables according to the given
    /// ordering, producing a new Values.
    pub fn retract(&self, delta: &DVector<f64>, ordering: &[Key]) -> Values {
        let mut result = self.clone();
        let mut offset = 0;
        for key in ordering {
            if let Some(var) = self.values.get(key) {
                let d = var.dim();
                let dv = delta.rows(offset, d).clone_owned();
                result.insert(*key, var.retract(&dv));
                offset += d;
            }
        }
        result
    }
}

// ── Solver configs ───────────────────────────────────────────────────────────

/// Configuration for Gauss-Newton optimization.
pub struct GNConfig {
    pub max_iters: usize,
    pub tolerance: f64,
}

impl Default for GNConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tolerance: 1e-6,
        }
    }
}

/// Configuration for Levenberg-Marquardt optimization.
pub struct LMParams {
    pub max_iters: usize,
    pub tolerance: f64,
    pub initial_lambda: f64,
    pub lambda_factor: f64,
}

impl Default for LMParams {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tolerance: 1e-6,
            initial_lambda: 1e-3,
            lambda_factor: 10.0,
        }
    }
}

// ── FactorGraph ──────────────────────────────────────────────────────────────

/// A nonlinear factor graph.
#[derive(Default)]
pub struct FactorGraph {
    factors: Vec<Box<dyn Factor>>,
}

impl FactorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a factor to the graph.
    pub fn add<F: Factor + 'static>(&mut self, factor: F) {
        self.factors.push(Box::new(factor));
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Total (squared) whitened error: sum of ||whitened_error||^2.
    pub fn total_error(&self, values: &Values) -> f64 {
        self.factors
            .iter()
            .map(|f| {
                let we = f.whitened_error(values);
                we.dot(&we)
            })
            .sum()
    }

    /// Build variable ordering from factor keys (deterministic, sorted).
    fn build_ordering(&self, values: &Values) -> Vec<Key> {
        let mut keys: Vec<Key> = values.values.keys().copied().collect();
        keys.sort_by_key(|k| k.0);
        keys
    }

    /// Linearise all factors and assemble the normal equations H*dx = -b.
    fn linearize(&self, values: &Values, ordering: &[Key]) -> (DMatrix<f64>, DVector<f64>) {
        // Build offset map
        let mut offsets: HashMap<Key, usize> = HashMap::new();
        let mut total_dim = 0;
        for key in ordering {
            offsets.insert(*key, total_dim);
            total_dim += values.get(key).unwrap().dim();
        }

        let mut h = DMatrix::zeros(total_dim, total_dim);
        let mut b = DVector::zeros(total_dim);

        for factor in &self.factors {
            let sqrt_info = factor.noise_model().sqrt_information();
            let raw_jacs = factor
                .jacobians(values)
                .unwrap_or_else(|| numerical_jacobians(factor.as_ref(), values));
            let r = factor.error(values);

            // Whiten Jacobians and residual
            let jacs: Vec<DMatrix<f64>> = raw_jacs.iter().map(|j| &sqrt_info * j).collect();
            let wr = &sqrt_info * &r;

            let keys = factor.keys();
            for (i, ki) in keys.iter().enumerate() {
                let oi = offsets[ki];
                let di = jacs[i].ncols();

                // b -= Ji^T * wr
                let jt_r = jacs[i].transpose() * &wr;
                b.rows_mut(oi, di).add_assign(&jt_r);

                for (j, kj) in keys.iter().enumerate() {
                    let oj = offsets[kj];
                    let dj = jacs[j].ncols();

                    // H += Ji^T * Jj
                    let jt_j = jacs[i].transpose() * &jacs[j];
                    h.view_mut((oi, oj), (di, dj)).add_assign(&jt_j);
                }
            }
        }

        (h, b)
    }

    /// Gauss-Newton optimization.
    pub fn optimize_gn(&self, initial: &Values, config: &GNConfig) -> Result<Values, String> {
        let ordering = self.build_ordering(initial);
        let mut values = initial.clone();

        for _iter in 0..config.max_iters {
            let (h, b) = self.linearize(&values, &ordering);

            // Solve H * dx = -b
            let neg_b = -&b;
            let dx = match h.clone().cholesky() {
                Some(chol) => chol.solve(&neg_b),
                None => {
                    // Fall back: add small regularization
                    let n = h.nrows();
                    let mut h_reg = h;
                    for i in 0..n {
                        h_reg[(i, i)] += 1e-6;
                    }
                    match h_reg.cholesky() {
                        Some(chol) => chol.solve(&neg_b),
                        None => return Err("Cholesky decomposition failed".into()),
                    }
                }
            };

            values = values.retract(&dx, &ordering);

            if dx.norm() < config.tolerance {
                break;
            }
        }

        Ok(values)
    }

    /// Levenberg-Marquardt optimization.
    pub fn optimize_lm(&self, initial: &Values, config: &LMParams) -> Result<Values, String> {
        let ordering = self.build_ordering(initial);
        let mut values = initial.clone();
        let mut lambda = config.initial_lambda;
        let mut current_error = self.total_error(&values);

        for _iter in 0..config.max_iters {
            let (h, b) = self.linearize(&values, &ordering);
            let n = h.nrows();

            // Damped normal equations: (H + lambda * diag(H)) * dx = -b
            let mut h_damped = h;
            for i in 0..n {
                h_damped[(i, i)] += lambda * h_damped[(i, i)].max(1e-10);
            }

            let neg_b = -&b;
            let dx = match h_damped.cholesky() {
                Some(chol) => chol.solve(&neg_b),
                None => {
                    lambda *= config.lambda_factor;
                    continue;
                }
            };

            let candidate = values.retract(&dx, &ordering);
            let candidate_error = self.total_error(&candidate);

            if candidate_error < current_error {
                values = candidate;
                current_error = candidate_error;
                lambda /= config.lambda_factor;
                if dx.norm() < config.tolerance {
                    break;
                }
            } else {
                lambda *= config.lambda_factor;
            }
        }

        Ok(values)
    }
}

use std::ops::AddAssign;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal prior factor for testing
    struct TestPriorPose3 {
        key: Key,
        prior: Isometry3<f64>,
        noise: NoiseModel,
    }

    impl Factor for TestPriorPose3 {
        fn keys(&self) -> &[Key] {
            std::slice::from_ref(&self.key)
        }
        fn dim(&self) -> usize {
            6
        }
        fn error(&self, values: &Values) -> DVector<f64> {
            let pose = values.at_pose3(&self.key).unwrap();
            let d = self.prior.inverse() * pose;
            let t = d.translation.vector;
            let r = d.rotation.scaled_axis();
            DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
        }
        fn noise_model(&self) -> &NoiseModel {
            &self.noise
        }
    }

    // A minimal between factor for testing
    struct TestBetweenPose3 {
        keys: [Key; 2],
        measured: Isometry3<f64>,
        noise: NoiseModel,
    }

    impl Factor for TestBetweenPose3 {
        fn keys(&self) -> &[Key] {
            &self.keys
        }
        fn dim(&self) -> usize {
            6
        }
        fn error(&self, values: &Values) -> DVector<f64> {
            let p1 = values.at_pose3(&self.keys[0]).unwrap();
            let p2 = values.at_pose3(&self.keys[1]).unwrap();
            let predicted = p1.inverse() * p2;
            let d = self.measured.inverse() * predicted;
            let t = d.translation.vector;
            let r = d.rotation.scaled_axis();
            DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
        }
        fn noise_model(&self) -> &NoiseModel {
            &self.noise
        }
    }

    #[test]
    fn test_key_symbol() {
        let k = Key::symbol('x', 42);
        assert_eq!(k.symbol_char(), 'x');
        assert_eq!(k.index(), 42);
    }

    #[test]
    fn test_variable_dim() {
        assert_eq!(Variable::Pose2(Vector3::zeros()).dim(), 3);
        assert_eq!(Variable::Pose3(Isometry3::identity()).dim(), 6);
        assert_eq!(Variable::Point2([0.0; 2]).dim(), 2);
        assert_eq!(Variable::Point3(Point3::origin()).dim(), 3);
        assert_eq!(Variable::Scalar(0.0).dim(), 1);
        assert_eq!(Variable::Vector(DVector::zeros(7)).dim(), 7);
    }

    #[test]
    fn test_variable_retract_pose3() {
        let iso = Isometry3::identity();
        let var = Variable::Pose3(iso);
        let delta = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let updated = var.retract(&delta);
        if let Variable::Pose3(new_iso) = updated {
            assert!((new_iso.translation.vector.x - 1.0).abs() < 1e-10);
        } else {
            panic!("expected Pose3");
        }
    }

    #[test]
    fn test_noise_model_diagonal() {
        let nm = NoiseModel::Diagonal(DVector::from_vec(vec![0.1, 0.2]));
        assert_eq!(nm.dim(), 2);
        let sq = nm.sqrt_information();
        assert!((sq[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((sq[(1, 1)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_noise_model_isotropic() {
        let nm = NoiseModel::Isotropic(0.5, 3);
        assert_eq!(nm.dim(), 3);
        let sq = nm.sqrt_information();
        for i in 0..3 {
            assert!((sq[(i, i)] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_values_retract() {
        let mut vals = Values::new();
        let k1 = Key::symbol('x', 0);
        let k2 = Key::symbol('x', 1);
        vals.insert(k1, Variable::Point3(Point3::origin()));
        vals.insert(k2, Variable::Point3(Point3::new(1.0, 0.0, 0.0)));

        let mut ordering = vec![k1, k2];
        ordering.sort_by_key(|k| k.0);

        let delta = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let new_vals = vals.retract(&delta, &ordering);

        let p1 = new_vals.at_point3(&k1).unwrap();
        let p2 = new_vals.at_point3(&k2).unwrap();
        assert!((p1.x - 0.1).abs() < 1e-10);
        assert!((p2.x - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_factor_graph_prior_only() {
        let mut graph = FactorGraph::new();
        let k = Key::symbol('x', 0);

        let prior = Isometry3::translation(1.0, 0.0, 0.0);
        graph.add(TestPriorPose3 {
            key: k,
            prior,
            noise: NoiseModel::Isotropic(0.1, 6),
        });

        let mut initial = Values::new();
        initial.insert(k, Variable::Pose3(Isometry3::translation(1.5, 0.1, -0.1)));

        let result = graph.optimize_gn(&initial, &GNConfig::default()).unwrap();
        let pose = result.at_pose3(&k).unwrap();
        assert!(
            (pose.translation.vector.x - 1.0).abs() < 0.01,
            "expected x~1.0, got {}",
            pose.translation.vector.x
        );
        assert!(
            (pose.translation.vector.y).abs() < 0.01,
            "expected y~0.0, got {}",
            pose.translation.vector.y
        );
    }

    #[test]
    fn test_factor_graph_pose_chain_gn() {
        // 3-pose chain with between factors
        let mut graph = FactorGraph::new();
        let x0 = Key::symbol('x', 0);
        let x1 = Key::symbol('x', 1);
        let x2 = Key::symbol('x', 2);

        // Strong prior on x0 at the origin
        graph.add(TestPriorPose3 {
            key: x0,
            prior: Isometry3::identity(),
            noise: NoiseModel::Isotropic(0.001, 6),
        });

        // Between x0->x1: 1m in x
        graph.add(TestBetweenPose3 {
            keys: [x0, x1],
            measured: Isometry3::translation(1.0, 0.0, 0.0),
            noise: NoiseModel::Isotropic(0.1, 6),
        });

        // Between x1->x2: 1m in x
        graph.add(TestBetweenPose3 {
            keys: [x1, x2],
            measured: Isometry3::translation(1.0, 0.0, 0.0),
            noise: NoiseModel::Isotropic(0.1, 6),
        });

        // Initial estimates with noise
        let mut initial = Values::new();
        initial.insert(x0, Variable::Pose3(Isometry3::translation(0.1, 0.05, 0.0)));
        initial.insert(x1, Variable::Pose3(Isometry3::translation(1.2, -0.1, 0.0)));
        initial.insert(x2, Variable::Pose3(Isometry3::translation(2.3, 0.1, 0.0)));

        let result = graph.optimize_gn(&initial, &GNConfig::default()).unwrap();

        let p0 = result.at_pose3(&x0).unwrap();
        let p1 = result.at_pose3(&x1).unwrap();
        let p2 = result.at_pose3(&x2).unwrap();

        assert!(p0.translation.vector.x.abs() < 0.05);
        assert!((p1.translation.vector.x - 1.0).abs() < 0.1);
        assert!((p2.translation.vector.x - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_factor_graph_lm() {
        let mut graph = FactorGraph::new();
        let k = Key::symbol('x', 0);

        graph.add(TestPriorPose3 {
            key: k,
            prior: Isometry3::translation(2.0, 3.0, 0.0),
            noise: NoiseModel::Isotropic(0.1, 6),
        });

        let mut initial = Values::new();
        initial.insert(k, Variable::Pose3(Isometry3::translation(2.5, 3.5, 0.5)));

        let result = graph.optimize_lm(&initial, &LMParams::default()).unwrap();
        let pose = result.at_pose3(&k).unwrap();
        assert!((pose.translation.vector.x - 2.0).abs() < 0.01);
        assert!((pose.translation.vector.y - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_total_error_decreases() {
        let mut graph = FactorGraph::new();
        let k = Key::symbol('x', 0);

        graph.add(TestPriorPose3 {
            key: k,
            prior: Isometry3::identity(),
            noise: NoiseModel::Isotropic(1.0, 6),
        });

        let mut initial = Values::new();
        initial.insert(k, Variable::Pose3(Isometry3::translation(5.0, 5.0, 5.0)));

        let err_before = graph.total_error(&initial);
        let result = graph.optimize_gn(&initial, &GNConfig::default()).unwrap();
        let err_after = graph.total_error(&result);

        assert!(
            err_after < err_before,
            "error should decrease: {} -> {}",
            err_before,
            err_after
        );
    }
}
