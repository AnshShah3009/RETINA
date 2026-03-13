//! iSAM2: Incremental Smoothing and Mapping
//!
//! Pure Rust implementation of the iSAM2 algorithm for incremental nonlinear
//! optimization. Based on "iSAM2: Incremental Smoothing and Mapping Using the
//! Bayes Tree" by Kaess et al. (IJRR 2012).
//!
//! Provides both incremental ([`Isam2Solver::update`]) and batch
//! ([`Isam2Solver::optimize_batch`]) modes. The incremental mode only
//! re-linearizes factors touching variables whose estimates have changed beyond
//! a configurable threshold.
//!
//! All factor graph types come from [`crate::factor_graph`].

use crate::factor_graph::{numerical_jacobians, Factor, Key, NoiseModel, Values, Variable};
use nalgebra::{DMatrix, DVector, Isometry3, Point3};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::AddAssign;

// ---------------------------------------------------------------------------
// Configuration
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
// Solver
// ---------------------------------------------------------------------------

/// iSAM2 incremental nonlinear solver.
pub struct Isam2Solver {
    config: Isam2Config,
    theta: Values,
    factors: Vec<Box<dyn Factor>>,
    key_to_factors: HashMap<Key, Vec<usize>>,
    last_linearized: Values,
    update_count: usize,
    last_error: f64,
}

impl Isam2Solver {
    pub fn new(config: Isam2Config) -> Self {
        Self {
            config,
            theta: Values::new(),
            factors: Vec::new(),
            key_to_factors: HashMap::new(),
            last_linearized: Values::new(),
            update_count: 0,
            last_error: f64::INFINITY,
        }
    }

    /// Incremental update: add new factors and initial values, then optimize
    /// only the affected sub-problem.
    pub fn update(
        &mut self,
        new_factors: Vec<Box<dyn Factor>>,
        new_values: Values,
    ) -> Result<(), String> {
        let new_keys: Vec<Key> = new_values.keys().copied().collect();
        for &k in &new_keys {
            let v = new_values.get(&k).unwrap().clone();
            self.last_linearized.insert(k, v.clone());
            self.theta.insert(k, v);
        }

        let base_idx = self.factors.len();
        for (i, factor) in new_factors.into_iter().enumerate() {
            let fi = base_idx + i;
            for &k in factor.keys() {
                self.key_to_factors.entry(k).or_default().push(fi);
            }
            self.factors.push(factor);
        }

        self.update_count += 1;

        let mut affected: HashSet<Key> = new_keys.into_iter().collect();

        if self.config.enable_partial_relinearization
            && self
                .update_count
                .is_multiple_of(self.config.relinearize_skip)
        {
            affected.extend(self.needs_relinearization());
        }

        // Expand to neighbours
        let mut expanded: HashSet<Key> = affected.clone();
        for &k in &affected {
            if let Some(fi_list) = self.key_to_factors.get(&k) {
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

        let mut affected_factor_set: BTreeSet<usize> = BTreeSet::new();
        for &k in &affected {
            if let Some(fi_list) = self.key_to_factors.get(&k) {
                for &fi in fi_list {
                    affected_factor_set.insert(fi);
                }
            }
        }
        let affected_factors: Vec<usize> = affected_factor_set.into_iter().collect();

        let mut ordered_keys: Vec<Key> = affected.into_iter().collect();
        ordered_keys.sort_by_key(|k| k.0);

        let (key_to_col, total_dim) = Self::build_ordering(&self.theta, &ordered_keys);

        for _gn in 0..self.config.gauss_newton_params.max_iters {
            let (h, b) = self.build_linear_system(&affected_factors, &key_to_col, total_dim);

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
                        .ok_or("Cholesky decomposition failed")?
                        .solve(&b)
                }
            };

            let delta_norm = delta.norm();
            Self::apply_delta(&mut self.theta, &ordered_keys, &key_to_col, &delta);

            if delta_norm < self.config.gauss_newton_params.tolerance {
                break;
            }
        }

        for &k in &ordered_keys {
            self.last_linearized
                .insert(k, self.theta.get(&k).unwrap().clone());
        }
        self.last_error = self.compute_total_error();
        Ok(())
    }

    pub fn estimate(&self, key: &Key) -> Option<&Variable> {
        self.theta.get(key)
    }

    pub fn estimate_pose3(&self, key: &Key) -> Option<Isometry3<f64>> {
        self.theta.at_pose3(key).copied()
    }

    pub fn estimate_point3(&self, key: &Key) -> Option<Point3<f64>> {
        self.theta.at_point3(key).copied()
    }

    pub fn total_error(&self) -> f64 {
        self.last_error
    }

    pub fn num_variables(&self) -> usize {
        self.theta.len()
    }

    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Full batch optimization over all factors.
    pub fn optimize_batch(&mut self) -> Result<f64, String> {
        if self.factors.is_empty() || self.theta.is_empty() {
            return Ok(0.0);
        }

        let all_factors: Vec<usize> = (0..self.factors.len()).collect();
        let mut ordered_keys: Vec<Key> = self.theta.keys().copied().collect();
        ordered_keys.sort_by_key(|k| k.0);
        let (key_to_col, total_dim) = Self::build_ordering(&self.theta, &ordered_keys);

        for _gn in 0..self.config.gauss_newton_params.max_iters {
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
            Self::apply_delta(&mut self.theta, &ordered_keys, &key_to_col, &delta);

            if delta_norm < self.config.gauss_newton_params.tolerance {
                break;
            }
        }

        for &k in &ordered_keys {
            self.last_linearized
                .insert(k, self.theta.get(&k).unwrap().clone());
        }
        self.last_error = self.compute_total_error();
        Ok(self.last_error)
    }

    // -- helpers --

    fn build_ordering(theta: &Values, ordered_keys: &[Key]) -> (HashMap<Key, usize>, usize) {
        let mut map = HashMap::new();
        let mut offset = 0usize;
        for &k in ordered_keys {
            map.insert(k, offset);
            offset += theta.get(&k).unwrap().dim();
        }
        (map, offset)
    }

    fn apply_delta(
        theta: &mut Values,
        ordered_keys: &[Key],
        key_to_col: &HashMap<Key, usize>,
        delta: &DVector<f64>,
    ) {
        for &k in ordered_keys {
            let col = key_to_col[&k];
            let var = theta.get(&k).unwrap();
            let dim = var.dim();
            let dk = delta.rows(col, dim).into_owned();
            let updated = var.retract(&dk);
            theta.insert(k, updated);
        }
    }

    fn needs_relinearization(&self) -> HashSet<Key> {
        let mut result = HashSet::new();
        for k in self.last_linearized.keys() {
            if let (Some(last_val), Some(cur_val)) =
                (self.last_linearized.get(k), self.theta.get(k))
            {
                let delta = last_val.local(cur_val);
                if delta.norm() > self.config.relinearize_threshold {
                    result.insert(*k);
                }
            }
        }
        result
    }

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
            let sqrt_info = factor.noise_model().sqrt_information();
            let raw_jacs = factor
                .jacobians(&self.theta)
                .unwrap_or_else(|| numerical_jacobians(factor.as_ref(), &self.theta));
            let jacs: Vec<DMatrix<f64>> = raw_jacs.iter().map(|j| &sqrt_info * j).collect();
            let wr = &sqrt_info * &err;
            let keys = factor.keys();

            for (i, ki) in keys.iter().enumerate() {
                if let Some(&ci) = key_to_col.get(ki) {
                    let di = jacs[i].ncols();
                    let jt_wr = jacs[i].transpose() * &wr;
                    b.rows_mut(ci, di).add_assign(&(-&jt_wr));

                    for (j, kj) in keys.iter().enumerate() {
                        if let Some(&cj) = key_to_col.get(kj) {
                            let dj = jacs[j].ncols();
                            let hij = jacs[i].transpose() * &jacs[j];
                            h.view_mut((ci, cj), (di, dj)).add_assign(&hij);
                        }
                    }
                }
            }
        }
        (h, b)
    }

    fn compute_total_error(&self) -> f64 {
        self.factors
            .iter()
            .map(|f| {
                let we = f.whitened_error(&self.theta);
                we.dot(&we)
            })
            .sum()
    }
}

impl Default for Isam2Solver {
    fn default() -> Self {
        Self::new(Isam2Config::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::{BetweenFactor, PriorFactor};

    fn iso_noise(dim: usize, sigma: f64) -> NoiseModel {
        NoiseModel::Isotropic(sigma, dim)
    }

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

        let prior = PriorFactor::new(
            k0,
            Variable::Vector(DVector::from_element(1, 0.0)),
            iso_noise(1, 0.01),
        );
        let b01 = BetweenFactor::new(
            k0,
            k1,
            Variable::Vector(DVector::from_element(1, 1.0)),
            iso_noise(1, 0.1),
        );
        let b12 = BetweenFactor::new(
            k1,
            k2,
            Variable::Vector(DVector::from_element(1, 2.0)),
            iso_noise(1, 0.1),
        );

        let mut init = Values::new();
        init.insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
        init.insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
        init.insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));

        solver
            .update(vec![Box::new(prior), Box::new(b01), Box::new(b12)], init)
            .unwrap();

        let x0 = match solver.estimate(&k0).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!(),
        };
        assert!(x0.abs() < 0.05, "x0 = {} should be near 0.0", x0);

        let x1 = match solver.estimate(&k1).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!(),
        };
        assert!((x1 - 1.0).abs() < 0.2, "x1 = {} should be near 1.0", x1);

        let x2 = match solver.estimate(&k2).unwrap() {
            Variable::Vector(v) => v[0],
            _ => panic!(),
        };
        assert!((x2 - 3.0).abs() < 0.2, "x2 = {} should be near 3.0", x2);
    }

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

        // Step 1: prior on x0
        {
            let mut vals = Values::new();
            vals.insert(k0, Variable::Vector(DVector::from_element(1, 0.1)));
            solver
                .update(
                    vec![Box::new(PriorFactor::new(
                        k0,
                        Variable::Vector(DVector::from_element(1, 0.0)),
                        iso_noise(1, 0.01),
                    ))],
                    vals,
                )
                .unwrap();
            let x0 = match solver.estimate(&k0).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!(x0.abs() < 0.05, "x0 = {} should be near 0", x0);
        }

        // Step 2: between x0->x1
        {
            let mut vals = Values::new();
            vals.insert(k1, Variable::Vector(DVector::from_element(1, 0.5)));
            solver
                .update(
                    vec![Box::new(BetweenFactor::new(
                        k0,
                        k1,
                        Variable::Vector(DVector::from_element(1, 1.0)),
                        iso_noise(1, 0.1),
                    ))],
                    vals,
                )
                .unwrap();
            let x1 = match solver.estimate(&k1).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!((x1 - 1.0).abs() < 0.3, "x1 = {} should be near 1.0", x1);
        }

        // Step 3: between x1->x2
        {
            let mut vals = Values::new();
            vals.insert(k2, Variable::Vector(DVector::from_element(1, 2.0)));
            solver
                .update(
                    vec![Box::new(BetweenFactor::new(
                        k1,
                        k2,
                        Variable::Vector(DVector::from_element(1, 2.0)),
                        iso_noise(1, 0.1),
                    ))],
                    vals,
                )
                .unwrap();
            let x2 = match solver.estimate(&k2).unwrap() {
                Variable::Vector(v) => v[0],
                _ => panic!(),
            };
            assert!((x2 - 3.0).abs() < 0.5, "x2 = {} should be near 3.0", x2);
        }

        assert_eq!(solver.num_variables(), 3);
        assert_eq!(solver.num_factors(), 3);
    }

    #[test]
    fn test_batch_vs_incremental() {
        let sigma = 0.1;
        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);

        // Incremental
        let mut inc = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-10,
            },
            ..Default::default()
        });
        {
            let mut v = Values::new();
            v.insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
            inc.update(
                vec![Box::new(PriorFactor::new(
                    k0,
                    Variable::Vector(DVector::from_element(1, 0.0)),
                    iso_noise(1, 0.01),
                ))],
                v,
            )
            .unwrap();
        }
        {
            let mut v = Values::new();
            v.insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
            inc.update(
                vec![Box::new(BetweenFactor::new(
                    k0,
                    k1,
                    Variable::Vector(DVector::from_element(1, 1.0)),
                    iso_noise(1, sigma),
                ))],
                v,
            )
            .unwrap();
        }
        {
            let mut v = Values::new();
            v.insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));
            inc.update(
                vec![Box::new(BetweenFactor::new(
                    k1,
                    k2,
                    Variable::Vector(DVector::from_element(1, 2.0)),
                    iso_noise(1, sigma),
                ))],
                v,
            )
            .unwrap();
        }

        // Batch
        let mut batch = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-10,
            },
            ..Default::default()
        });
        batch
            .theta
            .insert(k0, Variable::Vector(DVector::from_element(1, 0.5)));
        batch
            .theta
            .insert(k1, Variable::Vector(DVector::from_element(1, 1.5)));
        batch
            .theta
            .insert(k2, Variable::Vector(DVector::from_element(1, 3.5)));
        batch.last_linearized = batch.theta.clone();

        let fv: Vec<Box<dyn Factor>> = vec![
            Box::new(PriorFactor::new(
                k0,
                Variable::Vector(DVector::from_element(1, 0.0)),
                iso_noise(1, 0.01),
            )),
            Box::new(BetweenFactor::new(
                k0,
                k1,
                Variable::Vector(DVector::from_element(1, 1.0)),
                iso_noise(1, sigma),
            )),
            Box::new(BetweenFactor::new(
                k1,
                k2,
                Variable::Vector(DVector::from_element(1, 2.0)),
                iso_noise(1, sigma),
            )),
        ];
        for (i, f) in fv.into_iter().enumerate() {
            for &k in f.keys() {
                batch.key_to_factors.entry(k).or_default().push(i);
            }
            batch.factors.push(f);
        }
        batch.optimize_batch().unwrap();

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
                "Key {:?}: inc={} batch={}",
                k,
                vi,
                vb
            );
        }
    }

    #[test]
    fn test_2d_pose_slam_loop() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 30,
                tolerance: 1e-8,
            },
            ..Default::default()
        });
        let k0 = Key(0);
        let k1 = Key(1);
        let k2 = Key(2);
        let k3 = Key(3);
        let sigma = 0.05;

        let mut vals = Values::new();
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
                    Box::new(PriorFactor::new(
                        k0,
                        Variable::Vector(DVector::from_vec(vec![0.0, 0.0, 0.0])),
                        iso_noise(3, 0.01),
                    )),
                    Box::new(BetweenFactor::new(
                        k0,
                        k1,
                        Variable::Vector(DVector::from_vec(vec![1.0, 0.0, 0.0])),
                        iso_noise(3, sigma),
                    )),
                    Box::new(BetweenFactor::new(
                        k1,
                        k2,
                        Variable::Vector(DVector::from_vec(vec![0.0, 1.0, 0.0])),
                        iso_noise(3, sigma),
                    )),
                    Box::new(BetweenFactor::new(
                        k2,
                        k3,
                        Variable::Vector(DVector::from_vec(vec![-1.0, 0.0, 0.0])),
                        iso_noise(3, sigma),
                    )),
                    Box::new(BetweenFactor::new(
                        k3,
                        k0,
                        Variable::Vector(DVector::from_vec(vec![0.0, -1.0, 0.0])),
                        iso_noise(3, sigma),
                    )),
                ],
                vals,
            )
            .unwrap();

        let expected: [(Key, [f64; 3]); 4] = [
            (k0, [0.0, 0.0, 0.0]),
            (k1, [1.0, 0.0, 0.0]),
            (k2, [1.0, 1.0, 0.0]),
            (k3, [0.0, 1.0, 0.0]),
        ];
        for (k, gt) in &expected {
            let est = match solver.estimate(k).unwrap() {
                Variable::Vector(v) => v.clone(),
                _ => panic!(),
            };
            for i in 0..3 {
                assert!(
                    (est[i] - gt[i]).abs() < 0.15,
                    "Key {:?}[{}]: est={} gt={}",
                    k,
                    i,
                    est[i],
                    gt[i]
                );
            }
        }
        assert!(solver.total_error() < 1.0, "error={}", solver.total_error());
    }

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

        let mut vals = Values::new();
        vals.insert(k0, Variable::Pose3(Isometry3::translation(0.1, 0.0, 0.0)));
        vals.insert(k1, Variable::Pose3(Isometry3::translation(0.8, 0.1, 0.0)));

        solver
            .update(
                vec![
                    Box::new(PriorFactor::new(
                        k0,
                        Variable::Pose3(gt0),
                        iso_noise(6, 0.01),
                    )),
                    Box::new(BetweenFactor::new(
                        k0,
                        k1,
                        Variable::Pose3(measured),
                        iso_noise(6, 0.05),
                    )),
                ],
                vals,
            )
            .unwrap();

        let est0 = solver.estimate_pose3(&k0).unwrap();
        assert!(
            est0.translation.vector.norm() < 0.1,
            "Pose0 near origin: {:?}",
            est0.translation.vector
        );
        let est1 = solver.estimate_pose3(&k1).unwrap();
        assert!(
            (est1.translation.vector.x - 1.0).abs() < 0.15,
            "Pose1.x near 1.0: {}",
            est1.translation.vector.x
        );
    }

    #[test]
    fn test_point3_estimation() {
        let mut solver = Isam2Solver::new(Isam2Config {
            gauss_newton_params: GNParams {
                max_iters: 20,
                tolerance: 1e-8,
            },
            ..Default::default()
        });
        let k0 = Key(1_000_000);
        let k1 = Key(1_000_001);

        let mut vals = Values::new();
        vals.insert(k0, Variable::Point3(Point3::new(1.2, 1.8, 3.1)));
        vals.insert(k1, Variable::Point3(Point3::new(2.0, 1.5, 3.0)));

        solver
            .update(
                vec![
                    Box::new(PriorFactor::new(
                        k0,
                        Variable::Point3(Point3::new(1.0, 2.0, 3.0)),
                        iso_noise(3, 0.01),
                    )),
                    Box::new(BetweenFactor::new(
                        k0,
                        k1,
                        Variable::Point3(Point3::new(1.0, 0.0, 0.0)),
                        iso_noise(3, 0.05),
                    )),
                ],
                vals,
            )
            .unwrap();

        let p0 = solver.estimate_point3(&k0).unwrap();
        assert!(
            (p0.x - 1.0).abs() < 0.1 && (p0.y - 2.0).abs() < 0.1 && (p0.z - 3.0).abs() < 0.1,
            "p0={:?}",
            p0
        );
        let p1 = solver.estimate_point3(&k1).unwrap();
        assert!(
            (p1.x - 2.0).abs() < 0.15 && (p1.y - 2.0).abs() < 0.15 && (p1.z - 3.0).abs() < 0.15,
            "p1={:?}",
            p1
        );
    }

    #[test]
    fn test_default_config() {
        let solver = Isam2Solver::default();
        assert_eq!(solver.num_variables(), 0);
        assert_eq!(solver.num_factors(), 0);
        assert!(solver.total_error().is_infinite());
    }

    #[test]
    fn test_empty_update() {
        let mut solver = Isam2Solver::default();
        solver.update(vec![], Values::new()).unwrap();
        assert_eq!(solver.num_variables(), 0);
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible Isam2 wrapper (used by Python bindings)
// ---------------------------------------------------------------------------

use crate::factors::BetweenFactor;
use std::sync::RwLock;

/// Legacy Isam2 interface for backward compatibility with Python bindings.
pub struct Isam2 {
    solver: RwLock<Isam2Solver>,
    pending_factors: RwLock<Vec<Box<dyn Factor>>>,
    pending_values: RwLock<Values>,
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
            pending_values: RwLock::new(Values::new()),
            optimize_on_update,
        }
    }

    pub fn add_pose(&self, id: usize, initial: nalgebra::Vector3<f64>) {
        if let Ok(mut vals) = self.pending_values.write() {
            vals.insert(
                Key(id as u64),
                Variable::Vector(DVector::from_column_slice(&[
                    initial.x, initial.y, initial.z,
                ])),
            );
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
        let factors =
            std::mem::take(&mut *self.pending_factors.write().map_err(|e| e.to_string())?);
        let values = std::mem::replace(
            &mut *self.pending_values.write().map_err(|e| e.to_string())?,
            Values::new(),
        );
        let mut solver = self.solver.write().map_err(|e| e.to_string())?;
        solver.update(factors, values)
    }

    pub fn optimize(&self) -> Result<(), String> {
        self.solver
            .write()
            .map_err(|e| e.to_string())?
            .optimize_batch()?;
        Ok(())
    }

    pub fn get_pose(&self, id: usize) -> Option<nalgebra::Vector3<f64>> {
        let solver = self.solver.read().ok()?;
        match solver.estimate(&Key(id as u64))? {
            Variable::Vector(v) if v.len() >= 3 => Some(nalgebra::Vector3::new(v[0], v[1], v[2])),
            Variable::Pose3(iso) => Some(iso.translation.vector),
            _ => None,
        }
    }

    pub fn get_point(&self, id: usize) -> Option<Point3<f64>> {
        let solver = self.solver.read().ok()?;
        solver.estimate_point3(&Key(1_000_000 + id as u64))
    }

    pub fn get_all_poses(&self) -> Vec<(usize, nalgebra::Vector3<f64>)> {
        let solver = self.solver.read().unwrap();
        let mut result = Vec::new();
        for id in 0..1_000_000u64 {
            if let Some(var) = solver.estimate(&Key(id)) {
                match var {
                    Variable::Vector(v) if v.len() >= 3 => {
                        result.push((id as usize, nalgebra::Vector3::new(v[0], v[1], v[2])))
                    }
                    Variable::Pose3(iso) => result.push((id as usize, iso.translation.vector)),
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
