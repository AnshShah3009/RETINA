use nalgebra::{DMatrix, DVector, Isometry3, Matrix6, Vector3, Vector6};
use std::collections::HashMap;

pub struct PoseGraph {
    pub nodes: HashMap<usize, Isometry3<f64>>,
    pub edges: Vec<Edge>,
    pub fixed_nodes: std::collections::HashSet<usize>,
}

pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub measurement: Isometry3<f64>,
    pub information: Matrix6<f64>,
}

impl Default for PoseGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PoseGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            fixed_nodes: std::collections::HashSet::new(),
        }
    }

    pub fn add_node(&mut self, id: usize, pose: Isometry3<f64>) {
        self.nodes.insert(id, pose);
    }

    pub fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        measurement: Isometry3<f64>,
        information: Matrix6<f64>,
    ) {
        self.edges.push(Edge {
            from,
            to,
            measurement,
            information,
        });
    }

    pub fn set_fixed(&mut self, id: usize) {
        self.fixed_nodes.insert(id);
    }

    pub fn optimize(&mut self, iterations: usize) -> Result<f64, String> {
        let num_nodes = self.nodes.len();
        if num_nodes == 0 {
            return Ok(0.0);
        }

        let mut node_indices: HashMap<usize, usize> = HashMap::new();
        let mut idx = 0;
        let mut sorted_keys: Vec<usize> = self.nodes.keys().cloned().collect();
        sorted_keys.sort();

        for id in sorted_keys {
            if !self.fixed_nodes.contains(&id) {
                node_indices.insert(id, idx);
                idx += 1;
            }
        }
        let system_size = idx * 6;
        if system_size == 0 {
            return Ok(0.0);
        }

        let mut error_sum = 0.0;

        for _ in 0..iterations {
            let mut h_mat = DMatrix::zeros(system_size, system_size);
            let mut b_vec = DVector::zeros(system_size);
            error_sum = 0.0;

            for edge in &self.edges {
                let pose_i = self.nodes[&edge.from];
                let pose_j = self.nodes[&edge.to];
                let pose_ij = edge.measurement;
                let info = edge.information;

                // Error: e = log(Z^-1 * Xi^-1 * Xj)
                let prediction = pose_i.inverse() * pose_j;
                let error_se3 = pose_ij.inverse() * prediction;

                let error = Vector6::new(
                    error_se3.translation.vector.x,
                    error_se3.translation.vector.y,
                    error_se3.translation.vector.z,
                    error_se3.rotation.scaled_axis().x,
                    error_se3.rotation.scaled_axis().y,
                    error_se3.rotation.scaled_axis().z,
                );

                error_sum += error.dot(&(info * error));

                let idx_i = node_indices.get(&edge.from);
                let idx_j = node_indices.get(&edge.to);

                if idx_i.is_none() && idx_j.is_none() {
                    continue;
                }

                let jj: Matrix6<f64> = Matrix6::identity();
                let rel_pose = pose_j.inverse() * pose_i;
                let ji: Matrix6<f64> = -adjoint(&rel_pose);

                let h_ii: Matrix6<f64> = ji.transpose() * info * ji;
                let h_jj: Matrix6<f64> = jj.transpose() * info * jj;
                let h_ij: Matrix6<f64> = ji.transpose() * info * jj;

                let b_i: Vector6<f64> = ji.transpose() * info * error;
                let b_j: Vector6<f64> = jj.transpose() * info * error;

                if let Some(&i) = idx_i {
                    let r = i * 6;
                    h_mat.fixed_view_mut::<6, 6>(r, r).add_assign(h_ii);
                    b_vec.fixed_rows_mut::<6>(r).sub_assign(b_i);

                    if let Some(&j) = idx_j {
                        let c = j * 6;
                        h_mat.fixed_view_mut::<6, 6>(r, c).add_assign(h_ij);
                        h_mat
                            .fixed_view_mut::<6, 6>(c, r)
                            .add_assign(h_ij.transpose());
                        h_mat.fixed_view_mut::<6, 6>(c, c).add_assign(h_jj);
                        b_vec.fixed_rows_mut::<6>(c).sub_assign(b_j);
                    }
                } else if let Some(&j) = idx_j {
                    let c = j * 6;
                    h_mat.fixed_view_mut::<6, 6>(c, c).add_assign(h_jj);
                    b_vec.fixed_rows_mut::<6>(c).sub_assign(b_j);
                }
            }

            // Regularization
            for i in 0..system_size {
                h_mat[(i, i)] += 1e-6;
            }

            let dx = match h_mat.cholesky() {
                Some(chol) => chol.solve(&b_vec),
                None => return Err("Cholesky failed".into()),
            };

            for (&id, &idx) in &node_indices {
                let update = dx.fixed_rows::<6>(idx * 6);
                let translation = Vector3::new(update[0], update[1], update[2]);
                let rotation = Vector3::new(update[3], update[4], update[5]);

                // Exponential map update
                let delta = Isometry3::new(translation, rotation);
                if let Some(pose) = self.nodes.get_mut(&id) {
                    // X = X * exp(-delta)? No, we use left or right update consistently.
                    // If e = log(Z^-1 * Xi^-1 * Xj), then updates are:
                    // Xi = Xi * exp(dxi)
                    // Xj = Xj * exp(dxj)
                    *pose *= delta;
                }
            }

            if error_sum < 1e-9 {
                break;
            }
        }

        Ok(error_sum)
    }
}

fn adjoint(iso: &Isometry3<f64>) -> Matrix6<f64> {
    let r = iso.rotation.to_rotation_matrix().matrix().clone_owned();
    let t = iso.translation.vector;
    let t_skew = nalgebra::Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    let tr = t_skew * r;

    let mut ad = Matrix6::zeros();
    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&tr);
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    ad
}

use std::ops::{AddAssign, SubAssign};
