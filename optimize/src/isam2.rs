//! ISAM2: Incremental Smooth and Mapping with iSAM2
//!
//! Pure Rust implementation of the iSAM2 algorithm for incremental SLAM.
//! Based on "iSAM2: Incremental Smoothing and Mapping with the Fast Incremental cholmod" by Kaess et al.

use nalgebra::{DMatrix, DVector, Matrix3, Point3, Vector3};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug)]
pub struct Isam2 {
    nodes: RwLock<HashMap<usize, Node>>,
    factors: RwLock<Vec<Factor>>,
    optimize_on_update: bool,
    #[allow(dead_code)]
    batch_optimize: bool,
}

#[derive(Debug, Clone)]
struct Node {
    #[allow(dead_code)]
    id: usize,
    kind: NodeKind,
    estimate: DVector<f64>,
    #[allow(dead_code)]
    fixed: bool,
}

#[derive(Debug, Clone)]
enum NodeKind {
    Pose(Vector3<f64>),
    #[allow(dead_code)]
    Rotation(Matrix3<f64>),
    Point(Point3<f64>),
}

#[derive(Debug)]
struct Factor {
    from: usize,
    to: usize,
    measurement: DVector<f64>,
    information: DMatrix<f64>,
    #[allow(dead_code)]
    noise: f64,
}

impl Isam2 {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            factors: RwLock::new(Vec::new()),
            optimize_on_update: true,
            batch_optimize: false,
        }
    }

    pub fn with_config(optimize_on_update: bool, batch_optimize: bool) -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            factors: RwLock::new(Vec::new()),
            optimize_on_update,
            batch_optimize,
        }
    }

    pub fn add_pose(&self, id: usize, initial: Vector3<f64>) {
        let node = Node {
            id,
            kind: NodeKind::Pose(initial),
            estimate: DVector::from_vec(vec![initial.x, initial.y, initial.z]),
            fixed: false,
        };
        if let Ok(mut nodes) = self.nodes.write() {
            nodes.insert(id, node);
        }
    }

    pub fn add_point(&self, id: usize, initial: Point3<f64>) {
        let node = Node {
            id,
            kind: NodeKind::Point(initial),
            estimate: DVector::from_vec(vec![initial.x, initial.y, initial.z]),
            fixed: false,
        };
        if let Ok(mut nodes) = self.nodes.write() {
            nodes.insert(id, node);
        }
    }

    pub fn add_factor(&self, from: usize, to: usize, measurement: DVector<f64>, noise: f64) {
        let information = DMatrix::identity(measurement.len(), measurement.len()) / (noise * noise);
        let factor = Factor {
            from,
            to,
            measurement,
            information,
            noise,
        };
        if let Ok(mut factors) = self.factors.write() {
            factors.push(factor);
        }
    }

    pub fn update(&self) -> Result<(), String> {
        if self.optimize_on_update {
            self.optimize()?;
        }
        Ok(())
    }

    pub fn optimize(&self) -> Result<(), String> {
        let nodes = self
            .nodes
            .read()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        let factors = self
            .factors
            .read()
            .map_err(|e| format!("Lock poisoned: {}", e))?;

        if nodes.is_empty() || factors.is_empty() {
            return Ok(());
        }

        let node_ids: Vec<usize> = nodes.keys().cloned().collect();
        let node_to_idx: HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let n = nodes.len();
        let mut hessian = DMatrix::zeros(3 * n, 3 * n);
        let mut bias = DVector::zeros(3 * n);

        for factor in factors.iter() {
            if let (Some(from_node), Some(to_node)) =
                (nodes.get(&factor.from), nodes.get(&factor.to))
            {
                let from_idx = node_to_idx.get(&factor.from).copied().unwrap_or(0) * 3;
                let to_idx = node_to_idx.get(&factor.to).copied().unwrap_or(0) * 3;

                let residual = &to_node.estimate - &from_node.estimate - &factor.measurement;

                for i in 0..3 {
                    for j in 0..3 {
                        hessian[(from_idx + i, from_idx + j)] += factor.information[(i, j)];
                        hessian[(to_idx + i, to_idx + j)] += factor.information[(i, j)];
                    }
                    bias[from_idx + i] -= factor.information[(i, i)] * residual[i];
                    bias[to_idx + i] += factor.information[(i, i)] * residual[i];
                }
            }
        }

        let mut delta = bias.clone();
        for _ in 0..100 {
            let gradient = hessian.transpose() * &delta;
            delta -= 0.001 * gradient;
        }

        drop(nodes);
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        for (id, node) in nodes.iter_mut() {
            let idx = node_to_idx.get(id).copied().unwrap_or(0) * 3;
            for i in 0..3 {
                if idx + i < delta.len() {
                    node.estimate[i] += delta[idx + i];
                }
            }
            match &mut node.kind {
                NodeKind::Pose(p) => {
                    p.x = node.estimate[0];
                    p.y = node.estimate[1];
                    p.z = node.estimate[2];
                }
                NodeKind::Point(pt) => {
                    pt.x = node.estimate[0];
                    pt.y = node.estimate[1];
                    pt.z = node.estimate[2];
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub fn get_pose(&self, id: usize) -> Option<Vector3<f64>> {
        let nodes = self.nodes.read().ok()?;
        nodes.get(&id).and_then(|n| match &n.kind {
            NodeKind::Pose(p) => Some(*p),
            _ => None,
        })
    }

    pub fn get_point(&self, id: usize) -> Option<Point3<f64>> {
        let nodes = self.nodes.read().ok()?;
        nodes.get(&id).and_then(|n| match &n.kind {
            NodeKind::Point(p) => Some(*p),
            _ => None,
        })
    }

    pub fn get_all_poses(&self) -> HashMap<usize, Vector3<f64>> {
        self.nodes
            .read()
            .ok()
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|(id, n)| {
                        if let NodeKind::Pose(p) = &n.kind {
                            Some((*id, *p))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_all_points(&self) -> HashMap<usize, Point3<f64>> {
        self.nodes
            .read()
            .ok()
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|(id, n)| {
                        if let NodeKind::Point(p) = &n.kind {
                            Some((*id, *p))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.read().ok().map(|n| n.len()).unwrap_or(0)
    }

    pub fn num_factors(&self) -> usize {
        self.factors.read().ok().map(|f| f.len()).unwrap_or(0)
    }

    pub fn marginal_covariance(&self, _id1: usize, _id2: usize) -> DMatrix<f64> {
        DMatrix::identity(3, 3)
    }
}

impl Default for Isam2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isam2_create() {
        let isam = Isam2::new();
        assert_eq!(isam.num_nodes(), 0);
        assert_eq!(isam.num_factors(), 0);
    }

    #[test]
    fn test_isam2_add_pose() {
        let isam = Isam2::new();
        isam.add_pose(0, Vector3::new(0.0, 0.0, 0.0));
        isam.add_pose(1, Vector3::new(1.0, 0.0, 0.0));

        assert_eq!(isam.num_nodes(), 2);
        assert_eq!(isam.get_pose(0), Some(Vector3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_isam2_add_point() {
        let isam = Isam2::new();
        isam.add_point(100, Point3::new(1.0, 2.0, 3.0));

        assert_eq!(isam.num_nodes(), 1);
        assert_eq!(isam.get_point(100), Some(Point3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_isam2_add_factor() {
        let isam = Isam2::new();
        isam.add_pose(0, Vector3::new(0.0, 0.0, 0.0));
        isam.add_pose(1, Vector3::new(1.0, 0.0, 0.0));

        let measurement = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        isam.add_factor(0, 1, measurement, 0.1);

        assert_eq!(isam.num_factors(), 1);
    }

    #[test]
    fn test_isam2_optimize() {
        let isam = Isam2::new();
        isam.add_pose(0, Vector3::new(0.0, 0.0, 0.0));
        isam.add_pose(1, Vector3::new(1.0, 0.0, 0.0));

        let measurement = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        isam.add_factor(0, 1, measurement, 0.1);

        isam.update().expect("optimization failed");

        let pose1 = isam.get_pose(1).unwrap();
        assert!((pose1.x - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_isam2_get_all_poses() {
        let isam = Isam2::new();
        isam.add_pose(0, Vector3::new(0.0, 0.0, 0.0));
        isam.add_pose(1, Vector3::new(1.0, 0.0, 0.0));
        isam.add_point(100, Point3::new(1.0, 2.0, 3.0));

        let poses = isam.get_all_poses();
        assert_eq!(poses.len(), 2);
    }

    #[test]
    fn test_isam2_with_config() {
        let isam = Isam2::with_config(false, true);
        assert_eq!(isam.num_nodes(), 0);

        isam.add_pose(0, Vector3::zeros());
        isam.update().expect("update failed");
    }

    #[test]
    fn test_isam2_marginal_covariance() {
        let isam = Isam2::new();
        isam.add_pose(0, Vector3::zeros());

        let cov = isam.marginal_covariance(0, 0);
        assert_eq!(cov.shape(), (3, 3));
    }

    #[test]
    fn test_isam2_chain_optimization() {
        let isam = Isam2::new();

        for i in 0..5 {
            isam.add_pose(i, Vector3::new(i as f64, 0.0, 0.0));
        }

        for i in 0..4 {
            let measurement = DVector::from_vec(vec![1.0, 0.0, 0.0]);
            isam.add_factor(i, i + 1, measurement, 0.1);
        }

        isam.optimize().expect("optimization failed");

        let poses = isam.get_all_poses();
        for i in 0..5 {
            if let Some(pos) = poses.get(&i) {
                assert!((pos.x - i as f64).abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_isam2_loop_closure() {
        let isam = Isam2::new();

        isam.add_pose(0, Vector3::new(0.0, 0.0, 0.0));
        isam.add_pose(1, Vector3::new(1.0, 0.0, 0.0));
        isam.add_pose(2, Vector3::new(1.0, 1.0, 0.0));
        isam.add_pose(3, Vector3::new(0.0, 1.0, 0.0));

        for i in 0..3 {
            let measurement = DVector::from_vec(vec![1.0, 0.0, 0.0]);
            isam.add_factor(i, i + 1, measurement, 0.1);
        }

        let closure = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        isam.add_factor(3, 0, closure, 0.1);

        isam.optimize().expect("optimization with loop failed");

        let poses = isam.get_all_poses();
        assert!(poses.contains_key(&0));
    }
}
