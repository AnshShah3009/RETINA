//! Common factor types for the factor graph framework.
//!
//! Provides ready-to-use factors: priors, relative transforms, range/bearing
//! measurements, and camera projection factors.

use crate::factor_graph::{Factor, Key, NoiseModel, Values, Variable};
use nalgebra::{DVector, Vector2, Vector3};

// ── PriorFactor ──────────────────────────────────────────────────────────────

/// Unary factor anchoring a variable to a fixed prior value.
pub struct PriorFactor {
    key: Key,
    prior: Variable,
    noise: NoiseModel,
}

impl PriorFactor {
    pub fn new(key: Key, prior: Variable, noise: NoiseModel) -> Self {
        debug_assert_eq!(prior.dim(), noise.dim());
        Self { key, prior, noise }
    }
}

impl Factor for PriorFactor {
    fn keys(&self) -> &[Key] {
        std::slice::from_ref(&self.key)
    }

    fn dim(&self) -> usize {
        self.prior.dim()
    }

    fn error(&self, values: &Values) -> DVector<f64> {
        let var = values.get(&self.key).expect("variable not found");
        self.prior.local(var)
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

// ── BetweenFactor ────────────────────────────────────────────────────────────

/// Binary factor measuring the relative transform between two poses.
///
/// For Pose3 variables the error is `log(measured^{-1} * (p1^{-1} * p2))`.
/// For Euclidean variables it is `(p2 - p1) - measured`.
pub struct BetweenFactor {
    keys: [Key; 2],
    measured: Variable,
    noise: NoiseModel,
}

impl BetweenFactor {
    pub fn new(key1: Key, key2: Key, measured: Variable, noise: NoiseModel) -> Self {
        debug_assert_eq!(measured.dim(), noise.dim());
        Self {
            keys: [key1, key2],
            measured,
            noise,
        }
    }
}

impl Factor for BetweenFactor {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn dim(&self) -> usize {
        self.measured.dim()
    }

    fn error(&self, values: &Values) -> DVector<f64> {
        let v1 = values.get(&self.keys[0]).expect("key1 not found");
        let v2 = values.get(&self.keys[1]).expect("key2 not found");

        match (&self.measured, v1, v2) {
            (Variable::Pose3(meas), Variable::Pose3(p1), Variable::Pose3(p2)) => {
                let predicted = p1.inverse() * p2;
                let d = meas.inverse() * predicted;
                let t = d.translation.vector;
                let r = d.rotation.scaled_axis();
                DVector::from_vec(vec![t.x, t.y, t.z, r.x, r.y, r.z])
            }
            (Variable::Pose2(meas), Variable::Pose2(p1), Variable::Pose2(p2)) => {
                // For Pose2 we compute relative and subtract measurement
                let c = p1.z.cos();
                let s = p1.z.sin();
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let local_x = c * dx + s * dy;
                let local_y = -s * dx + c * dy;
                let local_theta = p2.z - p1.z;
                DVector::from_vec(vec![
                    local_x - meas.x,
                    local_y - meas.y,
                    angle_wrap(local_theta - meas.z),
                ])
            }
            (Variable::Point3(meas), Variable::Point3(p1), Variable::Point3(p2)) => {
                let diff = p2 - p1;
                DVector::from_vec(vec![
                    diff.x - meas.x,
                    diff.y - meas.y,
                    diff.z - meas.z,
                ])
            }
            _ => {
                // Generic fallback
                let predicted_diff = v2.to_vector() - v1.to_vector();
                predicted_diff - self.measured.to_vector()
            }
        }
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

/// Wrap angle to [-pi, pi].
fn angle_wrap(mut a: f64) -> f64 {
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

// ── RangeFactor ──────────────────────────────────────────────────────────────

/// Measures Euclidean distance between two 3-D variables (poses or points).
///
/// For Pose3 variables the position (translation) is used.
pub struct RangeFactor {
    keys: [Key; 2],
    measured_range: f64,
    noise: NoiseModel,
}

impl RangeFactor {
    pub fn new(key1: Key, key2: Key, measured_range: f64, noise: NoiseModel) -> Self {
        debug_assert_eq!(noise.dim(), 1);
        Self {
            keys: [key1, key2],
            measured_range,
            noise,
        }
    }
}

fn position_of(var: &Variable) -> Vector3<f64> {
    match var {
        Variable::Pose3(iso) => iso.translation.vector,
        Variable::Point3(p) => p.coords,
        Variable::Pose2(v) => Vector3::new(v.x, v.y, 0.0),
        Variable::Point2(p) => Vector3::new(p[0], p[1], 0.0),
        _ => panic!("RangeFactor: unsupported variable type"),
    }
}

impl Factor for RangeFactor {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn dim(&self) -> usize {
        1
    }

    fn error(&self, values: &Values) -> DVector<f64> {
        let v1 = values.get(&self.keys[0]).expect("key1 not found");
        let v2 = values.get(&self.keys[1]).expect("key2 not found");
        let p1 = position_of(v1);
        let p2 = position_of(v2);
        let predicted = (p2 - p1).norm();
        DVector::from_vec(vec![predicted - self.measured_range])
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

// ── BearingFactor ────────────────────────────────────────────────────────────

/// Measures bearing (azimuth, elevation) from a Pose3 to a Point3.
///
/// The bearing is computed in the pose's local frame.
pub struct BearingFactor {
    keys: [Key; 2],
    measured: Vector2<f64>, // (azimuth, elevation) in radians
    noise: NoiseModel,
}

impl BearingFactor {
    /// `key_pose`: the observer pose, `key_point`: the observed landmark.
    /// `measured`: (azimuth, elevation) in radians.
    pub fn new(
        key_pose: Key,
        key_point: Key,
        measured: Vector2<f64>,
        noise: NoiseModel,
    ) -> Self {
        debug_assert_eq!(noise.dim(), 2);
        Self {
            keys: [key_pose, key_point],
            measured,
            noise,
        }
    }
}

impl Factor for BearingFactor {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn dim(&self) -> usize {
        2
    }

    fn error(&self, values: &Values) -> DVector<f64> {
        let pose = values.at_pose3(&self.keys[0]).expect("pose not found");
        let point = values.at_point3(&self.keys[1]).expect("point not found");

        // Transform point to pose local frame
        let local = pose.inverse_transform_point(point);
        let azimuth = local.y.atan2(local.x);
        let range_xy = (local.x * local.x + local.y * local.y).sqrt();
        let elevation = local.z.atan2(range_xy);

        DVector::from_vec(vec![
            angle_wrap(azimuth - self.measured.x),
            angle_wrap(elevation - self.measured.y),
        ])
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

// ── ProjectionFactor ─────────────────────────────────────────────────────────

/// Projects a 3-D point through a pinhole camera model to a 2-D observation.
///
/// Uses intrinsic parameters (fx, fy, cx, cy).  The pose maps world→camera
/// (camera-from-world convention).
pub struct ProjectionFactor {
    keys: [Key; 2],
    measured: Vector2<f64>, // observed (u, v) pixel coordinates
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    noise: NoiseModel,
}

impl ProjectionFactor {
    /// `key_pose`: camera pose (world-from-camera), `key_point`: 3-D landmark.
    /// `measured`: observed pixel (u, v).
    pub fn new(
        key_pose: Key,
        key_point: Key,
        measured: Vector2<f64>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        noise: NoiseModel,
    ) -> Self {
        debug_assert_eq!(noise.dim(), 2);
        Self {
            keys: [key_pose, key_point],
            measured,
            fx,
            fy,
            cx,
            cy,
            noise,
        }
    }
}

impl Factor for ProjectionFactor {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn dim(&self) -> usize {
        2
    }

    fn error(&self, values: &Values) -> DVector<f64> {
        let pose = values.at_pose3(&self.keys[0]).expect("camera pose not found");
        let point = values.at_point3(&self.keys[1]).expect("landmark not found");

        // Transform point to camera frame (pose is world-from-camera, so
        // camera_point = pose^{-1} * world_point)
        let cam_pt = pose.inverse_transform_point(point);

        if cam_pt.z.abs() < 1e-10 {
            // Point behind/at camera — return large residual
            return DVector::from_vec(vec![1e6, 1e6]);
        }

        let u = self.fx * cam_pt.x / cam_pt.z + self.cx;
        let v = self.fy * cam_pt.y / cam_pt.z + self.cy;

        DVector::from_vec(vec![u - self.measured.x, v - self.measured.y])
    }

    fn noise_model(&self) -> &NoiseModel {
        &self.noise
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_graph::{FactorGraph, GNConfig, LMParams};
    use nalgebra::{Isometry3, Point3};

    #[test]
    fn test_prior_factor_zero_error_at_prior() {
        let k = Key::symbol('x', 0);
        let prior_val = Variable::Point3(Point3::new(1.0, 2.0, 3.0));
        let factor = PriorFactor::new(k, prior_val, NoiseModel::Isotropic(1.0, 3));

        let mut values = Values::new();
        values.insert(k, Variable::Point3(Point3::new(1.0, 2.0, 3.0)));

        let err = factor.error(&values);
        assert!(err.norm() < 1e-10);
    }

    #[test]
    fn test_prior_factor_nonzero_error() {
        let k = Key::symbol('x', 0);
        let prior_val = Variable::Point3(Point3::origin());
        let factor = PriorFactor::new(k, prior_val, NoiseModel::Isotropic(1.0, 3));

        let mut values = Values::new();
        values.insert(k, Variable::Point3(Point3::new(1.0, 0.0, 0.0)));

        let err = factor.error(&values);
        assert!((err[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_between_factor_pose3() {
        let k1 = Key::symbol('x', 0);
        let k2 = Key::symbol('x', 1);
        let meas = Isometry3::translation(1.0, 0.0, 0.0);
        let factor = BetweenFactor::new(
            k1,
            k2,
            Variable::Pose3(meas),
            NoiseModel::Isotropic(0.1, 6),
        );

        let mut values = Values::new();
        values.insert(k1, Variable::Pose3(Isometry3::identity()));
        values.insert(k2, Variable::Pose3(Isometry3::translation(1.0, 0.0, 0.0)));

        let err = factor.error(&values);
        assert!(err.norm() < 1e-10, "error at exact should be zero: {:?}", err);
    }

    #[test]
    fn test_range_factor() {
        let k1 = Key::symbol('x', 0);
        let k2 = Key::symbol('l', 0);
        let factor = RangeFactor::new(k1, k2, 5.0, NoiseModel::Isotropic(0.1, 1));

        let mut values = Values::new();
        values.insert(k1, Variable::Point3(Point3::origin()));
        values.insert(k2, Variable::Point3(Point3::new(3.0, 4.0, 0.0)));

        let err = factor.error(&values);
        assert!(err[0].abs() < 1e-10, "range 5 at distance 5 should be 0");
    }

    #[test]
    fn test_bearing_factor() {
        let kp = Key::symbol('x', 0);
        let kl = Key::symbol('l', 0);

        // Point at (1, 0, 0) in world frame, pose at origin looking along +x
        let measured = Vector2::new(0.0, 0.0); // azimuth=0, elevation=0
        let factor = BearingFactor::new(kp, kl, measured, NoiseModel::Isotropic(0.01, 2));

        let mut values = Values::new();
        values.insert(kp, Variable::Pose3(Isometry3::identity()));
        values.insert(kl, Variable::Point3(Point3::new(1.0, 0.0, 0.0)));

        let err = factor.error(&values);
        assert!(err.norm() < 1e-10, "bearing should be exact: {:?}", err);
    }

    #[test]
    fn test_projection_factor() {
        let kp = Key::symbol('x', 0);
        let kl = Key::symbol('l', 0);

        // Camera at origin, point at (0, 0, 5)
        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // Expected projection: u = fx*0/5 + cx = 320, v = fy*0/5 + cy = 240
        let measured = Vector2::new(320.0, 240.0);
        let factor =
            ProjectionFactor::new(kp, kl, measured, fx, fy, cx, cy, NoiseModel::Isotropic(1.0, 2));

        let mut values = Values::new();
        values.insert(kp, Variable::Pose3(Isometry3::identity()));
        values.insert(kl, Variable::Point3(Point3::new(0.0, 0.0, 5.0)));

        let err = factor.error(&values);
        assert!(err.norm() < 1e-10, "projection at center should match: {:?}", err);
    }

    #[test]
    fn test_projection_factor_off_center() {
        let kp = Key::symbol('x', 0);
        let kl = Key::symbol('l', 0);

        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // Point at (1, 2, 10): u = 500*1/10 + 320 = 370, v = 500*2/10 + 240 = 340
        let measured = Vector2::new(370.0, 340.0);
        let factor =
            ProjectionFactor::new(kp, kl, measured, fx, fy, cx, cy, NoiseModel::Isotropic(1.0, 2));

        let mut values = Values::new();
        values.insert(kp, Variable::Pose3(Isometry3::identity()));
        values.insert(kl, Variable::Point3(Point3::new(1.0, 2.0, 10.0)));

        let err = factor.error(&values);
        assert!(err.norm() < 1e-10, "projection off-center: {:?}", err);
    }

    #[test]
    fn test_pose_chain_optimization_with_factors() {
        // 3-pose chain: x0 --[1m]--> x1 --[1m]--> x2
        let mut graph = FactorGraph::new();
        let x0 = Key::symbol('x', 0);
        let x1 = Key::symbol('x', 1);
        let x2 = Key::symbol('x', 2);

        // Anchor x0
        graph.add(PriorFactor::new(
            x0,
            Variable::Pose3(Isometry3::identity()),
            NoiseModel::Isotropic(0.001, 6),
        ));

        graph.add(BetweenFactor::new(
            x0,
            x1,
            Variable::Pose3(Isometry3::translation(1.0, 0.0, 0.0)),
            NoiseModel::Isotropic(0.1, 6),
        ));

        graph.add(BetweenFactor::new(
            x1,
            x2,
            Variable::Pose3(Isometry3::translation(1.0, 0.0, 0.0)),
            NoiseModel::Isotropic(0.1, 6),
        ));

        let mut initial = Values::new();
        initial.insert(x0, Variable::Pose3(Isometry3::translation(0.1, 0.05, 0.0)));
        initial.insert(x1, Variable::Pose3(Isometry3::translation(1.2, -0.1, 0.0)));
        initial.insert(x2, Variable::Pose3(Isometry3::translation(2.3, 0.1, 0.0)));

        let result = graph
            .optimize_gn(&initial, &GNConfig::default())
            .unwrap();

        let p0 = result.at_pose3(&x0).unwrap();
        let p1 = result.at_pose3(&x1).unwrap();
        let p2 = result.at_pose3(&x2).unwrap();

        assert!(p0.translation.vector.norm() < 0.05, "x0 near origin");
        assert!((p1.translation.vector.x - 1.0).abs() < 0.1, "x1 near 1m");
        assert!((p2.translation.vector.x - 2.0).abs() < 0.1, "x2 near 2m");
    }

    #[test]
    fn test_landmark_optimization() {
        // Two known poses observing a single landmark via range
        let mut graph = FactorGraph::new();
        let x0 = Key::symbol('x', 0);
        let x1 = Key::symbol('x', 1);
        let l0 = Key::symbol('l', 0);

        // Fix both poses with strong priors
        graph.add(PriorFactor::new(
            x0,
            Variable::Point3(Point3::origin()),
            NoiseModel::Isotropic(0.001, 3),
        ));
        graph.add(PriorFactor::new(
            x1,
            Variable::Point3(Point3::new(4.0, 0.0, 0.0)),
            NoiseModel::Isotropic(0.001, 3),
        ));

        // Range from x0 to l0: 5.0
        graph.add(RangeFactor::new(
            x0,
            l0,
            5.0,
            NoiseModel::Isotropic(0.1, 1),
        ));
        // Range from x1 to l0: 3.0 (landmark at (4, 3, 0) satisfies both)
        // dist((0,0,0), (4,3,0)) = 5, dist((4,0,0), (4,3,0)) = 3
        graph.add(RangeFactor::new(
            x1,
            l0,
            3.0,
            NoiseModel::Isotropic(0.1, 1),
        ));

        let mut initial = Values::new();
        initial.insert(x0, Variable::Point3(Point3::origin()));
        initial.insert(x1, Variable::Point3(Point3::new(4.0, 0.0, 0.0)));
        // Start landmark estimate with some noise
        initial.insert(l0, Variable::Point3(Point3::new(3.0, 2.0, 0.0)));

        let result = graph
            .optimize_lm(&initial, &LMParams::default())
            .unwrap();

        let lm = result.at_point3(&l0).unwrap();
        // Landmark should converge near (4, 3, 0) (or mirror (4, -3, 0))
        let d0 = (lm.coords - Vector3::new(4.0, 3.0, 0.0)).norm();
        let d1 = (lm.coords - Vector3::new(4.0, -3.0, 0.0)).norm();
        assert!(
            d0 < 0.5 || d1 < 0.5,
            "landmark should be near (4,3,0) or mirror, got {:?}",
            lm
        );
    }

    #[test]
    fn test_projection_bundle_adjustment() {
        // Simple BA: 1 camera, 2 points
        let mut graph = FactorGraph::new();
        let cam = Key::symbol('x', 0);
        let l0 = Key::symbol('l', 0);
        let l1 = Key::symbol('l', 1);

        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // Fix camera at origin
        graph.add(PriorFactor::new(
            cam,
            Variable::Pose3(Isometry3::identity()),
            NoiseModel::Isotropic(0.001, 6),
        ));

        // Point at (1, 0, 5) -> pixel (420, 240)
        graph.add(ProjectionFactor::new(
            cam,
            l0,
            Vector2::new(420.0, 240.0),
            fx, fy, cx, cy,
            NoiseModel::Isotropic(1.0, 2),
        ));

        // Point at (0, 1, 5) -> pixel (320, 340)
        graph.add(ProjectionFactor::new(
            cam,
            l1,
            Vector2::new(320.0, 340.0),
            fx, fy, cx, cy,
            NoiseModel::Isotropic(1.0, 2),
        ));

        let mut initial = Values::new();
        initial.insert(cam, Variable::Pose3(Isometry3::identity()));
        initial.insert(l0, Variable::Point3(Point3::new(1.2, 0.1, 5.5)));
        initial.insert(l1, Variable::Point3(Point3::new(0.1, 1.2, 4.5)));

        let result = graph
            .optimize_lm(&initial, &LMParams::default())
            .unwrap();

        let p0 = result.at_point3(&l0).unwrap();
        let p1 = result.at_point3(&l1).unwrap();

        assert!(
            (p0.x - 1.0).abs() < 0.5,
            "l0.x ~ 1.0, got {}",
            p0.x
        );
        assert!(
            (p1.y - 1.0).abs() < 0.5,
            "l1.y ~ 1.0, got {}",
            p1.y
        );
    }
}
