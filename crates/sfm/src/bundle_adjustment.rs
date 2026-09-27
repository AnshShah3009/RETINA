use cv_core::{CameraIntrinsics, Pose};
use cv_runtime::orchestrator::{scheduler, ResourceGroup};
use nalgebra::{DMatrix, DVector, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use rayon::prelude::*;

use cv_optimize::sparse::{CgSolver, LinearSolver, SparseMatrix, Triplet};

#[derive(Clone)]
pub struct Landmark {
    pub position: Point3<f64>,
    pub observations: Vec<(usize, Point2<f64>)>,
    pub is_valid: bool,
}

impl Landmark {
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            observations: Vec::new(),
            is_valid: true,
        }
    }

    pub fn add_observation(&mut self, cam_idx: usize, obs: Point2<f64>) {
        self.observations.push((cam_idx, obs));
    }
}

#[derive(Clone)]
pub struct SfMState {
    pub cameras: Vec<Pose>,
    pub landmarks: Vec<Landmark>,
    pub intrinsics: CameraIntrinsics,
}

impl SfMState {
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            cameras: Vec::new(),
            landmarks: Vec::new(),
            intrinsics,
        }
    }

    pub fn add_camera(&mut self, pose: Pose) -> usize {
        let id = self.cameras.len();
        self.cameras.push(pose);
        id
    }

    pub fn add_landmark(
        &mut self,
        position: Point3<f64>,
        observations: Vec<(usize, Point2<f64>)>,
    ) -> usize {
        let id = self.landmarks.len();
        let mut landmark = Landmark::new(position);
        for (cam_idx, obs) in observations {
            landmark.add_observation(cam_idx, obs);
        }
        self.landmarks.push(landmark);
        id
    }

    pub fn total_reprojection_error(&self) -> f64 {
        let mut total_err = 0.0;
        let mut count = 0;

        for landmark in &self.landmarks {
            if !landmark.is_valid {
                continue;
            }

            for (cam_idx, obs) in &landmark.observations {
                if *cam_idx >= self.cameras.len() {
                    continue;
                }

                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;

                if pt_cam.z <= 0.0 {
                    total_err += 1e6;
                    // Count penalized observations toward the mean — dividing
                    // by valid-only counts made the metric grow without bound
                    // as more points fell behind the camera.
                    count += 1;
                    continue;
                }

                let projected = self.intrinsics.project(&pt_cam);
                let err = (projected.x - obs.x).powi(2) + (projected.y - obs.y).powi(2);
                total_err += err;
                count += 1;
            }
        }

        if count > 0 {
            total_err / count as f64
        } else {
            0.0
        }
    }

    pub fn to_parameters(&self) -> DVector<f64> {
        let n_cam = self.cameras.len();
        let n_lm = self.landmarks.len();
        let mut params = DVector::zeros(6 * n_cam + 3 * n_lm);

        for (i, cam) in self.cameras.iter().enumerate() {
            let rotation = cam.rotation.to_rotation_matrix();
            let axis_angle = rotation.scaled_axis();
            params[6 * i] = axis_angle.x;
            params[6 * i + 1] = axis_angle.y;
            params[6 * i + 2] = axis_angle.z;
            params[6 * i + 3] = cam.translation.x;
            params[6 * i + 4] = cam.translation.y;
            params[6 * i + 5] = cam.translation.z;
        }

        let offset = 6 * n_cam;
        for (i, lm) in self.landmarks.iter().enumerate() {
            params[offset + 3 * i] = lm.position.x;
            params[offset + 3 * i + 1] = lm.position.y;
            params[offset + 3 * i + 2] = lm.position.z;
        }

        params
    }

    pub fn from_parameters(&mut self, params: &DVector<f64>) {
        let n_cam = self.cameras.len();
        for (i, cam) in self.cameras.iter_mut().enumerate() {
            let axis_angle = Vector3::new(params[6 * i], params[6 * i + 1], params[6 * i + 2]);
            cam.rotation = UnitQuaternion::new(axis_angle);
            cam.translation = Vector3::new(params[6 * i + 3], params[6 * i + 4], params[6 * i + 5]);
        }

        let offset = 6 * n_cam;
        for (i, lm) in self.landmarks.iter_mut().enumerate() {
            lm.position = Point3::new(
                params[offset + 3 * i],
                params[offset + 3 * i + 1],
                params[offset + 3 * i + 2],
            );
        }
    }

    pub fn residuals(&self) -> DVector<f64> {
        let mut residuals = Vec::new();
        for landmark in &self.landmarks {
            for (cam_idx, obs) in &landmark.observations {
                if *cam_idx >= self.cameras.len() {
                    continue;
                }

                if !landmark.is_valid {
                    // Push zeros for invalid landmarks to maintain consistent dimensions
                    residuals.push(0.0);
                    residuals.push(0.0);
                    continue;
                }

                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;
                let projected = self.intrinsics.project(&pt_cam);
                residuals.push(projected.x - obs.x);
                residuals.push(projected.y - obs.y);
            }
        }
        DVector::from_vec(residuals)
    }

    /// Dense Jacobian of the reprojection residuals.
    ///
    /// This is the same analytic derivative as [`Self::numerical_jacobian_sparse`],
    /// materialised dense. It used to be built by central differences: for each
    /// parameter it cloned the whole parameter vector twice and recomputed every
    /// residual, i.e. O(parameters x residuals) full state rebuilds — with ~1,400
    /// landmarks that is thousands of rebuilds per bundle-adjustment iteration,
    /// which is what made the sequential solver appear to hang.
    pub fn numerical_jacobian(&self) -> DMatrix<f64> {
        let sparse = self.numerical_jacobian_sparse();
        sparse.to_dense()
    }

    /// Dense Jacobian computed on the compute device.
    ///
    /// Kept for API compatibility. The analytic derivative is cheap enough that
    /// splitting it across the pool costs more in scheduling than it saves, so
    /// this now simply returns [`Self::numerical_jacobian`]; the group is unused.
    #[allow(clippy::needless_range_loop)]
    pub fn numerical_jacobian_ctx(&self, _group: &ResourceGroup) -> DMatrix<f64> {
        self.numerical_jacobian()
    }

    /// Analytic sparse Jacobian of the reprojection residuals.
    ///
    /// The residual of an observation is the 2D projection error of a landmark
    /// seen by a camera, parameterised as in [`Self::to_parameters`]: each camera
    /// is a left-multiplied axis-angle (scaled-axis) vector plus a translation,
    /// each landmark is a 3D point. Differentiating that closed form is exact and
    /// costs O(observations), because a row depends only on the one camera and
    /// the one landmark it observes.
    ///
    /// This replaces a finite-difference construction that cloned the whole
    /// parameter vector once per parameter per observation — O(observations x
    /// parameters) full state rebuilds, which made bundle adjustment appear to
    /// hang (a 12-camera / 1400-landmark reconstruction spent minutes in one
    /// Jacobian). `numerical_jacobian_fd` keeps the old scheme as a test oracle.
    pub fn numerical_jacobian_sparse(&self) -> SparseMatrix {
        let params = self.to_parameters();
        let n_res = self.residuals().len();
        let n_params = params.len();
        let n_cam = self.cameras.len();
        let landmark_offset = 6 * n_cam;

        let mut triplets = Vec::new();

        let mut res_idx = 0;
        for (lm_idx, lm) in self.landmarks.iter().enumerate() {
            if !lm.is_valid {
                // residuals() emits two zero rows for invalid landmarks, but only
                // for observations whose camera index is in range; out-of-range
                // camera indices emit no rows at all. Match that exactly so the
                // row cursor stays aligned with the residual vector.
                res_idx += 2 * lm.observations.iter().filter(|(ci, _)| *ci < n_cam).count();
                continue;
            }
            for (cam_idx, _obs) in &lm.observations {
                // residuals() skips out-of-range cameras without emitting rows
                if *cam_idx >= n_cam {
                    continue;
                }
                let cam = &self.cameras[*cam_idx];

                // Point in camera coordinates and its projection.
                let p_cam = cam.rotation * lm.position + cam.translation;
                let (x, y, z) = (p_cam.x, p_cam.y, p_cam.z);
                if z.abs() < 1e-10 {
                    // The projection is clamped here; the derivative is not
                    // meaningful, so emit zeros for this observation.
                    res_idx += 2;
                    continue;
                }
                let inv_z = 1.0 / z;
                let inv_z2 = inv_z * inv_z;
                // d(pixel)/d(camera point) — the standard pinhole projection
                // Jacobian, rows for x and y respectively.
                let (dx_dxc, dx_dyc, dx_dzc) = (
                    self.intrinsics.fx * inv_z,
                    0.0,
                    -self.intrinsics.fx * x * inv_z2,
                );
                let (dy_dxc, dy_dyc, dy_dzc) = (
                    0.0,
                    self.intrinsics.fy * inv_z,
                    -self.intrinsics.fy * y * inv_z2,
                );

                // `from_parameters` builds the rotation as
                // `UnitQuaternion::new(scaled_axis)`, which nalgebra evaluates
                // through the quaternion (Halley) form
                //     R = I + 2aK + 2K^2,   a = cos(t/2),   K = skew(s*w),
                // with t = |w| and s = sin(t/2)/t. The textbook closed form
                // expm([w]_x) is not what the library computes, so the derivative
                // has to follow the same parameterisation; the finite-difference
                // oracle in the tests pins this.
                //
                // With e_i the i-th basis vector:
                //   da/dw_i = -(sin(t/2)/2) * (w_i/t)
                //   ds/dw_i = ds/dt * (w_i/t),
                //             ds/dt = (t*cos(t/2)/2 - sin(t/2)) / t^2
                //   dK/dw_i = skew(s*e_i + (ds/dw_i)*w)
                //   dR/dw_i = 2(da/dw_i)K + 2a(dK/dw_i) + 2(dK/dw_i * K + K * dK/dw_i)
                // and d(p_cam)/d(w_i) = dR/dw_i * p, p being the landmark in
                // world coordinates.
                let w = cam.rotation.to_rotation_matrix().scaled_axis();
                let theta = w.norm();
                let half = 0.5 * theta;
                let (a, da_dt, s, ds_dt): (f64, f64, f64, f64) = if theta < 1e-12 {
                    // limits as t -> 0: cos(t/2) -> 1, sin(t/2)/t -> 1/2.
                    (1.0f64, 0.0f64, 0.5f64, -1.0 / 24.0)
                } else {
                    (
                        half.cos(),
                        -half.sin() * 0.5,
                        half.sin() / theta,
                        (half * half.cos() - half.sin()) / (theta * theta),
                    )
                };
                let p_vec = Vector3::new(lm.position.x, lm.position.y, lm.position.z);
                let skew_of = |v: Vector3<f64>| {
                    nalgebra::Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
                };
                let k_mat = skew_of(s * w);

                // Camera block: columns 6*cam + 0..6 (axis-angle, then translation).
                for i in 0..3 {
                    let dtw = if theta < 1e-12 { 0.0 } else { w[i] / theta };
                    let da = da_dt * dtw;
                    let ds = ds_dt * dtw;
                    let mut e_i = Vector3::zeros();
                    e_i[i] = s;
                    e_i += ds * w;
                    let dk = skew_of(e_i);
                    let dpc = (2.0 * da * (k_mat * p_vec)
                        + 2.0 * a * (dk * p_vec)
                        + 2.0 * ((dk * k_mat + k_mat * dk) * p_vec));
                    let dx = dx_dxc * dpc.x + dx_dyc * dpc.y + dx_dzc * dpc.z;
                    let dy = dy_dxc * dpc.x + dy_dyc * dpc.y + dy_dzc * dpc.z;
                    let col = 6 * cam_idx + i;
                    triplets.push(Triplet::new(res_idx, col, dx));
                    triplets.push(Triplet::new(res_idx + 1, col, dy));
                }
                for k in 0..3 {
                    let dx = if k == 0 {
                        dx_dxc
                    } else if k == 1 {
                        dx_dyc
                    } else {
                        dx_dzc
                    };
                    let dy = if k == 0 {
                        dy_dxc
                    } else if k == 1 {
                        dy_dyc
                    } else {
                        dy_dzc
                    };
                    let col = 6 * cam_idx + 3 + k;
                    triplets.push(Triplet::new(res_idx, col, dx));
                    triplets.push(Triplet::new(res_idx + 1, col, dy));
                }

                // Landmark block: d(camera point)/d(landmark) = R.
                let r = cam.rotation.to_rotation_matrix().into_inner();
                for k in 0..3 {
                    let col = landmark_offset + 3 * lm_idx + k;
                    let dx = dx_dxc * r[(0, k)] + dx_dyc * r[(1, k)] + dx_dzc * r[(2, k)];
                    let dy = dy_dxc * r[(0, k)] + dy_dyc * r[(1, k)] + dy_dzc * r[(2, k)];
                    triplets.push(Triplet::new(res_idx, col, dx));
                    triplets.push(Triplet::new(res_idx + 1, col, dy));
                }
                res_idx += 2;
            }
        }

        let _ = params;
        SparseMatrix::from_triplets(n_res, n_params, &triplets)
    }

    /// Finite-difference Jacobian, kept as a test oracle for the analytic one.
    ///
    /// This is the original O(observations x parameters) construction. It is
    /// never used by the solver; `#[cfg(test)]` is applied by the caller.
    #[cfg(test)]
    fn numerical_jacobian_fd(&self) -> SparseMatrix {
        let params = self.to_parameters();
        let n_res = self.residuals().len();
        let n_params = params.len();
        let n_cam = self.cameras.len();
        let eps = 1e-6;

        let mut triplets = Vec::new();

        let mut res_idx = 0;
        for (lm_idx, lm) in self.landmarks.iter().enumerate() {
            if !lm.is_valid {
                res_idx += 2 * lm.observations.iter().filter(|(ci, _)| *ci < n_cam).count();
                continue;
            }
            for (cam_idx, _obs) in &lm.observations {
                if *cam_idx >= n_cam {
                    continue;
                }

                for k in 0..6 {
                    let idx = 6 * cam_idx + k;
                    let mut p_plus = params.clone();
                    p_plus[idx] += eps;
                    let mut p_minus = params.clone();
                    p_minus[idx] -= eps;
                    let (pp, _) = self.compute_residuals_for_param_local(&p_plus, *cam_idx, lm_idx);
                    let (pm, _) =
                        self.compute_residuals_for_param_local(&p_minus, *cam_idx, lm_idx);
                    triplets.push(Triplet::new(res_idx, idx, (pp.x - pm.x) / (2.0 * eps)));
                    triplets.push(Triplet::new(res_idx + 1, idx, (pp.y - pm.y) / (2.0 * eps)));
                }

                let offset = 6 * n_cam;
                for k in 0..3 {
                    let idx = offset + 3 * lm_idx + k;
                    let mut p_plus = params.clone();
                    p_plus[idx] += eps;
                    let mut p_minus = params.clone();
                    p_minus[idx] -= eps;
                    let (pp, _) = self.compute_residuals_for_param_local(&p_plus, *cam_idx, lm_idx);
                    let (pm, _) =
                        self.compute_residuals_for_param_local(&p_minus, *cam_idx, lm_idx);
                    triplets.push(Triplet::new(res_idx, idx, (pp.x - pm.x) / (2.0 * eps)));
                    triplets.push(Triplet::new(res_idx + 1, idx, (pp.y - pm.y) / (2.0 * eps)));
                }
                res_idx += 2;
            }
        }

        SparseMatrix::from_triplets(n_res, n_params, &triplets)
    }

    fn compute_residuals_for_param_local(
        &self,
        params: &DVector<f64>,
        cam_idx: usize,
        lm_idx: usize,
    ) -> (Point2<f64>, Point2<f64>) {
        let axis_angle = Vector3::new(
            params[6 * cam_idx],
            params[6 * cam_idx + 1],
            params[6 * cam_idx + 2],
        );
        let rot = Rotation3::new(axis_angle).into_inner();
        let trans = Vector3::new(
            params[6 * cam_idx + 3],
            params[6 * cam_idx + 4],
            params[6 * cam_idx + 5],
        );

        let n_cam = self.cameras.len();
        let offset = 6 * n_cam;
        let lm_pos = Point3::new(
            params[offset + 3 * lm_idx],
            params[offset + 3 * lm_idx + 1],
            params[offset + 3 * lm_idx + 2],
        );

        let pt_cam = rot * lm_pos + trans;
        let proj = self.intrinsics.project(&pt_cam);
        (proj, proj) // Dummy second for signature parity
    }

    fn compute_residuals_for_param(
        &self,
        params_plus: &DVector<f64>,
        params_minus: &DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        let mut state_plus = self.clone();
        let mut state_minus = self.clone();

        state_plus.from_parameters(params_plus);
        state_minus.from_parameters(params_minus);

        (state_plus.residuals(), state_minus.residuals())
    }

    fn compute_point_reprojection_error_index(&self, lm: &Landmark) -> f64 {
        let mut error = 0.0;
        let mut count = 0;

        for (cam_idx, obs) in &lm.observations {
            if *cam_idx >= self.cameras.len() {
                continue;
            }
            let cam = &self.cameras[*cam_idx];
            let pt_cam = cam.rotation * lm.position + cam.translation;

            if pt_cam.z > 0.0 {
                let proj = self.intrinsics.project(&pt_cam);
                error += (proj.x - obs.x).powi(2) + (proj.y - obs.y).powi(2);
                count += 1;
            }
        }

        if count > 0 {
            error / count as f64
        } else {
            0.0
        }
    }

    pub fn remove_outliers(&mut self, threshold: f64) {
        let outlier_indices: Vec<usize> = self
            .landmarks
            .iter()
            .enumerate()
            .filter(|(_, lm)| self.compute_point_reprojection_error_index(lm) > threshold)
            .map(|(i, _)| i)
            .collect();

        for idx in outlier_indices {
            self.landmarks[idx].is_valid = false;
        }
    }
}

use cv_optimize::{CostFunction, SparseLMSolver};

impl CostFunction for SfMState {
    fn dimensions(&self) -> (usize, usize) {
        // Must match residuals(): every observation contributes two rows,
        // including zero rows for invalid landmarks; out-of-range camera
        // indices are skipped.
        let n_res = self
            .landmarks
            .iter()
            .map(|l| {
                l.observations
                    .iter()
                    .filter(|(ci, _)| *ci < self.cameras.len())
                    .count()
                    * 2
            })
            .sum();
        let n_params = 6 * self.cameras.len() + 3 * self.landmarks.len();
        (n_res, n_params)
    }

    fn residuals(&self, params: &DVector<f64>) -> DVector<f64> {
        let mut temp_state = self.clone();
        temp_state.from_parameters(params);
        temp_state.residuals()
    }

    fn jacobian(&self, params: &DVector<f64>) -> SparseMatrix {
        let mut temp_state = self.clone();
        temp_state.from_parameters(params);
        temp_state.numerical_jacobian_sparse()
    }
}

pub struct BundleAdjustmentConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub lambda: f64,
    pub use_sparsity: bool,
    pub robust_kernel: bool,
}

impl Default for BundleAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            lambda: 0.001,
            use_sparsity: true,
            robust_kernel: true,
        }
    }
}

pub fn bundle_adjust(state: &mut SfMState, config: &BundleAdjustmentConfig) {
    // use_sparsity=false requests the dense (sequential) solver explicitly;
    // a previous revision ignored the flag entirely. The ctx (sparse) path does
    // not implement the robust-kernel outlier rejection, so a request for it
    // must not silently fall through to that path.
    if !config.use_sparsity || config.robust_kernel {
        bundle_adjust_sequential(state, config);
        return;
    }

    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            if bundle_adjust_ctx(state, config, &group) {
                return;
            }
            // Ctx path unavailable (no compute device / solver failure):
            // fall through to the sequential CPU implementation.
        }
    }
    bundle_adjust_sequential(state, config);
}

/// Sequential CPU Levenberg-Marquardt used when the runtime ctx path is
/// unavailable or fails.
fn bundle_adjust_sequential(state: &mut SfMState, config: &BundleAdjustmentConfig) {
    let mut current_params = state.to_parameters();
    let mut current_residuals = state.residuals();
    let mut current_err = current_residuals.norm_squared();
    let mut lambda = config.lambda;
    // Stall detection: see SparseLMSolver::minimize.
    let mut rejections = 0u32;

    // The normal equations stay sparse. Materialising J dense and Cholesky-ing
    // J^T J was O(parameters^3) on a system that is mostly empty: with 9 cameras
    // and ~1,000 landmarks (3,060 parameters) a single bundle adjustment took
    // 35 seconds, almost all of it in the dense factorisation. A sparse
    // conjugate-gradient solve on J^T J + lambda*diag(J^T J) touches only the
    // nonzeros and costs milliseconds for the same result.
    let n_params = current_params.len();
    let cpu = match cv_hal::cpu::CpuBackend::new() {
        Some(cpu) => cpu,
        None => return, // no CPU backend: leave the state untouched
    };
    let device = cv_hal::compute::ComputeDevice::Cpu(&cpu);
    let cg = CgSolver {
        max_iters: 200,
        tolerance: 1e-10,
    };

    for iteration in 0..config.max_iterations {
        let j = state.numerical_jacobian_sparse();
        let r = &current_residuals;

        // (J^T J + lambda*diag(J^T J)) delta = -J^T r, built from the sparse
        // Jacobian so the system never becomes dense.
        let (mut lhs, jtr) = j.normal_equations_sparse(r);
        let neg_jtr = -jtr;
        // Marquardt damping: scale the diagonal, keeping the system SPD.
        lhs.scale_diagonal(1.0 + lambda);

        let delta = cg
            .solve(&device, &lhs, &neg_jtr)
            .unwrap_or_else(|_| DVector::zeros(n_params));

        let next_params = &current_params + &delta;
        state.from_parameters(&next_params);
        let next_residuals = state.residuals();
        let next_err = next_residuals.norm_squared();

        if next_err < current_err {
            current_params = next_params;
            current_residuals = next_residuals;
            current_err = next_err;
            lambda /= 10.0;
            rejections = 0;
            if delta.norm() < config.convergence_threshold {
                break;
            }
        } else {
            lambda *= 10.0;
            rejections += 1;
            state.from_parameters(&current_params);
            if rejections >= 12 || !lambda.is_finite() {
                break; // stalled: keep best parameters
            }
        }

        if config.robust_kernel && iteration % 5 == 0 {
            state.remove_outliers(10.0);
        }
    }
}

pub fn bundle_adjust_ctx(
    state: &mut SfMState,
    config: &BundleAdjustmentConfig,
    group: &ResourceGroup,
) -> bool {
    let device = match group.device() {
        Ok(dev) => dev,
        Err(_) => return false, // No compute device: caller falls back to CPU
    };
    let solver = SparseLMSolver {
        ctx: &device,
        config: cv_optimize::LMConfig {
            max_iters: config.max_iterations,
            lambda: config.lambda,
            tolerance: config.convergence_threshold,
        },
    };

    let initial_params = state.to_parameters();
    match solver.minimize(state, initial_params) {
        Ok(final_params) => {
            state.from_parameters(&final_params);
            true
        }
        Err(_) => false, // Solver failed: caller falls back to CPU
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Pose;
    use nalgebra::{Matrix3, Vector3};

    fn create_test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        }
    }

    fn create_test_pose(translation: Vector3<f64>) -> Pose {
        Pose::new(Matrix3::identity(), translation)
    }

    #[test]
    fn test_sfm_state_creation() {
        let intrinsics = create_test_intrinsics();
        let state = SfMState::new(intrinsics);

        assert_eq!(state.cameras.len(), 0);
        assert_eq!(state.landmarks.len(), 0);
    }

    #[test]
    fn test_add_camera() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let pose = create_test_pose(Vector3::zeros());
        let id1 = state.add_camera(pose.clone());
        assert_eq!(id1, 0);
        assert_eq!(state.cameras.len(), 1);

        let pose2 = create_test_pose(Vector3::new(1.0, 0.0, 0.0));
        let id2 = state.add_camera(pose2);
        assert_eq!(id2, 1);
        assert_eq!(state.cameras.len(), 2);
    }

    #[test]
    fn test_landmark_creation() {
        let pos = Point3::new(0.0, 0.0, 5.0);
        let landmark = Landmark::new(pos);

        assert_eq!(landmark.position, pos);
        assert_eq!(landmark.observations.len(), 0);
        assert!(landmark.is_valid);
    }

    #[test]
    fn test_add_landmark() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let observations = vec![(0, Point2::new(320.0, 240.0))];
        let id = state.add_landmark(Point3::new(0.0, 0.0, 5.0), observations);

        assert_eq!(id, 0);
        assert_eq!(state.landmarks.len(), 1);
        assert_eq!(state.landmarks[0].observations.len(), 1);
    }

    #[test]
    fn test_landmark_add_observation() {
        let mut landmark = Landmark::new(Point3::new(0.0, 0.0, 5.0));

        landmark.add_observation(0, Point2::new(320.0, 240.0));
        assert_eq!(landmark.observations.len(), 1);

        landmark.add_observation(1, Point2::new(300.0, 250.0));
        assert_eq!(landmark.observations.len(), 2);
    }

    #[test]
    fn test_total_reprojection_error_empty() {
        let intrinsics = create_test_intrinsics();
        let state = SfMState::new(intrinsics);

        let error = state.total_reprojection_error();
        assert_eq!(error, 0.0);
    }

    #[test]
    fn test_total_reprojection_error_perfect_projection() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // Add camera at origin
        let pose = create_test_pose(Vector3::zeros());
        state.add_camera(pose);

        // Add 3D point in front of camera
        let point_3d = Point3::new(0.0, 0.0, 5.0);
        // Project it to expected 2D location (center of image for point at origin)
        let obs_2d = Point2::new(320.0, 240.0); // Principal point
        let obs = vec![(0, obs_2d)];

        state.add_landmark(point_3d, obs);

        let error = state.total_reprojection_error();
        // Error should be very small for perfectly projected point
        assert!(error < 1.0);
    }

    #[test]
    fn test_total_reprojection_error_with_offset() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let pose = create_test_pose(Vector3::zeros());
        state.add_camera(pose);

        let point_3d = Point3::new(0.0, 0.0, 5.0);
        // Observation with intentional offset
        let obs_2d = Point2::new(330.0, 240.0); // 10 pixels off
        let obs = vec![(0, obs_2d)];

        state.add_landmark(point_3d, obs);

        let error = state.total_reprojection_error();
        // Error should reflect the 10-pixel offset
        assert!(error > 0.0);
    }

    #[test]
    fn test_to_and_from_parameters() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // Add camera and landmarks
        let pose = create_test_pose(Vector3::new(1.0, 2.0, 3.0));
        state.add_camera(pose);

        let point_3d = Point3::new(1.0, 2.0, 5.0);
        state.add_landmark(point_3d, vec![]);

        // Extract parameters
        let params = state.to_parameters();

        // Should have 6 camera params + 3 landmark params
        assert_eq!(params.len(), 6 + 3);

        // Create new state with same intrinsics
        let mut state2 = SfMState::new(intrinsics);
        state2.add_camera(create_test_pose(Vector3::zeros()));
        state2.add_landmark(Point3::new(0.0, 0.0, 0.0), vec![]);

        // Load parameters from state1 into state2
        state2.from_parameters(&params);

        // Verify translation matches
        assert!((state2.cameras[0].translation.x - 1.0).abs() < 1e-6);
        assert!((state2.cameras[0].translation.y - 2.0).abs() < 1e-6);
        assert!((state2.cameras[0].translation.z - 3.0).abs() < 1e-6);

        // Verify landmark position matches
        assert!((state2.landmarks[0].position.x - 1.0).abs() < 1e-6);
        assert!((state2.landmarks[0].position.y - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_residuals_computation() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let pose = create_test_pose(Vector3::zeros());
        state.add_camera(pose);

        let point_3d = Point3::new(0.0, 0.0, 5.0);
        let obs = vec![(0, Point2::new(320.0, 240.0))];
        state.add_landmark(point_3d, obs);

        let residuals = state.residuals();
        // Should have 2 residuals per observation (x and y error)
        assert_eq!(residuals.len(), 2);
    }

    #[test]
    fn test_remove_outliers() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let pose = create_test_pose(Vector3::zeros());
        state.add_camera(pose);

        // Add landmark with perfect observation
        let point_3d = Point3::new(0.0, 0.0, 5.0);
        let obs_good = vec![(0, Point2::new(320.0, 240.0))];
        state.add_landmark(point_3d, obs_good);

        // Add landmark with bad observation (large error)
        let point_3d_bad = Point3::new(10.0, 10.0, 5.0);
        let obs_bad = vec![(0, Point2::new(100.0, 100.0))];
        state.add_landmark(point_3d_bad, obs_bad);

        assert_eq!(state.landmarks.len(), 2);
        assert!(state.landmarks[0].is_valid);
        assert!(state.landmarks[1].is_valid);

        // Remove outliers with low threshold
        state.remove_outliers(1.0);

        // Good landmark should still be valid, bad one should be invalid
        assert!(state.landmarks[0].is_valid);
        assert!(!state.landmarks[1].is_valid);
    }

    #[test]
    fn test_bundle_adjustment_config_default() {
        let config = BundleAdjustmentConfig::default();

        assert_eq!(config.max_iterations, 100);
        assert!(config.convergence_threshold < 1e-5);
        assert!(config.use_sparsity);
        assert!(config.robust_kernel);
    }

    #[test]
    fn test_bundle_adjustment_config_custom() {
        let config = BundleAdjustmentConfig {
            max_iterations: 50,
            convergence_threshold: 1e-4,
            lambda: 0.01,
            use_sparsity: false,
            robust_kernel: false,
        };

        assert_eq!(config.max_iterations, 50);
        assert!((config.convergence_threshold - 1e-4).abs() < 1e-10);
        assert!((config.lambda - 0.01).abs() < 1e-6);
        assert!(!config.use_sparsity);
        assert!(!config.robust_kernel);
    }

    #[test]
    fn test_dimensions_consistency() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // Add 2 cameras
        state.add_camera(create_test_pose(Vector3::new(0.0, 0.0, 0.0)));
        state.add_camera(create_test_pose(Vector3::new(1.0, 0.0, 0.0)));

        // Add 3 landmarks
        state.add_landmark(
            Point3::new(0.0, 0.0, 5.0),
            vec![(0, Point2::new(320.0, 240.0))],
        );
        state.add_landmark(
            Point3::new(1.0, 0.0, 5.0),
            vec![(0, Point2::new(330.0, 240.0))],
        );
        state.add_landmark(
            Point3::new(2.0, 0.0, 5.0),
            vec![
                (0, Point2::new(340.0, 240.0)),
                (1, Point2::new(320.0, 240.0)),
            ],
        );

        let (n_res, n_params) = state.dimensions();
        // n_res: 2 observations per landmark = 2*1 + 2*1 + 2*2 = 8
        // n_params: 6*2 cameras + 3*3 landmarks = 12 + 9 = 21
        assert_eq!(n_res, 8);
        assert_eq!(n_params, 21);
    }

    #[test]
    fn test_multiple_cameras_single_landmark() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // Add 3 cameras
        state.add_camera(create_test_pose(Vector3::new(0.0, 0.0, 0.0)));
        state.add_camera(create_test_pose(Vector3::new(1.0, 0.0, 0.0)));
        state.add_camera(create_test_pose(Vector3::new(0.0, 1.0, 0.0)));

        // Single landmark observed by all cameras
        let observations = vec![
            (0, Point2::new(320.0, 240.0)),
            (1, Point2::new(310.0, 240.0)),
            (2, Point2::new(320.0, 250.0)),
        ];
        state.add_landmark(Point3::new(0.0, 0.0, 5.0), observations);

        assert_eq!(state.landmarks[0].observations.len(), 3);

        // All landmarks should have their 3 observations
        let residuals = state.residuals();
        assert_eq!(residuals.len(), 6); // 3 observations * 2 residuals each
    }

    #[test]
    fn test_bundle_adjust_reduces_error() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // Camera 0: at origin, looking down +Z
        let cam0 = create_test_pose(Vector3::new(0.0, 0.0, 0.0));
        state.add_camera(cam0);

        // Camera 1: translated 0.5 to the right
        let cam1 = create_test_pose(Vector3::new(0.5, 0.0, 0.0));
        state.add_camera(cam1);

        // 4 3D points in front of both cameras (at z=5)
        let points_3d = [
            Point3::new(-0.5, -0.5, 5.0),
            Point3::new(0.5, -0.5, 5.0),
            Point3::new(-0.5, 0.5, 5.0),
            Point3::new(0.5, 0.5, 5.0),
        ];

        // Compute ground-truth observations by projecting with the true poses
        for &pt in &points_3d {
            let mut observations = Vec::new();
            for (cam_idx, cam) in state.cameras.iter().enumerate() {
                let pt_cam = cam.rotation * pt + cam.translation;
                let proj = intrinsics.project(&pt_cam);
                observations.push((cam_idx, proj));
            }
            // Add landmarks with noisy positions (perturbed by 0.3 in each axis)
            let noisy_pos = Point3::new(pt.x + 0.3, pt.y - 0.2, pt.z + 0.1);
            state.add_landmark(noisy_pos, observations);
        }

        let initial_error = state.total_reprojection_error();
        assert!(
            initial_error > 0.0,
            "Initial error should be non-zero due to noisy landmarks"
        );

        let config = BundleAdjustmentConfig {
            max_iterations: 50,
            convergence_threshold: 1e-8,
            lambda: 0.001,
            use_sparsity: false,
            robust_kernel: false,
        };
        bundle_adjust(&mut state, &config);

        let final_error = state.total_reprojection_error();
        assert!(
            final_error < initial_error,
            "Bundle adjustment should reduce reprojection error: initial={}, final={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_sparse_jacobian_row_alignment_with_invalid_landmark() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);
        state.add_camera(create_test_pose(Vector3::zeros()));

        // Invalid landmark with one in-range and one out-of-range observation.
        // residuals() emits rows only for the in-range camera (2 rows), so the
        // sparse Jacobian cursor must not jump for the out-of-range one.
        let mut lm = Landmark::new(Point3::new(0.0, 0.0, 5.0));
        lm.add_observation(0, Point2::new(320.0, 240.0));
        lm.add_observation(7, Point2::new(320.0, 240.0)); // camera 7 does not exist
        lm.is_valid = false;
        state.landmarks.push(lm);

        // A valid landmark whose Jacobian rows follow the invalid one.
        state.add_landmark(
            Point3::new(1.0, 0.0, 5.0),
            vec![(0, Point2::new(330.0, 240.0))],
        );

        // 2 zero rows (invalid landmark, in-range obs) + 2 rows (valid) = 4.
        assert_eq!(state.residuals().len(), 4);

        // Before the fix the cursor advanced by 2 * observations, placing the
        // valid landmark's triplets at rows 4..6 and panicking in from_triplets.
        let j = state.numerical_jacobian_sparse();
        assert_eq!(j.rows, 4);
        // The valid landmark's rows must still carry non-zero derivatives.
        assert!(j.values.iter().any(|v| v.abs() > 0.0));
    }

    /// The analytic Jacobian must agree with the finite-difference construction
    /// it replaced. The two use the same parameterisation (`to_parameters`) and
    /// the same residual, so every stored entry must match to the accuracy of a
    /// central-free forward difference at eps = 1e-6.
    #[test]
    fn test_analytic_jacobian_matches_finite_difference() {
        use nalgebra::UnitQuaternion;

        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        // A few cameras with non-trivial rotations and translations, so no
        // derivative degenerates to zero or a pure identity.
        let cams: Vec<(Vector3<f64>, Vector3<f64>)> = vec![
            (Vector3::zeros(), Vector3::zeros()),
            (
                Vector3::new(0.08, -0.15, 0.05),
                Vector3::new(0.4, -0.2, 0.1),
            ),
            (
                Vector3::new(-0.22, 0.11, 0.30),
                Vector3::new(-0.5, 0.35, -0.15),
            ),
        ];
        for (axis, trans) in &cams {
            state.add_camera(Pose {
                rotation: UnitQuaternion::new(*axis),
                translation: *trans,
            });
        }

        // Landmarks in front of every camera, each seen by all three.
        for i in 0..12 {
            let p = Point3::new(
                -1.5 + 0.31 * i as f64,
                -0.8 + 0.19 * i as f64,
                4.0 + 0.13 * i as f64,
            );
            let obs: Vec<(usize, Point2<f64>)> = cams
                .iter()
                .enumerate()
                .map(|(ci, (axis, trans))| {
                    let rot: nalgebra::Rotation3<f64> = nalgebra::Rotation3::new(*axis);
                    let pc = rot * p + *trans;
                    (ci, intrinsics.project(&pc))
                })
                .collect();
            state.add_landmark(p, obs);
        }

        let analytic = state.numerical_jacobian_sparse();
        let fd = state.numerical_jacobian_fd();

        assert_eq!(analytic.rows, fd.rows, "row count");
        assert_eq!(analytic.cols, fd.cols, "column count");
        // CSR: same stored entries, and the same per-row column pattern.
        assert_eq!(analytic.values.len(), fd.values.len(), "stored entries");
        assert_eq!(analytic.row_ptr, fd.row_ptr, "row pointer");

        let mut max_rel = 0.0f64;
        let mut checked = 0usize;
        for row in 0..analytic.rows {
            let start = analytic.row_ptr[row] as usize;
            let end = analytic.row_ptr[row + 1] as usize;
            assert_eq!(
                &analytic.col_indices[start..end],
                &fd.col_indices[start..end],
                "column pattern of row {row}"
            );
            for i in start..end {
                let a = analytic.values[i];
                let f = fd.values[i];
                let scale = a.abs().max(f.abs()).max(1.0);
                max_rel = max_rel.max((a - f).abs() / scale);
                checked += 1;
            }
        }
        assert!(checked > 0, "the comparison must actually inspect entries");
        assert!(
            max_rel < 1e-3,
            "analytic Jacobian differs from finite differences (max relative \
             difference {max_rel} over {checked} entries); a sign or transpose \
             error is the usual cause"
        );
    }

    #[test]
    fn test_invalid_landmark_handling() {
        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);

        let pose = create_test_pose(Vector3::zeros());
        state.add_camera(pose);

        state.add_landmark(
            Point3::new(0.0, 0.0, 5.0),
            vec![(0, Point2::new(320.0, 240.0))],
        );
        state.add_landmark(
            Point3::new(1.0, 0.0, 5.0),
            vec![(0, Point2::new(330.0, 240.0))],
        );

        // Mark first landmark as invalid
        state.landmarks[0].is_valid = false;

        let residuals = state.residuals();
        // Should only have residuals for valid landmarks
        assert_eq!(residuals.len(), 4); // Only 2nd landmark: 2 residuals
    }

    /// The sparse normal-equation system must agree with the dense one, since
    /// bundle adjustment now solves the sparse form and every other code path
    /// (including the tests) still reasons about the dense one.
    #[test]
    fn test_sparse_normal_equations_match_dense() {
        use nalgebra::UnitQuaternion;

        let intrinsics = create_test_intrinsics();
        let mut state = SfMState::new(intrinsics);
        let cams = [
            (Vector3::zeros(), Vector3::zeros()),
            (Vector3::new(0.1, -0.2, 0.05), Vector3::new(0.3, -0.1, 0.2)),
        ];
        for (axis, trans) in &cams {
            state.add_camera(Pose {
                rotation: UnitQuaternion::new(*axis),
                translation: *trans,
            });
        }
        for i in 0..8 {
            let p = Point3::new(
                -1.0 + 0.4 * i as f64,
                -0.5 + 0.2 * i as f64,
                4.0 + 0.3 * i as f64,
            );
            let obs: Vec<(usize, Point2<f64>)> = cams
                .iter()
                .enumerate()
                .map(|(ci, (axis, trans))| {
                    let rot: Rotation3<f64> = Rotation3::new(*axis);
                    (ci, intrinsics.project(&(rot * p + *trans)))
                })
                .collect();
            state.add_landmark(p, obs);
        }

        let j_sparse = state.numerical_jacobian_sparse();
        let j_dense = state.numerical_jacobian();
        let r = state.residuals();

        let (lhs_sparse, jtr_sparse) = j_sparse.normal_equations_sparse(&r);
        let (lhs_dense, jtr_dense) = j_sparse.normal_equations(&r);

        assert_eq!(jtr_sparse.len(), jtr_dense.len());
        for i in 0..jtr_sparse.len() {
            assert!(
                (jtr_sparse[i] - jtr_dense[i]).abs() < 1e-9,
                "J^T r differs at {i}: {} vs {}",
                jtr_sparse[i],
                jtr_dense[i]
            );
        }

        let dense_lhs = j_dense.transpose() * &j_dense;
        let sparse_dense = lhs_sparse.to_dense();
        for r0 in 0..dense_lhs.nrows() {
            for c0 in 0..dense_lhs.ncols() {
                let a = dense_lhs[(r0, c0)];
                let b = sparse_dense[(r0, c0)];
                if a.abs() < 1e-12 && b.abs() < 1e-12 {
                    continue;
                }
                assert!(
                    (a - b).abs() < 1e-9,
                    "J^T J differs at ({r0},{c0}): {a} vs {b}"
                );
            }
        }
    }
}
