//! Kalman Filter implementations
//!
//! Two variants:
//! - [`DynamicKalmanFilter`]: Runtime-sized matrices, OpenCV-compatible API
//! - [`KalmanFilter`]: Compile-time-sized, const-generic, with EKF support

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use std::fmt::Debug;

// ============================================================
// Dynamic Kalman Filter (runtime-sized, OpenCV-style API)
// ============================================================

pub struct DynamicKalmanFilter {
    pub state_pre: DVector<f64>,
    pub state_post: DVector<f64>,
    pub trans_matrix: DMatrix<f64>,
    pub control_matrix: DMatrix<f64>,
    pub meas_matrix: DMatrix<f64>,
    pub process_noise_cov: DMatrix<f64>,
    pub meas_noise_cov: DMatrix<f64>,
    pub error_cov_pre: DMatrix<f64>,
    pub gain: DMatrix<f64>,
    pub error_cov_post: DMatrix<f64>,
}

impl DynamicKalmanFilter {
    pub fn new(dynam_params: usize, meas_params: usize, control_params: usize) -> Self {
        Self {
            state_pre: DVector::zeros(dynam_params),
            state_post: DVector::zeros(dynam_params),
            trans_matrix: DMatrix::identity(dynam_params, dynam_params),
            control_matrix: DMatrix::zeros(dynam_params, control_params),
            meas_matrix: DMatrix::zeros(meas_params, dynam_params),
            process_noise_cov: DMatrix::identity(dynam_params, dynam_params),
            meas_noise_cov: DMatrix::identity(meas_params, meas_params),
            error_cov_pre: DMatrix::zeros(dynam_params, dynam_params),
            gain: DMatrix::zeros(dynam_params, meas_params),
            error_cov_post: DMatrix::identity(dynam_params, dynam_params),
        }
    }

    pub fn predict(&mut self, control: Option<&DVector<f64>>) -> &DVector<f64> {
        self.state_pre = &self.trans_matrix * &self.state_post;
        if let Some(u) = control {
            self.state_pre += &self.control_matrix * u;
        }
        self.error_cov_pre =
            &self.trans_matrix * &self.error_cov_post * self.trans_matrix.transpose()
                + &self.process_noise_cov;
        &self.state_pre
    }

    pub fn correct(&mut self, measurement: &DVector<f64>) -> &DVector<f64> {
        let s = &self.meas_matrix * &self.error_cov_pre * self.meas_matrix.transpose()
            + &self.meas_noise_cov;
        // Fixed: capture dimensions before moving s
        let s_dims = (s.nrows(), s.ncols());
        let s_inv = s
            .pseudo_inverse(1e-6)
            .unwrap_or_else(|_| DMatrix::identity(s_dims.0, s_dims.1));
        self.gain = &self.error_cov_pre * self.meas_matrix.transpose() * s_inv;
        let innovation = measurement - &self.meas_matrix * &self.state_pre;
        self.state_post = &self.state_pre + &self.gain * innovation;
        let i = DMatrix::identity(self.state_post.len(), self.state_post.len());
        self.error_cov_post = (&i - &self.gain * &self.meas_matrix) * &self.error_cov_pre;
        &self.state_post
    }
}

// ============================================================
// Const-generic Kalman Filter + EKF (from slam/src/kalman.rs)
// ============================================================

#[derive(Debug, Clone)]
pub struct KalmanFilterState<const N: usize> {
    pub x: SVector<f64, N>,
    pub p: SMatrix<f64, N, N>,
}

impl<const N: usize> KalmanFilterState<N> {
    pub fn new(x: SVector<f64, N>, p: SMatrix<f64, N, N>) -> Self {
        Self { x, p }
    }
    pub fn zero() -> Self {
        Self {
            x: SVector::zeros(),
            p: SMatrix::identity() * 1000.0,
        }
    }
    pub fn std_dev(&self) -> SVector<f64, N> {
        self.p.diagonal().map(|v| v.sqrt())
    }
}

#[derive(Debug, Clone)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    pub f: SMatrix<f64, N, N>,
    pub b: SMatrix<f64, N, N>,
    pub h: SMatrix<f64, M, N>,
    pub q: SMatrix<f64, N, N>,
    pub r: SMatrix<f64, M, M>,
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    pub fn new(
        f: SMatrix<f64, N, N>,
        b: SMatrix<f64, N, N>,
        h: SMatrix<f64, M, N>,
        q: SMatrix<f64, N, N>,
        r: SMatrix<f64, M, M>,
    ) -> Self {
        Self { f, b, h, q, r }
    }

    pub fn predict(&self, state: &mut KalmanFilterState<N>, u: &SVector<f64, N>) {
        state.x = self.f * state.x + self.b * u;
        state.p = self.f * state.p * self.f.transpose() + self.q;
    }

    pub fn update(&self, state: &mut KalmanFilterState<N>, z: &SVector<f64, M>) {
        let y = z - self.h * state.x;
        let s = self.h * state.p * self.h.transpose() + self.r;
        let k = state.p * self.h.transpose() * s.try_inverse().unwrap_or(SMatrix::identity());
        state.x += k * y;
        let i = SMatrix::<f64, N, N>::identity();
        let i_kh = i - k * self.h;
        state.p = i_kh * state.p * i_kh.transpose() + k * self.r * k.transpose();
    }

    pub fn step(&self, state: &mut KalmanFilterState<N>, u: &SVector<f64, N>, z: &SVector<f64, M>) {
        self.predict(state, u);
        self.update(state, z);
    }
}

pub struct ExtendedKalmanFilter<const N: usize, const M: usize> {
    pub q: SMatrix<f64, N, N>,
    pub r: SMatrix<f64, M, M>,
}

impl<const N: usize, const M: usize> ExtendedKalmanFilter<N, M> {
    pub fn new(q: SMatrix<f64, N, N>, r: SMatrix<f64, M, M>) -> Self {
        Self { q, r }
    }

    pub fn predict<F: Fn(&SVector<f64, N>) -> SVector<f64, N>>(
        &self,
        state: &mut KalmanFilterState<N>,
        f: F,
        jacobian_f: &SMatrix<f64, N, N>,
    ) {
        state.x = f(&state.x);
        state.p = jacobian_f * state.p * jacobian_f.transpose() + self.q;
    }

    pub fn update<H: Fn(&SVector<f64, N>) -> SVector<f64, M>>(
        &self,
        state: &mut KalmanFilterState<N>,
        h: H,
        jacobian_h: &SMatrix<f64, M, N>,
        z: &SVector<f64, M>,
    ) {
        let y = z - h(&state.x);
        let s = jacobian_h * state.p * jacobian_h.transpose() + self.r;
        let k = state.p * jacobian_h.transpose() * s.try_inverse().unwrap_or(SMatrix::identity());
        state.x += k * y;
        let i = SMatrix::<f64, N, N>::identity();
        let i_kh = i - k * jacobian_h;
        state.p = i_kh * state.p * i_kh.transpose() + k * self.r * k.transpose();
    }
}

pub mod utils {
    use super::*;

    pub fn constant_velocity_2d(
        dt: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> KalmanFilter<4, 2> {
        let f = SMatrix::<f64, 4, 4>::new(
            1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let b = SMatrix::<f64, 4, 4>::identity();
        let h = SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let q = SMatrix::<f64, 4, 4>::identity() * process_noise;
        let r = SMatrix::<f64, 2, 2>::identity() * measurement_noise;
        KalmanFilter::new(f, b, h, q, r)
    }

    pub fn constant_acceleration_2d(
        dt: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> KalmanFilter<6, 2> {
        let dt2 = dt * dt / 2.0;
        let f = SMatrix::<f64, 6, 6>::new(
            1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0,
        );
        let b = SMatrix::<f64, 6, 6>::identity();
        let h =
            SMatrix::<f64, 2, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let q = SMatrix::<f64, 6, 6>::identity() * process_noise;
        let r = SMatrix::<f64, 2, 2>::identity() * measurement_noise;
        KalmanFilter::new(f, b, h, q, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_kalman_predict() {
        let mut kf = DynamicKalmanFilter::new(2, 2, 0);
        kf.trans_matrix = DMatrix::identity(2, 2);
        kf.state_post = DVector::from_vec(vec![1.0, 2.0]);
        kf.predict(None);
        assert!((kf.state_pre[0] - 1.0).abs() < 1e-9);
        assert!((kf.state_pre[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_dynamic_kalman_correct() {
        let mut kf = DynamicKalmanFilter::new(2, 2, 0);
        kf.meas_matrix = DMatrix::identity(2, 2);
        kf.state_pre = DVector::from_vec(vec![0.0, 0.0]);
        kf.error_cov_pre = DMatrix::identity(2, 2);
        let z = DVector::from_vec(vec![1.0, 1.0]);
        kf.correct(&z);
        // State should move towards measurement
        assert!(kf.state_post[0] > 0.0);
        assert!(kf.state_post[1] > 0.0);
    }

    #[test]
    fn test_kalman_filter_2d() {
        let kf = utils::constant_velocity_2d(0.1, 0.01, 0.1);
        let mut state = KalmanFilterState::zero();
        let u = SVector::<f64, 4>::zeros();
        for z in [
            SVector::<f64, 2>::new(1.0, 1.0),
            SVector::<f64, 2>::new(1.1, 1.1),
            SVector::<f64, 2>::new(1.2, 1.2),
        ] {
            kf.step(&mut state, &u, &z);
        }
        assert!(state.x[0] > 0.5);
        assert!(state.x[1] > 0.5);
    }

    #[test]
    fn test_ekf_update() {
        let ekf = ExtendedKalmanFilter::<4, 2>::new(
            SMatrix::<f64, 4, 4>::identity() * 0.01,
            SMatrix::<f64, 2, 2>::identity() * 0.1,
        );
        let mut state = KalmanFilterState::new(
            SVector::<f64, 4>::new(0.0, 0.0, 1.0, 1.0),
            SMatrix::<f64, 4, 4>::identity(),
        );
        let h = |x: &SVector<f64, 4>| SVector::<f64, 2>::new(x[0], x[1]);
        let jacobian_h = SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let z = SVector::<f64, 2>::new(1.0, 1.0);
        ekf.update(&mut state, h, &jacobian_h, &z);
        assert!(state.x[0] > 0.0);
        assert!(state.x[1] > 0.0);
    }

    #[test]
    fn test_std_dev() {
        let state = KalmanFilterState::new(
            SVector::<f64, 2>::new(0.0, 0.0),
            SMatrix::<f64, 2, 2>::new(4.0, 0.0, 0.0, 9.0),
        );
        let std_dev = state.std_dev();
        assert!((std_dev[0] - 2.0).abs() < 1e-6);
        assert!((std_dev[1] - 3.0).abs() < 1e-6);
    }
}
