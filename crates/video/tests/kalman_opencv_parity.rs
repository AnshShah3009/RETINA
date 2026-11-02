//! `DynamicKalmanFilter` is the OpenCV-style filter (`cv::KalmanFilter`), so a
//! sequence of `predict` / `correct` calls must leave it in the state OpenCV's
//! filter reaches.
//!
//! Reference values come from OpenCV 4.13 in double precision:
//!
//! ```python
//! k = cv2.KalmanFilter(2, 1, 0, cv2.CV_64F)
//! k.transitionMatrix = np.array([[1, 1], [0, 1]], np.float64)
//! k.measurementMatrix = np.array([[1, 0]], np.float64)
//! k.processNoiseCov = np.eye(2) * 0.01
//! k.measurementNoiseCov = np.array([[0.25]])
//! k.statePost = np.array([[0], [1]], np.float64)
//! k.errorCovPost = np.eye(2)
//! for _ in range(3): k.predict()
//! k.correct(np.array([[3.5]]))
//! k.predict()
//! ```

use cv_video::kalman::DynamicKalmanFilter;
use nalgebra::{DMatrix, DVector};

fn constant_velocity() -> DynamicKalmanFilter {
    let mut kf = DynamicKalmanFilter::new(2, 1, 0);
    kf.trans_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
    kf.meas_matrix = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
    kf
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "{what}[{i}] = {g}, OpenCV gives {w} (full: {got:?} vs {want:?})"
        );
    }
}

/// `cv::KalmanFilter::predict` ends with `statePre.copyTo(statePost);
/// errorCovPre.copyTo(errorCovPost);` so that consecutive predictions (frames
/// with no measurement, e.g. an occluded target) keep extrapolating.
///
/// Before the fix every `predict` restarted from the last *corrected* state,
/// so three predictions in a row all returned `[1, 1]` - the target stopped
/// moving the moment it was occluded - and the covariance never grew past one
/// step's worth of process noise.
#[test]
fn consecutive_predictions_keep_extrapolating_like_opencv() {
    let mut kf = constant_velocity();
    kf.process_noise_cov = DMatrix::identity(2, 2) * 0.01;
    kf.meas_noise_cov = DMatrix::from_element(1, 1, 0.25);
    kf.state_post = DVector::from_vec(vec![0.0, 1.0]);
    kf.error_cov_post = DMatrix::identity(2, 2);

    let expected_states = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];
    let expected_covs = [
        [2.01, 1.0, 1.0, 1.01],
        [5.029999999999999, 2.01, 2.01, 1.02],
        [10.079999999999998, 3.03, 3.03, 1.03],
    ];
    for (step, (state, cov)) in expected_states.iter().zip(&expected_covs).enumerate() {
        let pre = kf.predict(None).clone();
        assert_close(
            &format!("statePre after predict #{}", step + 1),
            pre.as_slice(),
            state,
        );
        let p: Vec<f64> = kf.error_cov_pre.transpose().iter().copied().collect();
        assert_close(&format!("errorCovPre after predict #{}", step + 1), &p, cov);
    }

    let post = kf.correct(&DVector::from_element(1, 3.5)).clone();
    assert_close(
        "statePost after correct",
        post.as_slice(),
        &[3.4878993223620522, 1.1466602129719266],
    );
    let p: Vec<f64> = kf.error_cov_post.transpose().iter().copied().collect();
    assert_close(
        "errorCovPost after correct",
        &p,
        &[
            0.243949661181027,
            0.0733301064859635,
            0.07333010648596305,
            0.14123910939012585,
        ],
    );

    let pre = kf.predict(None).clone();
    assert_close(
        "statePre after the post-correct predict",
        pre.as_slice(),
        &[4.634559535333979, 1.1466602129719266],
    );
}

/// `cv::KalmanFilter::init` sets `errorCovPost` to **zeros** (only
/// `processNoiseCov`, `measurementNoiseCov` and `transitionMatrix` start as
/// identity). Starting from an identity `errorCovPost` instead inflated the
/// first prediction's covariance by `F Fᵀ` and moved the first correction:
/// `[1.5, 0.5]` where OpenCV gives `[1, 0]`.
#[test]
fn default_error_covariance_matches_opencv() {
    let mut kf = constant_velocity();
    assert_close(
        "default errorCovPost",
        kf.error_cov_post.as_slice(),
        &[0.0; 4],
    );
    kf.predict(None);
    let p: Vec<f64> = kf.error_cov_pre.transpose().iter().copied().collect();
    assert_close("errorCovPre after first predict", &p, &[1.0, 0.0, 0.0, 1.0]);
    let post = kf.correct(&DVector::from_element(1, 2.0)).clone();
    assert_close(
        "statePost after first correct",
        post.as_slice(),
        &[1.0, 0.0],
    );
}
