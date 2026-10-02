//! A singular innovation covariance must not make the filter ignore its
//! measurement.
//!
//! `KalmanFilter::update` and `ExtendedKalmanFilter::update` computed the gain as
//! `P Hᵀ · try_inverse(S).unwrap_or(zeros)`. When `S = H P Hᵀ + R` is singular,
//! `try_inverse` returns `None` and the fallback made `K` the **zero matrix**.
//!
//! `K = 0` is not a neutral value — it means *ignore the measurement*:
//!
//! ```text
//! state.x += 0 * y        -> the state does not move
//! i_kh = I - 0 = I        -> the covariance does not shrink either
//! ```
//!
//! So the filter went on reporting a maximally confident state that was perfectly
//! consistent with its own prediction, having silently stopped listening. Nothing
//! a caller can observe distinguishes that from a filter that is working.
//!
//! `S` is singular whenever `R` is and `H P Hᵀ` is rank-deficient, which a caller
//! reaches through the module's own constructor by asking for perfect trust in its
//! measurements: `utils::constant_velocity_2d(dt, q, 0.0)`.
//!
//! The correct limit is the pseudo-inverse, which is well defined for a singular
//! `S`, agrees with the true inverse to numerical precision when `S` is
//! invertible, and is what `DynamicKalmanFilter::correct` in the same module
//! already used.

use cv_video::kalman::{utils, ExtendedKalmanFilter, KalmanFilter, KalmanFilterState};
use nalgebra::{SMatrix, SVector};

/// `H` observing position `(x, y)` of a `(x, y, vx, vy)` state.
fn position_observer() -> SMatrix<f64, 2, 4> {
    SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
}

fn diagonal_state(diag: [f64; 4]) -> KalmanFilterState<4> {
    let mut p = SMatrix::<f64, 4, 4>::zeros();
    for i in 0..4 {
        p[(i, i)] = diag[i];
    }
    KalmanFilterState::new(SVector::<f64, 4>::zeros(), p)
}

/// The decisive case. `x` is *perfectly* observable: `P` has unit variance in `x`
/// and the measurement is noiseless, so the measurement should be taken exactly —
/// the correct gain on `x` is 1.0, not 0.
///
/// Measured before the fix: the state moved `0.000e0` and the covariance changed
/// `0.000e0`. After: the state moves exactly `10.0` to `x = 10`.
#[test]
fn a_singular_innovation_covariance_does_not_discard_the_measurement() {
    let h = position_observer();
    let f = SMatrix::<f64, 4, 4>::identity();
    let b = SMatrix::<f64, 4, 4>::zeros();
    // R = 0, so S = H P H^T is rank 1 and therefore singular.
    let kf = KalmanFilter::<4, 2>::new(f, b, h, SMatrix::zeros(), SMatrix::zeros());

    let mut state = diagonal_state([1.0, 0.0, 0.0, 0.0]);
    let before = state.x;

    kf.update(&mut state, &SVector::<f64, 2>::new(10.0, -10.0));

    let moved = (state.x - before).norm();
    assert!(
        moved > 1e-9,
        "a measurement of x, which the filter has unit variance in and zero \
         measurement noise, must move the state; it moved {moved:.3e}. A gain of \
         zero means the measurement was silently discarded."
    );
    // Gain 1.0 on x means the estimate becomes the measurement exactly.
    assert!(
        (state.x[0] - 10.0).abs() < 1e-9,
        "x is perfectly observable with a noiseless measurement, so the gain is \
         1.0 and the estimate should equal the measurement; got {}",
        state.x[0]
    );
    // y is unobservable here (zero variance), so it must stay put.
    assert!(
        state.x[1].abs() < 1e-9,
        "y has zero variance and zero measurement noise, so nothing determines \
         it; it moved to {}",
        state.x[1]
    );
}

/// The same defect in the extended filter.
#[test]
fn the_extended_filter_does_not_discard_the_measurement_either() {
    let h = position_observer();
    let ekf = ExtendedKalmanFilter::<4, 2>::new(SMatrix::zeros(), SMatrix::zeros());
    let mut state = diagonal_state([1.0, 0.0, 0.0, 0.0]);

    ekf.update(
        &mut state,
        |x| SVector::<f64, 2>::new(x[0], x[1]),
        &h,
        &SVector::<f64, 2>::new(10.0, -10.0),
    );

    assert!(
        (state.x[0] - 10.0).abs() < 1e-9,
        "the EKF carries the same fallback and must take the measurement; x = {}",
        state.x[0]
    );
}

/// CONTROL: an ordinary positive-definite `R` still behaves as before, so the fix
/// is not simply "always take the measurement".
#[test]
fn a_healthy_filter_still_fuses_rather_than_obeying() {
    let h = position_observer();
    let f = SMatrix::<f64, 4, 4>::identity();
    let b = SMatrix::<f64, 4, 4>::zeros();
    let r = SMatrix::<f64, 2, 2>::identity() * 1e-3;
    let kf = KalmanFilter::<4, 2>::new(f, b, h, SMatrix::zeros(), r);

    let mut state = diagonal_state([1.0, 1.0, 1.0, 1.0]);
    kf.update(&mut state, &SVector::<f64, 2>::new(10.0, -10.0));

    // With measurement noise comparable to the state variance the estimate is a
    // weighted blend, strictly between the prediction (0) and the measurement.
    assert!(
        state.x[0] > 0.0 && state.x[0] < 10.0,
        "a finite R must blend rather than obey; x = {}",
        state.x[0]
    );
}

/// CONTROL: `K = 0` is genuinely correct when the state has no uncertainty at all
/// in any direction — the gain is `P Hᵀ S⁺` and `P = 0` makes it zero. Without
/// this, the test above could be read as "always expect the state to move".
#[test]
fn zero_prior_covariance_legitimately_yields_no_update() {
    let h = position_observer();
    let kf = KalmanFilter::<4, 2>::new(
        SMatrix::identity(),
        SMatrix::zeros(),
        h,
        SMatrix::zeros(),
        SMatrix::zeros(),
    );

    let mut state = diagonal_state([0.0, 0.0, 0.0, 0.0]);
    let before = state.x;
    kf.update(&mut state, &SVector::<f64, 2>::new(10.0, -10.0));

    assert_eq!(
        state.x, before,
        "with P = 0 there is no uncertainty to reduce, so the gain is legitimately \
         zero and the state must not move"
    );
}

/// The fix must agree with the true inverse wherever that inverse exists — a
/// pseudo-inverse is not allowed to change well-conditioned behaviour.
#[test]
fn the_pseudo_inverse_agrees_with_the_true_inverse_when_it_exists() {
    let h = position_observer();
    let f = SMatrix::<f64, 4, 4>::identity();
    let b = SMatrix::<f64, 4, 4>::zeros();
    let r = SMatrix::<f64, 2, 2>::identity() * 0.5;
    let kf = KalmanFilter::<4, 2>::new(f, b, h, SMatrix::zeros(), r);

    let mut state = diagonal_state([2.0, 2.0, 2.0, 2.0]);
    let p_before = state.p;
    kf.update(&mut state, &SVector::<f64, 2>::new(3.0, -3.0));

    // Analytic one-step result for this system: with H observing both positions,
    // S = H P H^T + R = diag(2.5, 2.5) and K = P H^T S^-1 = diag(0.8, 0.8) in the
    // observed block, giving x = 0 + 0.8 * 3 = 2.4.
    assert!(
        (state.x[0] - 2.4).abs() < 1e-9,
        "the pseudo-inverse must reproduce the analytic one-step estimate 2.4; \
         got {}",
        state.x[0]
    );
    // Covariance must shrink, and be symmetric.
    assert!(
        state.p[(0, 0)] < p_before[(0, 0)],
        "an informative measurement must reduce the variance in x, {} -> {}",
        p_before[(0, 0)],
        state.p[(0, 0)]
    );
    let asymmetry = (state.p - state.p.transpose()).norm();
    assert!(
        asymmetry < 1e-12,
        "the covariance must stay symmetric; ||P - P^T|| = {asymmetry:.3e}"
    );
}

/// The defect through the module's own constructor, which is how a caller would
/// actually reach it: `measurement_noise = 0.0` reads as "I trust my sensor
/// perfectly", and it is the `R = 0` that makes `S` singular here.
#[test]
fn perfect_trust_in_the_sensor_does_not_freeze_the_filter() {
    let kf = utils::constant_velocity_2d(0.1, 1e-6, 0.0);
    // A rank-deficient P, as arises when only position has been observed.
    let mut state = diagonal_state([1.0, 0.0, 0.0, 0.0]);

    let before = state.x;
    kf.update(&mut state, &SVector::<f64, 2>::new(5.0, 0.0));

    assert!(
        (state.x - before).norm() > 1e-9,
        "constant_velocity_2d(dt, q, 0.0) must still fuse measurements; the state \
         moved {:.3e}, so the filter is frozen",
        (state.x - before).norm()
    );
}
