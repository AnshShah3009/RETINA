//! Nistér's 5-point solver must return the true essential matrix.
//!
//! The only previous checks were "five points in, some finite matrices out".
//! That passed while the solver never returned the true `E`: the returned
//! matrices lay in the 4-dimensional null space of the epipolar constraints
//! (so `x2^T E x1 = 0` held for the five points) but violated `det E = 0` and
//! the trace constraint `2 E E^T E - tr(E E^T) E = 0`, i.e. they were not
//! essential matrices at all. This test draws random exact two-view scenes and
//! requires the true `E` (up to scale and sign) to be among the solutions, and
//! every returned solution to be a genuine essential matrix.

use cv_calib3d::EssentialSolver;
use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn unit(&mut self) -> Unit<Vector3<f64>> {
        loop {
            let v = Vector3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let n = v.norm();
            if n > 0.1 && n <= 1.0 {
                return Unit::new_normalize(v);
            }
        }
    }
}

fn skew(t: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
}

/// Unit Frobenius norm with a deterministic sign (largest-magnitude entry
/// positive), so two matrices equal up to scale and sign compare directly.
fn canonical(e: &Matrix3<f64>) -> Matrix3<f64> {
    let n = e / e.norm();
    let pivot = n
        .iter()
        .copied()
        .fold(0.0f64, |acc, v| if v.abs() > acc.abs() { v } else { acc });
    if pivot < 0.0 {
        -n
    } else {
        n
    }
}

#[test]
fn five_point_recovers_the_true_essential_matrix() {
    let mut rng = Lcg(0x5eed_e55e);
    let trials = 200;
    let mut found = 0;
    let mut worst_det = 0.0f64;
    let mut worst_trace = 0.0f64;
    let mut worst_epipolar = 0.0f64;

    for _ in 0..trials {
        let axis = rng.unit();
        let r = Rotation3::from_axis_angle(&axis, rng.range(0.0, std::f64::consts::FRAC_PI_3))
            .into_inner();
        let t = rng.unit().into_inner();
        let e_true = skew(&t) * r;

        let mut pts1 = [[0.0f64; 2]; 5];
        let mut pts2 = [[0.0f64; 2]; 5];
        for i in 0..5 {
            let p1 = Vector3::new(
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(6.0, 10.0),
            );
            let p2 = r * p1 + t;
            pts1[i] = [p1.x / p1.z, p1.y / p1.z];
            pts2[i] = [p2.x / p2.z, p2.y / p2.z];
        }

        // An exact, generic scene always has the true E as a real solution,
        // so "no real solutions" is a miss like any other.
        let solutions = EssentialSolver::estimate_5point(&pts1, &pts2).unwrap_or_default();
        assert!(solutions.len() <= 10, "at most 10 real solutions exist");

        let target = canonical(&e_true);
        let mut hit = false;
        for e in &solutions {
            let e = e / e.norm();
            worst_det = worst_det.max(e.determinant().abs());
            let eet = e * e.transpose();
            let trace = 2.0 * eet * e - eet.trace() * e;
            worst_trace = worst_trace.max(trace.norm());
            for i in 0..5 {
                let x1 = Vector3::new(pts1[i][0], pts1[i][1], 1.0);
                let x2 = Vector3::new(pts2[i][0], pts2[i][1], 1.0);
                worst_epipolar = worst_epipolar.max((x2.transpose() * e * x1)[(0, 0)].abs());
            }
            if (canonical(&e) - target).norm() < 1e-6 {
                hit = true;
            }
        }
        if hit {
            found += 1;
        }
    }

    eprintln!(
        "true E recovered in {found}/{trials} scenes; worst |det E| = {worst_det:.3e}, \
         worst trace residual = {worst_trace:.3e}, worst epipolar residual = {worst_epipolar:.3e}"
    );
    assert!(
        found >= trials - 1,
        "true E recovered in only {found}/{trials} scenes"
    );
    // Every returned matrix must be essential. The broken solver returned
    // residuals of 0.1-0.9; correct output sits at ~1e-8 with a thin tail
    // (one spurious, ill-conditioned root of ~1000 reaches ~2e-6), so 1e-5
    // separates the two by four orders of magnitude.
    assert!(worst_det < 1e-5, "returned E with |det| = {worst_det:e}");
    assert!(
        worst_trace < 1e-5,
        "returned E violates the trace constraint by {worst_trace:e}"
    );
    assert!(
        worst_epipolar < 1e-8,
        "returned E violates x2^T E x1 = 0 by {worst_epipolar:e}"
    );
}
