//! The 8-point essential solver must work at its own documented minimum.
//!
//! nalgebra's thin SVD returns a `v_t` of shape `min(m, n)`. For the `m x 9`
//! design matrix the 8-point algorithm builds, the null vector is row 8 — so with
//! exactly 8 correspondences it is *not among the rows returned*, and
//! `v_t.row(v_t.nrows() - 1)` picked row 7: an arbitrary vector with a strictly
//! positive `‖A v‖`. `enforce_essential_constraints` then projected it onto the
//! rank-2 manifold, producing a well-formed, plausible, wrong `E`.
//!
//! Measured on noise-free correspondences from a known relative pose, worst
//! Sampson residual in pixels at f = 800 — the true `E` gives ~4.5e-14 px:
//!
//! | points | recovered | correct? |
//! | ---: | ---: | :--- |
//! | 8 | **21.71 px** | no |
//! | 9 | 4.01e-12 px | yes |
//! | 12 | 1.50e-12 px | yes |
//! | 16 | 3.35e-13 px | yes |
//!
//! Only the minimal sample was broken, which is why the general case looks fine.
//!
//! This reaches past the direct call: `EssentialEstimator::min_sample_size()` is
//! 8 and `find_essential_mat_ransac` drives it with `Ransac::run`, which samples
//! exactly that many — so **every** minimal sample in RANSAC came from a solver
//! returning a wrong `E`.

use cv_calib3d::{essential_from_extrinsics, find_essential_mat};
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Matrix3, Point2, Vector3};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(800.0, 795.0, 320.0, 240.0, 640, 480)
}

/// The relative pose the correspondences are generated from.
fn relative_pose() -> Pose {
    let r = nalgebra::Rotation3::from_euler_angles(0.05, -0.12, 0.03).into_inner();
    Pose::new(r, Vector3::new(0.4, 0.05, 0.02))
}

/// `n` exact correspondences from points spread through a genuine 3-D volume.
///
/// A grid on a thin slab makes the 8-point algorithm ill-conditioned, which would
/// confound the row-selection question with a conditioning one.
fn correspondences(n: usize) -> (Vec<Point2<f64>>, Vec<Point2<f64>>) {
    let k = intrinsics();
    let p0 = Pose::new(Matrix3::identity(), Vector3::zeros());
    let p1 = relative_pose();

    let mut pts1 = Vec::with_capacity(n);
    let mut pts2 = Vec::with_capacity(n);
    let mut seed = 12345u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };

    while pts1.len() < n {
        let p = Vector3::new(next() * 4.0, next() * 4.0, 3.0 + next() * 2.0);
        let c1 = p0.rotation * p + p0.translation;
        let c2 = p1.rotation * p + p1.translation;
        if c1.z <= 0.1 || c2.z <= 0.1 {
            continue;
        }
        pts1.push(Point2::new(
            k.fx * c1.x / c1.z + k.cx,
            k.fy * c1.y / c1.z + k.cy,
        ));
        pts2.push(Point2::new(
            k.fx * c2.x / c2.z + k.cx,
            k.fy * c2.y / c2.z + k.cy,
        ));
    }
    (pts1, pts2)
}

/// Worst Sampson distance in pixels: how far the epipolar geometry is from
/// satisfied. 0 means the correspondences satisfy the constraint exactly.
fn max_sampson_px(e: &Matrix3<f64>, pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> f64 {
    let k = intrinsics();
    let k_inv = k.matrix().try_inverse().expect("valid intrinsics");
    pts1.iter()
        .zip(pts2.iter())
        .map(|(a, b)| {
            let x1 = (k_inv * Vector3::new(a.x, a.y, 1.0)).normalize();
            let x2 = (k_inv * Vector3::new(b.x, b.y, 1.0)).normalize();
            let ex1 = e * x1;
            let etx2 = e.transpose() * x2;
            let num = x2.dot(&ex1).powi(2);
            let den = ex1[0].powi(2) + ex1[1].powi(2) + etx2[0].powi(2) + etx2[1].powi(2);
            if den < 1e-24 {
                return f64::INFINITY;
            }
            (num / den).sqrt() * k.fx
        })
        .fold(0.0f64, f64::max)
}

/// The minimal sample. This is the case that was wrong by 21.7 px.
#[test]
fn the_minimal_eight_point_sample_recovers_the_essential_matrix() {
    let (pts1, pts2) = correspondences(8);
    assert_eq!(pts1.len(), 8, "the documented minimum is 8 correspondences");

    let e = find_essential_mat(&pts1, &pts2, &intrinsics())
        .expect("exactly 8 correspondences is the documented minimum and must solve");

    let residual = max_sampson_px(&e, &pts1, &pts2);
    assert!(
        residual < 1e-6,
        "an exact correspondence set must be recovered exactly, but the recovered \
         E leaves a worst Sampson residual of {residual:.3e} px; the true E gives \
         ~4.5e-14 px. Reading the wrong row of the thin SVD is the cause."
    );
}

/// Above the minimum, everything already worked - these are the controls that
/// make the minimal-sample test meaningful rather than merely strict.
#[test]
fn above_the_minimum_sample_sizes_were_and_remain_exact() {
    for n in [9usize, 12, 16] {
        let (pts1, pts2) = correspondences(n);
        let e = find_essential_mat(&pts1, &pts2, &intrinsics())
            .unwrap_or_else(|err| panic!("n={n} should solve: {err}"));
        let residual = max_sampson_px(&e, &pts1, &pts2);
        assert!(
            residual < 1e-6,
            "n={n}: worst Sampson residual {residual:.3e} px"
        );
    }
}

/// Sanity check on the measurement itself: the ground-truth `E` scores ~0 on the
/// same correspondences. Without this, a bug in the residual would make every
/// other assertion here pass vacuously.
#[test]
fn the_true_essential_matrix_scores_zero_on_the_same_points() {
    let (pts1, pts2) = correspondences(8);
    let true_e = essential_from_extrinsics(&relative_pose());
    let residual = max_sampson_px(&true_e, &pts1, &pts2);
    assert!(
        residual < 1e-9,
        "the measurement is wrong: the true E leaves {residual:.3e} px, so every \
         residual in this file would be meaningless"
    );
}
