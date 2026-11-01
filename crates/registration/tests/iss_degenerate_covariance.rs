//! `compute_iss_features` must terminate on small clouds whose ISS covariance
//! is identically zero.
//!
//! The descriptor stage solved `cov.eigenvalues()` - nalgebra's *general* real
//! eigensolver, a Schur iteration - on a covariance that is exactly the zero
//! matrix. That solver does not converge on a zero matrix, so the function spun
//! forever: two points ten units apart (no neighbour within the saliency
//! radius) never returned. Coincident points reach the same state with a
//! non-empty neighbourhood, because every difference from the centroid is zero.
//!
//! A hang is not a test failure, so each case runs on a worker thread and the
//! test fails on a timeout instead of wedging the whole suite.

use cv_core::PointCloud;
use cv_registration::registration::global::{compute_iss_features, ISSDetector, ISSFeature};
use nalgebra::Point3;
use std::sync::mpsc;
use std::time::Duration;

fn cloud(points: &[(f32, f32, f32)]) -> PointCloud {
    PointCloud {
        points: points.iter().map(|p| Point3::new(p.0, p.1, p.2)).collect(),
        colors: None,
        normals: None,
    }
}

/// Run with a wall-clock budget so a non-terminating call is a failure.
fn within_budget(
    cloud: PointCloud,
    detector: ISSDetector,
    budget: Duration,
) -> Option<Vec<ISSFeature>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let features = compute_iss_features(&cloud, detector);
        let _ = tx.send(features);
    });
    rx.recv_timeout(budget).ok()
}

const BUDGET: Duration = Duration::from_secs(5);

/// Two points ten units apart: neither has a neighbour within the 0.1 saliency
/// radius, so both keypoints have a zero descriptor covariance.
#[test]
fn isolated_points_do_not_hang() {
    let features = within_budget(
        cloud(&[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]),
        ISSDetector::default(),
        BUDGET,
    )
    .expect(
        "compute_iss_features never returned for two isolated points: the zero descriptor \
         covariance was handed to nalgebra's general eigensolver, which does not converge \
         on the zero matrix",
    );
    assert_eq!(
        features.len(),
        2,
        "both isolated points survive non-maximum suppression"
    );
    for f in &features {
        assert_eq!(
            f.covariance_eigenvalues,
            [0.0, 0.0, 0.0],
            "an isolated point's covariance is zero, so its eigenvalues are zero"
        );
    }
}

/// Coincident points are the same zero covariance with a *non-empty*
/// neighbourhood: every neighbour sits on the centroid.
#[test]
fn coincident_points_do_not_hang() {
    let features = within_budget(
        cloud(&[(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)]),
        ISSDetector::default(),
        BUDGET,
    )
    .expect(
        "compute_iss_features never returned for three coincident points: their covariance \
         is exactly zero, which the general eigensolver cannot decompose",
    );
    assert_eq!(
        features.len(),
        1,
        "non-maximum suppression keeps one of the coincident points"
    );
    assert_eq!(features[0].covariance_eigenvalues, [0.0, 0.0, 0.0]);
}

/// CONTROL: a normal cloud still produces features, so the guard is not a
/// blanket early return.
#[test]
fn a_normal_cloud_still_produces_features() {
    let mut pts = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            pts.push((i as f32 * 0.02, j as f32 * 0.02, 0.0));
        }
    }
    let features = within_budget(cloud(&pts), ISSDetector::default(), BUDGET)
        .expect("a 5x5 grid must be processed");
    assert!(
        !features.is_empty(),
        "a well-formed cloud must still yield keypoints"
    );
}
