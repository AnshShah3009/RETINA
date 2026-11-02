//! Global RANSAC registration must not gate FPFH feature matches with the
//! *metric* inlier distance. Feature distances are between 33-bin histograms
//! (each sub-histogram sums to ~100), the inlier distance is in the cloud's
//! units; with a 5 cm inlier distance the old gate dropped every match and
//! registration failed with "Insufficient correspondences".

use cv_core::PointCloud;
use cv_registration::{compute_fpfh_features, registration_ransac_based_on_feature_matching};
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

#[test]
fn small_metric_threshold_does_not_discard_feature_matches() {
    // An asymmetric surface (a ridged, tilted bowl) so FPFH is distinctive.
    let mut pts = Vec::new();
    for i in 0..30 {
        for j in 0..30 {
            let (x, y) = (i as f32 * 0.04, j as f32 * 0.04);
            let z = 0.3 * (x * 3.0).sin() * (y * 2.0).cos() + 0.2 * x * x + 0.1 * y;
            pts.push(Point3::new(x, y, z));
        }
    }
    let r = Rotation3::from_scaled_axis(Vector3::new(0.1, -0.2, 0.3));
    let t = Translation3::new(0.2, -0.1, 0.05);
    // 3 mm of noise on the target: real scans never give identical
    // descriptors, so true matches have feature distances well above 0.05.
    let mut seed: u32 = 5;
    let mut noise = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.006
    };
    let moved: Vec<Point3<f32>> = pts
        .iter()
        .map(|p| t * (r * p) + Vector3::new(noise(), noise(), noise()))
        .collect();
    let source = PointCloud {
        points: pts,
        colors: None,
        normals: None,
    };
    let target = PointCloud {
        points: moved,
        colors: None,
        normals: None,
    };
    let fs = compute_fpfh_features(&source, 0.15).unwrap();
    let ft = compute_fpfh_features(&target, 0.15).unwrap();

    let res =
        registration_ransac_based_on_feature_matching(&source, &target, &fs, &ft, 0.05, 3, 4000)
            .expect("registration must find correspondences");
    let expected = (t * r).to_homogeneous();
    let err = (res.transformation - expected).norm();
    // Coarse alignment (Open3D refines this with ICP). RANSAC is randomised:
    // 60 local runs gave median 0.055, p95 0.084, max 0.094, and one CI run
    // reached 0.1505 - over the old 0.15 bound. The metric-gated matcher this
    // guards against was off by 2.8, so 0.5 keeps a 5x margin from the bug and
    // a 3x margin above every observed value.
    assert!(err < 0.5, "transformation error {err}");
    assert!(res.fitness > 0.5, "fitness {}", res.fitness);
}
