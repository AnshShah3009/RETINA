//! Degenerate-input contract for `cv-pointcloud`.
//!
//! Each test pins a defect that produced a *plausible* result rather than a
//! visible failure:
//!
//! * a NaN voxel size collapsed the whole cloud into one point (NaN is neither
//!   `<= 0.0` nor `> 0.0`, so it slipped past the guard and every
//!   `(x / NaN).floor() as i32` saturated to the same voxel);
//! * one non-finite coordinate panicked the k-nearest-neighbour loops
//!   (`rstar`'s `nearest_neighbor_iter` unwraps the closest candidate), so a
//!   cloud with a single bad point took the process down through
//!   `estimate_normals`, `orient_normals` and `remove_statistical_outliers`;
//! * a negative radius behaved exactly like its absolute value, because the
//!   search works on a squared radius, so `eps = -1.0` produced the same
//!   clusters as `eps = 1.0` and `radius = -1.0` kept the whole cloud.

use cv_core::PointCloud;
use cv_pointcloud::point_cloud::*;
use nalgebra::{Point3, Vector3};

fn grid(n: i32, step: f32) -> Vec<Point3<f32>> {
    let mut points = Vec::new();
    for i in 0..n {
        for j in 0..n {
            points.push(Point3::new(i as f32 * step, j as f32 * step, 0.0));
        }
    }
    points
}

#[test]
fn a_nan_voxel_size_does_not_collapse_the_cloud() {
    let points = grid(4, 0.1);
    let pc = PointCloud::new(points.clone());

    let collapsed = voxel_down_sample(&pc, f32::NAN);
    assert_eq!(
        collapsed.len(),
        pc.len(),
        "a NaN voxel size must not silently collapse {} points into {}",
        pc.len(),
        collapsed.len()
    );

    // Control: a real voxel size still downsamples (all 16 points sit inside one
    // 0.5 voxel).
    assert_eq!(voxel_down_sample(&pc, 0.5).len(), 1);
    // Control: a non-positive size is a documented no-op, unchanged.
    assert_eq!(voxel_down_sample(&pc, 0.0).len(), pc.len());
    assert_eq!(voxel_down_sample(&pc, -1.0).len(), pc.len());
}

#[test]
fn a_non_finite_point_does_not_panic_normal_estimation() {
    let points = grid(5, 1.0);
    let clean = PointCloud::new(points.clone());
    let mut with_bad = clean.clone();
    with_bad.points.push(Point3::new(f32::NAN, 0.0, 0.0));

    // Before: panicked inside rstar (`Option::unwrap()` on `None`).
    let mut bad = with_bad.clone();
    estimate_normals(&mut bad, 8);
    let mut good = clean.clone();
    estimate_normals(&mut good, 8);

    let bad_normals = bad.normals.as_ref().expect("normals must be set");
    let good_normals = good.normals.as_ref().expect("normals must be set");
    assert_eq!(bad_normals.len(), bad.len());

    // The one bad point must not change any other point's normal: it is not in
    // the search tree, so the neighbourhoods are the same as the clean cloud's.
    let worst = (0..good.len())
        .map(|i| (bad_normals[i] - good_normals[i]).norm())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-6,
        "a non-finite point changed the other normals by {worst}"
    );

    // It gets the same default as a point with too few neighbours, and every
    // normal is finite and unit length.
    let last = bad_normals[good.len()];
    assert_eq!(last, Vector3::new(0.0, 0.0, 1.0));
    assert!(bad_normals.iter().all(|n| n.iter().all(|v| v.is_finite())));
    assert!(
        (bad_normals[..good.len()]
            .iter()
            .map(|n| n.norm())
            .fold(0.0f32, f32::max)
            - 1.0)
            .abs()
            < 1e-4
    );

    // Control: the clean cloud still gets vertical normals for a plane at z = 0.
    assert!(good_normals.iter().all(|n| n.z.abs() > 0.99));
}

#[test]
fn orient_normals_survives_a_non_finite_point() {
    let mut pc = PointCloud::new(grid(5, 1.0));
    pc.points.push(Point3::new(0.0, 0.0, f32::NAN));
    let len = pc.len();
    estimate_normals(&mut pc, 5);
    // Before: panicked inside rstar while querying the tree for a seed.
    orient_normals(&mut pc, 5);

    let normals = pc.normals.as_ref().expect("normals must be set");
    assert_eq!(normals.len(), len);
    assert!(normals.iter().all(|n| n.iter().all(|v| v.is_finite())));

    // Control: the plane's finite points end up consistently oriented (the
    // unoriented eigenvector from the solver is arbitrary per point).
    let sign = normals[0].z.signum();
    assert!(sign != 0.0);
    assert!(
        normals[..len - 1].iter().all(|n| n.z * sign > 0.0),
        "the finite points of a plane must share one orientation"
    );
}

#[test]
fn outlier_removal_survives_a_non_finite_point() {
    let mut points: Vec<Point3<f32>> = (0..10).map(|_| Point3::new(0.0, 0.0, 0.0)).collect();
    points.push(Point3::new(10.0, 10.0, 10.0)); // index 10, the outlier
    let clean = PointCloud::new(points.clone());
    let mut with_bad = clean.clone();
    with_bad.points.push(Point3::new(f32::INFINITY, 0.0, 0.0)); // index 11

    let (kept_clean, inliers_clean) = remove_statistical_outliers(&clean, 5, 1.0);
    // Before: panicked inside rstar.
    let (kept_bad, inliers_bad) = remove_statistical_outliers(&with_bad, 5, 1.0);

    assert_eq!(kept_clean.len(), 10);
    assert_eq!(
        kept_bad.len(),
        kept_clean.len(),
        "the non-finite point changed which points were kept"
    );
    assert_eq!(inliers_bad, inliers_clean);
    assert!(
        !inliers_bad.contains(&11),
        "a point with no position cannot be an inlier"
    );

    // The same two functions are also reached through the radius filter, which
    // never used the panicking query and keeps only points with neighbours.
    let (kept_radius, inliers_radius) = remove_radius_outliers(&with_bad, 1.0, 5);
    assert_eq!(kept_radius.len(), 10);
    assert!(!inliers_radius.contains(&11));
}

#[test]
fn a_negative_radius_does_not_behave_like_a_positive_one() {
    let two = PointCloud::new(vec![
        Point3::new(0.0f32, 0.0, 0.0),
        Point3::new(0.5f32, 0.0, 0.0),
    ]);

    // Nothing is within a negative radius, so nothing is an inlier.
    let (kept_negative, inliers_negative) = remove_radius_outliers(&two, -1.0, 2);
    assert_eq!(
        kept_negative.len(),
        0,
        "radius = -1.0 kept {} of {} points, exactly like radius = 1.0",
        kept_negative.len(),
        two.len()
    );
    assert!(inliers_negative.is_empty());

    // Control: the positive radius keeps both (each has one neighbour).
    let (kept_positive, inliers_positive) = remove_radius_outliers(&two, 1.0, 2);
    assert_eq!(kept_positive.len(), 2);
    assert_eq!(inliers_positive, vec![0, 1]);
}

#[test]
fn a_negative_eps_produces_no_clusters() {
    let mut points: Vec<Point3<f32>> = (0..5).map(|_| Point3::new(0.0, 0.0, 0.0)).collect();
    points.extend((0..5).map(|_| Point3::new(10.0, 10.0, 10.0)));
    let pc = PointCloud::new(points);

    let negative = cluster_dbscan(&pc, -1.0, 3);
    assert!(
        negative.iter().all(|&label| label == -1),
        "eps = -1.0 clustered points that are not within a negative radius: {negative:?}"
    );

    // Control: the positive radius finds the two clusters.
    let positive = cluster_dbscan(&pc, 1.0, 3);
    assert_eq!(positive[0], positive[4]);
    assert_eq!(positive[5], positive[9]);
    assert_ne!(positive[0], positive[5]);
}

#[test]
fn fpfh_uses_the_published_darboux_frame() {
    // Two points, one pair. p1 at the origin with its normal along +z, p2 on the
    // +x axis with its normal along +y. Rusu et al. (ICRA 2009) build the frame
    // as u = n1, v = u x d, w = u x v, giving alpha = v . n2 = +1 (the last
    // alpha bin), phi = u . d / |d| = 0 and theta = atan2(w . n2, u . n2) = 0.
    let pc = PointCloud::new(vec![
        Point3::new(0.0f32, 0.0, 0.0),
        Point3::new(1.0f32, 0.0, 0.0),
    ])
    .with_normals(vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
    ])
    .unwrap();

    let features = compute_fpfh_feature(&pc, 2.0).expect("normals are present");
    let hist = &features[0];

    let alpha_bin = ((1.0f32 + 1.0) * 5.5f32).floor().clamp(0.0, 10.0) as usize;
    assert_eq!(alpha_bin, 10);
    assert!(
        (hist[alpha_bin] - 100.0 / 3.0).abs() < 1e-3,
        "alpha = +1 belongs in bin {alpha_bin}; histogram = {:?}",
        hist
    );
    assert_eq!(
        hist[0], 0.0,
        "alpha = -1 is the mirrored frame: histogram = {:?}",
        hist
    );

    // Control: the histogram is normalised and finite, and phi/theta are centred
    // for this configuration.
    assert!((hist.iter().sum::<f32>() - 100.0).abs() < 1e-3);
    assert!(hist.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!((hist[11 + 5] - 100.0 / 3.0).abs() < 1e-3);
    assert!((hist[22 + 5] - 100.0 / 3.0).abs() < 1e-3);
}
