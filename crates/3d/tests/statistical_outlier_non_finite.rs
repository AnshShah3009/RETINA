//! A point without a position (a NaN or infinite coordinate) must not change
//! which other points `statistical_outlier_removal` keeps.
//!
//! The filter averaged every point's mean neighbour distance into the global
//! mean and standard deviation. A NaN point has a NaN mean, so the threshold
//! became NaN, `md < NaN` was false for every point, and only the points whose
//! mean was exactly 0 survived. Non-finite points are now left out of the
//! search tree and the statistics, and are never kept.

use cv_3d::filters::statistical_outlier_removal;
use nalgebra::Point3;

/// 29 irregularly spread points near the origin and one far outlier.
fn cloud() -> Vec<Point3<f64>> {
    let mut pts: Vec<Point3<f64>> = (0..29)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                (t * 0.37).sin() + 0.1 * (t * 0.011).cos(),
                (t * 0.53).cos() * 0.8,
                (t * 0.29).sin() * 0.3,
            )
        })
        .collect();
    pts.push(Point3::new(40.0, -35.0, 25.0));
    pts
}

#[test]
fn a_nan_point_does_not_change_which_points_survive() {
    let clean = cloud();
    let (_, kept_clean) = statistical_outlier_removal(&clean, 6, 1.0);
    // Control: the far outlier goes and most of the cluster stays.
    assert!(!kept_clean.contains(&29), "the far outlier must be removed");
    assert!(kept_clean.len() > 20, "kept only {kept_clean:?}");

    for (insert_at, bad) in [
        (10usize, Point3::new(f64::NAN, 0.0, 0.0)),
        (0, Point3::new(0.0, f64::NAN, f64::NAN)),
        (30, Point3::new(f64::INFINITY, 0.0, 0.0)),
    ] {
        let mut with_bad = clean.clone();
        with_bad.insert(insert_at, bad);

        let (kept_pts, kept_bad) = statistical_outlier_removal(&with_bad, 6, 1.0);
        assert!(
            !kept_bad.contains(&insert_at),
            "the non-finite point at {insert_at} was kept"
        );
        // Map the indices back onto the clean cloud.
        let remapped: Vec<usize> = kept_bad
            .iter()
            .map(|&i| if i > insert_at { i - 1 } else { i })
            .collect();
        assert_eq!(
            remapped, kept_clean,
            "a non-finite point at {insert_at} changed which points were kept"
        );
        assert_eq!(kept_pts.len(), kept_bad.len());
        assert!(
            kept_pts
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        );
    }
}

#[test]
fn only_non_finite_points_leave_nothing() {
    let pts = vec![
        Point3::new(f64::NAN, 0.0, 0.0),
        Point3::new(0.0, f64::INFINITY, 0.0),
        Point3::new(0.0, 0.0, f64::NEG_INFINITY),
    ];
    let (kept, idx) = statistical_outlier_removal(&pts, 3, 1.0);
    assert!(kept.is_empty() && idx.is_empty(), "kept {idx:?}");

    // One finite point among them is kept, as a lone point always is.
    let mut pts = pts;
    pts.push(Point3::new(1.0, 2.0, 3.0));
    let (_, idx) = statistical_outlier_removal(&pts, 3, 1.0);
    assert_eq!(idx, vec![3]);
}
