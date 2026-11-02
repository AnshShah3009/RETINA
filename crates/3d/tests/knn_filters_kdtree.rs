//! `statistical_outlier_removal` and `estimate_normals_knn` find neighbours
//! with an f64 KD-tree instead of an O(n) scan per point. The `brute_*`
//! functions below are the previous implementations, transcribed; the tree
//! versions must select the same neighbours (bit-identical distances), so the
//! inlier sets are equal and the normals agree to summation-order rounding.
//!
//! `brute_statistical` follows Open3D's neighbourhood rule (the point itself
//! is one of its `nb` nearest), which the filter adopted after this file was
//! written; see `statistical_outlier_open3d.rs`.

use cv_3d::filters::{estimate_normals_knn, statistical_outlier_removal};
use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;
use std::time::Instant;

fn dist_sq(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}

fn brute_statistical(points: &[Point3<f64>], nb: usize, std_ratio: f64) -> Vec<usize> {
    // Open3D's rule (see `statistical_outlier_removal`): the `nb` nearest
    // points *including the point itself*, averaged over all of them; the
    // sample standard deviation; a strict threshold. Points whose mean is 0
    // (every neighbour coincides with them) are kept.
    if points.len() < 2 {
        return (0..points.len()).collect();
    }
    let k = nb.max(1).min(points.len());
    let mean_dists: Vec<f64> = points
        .par_iter()
        .map(|p| {
            let mut d: Vec<f64> = points.iter().map(|q| dist_sq(p, q)).collect();
            d.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
            d[..k].iter().map(|v| v.sqrt()).sum::<f64>() / k as f64
        })
        .collect();
    let n = mean_dists.len() as f64;
    let mean = mean_dists.iter().sum::<f64>() / n;
    let std = (mean_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let thr = mean + std_ratio * std;
    (0..points.len())
        .filter(|&i| mean_dists[i] < thr || mean_dists[i] == 0.0)
        .collect()
}

fn brute_normals(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    // Open3D's rule (see `normals_knn_open3d.rs`): the `k` nearest points
    // *including the point itself*; fewer than three give (0, 0, 1).
    let m = k.min(points.len());
    if m < 3 {
        return vec![Vector3::z(); points.len()];
    }
    let k = m - 1;
    points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let mut d: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, q)| (j, dist_sq(p, q)))
                .collect();
            d.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut c = p.coords;
            for &(j, _) in &d[..k] {
                c += points[j].coords;
            }
            c /= (k + 1) as f64;
            let mut cov = Matrix3::<f64>::zeros();
            for &(j, _) in &d[..k] {
                let v = points[j].coords - c;
                cov += v * v.transpose();
            }
            let v = p.coords - c;
            cov += v * v.transpose();
            let n = cv_math::linalg::min_eigenvector_3x3(&cov);
            if n.z < 0.0 { -n } else { n }
        })
        .collect()
}

/// A noisy, curved surface patch plus scattered far outliers.
fn cloud(n: usize, outliers: usize, seed: u64) -> Vec<Point3<f64>> {
    let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut pts: Vec<Point3<f64>> = (0..n)
        .map(|_| {
            let (u, v) = (r() * 4.0 - 2.0, r() * 4.0 - 2.0);
            Point3::new(u, v, 0.1 * (u * u - v * v) + 0.01 * (r() - 0.5))
        })
        .collect();
    for _ in 0..outliers {
        pts.push(Point3::new(
            r() * 40.0 - 20.0,
            r() * 40.0 - 20.0,
            r() * 40.0 - 20.0,
        ));
    }
    pts
}

#[test]
fn kdtree_statistical_filter_keeps_the_same_inliers() {
    for (seed, &(n, out)) in [(5usize, 0usize), (40, 3), (600, 25), (3000, 60)]
        .iter()
        .enumerate()
    {
        let pts = cloud(n, out, seed as u64);
        for (nb, ratio) in [
            (1usize, 1.0),
            (5, 2.0),
            (16, 1.0),
            (30, 0.5),
            (n + out, 1.0),
        ] {
            let (kept, idx) = statistical_outlier_removal(&pts, nb, ratio);
            assert_eq!(idx, brute_statistical(&pts, nb, ratio), "n={n} nb={nb}");
            assert_eq!(kept.len(), idx.len());
        }
    }
    // Coincident points: a duplicate of the query may be returned in its
    // place; both are at distance 0, so the mean must not change.
    let mut pts = cloud(200, 5, 9);
    for i in 0..30 {
        pts.push(pts[i]);
        pts.push(pts[i]);
    }
    for nb in [1usize, 2, 3, 8] {
        let (_, idx) = statistical_outlier_removal(&pts, nb, 1.0);
        assert_eq!(idx, brute_statistical(&pts, nb, 1.0), "duplicates nb={nb}");
    }
}

#[test]
fn kdtree_normals_match_brute_force() {
    for (seed, &n) in [2usize, 3, 50, 700, 3000].iter().enumerate() {
        let pts = cloud(n, 0, 100 + seed as u64);
        for k in [1usize, 3, 8, 20] {
            let got = estimate_normals_knn(&pts, k);
            let want = brute_normals(&pts, k);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                // Same neighbour set; only the covariance summation order
                // differs (a few ulps of the entries), amplified by at most
                // ||C|| / eigengap in the eigenvector.
                assert!(
                    (g - w).amax() < 1e-9,
                    "n={n} k={k} point {i}: {g:?} vs {w:?}"
                );
            }
        }
    }
}

/// `cargo test --release -p cv-3d --test knn_filters_kdtree -- --ignored --nocapture`
#[test]
#[ignore]
fn knn_filters_timing() {
    for n in [5_000usize, 20_000, 100_000] {
        let pts = cloud(n, n / 100, 1);
        let t = Instant::now();
        let (_, idx) = statistical_outlier_removal(&pts, 20, 2.0);
        let t_sor = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let normals = estimate_normals_knn(&pts, 20);
        let t_nrm = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let want = brute_statistical(&pts, 20, 2.0);
        let b_sor = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let want_n = brute_normals(&pts, 20);
        let b_nrm = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(idx, want);
        assert!(
            normals
                .iter()
                .zip(&want_n)
                .all(|(a, b)| (a - b).amax() < 1e-9)
        );
        println!(
            "n={:6}: statistical_outlier_removal brute {b_sor:9.2} ms, kd-tree {t_sor:7.2} ms; \
             estimate_normals_knn brute {b_nrm:9.2} ms, kd-tree {t_nrm:7.2} ms",
            pts.len()
        );
    }
}
