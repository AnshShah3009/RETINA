//! Open3D's FPFH (Fast Point Feature Histogram) descriptor, after PCL.
//!
//! This is the one implementation of the descriptor in the workspace:
//! [`crate::compute_fpfh_feature`] and `cv-registration`'s
//! `compute_fpfh_features` differ only in how they find neighbours and what
//! they do when normals are missing, and both call
//! [`fpfh_from_neighbourhoods`].
//!
//! It follows Open3D's `ComputePairFeatures`, `ComputeSPFHFeature` and
//! `ComputeFPFHFeature` (`open3d/pipelines/registration/Feature.cpp`) in f64:
//!
//! * the Darboux frame of a pair has as source whichever point's normal makes
//!   the smaller angle with the line joining them, so a pair's features do not
//!   depend on which point is the query, and `v = d x n_source`;
//! * the 33 bins are theta over `[-pi, pi]`, then alpha and phi over `[-1, 1]`,
//!   11 each, and each neighbour adds `100 / k` to one bin of each, so every
//!   SPFH sub-histogram sums to 100;
//! * neighbour SPFHs are weighted by the inverse *squared* distance, each
//!   sub-histogram of that sum is rescaled to 100, and the point's own SPFH is
//!   added, so a descriptor with neighbours sums to 600.

use nalgebra::Vector3;
use rayon::prelude::*;

/// Number of bins in an FPFH descriptor.
pub const FPFH_BINS: usize = 33;

/// Open3D's `ComputePairFeatures`: the Darboux-frame angles
/// `(theta, alpha, phi)` of the pair `(p1, n1)`, `(p2, n2)`, or zeros for
/// coincident points or a degenerate frame (which Open3D still bins).
pub fn fpfh_pair_features(
    p1: &Vector3<f64>,
    n1: &Vector3<f64>,
    p2: &Vector3<f64>,
    n2: &Vector3<f64>,
) -> [f64; 3] {
    let mut dp = p2 - p1;
    let dist = dp.norm();
    if dist == 0.0 {
        return [0.0; 3];
    }
    let angle1 = n1.dot(&dp) / dist;
    let angle2 = n2.dot(&dp) / dist;
    let (src, tgt, phi) = if angle1.abs().acos() > angle2.abs().acos() {
        dp = -dp;
        (n2, n1, -angle2)
    } else {
        (n1, n2, angle1)
    };
    let v = dp.cross(src);
    let v_norm = v.norm();
    if v_norm == 0.0 {
        return [0.0; 3];
    }
    let v = v / v_norm;
    let w = src.cross(&v);
    let alpha = v.dot(tgt);
    let theta = w.dot(tgt).atan2(src.dot(tgt));
    [theta, alpha, phi]
}

/// Bin of `value` in 11 equal bins over `[lo, hi]`, clamped.
fn bin11(value: f64, lo: f64, hi: f64) -> usize {
    ((11.0 * (value - lo) / (hi - lo)).floor() as i64).clamp(0, 10) as usize
}

/// Open3D's `ComputeSPFHFeature` for point `i` with neighbours `neighbours`
/// (not containing `i`).
fn spfh(
    i: usize,
    neighbours: &[usize],
    points: &[Vector3<f64>],
    normals: &[Vector3<f64>],
) -> [f64; FPFH_BINS] {
    let mut histogram = [0.0f64; FPFH_BINS];
    if neighbours.is_empty() {
        return histogram;
    }
    let increment = 100.0 / neighbours.len() as f64;
    for &j in neighbours {
        let [theta, alpha, phi] =
            fpfh_pair_features(&points[i], &normals[i], &points[j], &normals[j]);
        histogram[bin11(theta, -std::f64::consts::PI, std::f64::consts::PI)] += increment;
        histogram[11 + bin11(alpha, -1.0, 1.0)] += increment;
        histogram[22 + bin11(phi, -1.0, 1.0)] += increment;
    }
    histogram
}

/// Open3D's `ComputeFPFHFeature` for point `i` from every point's SPFH.
fn fpfh(
    i: usize,
    neighbours: &[usize],
    points: &[Vector3<f64>],
    all_spfh: &[[f64; FPFH_BINS]],
) -> [f64; FPFH_BINS] {
    let mut out = [0.0f64; FPFH_BINS];
    let mut sums = [0.0f64; 3];
    for &j in neighbours {
        let dist_sq = (points[j] - points[i]).norm_squared();
        if dist_sq == 0.0 {
            continue;
        }
        for bin in 0..FPFH_BINS {
            let value = all_spfh[j][bin] / dist_sq;
            sums[bin / 11] += value;
            out[bin] += value;
        }
    }
    for bin in 0..FPFH_BINS {
        if sums[bin / 11] != 0.0 {
            out[bin] *= 100.0 / sums[bin / 11];
        }
        out[bin] += all_spfh[i][bin];
    }
    out
}

/// FPFH descriptors of every point, given each point's neighbourhood.
///
/// `neighbourhoods[i]` lists the indices of the points within the search
/// radius of point `i`, *excluding `i` itself* (Open3D drops the first search
/// result, the query). A point with no neighbours gets an all-zero descriptor,
/// as in Open3D.
///
/// # Panics
///
/// If `points`, `normals` and `neighbourhoods` differ in length, or a
/// neighbourhood holds an out-of-range index.
pub fn fpfh_from_neighbourhoods(
    points: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    neighbourhoods: &[Vec<usize>],
) -> Vec<[f64; FPFH_BINS]> {
    let n = points.len();
    assert_eq!(normals.len(), n, "one normal per point");
    assert_eq!(neighbourhoods.len(), n, "one neighbourhood per point");
    let all_spfh: Vec<[f64; FPFH_BINS]> = (0..n)
        .into_par_iter()
        .map(|i| spfh(i, &neighbourhoods[i], points, normals))
        .collect();
    (0..n)
        .into_par_iter()
        .map(|i| fpfh(i, &neighbourhoods[i], points, &all_spfh))
        .collect()
}
