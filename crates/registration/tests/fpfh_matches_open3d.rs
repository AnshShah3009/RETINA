//! FPFH descriptors equal Open3D's `compute_fpfh_feature`.
//!
//! Open3D (after PCL) bins theta, alpha, phi in that order, builds the
//! Darboux frame as `v = d x n_source` from whichever point's normal is closer
//! to the joining line, adds `100 / k` per neighbour so each SPFH
//! sub-histogram sums to 100, and weights neighbour SPFHs by the inverse
//! *squared* distance, rescaling each sub-histogram of that sum to 100 before
//! adding the point's own SPFH. `cv-registration` differed in all of these.
//!
//! References: Open3D 0.20.0, `KDTreeSearchParamRadius(0.25)`, on the
//! closed-form cloud below (coordinates and normals rounded to f32 first, as
//! this crate stores them).

use cv_core::point_cloud::PointCloud;
use cv_registration::compute_fpfh_features;
use nalgebra::{Point3, Vector3};

fn cloud() -> PointCloud {
    let mut points = Vec::new();
    let mut normals = Vec::new();
    for i in 0..300 {
        let t = i as f64;
        points.push(Point3::new(
            ((t * 0.37).sin() * 0.5 + (t * 0.011).cos() * 0.2) as f32,
            ((t * 0.53).cos() * 0.4 + (t * 0.007).sin() * 0.3) as f32,
            ((t * 0.29).sin() * (t * 0.13).cos() * 0.3) as f32,
        ));
        let n =
            Vector3::new((t * 0.21).cos(), (t * 0.17).sin(), (t * 0.05).cos() + 0.5).normalize();
        normals.push(n.cast::<f32>());
    }
    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

const OPEN3D: [(usize, [f64; 33]); 4] = [
    (
        0,
        [
            4.995487, 4.770630, 10.407092, 16.873029, 21.335327, 54.528741, 38.626732, 27.284239,
            10.788134, 4.119845, 6.270743, 21.548273, 22.993848, 10.452207, 8.094944, 10.966196,
            23.480782, 22.628055, 11.652116, 17.114189, 12.611701, 38.457689, 29.482613, 27.990672,
            21.932897, 16.251504, 18.790544, 0.273804, 5.156841, 5.327106, 21.812679, 21.367727,
            31.613612,
        ],
    ),
    (
        7,
        [
            3.133698, 2.286069, 12.558194, 2.452022, 13.259406, 74.464032, 29.201860, 21.061580,
            13.693625, 18.216647, 9.672867, 15.424866, 15.775233, 11.640152, 4.189423, 6.101701,
            41.157657, 28.822328, 12.090751, 11.569749, 23.389117, 29.839023, 51.975048, 47.505392,
            19.468883, 5.871485, 21.442286, 0.251084, 4.361817, 2.795648, 19.365353, 3.627401,
            23.335603,
        ],
    ),
    (
        123,
        [
            1.186277, 0.799961, 6.284524, 1.724110, 18.062752, 105.909414, 45.601958, 7.979834,
            4.141230, 4.509350, 3.800590, 14.109106, 8.573746, 22.962287, 20.761114, 10.816730,
            37.259163, 30.451504, 2.481983, 4.010185, 21.019248, 27.554936, 25.702570, 47.707003,
            27.279090, 2.735133, 21.102283, 2.823351, 16.842412, 6.608774, 23.043890, 8.683984,
            17.471510,
        ],
    ),
    (
        250,
        [
            6.619813, 11.752899, 2.743147, 17.312931, 54.011529, 64.364021, 14.017330, 11.697118,
            3.829911, 2.675440, 10.975861, 27.323592, 35.147963, 10.319787, 30.338078, 16.733691,
            37.755284, 7.398491, 12.792086, 5.099045, 10.226543, 6.865439, 10.620727, 24.113274,
            7.545142, 9.435839, 18.143084, 2.999760, 14.020322, 23.484861, 19.142645, 48.840366,
            21.653982,
        ],
    ),
];

fn assert_matches(name: &str, got: &[f32; 33], i: usize, want: &[f64; 33]) {
    for (bin, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g as f64 - w).abs() < 1e-3,
            "{name}: point {i} bin {bin} is {g}, Open3D {w}"
        );
    }
}

#[test]
fn registration_fpfh_matches_open3d() {
    let features = compute_fpfh_features(&cloud(), 0.25).unwrap();
    for (i, want) in OPEN3D.iter() {
        assert_matches("cv-registration", &features[*i].histogram, *i, want);
    }
    for f in &features {
        let sum: f32 = f.histogram.iter().sum();
        assert!((sum - 600.0).abs() < 1e-2, "sum {sum}, Open3D 600");
    }
}
