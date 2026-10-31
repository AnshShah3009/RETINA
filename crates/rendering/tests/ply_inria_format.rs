//! The Gaussian PLY reader must read files laid out the way the INRIA 3DGS
//! reference implementation writes them (`GaussianModel.save_ply`):
//!
//! ```text
//! x y z nx ny nz f_dc_0..2 f_rest_0..(3*(K-1)-1) opacity scale_0..2 rot_0..3
//! ```
//!
//! stored as `binary_little_endian`, with
//!
//! * `opacity` the logit of the opacity (activated with a sigmoid),
//! * `scale_*` the natural log of the scale (activated with `exp`),
//! * `f_dc_*` the degree-0 SH coefficients, colour = `0.5 + C0 * f_dc`,
//! * `f_rest_*` channel-major (all R coefficients, then G, then B),
//! * `rot_0..3` the quaternion as `w, x, y, z`.
//!
//! The reader used to collect the property names and then ignore them,
//! assuming its own writer's column order, and returned `Ok(empty)` for a
//! binary file. Every fixture here is generated in the test.

use std::io::Write;

use cv_rendering::gaussian_splatting::io::gaussian_cloud_to_ply_string;
use cv_rendering::gaussian_splatting::{
    read_ply_gaussian_cloud, write_ply_gaussian_cloud, Gaussian, GaussianCloud,
};
use nalgebra::{Point3, Vector3, Vector4};

const SH_C0: f32 = 0.282_094_79;

/// Raw INRIA-order property names for SH degree 1 (9 `f_rest`).
fn inria_property_names() -> Vec<String> {
    let mut names: Vec<String> = ["x", "y", "z", "nx", "ny", "nz"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for i in 0..3 {
        names.push(format!("f_dc_{i}"));
    }
    for i in 0..9 {
        names.push(format!("f_rest_{i}"));
    }
    names.push("opacity".into());
    for i in 0..3 {
        names.push(format!("scale_{i}"));
    }
    for i in 0..4 {
        names.push(format!("rot_{i}"));
    }
    names
}

/// One raw INRIA record: position (1, 2, 3), f_dc (0.5, 0.6, 0.7), f_rest
/// 0.01..0.09, opacity logit 2.0, log scale (-3, -2, -1), and a 45 degree
/// rotation about z stored as (w, x, y, z).
fn inria_raw_record() -> Vec<f32> {
    let half = std::f32::consts::FRAC_PI_8;
    let mut v = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7];
    for i in 0..9 {
        v.push(0.01 * (i + 1) as f32);
    }
    v.push(2.0);
    v.extend_from_slice(&[-3.0, -2.0, -1.0]);
    v.extend_from_slice(&[half.cos(), 0.0, 0.0, half.sin()]);
    v
}

fn header(format: &str, names: &[String], count: usize) -> String {
    let mut h = format!("ply\nformat {format} 1.0\nelement vertex {count}\n");
    for n in names {
        h.push_str(&format!("property float {n}\n"));
    }
    h.push_str("end_header\n");
    h
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Assert the decoded Gaussian matches `inria_raw_record`.
fn assert_inria_record(g: &Gaussian) {
    let half = std::f32::consts::FRAC_PI_8;
    assert!((g.position - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
    let expected_scale = Vector3::new((-3.0f32).exp(), (-2.0f32).exp(), (-1.0f32).exp());
    assert!(
        (g.scale - expected_scale).norm() < 1e-5,
        "scale {:?}, expected {:?}",
        g.scale,
        expected_scale
    );
    assert!(
        (g.opacity - sigmoid(2.0)).abs() < 1e-5,
        "opacity {}, expected {}",
        g.opacity,
        sigmoid(2.0)
    );
    let expected_colour = Vector3::new(0.5 + SH_C0 * 0.5, 0.5 + SH_C0 * 0.6, 0.5 + SH_C0 * 0.7);
    assert!(
        (g.spherical_harmonics.dc() - expected_colour).norm() < 1e-5,
        "dc {:?}, expected {:?}",
        g.spherical_harmonics.dc(),
        expected_colour
    );
    // This crate stores quaternions as (x, y, z, w).
    let expected_rot = Vector4::new(0.0, 0.0, half.sin(), half.cos());
    assert!(
        (g.rotation - expected_rot).norm() < 1e-5,
        "rotation {:?}, expected {:?}",
        g.rotation,
        expected_rot
    );
    // f_rest is channel-major in the file; the crate layout is interleaved
    // [R0 G0 B0 R1 G1 B1 ...]. f_rest_0..2 = R1..R3, 3..5 = G1..G3, 6..8 = B1..B3.
    let c = &g.spherical_harmonics.coeffs;
    assert_eq!(c.len(), 12);
    assert_eq!(g.spherical_harmonics.degree, 1);
    // R1, G1, B1
    assert!((c[3] - 0.01).abs() < 1e-6, "{c:?}");
    assert!((c[4] - 0.04).abs() < 1e-6, "{c:?}");
    assert!((c[5] - 0.07).abs() < 1e-6, "{c:?}");
    // R3, G3, B3
    assert!((c[9] - 0.03).abs() < 1e-6, "{c:?}");
    assert!((c[10] - 0.06).abs() < 1e-6, "{c:?}");
    assert!((c[11] - 0.09).abs() < 1e-6, "{c:?}");
}

fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(bytes).expect("write");
    f.flush().expect("flush");
    f
}

#[test]
fn inria_ascii_file_reads_by_property_name() {
    let names = inria_property_names();
    let mut text = header("ascii", &names, 1);
    let rec: Vec<String> = inria_raw_record().iter().map(|v| v.to_string()).collect();
    text.push_str(&rec.join(" "));
    text.push('\n');
    let f = write_temp(text.as_bytes());

    let cloud = read_ply_gaussian_cloud(f.path()).expect("read");
    assert_eq!(cloud.num_gaussians(), 1);
    assert_inria_record(&cloud.gaussians[0]);
}

#[test]
fn inria_binary_little_endian_file_reads() {
    let names = inria_property_names();
    let mut bytes = header("binary_little_endian", &names, 2).into_bytes();
    for _ in 0..2 {
        for v in inria_raw_record() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let f = write_temp(&bytes);

    let cloud = read_ply_gaussian_cloud(f.path()).expect("binary PLY must read");
    assert_eq!(cloud.num_gaussians(), 2);
    assert_inria_record(&cloud.gaussians[0]);
    assert_inria_record(&cloud.gaussians[1]);
}

#[test]
fn binary_big_endian_and_double_properties_read() {
    // Mixed property types: position as double, everything else float.
    let names = inria_property_names();
    let mut h = String::from("ply\nformat binary_big_endian 1.0\nelement vertex 1\n");
    for n in &names {
        let ty = if n == "x" || n == "y" || n == "z" {
            "double"
        } else {
            "float"
        };
        h.push_str(&format!("property {ty} {n}\n"));
    }
    h.push_str("end_header\n");
    let mut bytes = h.into_bytes();
    for (n, v) in names.iter().zip(inria_raw_record()) {
        if n == "x" || n == "y" || n == "z" {
            bytes.extend_from_slice(&(v as f64).to_be_bytes());
        } else {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
    }
    let f = write_temp(&bytes);
    let cloud = read_ply_gaussian_cloud(f.path()).expect("read");
    assert_eq!(cloud.num_gaussians(), 1);
    assert_inria_record(&cloud.gaussians[0]);
}

#[test]
fn permuted_columns_read_identically() {
    let names = inria_property_names();
    let rec = inria_raw_record();
    // Reverse the column order.
    let rev_names: Vec<String> = names.iter().rev().cloned().collect();
    let rev_rec: Vec<String> = rec.iter().rev().map(|v| v.to_string()).collect();
    let mut text = header("ascii", &rev_names, 1);
    text.push_str(&rev_rec.join(" "));
    text.push('\n');
    let f = write_temp(text.as_bytes());

    let cloud = read_ply_gaussian_cloud(f.path()).expect("read");
    assert_inria_record(&cloud.gaussians[0]);
}

#[test]
fn truncated_binary_is_an_error() {
    let names = inria_property_names();
    let mut bytes = header("binary_little_endian", &names, 2).into_bytes();
    for v in inria_raw_record() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    // Only one of two records present.
    let f = write_temp(&bytes);
    assert!(read_ply_gaussian_cloud(f.path()).is_err());
}

#[test]
fn missing_position_property_is_an_error() {
    let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float opacity\nend_header\n0.5\n";
    let f = write_temp(text.as_bytes());
    assert!(read_ply_gaussian_cloud(f.path()).is_err());
}

#[test]
fn short_ascii_row_is_an_error() {
    let names = inria_property_names();
    let mut text = header("ascii", &names, 1);
    text.push_str("1 2 3\n");
    let f = write_temp(text.as_bytes());
    assert!(read_ply_gaussian_cloud(f.path()).is_err());
}

#[test]
fn unknown_format_is_an_error() {
    let text = "ply\nformat binary_middle_endian 1.0\nelement vertex 0\nend_header\n";
    let f = write_temp(text.as_bytes());
    assert!(read_ply_gaussian_cloud(f.path()).is_err());
}

#[test]
fn preceding_element_is_skipped() {
    // An element declared before `vertex` must be skipped, not read as vertices.
    let names = inria_property_names();
    let mut text = String::from("ply\nformat ascii 1.0\ncomment made by a test\n");
    text.push_str("element camera 2\nproperty float fx\nproperty float fy\n");
    text.push_str("element vertex 1\n");
    for n in &names {
        text.push_str(&format!("property float {n}\n"));
    }
    text.push_str("end_header\n100 100\n200 200\n");
    let rec: Vec<String> = inria_raw_record().iter().map(|v| v.to_string()).collect();
    text.push_str(&rec.join(" "));
    text.push('\n');
    let f = write_temp(text.as_bytes());
    let cloud = read_ply_gaussian_cloud(f.path()).expect("read");
    assert_eq!(cloud.num_gaussians(), 1);
    assert_inria_record(&cloud.gaussians[0]);
}

fn rotated_cloud() -> GaussianCloud {
    // A non-identity rotation, so an (x,y,z,w) <-> (w,x,y,z) mixup shows.
    let q = Vector4::new(0.1, -0.3, 0.5, 0.8).normalize();
    let mut g = Gaussian::new(
        Point3::new(-1.0, 0.5, 4.0),
        Vector3::new(0.02, 0.05, 0.3),
        q,
        Vector3::new(0.1, 0.45, 0.9),
    )
    .with_opacity(0.37);
    g.spherical_harmonics
        .coeffs
        .extend((0..9).map(|i| 0.1 * i as f32 - 0.4));
    g.spherical_harmonics.degree = 1;
    GaussianCloud::from_gaussians(vec![g])
}

#[test]
fn write_then_read_round_trips_rotation_colour_and_sh() {
    let cloud = rotated_cloud();
    let f = tempfile::NamedTempFile::new().unwrap();
    write_ply_gaussian_cloud(&cloud, f.path()).unwrap();
    let back = read_ply_gaussian_cloud(f.path()).unwrap();
    let (a, b) = (&cloud.gaussians[0], &back.gaussians[0]);
    assert!((a.position - b.position).norm() < 1e-6);
    assert!((a.scale - b.scale).norm() < 1e-6, "{:?}", b.scale);
    assert!((a.rotation - b.rotation).norm() < 1e-6, "{:?}", b.rotation);
    assert!((a.opacity - b.opacity).abs() < 1e-6);
    assert_eq!(b.spherical_harmonics.degree, 1);
    assert_eq!(b.spherical_harmonics.coeffs.len(), 12);
    for (x, y) in a
        .spherical_harmonics
        .coeffs
        .iter()
        .zip(&b.spherical_harmonics.coeffs)
    {
        assert!(
            (x - y).abs() < 1e-5,
            "{:?} vs {:?}",
            a.spherical_harmonics,
            b.spherical_harmonics
        );
    }
}

#[test]
fn written_file_uses_inria_storage_conventions() {
    // Read the writer's output as raw columns by name and check the stored
    // quantities are the INRIA ones, so the INRIA viewer / tooling reads it.
    let cloud = rotated_cloud();
    let text = gaussian_cloud_to_ply_string(&cloud);
    let mut names = Vec::new();
    let mut lines = text.lines();
    for line in lines.by_ref() {
        if let Some(n) = line.strip_prefix("property float ") {
            names.push(n.to_string());
        }
        if line == "end_header" {
            break;
        }
    }
    let row: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let col = |n: &str| row[names.iter().position(|x| x == n).unwrap()];
    let g = &cloud.gaussians[0];
    assert!(
        (col("rot_0") - g.rotation.w).abs() < 1e-6,
        "rot_0 must be w"
    );
    assert!((col("rot_1") - g.rotation.x).abs() < 1e-6);
    assert!((col("rot_3") - g.rotation.z).abs() < 1e-6);
    assert!((col("scale_2") - g.scale.z.ln()).abs() < 1e-6);
    assert!((sigmoid(col("opacity")) - g.opacity).abs() < 1e-6);
    assert!((0.5 + SH_C0 * col("f_dc_1") - g.spherical_harmonics.coeffs[1]).abs() < 1e-6);
    // Channel-major rest: f_rest_0 is R1 (crate index 3), f_rest_3 is G1 (index 4).
    assert!((col("f_rest_0") - g.spherical_harmonics.coeffs[3]).abs() < 1e-6);
    assert!((col("f_rest_3") - g.spherical_harmonics.coeffs[4]).abs() < 1e-6);
}
