//! A PLY round trip must return the opacity that went in.
//!
//! The reader decodes `opacity` with the standard 3DGS inverse-sigmoid: raw
//! values are logits, so `opacity = 1 / (1 + exp(-raw))`. The writer used the
//! *negated* logit:
//!
//! ```text
//! log_opacity = ln((1 - o) / o)      // = logit(1 - o), not logit(o)
//! ```
//!
//! so a written file decodes to `1 - o`: opacity 0.9 came back as 0.1, and 0.7 as
//! 0.3. 0.5 is a fixed point of `o ↦ 1 - o`, which is why the existing round-trip
//! test - which only checks positions - never noticed. The same expression is in
//! both `write_ply_gaussian_cloud` and `gaussian_cloud_to_ply_string`.

use cv_rendering::gaussian_splatting::io::gaussian_cloud_to_ply_string;
use cv_rendering::gaussian_splatting::{
    read_ply_gaussian_cloud, write_ply_gaussian_cloud, Gaussian, GaussianCloud, SphericalHarmonics,
};
use nalgebra::{Point3, Vector3, Vector4};

fn cloud_with_opacity(opacity: f32) -> GaussianCloud {
    let g = Gaussian::new(
        Point3::new(1.0, 2.0, 3.0),
        Vector3::new(0.1, 0.2, 0.3),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(0.25, 0.5, 0.75),
    )
    .with_opacity(opacity);
    GaussianCloud::from_gaussians(vec![g])
}

/// The invariant, over a range that includes and straddles the 0.5 fixed point.
#[test]
fn opacity_survives_a_file_round_trip() {
    for opacity in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
        let cloud = cloud_with_opacity(opacity);
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let path = file.path().to_path_buf();
        write_ply_gaussian_cloud(&cloud, &path).expect("write");

        let back = read_ply_gaussian_cloud(&path).expect("read");
        assert_eq!(back.num_gaussians(), 1);
        let read_back = back.gaussians[0].opacity;
        println!("opacity {opacity} -> {}", read_back);
        assert!(
            (read_back - opacity).abs() < 1e-3,
            "opacity {opacity} round-tripped to {read_back} (1 - {opacity} is {})",
            1.0 - opacity
        );
    }
}

/// The string form takes the same path and had the same defect.
#[test]
fn opacity_survives_a_string_round_trip() {
    let cloud = cloud_with_opacity(0.9);
    let text = gaussian_cloud_to_ply_string(&cloud);

    let file = tempfile::NamedTempFile::new().expect("temp file");
    let path = file.path().to_path_buf();
    std::fs::write(&path, text).expect("write text");

    let back = read_ply_gaussian_cloud(&path).expect("read");
    let read_back = back.gaussians[0].opacity;
    println!("string round trip: 0.9 -> {read_back}");
    assert!(
        (read_back - 0.9).abs() < 1e-3,
        "opacity 0.9 round-tripped to {read_back} through the string form"
    );
}

/// Control: the rest of the record already round-trips, so the assertions above
/// are about opacity and not about a broken reader.
#[test]
fn position_scale_colour_and_rotation_still_round_trip() {
    let cloud = cloud_with_opacity(0.6);
    let file = tempfile::NamedTempFile::new().expect("temp file");
    let path = file.path().to_path_buf();
    write_ply_gaussian_cloud(&cloud, &path).expect("write");
    let back = read_ply_gaussian_cloud(&path).expect("read");

    let g0 = &cloud.gaussians[0];
    let g1 = &back.gaussians[0];
    assert!((g1.position - g0.position).norm() < 1e-4, "{g1:?}");
    assert!((g1.scale - g0.scale).norm() < 1e-4, "{:?}", g1.scale);
    assert!(
        (g1.rotation - g0.rotation).norm() < 1e-4,
        "{:?}",
        g1.rotation
    );
    let dc = SphericalHarmonics::from_dc(Vector3::new(0.25, 0.5, 0.75));
    assert!((g1.spherical_harmonics.dc() - dc.dc()).norm() < 1e-4);
}
