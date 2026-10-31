//! `SphericalHarmonics::eval` must use the 3DGS reference's real SH basis
//! (bands 1-3, its order, signs and constants) per colour channel. It used to
//! mix basis functions across channels with no constants and no band 3, so
//! view-dependent colour from a loaded `f_rest` was wrong. Reference values
//! from the reference `eval_sh` formula evaluated in numpy.

use cv_rendering::gaussian_splatting::SphericalHarmonics;
use nalgebra::Vector3;

#[test]
fn degree_three_matches_the_3dgs_reference() {
    let coeffs: Vec<f32> = (0..16)
        .flat_map(|k| (0..3).map(move |c| ((k * 7 + c * 3) % 11) as f32 / 10.0 - 0.5))
        .collect();
    let sh = SphericalHarmonics { coeffs, degree: 3 };
    let out = sh.eval(Vector3::new(0.3, -0.5, 0.8));
    let want = [
        -0.8259056471306162f32,
        -0.3170826125748017,
        0.14489843132607078,
    ];
    for c in 0..3 {
        assert!(
            (out[c] - want[c]).abs() < 1e-5,
            "channel {c}: {} vs {}",
            out[c],
            want[c]
        );
    }
}

#[test]
fn band_one_is_minus_c1_y_for_the_first_coefficient() {
    let mut sh = SphericalHarmonics::new(1);
    sh.coeffs[3] = 1.0; // R channel, basis 1 (-C1 * y)
    let out = sh.eval(Vector3::new(0.0, 1.0, 0.0));
    assert!((out.x + 0.488_602_5).abs() < 1e-6, "{out:?}");
    assert_eq!((out.y, out.z), (0.0, 0.0), "channels must not mix");
}
