//! SIFT's Gaussian scale space must have the blur it claims.
//!
//! Lowe / OpenCV (`buildGaussianPyramid`): within an octave layer `i` has
//! total blur `sigma * k^i`, `k = 2^(1/n_layers)`, built by incremental
//! `GaussianBlur`s whose kernel is `cvRound(sigma * 8 + 1) | 1` taps for a
//! float image; the next octave's base is layer `n_layers` decimated by 2
//! (`resize(.., INTER_NEAREST)`), with no further blur.
//!
//! The port blurred every layer with a fixed 7-tap kernel. At the top of an
//! octave the incremental sigma is 3.1, so 7 taps cover +-1 sigma and the
//! normalised remainder is a much narrower filter. Measured on an impulse
//! (sigma of each layer, nominal in brackets): 2.003 (2.016), 2.465 (2.540),
//! 2.951 (3.200), 3.434 (4.032), 3.898 (5.080). Each octave base was then
//! made by a bilinear 2x resize (a 2x2 average) and blurred by `sigma`
//! again, so octave 1 started at ~2.1 px instead of 1.6. Keypoint scales
//! and DoG responses were therefore assigned to the wrong level.

use cv_core::{CpuTensor, TensorShape, storage::Storage};
use cv_features::Sift;
use cv_hal::compute::ComputeDevice;
use cv_hal::cpu::CpuBackend;

const N: usize = 129;

fn impulse() -> CpuTensor<u8> {
    let mut data = vec![0u8; N * N];
    data[(N / 2) * N + N / 2] = 255;
    CpuTensor::from_vec(data, TensorShape::new(1, N, N)).unwrap()
}

/// Second moment of a blurred impulse along x, i.e. the variance of the
/// cumulative blur kernel.
fn variance_x(v: &[f32], w: usize, h: usize) -> f64 {
    let c = (w / 2) as f64;
    let (mut s, mut m2) = (0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let p = v[y * w + x] as f64;
            s += p;
            m2 += p * (x as f64 - c) * (x as f64 - c);
        }
    }
    m2 / s
}

#[test]
fn layer_blur_grows_by_k_per_layer() {
    let cpu = CpuBackend::new().unwrap();
    let ctx = ComputeDevice::Cpu(&cpu);
    let sift = Sift {
        n_octaves: 2,
        ..Sift::default()
    };
    let pyr = sift.build_scale_space(&ctx, &impulse()).unwrap();
    let k = 2f64.powf(1.0 / sift.n_layers as f64);
    let s0 = sift.sigma as f64;
    let layers = &pyr[0];
    let v0 = variance_x(layers[0].storage.as_slice().unwrap(), N, N);
    let mut report = Vec::new();
    let mut ok = true;
    for (i, layer) in layers.iter().enumerate().skip(1) {
        // Blur added on top of layer 0: sigma_i^2 - sigma_0^2.
        let got = variance_x(layer.storage.as_slice().unwrap(), N, N) - v0;
        let want = s0 * s0 * (k.powi(2 * i as i32) - 1.0);
        ok &= (got - want).abs() <= 0.01 * want;
        report.push(format!(
            "layer {i}: sigma {:.3} want {:.3}",
            (s0 * s0 + got).sqrt(),
            (s0 * s0 + want).sqrt()
        ));
    }
    assert!(
        ok,
        "scale-space blur off by more than 1%:\n{}",
        report.join("\n")
    );
}

#[test]
fn octave_base_is_decimated_layer_without_extra_blur() {
    let cpu = CpuBackend::new().unwrap();
    let ctx = ComputeDevice::Cpu(&cpu);
    let sift = Sift {
        n_octaves: 2,
        ..Sift::default()
    };
    let pyr = sift.build_scale_space(&ctx, &impulse()).unwrap();
    let src = pyr[0][sift.n_layers].storage.as_slice().unwrap();
    let base = &pyr[1][0];
    let (h, w) = base.shape.hw();
    assert_eq!((h, w), (N / 2, N / 2));
    let base = base.storage.as_slice().unwrap();
    for y in 0..h {
        for x in 0..w {
            let want = src[(2 * y) * N + 2 * x];
            assert_eq!(
                base[y * w + x],
                want,
                "octave 1 base at ({x}, {y}) is not octave 0 layer {} at ({}, {})",
                sift.n_layers,
                2 * x,
                2 * y
            );
        }
    }
}
