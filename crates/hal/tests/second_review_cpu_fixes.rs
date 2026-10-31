//! Second-review CPU HAL fixes (GAPS g-h3, g-h9, g-h10).

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{ComputeContext, DiffusivityType, StereoMatchMethod, StereoMatchParams};
use cv_hal::cpu::CpuBackend;

fn cpu() -> CpuBackend {
    CpuBackend::new().expect("CPU backend unavailable")
}

/// g-h3: on a mostly flat image the zero gradients used to put the 70th
/// percentile at k = 0, and one Perona-Malik step then wrote NaN over almost
/// the whole image (4032/4096 pixels).
#[test]
fn akaze_k_ignores_zero_gradients_and_diffusion_stays_finite() {
    let (w, h) = (64usize, 64usize);
    let mut data = vec![0.0f32; w * h];
    for y in 28..36 {
        for x in 28..36 {
            data[y * w + x] = 1.0;
        }
    }
    let img: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let backend = cpu();
    let k = backend.akaze_contrast_k(&img).unwrap();
    assert!(k > 0.0, "k = {k}");
    let out = backend
        .akaze_diffusion(&img, k, 0.25, DiffusivityType::PeronaMalik2)
        .unwrap();
    let nan = out
        .as_slice()
        .unwrap()
        .iter()
        .filter(|v| !v.is_finite())
        .count();
    assert_eq!(nan, 0, "{nan} non-finite pixels after one diffusion step");

    // No gradient at all: OpenCV's fallback, not 0.
    let flat: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(vec![0.5; w * h], TensorShape::new(1, h, w)).unwrap();
    assert_eq!(backend.akaze_contrast_k(&flat).unwrap(), 0.03);
}

/// g-h9: a pixel where no disparity in the range can be tested is
/// "no match" (-1), not disparity 0.
#[test]
fn stereo_pixels_with_no_testable_disparity_are_no_match() {
    let (w, h) = (32usize, 9usize);
    let data: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32).collect();
    let l: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data.clone(), TensorShape::new(1, h, w)).unwrap();
    let r: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let params = StereoMatchParams {
        method: StereoMatchMethod::BlockMatching,
        min_disparity: 8,
        num_disparities: 8,
        block_size: 5,
    };
    let out: Tensor<f32, CpuStorage<f32>> = cpu().stereo_match(&l, &r, &params).unwrap();
    let out = out.as_slice().unwrap();
    // Row 4, x in 2..=9: every candidate x - d lies left of the image.
    for x in 2..10 {
        assert_eq!(out[4 * w + x], -1.0, "x = {x}");
    }
}

/// g-h10: an empty train set must give no matches, whatever the ratio.
#[test]
fn matching_against_nothing_matches_nothing() {
    let q: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(vec![0xAA; 32], TensorShape::new(1, 1, 32)).unwrap();
    let t: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(Vec::new(), TensorShape::new(1, 0, 32)).unwrap();
    let m = cpu().match_descriptors(&q, &t, 1.0).unwrap();
    assert!(m.matches.is_empty(), "{:?}", m.matches);
}

/// g-h5/g-h6: one straight segment is one line, strongest first.
#[test]
fn hough_lines_suppresses_neighbouring_bins() {
    let (w, h) = (64usize, 64usize);
    let mut data = vec![0.0f32; w * h];
    for x in 0..w {
        data[32 * w + x] = 255.0;
    }
    let img: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let lines = cpu()
        .hough_lines(&img, 1.0, std::f32::consts::PI / 180.0, 40)
        .unwrap();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0].score, 64);
    assert!((lines[0].theta - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
    // cv2.HoughLines on the same image: [[32.0, 1.5707964]], 64 votes.
    assert!((lines[0].rho - 32.0).abs() < 1e-4, "{:?}", lines[0]);
}

/// g-h12: AKAZE first derivatives are Scharr with unit gain - a unit ramp
/// has derivative 1 (the [1, 3, 1] taps gave 0.3125).
#[test]
fn akaze_first_derivatives_have_unit_gain() {
    let (w, h) = (16usize, 16usize);
    let ramp: Vec<f32> = (0..w * h)
        .map(|i| (i % w) as f32 + 2.0 * (i / w) as f32)
        .collect();
    let img: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(ramp, TensorShape::new(1, h, w)).unwrap();
    let (lx, ly, _) = cpu().akaze_derivatives(&img).unwrap();
    let (lx, ly) = (lx.as_slice().unwrap(), ly.as_slice().unwrap());
    let c = 8 * w + 8;
    assert!((lx[c] - 1.0).abs() < 1e-6, "d/dx of x is {}", lx[c]);
    assert!((ly[c] - 2.0).abs() < 1e-6, "d/dy of 2y is {}", ly[c]);
}

/// g-h11: SIFT descriptors are renormalised after the 0.2 clamp, so every
/// descriptor has L2 norm 512 (up to rounding), as OpenCV's do.
#[test]
fn sift_descriptors_are_renormalised_after_clamping() {
    let (w, h) = (64usize, 64usize);
    // A vertical step edge: every gradient falls in one orientation bin, so
    // the 0.2 clamp bites hard and only renormalisation restores the length.
    let data: Vec<f32> = (0..w * h)
        .map(|i| if i % w < 32 { 0.0 } else { 1.0 })
        .collect();
    let img: Tensor<f32, CpuStorage<f32>> =
        Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let mut kps = cv_core::KeyPoints::new();
    kps.push(
        cv_core::KeyPoint::new(32.0, 32.0)
            .with_size(8.0)
            .with_angle(0.0),
    );
    let d = cpu().compute_sift_descriptors(&img, &kps).unwrap();
    let norm = d.descriptors[0]
        .data
        .iter()
        .map(|&v| (v as f32).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!((norm - 512.0).abs() < 8.0, "descriptor norm {norm}");
}
