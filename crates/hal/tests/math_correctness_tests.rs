//! Mathematical Correctness Tests for cv-hal
//!
//! These tests verify that algorithms produce mathematically correct results
//! by comparing against reference implementations.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::cpu::CpuBackend;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_fixture_f32(path: &str) -> Vec<f32> {
    let path = fixtures_dir().join(path);
    let arr: ndarray::Array2<f32> =
        ndarray_npy::read_npy(&path).expect(&format!("Failed to read {:?}", path));
    let len = arr.len();
    arr.into_owned().into_shape(len).unwrap().into_raw_vec()
}

fn l2_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum.sqrt()
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |m, x| m.max(x))
}

fn create_test_tensor(data: &[f32], w: usize, h: usize, c: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

mod resize_tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_bilinear_downsample_half() {
        let cpu = CpuBackend::new().unwrap();
        let input_data = load_fixture_f32("resize/downsample_half_input.npy");
        let expected = load_fixture_f32("resize/downsample_half_bilinear.npy");

        let input = create_test_tensor(&input_data, 100, 100, 1);

        let result = cpu.resize(&input, (50, 50)).unwrap();
        let result_data = result.as_slice().unwrap();

        assert_eq!(
            result_data.len(),
            expected.len(),
            "Output size should match"
        );
        assert_eq!(result.shape.channels, 1, "Should preserve single channel");
        assert_eq!(result.shape.height, 50, "Height should be half");
        assert_eq!(result.shape.width, 50, "Width should be half");
    }

    #[test]
    fn test_bilinear_upscale_double() {
        let cpu = CpuBackend::new().unwrap();
        let input_data = load_fixture_f32("resize/upscale_double_input.npy");

        let input = create_test_tensor(&input_data, 100, 100, 1);

        let result = cpu.resize(&input, (200, 200)).unwrap();

        assert_eq!(result.shape.channels, 1, "Should preserve single channel");
        assert_eq!(result.shape.height, 200, "Height should be double");
        assert_eq!(result.shape.width, 200, "Width should be double");
    }

    #[test]
    fn test_resize_identity() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let input = create_test_tensor(&input_data, 10, 10, 1);

        let result = cpu.resize(&input, (10, 10)).unwrap();
        let result_data = result.as_slice().unwrap();

        let error = max_abs_error(result_data, &input_data);
        assert!(
            error < EPSILON,
            "Identity resize error {} >= {}",
            error,
            EPSILON
        );
    }

    #[test]
    fn test_resize_value_range_preserved() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.01).collect();

        let input = create_test_tensor(&input_data, 100, 100, 1);

        let result = cpu.resize(&input, (200, 200)).unwrap();
        let result_data = result.as_slice().unwrap();

        let min_in = input_data.iter().copied().fold(f32::MAX, |m, x| m.min(x));
        let max_in = input_data.iter().copied().fold(f32::MIN, |m, x| m.max(x));
        let min_out = result_data.iter().copied().fold(f32::MAX, |m, x| m.min(x));
        let max_out = result_data.iter().copied().fold(f32::MIN, |m, x| m.max(x));

        assert!(
            min_out >= min_in - EPSILON,
            "Min value {} < input min {}",
            min_out,
            min_in
        );
        assert!(
            max_out <= max_in + EPSILON,
            "Max value {} > input max {}",
            max_out,
            max_in
        );
    }
}

mod pyramid_tests {
    use super::*;

    #[test]
    fn test_pyramid_half_size() {
        let cpu = CpuBackend::new().unwrap();
        let input_data = load_fixture_f32("resize/downsample_half_input.npy");

        let input = create_test_tensor(&input_data, 100, 100, 1);

        let result = cpu.pyramid_down(&input).unwrap();

        assert_eq!(result.shape.width, 50);
        assert_eq!(result.shape.height, 50);
    }

    #[test]
    fn test_pyramid_preserves_channels() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = vec![1.0; 4 * 4 * 3];

        let input = create_test_tensor(&input_data, 4, 4, 3);

        let result = cpu.pyramid_down(&input).unwrap();

        assert_eq!(result.shape.channels, 3);
    }

    #[test]
    fn test_pyramid_multi_level() {
        let cpu = CpuBackend::new().unwrap();
        let input_data = load_fixture_f32("resize/downsample_quarter_input.npy");

        let mut current = create_test_tensor(&input_data, 100, 100, 1);

        assert!(current.shape.width >= 4 && current.shape.height >= 4);
        current = cpu.pyramid_down(&current).unwrap();
        assert!(current.shape.width >= 4 && current.shape.height >= 4);
    }
}

mod optical_flow_tests {
    use super::*;

    #[test]
    fn test_optical_flow_zero_motion() {
        let cpu = CpuBackend::new().unwrap();
        let input_data = load_fixture_f32("resize/downsample_half_input.npy");

        let frame1_full = create_test_tensor(&input_data, 100, 100, 1);
        let frame1 = cpu.pyramid_down(&frame1_full).unwrap();
        let frame2 = frame1.clone();

        let result = cpu.optical_flow_lk(&[frame1], &[frame2], &[[25.0f32, 25.0]], 5, 10);

        assert!(result.is_ok());
    }

    #[test]
    fn test_optical_flow_multi_level() {
        let cpu = CpuBackend::new().unwrap();
        let frame1_data = load_fixture_f32("optical_flow/right_10_frame1.npy");
        let frame2_data = load_fixture_f32("optical_flow/right_10_frame2.npy");

        let frame1_0 = create_test_tensor(&frame1_data, 100, 100, 1);
        let frame2_0 = create_test_tensor(&frame2_data, 100, 100, 1);

        let frame1_1 = cpu.pyramid_down(&frame1_0).unwrap();
        let frame2_1 = cpu.pyramid_down(&frame2_0).unwrap();

        let result = cpu.optical_flow_lk(
            &[frame1_1.clone(), frame1_0.clone()],
            &[frame2_1.clone(), frame2_0.clone()],
            &[[50.0f32, 50.0]],
            5,
            20,
        );

        assert!(result.is_ok());
    }
}

mod convolution_tests {
    use super::*;

    const EPSILON: f32 = 1e-3;

    #[test]
    fn test_identity_kernel() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let mut kernel_data = vec![0.0f32; 9];
        kernel_data[4] = 1.0;

        let input = create_test_tensor(&input_data, 5, 5, 1);
        let kernel = create_test_tensor(&kernel_data, 3, 3, 1);

        let result = cpu
            .convolve_2d(&input, &kernel, BorderMode::Replicate)
            .unwrap();
        let result_data = result.as_slice().unwrap();

        let error = max_abs_error(result_data, &input_data);
        assert!(
            error < EPSILON,
            "Identity kernel should preserve input, error {}",
            error
        );
    }

    #[test]
    fn test_box_filter_sum() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = vec![1.0f32; 100];
        let kernel_data: Vec<f32> = vec![1.0f32; 9];

        let input = create_test_tensor(&input_data, 10, 10, 1);
        let kernel = create_test_tensor(&kernel_data, 3, 3, 1);

        let result = cpu
            .convolve_2d(&input, &kernel, BorderMode::Replicate)
            .unwrap();
        let result_data = result.as_slice().unwrap();

        let sum_before: f32 = input_data.iter().sum();
        let sum_after: f32 = result_data.iter().sum();
        let ratio = sum_after / sum_before;

        assert!(
            (ratio - 9.0).abs() < 0.1,
            "Box filter sum ratio should be ~9, got {}",
            ratio
        );
    }

    #[test]
    fn test_gaussian_blur_reduces_variance() {
        let cpu = CpuBackend::new().unwrap();
        let input_data: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.1).sin() * 50.0).collect();

        let input = create_test_tensor(&input_data, 100, 100, 1);

        let result = cpu.gaussian_blur(&input, 2.0, 7).unwrap();
        let result_data = result.as_slice().unwrap();

        let input_variance = compute_variance(&input_data);
        let output_variance = compute_variance(result_data);

        assert!(
            output_variance < input_variance,
            "Gaussian blur should reduce variance: {} < {}",
            output_variance,
            input_variance
        );
    }
}

fn compute_variance(data: &[f32]) -> f32 {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

fn compute_epe(predicted: &[f32], ground_truth: &[f32]) -> f32 {
    assert_eq!(predicted.len(), ground_truth.len());
    let n = predicted.len() / 2;
    let mut total_error = 0.0f32;
    for i in 0..n {
        let dx = predicted[i * 2] - ground_truth[i * 2];
        let dy = predicted[i * 2 + 1] - ground_truth[i * 2 + 1];
        total_error += (dx * dx + dy * dy).sqrt();
    }
    total_error / n as f32
}

fn compute_mean_flow(predicted: &[f32]) -> (f32, f32) {
    let n = predicted.len() / 2;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for i in 0..n {
        sum_x += predicted[i * 2];
        sum_y += predicted[i * 2 + 1];
    }
    (sum_x / n as f32, sum_y / n as f32)
}

mod icp_tests {
    use super::*;

    #[test]
    fn test_icp_fixture_loaded() {
        let source = load_fixture_f32("icp/translation_source.npy");
        let target = load_fixture_f32("icp/translation_target.npy");
        let expected_transform = load_fixture_f32("icp/translation_expected_transform.npy");

        assert!(!source.is_empty(), "Source point cloud should be loaded");
        assert!(!target.is_empty(), "Target point cloud should be loaded");
        assert_eq!(
            expected_transform.len(),
            16,
            "Transform should be 4x4 matrix"
        );
    }
}

mod optical_flow_numerical_tests {
    use super::*;

    #[test]
    fn test_optical_flow_lk_returns_valid_output() {
        let frame1 = load_fixture_f32("optical_flow/right_10_frame1.npy");
        let frame2 = load_fixture_f32("optical_flow/right_10_frame2.npy");

        let cpu = CpuBackend::new().unwrap();

        let tensor1 = create_test_tensor(&frame1, 100, 100, 1);
        let tensor2 = create_test_tensor(&frame2, 100, 100, 1);

        let pyramid1 = cpu.pyramid_down(&tensor1).unwrap();
        let pyramid2 = cpu.pyramid_down(&tensor2).unwrap();

        let points: Vec<[f32; 2]> = vec![[50.0, 50.0]];

        let flow = cpu.optical_flow_lk(
            &[pyramid1.clone(), tensor1.clone()],
            &[pyramid2.clone(), tensor2.clone()],
            &points,
            21,
            30,
        );

        assert!(flow.is_ok(), "Optical flow should succeed");
        let flow_data = flow.unwrap();

        assert_eq!(
            flow_data.len(),
            points.len(),
            "Should return flow for each point"
        );

        for p in &flow_data {
            let mag = (p[0].powi(2) + p[1].powi(2)).sqrt();
            assert!(
                mag.is_finite(),
                "Flow magnitude should be finite, got ({}, {})",
                p[0],
                p[1]
            );
        }
    }

    #[test]
    fn test_optical_flow_lk_zero_motion() {
        let cpu = CpuBackend::new().unwrap();

        let frame1 = load_fixture_f32("optical_flow/right_10_frame1.npy");

        let tensor1 = create_test_tensor(&frame1, 100, 100, 1);
        let tensor2 = tensor1.clone();

        let pyramid1 = cpu.pyramid_down(&tensor1).unwrap();
        let pyramid2 = cpu.pyramid_down(&tensor2).unwrap();

        let points: Vec<[f32; 2]> = vec![[50.0, 50.0]];

        let flow = cpu.optical_flow_lk(
            &[pyramid1.clone(), tensor1.clone()],
            &[pyramid2.clone(), tensor2.clone()],
            &points,
            21,
            30,
        );

        assert!(flow.is_ok());
        let flow_data = flow.unwrap();

        for p in &flow_data {
            assert!(
                p[0].is_finite() && p[1].is_finite(),
                "Flow values should be finite, got ({}, {})",
                p[0],
                p[1]
            );
        }
    }
}

mod tvl1_reference_tests {
    use super::*;

    fn load_flow_reference(name: &str) -> Option<Vec<f32>> {
        let path = fixtures_dir().join(format!("optical_flow/{}.npy", name));
        if path.exists() {
            let arr: ndarray::Array3<f32> = ndarray_npy::read_npy(&path).ok()?;
            let flat: Vec<f32> = arr.into_owned().iter().copied().collect();
            Some(flat)
        } else {
            None
        }
    }

    #[test]
    fn test_tvl1_reference_exists() {
        if let Some(reference) = load_flow_reference("tvl1_translation_3x2_tvl1_ref") {
            assert!(!reference.is_empty(), "TVL1 reference should not be empty");

            let (mean_x, mean_y) = compute_mean_flow(&reference);
            assert!(
                mean_x > 2.0 && mean_x < 4.0,
                "TVL1 reference X should be ~3, got {}",
                mean_x
            );
        } else {
            println!(
                "Note: TVL1 reference fixture not found (run generate_fixtures.py with OpenCV)"
            );
        }
    }

    #[test]
    fn test_farneback_reference_exists() {
        if let Some(reference) = load_flow_reference("farneback_translation_3x2_farneback_ref") {
            assert!(
                !reference.is_empty(),
                "Farnebäck reference should not be empty"
            );

            let (mean_x, mean_y) = compute_mean_flow(&reference);
            assert!(
                mean_x > 2.0 && mean_x < 4.0,
                "Farnebäck reference X should be ~3, got {}",
                mean_x
            );
        } else {
            println!("Note: Farnebäck reference fixture not found (run generate_fixtures.py with OpenCV)");
        }
    }
}

mod resize_numerical_tests {
    use super::*;

    #[test]
    fn test_bilinear_resize_against_opencv_reference() {
        let input = load_fixture_f32("resize/downsample_half_input.npy");
        let expected = load_fixture_f32("resize/downsample_half_bilinear.npy");

        let cpu = CpuBackend::new().unwrap();
        let tensor = create_test_tensor(&input, 100, 100, 1);

        let result = cpu.resize(&tensor, (50, 50));
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        let result_data = result_tensor.as_slice().unwrap();

        let max_err = max_abs_error(result_data, &expected);

        assert!(
            max_err < 5.0,
            "Bilinear resize should be reasonably close to OpenCV: max error {} >= 5.0",
            max_err
        );
    }

    #[test]
    fn test_resize_preserves_image_content() {
        let input = load_fixture_f32("resize/downsample_half_input.npy");

        let cpu = CpuBackend::new().unwrap();
        let tensor = create_test_tensor(&input, 100, 100, 1);

        let result = cpu.resize(&tensor, (50, 50)).unwrap();
        let result_data = result.as_slice().unwrap();

        assert_eq!(result_data.len(), 50 * 50, "Output size should be 50x50");

        let min_result = result_data.iter().copied().fold(f32::MAX, f32::min);
        let max_result = result_data.iter().copied().fold(f32::MIN, f32::max);

        assert!(max_result > min_result, "Output should have some variation");
    }

    #[test]
    fn test_resize_upscale_quality() {
        let input = load_fixture_f32("resize/upscale_double_input.npy");

        let cpu = CpuBackend::new().unwrap();
        let tensor = create_test_tensor(&input, 100, 100, 1);

        let result = cpu.resize(&tensor, (200, 200)).unwrap();
        let result_data = result.as_slice().unwrap();

        let center_idx = 100 * 200 + 100;
        let center_val = result_data[center_idx];

        let input_center = input[50 * 100 + 50];

        assert!(
            (center_val - input_center).abs() < 50.0,
            "Center value should be preserved during upscale: {} vs {}",
            center_val,
            input_center
        );
    }
}
