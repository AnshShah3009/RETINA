//! CPU Mathematical Correctness Tests
//!
//! These tests verify that CPU implementations produce mathematically correct results
//! by comparing against reference implementations and known analytical solutions.

use cv_core::{CpuStorage, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, MorphologyType, ThresholdType};
use cv_hal::cpu::CpuBackend;

fn get_cpu_backend() -> CpuBackend {
    CpuBackend::new().expect("CPU backend unavailable")
}

// ============================================================================
// Threshold Tests
// ============================================================================

#[test]
fn test_threshold_binary_basic() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![0.0, 50.0, 100.0, 150.0],
        TensorShape::new(1, 2, 2),
    )
    .unwrap();

    let result = cpu
        .threshold(&input, 75.0f32, 255.0, ThresholdType::Binary)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert_eq!(output[0], 0.0);
    assert_eq!(output[1], 0.0);
    assert_eq!(output[2], 255.0);
    assert_eq!(output[3], 255.0);
}

#[test]
fn test_threshold_trunc_basic() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![10.0, 50.0, 100.0, 150.0],
        TensorShape::new(1, 2, 2),
    )
    .unwrap();

    let result = cpu
        .threshold(&input, 75.0f32, 255.0, ThresholdType::Trunc)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert_eq!(output[0], 10.0);
    assert_eq!(output[1], 50.0);
    assert_eq!(output[2], 75.0);
    assert_eq!(output[3], 75.0);
}

// ============================================================================
// Gaussian Kernel Tests
// ============================================================================

#[test]
fn test_gaussian_kernel_sum_equals_one() {
    let sizes = [3, 5, 7, 9];
    let sigma = 1.0f32;

    for size in sizes {
        let kernel = cv_hal::cpu::gaussian_kernel_1d(sigma, size);

        let sum: f32 = kernel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Kernel size {} sum = {}",
            size,
            sum
        );
    }
}

#[test]
fn test_gaussian_kernel_symmetric() {
    let sigma = 2.0f32;
    let size = 7;

    let kernel = cv_hal::cpu::gaussian_kernel_1d(sigma, size);

    for i in 0..(size / 2) {
        assert!(
            (kernel[i] - kernel[size - 1 - i]).abs() < 1e-6,
            "Not symmetric at indices {} and {}",
            i,
            size - 1 - i
        );
    }
}

#[test]
fn test_gaussian_kernel_max_at_center() {
    let sigma = 1.5f32;
    let size = 5;

    let kernel = cv_hal::cpu::gaussian_kernel_1d(sigma, size);

    let center = size / 2;
    for (i, &val) in kernel.iter().enumerate() {
        assert!(
            val <= kernel[center],
            "Index {} value {} > center {} value {}",
            i,
            val,
            center,
            kernel[center]
        );
    }
}

// ============================================================================
// Resize Tests
// ============================================================================

#[test]
fn test_resize_preserves_value_range() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![0.0, 100.0, 50.0, 150.0],
        TensorShape::new(1, 2, 2),
    )
    .unwrap();

    let result = cpu.resize(&input, (4, 4)).unwrap();

    let output = result.as_slice().unwrap();

    for &val in output.iter() {
        assert!(val >= 0.0 && val <= 150.0, "Value {} out of range", val);
    }
}

#[test]
fn test_resize_4x4_to_2x2() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
        TensorShape::new(1, 4, 4),
    )
    .unwrap();

    let result = cpu.resize(&input, (2, 2)).unwrap();

    assert_eq!(result.as_slice().unwrap().len(), 4);
}

// ============================================================================
// Point Cloud Transform Tests
// ============================================================================

#[test]
fn test_pointcloud_transform_basic() {
    let cpu = get_cpu_backend();

    // 3 points with 4 components each
    let points = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        TensorShape::new(4, 3, 1),
    )
    .unwrap();

    // Identity transform
    let transform = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let result = cpu.pointcloud_transform(&points, &transform).unwrap();

    let output = result.as_slice().unwrap();
    // Identity should preserve values
    assert_eq!(output[0], 1.0);
    assert_eq!(output[4], 5.0);
    assert_eq!(output[8], 9.0);
}

// ============================================================================
// Color Conversion Tests (f32)
// ============================================================================

#[test]
fn test_color_rgb_to_gray_red() {
    let cpu = get_cpu_backend();

    let input =
        Tensor::<f32, CpuStorage<f32>>::from_vec(vec![255.0, 0.0, 0.0], TensorShape::new(3, 1, 1))
            .unwrap();

    let result = cpu
        .cvt_color(&input, cv_hal::context::ColorConversion::RgbToGray)
        .unwrap();

    let output = result.as_slice().unwrap();
    let expected = 0.299 * 255.0;
    assert!((output[0] - expected).abs() < 1.0);
}

#[test]
fn test_color_rgb_to_gray_white() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![255.0, 255.0, 255.0],
        TensorShape::new(3, 1, 1),
    )
    .unwrap();

    let result = cpu
        .cvt_color(&input, cv_hal::context::ColorConversion::RgbToGray)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert!(output[0] >= 250.0);
}

#[test]
fn test_color_rgb_to_gray_black() {
    let cpu = get_cpu_backend();

    let input =
        Tensor::<f32, CpuStorage<f32>>::from_vec(vec![0.0, 0.0, 0.0], TensorShape::new(3, 1, 1))
            .unwrap();

    let result = cpu
        .cvt_color(&input, cv_hal::context::ColorConversion::RgbToGray)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert!(output[0] <= 5.0);
}

// ============================================================================
// Sobel Tests
// ============================================================================

#[test]
fn test_sobel_uniform_image() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(vec![128.0; 9], TensorShape::new(1, 3, 3))
        .unwrap();

    let (gx, gy) = cpu.sobel(&input, 1, 1, 3).unwrap();

    let gx_data = gx.as_slice().unwrap();
    let gy_data = gy.as_slice().unwrap();

    for &gx_val in gx_data.iter().take(9) {
        assert!(gx_val.abs() < 1.0);
    }
    for &gy_val in gy_data.iter().take(9) {
        assert!(gy_val.abs() < 1.0);
    }
}

#[test]
fn test_sobel_horizontal_edge() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0],
        TensorShape::new(1, 3, 3),
    )
    .unwrap();

    let (_gx, gy) = cpu.sobel(&input, 1, 1, 3).unwrap();

    let gy_data = gy.as_slice().unwrap();
    assert!(gy_data[4].abs() > 50.0);
}

// ============================================================================
// Pyramid Tests
// ============================================================================

#[test]
fn test_pyramid_down_2x2_to_1x1() {
    let cpu = get_cpu_backend();

    let input = Tensor::<f32, CpuStorage<f32>>::from_vec(
        vec![100.0, 100.0, 100.0, 100.0],
        TensorShape::new(1, 2, 2),
    )
    .unwrap();

    let result = cpu.pyramid_down(&input).unwrap();

    let output = result.as_slice().unwrap();
    assert!((output[0] - 100.0).abs() < 1.0);
}

#[test]
fn test_pyramid_down_preserves_range() {
    let cpu = get_cpu_backend();

    let input_data: Vec<f32> = (0..16).map(|i| i as f32 * 16.0).collect();
    let input =
        Tensor::<f32, CpuStorage<f32>>::from_vec(input_data, TensorShape::new(1, 4, 4)).unwrap();

    let result = cpu.pyramid_down(&input).unwrap();

    let output = result.as_slice().unwrap();
    for &val in output.iter().take(4) {
        assert!(val >= 0.0 && val <= 255.0, "Value {} out of range", val);
    }
}

// ============================================================================
// Morphology Tests
// ============================================================================

#[test]
fn test_morphology_erode_white_image() {
    let cpu = get_cpu_backend();

    let input =
        Tensor::<u8, CpuStorage<u8>>::from_vec(vec![255; 9], TensorShape::new(1, 3, 3)).unwrap();

    let kernel =
        Tensor::<u8, CpuStorage<u8>>::from_vec(vec![1; 9], TensorShape::new(1, 3, 3)).unwrap();

    let result = cpu
        .morphology(&input, MorphologyType::Erode, &kernel, 1)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert_eq!(output.iter().filter(|&&v| v == 255).count(), 9);
}

#[test]
fn test_morphology_dilate_black_image() {
    let cpu = get_cpu_backend();

    let input =
        Tensor::<u8, CpuStorage<u8>>::from_vec(vec![0; 9], TensorShape::new(1, 3, 3)).unwrap();

    let kernel =
        Tensor::<u8, CpuStorage<u8>>::from_vec(vec![1; 9], TensorShape::new(1, 3, 3)).unwrap();

    let result = cpu
        .morphology(&input, MorphologyType::Dilate, &kernel, 1)
        .unwrap();

    let output = result.as_slice().unwrap();
    assert_eq!(output.iter().filter(|&&v| v == 0).count(), 9);
}

// ============================================================================
// Subtract Tests
// ============================================================================

#[test]
fn test_subtract_identity() {
    let cpu = get_cpu_backend();

    let a =
        Tensor::<f32, CpuStorage<f32>>::from_vec(vec![10.0, 20.0, 30.0], TensorShape::new(1, 1, 3))
            .unwrap();

    let b =
        Tensor::<f32, CpuStorage<f32>>::from_vec(vec![0.0, 0.0, 0.0], TensorShape::new(1, 1, 3))
            .unwrap();

    let result = cpu.subtract(&a, &b).unwrap();

    let output = result.as_slice().unwrap();
    assert_eq!(output[0], 10.0);
    assert_eq!(output[1], 20.0);
    assert_eq!(output[2], 30.0);
}

// ============================================================================
// Single Pixel Tests
// ============================================================================

#[test]
fn test_single_pixel_threshold() {
    let cpu = get_cpu_backend();

    let input =
        Tensor::<f32, CpuStorage<f32>>::from_vec(vec![100.0], TensorShape::new(1, 1, 1)).unwrap();

    let result = cpu
        .threshold(&input, 50.0f32, 255.0, ThresholdType::Binary)
        .unwrap();
    assert_eq!(result.as_slice().unwrap()[0], 255.0);

    let result = cpu
        .threshold(&input, 150.0f32, 255.0, ThresholdType::Binary)
        .unwrap();
    assert_eq!(result.as_slice().unwrap()[0], 0.0);
}
