//! Edge case tests for cv-hal
//!
//! Tests boundary conditions, empty/zero inputs, overflow scenarios,
//! and other edge cases that could cause functional bugs.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::cpu::CpuBackend;
use cv_hal::BackendType;

fn create_test_tensor_f32(
    data: &[f32],
    w: usize,
    h: usize,
    c: usize,
) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data.to_vec(), cv_core::TensorShape::new(c, h, w)).unwrap()
}

fn create_test_tensor_u8(data: &[u8], w: usize, h: usize, c: usize) -> Tensor<u8, CpuStorage<u8>> {
    Tensor::from_vec(data.to_vec(), cv_core::TensorShape::new(c, h, w)).unwrap()
}

mod convolve_tests {
    use super::*;

    #[test]
    fn test_convolve_single_pixel() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0], 1, 1, 1);
        let kernel = create_test_tensor_f32(&[1.0], 1, 1, 1);

        let result = cpu.convolve_2d(&input, &kernel, BorderMode::Replicate);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convolve_3x3_input() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0; 9], 3, 3, 1);
        let kernel = create_test_tensor_f32(&[1.0; 9], 3, 3, 1);

        let result = cpu.convolve_2d(&input, &kernel, BorderMode::Replicate);
        assert!(result.is_ok());
    }
}

mod resize_tests {
    use super::*;

    #[test]
    fn test_resize_1x1() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0], 1, 1, 1);

        let result = cpu.resize(&input, (1, 1));
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_multichannel() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0; 12], 2, 2, 3);

        let result = cpu.resize(&input, (4, 4));
        assert!(result.is_ok());
    }
}

mod pyramid_tests {
    use super::*;

    #[test]
    fn test_pyramid_down_2x2() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0; 4], 2, 2, 1);

        let result = cpu.pyramid_down(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pyramid_down_3x3() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0; 9], 3, 3, 1);

        let result = cpu.pyramid_down(&input);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape.width, 2);
        assert_eq!(output.shape.height, 2);
    }

    #[test]
    fn test_pyramid_down_1x1_fails() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0], 1, 1, 1);

        let result = cpu.pyramid_down(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_pyramid_down_multichannel() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[1.0; 12], 2, 2, 3);

        let result = cpu.pyramid_down(&input);
        assert!(result.is_ok());
    }
}

mod optical_flow_tests {
    use super::*;

    #[test]
    fn test_optical_flow_single_level() {
        let cpu = CpuBackend::new().unwrap();
        let prev = create_test_tensor_f32(&[1.0; 100], 10, 10, 1);
        let next = create_test_tensor_f32(&[1.0; 100], 10, 10, 1);
        let points: [[f32; 2]; 1] = [[5.0, 5.0]];

        let result = cpu.optical_flow_lk::<f32, CpuStorage<f32>>(&[prev], &[next], &points, 3, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optical_flow_empty_pyramid() {
        let cpu = CpuBackend::new().unwrap();
        let points: [[f32; 2]; 1] = [[5.0, 5.0]];

        let result = cpu.optical_flow_lk::<f32, CpuStorage<f32>>(&[], &[], &points, 3, 10);
        assert!(result.is_err());
    }
}

mod hough_tests {
    use super::*;

    #[test]
    fn test_hough_lines_single_edge_pixel() {
        let cpu = CpuBackend::new().unwrap();
        let mut data = vec![0.0; 100];
        data[50] = 1.0;
        let input = create_test_tensor_f32(&data, 10, 10, 1);

        let result = cpu.hough_lines(&input, 1.0, 0.01, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hough_lines_empty_image() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_test_tensor_f32(&[0.0; 100], 10, 10, 1);

        let result = cpu.hough_lines(&input, 1.0, 0.01, 1);
        assert!(result.is_ok());
        let lines = result.unwrap();
        assert!(lines.is_empty());
    }
}

mod device_tests {
    use super::*;

    #[test]
    fn test_cpu_device_id_unique() {
        let cpu1 = CpuBackend::new().unwrap();
        let cpu2 = CpuBackend::new().unwrap();

        let id1 = cpu1.device_id();
        let id2 = cpu2.device_id();

        assert_ne!(id1, id2, "Multiple CPU backends should have unique IDs");
    }

    #[test]
    fn test_backend_type() {
        let cpu = CpuBackend::new().unwrap();
        assert_eq!(cpu.backend_type(), BackendType::Cpu);
    }
}
