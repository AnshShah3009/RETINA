use cv_core::float::Float;
use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

fn test_convolution_for_type<T: Float + std::ops::Neg<Output = T> + 'static>() {
    let cpu = CpuBackend::new().unwrap();

    let mut data = vec![T::ZERO; 9];
    data[0] = T::ONE;
    data[1] = T::ZERO;
    data[2] = T::ZERO;
    data[3] = T::ONE;
    data[4] = T::ZERO;
    data[5] = T::ZERO;
    data[6] = T::ONE;
    data[7] = T::ZERO;
    data[8] = T::ZERO;

    let input: Tensor<T, CpuStorage<T>> =
        Tensor::from_vec(data, cv_core::TensorShape::new(1, 3, 3)).unwrap();

    let kernel_data = vec![
        T::ONE,
        T::ZERO,
        T::ONE.neg(),
        T::ONE,
        T::ZERO,
        T::ONE.neg(),
        T::ONE,
        T::ZERO,
        T::ONE.neg(),
    ];
    let kernel: Tensor<T, CpuStorage<T>> =
        Tensor::from_vec(kernel_data, cv_core::TensorShape::new(1, 3, 3)).unwrap();

    let border = cv_hal::context::BorderMode::Replicate;
    let result = cpu.convolve_2d(&input, &kernel, border).unwrap();
    let result_data = result.as_slice().unwrap();

    assert!(result_data[4].to_f32() > 0.0);
}

#[test]
fn test_convolution_f32() {
    test_convolution_for_type::<f32>();
}

#[test]
fn test_convolution_f64() {
    test_convolution_for_type::<f64>();
}

#[cfg(feature = "half-precision")]
#[test]
fn test_convolution_f16() {
    test_convolution_for_type::<half::f16>();
}

#[cfg(feature = "half-precision")]
#[test]
fn test_convolution_bf16() {
    test_convolution_for_type::<half::bf16>();
}
