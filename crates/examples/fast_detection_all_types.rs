use cv_core::float::Float;
use cv_core::storage::CpuStorage;
/// Example demonstrating FAST detection with multiple float precision formats
use cv_core::tensor::Tensor;
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

fn detect_with_precision<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    name: &str,
    cpu: &CpuBackend,
    threshold: T,
) {
    let width = 640;
    let height = 480;

    // Create a dummy image. Real use cases will load a file and map over cast variants
    let mut data = vec![T::ZERO; width * height];
    // Add point for detection
    data[120 * width + 120] = T::ONE;
    let tensor: Tensor<T, CpuStorage<T>> =
        Tensor::from_vec(data, cv_core::TensorShape::new(1, height, width)).unwrap();

    // Call FAST feature detection (Generic parameter T is inferred from tensor)
    let score_map = cpu.fast_detect(&tensor, threshold, true).unwrap();

    // We just return the score map since extract_keypoints is now handled by cv-features
    println!(
        "Found FAST score map of size {} using precision format {}",
        score_map.shape.len(),
        name
    );
}

fn main() {
    let cpu = CpuBackend::new().unwrap();

    // Default precision
    detect_with_precision::<f32>("f32", &cpu, 20.0);

    // High precision
    detect_with_precision::<f64>("f64", &cpu, 20.0);

    // Half-precision variants are typically hidden behind flags
    #[cfg(feature = "half-precision")]
    {
        detect_with_precision::<half::f16>("f16", &cpu, half::f16::from_f32(20.0));
        detect_with_precision::<half::bf16>("bf16", &cpu, half::bf16::from_f32(20.0));
    }
}
