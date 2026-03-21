use cv_core::tensor::{CpuTensor, TensorShape};
use cv_photo::{fast_nl_means_denoising, inpaint_telea};
use std::time::Instant;

#[test]
fn profile_photo_algorithms() {
    println!("--- Profiling cv-photo ---");
    let width = 1920;
    let height = 1080;

    // Create mock 1080p data
    let data_f32 = vec![0.5f32; width * height];
    let tensor_f32 = CpuTensor::from_vec(data_f32, TensorShape::new(1, height, width)).unwrap();

    let mask_u8 = vec![0u8; width * height];
    let tensor_mask = CpuTensor::from_vec(mask_u8, TensorShape::new(1, height, width)).unwrap();

    // 1. Fast NL Means Denoising
    let start = Instant::now();
    let _ = fast_nl_means_denoising(&tensor_f32, 3.0, 7, 21).unwrap();
    let elapsed = start.elapsed();
    println!("Fast NL Means Denoising (1080p): {:?}", elapsed);

    // 2. Inpaint Telea
    let start = Instant::now();
    let _ = inpaint_telea(&tensor_f32, &tensor_mask, 3.0).unwrap();
    let elapsed = start.elapsed();
    println!("Inpaint Telea (1080p, r=3): {:?}", elapsed);
}
