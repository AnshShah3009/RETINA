use cv_core::tensor::{CpuTensor, TensorShape};
use cv_imgproc::{
    distance_transform::{distance_transform, DistanceType},
    gaussian_blur, threshold, ThresholdType,
};
use image::{GrayImage, Luma};
use std::time::Instant;

#[test]
fn profile_imgproc_algorithms() {
    println!("--- Profiling cv-imgproc ---");
    let width = 1920;
    let height = 1080;

    // Create mock 1080p data
    let gray_image = GrayImage::from_fn(width, height, |_, _| Luma([128u8]));

    // 1. Thresholding
    let start = Instant::now();
    let _ = threshold(&gray_image, 100, 255, ThresholdType::Binary);
    let elapsed = start.elapsed();
    println!("Threshold (1080p): {:?}", elapsed);

    // 2. Gaussian Blur
    let start = Instant::now();
    let _ = gaussian_blur(&gray_image, 3.0);
    let elapsed = start.elapsed();
    println!("Gaussian Blur (1080p, sigma=3.0): {:?}", elapsed);

    // 3. Distance Transform
    let tensor_f32 = CpuTensor::from_vec(
        vec![0.0f32; (width * height) as usize],
        TensorShape::new(1, height as usize, width as usize),
    )
    .unwrap();
    let start = Instant::now();
    let _ = distance_transform(&tensor_f32, DistanceType::L2);
    let elapsed = start.elapsed();
    println!("Distance Transform (1080p, L2): {:?}", elapsed);
}
