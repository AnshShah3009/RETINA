use cv_core::{Tensor, TensorShape};
use cv_features::orb::{detect_and_compute_ctx, orb_detect_and_compute, Orb};
use cv_hal::compute::ComputeDevice;
use cv_hal::gpu::GpuContext;
use cv_runtime::orchestrator::scheduler;
use image::{GrayImage, Luma};
use std::time::Instant;

fn create_benchmark_image(width: u32, height: u32) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    // Create a complex pattern for FAST to find features
    for y in 0..height {
        for x in 0..width {
            let is_white = ((x / 16) + (y / 16)) % 2 == 0;
            let noise = (rand::random::<u8>() % 32) as i32 - 16;
            let val = if is_white { 200 } else { 50 };
            img.put_pixel(x, y, Luma([(val as i32 + noise).clamp(0, 255) as u8]));
        }
    }
    img
}

fn main() {
    let width = 1280;
    let height = 720;
    println!("Creating benchmark image {}x{}...", width, height);
    let img = create_benchmark_image(width, height);

    let n_features = 1000;
    let n_levels = 8;
    let orb = Orb::new()
        .with_n_features(n_features)
        .with_n_levels(n_levels);

    println!(
        "Starting ORB Benchmark (Features: {}, Levels: {})",
        n_features, n_levels
    );

    // 1. CPU Benchmark
    println!(
        "
[CPU Benchmark]"
    );
    let start_cpu = Instant::now();
    let (kps_cpu, _) = orb_detect_and_compute(&img, n_features);
    let duration_cpu = start_cpu.elapsed();
    println!(
        "CPU: Detected {} keypoints in {:?}",
        kps_cpu.len(),
        duration_cpu
    );

    // 2. GPU Benchmark
    println!(
        "
[GPU Benchmark]"
    );
    let gpu = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("GPU unavailable: {}", e);
            return;
        }
    };

    let device = ComputeDevice::Gpu(&gpu);
    let group = scheduler().unwrap().get_default_group().unwrap();

    // Upload image to GPU
    let shape = TensorShape::new(1, height as usize, width as usize);
    let tensor_cpu: Tensor<u8, cv_core::storage::CpuStorage<u8>> =
        Tensor::from_vec(img.to_vec(), shape).unwrap();
    let tensor_gpu = cv_hal::tensor_ext::TensorToGpu::to_gpu_ctx(&tensor_cpu, &gpu).unwrap();

    // Warm-up
    let _ = detect_and_compute_ctx(&orb, &device, &group, &tensor_gpu);

    let start_gpu = Instant::now();
    let (kps_gpu, _) = detect_and_compute_ctx(&orb, &device, &group, &tensor_gpu);
    let duration_gpu = start_gpu.elapsed();

    println!(
        "GPU: Detected {} keypoints in {:?}",
        kps_gpu.len(),
        duration_gpu
    );

    println!(
        "
Speedup: {:.2}x",
        duration_cpu.as_secs_f64() / duration_gpu.as_secs_f64()
    );

    if !kps_gpu.is_empty() && !kps_cpu.is_empty() {
        println!("Benchmark completed successfully.");
    }
}
