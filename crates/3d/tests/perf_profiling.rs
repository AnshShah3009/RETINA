use cv_3d::{
    estimate_normals_auto, estimate_normals_cpu, estimate_normals_gpu, estimate_normals_hybrid,
};
use cv_hal::gpu::GpuContext;
use nalgebra::Point3;
use std::time::Instant;

#[test]
fn profile_3d_algorithms() {
    println!("--- Profiling cv-3d (Normal Estimation) ---");
    let num_points = 500_000;

    // Ensure scheduler is initialized
    let s = cv_runtime::scheduler().unwrap();

    // Create a GPU group if not exists
    if s.get_best_group(
        cv_hal::BackendType::WebGPU,
        cv_runtime::orchestrator::WorkloadHint::Default,
    )
    .unwrap()
    .is_none()
    {
        if let Some(gpu) = cv_runtime::registry().unwrap().default_gpu() {
            let policy = cv_runtime::GroupPolicy::default();
            let _ = s.create_group_with_device("gpu_perf", 1, None, policy, gpu.id());
        }
    }

    // Check if we are on Unified Memory Architecture
    if let Ok(gpu) = GpuContext::global() {
        println!("Unified Memory: {}", gpu.is_unified_memory());
    }

    // Create mock point cloud (500k points)
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        points.push(Point3::new(
            (i as f32) * 0.001,
            (i as f32) * 0.002,
            (i as f32) * 0.003,
        ));
    }

    // 1. CPU Normal Estimation (Optimized Grid-Link-List)
    let start = Instant::now();
    let normals_cpu = estimate_normals_cpu(&points, 15);
    let elapsed_cpu = start.elapsed();
    println!("1. CPU Normal Estimation (500k):      {:?}", elapsed_cpu);

    // 2. GPU Normal Estimation (Full On-Device LBVH Pipeline)
    let start = Instant::now();
    let (gpu_runner, is_gpu) = cv_runtime::orchestrator::best_runner_gpu_wait_for(
        cv_runtime::orchestrator::WorkloadHint::Throughput,
        100, // Request 100MB
        None,
    )
    .unwrap();

    if is_gpu {
        let normals_gpu = cv_3d::gpu::point_cloud::compute_normals_ctx(&points, 15, &gpu_runner);
        let elapsed_gpu = start.elapsed();
        println!("2. GPU Normal Estimation (500k):      {:?}", elapsed_gpu);
        assert_eq!(normals_gpu.len(), num_points);
    } else {
        println!("2. GPU Normal Estimation (500k):      SKIPPED (No GPU runner)");
    }

    // 3. Hybrid Normal Estimation (CPU kNN + GPU PCA)
    let start = Instant::now();
    let normals_hybrid = estimate_normals_hybrid(&points, 15);
    let elapsed_hybrid = start.elapsed();
    println!("3. Hybrid Normal Estimation (500k):   {:?}", elapsed_hybrid);

    // 4. Large Scale Benchmark (1M points)
    println!("--- Large Scale Benchmark (1,000,000 points) ---");
    let num_points_large = 1_000_000;
    let mut points_large = Vec::with_capacity(num_points_large);
    for i in 0..num_points_large {
        points_large.push(Point3::new(
            (i as f32) * 0.0005,
            (i as f32) * 0.001,
            (i as f32) * 0.0015,
        ));
    }

    let start = Instant::now();
    let _ = estimate_normals_cpu(&points_large, 15);
    println!(
        "CPU Normal Estimation (1M):           {:?}",
        start.elapsed()
    );

    let start = Instant::now();
    let (gpu_runner_large, is_gpu_large) = cv_runtime::orchestrator::best_runner_gpu_wait_for(
        cv_runtime::orchestrator::WorkloadHint::Throughput,
        200, // Request 200MB
        None,
    )
    .unwrap();
    if is_gpu_large {
        let _ = cv_3d::gpu::point_cloud::compute_normals_ctx(&points_large, 15, &gpu_runner_large);
        println!(
            "GPU Normal Estimation (1M):           {:?}",
            start.elapsed()
        );
    }
}
