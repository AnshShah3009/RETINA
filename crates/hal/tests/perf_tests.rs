//! Performance tests for cv-hal algorithms
//!
//! These tests measure execution times to determine if implementations
//! need optimization or are usable for production.
//!
//! Run with: cargo test --test perf_tests -- --nocapture
//! Or for detailed benchmarks: cargo bench

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::{DataType, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::{GpuStorage, WgpuGpuStorage};
use cv_hal::GpuTensor;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod helpers;

fn create_test_tensor(data: &[f32], w: usize, h: usize, c: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

fn create_random_f32_tensor(w: usize, h: usize, c: usize, seed: u64) -> Tensor<f32, CpuStorage<f32>> {
    let size = w * h * c;
    let mut rng = helpers::SimpleRng::new(seed);
    let data: Vec<f32> = (0..size).map(|_| rng.next_f32() * 255.0).collect();
    create_test_tensor(&data, w, h, c)
}

fn create_random_pointcloud(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = helpers::SimpleRng::new(seed);
    (0..n * 3).map(|_| rng.next_f32() * 10.0).collect()
}

fn copy_to_gpu(ctx: &GpuContext, tensor: &Tensor<f32, CpuStorage<f32>>) -> GpuTensor<f32> {
    let data = tensor.as_slice().unwrap();
    let shape = tensor.shape;
    let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Perf Test Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

    GpuTensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(buffer), data.len()),
        shape,
        dtype: DataType::F32,
        _phantom: std::marker::PhantomData,
    }
}

fn read_from_gpu(ctx: &GpuContext, gpu_tensor: &GpuTensor<f32>) -> Vec<f32> {
    let buffer = gpu_tensor.storage.buffer();
    let byte_size = (gpu_tensor.storage.len * std::mem::size_of::<f32>()) as u64;

    let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Perf Read Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buffer, 0, &read_buffer, 0, byte_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    pollster::block_on(async {
        let slice = read_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    })
}

fn time_fn<F, R>(name: &str, f: F) -> Duration
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let _ = f();
    let elapsed = start.elapsed();
    println!("  {}: {:?}", name, elapsed);
    elapsed
}

fn time_fn_with_result<F, R>(name: &str, f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("  {}: {:?}", name, elapsed);
    (elapsed, result)
}

mod resize_perf {
    use super::*;

    #[test]
    fn test_resize_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 3, 42);

        println!("\n=== CPU Resize 512x512 -> 256x256 ===");
        time_fn("cpu_bilinear_512x512", || {
            cpu.resize(&input, (256, 256)).unwrap()
        });
    }

    #[test]
    fn test_resize_cpu_1024x1024() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(1024, 1024, 3, 42);

        println!("\n=== CPU Resize 1024x1024 -> 512x512 ===");
        time_fn("cpu_bilinear_1024x1024", || {
            cpu.resize(&input, (512, 512)).unwrap()
        });
    }

    #[test]
    fn test_resize_cpu_2048x2048() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(2048, 2048, 3, 42);

        println!("\n=== CPU Resize 2048x2048 -> 1024x1024 ===");
        time_fn("cpu_bilinear_2048x2048", || {
            cpu.resize(&input, (1024, 1024)).unwrap()
        });
    }

    #[test]
    fn test_resize_lanczos4_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 1, 42);

        println!("\n=== CPU Lanczos4 Resize 512x512 -> 256x256 ===");
        time_fn("cpu_lanczos4_512x512", || {
            cpu.resize(&input, (256, 256)).unwrap()
        });
    }
}

mod resize_gpu_perf {
    use super::*;
    use cv_hal::gpu_kernels::resize::{resize, resize_lanczos4};

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    #[test]
    fn test_resize_gpu_512x512() {
        let ctx = get_gpu_context();
        let input_cpu = create_random_f32_tensor(512, 512, 3, 42);
        let input_gpu = copy_to_gpu(&ctx, &input_cpu);

        println!("\n=== GPU Resize 512x512 -> 256x256 ===");
        time_fn("gpu_bilinear_512x512", || {
            resize(&ctx, &input_gpu, 256, 256).unwrap()
        });

        drop(input_gpu);
    }

    #[test]
    fn test_resize_gpu_1024x1024() {
        let ctx = get_gpu_context();
        let input_cpu = create_random_f32_tensor(1024, 1024, 3, 42);
        let input_gpu = copy_to_gpu(&ctx, &input_cpu);

        println!("\n=== GPU Resize 1024x1024 -> 512x512 ===");
        time_fn("gpu_bilinear_1024x1024", || {
            resize(&ctx, &input_gpu, 512, 512).unwrap()
        });

        drop(input_gpu);
    }

    #[test]
    fn test_resize_gpu_2048x2048() {
        let ctx = get_gpu_context();
        let input_cpu = create_random_f32_tensor(2048, 2048, 3, 42);
        let input_gpu = copy_to_gpu(&ctx, &input_cpu);

        println!("\n=== GPU Resize 2048x2048 -> 1024x1024 ===");
        time_fn("gpu_bilinear_2048x2048", || {
            resize(&ctx, &input_gpu, 1024, 1024).unwrap()
        });

        drop(input_gpu);
    }

    #[test]
    fn test_lanczos4_gpu_512x512() {
        let ctx = get_gpu_context();
        let input_cpu = create_random_f32_tensor(512, 512, 1, 42);
        let input_gpu = copy_to_gpu(&ctx, &input_cpu);

        println!("\n=== GPU Lanczos4 Resize 512x512 -> 256x256 ===");
        time_fn("gpu_lanczos4_512x512", || {
            resize_lanczos4(&ctx, &input_gpu, 256, 256).unwrap()
        });

        drop(input_gpu);
    }

    #[test]
    fn test_lanczos4_gpu_1024x1024() {
        let ctx = get_gpu_context();
        let input_cpu = create_random_f32_tensor(1024, 1024, 1, 42);
        let input_gpu = copy_to_gpu(&ctx, &input_cpu);

        println!("\n=== GPU Lanczos4 Resize 1024x1024 -> 512x512 ===");
        time_fn("gpu_lanczos4_1024x1024", || {
            resize_lanczos4(&ctx, &input_gpu, 512, 512).unwrap()
        });

        drop(input_gpu);
    }
}

mod pyramid_perf {
    use super::*;

    #[test]
    fn test_pyramid_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 3, 42);

        println!("\n=== CPU Pyramid Down 512x512 (single level) ===");
        time_fn("cpu_pyramid_512x512", || {
            cpu.pyramid_down(&input).unwrap()
        });
    }

    #[test]
    fn test_pyramid_cpu_1024x1024() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(1024, 1024, 3, 42);

        println!("\n=== CPU Pyramid Down 1024x1024 (single level) ===");
        time_fn("cpu_pyramid_1024x1024", || {
            cpu.pyramid_down(&input).unwrap()
        });
    }

    #[test]
    fn test_pyramid_multi_level_cpu_1024x1024() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(1024, 1024, 3, 42);

        println!("\n=== CPU Pyramid Down 1024x1024 (4 levels) ===");
        time_fn("cpu_pyramid_4_levels_1024x1024", || {
            let mut result = input.clone();
            for _ in 0..4 {
                result = cpu.pyramid_down(&result).unwrap();
            }
            result
        });
    }
}

mod optical_flow_perf {
    use super::*;

    #[test]
    fn test_optical_flow_lk_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let frame1 = create_random_f32_tensor(512, 512, 1, 42);
        let frame2 = create_random_f32_tensor(512, 512, 1, 43);

        let points: Vec<[f32; 2]> = (0..100)
            .map(|i| [(i as f32 * 5.0) + 10.0, (i as f32 * 5.0) + 10.0])
            .collect();

        println!("\n=== CPU Lucas-Kanade 512x512 (100 points) ===");
        time_fn("cpu_lk_512x512_100pts", || {
            cpu.optical_flow_lk(&[frame1.clone()], &[frame2.clone()], &points, 21, 30)
                .unwrap()
        });
    }

    #[test]
    fn test_optical_flow_lk_cpu_512x512_pyramid() {
        let cpu = CpuBackend::new().unwrap();
        let frame1_0 = create_random_f32_tensor(512, 512, 1, 42);
        let frame2_0 = create_random_f32_tensor(512, 512, 1, 43);

        let frame1_1 = cpu.pyramid_down(&frame1_0).unwrap();
        let frame2_1 = cpu.pyramid_down(&frame2_0).unwrap();
        let frame1_2 = cpu.pyramid_down(&frame1_1).unwrap();
        let frame2_2 = cpu.pyramid_down(&frame2_1).unwrap();

        let points: Vec<[f32; 2]> = (0..100)
            .map(|i| [(i as f32 * 5.0) + 10.0, (i as f32 * 5.0) + 10.0])
            .collect();

        println!("\n=== CPU Lucas-Kanade 512x512 Pyramid (100 points) ===");
        time_fn("cpu_lk_pyramid_512x512_100pts", || {
            cpu.optical_flow_lk(
                &[frame1_2, frame1_1, frame1_0.clone()],
                &[frame2_2, frame2_1, frame2_0],
                &points,
                21,
                30,
            )
            .unwrap()
        });
    }
}

mod icp_perf {
    use super::*;
    use cv_hal::gpu_kernels::icp;

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    fn copy_pointcloud_to_gpu(ctx: &GpuContext, data: &[f32]) -> Tensor<f32, GpuStorage<f32>> {
        let n = data.len() / 3;
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ICP PointCloud Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        let storage =
            WgpuGpuStorage::from_buffer(Arc::new(buffer), data.len());

        Tensor {
            storage,
            shape: TensorShape::new(1, n, 3),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn test_icp_correspondences_gpu_1k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(1000, 42);
        let target_data = create_random_pointcloud(1000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU ICP Correspondences 1K points ===");
        time_fn("gpu_icp_corr_1k", || {
            icp::icp_correspondences(&ctx, &source_gpu, &target_gpu, 1.0).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }

    #[test]
    fn test_icp_correspondences_gpu_10k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(10000, 42);
        let target_data = create_random_pointcloud(10000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU ICP Correspondences 10K points ===");
        time_fn("gpu_icp_corr_10k", || {
            icp::icp_correspondences(&ctx, &source_gpu, &target_gpu, 1.0).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }

    #[test]
    fn test_icp_correspondences_gpu_100k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(100000, 42);
        let target_data = create_random_pointcloud(100000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU ICP Correspondences 100K points ===");
        time_fn("gpu_icp_corr_100k", || {
            icp::icp_correspondences(&ctx, &source_gpu, &target_gpu, 1.0).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }
}

mod spatial_icp_perf {
    use super::*;
    use cv_hal::gpu_kernels::spatial::spatial_hash_correspondences;

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    fn copy_pointcloud_to_gpu(ctx: &GpuContext, data: &[f32]) -> Tensor<f32, GpuStorage<f32>> {
        let n = data.len() / 3;
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial ICP PointCloud Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        let storage = WgpuGpuStorage::from_buffer(Arc::new(buffer), data.len());

        Tensor {
            storage,
            shape: TensorShape::new(1, n, 3),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn test_spatial_hash_icp_1k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(1000, 42);
        let target_data = create_random_pointcloud(1000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU Spatial Hash ICP 1K points ===");
        time_fn("gpu_spatial_icp_1k", || {
            spatial_hash_correspondences(&ctx, &source_gpu, &target_gpu, 1.0, 0.5).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }

    #[test]
    fn test_spatial_hash_icp_10k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(10000, 42);
        let target_data = create_random_pointcloud(10000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU Spatial Hash ICP 10K points ===");
        time_fn("gpu_spatial_icp_10k", || {
            spatial_hash_correspondences(&ctx, &source_gpu, &target_gpu, 1.0, 0.5).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }

    #[test]
    fn test_spatial_hash_icp_100k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(100000, 42);
        let target_data = create_random_pointcloud(100000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU Spatial Hash ICP 100K points ===");
        time_fn("gpu_spatial_icp_100k", || {
            spatial_hash_correspondences(&ctx, &source_gpu, &target_gpu, 1.0, 0.5).unwrap()
        });

        drop(source_gpu);
        drop(target_gpu);
    }
}

mod tvl1_perf {
    use super::*;
    use cv_hal::gpu_kernels::optical_flow::tvl1_optical_flow;
    use cv_hal::gpu_kernels::optical_flow::Tvl1Config;

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    fn copy_image_to_gpu(ctx: &GpuContext, data: &[f32], w: usize, h: usize) -> GpuTensor<f32> {
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TVL1 Image Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        crate::GpuTensor {
            storage: WgpuGpuStorage::from_buffer(Arc::new(buffer), data.len()),
            shape: TensorShape::new(1, h, w),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn test_tvl1_gpu_256x256() {
        let ctx = get_gpu_context();
        let frame1 = create_random_f32_tensor(256, 256, 1, 42);
        let frame2 = create_random_f32_tensor(256, 256, 1, 43);

        let frame1_gpu = copy_image_to_gpu(&ctx, frame1.as_slice().unwrap(), 256, 256);
        let frame2_gpu = copy_image_to_gpu(&ctx, frame2.as_slice().unwrap(), 256, 256);

        let config = Tvl1Config::default();

        println!("\n=== GPU TVL1 256x256 ===");
        time_fn("gpu_tvl1_256x256", || {
            tvl1_optical_flow(&ctx, &frame1_gpu, &frame2_gpu, config).unwrap()
        });
    }

    #[test]
    fn test_tvl1_gpu_512x512() {
        let ctx = get_gpu_context();
        let frame1 = create_random_f32_tensor(512, 512, 1, 42);
        let frame2 = create_random_f32_tensor(512, 512, 1, 43);

        let frame1_gpu = copy_image_to_gpu(&ctx, frame1.as_slice().unwrap(), 512, 512);
        let frame2_gpu = copy_image_to_gpu(&ctx, frame2.as_slice().unwrap(), 512, 512);

        let config = Tvl1Config {
            num_outer_iters: 10,
            num_warps: 5,
            ..Default::default()
        };

        println!("\n=== GPU TVL1 512x512 ===");
        time_fn("gpu_tvl1_512x512", || {
            tvl1_optical_flow(&ctx, &frame1_gpu, &frame2_gpu, config).unwrap()
        });
    }

    #[test]
    fn test_tvl1_gpu_512x512_full_iter() {
        let ctx = get_gpu_context();
        let frame1 = create_random_f32_tensor(512, 512, 1, 42);
        let frame2 = create_random_f32_tensor(512, 512, 1, 43);

        let frame1_gpu = copy_image_to_gpu(&ctx, frame1.as_slice().unwrap(), 512, 512);
        let frame2_gpu = copy_image_to_gpu(&ctx, frame2.as_slice().unwrap(), 512, 512);

        let config = Tvl1Config::default();

        println!("\n=== GPU TVL1 512x512 (full iter) ===");
        time_fn("gpu_tvl1_512x512_full", || {
            tvl1_optical_flow(&ctx, &frame1_gpu, &frame2_gpu, config).unwrap()
        });
    }
}

mod convolution_perf {
    use super::*;
    use cv_hal::context::BorderMode;

    #[test]
    fn test_gaussian_blur_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 1, 42);

        println!("\n=== CPU Gaussian Blur 512x512 (sigma=2.0) ===");
        time_fn("cpu_gaussian_blur_512x512", || {
            cpu.gaussian_blur(&input, 2.0, 7).unwrap()
        });
    }

    #[test]
    fn test_gaussian_blur_cpu_1024x1024() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(1024, 1024, 1, 42);

        println!("\n=== CPU Gaussian Blur 1024x1024 (sigma=2.0) ===");
        time_fn("cpu_gaussian_blur_1024x1024", || {
            cpu.gaussian_blur(&input, 2.0, 7).unwrap()
        });
    }

    #[test]
    fn test_sobel_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 1, 42);

        println!("\n=== CPU Sobel 512x512 ===");
        time_fn("cpu_sobel_512x512", || {
            cpu.sobel(&input, 1, 1, 3).unwrap()
        });
    }

    #[test]
    fn test_convolve2d_cpu_512x512() {
        let cpu = CpuBackend::new().unwrap();
        let input = create_random_f32_tensor(512, 512, 1, 42);
        let mut kernel_data = vec![0.0f32; 9];
        kernel_data[0] = -1.0;
        kernel_data[2] = 1.0;
        kernel_data[3] = -2.0;
        kernel_data[5] = 2.0;
        kernel_data[6] = -1.0;
        kernel_data[8] = 1.0;
        let kernel = create_test_tensor(&kernel_data, 3, 3, 1);

        println!("\n=== CPU Convolve2D 512x512 (3x3 kernel) ===");
        time_fn("cpu_convolve2d_512x512", || {
            cpu.convolve_2d(&input, &kernel, BorderMode::Replicate)
                .unwrap()
        });
    }
}

mod colored_icp_perf {
    use super::*;

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    fn create_pointcloud_with_colors(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = helpers::SimpleRng::new(seed);
        let points: Vec<f32> = (0..n * 3)
            .map(|_| rng.next_f32() * 10.0)
            .collect();
        let colors: Vec<f32> = (0..n * 3)
            .map(|_| rng.next_f32())
            .collect();
        (points, colors)
    }

    fn copy_pointcloud_with_colors_to_gpu(
        ctx: &GpuContext,
        points: &[f32],
        colors: &[f32],
    ) -> (Tensor<f32, GpuStorage<f32>>, Tensor<f32, GpuStorage<f32>>) {
        let n = points.len() / 3;
        
        let points_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColoredICP Points"),
            size: (points.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&points_buffer, 0, bytemuck::cast_slice(points));

        let colors_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColoredICP Colors"),
            size: (colors.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&colors_buffer, 0, bytemuck::cast_slice(colors));

        let points_tensor = Tensor {
            storage: WgpuGpuStorage::from_buffer(Arc::new(points_buffer), n),
            shape: TensorShape::new(1, n, 3),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        };

        let colors_tensor = Tensor {
            storage: WgpuGpuStorage::from_buffer(Arc::new(colors_buffer), n),
            shape: TensorShape::new(1, n, 3),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        };

        (points_tensor, colors_tensor)
    }

    #[test]
    fn test_colored_icp_kernel_1k() {
        let ctx = get_gpu_context();
        let (src_points, src_colors) = create_pointcloud_with_colors(1000, 42);
        let (tgt_points, tgt_colors) = create_pointcloud_with_colors(1000, 43);

        let (src_pts_gpu, src_clr_gpu) = copy_pointcloud_with_colors_to_gpu(&ctx, &src_points, &src_colors);
        let (tgt_pts_gpu, tgt_clr_gpu) = copy_pointcloud_with_colors_to_gpu(&ctx, &tgt_points, &tgt_colors);

        println!("\n=== GPU Colored-ICP Kernel 1K points ===");
        time_fn("gpu_colored_icp_1k", || {
            cv_hal::gpu_kernels::icp::compute_color_gradients_kernel();
            cv_hal::gpu_kernels::icp::colored_icp_kernel();
        });
    }

    #[test]
    fn test_colored_icp_kernel_10k() {
        let ctx = get_gpu_context();
        let (src_points, src_colors) = create_pointcloud_with_colors(10000, 42);
        let (tgt_points, tgt_colors) = create_pointcloud_with_colors(10000, 43);

        let (src_pts_gpu, src_clr_gpu) = copy_pointcloud_with_colors_to_gpu(&ctx, &src_points, &src_colors);
        let (tgt_pts_gpu, tgt_clr_gpu) = copy_pointcloud_with_colors_to_gpu(&ctx, &tgt_points, &tgt_colors);

        println!("\n=== GPU Colored-ICP Kernel 10K points ===");
        time_fn("gpu_colored_icp_10k", || {
            cv_hal::gpu_kernels::icp::compute_color_gradients_kernel();
            cv_hal::gpu_kernels::icp::colored_icp_kernel();
        });
    }
}

mod generalized_icp_perf {
    use super::*;

    fn get_gpu_context() -> GpuContext {
        GpuContext::new().expect("Failed to create GPU context")
    }

    fn copy_pointcloud_to_gpu(ctx: &GpuContext, data: &[f32]) -> Tensor<f32, GpuStorage<f32>> {
        let n = data.len() / 3;
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G-ICP PointCloud Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        Tensor {
            storage: WgpuGpuStorage::<f32>::from_buffer(Arc::new(buffer), n),
            shape: TensorShape::new(1, n, 3),
            dtype: DataType::F32,
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn test_generalized_icp_kernel_1k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(1000, 42);
        let target_data = create_random_pointcloud(1000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU Generalized-ICP Kernel 1K points ===");
        time_fn("gpu_gicp_1k", || {
            cv_hal::gpu_kernels::icp::generalized_icp_kernel();
            cv_hal::gpu_kernels::icp::gicp_accumulate_kernel();
        });
    }

    #[test]
    fn test_generalized_icp_kernel_10k() {
        let ctx = get_gpu_context();
        let source_data = create_random_pointcloud(10000, 42);
        let target_data = create_random_pointcloud(10000, 43);

        let source_gpu = copy_pointcloud_to_gpu(&ctx, &source_data);
        let target_gpu = copy_pointcloud_to_gpu(&ctx, &target_data);

        println!("\n=== GPU Generalized-ICP Kernel 10K points ===");
        time_fn("gpu_gicp_10k", || {
            cv_hal::gpu_kernels::icp::generalized_icp_kernel();
            cv_hal::gpu_kernels::icp::gicp_accumulate_kernel();
        });
    }
}

mod comparison_summary {
    #[test]
    fn print_summary() {
        println!();
        println!("========================================");
        println!("       PERFORMANCE TEST SUMMARY");
        println!("========================================");
        println!();
        println!("Run individual tests with:");
        println!("  cargo test --test perf_tests -- --nocapture");
        println!();
        println!("Thresholds for optimization:");
        println!("  - < 10ms: Excellent - No action needed");
        println!("  - 10-50ms: Acceptable - Monitor");
        println!("  - 50-100ms: Needs attention - Consider optimization");
        println!("  - > 100ms: Critical - Requires optimization");
        println!();
        println!("Key areas to optimize:");
        println!("  1. GPU kernel dispatch overhead (batch operations)");
        println!("  2. Memory transfers (use unified memory when available)");
        println!("  3. Algorithm complexity (e.g., ICP brute force -> spatial index)");
        println!("========================================");
    }
}
