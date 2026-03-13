use cv_core::{storage::Storage, CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ColorConversion, ComputeContext, ThresholdType};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use futures::executor::block_on;

#[test]
fn test_cross_device_parity() {
    // 1. Setup CPU reference
    let cpu = CpuBackend::new().expect("CPU backend unavailable");

    // 2. Enumerate all GPUs
    let adapters = block_on(GpuContext::enumerate_adapters());
    println!("Found {} GPU adapters", adapters.len());

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!(
            "Adapter {}: {} ({:?}) - Backend: {:?}",
            i, info.name, info.device_type, info.backend
        );
    }

    for (i, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        println!("--- Testing Adapter {}: {} ---", i, info.name);

        // Skip problematic AMD Integrated GPU on Vulkan that panics in driver
        if info.name.contains("AMD") && info.backend == wgpu::Backend::Vulkan {
            println!("  ! Skipping AMD Integrated GPU on Vulkan to avoid driver panic");
            continue;
        }

        // Skip llvmpipe
        if info.name.contains("llvmpipe") {
            println!("  ! Skipping software renderer");
            continue;
        }

        // Skip Apple Paravirtual device (CI virtual GPU that doesn't support compute)
        if info.name.contains("Paravirtual") || info.name.contains("paravirtual") {
            println!("  ! Skipping paravirtual GPU (CI virtual device)");
            continue;
        }

        let gpu_res = block_on(GpuContext::from_adapter(adapter));
        let gpu = match gpu_res {
            Ok(g) => g,
            Err(e) => {
                println!(
                    "  ! Skipping adapter {}: GPU context creation failed: {}",
                    info.name, e
                );
                continue;
            }
        };

        // --- Run Parity Tests ---
        test_threshold_parity::<f32>(&cpu, &gpu, &info.name);
        test_pc_transform_parity::<f32>(&cpu, &gpu, &info.name);
        test_color_cvt_parity::<f32>(&cpu, &gpu, &info.name);
        test_resize_parity(&cpu, &gpu, &info.name);
        test_bilateral_parity::<f32>(&cpu, &gpu, &info.name);
        test_fast_parity::<f32>(&cpu, &gpu, &info.name);
        test_matching_parity(&cpu, &gpu, &info.name);
        test_icp_parity(&cpu, &gpu, &info.name);
    }
}

fn test_icp_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let num_src = 100;
    let num_tgt = 200;

    let mut src_data = vec![0.0f32; num_src * 4];
    let mut tgt_data = vec![0.0f32; num_tgt * 4];

    // Create some exact matches
    for i in 0..num_src {
        src_data[i * 4] = i as f32;
        src_data[i * 4 + 1] = (i * 2) as f32;
        src_data[i * 4 + 2] = (i * 3) as f32;
        src_data[i * 4 + 3] = 1.0;

        tgt_data[i * 4] = i as f32;
        tgt_data[i * 4 + 1] = (i * 2) as f32;
        tgt_data[i * 4 + 2] = (i * 3) as f32;
        tgt_data[i * 4 + 3] = 1.0;
    }

    let src_cpu: CpuTensor<f32> =
        Tensor::from_vec(src_data, TensorShape::new(1, num_src, 4)).unwrap();
    let tgt_cpu: CpuTensor<f32> =
        Tensor::from_vec(tgt_data, TensorShape::new(1, num_tgt, 4)).unwrap();

    let src_gpu = src_cpu.to_gpu_ctx(gpu).unwrap();
    let tgt_gpu = tgt_cpu.to_gpu_ctx(gpu).unwrap();

    // CPU
    let res_cpu = cpu.icp_correspondences(&src_cpu, &tgt_cpu, 1.0).unwrap();

    // GPU
    let res_gpu = gpu.icp_correspondences(&src_gpu, &tgt_gpu, 1.0).unwrap();

    assert_eq!(
        res_cpu.len(),
        res_gpu.len(),
        "ICP count mismatch on {}",
        gpu_name
    );
    for i in 0..res_cpu.len() {
        assert_eq!(res_cpu[i].0, res_gpu[i].0);
        assert_eq!(res_cpu[i].1, res_gpu[i].1);
        assert!((res_cpu[i].2 - res_gpu[i].2).abs() < 1e-5);
    }
    println!("  ✓ ICP parity passed for {}", gpu_name);
}

fn test_matching_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let q_len = 50;
    let t_len = 100;
    let d_size = 32;

    let mut q_data = vec![0u8; q_len * d_size];
    let mut t_data = vec![0u8; t_len * d_size];

    // Create some exact matches
    for i in 0..q_len {
        for j in 0..d_size {
            let val = (i + j) as u8;
            q_data[i * d_size + j] = val;
            t_data[i * d_size + j] = val; // Direct match at same index
        }
    }

    let query_cpu: CpuTensor<u8> =
        Tensor::from_vec(q_data, TensorShape::new(1, q_len, d_size)).unwrap();
    let train_cpu: CpuTensor<u8> =
        Tensor::from_vec(t_data, TensorShape::new(1, t_len, d_size)).unwrap();

    let query_gpu = query_cpu.to_gpu_ctx(gpu).unwrap();
    let train_gpu = train_cpu.to_gpu_ctx(gpu).unwrap();

    let ratio = 0.8f32;

    // CPU
    let res_cpu = cpu
        .match_descriptors(&query_cpu, &train_cpu, ratio)
        .unwrap();

    // GPU
    let res_gpu = gpu
        .match_descriptors(&query_gpu, &train_gpu, ratio)
        .unwrap();

    assert_eq!(
        res_cpu.matches.len(),
        res_gpu.matches.len(),
        "Match count mismatch on {}",
        gpu_name
    );
    for i in 0..res_cpu.matches.len() {
        assert_eq!(res_cpu.matches[i].query_idx, res_gpu.matches[i].query_idx);
        assert_eq!(res_cpu.matches[i].train_idx, res_gpu.matches[i].train_idx);
        assert_eq!(res_cpu.matches[i].distance, res_gpu.matches[i].distance);
    }
    println!("  ✓ Matching parity passed for {}", gpu_name);
}

fn test_fast_parity<T: cv_core::float::Float + bytemuck::Pod>(
    cpu: &CpuBackend,
    gpu: &GpuContext,
    gpu_name: &str,
) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![T::ZERO; shape.len()];
    // Create some corners
    for y in 30..60 {
        for x in 30..60 {
            data[y * 128 + x] = T::from_f32(255.0);
        }
    }

    let input_cpu: CpuTensor<T> = Tensor::from_vec(data, shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    // CPU
    let res_cpu = cpu
        .fast_detect(&input_cpu, T::from_f32(20.0), false)
        .unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // GPU (Should return NotSupported for now, we handle it gracefully in the test runner)
    let res_gpu_encoded = gpu.fast_detect(&input_gpu, T::from_f32(20.0), false);

    match res_gpu_encoded {
        Ok(res_gpu_t) => {
            let res_gpu = res_gpu_t.to_cpu_ctx(gpu).unwrap();
            let gpu_slice = res_gpu.storage.as_slice().unwrap();
            for i in 0..cpu_slice.len() {
                if cpu_slice[i] != gpu_slice[i] {
                    panic!(
                        "FAST Parity failure on {}: at index {}, CPU={}, GPU={}",
                        gpu_name, i, cpu_slice[i], gpu_slice[i]
                    );
                }
            }
            println!("  ✓ FAST parity passed for {}", gpu_name);
        }
        Err(cv_hal::Error::NotSupported(_)) => {
            println!("  - FAST parity skipped for {} (NotSupported)", gpu_name);
        }
        Err(e) => panic!("FAST failed on {}: {}", gpu_name, e),
    }
}

fn test_bilateral_parity<T: cv_core::float::Float + bytemuck::Pod>(
    cpu: &CpuBackend,
    gpu: &GpuContext,
    gpu_name: &str,
) {
    let shape = TensorShape::new(1, 64, 64);
    let mut data = vec![T::ZERO; shape.len()];
    for i in 0..data.len() {
        data[i] = T::from_f32((i % 256) as f32);
    }

    let input_cpu: CpuTensor<T> = Tensor::from_vec(data, shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    // CPU
    let res_cpu = cpu
        .bilateral_filter(&input_cpu, 5, T::from_f32(10.0), T::from_f32(10.0))
        .unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // GPU
    let res_gpu_encoded = gpu
        .bilateral_filter(&input_gpu, 5, T::from_f32(10.0), T::from_f32(10.0))
        .unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();

    for i in 0..cpu_slice.len() {
        if (cpu_slice[i].to_f32() as i32 - gpu_slice[i].to_f32() as i32).abs() > 1 {
            panic!(
                "Bilateral Parity failure on {}: at index {}, CPU={}, GPU={}",
                gpu_name, i, cpu_slice[i], gpu_slice[i]
            );
        }
    }
    println!("  ✓ Bilateral parity passed for {}", gpu_name);
}

fn test_resize_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![0.0f32; shape.len()];
    for i in 0..data.len() {
        data[i] = (i % 256) as f32;
    }

    let input_cpu: CpuTensor<f32> = Tensor::from_vec(data, shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    let new_shape = (64, 64);

    // CPU
    let res_cpu = cpu.resize(&input_cpu, new_shape).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // GPU
    let res_gpu_encoded = gpu.resize(&input_gpu, new_shape).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();

    for i in 0..cpu_slice.len() {
        if (cpu_slice[i] - gpu_slice[i]).abs() > 1.0 {
            panic!(
                "Resize Parity failure on {}: at index {}, CPU={}, GPU={}",
                gpu_name, i, cpu_slice[i], gpu_slice[i]
            );
        }
    }
    println!("  ✓ Resize parity passed for {}", gpu_name);
}

fn test_color_cvt_parity<T: cv_core::float::Float + bytemuck::Pod>(
    cpu: &CpuBackend,
    gpu: &GpuContext,
    gpu_name: &str,
) {
    let shape = TensorShape::new(3, 64, 64);
    let mut data = vec![T::ZERO; shape.len()];
    for i in 0..data.len() {
        data[i] = T::from_f32((i % 256) as f32);
    }

    let input_cpu: CpuTensor<T> = Tensor::from_vec(data, shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    // CPU
    let res_cpu = cpu
        .cvt_color(&input_cpu, ColorConversion::RgbToGray)
        .unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // GPU
    let res_gpu_encoded = gpu
        .cvt_color(&input_gpu, ColorConversion::RgbToGray)
        .unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();

    for i in 0..cpu_slice.len() {
        // Use tolerance of 1 due to potential rounding differences
        if (cpu_slice[i].to_f32() - gpu_slice[i].to_f32()).abs() > 1.0 {
            panic!(
                "Color Cvt Parity failure on {}: at index {}, CPU={}, GPU={}",
                gpu_name, i, cpu_slice[i], gpu_slice[i]
            );
        }
    }
    println!("  ✓ Color Cvt parity passed for {}", gpu_name);
}

fn test_pc_transform_parity<T: cv_core::float::Float + bytemuck::Pod>(
    cpu: &CpuBackend,
    gpu: &GpuContext,
    gpu_name: &str,
) {
    let num_points = 1000;
    let mut data = vec![T::ZERO; num_points * 4];
    for i in 0..num_points {
        data[i * 4] = T::from_f32((i % 100) as f32);
        data[i * 4 + 1] = T::from_f32((i % 100) as f32);
        data[i * 4 + 2] = T::from_f32(i as f32);
        data[i * 4 + 3] = T::ONE;
    }

    let shape = TensorShape::new(1, num_points, 4);
    let input_cpu: CpuTensor<T> = Tensor::from_vec(data, shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    let mut transform = [[T::ZERO; 4]; 4];
    transform[0][0] = T::ONE;
    transform[0][3] = T::from_f32(10.0);
    transform[1][1] = T::ONE;
    transform[1][3] = T::from_f32(-5.0);
    transform[2][2] = T::ONE;
    transform[2][3] = T::from_f32(2.0);
    transform[3][3] = T::ONE;

    // Execute on CPU
    let res_cpu = cpu.pointcloud_transform(&input_cpu, &transform).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // Execute on GPU (skip if not implemented)
    match gpu.pointcloud_transform(&input_gpu, &transform) {
        Err(cv_hal::Error::NotSupported(_)) => {
            println!(
                "  ⊘ PC Transform skipped for {} (GPU kernel not implemented)",
                gpu_name
            );
            return;
        }
        Err(e) => panic!("PC Transform failed on {}: {:?}", gpu_name, e),
        Ok(res_gpu_encoded) => {
            let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
            let gpu_slice = res_gpu.storage.as_slice().unwrap();

            for i in 0..cpu_slice.len() {
                if (cpu_slice[i] - gpu_slice[i]).abs() > T::from_f32(1e-5) {
                    panic!(
                        "PC Transform Parity failure on {}: at index {}, CPU={}, GPU={}",
                        gpu_name, i, cpu_slice[i], gpu_slice[i]
                    );
                }
            }

            println!("  ✓ PC Transform parity passed for {}", gpu_name);
        }
    }
}

fn test_threshold_parity<T: cv_core::float::Float + bytemuck::Pod>(
    cpu: &CpuBackend,
    gpu: &GpuContext,
    gpu_name: &str,
) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![T::ZERO; shape.len()];
    for i in 0..data.len() {
        data[i] = T::from_f32((i % 256) as f32);
    }

    let input_cpu: CpuTensor<T> = Tensor::from_vec(data.clone(), shape).unwrap();
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Failed to upload to GPU");

    let thresh = T::from_f32(128.0);
    let max_val = T::from_f32(255.0);

    // Execute on CPU
    let res_cpu = cpu
        .threshold(&input_cpu, thresh, max_val, ThresholdType::Binary)
        .unwrap();

    // Execute on GPU
    let res_gpu_encoded = gpu
        .threshold(&input_gpu, thresh, max_val, ThresholdType::Binary)
        .unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();

    // Compare
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();

    for i in 0..cpu_slice.len() {
        if cpu_slice[i] != gpu_slice[i] {
            panic!(
                "Parity failure on {}: at index {}, CPU={}, GPU={}",
                gpu_name, i, cpu_slice[i], gpu_slice[i]
            );
        }
    }
    println!("  ✓ Threshold parity passed for {}", gpu_name);
}
