#![forbid(unsafe_code)]
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

    let mut tested: Vec<(String, GpuContext)> = Vec::new();
    let mut skipped = 0usize;

    for (i, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        println!("--- Testing Adapter {}: {} ---", i, info.name);

        // Skip GL/OpenGL backends (radeonsi) — not suitable for GPU compute testing
        if info.backend != wgpu::Backend::Vulkan {
            println!("  ! Skipping non-Vulkan backend ({:?})", info.backend);
            skipped += 1;
            continue;
        }

        // Skip software / CI virtual renderers (can hang on map/readback).
        let name_l = info.name.to_ascii_lowercase();
        if name_l.contains("llvmpipe")
            || name_l.contains("lavapipe")
            || name_l.contains("swiftshader")
            || name_l.contains("paravirtual")
            || info.device_type == wgpu::DeviceType::Cpu
        {
            println!("  ! Skipping software/virtual renderer ({})", info.name);
            skipped += 1;
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
                skipped += 1;
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

        tested.push((info.name, gpu));
    }

    // A run where every adapter was skipped asserts nothing, so report what
    // actually ran instead of a green test that never touched a device. CI has no
    // adapter at all (a legitimate skip); an environment that claims to have one
    // can require it with CV_REQUIRE_GPU=1.
    println!(
        "device summary: {} adapter(s) verified, {} skipped",
        tested.len(),
        skipped
    );
    if std::env::var("CV_REQUIRE_GPU").is_ok() && tested.is_empty() {
        panic!(
            "CV_REQUIRE_GPU is set but no usable Vulkan adapter was verified ({} adapter(s) found, {} skipped)",
            tested.len() + skipped,
            skipped
        );
    }

    // The point of a *multi*-GPU test: with two usable devices, compare their
    // outputs with each other, not only with the CPU.
    if tested.len() >= 2 {
        let (name_a, gpu_a) = &tested[0];
        let (name_b, gpu_b) = &tested[1];
        println!("--- Device-to-device parity: {} vs {} ---", name_a, name_b);
        test_device_to_device_parity(gpu_a, gpu_b, name_a, name_b);
    } else {
        println!(
            "device-to-device parity skipped: needs 2 usable Vulkan adapters, found {}",
            tested.len()
        );
    }
}

/// Run the same operations on two devices and compare their outputs directly.
fn test_device_to_device_parity(
    gpu_a: &GpuContext,
    gpu_b: &GpuContext,
    name_a: &str,
    name_b: &str,
) {
    let shape = TensorShape::new(1, 128, 128);
    let data: Vec<f32> = (0..shape.len()).map(|i| (i % 256) as f32).collect();
    let input_cpu: CpuTensor<f32> = Tensor::from_vec(data, shape).unwrap();

    // Threshold (a 0/255 mask, so equality must be exact)
    let a = input_cpu.to_gpu_ctx(gpu_a).unwrap();
    let b = input_cpu.to_gpu_ctx(gpu_b).unwrap();
    let res_a = gpu_a
        .threshold(&a, 128.0, 255.0, ThresholdType::Binary)
        .unwrap()
        .to_cpu_ctx(gpu_a)
        .unwrap();
    let res_b = gpu_b
        .threshold(&b, 128.0, 255.0, ThresholdType::Binary)
        .unwrap()
        .to_cpu_ctx(gpu_b)
        .unwrap();
    assert_eq!(
        res_a.storage.as_slice().unwrap(),
        res_b.storage.as_slice().unwrap(),
        "threshold differs between {} and {}",
        name_a,
        name_b
    );

    // Resize (allow one intensity step, as the CPU/GPU comparison does)
    let res_a = gpu_a
        .resize(&a, (64, 64))
        .unwrap()
        .to_cpu_ctx(gpu_a)
        .unwrap();
    let res_b = gpu_b
        .resize(&b, (64, 64))
        .unwrap()
        .to_cpu_ctx(gpu_b)
        .unwrap();
    let (sa, sb) = (
        res_a.storage.as_slice().unwrap(),
        res_b.storage.as_slice().unwrap(),
    );
    for i in 0..sa.len() {
        assert!(
            (sa[i] - sb[i]).abs() <= 1.0,
            "resize differs between {} and {} at {}: {} vs {}",
            name_a,
            name_b,
            i,
            sa[i],
            sb[i]
        );
    }

    // Color conversion
    let rgb_shape = TensorShape::new(3, 64, 64);
    let rgb_data: Vec<f32> = (0..rgb_shape.len()).map(|i| (i % 256) as f32).collect();
    let rgb_cpu: CpuTensor<f32> = Tensor::from_vec(rgb_data, rgb_shape).unwrap();
    let a = rgb_cpu.to_gpu_ctx(gpu_a).unwrap();
    let b = rgb_cpu.to_gpu_ctx(gpu_b).unwrap();
    let res_a = gpu_a
        .cvt_color(&a, ColorConversion::RgbToGray)
        .unwrap()
        .to_cpu_ctx(gpu_a)
        .unwrap();
    let res_b = gpu_b
        .cvt_color(&b, ColorConversion::RgbToGray)
        .unwrap()
        .to_cpu_ctx(gpu_b)
        .unwrap();
    let (sa, sb) = (
        res_a.storage.as_slice().unwrap(),
        res_b.storage.as_slice().unwrap(),
    );
    for i in 0..sa.len() {
        assert!(
            (sa[i] - sb[i]).abs() <= 1.0,
            "color conversion differs between {} and {} at {}: {} vs {}",
            name_a,
            name_b,
            i,
            sa[i],
            sb[i]
        );
    }

    // Descriptor matching must agree exactly (integer indices and distances)
    let q_len = 50;
    let t_len = 100;
    let d_size = 32;
    let mut q_data = vec![0u8; q_len * d_size];
    let mut t_data = vec![0u8; t_len * d_size];
    for i in 0..q_len {
        for j in 0..d_size {
            let val = (i + j) as u8;
            q_data[i * d_size + j] = val;
            t_data[i * d_size + j] = val;
        }
    }
    let q_cpu: CpuTensor<u8> =
        Tensor::from_vec(q_data, TensorShape::new(1, q_len, d_size)).unwrap();
    let t_cpu: CpuTensor<u8> =
        Tensor::from_vec(t_data, TensorShape::new(1, t_len, d_size)).unwrap();
    let (qa, ta) = (
        q_cpu.to_gpu_ctx(gpu_a).unwrap(),
        t_cpu.to_gpu_ctx(gpu_a).unwrap(),
    );
    let (qb, tb) = (
        q_cpu.to_gpu_ctx(gpu_b).unwrap(),
        t_cpu.to_gpu_ctx(gpu_b).unwrap(),
    );
    let res_a = gpu_a.match_descriptors(&qa, &ta, 0.8).unwrap();
    let res_b = gpu_b.match_descriptors(&qb, &tb, 0.8).unwrap();
    assert_eq!(
        res_a.matches.len(),
        res_b.matches.len(),
        "match count differs between {} and {}",
        name_a,
        name_b
    );
    for (ma, mb) in res_a.matches.iter().zip(res_b.matches.iter()) {
        assert_eq!(ma.query_idx, mb.query_idx);
        assert_eq!(ma.train_idx, mb.train_idx);
        assert_eq!(ma.distance, mb.distance);
    }

    println!(
        "  ✓ Device-to-device parity passed: {} vs {} (threshold, resize, color cvt, matching)",
        name_a, name_b
    );
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

/// The device selector must resolve every GPU class the machine has, and a
/// selector matching nothing must fail rather than silently fall back to
/// another device — a run that believes it measured the iGPU while actually
/// using the dGPU is worse than a run that failed outright.
#[test]
fn test_device_selector_resolves_each_gpu_class() {
    let adapters = block_on(GpuContext::describe_adapters());
    println!("adapters: {adapters:?}");
    if adapters.is_empty() {
        println!("no adapters; selector resolution not exercised");
        return;
    }

    use cv_hal::gpu::DeviceSelector;

    assert!(
        block_on(GpuContext::select_adapter(&DeviceSelector::Default)).is_ok(),
        "the default selector must resolve when an adapter exists"
    );

    // Discrete/Integrated fall back to the other class, so a single-GPU machine
    // still resolves rather than failing.
    for selector in [DeviceSelector::Discrete, DeviceSelector::Integrated] {
        match block_on(GpuContext::select_adapter(&selector)) {
            Ok(a) => println!("{selector:?} resolved to {}", a.get_info().name),
            Err(e) => panic!("{selector:?} should resolve on a machine with a GPU: {e}"),
        }
    }

    // A name that does not exist is an error, and the message lists what exists.
    let bogus = block_on(GpuContext::select_adapter(&DeviceSelector::NameContains(
        "no-such-gpu-name".into(),
    )));
    assert!(
        bogus.is_err(),
        "a selector matching nothing must not silently succeed"
    );
    let err = bogus.err().map(|e| e.to_string()).unwrap_or_default();
    println!("bogus selector error: {err}");
    assert!(
        err.contains("no-such-gpu-name") || err.contains("available"),
        "the error should name the selector or list the adapters, got: {err}"
    );

    // A real name resolves.
    let token: String = adapters[0]
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();
    if !token.is_empty() {
        match block_on(GpuContext::select_adapter(&DeviceSelector::NameContains(
            token.to_ascii_lowercase(),
        ))) {
            Ok(a) => println!("name selector matched {}", a.get_info().name),
            Err(e) => println!("name selector for {token:?} did not match: {e}"),
        }
    }
}

/// Every GPU must agree with the CPU, and with the others, on the operations the
/// pipelines use. Device-to-device agreement is the point: a result that varies
/// by device is not reproducible.
#[test]
fn test_all_gpus_agree_with_cpu_and_each_other() {
    let cpu = CpuBackend::new().expect("CPU backend unavailable");
    let adapters = block_on(GpuContext::enumerate_adapters());
    if adapters.is_empty() {
        println!("no GPU adapters; skipping");
        return;
    }

    let mut contexts: Vec<(String, GpuContext)> = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        if info.backend != wgpu::Backend::Vulkan || info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }
        let name = info.name.clone();
        if let Ok(ctx) = block_on(GpuContext::from_adapter(adapter)) {
            contexts.push((name, ctx));
        }
    }
    println!(
        "comparing {} GPU(s) against the CPU: {:?}",
        contexts.len(),
        contexts.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
    );
    if contexts.is_empty() {
        return;
    }

    let shape = TensorShape::new(1, 192, 256);
    let data: Vec<f32> = (0..shape.len()).map(|i| ((i * 37) % 251) as f32).collect();
    let input_cpu: CpuTensor<f32> = Tensor::from_vec(data, shape).unwrap();

    let rgb_shape = TensorShape::new(3, 96, 128);
    let rgb_data: Vec<f32> = (0..rgb_shape.len())
        .map(|i| ((i * 53) % 241) as f32)
        .collect();
    let rgb_cpu: CpuTensor<f32> = Tensor::from_vec(rgb_data, rgb_shape).unwrap();

    let mut results: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, gpu) in &contexts {
        let t = input_cpu.to_gpu_ctx(gpu).unwrap();
        let thr = gpu
            .threshold(&t, 128.0, 255.0, ThresholdType::Binary)
            .unwrap()
            .to_cpu_ctx(gpu)
            .unwrap();
        let resized = gpu.resize(&t, (96, 128)).unwrap().to_cpu_ctx(gpu).unwrap();
        let rgb = rgb_cpu.to_gpu_ctx(gpu).unwrap();
        let gray = gpu
            .cvt_color(&rgb, ColorConversion::RgbToGray)
            .unwrap()
            .to_cpu_ctx(gpu)
            .unwrap();

        let mut all = thr.storage.as_slice().unwrap().to_vec();
        all.extend_from_slice(resized.storage.as_slice().unwrap());
        all.extend_from_slice(gray.storage.as_slice().unwrap());
        results.push((name.clone(), all));
    }

    let cpu_thr = cpu
        .threshold(&input_cpu, 128.0, 255.0, ThresholdType::Binary)
        .unwrap();
    let cpu_resized = cpu.resize(&input_cpu, (96, 128)).unwrap();
    let cpu_gray = cpu.cvt_color(&rgb_cpu, ColorConversion::RgbToGray).unwrap();
    let mut cpu_all = cpu_thr.storage.as_slice().unwrap().to_vec();
    cpu_all.extend_from_slice(cpu_resized.storage.as_slice().unwrap());
    cpu_all.extend_from_slice(cpu_gray.storage.as_slice().unwrap());

    for (name, got) in &results {
        assert_eq!(
            got.len(),
            cpu_all.len(),
            "{name}: result length differs from the CPU reference"
        );
        let mut diffs = 0usize;
        let mut worst = 0.0f32;
        for (a, b) in got.iter().zip(cpu_all.iter()) {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
            if d > 1.0 {
                diffs += 1;
            }
        }
        assert_eq!(
            diffs, 0,
            "{name}: {diffs} values differ from the CPU by more than 1.0 (worst {worst})"
        );
        println!("  {name}: matches the CPU (worst difference {worst:.3})");
    }

    for pair in results.windows(2) {
        let (an, a) = &pair[0];
        let (bn, b) = &pair[1];
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= 1.0, "{an} and {bn} disagree: {x} vs {y}");
        }
        println!("  {an} and {bn} agree");
    }
}

/// Canny must actually produce edges on the GPU.
///
/// The GPU shader declared its input `array<u32>` and read four packed bytes per
/// word, but the host uploads f32. The low byte of an f32 is 0x00 for every
/// whole-number value, so essentially every pixel read as zero, every Sobel
/// sample was zero, and the edge map came back uniformly black - with no error
/// anywhere, because the only guard was a clamp on the way out. The existing
/// parity tests did not cover Canny, which is why it survived.
///
/// This asserts the edge is found at all, on every GPU and on the CPU, rather
/// than only that the two agree - agreeing on all-zero is the failure mode.
#[test]
fn test_canny_finds_edges_on_every_device() {
    use cv_core::{CpuTensor, Storage, Tensor, TensorShape};

    let (w, h) = (64u32, 64u32);
    // Left half black, right half white: one strong vertical edge at x = 32.
    let mut data = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 32..w {
            data[(y * w + x) as usize] = 255.0;
        }
    }
    let shape = TensorShape::new(1, h as usize, w as usize);
    let input: CpuTensor<f32> = Tensor::from_vec(data, shape).unwrap();

    let count_nonzero = |t: &cv_core::CpuTensor<f32>| -> usize {
        t.storage
            .as_slice()
            .unwrap()
            .iter()
            .filter(|&&v| v > 0.0)
            .count()
    };

    let cpu = CpuBackend::new().expect("CPU backend unavailable");
    let cpu_edges = count_nonzero(&cpu.canny(&input, 50.0_f32, 150.0_f32).unwrap());
    assert!(
        cpu_edges > 0,
        "the CPU reference itself found no edges, so the test input is wrong"
    );
    println!("  CPU: {cpu_edges} edge pixels");

    for adapter in block_on(GpuContext::enumerate_adapters()) {
        let info = adapter.get_info();
        if info.backend != wgpu::Backend::Vulkan || info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }
        let name = info.name.clone();
        let Ok(gpu) = block_on(GpuContext::from_adapter(adapter)) else {
            continue;
        };
        let on_gpu = input.to_gpu_ctx(&gpu).unwrap();
        let out = gpu
            .canny(&on_gpu, 50.0_f32, 150.0_f32)
            .unwrap_or_else(|e| panic!("{name}: canny failed: {e}"));
        let back = out.to_cpu_ctx(&gpu).unwrap();
        let edges = count_nonzero(&back);
        assert!(
            edges > 0,
            "{name}: canny returned an all-zero edge map - the shader is reading the \
             input with the wrong element type"
        );
        println!("  {name}: {edges} edge pixels");
    }
}
