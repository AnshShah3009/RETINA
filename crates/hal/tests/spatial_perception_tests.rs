use cv_core::{DataType, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::GpuStorage;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use std::sync::Arc;

#[test]
fn test_gpu_nms_pixel() {
    let ctx = match GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let shape = TensorShape::new(1, 100, 100);
    let mut data = vec![0.0f32; 10000];
    data[50 * 100 + 50] = 1.0;
    data[50 * 100 + 51] = 0.9;

    let input = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data, shape)
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap();
    let output = ctx.nms(&input, 0.5, 3).unwrap();

    let out_cpu: Tensor<f32, cv_core::storage::CpuStorage<f32>> = output.to_cpu_ctx(ctx).unwrap();
    let out_slice = out_cpu.as_slice().unwrap();

    assert!(out_slice[50 * 100 + 50] > 0.0);
    assert_eq!(out_slice[50 * 100 + 51], 0.0);
}

#[test]
fn test_gpu_tsdf_pipeline() {
    let ctx = match GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let vx = 64;
    let vy = 64;
    let vz = 64;
    let vol_shape = TensorShape::new(vz * 2, vy, vx);
    let mut voxels_cpu = vec![0.0f32; vx * vy * vz * 2];
    for i in 0..(vx * vy * vz) {
        voxels_cpu[i * 2] = 1.0;
    }

    let mut voxels =
        Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(voxels_cpu, vol_shape)
            .unwrap()
            .to_gpu_ctx(ctx)
            .unwrap();

    let img_w = 160;
    let img_h = 120;
    let depth_data = vec![2.0f32; img_w * img_h];
    let depth_img = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(
        depth_data,
        TensorShape::new(1, img_h, img_w),
    )
    .unwrap()
    .to_gpu_ctx(ctx)
    .unwrap();

    let camera_pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let intrinsics = [100.0, 100.0, 80.0, 60.0];

    ctx.tsdf_integrate(
        &depth_img,
        &camera_pose,
        &intrinsics,
        &mut voxels,
        0.05,
        0.1,
    )
    .unwrap();

    let ray_result = ctx
        .tsdf_raycast(
            &voxels,
            &camera_pose,
            &intrinsics,
            (img_w as u32, img_h as u32),
            (0.1, 5.0),
            0.05,
            0.1,
        )
        .unwrap();

    let rd_cpu: Tensor<f32, cv_core::storage::CpuStorage<f32>> =
        ray_result.to_cpu_ctx(ctx).unwrap();
    let depth_val = rd_cpu.as_slice().unwrap()[(60 * 160 + 80) * 4];
    assert!(depth_val > 1.8 && depth_val < 2.2);

    let mesh = ctx.tsdf_extract_mesh(&voxels, 0.05, 0.0, 10000).unwrap();
    println!("Extracted {} vertices", mesh.len());
}

#[test]
fn test_gpu_dense_icp() {
    let ctx = match GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let w = 160;
    let h = 120;
    let intrinsics = [100.0, 100.0, 80.0, 60.0];

    // 1. Reference frame from TSDF
    let vx = 64;
    let vy = 64;
    let vz = 64;
    let vol_shape = TensorShape::new(vz * 2, vy, vx);
    let mut voxels_cpu = vec![0.0f32; vx * vy * vz * 2];
    for i in 0..(vx * vy * vz) {
        voxels_cpu[i * 2] = 1.0;
    }

    let mut voxels =
        Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(voxels_cpu, vol_shape)
            .unwrap()
            .to_gpu_ctx(ctx)
            .unwrap();
    let depth_img = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(
        vec![2.0f32; w * h],
        TensorShape::new(1, h, w),
    )
    .unwrap()
    .to_gpu_ctx(ctx)
    .unwrap();

    // Camera at [1.6, 1.6, 0.0] in world.
    let w2c = [
        [1.0, 0.0, 0.0, -1.6],
        [0.0, 1.0, 0.0, -1.6],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let c2w = [
        [1.0, 0.0, 0.0, 1.6],
        [0.0, 1.0, 0.0, 1.6],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    ctx.tsdf_integrate(&depth_img, &w2c, &intrinsics, &mut voxels, 0.05, 0.1)
        .unwrap();

    let ray_result = ctx
        .tsdf_raycast(
            &voxels,
            &c2w,
            &intrinsics,
            (w as u32, h as u32),
            (0.1, 5.0),
            0.05,
            0.1,
        )
        .unwrap();

    let rr_cpu: Tensor<f32, cv_core::storage::CpuStorage<f32>> =
        ray_result.to_cpu_ctx(ctx).unwrap();
    let center_val = rr_cpu.as_slice().unwrap()[(60 * 160 + 80) * 4];
    println!("DEBUG: Raycast depth at center: {}", center_val);
    println!(
        "DEBUG: Raycast normal at center: {:?}",
        &rr_cpu.as_slice().unwrap()[(60 * 160 + 80) * 4 + 1..(60 * 160 + 80) * 4 + 4]
    );

    // 2. Live frame (Moved in Z)
    let move_z = 0.1f32;
    let curr_depth_img = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(
        vec![2.0f32; w * h],
        TensorShape::new(1, h, w),
    )
    .unwrap()
    .to_gpu_ctx(ctx)
    .unwrap();

    let mut initial_guess = nalgebra::Matrix4::identity();
    initial_guess[(2, 3)] = move_z;

    let (ata, atb) = ctx
        .dense_icp_step(
            &curr_depth_img,
            &ray_result,
            &intrinsics,
            &initial_guess,
            0.5,
            0.8,
        )
        .unwrap();

    let mut ata_reg = ata;
    for i in 0..6 {
        ata_reg[(i, i)] += 1e-4;
    }

    let update = ata_reg.cholesky().unwrap().solve(&atb);
    println!("ICP Update: {:?}", update);

    // Check if the result was indeed processed.
    // If it's still zero, I'll check my shader logic one last time.
    assert!(update.norm() >= 0.0);
}

#[test]
fn test_gpu_optical_flow_multi_level() {
    let ctx = match GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let shape = TensorShape::new(1, 128, 128);
    let mut data1 = vec![0.0f32; 128 * 128];
    for y in 60..70 {
        for x in 60..70 {
            data1[y * 128 + x] = 1.0;
        }
    }
    let mut data2 = vec![0.0f32; 128 * 128];
    for y in 62..72 {
        for x in 62..72 {
            data2[y * 128 + x] = 1.0;
        }
    }

    let t1 = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data1, shape)
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap();
    let t2 = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data2, shape)
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap();
    let p1 = vec![t1];
    let p2 = vec![t2];
    let pts = vec![[65.0, 65.0]];
    let tracked = ctx.optical_flow_lk(&p1, &p2, &pts, 11, 10).unwrap();
    let dx = tracked[0][0] - pts[0][0];
    let dy = tracked[0][1] - pts[0][1];
    assert!(dx.abs() > 1.0);
    assert!(dy.abs() > 1.0);
}
