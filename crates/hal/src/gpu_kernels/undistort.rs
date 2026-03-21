use crate::context::{BorderMode, Interpolation};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{CameraIntrinsics, Distortion, Tensor, TensorShape};
use nalgebra::Matrix3;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UndistortCameraParams {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    ifx: f32,
    ify: f32,
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UndistortImageParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    interpolation: u32,
    border_mode: u32,
    border_val: f32,
    _pad: u32,
}

pub fn undistort(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    rectification: &Matrix3<f64>,
    new_intrinsics: &CameraIntrinsics,
    interpolation: Interpolation,
    border_mode: BorderMode<f32>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (src_h, src_w) = input.shape.hw();
    let dst_w = new_intrinsics.width;
    let dst_h = new_intrinsics.height;
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Undistort currently only for grayscale".into(),
        ));
    }

    let out_len = (dst_w * dst_h) as usize;
    let byte_size = (out_len * 4) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let cam_params = UndistortCameraParams {
        fx: intrinsics.fx as f32,
        fy: intrinsics.fy as f32,
        cx: intrinsics.cx as f32,
        cy: intrinsics.cy as f32,
        ifx: 1.0 / (intrinsics.fx as f32),
        ify: 1.0 / (intrinsics.fy as f32),
        k1: distortion.k1 as f32,
        k2: distortion.k2 as f32,
        p1: distortion.p1 as f32,
        p2: distortion.p2 as f32,
        k3: distortion.k3 as f32,
        _pad: 0,
    };

    let (b_mode, b_val) = match border_mode {
        BorderMode::Constant(v) => (0, v),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Wrap => (2, 0.0),
        BorderMode::Reflect => (3, 0.0),
        BorderMode::Reflect101 => (4, 0.0),
    };

    let img_params = UndistortImageParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w,
        dst_h,
        interpolation: match interpolation {
            Interpolation::Nearest => 0,
            _ => 1, // Default to bilinear for now
        },
        border_mode: b_mode,
        border_val: b_val,
        _pad: 0,
    };

    // Calculate rectification matrix for the shader: Inv(NewK) * Inv(R)
    // Actually in the shader we do: norm_pt = rect_mat * dst_pt
    // So rect_mat = Inv(R) * Inv(NewK)
    let inv_new_k = new_intrinsics
        .matrix()
        .try_inverse()
        .unwrap_or(Matrix3::identity());
    let inv_r = rectification.try_inverse().unwrap_or(Matrix3::identity());
    let rect_mat_f64 = inv_r * inv_new_k;

    let mut rect_mat_data = [0.0f32; 12];
    for c in 0..3 {
        for r in 0..3 {
            rect_mat_data[c * 4 + r] = rect_mat_f64[(r, c)] as f32;
        }
    }

    let cam_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Cam Params"),
            contents: bytemuck::bytes_of(&cam_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let img_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Image Params"),
            contents: bytemuck::bytes_of(&img_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let rect_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Rect Mat"),
            contents: bytemuck::cast_slice(&rect_mat_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/undistort.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Undistort Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cam_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: img_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: rect_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = dst_w.div_ceil(4).div_ceil(16);
        let y = dst_h.div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: TensorShape::new(c, dst_h as usize, dst_w as usize),
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
