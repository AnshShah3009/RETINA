use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use crate::GpuTensor;
use crate::Result;
use cv_core::storage::Storage;
use cv_core::Tensor;

#[derive(Debug, Clone)]
pub struct ImagePyramid<T: cv_core::Float + bytemuck::Pod + 'static, S: Storage<T>> {
    pub levels: Vec<Tensor<T, S>>,
    pub scales: Vec<f32>,
}

pub fn build_pyramid<T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    n_levels: usize,
    scale_factor: f32,
) -> Result<ImagePyramid<T, WgpuGpuStorage<T>>> {
    let mut levels = Vec::with_capacity(n_levels);
    let mut scales = Vec::with_capacity(n_levels);

    levels.push(input.clone());
    scales.push(1.0);

    let mut current = input.clone();
    let mut current_scale = 1.0;

    for _ in 1..n_levels {
        current_scale *= scale_factor;
        let inv_scale = 1.0 / current_scale;

        let (h_orig, w_orig) = input.shape.hw();
        let new_w = (w_orig as f32 * inv_scale).round() as usize;
        let new_h = (h_orig as f32 * inv_scale).round() as usize;

        // Ensure minimum size
        if new_w < 16 || new_h < 16 {
            break;
        }

        let next = crate::gpu_kernels::resize::resize(ctx, &current, new_w as u32, new_h as u32)?;
        levels.push(next.clone());
        scales.push(current_scale);
        current = next;
    }

    Ok(ImagePyramid { levels, scales })
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PyrDownParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub fn pyramid_down<T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
) -> Result<GpuTensor<T>> {
    let (h, w) = input.shape.hw();
    // OpenCV `pyrDown`: `((w + 1) / 2, (h + 1) / 2)`.
    let new_w = w.div_ceil(2);
    let new_h = h.div_ceil(2);

    if w == 0 || h == 0 || new_w == 0 || new_h == 0 {
        return Err(crate::Error::InvalidInput(
            "Image too small to downsample".into(),
        ));
    }

    use cv_core::storage::Storage;
    if let Some(input_f32) = input
        .storage
        .as_any()
        .downcast_ref::<crate::storage::WgpuGpuStorage<f32>>()
    {
        use std::marker::PhantomData;
        use std::sync::Arc;
        use wgpu::util::DeviceExt;
        let c = input.shape.channels.max(1);
        let out_len = c * new_h * new_w;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PyrDown"),
            size: (out_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = PyrDownParams {
            src_w: w as u32,
            src_h: h as u32,
            dst_w: new_w as u32,
            dst_h: new_h as u32,
            channels: c as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PyrDown Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let shader = include_str!("../../shaders/pyramid_down.wgsl");
        let pipeline = ctx.create_compute_pipeline(shader, "main");
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PyrDown"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_f32.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PyrDown"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                (new_w as u32).div_ceil(16),
                (new_h as u32).div_ceil(16),
                c as u32,
            );
        }
        ctx.submit(enc);
        let scaled = cv_core::Tensor {
            storage: crate::storage::WgpuGpuStorage::<f32>::from_buffer(
                Arc::new(output_buffer),
                out_len,
            ),
            shape: cv_core::TensorShape::new(c, new_h, new_w),
            dtype: input.dtype,
            _phantom: PhantomData,
        };
        let scaled_t = cv_core::Tensor {
            storage: scaled.storage.retype::<T>(),
            shape: scaled.shape,
            dtype: scaled.dtype,
            _phantom: PhantomData,
        };
        Ok(scaled_t)
    } else {
        crate::gpu_kernels::resize::resize(ctx, input, new_w as u32, new_h as u32)
    }
}
