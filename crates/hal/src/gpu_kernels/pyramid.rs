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

pub fn pyramid_down<T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
) -> Result<GpuTensor<T>> {
    let (h, w) = input.shape.hw();
    let new_w = w / 2;
    let new_h = h / 2;

    if new_w == 0 || new_h == 0 {
        return Err(crate::Error::InvalidInput(
            "Image too small to downsample".into(),
        ));
    }

    // Gaussian blur then resize (gaussian_blur is f32-only, so we use resize directly for non-f32)
    use cv_core::storage::Storage;
    if let Some(input_f32) = input
        .storage
        .as_any()
        .downcast_ref::<crate::storage::WgpuGpuStorage<f32>>()
    {
        use std::marker::PhantomData;
        let input_f32_tensor: crate::GpuTensor<f32> = cv_core::Tensor {
            storage: input_f32.clone(),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: PhantomData,
        };
        let blurred = crate::gpu_kernels::convolve::gaussian_blur(ctx, &input_f32_tensor, 1.0, 5)?;
        let scaled = crate::gpu_kernels::resize::resize(ctx, &blurred, new_w as u32, new_h as u32)?;
        // Safety: f32 is a special case of T (we only reach here when T == f32)
        let scaled_t =
            unsafe { std::mem::transmute::<crate::GpuTensor<f32>, crate::GpuTensor<T>>(scaled) };
        Ok(scaled_t)
    } else {
        // For non-f32: skip gaussian blur, just resize
        crate::gpu_kernels::resize::resize(ctx, input, new_w as u32, new_h as u32)
    }
}
