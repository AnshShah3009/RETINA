use crate::{BorderMode, Interpolation};
use cv_core::{storage::Storage, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use nalgebra::{Matrix3, Point2};
use rayon::prelude::*;

pub fn get_pixel_bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
    get_pixel_bilinear_with_border(img, x, y, BorderMode::Constant(0))
}

fn get_pixel_bilinear_with_border(img: &GrayImage, x: f32, y: f32, border: BorderMode) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let v00 = sample_pixel(img, x0, y0, border);
    let v10 = sample_pixel(img, x1, y0, border);
    let v01 = sample_pixel(img, x0, y1, border);
    let v11 = sample_pixel(img, x1, y1, border);

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    v0 * (1.0 - fy) + v1 * fy
}

fn get_pixel_nearest_with_border(img: &GrayImage, x: f32, y: f32, border: BorderMode) -> f32 {
    let xi = x.round() as isize;
    let yi = y.round() as isize;
    sample_pixel(img, xi, yi, border)
}

fn sample_pixel(img: &GrayImage, x: isize, y: isize, border: BorderMode) -> f32 {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let raw = img.as_raw();

    match (map_coord(x, width, border), map_coord(y, height, border)) {
        (Some(ix), Some(iy)) => raw[iy * width + ix] as f32,
        _ => match border {
            BorderMode::Constant(v) => v as f32,
            _ => 0.0,
        },
    }
}

fn map_coord(coord: isize, len: usize, mode: BorderMode) -> Option<usize> {
    let n = len as isize;
    if n <= 0 {
        return None;
    }

    match mode {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= n {
                None
            } else {
                Some(coord as usize)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, n - 1) as usize),
        BorderMode::Wrap => {
            let mut c = coord % n;
            if c < 0 {
                c += n;
            }
            Some(c as usize)
        }
        BorderMode::Reflect => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c - 1;
            }
            Some(c as usize)
        }
        BorderMode::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c;
            }
            Some(c as usize)
        }
    }
}

fn interpolate_sample(
    src: &GrayImage,
    x: f32,
    y: f32,
    interpolation: Interpolation,
    border: BorderMode,
) -> f32 {
    match interpolation {
        Interpolation::Nearest => get_pixel_nearest_with_border(src, x, y, border),
        Interpolation::Linear | Interpolation::Cubic | Interpolation::Lanczos => {
            get_pixel_bilinear_with_border(src, x, y, border)
        }
    }
}

pub fn warp_perspective_ex(
    src: &GrayImage,
    matrix: &Matrix3<f32>,
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
) -> GrayImage {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    warp_perspective_ex_ctx(src, matrix, width, height, interpolation, border, &runner)
}

pub fn warp_perspective_ex_ctx(
    src: &GrayImage,
    matrix: &Matrix3<f32>,
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    // Check for GPU acceleration
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if interpolation == Interpolation::Linear && border == BorderMode::Constant(0) {
            if let Ok(result) = warp_gpu(
                gpu,
                src,
                matrix,
                width,
                height,
                cv_hal::context::WarpType::Perspective,
            ) {
                return result;
            }
        }
    }

    let mut dst = GrayImage::new(width, height);

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let pt = Point2::new(x as f32, y as f32);
                let src_pt = transform_point(matrix, &pt);
                let val = interpolate_sample(src, src_pt.x, src_pt.y, interpolation, border);
                row[x as usize] = val.clamp(0.0, 255.0) as u8;
            }
        });

    dst
}

fn warp_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    matrix: &Matrix3<f32>,
    width: u32,
    height: u32,
    typ: cv_hal::context::WarpType,
) -> cv_hal::Result<GrayImage> {
    use cv_hal::context::ComputeContext;

    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    // Matrix to 3x3 array
    let mut mat_data = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            mat_data[r][c] = matrix[(r, c)];
        }
    }

    let res_gpu = gpu.warp(
        &input_gpu,
        &mat_data,
        (width as usize, height as usize),
        typ,
    )?;

    let res_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = res_gpu.to_cpu()?;
    let res_data: Vec<u8> = res_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect();

    GrayImage::from_raw(width, height, res_data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

pub fn warp_perspective(
    src: &GrayImage,
    matrix: &Matrix3<f32>,
    width: u32,
    height: u32,
) -> GrayImage {
    warp_perspective_ex(
        src,
        matrix,
        width,
        height,
        Interpolation::Linear,
        BorderMode::Constant(0),
    )
}

pub fn warp_affine(
    src: &GrayImage,
    matrix_2x3: [[f32; 3]; 2],
    width: u32,
    height: u32,
) -> GrayImage {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    warp_affine_ctx(src, matrix_2x3, width, height, &runner)
}

pub fn warp_affine_ctx(
    src: &GrayImage,
    matrix_2x3: [[f32; 3]; 2],
    width: u32,
    height: u32,
    group: &RuntimeRunner,
) -> GrayImage {
    warp_affine_ex_ctx(
        src,
        matrix_2x3,
        width,
        height,
        Interpolation::Linear,
        BorderMode::Constant(0),
        group,
    )
}

pub fn warp_affine_ex(
    src: &GrayImage,
    matrix_2x3: [[f32; 3]; 2],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
) -> GrayImage {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    warp_affine_ex_ctx(
        src,
        matrix_2x3,
        width,
        height,
        interpolation,
        border,
        &runner,
    )
}

pub fn warp_affine_ex_ctx(
    src: &GrayImage,
    matrix_2x3: [[f32; 3]; 2],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    let matrix = Matrix3::new(
        matrix_2x3[0][0],
        matrix_2x3[0][1],
        matrix_2x3[0][2],
        matrix_2x3[1][0],
        matrix_2x3[1][1],
        matrix_2x3[1][2],
        0.0,
        0.0,
        1.0,
    );

    // Check for GPU acceleration
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if interpolation == Interpolation::Linear && border == BorderMode::Constant(0) {
            if let Ok(result) = warp_gpu(
                gpu,
                src,
                &matrix,
                width,
                height,
                cv_hal::context::WarpType::Affine,
            ) {
                return result;
            }
        }
    }

    // Warp uses inverse mapping from destination -> source coordinates.
    let inv = matrix.try_inverse().unwrap_or(matrix);
    warp_perspective_ex_ctx(src, &inv, width, height, interpolation, border, group)
}

pub fn remap_ex(
    src: &GrayImage,
    map_x: &[f32],
    map_y: &[f32],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
) -> GrayImage {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    remap_ex_ctx(
        src,
        map_x,
        map_y,
        width,
        height,
        interpolation,
        border,
        &runner,
    )
}

pub fn remap_ex_ctx(
    src: &GrayImage,
    map_x: &[f32],
    map_y: &[f32],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
    group: &cv_runtime::orchestrator::RuntimeRunner,
) -> GrayImage {
    // Check for GPU acceleration
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(result) = remap_gpu(gpu, src, map_x, map_y, width, height, interpolation, border)
        {
            return result;
        }
    }

    let mut dst = GrayImage::new(width, height);

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let sx = map_x[idx];
                let sy = map_y[idx];
                let val = interpolate_sample(src, sx, sy, interpolation, border);
                row[x as usize] = val.clamp(0.0, 255.0) as u8;
            }
        });

    dst
}

fn remap_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    map_x: &[f32],
    map_y: &[f32],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
) -> cv_hal::Result<GrayImage> {
    use cv_hal::context::ComputeContext;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let map_x_tensor = cv_core::CpuTensor::from_vec(
        map_x.to_vec(),
        cv_core::TensorShape::new(1, height as usize, width as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let map_y_tensor = cv_core::CpuTensor::from_vec(
        map_y.to_vec(),
        cv_core::TensorShape::new(1, height as usize, width as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;
    let mx_gpu = map_x_tensor.to_gpu_ctx(gpu)?;
    let my_gpu = map_y_tensor.to_gpu_ctx(gpu)?;

    let interp = match interpolation {
        Interpolation::Nearest => cv_hal::context::Interpolation::Nearest,
        Interpolation::Linear => cv_hal::context::Interpolation::Linear,
        Interpolation::Cubic => cv_hal::context::Interpolation::Cubic,
        Interpolation::Lanczos => cv_hal::context::Interpolation::Lanczos,
    };

    let border_hal = match border {
        BorderMode::Constant(v) => cv_hal::context::BorderMode::Constant(v as f32),
        BorderMode::Replicate => cv_hal::context::BorderMode::Replicate,
        BorderMode::Reflect => cv_hal::context::BorderMode::Reflect,
        BorderMode::Reflect101 => cv_hal::context::BorderMode::Reflect101,
        BorderMode::Wrap => cv_hal::context::BorderMode::Wrap,
    };

    let res_gpu = gpu.remap::<f32, GpuStorage<f32>, GpuStorage<f32>>(
        &input_gpu, &mx_gpu, &my_gpu, interp, border_hal,
    )?;

    let res_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = res_gpu.to_cpu_ctx(gpu)?;
    use cv_core::storage::Storage;
    let res_data: Vec<u8> = res_cpu
        .storage
        .as_slice()
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to get CPU slice".into()))?
        .iter()
        .map(|v: &f32| v.clamp(0.0, 255.0) as u8)
        .collect();

    GrayImage::from_raw(width, height, res_data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

pub fn remap(
    src: &GrayImage,
    map_x: &[f32],
    map_y: &[f32],
    width: u32,
    height: u32,
    interpolation: Interpolation,
    border: BorderMode,
) -> GrayImage {
    assert_eq!(
        map_x.len(),
        (width * height) as usize,
        "map_x size must equal width*height"
    );
    assert_eq!(
        map_y.len(),
        (width * height) as usize,
        "map_y size must equal width*height"
    );

    let mut dst = GrayImage::new(width, height);

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let sx = map_x[idx];
                let sy = map_y[idx];
                let val = interpolate_sample(src, sx, sy, interpolation, border);
                row[x as usize] = val.clamp(0.0, 255.0) as u8;
            }
        });

    dst
}

pub fn remap_identity(
    src: &GrayImage,
    interpolation: Interpolation,
    border: BorderMode,
) -> GrayImage {
    let mut map_x = vec![0.0f32; (src.width() * src.height()) as usize];
    let mut map_y = vec![0.0f32; (src.width() * src.height()) as usize];

    for y in 0..src.height() {
        for x in 0..src.width() {
            let idx = (y * src.width() + x) as usize;
            map_x[idx] = x as f32;
            map_y[idx] = y as f32;
        }
    }

    remap(
        src,
        &map_x,
        &map_y,
        src.width(),
        src.height(),
        interpolation,
        border,
    )
}

fn transform_point(matrix: &Matrix3<f32>, pt: &Point2<f32>) -> Point2<f32> {
    let x = pt.x;
    let y = pt.y;

    let w = matrix[(2, 0)] * x + matrix[(2, 1)] * y + matrix[(2, 2)];

    if w.abs() > 1e-10 {
        Point2::new(
            (matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)]) / w,
            (matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)]) / w,
        )
    } else {
        Point2::new(
            matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)],
            matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)],
        )
    }
}

pub fn get_rotation_matrix(center: Point2<f32>, angle: f32, scale: f32) -> Matrix3<f32> {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let tx = center.x * (1.0 - scale * cos_a) + center.y * scale * sin_a;
    let ty = center.y * (1.0 - scale * cos_a) - center.x * scale * sin_a;

    Matrix3::new(
        scale * cos_a,
        scale * sin_a,
        tx,
        -scale * sin_a,
        scale * cos_a,
        ty,
        0.0,
        0.0,
        1.0,
    )
}

pub fn get_translation_matrix(dx: f32, dy: f32) -> Matrix3<f32> {
    Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)
}

pub fn get_scaling_matrix(sx: f32, sy: f32) -> Matrix3<f32> {
    Matrix3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn remap_identity_keeps_pixels() {
        let mut img = GrayImage::new(6, 4);
        img.put_pixel(2, 1, Luma([200]));
        img.put_pixel(4, 3, Luma([123]));

        let out = remap_identity(&img, Interpolation::Nearest, BorderMode::Replicate);
        assert_eq!(out.get_pixel(2, 1)[0], 200);
        assert_eq!(out.get_pixel(4, 3)[0], 123);
    }

    #[test]
    fn warp_affine_translation_moves_point() {
        let mut img = GrayImage::new(8, 8);
        img.put_pixel(2, 2, Luma([255]));

        // dst(x,y) = src(x-2, y-1)
        let m = [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        let out = warp_affine_ex(
            &img,
            m,
            8,
            8,
            Interpolation::Nearest,
            BorderMode::Constant(0),
        );
        assert_eq!(out.get_pixel(4, 3)[0], 255);
    }

    #[test]
    fn warp_perspective_identity_preserves_point() {
        let mut img = GrayImage::new(7, 7);
        img.put_pixel(5, 4, Luma([180]));
        let i = Matrix3::identity();
        let out = warp_perspective(&img, &i, 7, 7);
        assert_eq!(out.get_pixel(5, 4)[0], 180);
    }
}
