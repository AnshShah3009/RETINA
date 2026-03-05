use cv_core::{storage::Storage, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::TemplateMatchMethod;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl MatchResult {
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.width + x) as usize]
    }
}

pub fn match_template(
    image: &GrayImage,
    templ: &GrayImage,
    method: TemplateMatchMethod,
) -> MatchResult {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    match_template_ctx(image, templ, method, &runner)
}

pub fn match_template_ctx(
    image: &GrayImage,
    templ: &GrayImage,
    method: TemplateMatchMethod,
    group: &RuntimeRunner,
) -> MatchResult {
    assert!(
        image.width() >= templ.width() && image.height() >= templ.height(),
        "template must fit inside source image"
    );
    assert!(
        templ.width() > 0 && templ.height() > 0,
        "template cannot be empty"
    );

    let out_w = image.width() - templ.width() + 1;
    let out_h = image.height() - templ.height() + 1;

    // Check for GPU acceleration
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(res) = match_template_gpu(gpu, image, templ, method) {
            return res;
        }
    }

    let mut out = vec![0.0f32; (out_w * out_h) as usize];

    let tw = templ.width() as usize;
    let th = templ.height() as usize;
    let t_raw = templ.as_raw();

    let t_mean = mean_u8(t_raw);
    let t_sq_sum = sum_sq_u8(t_raw);
    let t_var_sum = t_raw
        .iter()
        .map(|&v| {
            let d = v as f32 - t_mean;
            d * d
        })
        .sum::<f32>();

    group.run(|| {
        out.par_chunks_mut(out_w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..out_w as usize {
                    let mut sum_i = 0.0f32;
                    let mut sum_i_sq = 0.0f32;
                    let mut cross = 0.0f32;

                    for j in 0..th {
                        let src_row = (y + j) as u32;
                        for i in 0..tw {
                            let src_col = (x + i) as u32;
                            let iv = image.get_pixel(src_col, src_row)[0] as f32;
                            let tv = t_raw[j * tw + i] as f32;
                            sum_i += iv;
                            sum_i_sq += iv * iv;
                            cross += iv * tv;
                        }
                    }

                    let n = (tw * th) as f32;
                    let i_mean = sum_i / n;

                    row[x] = match method {
                        TemplateMatchMethod::SqDiff => {
                            // ||I - T||^2 = ||I||^2 + ||T||^2 - 2 * <I, T>
                            sum_i_sq + t_sq_sum - 2.0 * cross
                        }
                        TemplateMatchMethod::SqDiffNormed => {
                            let sqdiff = sum_i_sq + t_sq_sum - 2.0 * cross;
                            let denom = (sum_i_sq * t_sq_sum).sqrt();
                            if denom > 1e-12 {
                                sqdiff / denom
                            } else {
                                0.0
                            }
                        }
                        TemplateMatchMethod::Ccorr => cross,
                        TemplateMatchMethod::CcorrNormed => {
                            let denom = (sum_i_sq * t_sq_sum).sqrt();
                            if denom > 1e-12 {
                                cross / denom
                            } else {
                                0.0
                            }
                        }
                        TemplateMatchMethod::Ccoeff => {
                            // sum((I - meanI) * (T - meanT)) = <I,T> - N*meanI*meanT
                            cross - n * i_mean * t_mean
                        }
                        TemplateMatchMethod::CcoeffNormed => {
                            let coeff = cross - n * i_mean * t_mean;
                            let i_var = sum_i_sq - n * i_mean * i_mean;
                            let denom = (i_var * t_var_sum).sqrt();
                            if denom > 1e-12 {
                                coeff / denom
                            } else {
                                0.0
                            }
                        }
                    };
                }
            });
    });

    MatchResult {
        data: out,
        width: out_w,
        height: out_h,
    }
}

fn match_template_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    image: &GrayImage,
    templ: &GrayImage,
    method: TemplateMatchMethod,
) -> cv_hal::Result<MatchResult> {
    use cv_hal::context::ComputeContext;
    let img_tensor = cv_core::CpuTensor::from_vec(
        image.as_raw().to_vec(),
        cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let templ_tensor = cv_core::CpuTensor::from_vec(
        templ.as_raw().to_vec(),
        cv_core::TensorShape::new(1, templ.height() as usize, templ.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let img_gpu = img_tensor.to_gpu_ctx(gpu)?;
    let templ_gpu = templ_tensor.to_gpu_ctx(gpu)?;

    // Output is f32 score map
    let res_gpu: Tensor<f32, cv_hal::storage::GpuStorage<f32>> =
        gpu.match_template(&img_gpu, &templ_gpu, method)?;
    let res_cpu = res_gpu.to_cpu_ctx(gpu)?;

    let out_w = image.width() - templ.width() + 1;
    let out_h = image.height() - templ.height() + 1;

    let data = res_cpu
        .storage
        .as_slice()
        .ok_or_else(|| {
            cv_hal::Error::RuntimeError(
                "Template match GPU result not accessible as CPU slice".to_string(),
            )
        })?
        .to_vec();
    Ok(MatchResult {
        data,
        width: out_w,
        height: out_h,
    })
}

pub fn min_max_loc(result: &MatchResult) -> ((u32, u32, f32), (u32, u32, f32)) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut min_xy = (0u32, 0u32);
    let mut max_xy = (0u32, 0u32);

    for y in 0..result.height {
        for x in 0..result.width {
            let v = result.get(x, y);
            if v < min_val {
                min_val = v;
                min_xy = (x, y);
            }
            if v > max_val {
                max_val = v;
                max_xy = (x, y);
            }
        }
    }

    ((min_xy.0, min_xy.1, min_val), (max_xy.0, max_xy.1, max_val))
}

fn mean_u8(values: &[u8]) -> f32 {
    values.iter().map(|&v| v as f32).sum::<f32>() / values.len() as f32
}

fn sum_sq_u8(values: &[u8]) -> f32 {
    values
        .iter()
        .map(|&v| {
            let f = v as f32;
            f * f
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn ccorr_finds_exact_patch_at_max() {
        let mut img = GrayImage::new(8, 8);
        for y in 2..5 {
            for x in 3..6 {
                img.put_pixel(x, y, Luma([200]));
            }
        }

        let mut templ = GrayImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                templ.put_pixel(x, y, Luma([200]));
            }
        }

        let res = match_template(&img, &templ, TemplateMatchMethod::CcorrNormed);
        let (_min, max) = min_max_loc(&res);
        assert_eq!((max.0, max.1), (3, 2));
    }

    #[test]
    fn sqdiff_finds_exact_patch_at_min() {
        let mut img = GrayImage::new(8, 8);
        for y in 1..4 {
            for x in 2..5 {
                img.put_pixel(x, y, Luma([120]));
            }
        }

        let mut templ = GrayImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                templ.put_pixel(x, y, Luma([120]));
            }
        }

        let res = match_template(&img, &templ, TemplateMatchMethod::SqDiff);
        let (min, _max) = min_max_loc(&res);
        assert_eq!((min.0, min.1), (2, 1));
        assert!(min.2.abs() < 1e-5);
    }
}
