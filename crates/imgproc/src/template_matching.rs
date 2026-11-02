use cv_core::{Tensor, storage::Storage};
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
    let runner = cv_runtime::best_runner()
        .map(|(r, _)| r)
        .unwrap_or_else(|_| {
            // Fallback: use CPU registry if available
            cv_runtime::registry()
                .ok()
                .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
                .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
        });
    match_template_ctx(image, templ, method, &runner)
}

#[allow(clippy::needless_range_loop)]
pub fn match_template_ctx(
    image: &GrayImage,
    templ: &GrayImage,
    method: TemplateMatchMethod,
    group: &RuntimeRunner,
) -> MatchResult {
    if image.width() == 0 || image.height() == 0 {
        return MatchResult {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
    }
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

    // The GPU shader computes SqDiff and Ccorr only; for the other four
    // methods it writes 0.0 to every output and still returns Ok, so routing
    // them to the GPU silently produced an all-zero score map.
    if matches!(
        method,
        TemplateMatchMethod::SqDiff | TemplateMatchMethod::Ccorr
    ) && let Ok(ComputeDevice::Gpu(gpu)) = group.device()
        && let Ok(res) = match_template_gpu(gpu, image, templ, method)
    {
        return res;
    }

    let mut out = vec![0.0f32; (out_w * out_h) as usize];

    let tw = templ.width() as usize;
    let th = templ.height() as usize;
    let iw = image.width() as usize;
    let i_raw = image.as_raw();
    let t_raw = templ.as_raw();

    // Every window sum is a sum of integers, so it is accumulated exactly in
    // u64 and only the normalisation is floating point (f64), as OpenCV does
    // with its CV_64F integral images.
    let n = (tw * th) as f64;
    let t_sum: u64 = t_raw.iter().map(|&v| v as u64).sum();
    let t_sum2: u64 = t_raw.iter().map(|&v| (v as u64) * (v as u64)).sum();
    let t_mean = t_sum as f64 / n;
    // sum((T - mean(T))^2), the CCOEFF template norm squared.
    let t_var_sum = (t_sum2 as f64 - t_sum as f64 * t_mean).max(0.0);

    // `cv::matchTemplate`: a constant template has no correlation
    // coefficient, and OpenCV defines the TM_CCOEFF_NORMED result as 1.
    if method == TemplateMatchMethod::CcoeffNormed && t_var_sum / n < f64::EPSILON {
        return MatchResult {
            data: vec![1.0; (out_w * out_h) as usize],
            width: out_w,
            height: out_h,
        };
    }
    let t_norm = match method {
        TemplateMatchMethod::CcoeffNormed => t_var_sum.sqrt(),
        _ => (t_sum2 as f64).sqrt(),
    };

    group.run(|| {
        out.par_chunks_mut(out_w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    let mut sum_i = 0u64;
                    let mut sum_i_sq = 0u64;
                    let mut cross = 0u64;

                    for j in 0..th {
                        let src = &i_raw[(y + j) * iw + x..][..tw];
                        let tpl = &t_raw[j * tw..][..tw];
                        for (&iv, &tv) in src.iter().zip(tpl) {
                            let (iv, tv) = (iv as u64, tv as u64);
                            sum_i += iv;
                            sum_i_sq += iv * iv;
                            cross += iv * tv;
                        }
                    }
                    *o = score(method, sum_i, sum_i_sq, cross, n, t_mean, t_sum2, t_norm) as f32;
                }
            });
    });

    MatchResult {
        data: out,
        width: out_w,
        height: out_h,
    }
}

/// One output value of `cv::matchTemplate` (`common_matchTemplate` in
/// OpenCV's templmatch.cpp), from the window's exact sums.
///
/// The normed methods divide by `t = sqrt(window energy) * |T|` only while the
/// quotient is in range: `|num| < t` divides, `|num| < 1.125 t` saturates to
/// +-1, and anything larger (including `t == 0`) is 0 - or 1 for
/// TM_SQDIFF_NORMED, the "no match" end of its scale. A window with no
/// energy (all black for SQDIFF/CCORR, flat for CCOEFF) therefore scores 1
/// under TM_SQDIFF_NORMED; dividing by a zero denominator used to return 0,
/// i.e. a black window was reported as a perfect match, and unclamped
/// quotients reached 600+ where OpenCV returns at most 1.
#[allow(clippy::too_many_arguments)]
fn score(
    method: TemplateMatchMethod,
    sum_i: u64,
    sum_i_sq: u64,
    cross: u64,
    n: f64,
    t_mean: f64,
    t_sum2: u64,
    t_norm: f64,
) -> f64 {
    let wnd_sum2 = sum_i_sq as f64;
    let mut wnd_mean2 = 0.0;
    let mut num = cross as f64;
    match method {
        TemplateMatchMethod::Ccorr => return num,
        TemplateMatchMethod::Ccoeff | TemplateMatchMethod::CcoeffNormed => {
            let s = sum_i as f64;
            wnd_mean2 = s * s / n;
            num -= s * t_mean;
            if method == TemplateMatchMethod::Ccoeff {
                return num;
            }
        }
        TemplateMatchMethod::SqDiff | TemplateMatchMethod::SqDiffNormed => {
            // Exact in integers; the subtraction cannot go negative.
            num = (sum_i_sq + t_sum2) as f64 - 2.0 * num;
            if method == TemplateMatchMethod::SqDiff {
                return num;
            }
        }
        TemplateMatchMethod::CcorrNormed => {}
    }

    let diff2 = (wnd_sum2 - wnd_mean2).max(0.0);
    let t = if diff2 <= (10.0 * f32::EPSILON as f64 * wnd_sum2).min(0.5) {
        0.0 // flat window: avoid dividing rounding noise
    } else {
        diff2.sqrt() * t_norm
    };
    if num.abs() < t {
        num / t
    } else if num.abs() < t * 1.125 {
        num.signum()
    } else if method == TemplateMatchMethod::SqDiffNormed {
        1.0
    } else {
        0.0
    }
}

fn match_template_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    image: &GrayImage,
    templ: &GrayImage,
    method: TemplateMatchMethod,
) -> cv_hal::Result<MatchResult> {
    use cv_hal::context::ComputeContext;
    let img_f32: Vec<f32> = image.as_raw().iter().map(|&p| p as f32).collect();
    let img_tensor = cv_core::CpuTensor::from_vec(
        img_f32,
        cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let templ_f32: Vec<f32> = templ.as_raw().iter().map(|&p| p as f32).collect();
    let templ_tensor = cv_core::CpuTensor::from_vec(
        templ_f32,
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
