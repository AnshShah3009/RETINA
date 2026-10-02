use crate::threshold::ThresholdType;
use image::GrayImage;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalThresholdMethod {
    Niblack,
    Sauvola,
}

/// Validate the parameters that would otherwise silently corrupt the result.
///
/// * `block_size` must be at least 1 and odd (OpenCV requires an odd window size);
///   `block_size == 0` would make `half_block == 0` and every neighbourhood a
///   single pixel.
/// * `r` is the Sauvola dynamic range of the standard deviation. It is only
///   meaningful for [`LocalThresholdMethod::Sauvola`], where the threshold is
///   `mean * (1 + k * (std_dev / r - 1))`. With `r == 0` the term `std_dev / r`
///   is `0/0` or `x/0`, i.e. `NaN`/`+inf`, which makes the threshold `+inf`
///   and turns every pixel of the output into background (an all-black image).
///
/// # Returns
/// `Err` with a message naming the offending parameter when validation fails.
pub fn validate_local_threshold_params(
    method: LocalThresholdMethod,
    block_size: u32,
    r: f32,
) -> crate::Result<()> {
    if block_size == 0 {
        return Err(crate::ImgprocError::InvalidInput(
            "local_threshold: block_size must be greater than 0".into(),
        ));
    }
    if matches!(method, LocalThresholdMethod::Sauvola) && !(r.is_finite() && r > 0.0) {
        return Err(crate::ImgprocError::InvalidInput(format!(
            "local_threshold: Sauvola requires a finite r > 0 (dynamic range of the standard \
             deviation), got r = {}. r == 0 makes `std_dev / r` infinite, so every pixel is \
             thresholded away (all-black output)",
            r
        )));
    }
    Ok(())
}

/// Adaptive (local) thresholding.
///
/// # Arguments
/// * `src` - Source grayscale image.
/// * `max_value` - Maximum value written for "above threshold" pixels.
/// * `method` - [`LocalThresholdMethod::Niblack`] or [`LocalThresholdMethod::Sauvola`].
/// * `typ` - [`ThresholdType::Binary`] or [`ThresholdType::BinaryInv`].
/// * `block_size` - Side length of the neighbourhood, must be non-zero.
/// * `k` - Bias factor multiplying the local spread.
/// * `r` - Sauvola dynamic range; must be finite and `> 0` for Sauvola.
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] when `block_size == 0`, or when
/// `method` is Sauvola and `r` is not finite and positive (which previously
/// produced an all-black image instead of an error).
pub fn local_threshold(
    src: &GrayImage,
    max_value: u8,
    method: LocalThresholdMethod,
    typ: ThresholdType,
    block_size: u32,
    k: f32,
    r: f32,
) -> crate::Result<GrayImage> {
    validate_local_threshold_params(method, block_size, r)?;

    let width = src.width();
    let height = src.height();
    let mut dst = GrayImage::new(width, height);

    let half_block = (block_size / 2) as i32;

    // We can optimize this with integral images (one for sum, one for sum of squares)
    // For now, let's implement the sliding window version.

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let mut count = 0;

                for dy in -half_block..=half_block {
                    let sy = y as i32 + dy;
                    if sy < 0 || sy >= height as i32 {
                        continue;
                    }
                    for dx in -half_block..=half_block {
                        let sx = x as i32 + dx;
                        if sx < 0 || sx >= width as i32 {
                            continue;
                        }

                        let val = src.get_pixel(sx as u32, sy as u32)[0] as f32;
                        sum += val;
                        sum_sq += val * val;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let variance = (sum_sq / count as f32) - (mean * mean);
                let std_dev = variance.max(0.0).sqrt();

                let thresh = match method {
                    LocalThresholdMethod::Niblack => mean + k * std_dev,
                    LocalThresholdMethod::Sauvola => mean * (1.0 + k * (std_dev / r - 1.0)),
                };

                let src_val = src.get_pixel(x, y)[0] as f32;
                let binary_val = match typ {
                    ThresholdType::Binary => {
                        if src_val > thresh {
                            max_value
                        } else {
                            0
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if src_val > thresh {
                            0
                        } else {
                            max_value
                        }
                    }
                    _ => 0,
                };

                row[x as usize] = binary_val;
            }
        });

    Ok(dst)
}
