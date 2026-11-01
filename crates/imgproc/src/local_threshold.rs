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
/// The neighbourhood sum and sum of squares come from integral images, so the
/// cost is one pass over the image plus a constant-time lookup per pixel.
/// Out-of-range window pixels are still omitted, the same clipped rectangle
/// the old nested loops summed. The sums are exact integer totals (f64); the
/// mean and standard deviation are then the same f32 expressions as before.
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
    if width == 0 || height == 0 {
        return Ok(dst);
    }

    let half_block = (block_size / 2) as i32;
    let (w, h) = (width as usize, height as usize);
    // Exact window sums. A global f32 integral would round a large image's
    // sum of squares; f64 holds every practical image exactly (pixels are
    // integers, and 2^53 is far past 255² times the pixel count). The
    // threshold below still uses f32, matching the old per-window f32 sum
    // whenever that sum itself was an exact f32 integer.
    let (sum_ii, sq_ii) = integral_sum_and_squares(src.as_raw(), w, h);
    let stride = w + 1;

    dst.as_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let y_i = y as i32;
            let y0 = (y_i - half_block).max(0) as usize;
            let y1 = (y_i + half_block + 1).min(height as i32) as usize;
            for x in 0..width {
                let x_i = x as i32;
                let x0 = (x_i - half_block).max(0) as usize;
                let x1 = (x_i + half_block + 1).min(width as i32) as usize;
                let sum = rect(&sum_ii, stride, x0, y0, x1, y1) as f32;
                let sum_sq = rect(&sq_ii, stride, x0, y0, x1, y1) as f32;
                let count = (x1 - x0) * (y1 - y0);

                let mean = sum / count as f32;
                let variance = (sum_sq / count as f32) - (mean * mean);
                let std_dev = variance.max(0.0).sqrt();

                let thresh = match method {
                    LocalThresholdMethod::Niblack => mean + k * std_dev,
                    LocalThresholdMethod::Sauvola => mean * (1.0 + k * (std_dev / r - 1.0)),
                };

                let src_val = src.get_pixel(x, y as u32)[0] as f32;
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

/// Inclusive-prefix integrals of the pixel values and of their squares.
/// Index `(y+1, x+1)` is the sum of `src[0..y][0..x]`.
fn integral_sum_and_squares(src: &[u8], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let stride = w + 1;
    let mut sum = vec![0.0f64; stride * (h + 1)];
    let mut sum_sq = vec![0.0f64; stride * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0.0f64;
        let mut row_sq = 0.0f64;
        let src_row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let v = src_row[x] as f64;
            row_sum += v;
            row_sq += v * v;
            let here = (y + 1) * stride + (x + 1);
            let above = y * stride + (x + 1);
            sum[here] = sum[above] + row_sum;
            sum_sq[here] = sum_sq[above] + row_sq;
        }
    }
    (sum, sum_sq)
}

fn rect(ii: &[f64], stride: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    ii[y1 * stride + x1] - ii[y0 * stride + x1] - ii[y1 * stride + x0] + ii[y0 * stride + x0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    /// The previous body: a fresh f32 sum of the clipped window at every pixel.
    fn sliding_window(
        src: &GrayImage,
        max_value: u8,
        method: LocalThresholdMethod,
        typ: ThresholdType,
        block_size: u32,
        k: f32,
        r: f32,
    ) -> GrayImage {
        let width = src.width();
        let height = src.height();
        let mut dst = GrayImage::new(width, height);
        let half = (block_size / 2) as i32;
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let mut count = 0i32;
                for dy in -half..=half {
                    let sy = y as i32 + dy;
                    if sy < 0 || sy >= height as i32 {
                        continue;
                    }
                    for dx in -half..=half {
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
                let bit = src_val > thresh;
                let on = match typ {
                    ThresholdType::BinaryInv => !bit,
                    _ => bit,
                };
                dst.put_pixel(x, y, Luma([if on { max_value } else { 0 }]));
            }
        }
        dst
    }

    fn noise(w: u32, h: u32, seed: u64) -> GrayImage {
        let mut s = seed;
        let mut img = GrayImage::new(w, h);
        for p in img.pixels_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = Luma([(s >> 33) as u8]);
        }
        img
    }

    /// Windows whose sum of squares fits in an f32 integer match the old loop
    /// exactly: 15² · 255² = 14_630_625 < 2²⁴. A 31-wide window does not, so
    /// the old running f32 sum was already rounded; the integral keeps the
    /// integer total and the two masks can differ.
    #[test]
    fn integral_matches_the_sliding_window_when_the_sum_is_exact() {
        let img = noise(48, 36, 7);
        for (method, r) in [
            (LocalThresholdMethod::Niblack, 0.0),
            (LocalThresholdMethod::Sauvola, 128.0),
        ] {
            for block in [3u32, 11, 15] {
                let got = local_threshold(&img, 255, method, ThresholdType::Binary, block, 0.2, r)
                    .unwrap();
                let old = sliding_window(&img, 255, method, ThresholdType::Binary, block, 0.2, r);
                assert_eq!(got.as_raw(), old.as_raw(), "{method:?} block {block}");
            }
        }
    }

    #[test]
    #[ignore]
    fn time_against_the_sliding_window() {
        let img = noise(640, 480, 1);
        let t0 = std::time::Instant::now();
        let _ = local_threshold(
            &img,
            255,
            LocalThresholdMethod::Sauvola,
            ThresholdType::Binary,
            31,
            0.2,
            128.0,
        )
        .unwrap();
        let neu = t0.elapsed();
        let t1 = std::time::Instant::now();
        // Same per-pixel loops as the old body, still parallel across rows,
        // so the comparison is the window sum and not the loss of rayon.
        let mut old_img = GrayImage::new(640, 480);
        old_img
            .as_mut()
            .par_chunks_mut(640)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..640u32 {
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    let mut count = 0i32;
                    for dy in -15i32..=15 {
                        let sy = y as i32 + dy;
                        if !(0..480).contains(&sy) {
                            continue;
                        }
                        for dx in -15i32..=15 {
                            let sx = x as i32 + dx;
                            if !(0..640).contains(&sx) {
                                continue;
                            }
                            let val = img.get_pixel(sx as u32, sy as u32)[0] as f32;
                            sum += val;
                            sum_sq += val * val;
                            count += 1;
                        }
                    }
                    let mean = sum / count as f32;
                    let variance = (sum_sq / count as f32) - (mean * mean);
                    let std_dev = variance.max(0.0).sqrt();
                    let thresh = mean * (1.0 + 0.2 * (std_dev / 128.0 - 1.0));
                    row[x as usize] = if img.get_pixel(x, y as u32)[0] as f32 > thresh {
                        255
                    } else {
                        0
                    };
                }
            });
        let old = t1.elapsed();
        let _ = old_img;
        eprintln!("640x480 block 31: sliding {old:?} -> integral {neu:?}");
    }
}
