use crate::threshold::ThresholdType;
use image::GrayImage;
use rayon::prelude::*;

pub enum LocalThresholdMethod {
    Niblack,
    Sauvola,
}

pub fn local_threshold(
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

    dst
}
