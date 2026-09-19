use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use rayon::ThreadPool;

pub fn convert_gray_to_rgb(gray: &GrayImage) -> RgbImage {
    convert_gray_to_rgb_in_pool(gray, None)
}

pub fn convert_gray_to_rgb_in_pool(gray: &GrayImage, pool: Option<&ThreadPool>) -> RgbImage {
    let run = || {
        let (w, h) = gray.dimensions();
        let count = (w * h) as usize;
        let mut rgb_data = vec![0u8; count * 3];
        let gray_data = gray.as_raw();

        rgb_data
            .par_chunks_mut(3)
            .zip(gray_data.par_iter())
            .for_each(|(rgb_pixel, &g)| {
                rgb_pixel[0] = g;
                rgb_pixel[1] = g;
                rgb_pixel[2] = g;
            });

        RgbImage::from_raw(w, h, rgb_data).unwrap()
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

use crate::simd::rgb_to_gray_simd;

pub fn convert_rgb_to_gray(rgb: &RgbImage) -> GrayImage {
    convert_rgb_to_gray_in_pool(rgb, None)
}

pub fn convert_rgb_to_gray_in_pool(rgb: &RgbImage, pool: Option<&ThreadPool>) -> GrayImage {
    let run = || {
        let (w, h) = rgb.dimensions();
        let count = (w * h) as usize;
        let mut gray_data = vec![0u8; count];
        let rgb_data = rgb.as_raw();

        // Process in parallel chunks, but use SIMD within each chunk
        gray_data
            .par_chunks_mut(4096) // Larger chunk size for SIMD efficiency
            .zip(rgb_data.par_chunks(4096 * 3))
            .for_each(|(g_chunk, rgb_chunk)| {
                rgb_to_gray_simd(rgb_chunk, g_chunk);
            });

        GrayImage::from_raw(w, h, gray_data).unwrap()
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn to_gray(img: &RgbImage) -> GrayImage {
    convert_rgb_to_gray(img)
}

pub fn to_rgb(img: &GrayImage) -> RgbImage {
    convert_gray_to_rgb(img)
}

pub fn adjust_brightness(src: &GrayImage, factor: f32) -> GrayImage {
    adjust_brightness_in_pool(src, factor, None)
}

pub fn adjust_brightness_in_pool(
    src: &GrayImage,
    factor: f32,
    pool: Option<&ThreadPool>,
) -> GrayImage {
    let run = || {
        let mut output = src.clone();
        output.as_mut().par_iter_mut().for_each(|p| {
            *p = (*p as f32 * factor).clamp(0.0, 255.0) as u8;
        });
        output
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn adjust_contrast(src: &GrayImage, factor: f32) -> GrayImage {
    adjust_contrast_in_pool(src, factor, None)
}

pub fn adjust_contrast_in_pool(
    src: &GrayImage,
    factor: f32,
    pool: Option<&ThreadPool>,
) -> GrayImage {
    let run = || {
        let mut output = src.clone();
        let mean = compute_mean_intensity(src);
        output.as_mut().par_iter_mut().for_each(|p| {
            *p = ((*p as f32 - mean) * factor + mean).clamp(0.0, 255.0) as u8;
        });
        output
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn compute_mean_intensity(src: &GrayImage) -> f32 {
    let raw = src.as_raw();
    if raw.is_empty() {
        return 0.0;
    }
    let sum: u64 = raw.par_iter().map(|&p| p as u64).sum();
    (sum as f32) / (raw.len() as f32)
}

pub fn invert(src: &GrayImage) -> GrayImage {
    invert_in_pool(src, None)
}

pub fn invert_in_pool(src: &GrayImage, pool: Option<&ThreadPool>) -> GrayImage {
    let run = || {
        let mut output = src.clone();
        output.as_mut().par_iter_mut().for_each(|p| {
            *p = 255 - *p;
        });
        output
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn equalize_histogram(src: &GrayImage) -> GrayImage {
    equalize_histogram_in_pool(src, None)
}

pub fn equalize_histogram_in_pool(src: &GrayImage, pool: Option<&ThreadPool>) -> GrayImage {
    let run = || {
        let raw = src.as_raw();
        if raw.is_empty() {
            return src.clone();
        }

        // Compute histogram in parallel
        let hist = raw
            .par_chunks(4096)
            .fold(
                || [0u32; 256],
                |mut local_hist, chunk| {
                    for &val in chunk {
                        local_hist[val as usize] += 1;
                    }
                    local_hist
                },
            )
            .reduce(
                || [0u32; 256],
                |mut h1, h2| {
                    for i in 0..256 {
                        h1[i] += h2[i];
                    }
                    h1
                },
            );

        // Compute CDF (cumulative distribution function) - Serial is fast enough for 256 elements
        let mut cdf = [0u32; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
        let total = raw.len() as u32;

        // Build LUT
        let mut lut = [0u8; 256];
        if total > cdf_min {
            let denom = (total - cdf_min) as f32;
            for i in 0..256 {
                lut[i] = ((cdf[i].saturating_sub(cdf_min)) as f32 / denom * 255.0).round() as u8;
            }
        } else {
            // Edge case: all pixels are same or image is empty
            return src.clone();
        }

        // Apply LUT in parallel
        let mut output = src.clone();
        output.as_mut().par_iter_mut().for_each(|p| {
            *p = lut[*p as usize];
        });
        output
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}
