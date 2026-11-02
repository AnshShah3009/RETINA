use image::{GrayImage, RgbImage};
use rayon::ThreadPool;
use rayon::prelude::*;

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

/// OpenCV's 8-bit colour-to-grey weights: `0.299`, `0.587`, `0.114` in 15-bit
/// fixed point (`cvtColor` `RGB2GRAY` / `BGR2GRAY` for `CV_8U`).
const R2Y: u32 = 9798;
const G2Y: u32 = 19235;
const B2Y: u32 = 3735;
const GRAY_SHIFT: u32 = 15;

#[inline]
fn gray_fixed(r: u8, g: u8, b: u8) -> u8 {
    // At most 255 * 32768 + 2^14 before the shift, so u32 cannot overflow and
    // the result is at most 255.
    ((r as u32 * R2Y + g as u32 * G2Y + b as u32 * B2Y + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT)
        as u8
}

/// Interleaved RGB `u8` to grey, exactly `cv2.cvtColor(.., COLOR_RGB2GRAY)`:
/// `(9798 R + 19235 G + 3735 B + 2^14) >> 15`.
///
/// OpenCV 4.13 uses these 15-bit weights; on all 2^24 colours this matches it
/// with no mismatch. The f32 `0.299 R + 0.587 G + 0.114 B` with half-away
/// rounding that `convert_rgb_to_gray` used before (via
/// `cv_hal::cpu::simd::rgb_to_gray_simd`) was one grey level off on 21,247 of
/// them (0.127%). See `tests/rgb_to_gray_opencv_parity.rs`.
///
/// # Panics
///
/// If `rgb.len() != 3 * gray.len()`.
pub fn rgb_to_gray_u8(rgb: &[u8], gray: &mut [u8]) {
    assert_eq!(rgb.len(), gray.len() * 3, "rgb_to_gray_u8: length mismatch");
    for (g, px) in gray.iter_mut().zip(rgb.chunks_exact(3)) {
        *g = gray_fixed(px[0], px[1], px[2]);
    }
}

/// Interleaved BGR `u8` to grey, exactly `cv2.cvtColor(.., COLOR_BGR2GRAY)`
/// (the weights of [`rgb_to_gray_u8`] with the channel order swapped).
///
/// # Panics
///
/// If `bgr.len() != 3 * gray.len()`.
pub fn bgr_to_gray_u8(bgr: &[u8], gray: &mut [u8]) {
    assert_eq!(bgr.len(), gray.len() * 3, "bgr_to_gray_u8: length mismatch");
    for (g, px) in gray.iter_mut().zip(bgr.chunks_exact(3)) {
        *g = gray_fixed(px[2], px[1], px[0]);
    }
}

/// RGB to grey with OpenCV's 8-bit weights; see [`rgb_to_gray_u8`].
pub fn convert_rgb_to_gray(rgb: &RgbImage) -> GrayImage {
    convert_rgb_to_gray_in_pool(rgb, None)
}

pub fn convert_rgb_to_gray_in_pool(rgb: &RgbImage, pool: Option<&ThreadPool>) -> GrayImage {
    let run = || {
        let (w, h) = rgb.dimensions();
        let count = (w * h) as usize;
        let mut gray_data = vec![0u8; count];
        let rgb_data = rgb.as_raw();

        gray_data
            .par_chunks_mut(4096)
            .zip(rgb_data.par_chunks(4096 * 3))
            .for_each(|(g_chunk, rgb_chunk)| {
                rgb_to_gray_u8(rgb_chunk, g_chunk);
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
            *p = (*p as f32 * factor).round_ties_even().clamp(0.0, 255.0) as u8;
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
            *p = ((*p as f32 - mean) * factor + mean)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8;
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

        let Some(lut) = crate::histogram::equalize_hist_lut(&hist, raw.len() as u64) else {
            // Single-valued image: cv2.equalizeHist leaves it unchanged.
            return src.clone();
        };

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
