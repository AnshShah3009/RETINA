use image::GrayImage;
use wide::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMethod {
    MeanC,
    GaussianC,
}

pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
    threshold_cpu(src, thresh, max_value, typ)
}

pub fn threshold_cpu(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
    use rayon::prelude::*;
    let mut dst = GrayImage::new(src.width(), src.height());
    if src.width() == 0 || src.height() == 0 {
        return dst;
    }
    let width = src.width() as usize;
    let src_raw = src.as_raw();
    let dst_raw = dst.as_mut();

    dst_raw
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_dst)| {
            let row_src = &src_raw[y * width..(y + 1) * width];

            // Every threshold type is an unsigned compare-and-select (or a
            // min) on the bytes themselves, so the vector loop stays in u8:
            // 32 pixels per step, no widening to f32 and back.
            let thresh_v = u8x32::splat(thresh);
            let max_v = u8x32::splat(max_value);
            let zero_v = u8x32::ZERO;

            let mut src_chunks = row_src.chunks_exact(32);
            let mut dst_chunks = row_dst.chunks_exact_mut(32);
            for (s, d) in (&mut src_chunks).zip(&mut dst_chunks) {
                let s_v = u8x32::new(s.try_into().expect("chunk is 32 bytes"));
                let res = match typ {
                    ThresholdType::Binary => s_v.simd_gt(thresh_v).select(max_v, zero_v),
                    ThresholdType::BinaryInv => s_v.simd_gt(thresh_v).select(zero_v, max_v),
                    ThresholdType::Trunc => s_v.min(thresh_v),
                    ThresholdType::ToZero => s_v.simd_gt(thresh_v).select(s_v, zero_v),
                    ThresholdType::ToZeroInv => s_v.simd_gt(thresh_v).select(zero_v, s_v),
                };
                d.copy_from_slice(&res.to_array());
            }
            for (s, d) in src_chunks
                .remainder()
                .iter()
                .zip(dst_chunks.into_remainder())
            {
                *d = apply_threshold(*s, thresh, max_value, typ);
            }
        });

    dst
}

pub fn threshold_otsu(src: &GrayImage, max_value: u8, typ: ThresholdType) -> (u8, GrayImage) {
    let hist = histogram(src);
    let total = (src.width() * src.height()) as f64;

    let mut sum_all = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_all += (i as f64) * (count as f64);
    }

    let mut weight_background = 0.0f64;
    let mut sum_background = 0.0f64;
    let mut best_between = -1.0f64;
    let mut best_threshold = 0u8;

    // Mode of the histogram: fallback threshold for uniform images (a
    // previous revision returned 0, inverting the output vs. OpenCV).
    let mode_level = hist
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i as u8)
        .unwrap_or(0);

    for t in 0u16..=255 {
        let idx = t as usize;
        weight_background += hist[idx] as f64;
        if weight_background <= f64::EPSILON {
            continue;
        }

        let weight_foreground = total - weight_background;
        if weight_foreground <= f64::EPSILON {
            break; // all mass at or below t — no valid split remains
        }

        sum_background += (t as f64) * (hist[idx] as f64);
        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum_all - sum_background) / weight_foreground;
        let diff = mean_background - mean_foreground;
        let between = weight_background * weight_foreground * diff * diff;

        if between > best_between {
            best_between = between;
            best_threshold = t as u8;
        }
    }

    // Uniform image: between-variance never became positive; fall back to
    // the histogram mode instead of 0.
    if best_between <= 0.0 {
        best_threshold = mode_level;
    }

    let dst = threshold_cpu(src, best_threshold, max_value, typ);
    (best_threshold, dst)
}

/// Adaptive threshold, `cv2.adaptiveThreshold(src, max_value, method, typ,
/// block_size, c)`.
///
/// The local mean is a `block_size` x `block_size` box mean (`MeanC`) or
/// Gaussian-weighted mean (`GaussianC`, `getGaussianKernel(block_size, 0)`,
/// computed in f32) with BORDER_REPLICATE, rounded to u8. The comparison is
/// integer, as in OpenCV: `Binary` sets `max_value` where
/// `src - mean > -ceil(c)`, `BinaryInv` where `src - mean <= -floor(c)`.
///
/// # Panics
/// If `block_size` is not odd and >= 3, or `typ` is not `Binary`/`BinaryInv`.
pub fn adaptive_threshold(
    src: &GrayImage,
    max_value: u8,
    method: AdaptiveMethod,
    typ: ThresholdType,
    block_size: u32,
    c: f32,
) -> GrayImage {
    use rayon::prelude::*;
    assert!(block_size >= 3, "block_size must be >= 3");
    assert!(block_size % 2 == 1, "block_size must be odd");
    assert!(
        matches!(typ, ThresholdType::Binary | ThresholdType::BinaryInv),
        "adaptive threshold supports Binary or BinaryInv types"
    );

    let (w, h) = (src.width() as usize, src.height() as usize);
    let mut dst = GrayImage::new(src.width(), src.height());
    if w == 0 || h == 0 {
        return dst;
    }
    let mean = match method {
        AdaptiveMethod::MeanC => local_mean_replicate(src, block_size as usize),
        AdaptiveMethod::GaussianC => local_gaussian_replicate(src, block_size as usize),
    };

    // OpenCV: idelta = BINARY ? cvCeil(delta) : cvFloor(delta), then a table
    // on src - mean.
    let binary = typ == ThresholdType::Binary;
    let idelta = if binary {
        (c as f64).ceil() as i32
    } else {
        (c as f64).floor() as i32
    };
    let src_raw = src.as_raw();
    dst.as_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let d = src_raw[y * w + x] as i32 - mean[y * w + x] as i32;
                let on = if binary { d > -idelta } else { d <= -idelta };
                *out = if on { max_value } else { 0 };
            }
        });
    dst
}

fn apply_threshold(value: u8, thresh: u8, max_value: u8, typ: ThresholdType) -> u8 {
    match typ {
        ThresholdType::Binary => {
            if value > thresh {
                max_value
            } else {
                0
            }
        }
        ThresholdType::BinaryInv => {
            if value > thresh {
                0
            } else {
                max_value
            }
        }
        ThresholdType::Trunc => value.min(thresh),
        ThresholdType::ToZero => {
            if value > thresh {
                value
            } else {
                0
            }
        }
        ThresholdType::ToZeroInv => {
            if value > thresh {
                0
            } else {
                value
            }
        }
    }
}

fn histogram(src: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &px in src.as_raw() {
        hist[px as usize] += 1;
    }
    hist
}

/// Box mean over a `block` x `block` window with BORDER_REPLICATE, rounded
/// to nearest (`cv::boxFilter(.., normalize = true)` on u8).
fn local_mean_replicate(src: &GrayImage, block: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let (w, h) = (src.width() as usize, src.height() as usize);
    let r = block / 2;
    let data = src.as_raw();
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;

    // Horizontal window sums. u64: 255 * block exceeds u32 once
    // block >= 16_843_011 (the vertical pass below is already u64).
    let mut rows = vec![0u64; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let src_row = &data[y * w..(y + 1) * w];
        let mut s: u64 = (0..block)
            .map(|i| src_row[clamp(i as isize - r as isize, w)] as u64)
            .sum();
        for (x, o) in out.iter_mut().enumerate() {
            *o = s;
            let add = src_row[clamp((x + r + 1) as isize, w)] as u64;
            let sub = src_row[clamp(x as isize - r as isize, w)] as u64;
            s = s + add - sub;
        }
    });

    let n = (block * block) as u64;
    // Vertical window is the same 1D running sum as the rows above, one
    // column at a time. Summing `block` row-sums per pixel was the remaining
    // O(block) term. Column-major so each column is written by one thread.
    let mut cols = vec![0u8; w * h];
    cols.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        let mut s: u64 = (0..block)
            .map(|i| rows[clamp(i as isize - r as isize, h) * w + x] as u64)
            .sum();
        for (y, v) in col.iter_mut().enumerate() {
            // round(s / n); n = block^2 is odd, so there are no ties.
            *v = ((2 * s + n) / (2 * n)).min(255) as u8;
            let add = rows[clamp(y as isize + r as isize + 1, h) * w + x] as u64;
            let sub = rows[clamp(y as isize - r as isize, h) * w + x] as u64;
            s = s + add - sub;
        }
    });
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = cols[x * h + y];
        }
    });
    out
}

/// `cv::getGaussianKernel(n, 0, CV_32F)`: the fixed tables for n <= 9, else
/// sigma = 0.3 * ((n - 1) / 2 - 1) + 0.8.
fn opencv_gaussian_kernel(n: usize) -> Vec<f32> {
    match n {
        1 => return vec![1.0],
        3 => return vec![0.25, 0.5, 0.25],
        5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
        7 => {
            return vec![
                0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
            ];
        }
        9 => {
            return [4.0, 13.0, 30.0, 51.0, 60.0, 51.0, 30.0, 13.0, 4.0]
                .iter()
                .map(|v| v / 256.0)
                .collect();
        }
        _ => {}
    }
    let sigma = ((n as f64 - 1.0) * 0.5 - 1.0) * 0.3 + 0.8;
    let scale2x = -0.5 / (sigma * sigma);
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - (n as f64 - 1.0) * 0.5;
            (scale2x * x * x).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter().map(|v| (v / sum) as f32).collect()
}

/// Gaussian-weighted local mean as OpenCV's ADAPTIVE_THRESH_GAUSSIAN_C
/// computes it: the u8 image converted to f32, `GaussianBlur(block, block)`
/// with BORDER_REPLICATE (row pass, then column pass, f32 accumulation), and
/// the result rounded half to even back to u8.
fn local_gaussian_replicate(src: &GrayImage, block: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let (w, h) = (src.width() as usize, src.height() as usize);
    let k = opencv_gaussian_kernel(block);
    let r = block / 2;
    let data = src.as_raw();
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;

    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &data[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            let mut s = 0f32;
            for (i, &kv) in k.iter().enumerate() {
                s += kv * row[clamp(x as isize + i as isize - r as isize, w)] as f32;
            }
            *o = s;
        }
    });
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        for (x, v) in o.iter_mut().enumerate() {
            let mut s = 0f32;
            for (i, &kv) in k.iter().enumerate() {
                s += kv * tmp[clamp(y as isize + i as isize - r as isize, h) * w + x];
            }
            *v = s.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn binary_threshold_basic() {
        let mut img = GrayImage::new(4, 1);
        img.put_pixel(0, 0, Luma([10]));
        img.put_pixel(1, 0, Luma([50]));
        img.put_pixel(2, 0, Luma([100]));
        img.put_pixel(3, 0, Luma([200]));

        let out = threshold(&img, 100, 255, ThresholdType::Binary);
        assert_eq!(out.as_raw(), &[0, 0, 0, 255]);
    }

    #[test]
    fn binary_inv_threshold() {
        let mut img = GrayImage::new(4, 1);
        for (i, v) in [50u8, 100, 150, 200].iter().enumerate() {
            img.put_pixel(i as u32, 0, Luma([*v]));
        }
        let out = threshold(&img, 100, 255, ThresholdType::BinaryInv);
        // > 100 → 0, <= 100 → 255
        assert_eq!(out.as_raw(), &[255, 255, 0, 0]);
    }

    #[test]
    fn trunc_threshold() {
        let mut img = GrayImage::new(3, 1);
        img.put_pixel(0, 0, Luma([50]));
        img.put_pixel(1, 0, Luma([100]));
        img.put_pixel(2, 0, Luma([200]));
        let out = threshold(&img, 100, 255, ThresholdType::Trunc);
        assert_eq!(out.as_raw(), &[50, 100, 100]);
    }

    #[test]
    fn to_zero_threshold() {
        let mut img = GrayImage::new(4, 1);
        for (i, v) in [50u8, 100, 150, 200].iter().enumerate() {
            img.put_pixel(i as u32, 0, Luma([*v]));
        }
        let out = threshold(&img, 100, 255, ThresholdType::ToZero);
        assert_eq!(out.as_raw(), &[0, 0, 150, 200]);
    }

    #[test]
    fn to_zero_inv_threshold() {
        let mut img = GrayImage::new(4, 1);
        for (i, v) in [50u8, 100, 150, 200].iter().enumerate() {
            img.put_pixel(i as u32, 0, Luma([*v]));
        }
        let out = threshold(&img, 100, 255, ThresholdType::ToZeroInv);
        assert_eq!(out.as_raw(), &[50, 100, 0, 0]);
    }

    #[test]
    fn threshold_custom_max_value() {
        let mut img = GrayImage::new(2, 1);
        img.put_pixel(0, 0, Luma([50]));
        img.put_pixel(1, 0, Luma([200]));
        let out = threshold(&img, 100, 128, ThresholdType::Binary);
        assert_eq!(out.as_raw(), &[0, 128]);
    }

    /// The 32-lane loop and the scalar tail must agree with `apply_threshold`
    /// for every byte value and every threshold, including the 127/128
    /// boundary where an unsigned compare done as a signed one would flip.
    #[test]
    fn vector_and_tail_match_scalar_for_every_value() {
        // 32 * 8 + 3: eight vector chunks and a scalar tail per row.
        let (w, h) = (259u32, 256u32);
        let img = GrayImage::from_fn(w, h, |x, y| Luma([(x + y) as u8]));
        for thresh in 0..=255u8 {
            for typ in [
                ThresholdType::Binary,
                ThresholdType::BinaryInv,
                ThresholdType::Trunc,
                ThresholdType::ToZero,
                ThresholdType::ToZeroInv,
            ] {
                let out = threshold(&img, thresh, 200, typ);
                for (i, (&s, &d)) in img.as_raw().iter().zip(out.as_raw()).enumerate() {
                    assert_eq!(
                        d,
                        apply_threshold(s, thresh, 200, typ),
                        "{typ:?} thresh {thresh} value {s} at x {}",
                        i % w as usize
                    );
                }
            }
        }
    }

    #[test]
    fn threshold_1x1_image() {
        let mut img = GrayImage::new(1, 1);
        img.put_pixel(0, 0, Luma([150]));
        let out = threshold(&img, 100, 255, ThresholdType::Binary);
        assert_eq!(out.as_raw(), &[255]);
    }

    #[test]
    fn otsu_bimodal_picks_correct_threshold() {
        let mut img = GrayImage::new(20, 1);
        for x in 0..10 {
            img.put_pixel(x, 0, Luma([50]));
        }
        for x in 10..20 {
            img.put_pixel(x, 0, Luma([200]));
        }
        let (thresh, result) = threshold_otsu(&img, 255, ThresholdType::Binary);
        // Verify the result properly segments the bimodal histogram
        assert!(
            (50..=200).contains(&thresh),
            "threshold should be in reasonable range: {}",
            thresh
        );
        // Check segmentation correctness - lower group should map to 0, upper to 255
        let mut lower_count_zero = 0;
        let mut upper_count_255 = 0;
        for x in 0..10 {
            if result.get_pixel(x, 0)[0] == 0 {
                lower_count_zero += 1;
            }
        }
        for x in 10..20 {
            if result.get_pixel(x, 0)[0] == 255 {
                upper_count_255 += 1;
            }
        }
        assert!(lower_count_zero >= 5, "most lower pixels should be 0");
        assert!(upper_count_255 >= 5, "most upper pixels should be 255");
    }

    #[test]
    fn otsu_uniform_image_no_panic() {
        let img = GrayImage::from_pixel(10, 10, Luma([128u8]));
        let (_, result) = threshold_otsu(&img, 255, ThresholdType::Binary);
        assert_eq!(result.width(), 10);
    }

    #[test]
    fn adaptive_mean_large_block_does_not_wrap_horizontal_sum() {
        // 255 * 16_843_011 = 2^32 + 509. A u32 horizontal sum wraps, the
        // replicated 1×1 window mean collapses to 0, and Binary c=0 then
        // emits max_value. The true mean is 255, so 255 > 255 is false.
        let img = GrayImage::from_pixel(1, 1, Luma([255]));
        let out = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::MeanC,
            ThresholdType::Binary,
            16_843_011,
            0.0,
        );
        assert_eq!(out.as_raw(), &[0]);
    }

    #[test]
    fn adaptive_mean_correct_dimensions() {
        let img = GrayImage::new(20, 20);
        let result = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::MeanC,
            ThresholdType::Binary,
            5,
            0.0,
        );
        assert_eq!((result.width(), result.height()), (20, 20));
    }

    #[test]
    fn adaptive_gaussian_correct_dimensions() {
        let img = GrayImage::new(20, 20);
        let result = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::GaussianC,
            ThresholdType::Binary,
            5,
            0.0,
        );
        assert_eq!((result.width(), result.height()), (20, 20));
    }

    #[test]
    fn adaptive_gaussian_uses_block_size_window() {
        // The Gaussian window must be exactly `block_size` wide, as in
        // OpenCV. Probe pixel (index 2, value 105) sits two columns away from
        // a bright spike (index 4, value 255):
        //   - 3-wide window [1..4]: local mean 102.6  -> 105 > mean -> 255
        //   - the old sigma-derived 5-wide window [0..5]: mean 105.9 -> 0
        let mut img = GrayImage::new(10, 5);
        for y in 0..5 {
            for x in 0..10 {
                img.put_pixel(x, y, Luma([100]));
            }
            img.put_pixel(2, y, Luma([105]));
            img.put_pixel(4, y, Luma([255]));
        }

        let result = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::GaussianC,
            ThresholdType::Binary,
            3,
            0.0,
        );

        assert_eq!(
            result.get_pixel(2, 2)[0],
            255,
            "local mean must use a 3-wide (block_size) Gaussian window"
        );
    }

    #[test]
    fn test_adaptive_mean_segments_gradient() {
        // 20x20 image: left half = 50, right half = 200.
        // With MeanC, block_size=11, c=0:
        //  - bright half (200) should be above the local mean => 255
        //  - dark half (50) should be below the local mean => 0
        //
        // The block_size=11 means a radius of 5 pixels. Pixels within 5
        // columns of the edge (x=10) have mixed neighborhoods, so we only
        // check pixels well inside each half.
        let width = 20u32;
        let height = 20u32;
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = if x < width / 2 { 50u8 } else { 200u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }

        let result = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::MeanC,
            ThresholdType::Binary,
            11,
            0.0,
        );

        // Dark interior: columns 0..4. The 11x11 window centered at x=3
        // spans x=-2..8, all within the dark half (value 50). Local mean=50,
        // pixel=50, and 50 > 50 is false => output 0.
        // Bright interior: columns 16..20. Similarly, fully within bright half.
        // Local mean=200, pixel=200, and 200 > 200 is false => output 0 (!).
        //
        // The Binary threshold outputs max_value when src > local_mean - c.
        // With c=0: src > local_mean. For uniform regions the pixel equals
        // the mean so the strict > comparison yields 0 everywhere in a
        // uniform region. The test should therefore verify the TRANSITION:
        // near the edge, the bright side has mean < 200, so 200 > mean => 255,
        // while the dark side has mean > 50, so 50 > mean is false => 0.
        //
        // Check pixels near (but not on) the transition zone.
        // Columns 6..9 on the dark side have a window that includes some 200s,
        // raising the local mean above 50, so 50 > mean is still false => 0.
        // Columns 11..14 on the bright side have a window that includes some 50s,
        // lowering the local mean below 200, so 200 > mean => 255.
        for y in 0..height {
            for x in 6..10 {
                let v = result.get_pixel(x, y)[0];
                assert_eq!(v, 0, "dark-side pixel ({},{}) should be 0, got {}", x, y, v);
            }
            for x in 11..15 {
                let v = result.get_pixel(x, y)[0];
                assert_eq!(
                    v, 255,
                    "bright-side pixel ({},{}) should be 255, got {}",
                    x, y, v
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn time_mean_window() {
        use rayon::prelude::*;
        use std::time::Instant;
        let (w, h, block) = (640usize, 480usize, 75usize);
        let mut s = 1u64;
        let mut img = GrayImage::new(w as u32, h as u32);
        for p in img.pixels_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *p = Luma([(s >> 33) as u8]);
        }
        let t0 = Instant::now();
        let _ = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::MeanC,
            ThresholdType::Binary,
            block as u32,
            2.0,
        );
        let neu = t0.elapsed();
        let r = block / 2;
        let data = img.as_raw();
        let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
        let mut rows = vec![0u32; w * h];
        rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
            let src_row = &data[y * w..(y + 1) * w];
            let mut acc: u32 = (0..block)
                .map(|i| src_row[clamp(i as isize - r as isize, w)] as u32)
                .sum();
            for (x, o) in out.iter_mut().enumerate() {
                *o = acc;
                acc = acc + src_row[clamp((x + r + 1) as isize, w)] as u32
                    - src_row[clamp(x as isize - r as isize, w)] as u32;
            }
        });
        let t1 = Instant::now();
        let mut out = vec![0u8; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
            for (x, v) in o.iter_mut().enumerate() {
                let acc: u64 = (0..block)
                    .map(|i| rows[clamp(y as isize + i as isize - r as isize, h) * w + x] as u64)
                    .sum();
                *v = ((2 * acc + (block * block) as u64) / (2 * (block * block) as u64)) as u8;
            }
        });
        let old = t1.elapsed();
        let _ = out;
        eprintln!("640x480 block {block}: vertical scan {old:?}, full mean {neu:?}");
    }
}
