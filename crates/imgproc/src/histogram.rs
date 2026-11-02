use image::GrayImage;
use rayon::prelude::*;

pub fn compute_histogram(image: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for pixel in image.pixels() {
        hist[pixel[0] as usize] += 1;
    }
    hist
}

pub fn compute_histogram_normalized(image: &GrayImage) -> [f32; 256] {
    let total = image.width() * image.height();
    if total == 0 {
        return [0.0f32; 256];
    }
    let hist = compute_histogram(image);
    hist.map(|h| h as f32 / total as f32)
}

pub fn compute_cdf(hist: &[u32; 256]) -> [u32; 256] {
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    cdf
}

/// `cv::equalizeHist`'s LUT for a histogram of `total` pixels, or `None` for
/// a single-valued image (which OpenCV returns unchanged).
///
/// `lut[v] = saturate_cast<uchar>(sum * scale)` where `sum` is the count of
/// pixels in `(first, v]`, `first` is the lowest occupied level and
/// `scale = 255.f / (total - hist[first])` in f32; `saturate_cast` rounds
/// half to even. Dividing first and rounding half away from zero, as both
/// copies here used to, put every `.5` entry one level above cv2.
pub(crate) fn equalize_hist_lut(hist: &[u32; 256], total: u64) -> Option<[u8; 256]> {
    let first = hist.iter().position(|&c| c > 0)?;
    if hist[first] as u64 == total {
        return None;
    }
    let scale = 255.0f32 / (total - hist[first] as u64) as f32;
    let mut lut = [0u8; 256];
    let mut sum = 0u64;
    for (v, l) in lut.iter_mut().enumerate().skip(first + 1) {
        sum += hist[v] as u64;
        *l = (sum as f32 * scale).round_ties_even().clamp(0.0, 255.0) as u8;
    }
    Some(lut)
}

/// `cv2.equalizeHist`.
#[allow(clippy::needless_range_loop)]
pub fn histogram_equalization(image: &GrayImage) -> GrayImage {
    if image.width() == 0 || image.height() == 0 {
        return image.clone();
    }
    let hist = compute_histogram(image);
    let total = image.width() as u64 * image.height() as u64;
    let Some(lut) = equalize_hist_lut(&hist, total) else {
        return image.clone();
    };

    let mut output = GrayImage::new(image.width(), image.height());
    let src_raw = image.as_raw();

    output
        .as_mut()
        .par_chunks_mut(image.width() as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let offset = y * image.width() as usize;
            for x in 0..image.width() as usize {
                let src_pixel = src_raw[offset + x];
                row[x] = lut[src_pixel as usize];
            }
        });

    output
}

// ── CLAHE ────────────────────────────────────────────────────────────────────

/// CLAHE (Contrast Limited Adaptive Histogram Equalization), as
/// `cv2.createCLAHE(clip_limit, tile_grid).apply(image)`.
///
/// Follows OpenCV's algorithm:
/// * if the size is not a multiple of the grid, tile statistics are taken on
///   an image padded right/bottom with BORDER_REFLECT_101 up to the next
///   multiple; the tile size is that padded size divided by the grid;
/// * each tile histogram is clipped at `max(int(clip_limit * tileArea / 256), 1)`
///   (no clipping when `clip_limit <= 0`), the excess is spread evenly over all
///   bins and the remainder added to every `256 / residual`-th bin;
/// * the LUT is `round(cdf * 255 / tileArea)`;
/// * each output pixel interpolates the LUTs of the four nearest tile
///   centres bilinearly.
///
/// # Arguments
/// * `image` — Input grayscale image
/// * `clip_limit` — Contrast limit (typically 2.0-4.0). Higher = more contrast.
/// * `tile_grid` — Grid size as `(cols, rows)`. Default is `(8, 8)`.
///
/// # Example
/// ```ignore
/// let enhanced = clahe(&gray_image, 2.0, (8, 8));
/// ```
pub fn clahe(image: &GrayImage, clip_limit: f32, tile_grid: (u32, u32)) -> GrayImage {
    const HIST: usize = 256;
    let (w, h) = (image.width() as usize, image.height() as usize);
    let (nx, ny) = (tile_grid.0.max(1) as usize, tile_grid.1.max(1) as usize);
    if w == 0 || h == 0 {
        return image.clone();
    }

    // OpenCV pads by `tiles - size % tiles` when the size is not divisible.
    let ext_w = if w % nx != 0 { w + nx - w % nx } else { w };
    let ext_h = if h % ny != 0 { h + ny - h % ny } else { h };
    let tw = ext_w / nx;
    let th = ext_h / ny;
    let src = image.as_raw();
    // BORDER_REFLECT_101 index for the padded region (pad < tile size <= size
    // is not guaranteed for tiny images, so fold periodically).
    let reflect101 = |i: usize, n: usize| -> usize {
        if n == 1 {
            return 0;
        }
        let period = 2 * (n - 1);
        let m = i % period;
        if m < n { m } else { period - m }
    };

    let tile_area = (tw * th) as i64;
    let clip = if clip_limit > 0.0 {
        ((clip_limit * tile_area as f32 / HIST as f32) as i64).max(1)
    } else {
        0
    };
    let lut_scale = (HIST - 1) as f32 / tile_area as f32;

    let mut luts = vec![[0u8; HIST]; nx * ny];
    luts.par_iter_mut().enumerate().for_each(|(t, lut)| {
        let (tx, ty) = (t % nx, t / nx);
        let mut hist = [0i64; HIST];
        for yy in ty * th..(ty + 1) * th {
            let row = reflect101(yy, h) * w;
            for xx in tx * tw..(tx + 1) * tw {
                hist[src[row + reflect101(xx, w)] as usize] += 1;
            }
        }
        if clip > 0 {
            let mut clipped = 0i64;
            for v in hist.iter_mut() {
                if *v > clip {
                    clipped += *v - clip;
                    *v = clip;
                }
            }
            let batch = clipped / HIST as i64;
            let mut residual = clipped - batch * HIST as i64;
            for v in hist.iter_mut() {
                *v += batch;
            }
            if residual != 0 {
                let step = ((HIST as i64) / residual).max(1) as usize;
                let mut i = 0usize;
                while i < HIST && residual > 0 {
                    hist[i] += 1;
                    i += step;
                    residual -= 1;
                }
            }
        }
        let mut sum = 0i64;
        for (l, v) in lut.iter_mut().zip(hist.iter()) {
            sum += v;
            *l = (sum as f32 * lut_scale).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });

    // Bilinear interpolation between the four surrounding tile LUTs.
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let xs: Vec<(usize, usize, f32)> = (0..w)
        .map(|x| {
            let txf = x as f32 * inv_tw - 0.5;
            let tx1 = txf.floor() as i64;
            let xa = txf - tx1 as f32;
            let t1 = tx1.max(0) as usize;
            let t2 = ((tx1 + 1) as usize).min(nx - 1);
            (t1, t2, xa)
        })
        .collect();

    let mut output = GrayImage::new(w as u32, h as u32);
    output
        .as_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let tyf = y as f32 * inv_th - 0.5;
            let ty1 = tyf.floor() as i64;
            let ya = tyf - ty1 as f32;
            let ya1 = 1.0 - ya;
            let r1 = ty1.max(0) as usize * nx;
            let r2 = ((ty1 + 1) as usize).min(ny - 1) * nx;
            for (x, out) in row.iter_mut().enumerate() {
                let v = src[y * w + x] as usize;
                let (t1, t2, xa) = xs[x];
                let xa1 = 1.0 - xa;
                let res = (luts[r1 + t1][v] as f32 * xa1 + luts[r1 + t2][v] as f32 * xa) * ya1
                    + (luts[r2 + t1][v] as f32 * xa1 + luts[r2 + t2][v] as f32 * xa) * ya;
                *out = res.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        });

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_histogram_equalization_identity() {
        // A 256x1 image with one pixel per intensity (0..=255) already has a
        // perfectly uniform histogram. After equalization the output should
        // approximately equal the input (the CDF is already linear).
        let mut img = GrayImage::new(256, 1);
        for i in 0u32..256 {
            img.put_pixel(i, 0, Luma([i as u8]));
        }

        let output = histogram_equalization(&img);

        for i in 0u32..256 {
            let src = i as u8;
            let dst = output.get_pixel(i, 0)[0];
            let diff = (src as i16 - dst as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "pixel {} expected ~{}, got {} (diff={})",
                i,
                src,
                dst,
                diff
            );
        }
    }

    #[test]
    fn test_clahe_basic() {
        // Low-contrast image: all pixels in 100-110 range
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.put_pixel(x, y, Luma([100 + ((x + y) % 10) as u8]));
            }
        }

        let result = clahe(&img, 2.0, (4, 4));
        assert_eq!(result.width(), 64);
        assert_eq!(result.height(), 64);

        // CLAHE should expand the contrast range beyond 100-110
        let min_val = result.pixels().map(|p| p[0]).min().unwrap();
        let max_val = result.pixels().map(|p| p[0]).max().unwrap();
        let range = max_val - min_val;
        assert!(
            range > 20,
            "CLAHE should expand contrast, got range {} (min={}, max={})",
            range,
            min_val,
            max_val
        );
    }

    #[test]
    fn test_clahe_preserves_dimensions() {
        let img = GrayImage::new(320, 240);
        let result = clahe(&img, 4.0, (8, 8));
        assert_eq!(result.dimensions(), (320, 240));
    }
}
