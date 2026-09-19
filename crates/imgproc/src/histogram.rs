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

#[allow(clippy::needless_range_loop)]
pub fn histogram_equalization(image: &GrayImage) -> GrayImage {
    let hist = compute_histogram(image);
    let cdf = compute_cdf(&hist);

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = image.width() * image.height();

    let mut lut = [0u8; 256];
    if total > cdf_min {
        for i in 0..256 {
            let val = ((cdf[i].saturating_sub(cdf_min)) as f32 / (total - cdf_min) as f32 * 255.0)
                .round() as u8;
            lut[i] = val;
        }
    } else {
        // If total == cdf_min (e.g. constant image), identity mapping
        for i in 0..256 {
            lut[i] = i as u8;
        }
    }

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

/// CLAHE (Contrast Limited Adaptive Histogram Equalization).
///
/// Divides the image into `tile_grid` tiles, computes a contrast-limited
/// histogram equalization per tile, and bilinearly interpolates between
/// neighboring tiles for smooth transitions.
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
    let (w, h) = (image.width(), image.height());
    let (nx, ny) = (tile_grid.0.max(1), tile_grid.1.max(1));
    let tw = w / nx; // tile width
    let th = h / ny; // tile height

    if tw == 0 || th == 0 {
        return image.clone();
    }

    let tile_pixels = (tw * th) as f32;
    // Compute the actual clip count from the normalized clip_limit
    let clip_count = (clip_limit * tile_pixels / 256.0).max(1.0) as u32;

    // Step 1: Compute clipped+redistributed LUT for each tile
    let mut luts = vec![[0u8; 256]; (nx * ny) as usize];

    for ty in 0..ny {
        for tx in 0..nx {
            let x0 = tx * tw;
            let y0 = ty * th;
            let x1 = if tx == nx - 1 { w } else { x0 + tw };
            let y1 = if ty == ny - 1 { h } else { y0 + th };
            let n_pixels = (x1 - x0) * (y1 - y0);

            // Compute histogram for this tile
            let mut hist = [0u32; 256];
            for py in y0..y1 {
                for px in x0..x1 {
                    hist[image.get_pixel(px, py)[0] as usize] += 1;
                }
            }

            // Clip histogram and redistribute excess
            let mut excess = 0u32;
            for h in hist.iter_mut() {
                if *h > clip_count {
                    excess += *h - clip_count;
                    *h = clip_count;
                }
            }
            let per_bin = excess / 256;
            let remainder = (excess % 256) as usize;
            for (i, h) in hist.iter_mut().enumerate() {
                *h += per_bin;
                if i < remainder {
                    *h += 1;
                }
            }

            // Build CDF → LUT
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }
            let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
            let denom = n_pixels.saturating_sub(cdf_min);

            let lut = &mut luts[(ty * nx + tx) as usize];
            if denom > 0 {
                for i in 0..256 {
                    lut[i] = ((cdf[i].saturating_sub(cdf_min) as f32 / denom as f32) * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            } else {
                for i in 0..256 {
                    lut[i] = i as u8;
                }
            }
        }
    }

    // Step 2: Bilinear interpolation between neighboring tile LUTs
    let mut output = GrayImage::new(w, h);

    for py in 0..h {
        for px in 0..w {
            let src = image.get_pixel(px, py)[0] as usize;

            // Find which tile centers surround this pixel
            // Tile center = (tx * tw + tw/2, ty * th + th/2)
            let fx = (px as f32 - tw as f32 / 2.0) / tw as f32;
            let fy = (py as f32 - th as f32 / 2.0) / th as f32;

            let tx0 = (fx.floor() as i32).clamp(0, nx as i32 - 1) as u32;
            let ty0 = (fy.floor() as i32).clamp(0, ny as i32 - 1) as u32;
            let tx1 = (tx0 + 1).min(nx - 1);
            let ty1 = (ty0 + 1).min(ny - 1);

            let ax = (fx - tx0 as f32).clamp(0.0, 1.0);
            let ay = (fy - ty0 as f32).clamp(0.0, 1.0);

            let v00 = luts[(ty0 * nx + tx0) as usize][src] as f32;
            let v10 = luts[(ty0 * nx + tx1) as usize][src] as f32;
            let v01 = luts[(ty1 * nx + tx0) as usize][src] as f32;
            let v11 = luts[(ty1 * nx + tx1) as usize][src] as f32;

            let val = v00 * (1.0 - ax) * (1.0 - ay)
                + v10 * ax * (1.0 - ay)
                + v01 * (1.0 - ax) * ay
                + v11 * ax * ay;

            output.put_pixel(px, py, image::Luma([val.round().clamp(0.0, 255.0) as u8]));
        }
    }

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
