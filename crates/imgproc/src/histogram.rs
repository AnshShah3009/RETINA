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
}
