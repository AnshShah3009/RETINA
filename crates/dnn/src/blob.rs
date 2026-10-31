use image::GrayImage;

pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut blob = Vec::with_capacity(width * height);

    for pixel in image.as_raw() {
        blob.push(*pixel as f32 / 255.0);
    }

    blob
}

/// Convert a `[0, 1]` blob back into a `width x height` luma image.
///
/// The blob is consumed in row-major order, exactly `width * height` values.
///
/// # Panics
///
/// Panics when `blob.len() != width * height`. Both mismatch directions used to
/// be silently absorbed:
///
/// * **too few values** — `image::GrayImage::from_raw` rejects a short buffer,
///   and the fallback was `GrayImage::new(width, height)`, a well-formed
///   **all-black** image. Measured: `blob_to_image(&[0.25, 0.5, 0.75], 2, 2)`
///   returned pixels `[0, 0, 0, 0]` — a plausible observation, so nothing
///   downstream could tell that every value it supplied had been dropped.
/// * **too many values** — `from_raw` accepts a buffer that is *at least*
///   `width * height` long, so the extra values were silently discarded
///   (`blob_to_image(&[1.0; 5], 2, 2)` returned `[255, 255, 255, 255]`).
///
/// There is no honest value to substitute for a missing sample, so the size
/// mismatch is reported instead.
pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage {
    let expected = width as usize * height as usize;
    assert_eq!(
        blob.len(),
        expected,
        "blob_to_image: blob has {} values but {width}x{height} needs {expected}",
        blob.len()
    );

    // Pixel-for-pixel copy in row-major order (same order as the blob), so
    // there is no fallible buffer rebuild that could silently fall back.
    let mut img = GrayImage::new(width, height);
    for (px, val) in img.pixels_mut().zip(blob.iter()) {
        *px = image::Luma([(val * 255.0).clamp(0.0, 255.0) as u8]);
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_to_blob_basic() {
        let img = GrayImage::new(2, 2);
        let blob = image_to_blob(&img);

        assert_eq!(blob.len(), 4);
        // All pixels should be 0.0 (black image)
        for &val in &blob {
            assert!((val - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_image_to_blob_normalization() {
        let mut img = GrayImage::new(1, 1);
        img.put_pixel(0, 0, image::Luma([255]));
        let blob = image_to_blob(&img);

        assert_eq!(blob.len(), 1);
        assert!((blob[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_image_to_blob_mid_value() {
        let mut img = GrayImage::new(1, 1);
        img.put_pixel(0, 0, image::Luma([128]));
        let blob = image_to_blob(&img);

        assert_eq!(blob.len(), 1);
        assert!((blob[0] - (128.0 / 255.0)).abs() < 0.01);
    }

    #[test]
    fn test_blob_to_image_basic() {
        let blob = vec![0.0, 0.5, 1.0, 0.0];
        let img = blob_to_image(&blob, 2, 2);

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.len(), 4);
    }

    #[test]
    fn test_blob_to_image_normalization() {
        let blob = vec![1.0];
        let img = blob_to_image(&blob, 1, 1);

        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
        assert_eq!(img.get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn test_blob_to_image_clamping() {
        let blob = vec![1.5, -0.5];
        let img = blob_to_image(&blob, 2, 1);

        // Values > 1.0 should clamp to 255
        assert_eq!(img.get_pixel(0, 0)[0], 255);
        // Values < 0.0 should clamp to 0
        assert_eq!(img.get_pixel(1, 0)[0], 0);
    }

    #[test]
    fn test_round_trip() {
        let mut original = GrayImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                original.put_pixel(x, y, image::Luma([64 + x as u8 * 50]));
            }
        }

        let blob = image_to_blob(&original);
        let recovered = blob_to_image(&blob, 3, 3);

        // After normalization and denormalization, values should be close
        for y in 0..3 {
            for x in 0..3 {
                let orig_val = original.get_pixel(x, y)[0] as f32;
                let recov_val = recovered.get_pixel(x, y)[0] as f32;
                let error = (orig_val - recov_val).abs();
                assert!(
                    error <= 1.0,
                    "Round trip error too large: {} vs {}",
                    orig_val,
                    recov_val
                );
            }
        }
    }
}
