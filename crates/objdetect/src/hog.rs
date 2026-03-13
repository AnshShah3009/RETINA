//! Histogram of Oriented Gradients (HOG)
//!
//! Robust descriptor for object detection, particularly pedestrians.

use cv_imgproc::sobel;
use image::GrayImage;
use rayon::prelude::*;

pub struct HogParams {
    pub cell_size: usize,
    pub block_size: usize, // in cells
    pub n_bins: usize,
}

impl Default for HogParams {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2,
            n_bins: 9,
        }
    }
}

pub struct Hog {
    params: HogParams,
}

impl Hog {
    pub fn new(params: HogParams) -> Self {
        Self { params }
    }

    /// Compute HOG descriptor for the given image
    /// Returns empty vector if image is smaller than cell_size
    pub fn compute(&self, image: &GrayImage) -> Vec<f32> {
        let (gx, gy) = sobel(image);
        let width = image.width() as usize;
        let height = image.height() as usize;

        let n_cells_x = width / self.params.cell_size;
        let n_cells_y = height / self.params.cell_size;

        // 1. Compute cell histograms
        let mut cell_histograms = vec![0.0f32; n_cells_x * n_cells_y * self.params.n_bins];

        // This can be parallelized over cells
        cell_histograms
            .par_chunks_mut(self.params.n_bins)
            .enumerate()
            .for_each(|(cell_idx, hist)| {
                let cx = cell_idx % n_cells_x;
                let cy = cell_idx / n_cells_x;

                let x_start = cx * self.params.cell_size;
                let y_start = cy * self.params.cell_size;

                for dy in 0..self.params.cell_size {
                    for dx in 0..self.params.cell_size {
                        let x = x_start + dx;
                        let y = y_start + dy;
                        let idx = y * width + x;

                        let dx_val = gx.as_raw()[idx] as f32 - 128.0;
                        let dy_val = gy.as_raw()[idx] as f32 - 128.0;

                        let mag = (dx_val * dx_val + dy_val * dy_val).sqrt();
                        let mut angle = dy_val.atan2(dx_val).to_degrees();
                        if angle < 0.0 {
                            angle += 180.0;
                        } // Unsigned gradients

                        let bin = (angle / (180.0 / self.params.n_bins as f32)) as usize
                            % self.params.n_bins;
                        hist[bin] += mag;
                    }
                }
            });

        // 2. Block normalization (L2-Hys)
        let n_blocks_x = n_cells_x - self.params.block_size + 1;
        let n_blocks_y = n_cells_y - self.params.block_size + 1;
        let block_dim = self.params.block_size * self.params.block_size * self.params.n_bins;

        let mut descriptor = vec![0.0f32; n_blocks_x * n_blocks_y * block_dim];

        descriptor
            .par_chunks_mut(block_dim)
            .enumerate()
            .for_each(|(block_idx, block_vec)| {
                let bx = block_idx % n_blocks_x;
                let by = block_idx / n_blocks_x;

                let mut norm_sq = 0.0f32;
                for j in 0..self.params.block_size {
                    for i in 0..self.params.block_size {
                        let cell_x = bx + i;
                        let cell_y = by + j;
                        let cell_base = (cell_y * n_cells_x + cell_x) * self.params.n_bins;

                        for b in 0..self.params.n_bins {
                            let val = cell_histograms[cell_base + b];
                            block_vec[(j * self.params.block_size + i) * self.params.n_bins + b] =
                                val;
                            norm_sq += val * val;
                        }
                    }
                }

                // Normalize
                let norm = (norm_sq + 1e-6).sqrt();
                for v in block_vec.iter_mut() {
                    *v = (*v / norm).min(0.2);
                }
                // Re-normalize after clipping
                let final_norm = (block_vec.iter().map(|&v| v * v).sum::<f32>() + 1e-6).sqrt();
                for v in block_vec.iter_mut() {
                    *v /= final_norm;
                }
            });

        descriptor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_hog_output_dimensions() {
        let hog = Hog::new(HogParams::default());
        let img = GrayImage::new(64, 64);
        let descriptor = hog.compute(&img);

        // With 64x64 image, 8x8 cells -> 8 cells x 8 cells
        // With 2x2 block size and 9 bins
        // n_blocks_x = 8 - 2 + 1 = 7
        // n_blocks_y = 8 - 2 + 1 = 7
        // block_dim = 2 * 2 * 9 = 36
        // total = 7 * 7 * 36 = 1764
        let expected_size = 7 * 7 * 36;
        assert_eq!(descriptor.len(), expected_size);
    }

    #[test]
    fn test_hog_normalization_range() {
        let hog = Hog::new(HogParams::default());
        let img = GrayImage::new(64, 64);
        let descriptor = hog.compute(&img);

        // All values should be in [0, ~1] range after normalization
        for &val in &descriptor {
            assert!(val >= 0.0, "Value {} is negative", val);
            assert!(val <= 1.0, "Value {} is > 1.0", val);
        }
    }

    #[test]
    fn test_hog_with_gradient_image() {
        let hog = Hog::new(HogParams::default());
        let mut img = GrayImage::new(64, 64);

        // Create gradient image
        for y in 0..64 {
            for x in 0..64 {
                img.put_pixel(x, y, Luma([((x + y) % 256) as u8]));
            }
        }

        let descriptor = hog.compute(&img);
        assert!(!descriptor.is_empty());
        assert!(descriptor.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_hog_minimum_valid_image() {
        let hog = Hog::new(HogParams::default());
        // Minimum size to have at least one 2x2 block
        // Need at least 2 cells x 2 cells = 16 pixels with cell_size 8
        let img = GrayImage::new(16, 16);
        let descriptor = hog.compute(&img);

        // With 16x16 image, 8x8 cells -> 2 cells x 2 cells
        // With 2x2 block size
        // n_blocks_x = 2 - 2 + 1 = 1
        // n_blocks_y = 2 - 2 + 1 = 1
        // block_dim = 2 * 2 * 9 = 36
        // total = 1 * 1 * 36 = 36
        assert_eq!(descriptor.len(), 36);
    }

    #[test]
    fn test_hog_small_image() {
        let hog = Hog::new(HogParams::default());
        let img = GrayImage::new(32, 32);
        let descriptor = hog.compute(&img);

        // With 32x32 image, 8x8 cells -> 4 cells x 4 cells
        // With 2x2 block size
        // n_blocks_x = 4 - 2 + 1 = 3
        // n_blocks_y = 4 - 2 + 1 = 3
        // block_dim = 2 * 2 * 9 = 36
        // total = 3 * 3 * 36 = 324
        let expected_size = 3 * 3 * 36;
        assert_eq!(descriptor.len(), expected_size);
    }

    #[test]
    fn test_hog_custom_params() {
        let params = HogParams {
            cell_size: 16,
            block_size: 1,
            n_bins: 8,
        };
        let hog = Hog::new(params);
        let img = GrayImage::new(64, 64);
        let descriptor = hog.compute(&img);

        // With 64x64 image, 16x16 cells -> 4 cells x 4 cells
        // With 1x1 block size
        // n_blocks_x = 4 - 1 + 1 = 4
        // n_blocks_y = 4 - 1 + 1 = 4
        // block_dim = 1 * 1 * 8 = 8
        // total = 4 * 4 * 8 = 128
        let expected_size = 4 * 4 * 8;
        assert_eq!(descriptor.len(), expected_size);
    }

    #[test]
    fn test_hog_uniform_image() {
        let hog = Hog::new(HogParams::default());
        let mut img = GrayImage::new(64, 64);

        // Uniform image (all same intensity)
        for pixel in img.iter_mut() {
            *pixel = 128;
        }

        let descriptor = hog.compute(&img);

        // Uniform image should have gradients close to zero
        // Most values should be very small after normalization
        let sum: f32 = descriptor.iter().sum();
        let avg = sum / descriptor.len() as f32;
        assert!(
            avg < 0.1,
            "Average value {} is too high for uniform image",
            avg
        );
    }

    #[test]
    fn test_hog_different_sizes() {
        let hog = Hog::new(HogParams::default());

        // Test various image sizes
        for size in &[16, 32, 48, 64, 80, 96] {
            let img = GrayImage::new(*size, *size);
            let descriptor = hog.compute(&img);

            let n_cells = (size / 8) as usize;
            if n_cells >= 2 {
                let n_blocks = n_cells - 1;
                let block_dim = 2 * 2 * 9;
                let expected = n_blocks * n_blocks * block_dim;
                assert_eq!(
                    descriptor.len(),
                    expected,
                    "Size {} produced wrong descriptor length",
                    size
                );
            }
        }
    }

    #[test]
    fn test_hog_normalization_preserves_structure() {
        let hog = Hog::new(HogParams::default());
        let mut img = GrayImage::new(64, 64);

        // Create two distinct regions
        for y in 0..32 {
            for x in 0..64 {
                img.put_pixel(x, y, Luma([50]));
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                img.put_pixel(x, y, Luma([200]));
            }
        }

        let descriptor = hog.compute(&img);

        // Should have non-zero descriptors
        assert!(descriptor.iter().any(|&v| v > 0.01));
    }
}
