use image::GrayImage;
use rayon::prelude::*;
use std::f32::consts::PI;

/// Parameters for HOG (Histogram of Oriented Gradients) descriptor computation.
pub struct HogParams {
    /// Width/height in pixels of each HOG cell.
    pub cell_size: usize,
    /// Number of cells per block edge (block is `block_size × block_size` cells).
    pub block_size: usize,
    /// Number of orientation bins per cell histogram (unsigned gradients, 0–180°).
    pub n_bins: usize,
}

impl Default for HogParams {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2, // 2x2 cells
            n_bins: 9,
        }
    }
}

/// Compute the HOG descriptor for a grayscale image.
///
/// Divides the image into cells, accumulates gradient orientation histograms per cell,
/// L2-normalizes over overlapping blocks, and concatenates the result into a flat vector.
pub fn compute_hog(image: &GrayImage, params: &HogParams) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;

    // 1. Compute gradients in parallel
    let mut grad_mag = vec![0.0f32; width * height];
    let mut grad_ori = vec![0.0f32; width * height];

    grad_mag
        .par_chunks_mut(width)
        .zip(grad_ori.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (row_mag, row_ori))| {
            if y == 0 || y == height - 1 {
                return;
            }
            for x in 1..width - 1 {
                let gx = image.get_pixel((x + 1) as u32, y as u32)[0] as f32
                    - image.get_pixel((x - 1) as u32, y as u32)[0] as f32;
                let gy = image.get_pixel(x as u32, (y + 1) as u32)[0] as f32
                    - image.get_pixel(x as u32, (y - 1) as u32)[0] as f32;

                row_mag[x] = (gx * gx + gy * gy).sqrt();
                let mut ori = gy.atan2(gx) * 180.0 / PI;
                if ori < 0.0 {
                    ori += 180.0;
                } // Unsigned gradients
                row_ori[x] = ori;
            }
        });

    // 2. Cell Histogramming
    let n_cells_x = width / params.cell_size;
    let n_cells_y = height / params.cell_size;
    let mut cell_hists = vec![0.0f32; n_cells_x * n_cells_y * params.n_bins];

    cell_hists
        .par_chunks_mut(n_cells_x * params.n_bins)
        .enumerate()
        .for_each(|(cy, row_hists)| {
            for dy in 0..params.cell_size {
                let y = cy * params.cell_size + dy;
                if y >= height {
                    continue;
                }

                for x in 0..width {
                    let cx = x / params.cell_size;
                    if cx >= n_cells_x {
                        continue;
                    }

                    let mag = grad_mag[y * width + x];
                    let ori = grad_ori[y * width + x];

                    let bin = (ori / (180.0 / params.n_bins as f32)).floor() as usize;
                    let bin = bin.min(params.n_bins - 1);

                    // Note: This part has multiple threads writing to the same cx-bin.
                    // However, each cy is processed by a single chunk here, so rows are independent.
                    // dy loop ensures we process all pixels in this cell row.
                    row_hists[cx * params.n_bins + bin] += mag;
                }
            }
        });

    // 3. Block Normalization
    let b_w = params.block_size;
    let n_blocks_x = n_cells_x - b_w + 1;
    let n_blocks_y = n_cells_y - b_w + 1;
    let block_dim = b_w * b_w * params.n_bins;
    let mut descriptor = vec![0.0f32; n_blocks_x * n_blocks_y * block_dim];

    descriptor
        .par_chunks_mut(n_blocks_x * block_dim)
        .enumerate()
        .for_each(|(by, row_desc)| {
            for bx in 0..n_blocks_x {
                let mut block_vec = Vec::with_capacity(block_dim);
                for dy in 0..b_w {
                    for dx in 0..b_w {
                        let c_idx = (by + dy) * n_cells_x + (bx + dx);
                        for b in 0..params.n_bins {
                            block_vec.push(cell_hists[c_idx * params.n_bins + b]);
                        }
                    }
                }

                // L2 normalization
                let norm = block_vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    for v in &mut block_vec {
                        *v /= norm;
                    }
                }

                let start = bx * block_dim;
                row_desc[start..start + block_dim].copy_from_slice(&block_vec);
            }
        });

    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_image() -> GrayImage {
        let mut img = GrayImage::new(64, 128);
        for y in 0..128 {
            for x in 0..64 {
                let val = ((x / 8 + y / 8) % 2 * 255) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }

    fn create_uniform_image() -> GrayImage {
        GrayImage::from_pixel(64, 128, Luma([128]))
    }

    #[test]
    fn test_hog_params_default() {
        let params = HogParams::default();
        assert_eq!(params.cell_size, 8);
        assert_eq!(params.block_size, 2);
        assert_eq!(params.n_bins, 9);
    }

    #[test]
    fn test_compute_hog_basic() {
        let img = create_test_image();
        let params = HogParams::default();
        let descriptor = compute_hog(&img, &params);

        assert!(!descriptor.is_empty());
    }

    #[test]
    fn test_compute_hog_descriptor_size() {
        let img = create_test_image();
        let params = HogParams::default();
        let descriptor = compute_hog(&img, &params);

        let n_cells_x = 64 / 8;
        let n_cells_y = 128 / 8;
        let n_blocks_x = n_cells_x - 2 + 1;
        let n_blocks_y = n_cells_y - 2 + 1;
        let block_dim = 2 * 2 * 9;
        let expected_size = n_blocks_x * n_blocks_y * block_dim;

        assert_eq!(descriptor.len(), expected_size);
    }

    #[test]
    fn test_compute_hog_uniform_image() {
        let img = create_uniform_image();
        let params = HogParams::default();
        let descriptor = compute_hog(&img, &params);

        assert!(!descriptor.is_empty());
        let has_nonzero: bool = descriptor.iter().any(|&v| v > 0.0);
        assert!(!has_nonzero, "Uniform image should produce zero gradients");
    }

    #[test]
    fn test_compute_hog_custom_params() {
        let img = create_test_image();
        let params = HogParams {
            cell_size: 16,
            block_size: 3,
            n_bins: 12,
        };
        let descriptor = compute_hog(&img, &params);

        assert!(!descriptor.is_empty());
    }

    #[test]
    fn test_compute_hog_normalized() {
        let img = create_test_image();
        let params = HogParams::default();
        let descriptor = compute_hog(&img, &params);

        let block_dim = 2 * 2 * 9;
        for block_start in (0..descriptor.len()).step_by(block_dim) {
            let block_end = (block_start + block_dim).min(descriptor.len());
            let block = &descriptor[block_start..block_end];
            let norm: f32 = block.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 1e-6 {
                assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-6);
            }
        }
    }
}
