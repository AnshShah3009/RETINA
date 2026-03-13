//! Advanced morphological operations: skeleton (thinning), morphological gradient,
//! top-hat, black-hat, and hit-or-miss transforms.
//!
//! These build on the basic dilate/erode primitives in [`crate::morph`].

use crate::morph::{closing, create_morph_kernel, opening, MorphShape};
use image::GrayImage;
use rayon::prelude::*;

/// Morphological skeleton via Zhang-Suen thinning.
///
/// Iteratively removes boundary pixels while preserving topology until the
/// result is one-pixel wide. Input should be a binary image (0 = background,
/// non-zero = foreground).
pub fn skeleton(binary: &GrayImage) -> GrayImage {
    let width = binary.width() as usize;
    let height = binary.height() as usize;
    if width == 0 || height == 0 {
        return binary.clone();
    }

    // Work on a bool grid for speed.
    let mut img: Vec<bool> = binary.as_raw().iter().map(|&v| v > 0).collect();
    let mut changed = true;

    while changed {
        changed = false;

        // Sub-iteration 1
        let to_remove = zhang_suen_pass(&img, width, height, true);
        if !to_remove.is_empty() {
            changed = true;
            for &idx in &to_remove {
                img[idx] = false;
            }
        }

        // Sub-iteration 2
        let to_remove = zhang_suen_pass(&img, width, height, false);
        if !to_remove.is_empty() {
            changed = true;
            for &idx in &to_remove {
                img[idx] = false;
            }
        }
    }

    let mut out = GrayImage::new(binary.width(), binary.height());
    for (i, &v) in img.iter().enumerate() {
        out.as_mut()[i] = if v { 255 } else { 0 };
    }
    out
}

/// One sub-iteration of the Zhang-Suen thinning algorithm.
/// `step_one` selects which of the two conditions to check.
fn zhang_suen_pass(img: &[bool], width: usize, height: usize, step_one: bool) -> Vec<usize> {
    let mut remove = Vec::new();

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            if !img[idx] {
                continue;
            }

            // 8-neighbors in order: P2..P9 (clockwise from top)
            //   P9 P2 P3
            //   P8 P1 P4
            //   P7 P6 P5
            let p2 = img[(y - 1) * width + x] as u8;
            let p3 = img[(y - 1) * width + (x + 1)] as u8;
            let p4 = img[y * width + (x + 1)] as u8;
            let p5 = img[(y + 1) * width + (x + 1)] as u8;
            let p6 = img[(y + 1) * width + x] as u8;
            let p7 = img[(y + 1) * width + (x - 1)] as u8;
            let p8 = img[y * width + (x - 1)] as u8;
            let p9 = img[(y - 1) * width + (x - 1)] as u8;

            let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];

            // B(P1): number of non-zero neighbors
            let b: u8 = neighbors.iter().sum();
            if !(2..=6).contains(&b) {
                continue;
            }

            // A(P1): number of 0->1 transitions in the ordered sequence P2..P9..P2
            let mut a = 0u8;
            for i in 0..8 {
                if neighbors[i] == 0 && neighbors[(i + 1) % 8] == 1 {
                    a += 1;
                }
            }
            if a != 1 {
                continue;
            }

            if step_one {
                // Step 1: P2 * P4 * P6 == 0  AND  P4 * P6 * P8 == 0
                if p2 * p4 * p6 != 0 {
                    continue;
                }
                if p4 * p6 * p8 != 0 {
                    continue;
                }
            } else {
                // Step 2: P2 * P4 * P8 == 0  AND  P2 * P6 * P8 == 0
                if p2 * p4 * p8 != 0 {
                    continue;
                }
                if p2 * p6 * p8 != 0 {
                    continue;
                }
            }

            remove.push(idx);
        }
    }

    remove
}

/// Top-hat transform: `image - opening(image)`.
///
/// Extracts bright features smaller than the structuring element from an
/// uneven background. Uses a rectangular kernel of `kernel_size x kernel_size`.
pub fn top_hat(image: &GrayImage, kernel_size: usize) -> GrayImage {
    let kernel = create_morph_kernel(
        MorphShape::Rectangle,
        kernel_size as u32,
        kernel_size as u32,
    );
    let opened = opening(image, &kernel, 1);

    let width = image.width() as usize;
    let mut out = GrayImage::new(image.width(), image.height());
    let src_raw = image.as_raw();
    let op_raw = opened.as_raw();
    let out_raw = out.as_mut();

    out_raw
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let off = y * width;
            for x in 0..width {
                row[x] = src_raw[off + x].saturating_sub(op_raw[off + x]);
            }
        });

    out
}

/// Black-hat transform: `closing(image) - image`.
///
/// Extracts dark features (holes, gaps) smaller than the structuring element.
/// Uses a rectangular kernel of `kernel_size x kernel_size`.
pub fn black_hat(image: &GrayImage, kernel_size: usize) -> GrayImage {
    let kernel = create_morph_kernel(
        MorphShape::Rectangle,
        kernel_size as u32,
        kernel_size as u32,
    );
    let closed = closing(image, &kernel, 1);

    let width = image.width() as usize;
    let mut out = GrayImage::new(image.width(), image.height());
    let cl_raw = closed.as_raw();
    let src_raw = image.as_raw();
    let out_raw = out.as_mut();

    out_raw
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let off = y * width;
            for x in 0..width {
                row[x] = cl_raw[off + x].saturating_sub(src_raw[off + x]);
            }
        });

    out
}

/// Hit-or-miss transform.
///
/// The kernel is a 2D array of `i8` values:
///   *  1 = must be foreground (non-zero)
///   *  0 = don't care
///   * -1 = must be background (zero)
///
/// The result pixel is 255 where the pattern matches, 0 otherwise.
/// `kernel` should be a flat row-major slice with dimensions `kw x kh`.
#[allow(clippy::needless_range_loop)]
pub fn hit_or_miss(image: &GrayImage, kernel: &[i8], kw: usize, kh: usize) -> GrayImage {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let cx = kw / 2;
    let cy = kh / 2;

    let src = image.as_raw();
    let mut out = GrayImage::new(image.width(), image.height());
    let out_raw = out.as_mut();

    out_raw
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            'pixel: for x in 0..width {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let k_val = kernel[ky * kw + kx];
                        if k_val == 0 {
                            continue;
                        }
                        let px = x as isize + kx as isize - cx as isize;
                        let py = y as isize + ky as isize - cy as isize;

                        if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                            // Out-of-bounds treated as background (0).
                            if k_val == 1 {
                                row[x] = 0;
                                continue 'pixel;
                            }
                            // k_val == -1 expects background, which is satisfied.
                            continue;
                        }

                        let pix = src[py as usize * width + px as usize];
                        if k_val == 1 && pix == 0 {
                            row[x] = 0;
                            continue 'pixel;
                        }
                        if k_val == -1 && pix != 0 {
                            row[x] = 0;
                            continue 'pixel;
                        }
                    }
                }
                row[x] = 255;
            }
        });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    // -----------------------------------------------------------------------
    // Skeleton tests
    // -----------------------------------------------------------------------

    #[test]
    fn skeleton_thin_line() {
        // Create a thick horizontal bar (3 pixels tall, 10 wide).
        let mut img = GrayImage::new(12, 5);
        for y in 1..4 {
            for x in 1..11 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let skel = skeleton(&img);

        // Result should be 1-pixel wide: count distinct y-values with white on each x.
        for x in 2..10 {
            let mut white_count = 0u32;
            for y in 0..5 {
                if skel.get_pixel(x, y)[0] > 0 {
                    white_count += 1;
                }
            }
            assert!(
                white_count <= 1,
                "Column {} has {} white pixels, expected at most 1",
                x,
                white_count
            );
        }
    }

    #[test]
    fn skeleton_single_pixel_unchanged() {
        let mut img = GrayImage::new(5, 5);
        img.put_pixel(2, 2, Luma([255]));
        let skel = skeleton(&img);
        // A single isolated pixel cannot be thinned further — it stays.
        assert_eq!(skel.get_pixel(2, 2)[0], 255);
    }

    #[test]
    fn skeleton_empty_image() {
        let img = GrayImage::new(5, 5);
        let skel = skeleton(&img);
        assert!(skel.as_raw().iter().all(|&v| v == 0));
    }

    // -----------------------------------------------------------------------
    // Morphological gradient tests
    // -----------------------------------------------------------------------

    #[test]
    fn morph_gradient_edge_detection() {
        let mut img = GrayImage::new(16, 16);
        for y in 4..12 {
            for x in 4..12 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let grad = crate::morph::morphological_gradient(&img, 3);

        // Interior pixels should be ~0 (dilate and erode both 255).
        assert_eq!(grad.get_pixel(8, 8)[0], 0);
        // Boundary pixels should be non-zero.
        assert!(grad.as_raw().iter().any(|&v| v > 0));
    }

    #[test]
    fn morph_gradient_uniform_is_zero() {
        let img = GrayImage::from_pixel(10, 10, Luma([128]));
        let grad = crate::morph::morphological_gradient(&img, 3);
        // Uniform image => gradient should be zero everywhere (ignoring border effects).
        let interior_zero = (2..8).all(|y| (2..8).all(|x| grad.get_pixel(x, y)[0] == 0));
        assert!(interior_zero);
    }

    // -----------------------------------------------------------------------
    // Top-hat tests
    // -----------------------------------------------------------------------

    #[test]
    fn top_hat_extracts_bright_blob() {
        // Uneven background (value 100) with a small bright blob (value 200).
        let mut img = GrayImage::from_pixel(20, 20, Luma([100]));
        // Small 2x2 bright spot.
        for y in 9..11 {
            for x in 9..11 {
                img.put_pixel(x, y, Luma([200]));
            }
        }
        let th = top_hat(&img, 7);

        // The bright spot should produce non-zero values in the top-hat result.
        let center_val = th.get_pixel(9, 9)[0];
        assert!(
            center_val > 0,
            "Top-hat should extract bright blob, got {}",
            center_val
        );
    }

    #[test]
    fn top_hat_uniform_is_zero() {
        let img = GrayImage::from_pixel(10, 10, Luma([128]));
        let th = top_hat(&img, 3);
        assert!(th.as_raw().iter().all(|&v| v == 0));
    }

    // -----------------------------------------------------------------------
    // Black-hat tests
    // -----------------------------------------------------------------------

    #[test]
    fn black_hat_extracts_dark_hole() {
        // Bright background with a small dark spot.
        let mut img = GrayImage::from_pixel(20, 20, Luma([200]));
        for y in 9..11 {
            for x in 9..11 {
                img.put_pixel(x, y, Luma([50]));
            }
        }
        let bh = black_hat(&img, 7);

        let center_val = bh.get_pixel(9, 9)[0];
        assert!(
            center_val > 0,
            "Black-hat should extract dark hole, got {}",
            center_val
        );
    }

    #[test]
    fn black_hat_uniform_is_zero() {
        let img = GrayImage::from_pixel(10, 10, Luma([128]));
        let bh = black_hat(&img, 3);
        assert!(bh.as_raw().iter().all(|&v| v == 0));
    }

    // -----------------------------------------------------------------------
    // Hit-or-miss tests
    // -----------------------------------------------------------------------

    #[test]
    fn hit_or_miss_detects_pattern() {
        // Detect isolated white pixel: center=1, all neighbors=-1.
        let kernel: Vec<i8> = vec![-1, -1, -1, -1, 1, -1, -1, -1, -1];
        let mut img = GrayImage::new(5, 5);
        img.put_pixel(2, 2, Luma([255])); // isolated pixel

        let result = hit_or_miss(&img, &kernel, 3, 3);
        assert_eq!(result.get_pixel(2, 2)[0], 255);
        // Non-matching locations should be 0.
        assert_eq!(result.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn hit_or_miss_no_match() {
        // Kernel expects isolated pixel, but we have a 3x3 block.
        let kernel: Vec<i8> = vec![-1, -1, -1, -1, 1, -1, -1, -1, -1];
        let mut img = GrayImage::new(5, 5);
        for y in 1..4 {
            for x in 1..4 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let result = hit_or_miss(&img, &kernel, 3, 3);
        // No pixel should match because neighbors are also white.
        assert!(result.as_raw().iter().all(|&v| v == 0));
    }

    #[test]
    fn hit_or_miss_dont_care() {
        // Kernel with don't-care: center=1, rest=0 (don't care).
        let kernel: Vec<i8> = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mut img = GrayImage::new(5, 5);
        img.put_pixel(2, 2, Luma([255]));
        img.put_pixel(1, 1, Luma([255])); // neighbor, but don't care

        let result = hit_or_miss(&img, &kernel, 3, 3);
        // Both white pixels have center=1 and rest don't care.
        assert_eq!(result.get_pixel(2, 2)[0], 255);
        assert_eq!(result.get_pixel(1, 1)[0], 255);
    }

    // -----------------------------------------------------------------------
    // Dimension preservation tests
    // -----------------------------------------------------------------------

    #[test]
    fn all_outputs_preserve_dimensions() {
        let img = GrayImage::new(32, 24);
        assert_eq!(skeleton(&img).dimensions(), (32, 24));
        assert_eq!(
            crate::morph::morphological_gradient(&img, 3).dimensions(),
            (32, 24)
        );
        assert_eq!(top_hat(&img, 3).dimensions(), (32, 24));
        assert_eq!(black_hat(&img, 3).dimensions(), (32, 24));
        let k = vec![0i8; 9];
        assert_eq!(hit_or_miss(&img, &k, 3, 3).dimensions(), (32, 24));
    }
}
