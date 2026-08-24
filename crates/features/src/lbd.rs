//! Line Binary Descriptor (LBD) for line matching
//!
//! LBD is a robust line descriptor that uses local gradient information.

use crate::line_matcher::LineDescriptor;
use image::GrayImage;
use rayon::prelude::*;

/// Parameters for the LBD (Line Binary Descriptor) extractor.
pub struct LbdParams {
    /// Number of scale layers used when sampling orthogonal gradients.
    pub n_layers: usize,
    /// Width of the sampling band around each line segment (in pixels).
    pub n_bandwidth: usize,
}

impl Default for LbdParams {
    fn default() -> Self {
        Self {
            n_layers: 3,
            n_bandwidth: 7,
        }
    }
}

/// Line Binary Descriptor extractor.
pub struct Lbd {
    params: LbdParams,
}

impl Lbd {
    /// Create a new LBD extractor with the given parameters.
    pub fn new(params: LbdParams) -> Self {
        Self { params }
    }

    /// Compute LBD descriptors for a set of line segments in the image.
    ///
    /// Samples 32 points along each segment; at each point compares gradient
    /// magnitudes across `params.n_bandwidth` paired offsets orthogonal to
    /// the line. The normal is CANONICALLY oriented (fixed half-plane rule)
    /// so that swapping a segment's endpoints — which Hough output does
    /// arbitrarily — yields the identical descriptor instead of a bitwise
    /// inversion.
    #[allow(clippy::needless_range_loop)]
    pub fn compute(
        &self,
        image: &GrayImage,
        segments: &[cv_imgproc::hough::LineSegment],
    ) -> Vec<LineDescriptor> {
        let (gx, gy) = cv_imgproc::sobel(image);
        let gx_raw = gx.as_raw();
        let gy_raw = gy.as_raw();
        let (w, h) = image.dimensions();

        let bandwidth = self.params.n_bandwidth.max(1);
        // Spacing between paired samples: keep the band inside ~7px total
        // span regardless of requested width.
        let step = 1.5f32 * 7.0 / bandwidth as f32;
        let n_bits = bandwidth.min(8).max(1);

        segments
            .par_iter()
            .map(|seg| {
                let mut packed = [0u8; 32];

                let dx = seg.x2 - seg.x1;
                let dy = seg.y2 - seg.y1;
                let len = (dx * dx + dy * dy).sqrt();

                if len > 1e-5 {
                    // Orthogonal vector, canonically oriented into the upper
                    // half-plane (or +x when horizontal) so endpoint order
                    // does not flip descriptor bits.
                    let mut nx = -dy / len;
                    let mut ny = dx / len;
                    if ny < 0.0 || (ny.abs() <= 1e-6 && nx < 0.0) {
                        nx = -nx;
                        ny = -ny;
                    }

                    for i in 0..32 {
                        let t = i as f32 / 31.0;
                        let lx = seg.x1 + t * dx;
                        let ly = seg.y1 + t * dy;

                        let mut byte_val = 0u8;
                        for bit in 0..n_bits {
                            // Center the offsets around zero, positive side
                            // along the canonical normal.
                            let offset =
                                (bit as f32 - (n_bits as f32 - 1.0) / 2.0) * step;
                            let x_p = (lx + nx * offset).round() as i32;
                            let y_p = (ly + ny * offset).round() as i32;
                            let x_n = (lx - nx * offset).round() as i32;
                            let y_n = (ly - ny * offset).round() as i32;

                            if x_p >= 0
                                && x_p < w as i32
                                && y_p >= 0
                                && y_p < h as i32
                                && x_n >= 0
                                && x_n < w as i32
                                && y_n >= 0
                                && y_n < h as i32
                            {
                                let idx_p = y_p as usize * w as usize + x_p as usize;
                                let idx_n = y_n as usize * w as usize + x_n as usize;

                                let g_p =
                                    (gx_raw[idx_p] as f32).abs() + (gy_raw[idx_p] as f32).abs();
                                let g_n =
                                    (gx_raw[idx_n] as f32).abs() + (gy_raw[idx_n] as f32).abs();

                                if g_p > g_n {
                                    byte_val |= 1 << bit;
                                }
                            }
                        }
                        packed[i] = byte_val;
                    }
                }

                LineDescriptor {
                    data: packed.to_vec(),
                    segment: *seg,
                }
            })
            .collect()
    }
}
