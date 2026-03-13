//! Haar Cascade Face Detection
//!
//! Implementation of the Viola-Jones object detection framework
//! using Haar-like features and boosted cascades.

use cv_core::{Error, Rect, Result, Tensor};
use image::GrayImage;

/// Haar Cascade Classifier for object detection
///
/// Implements the Viola-Jones cascade classifier using Haar-like features.
/// A cascade of decision trees enables fast detection by rejecting non-object
/// regions early in the pipeline.
///
/// # Algorithm Overview
///
/// The cascade evaluates a series of stages, each containing a set of boosted
/// Haar features. Detection succeeds only if a window passes all stages.
/// Multi-scale detection is performed by evaluating at different scales.
///
/// # Usage
///
/// ```no_run
/// # use cv_objdetect::haar::HaarCascade;
/// # use image::GrayImage;
/// let cascade = HaarCascade { stages: vec![], size: (24, 24) };
/// let image = GrayImage::new(640, 480);
/// let detections = cascade.detect(&image, 1.1, 3)?;
/// # Ok::<(), cv_core::Error>(())
/// ```
pub struct HaarCascade {
    /// Cascade stages, evaluated in sequence
    pub stages: Vec<CascadeStage>,
    /// Expected input window size (width, height) for this cascade
    pub size: (u32, u32),
}

/// A single stage in the Haar cascade classifier
///
/// Each stage contains a set of weighted Haar features that are summed to
/// produce a stage response. The window is rejected if this response falls
/// below the stage threshold.
pub struct CascadeStage {
    /// Decision threshold for this stage
    pub threshold: f32,
    /// Haar features evaluated in this stage
    pub features: Vec<HaarFeature>,
}

/// A single Haar-like feature with multiple weighted rectangular regions
///
/// Haar features compute weighted sums of pixel values over axis-aligned
/// rectangles. They approximate local image gradients and are fast to
/// evaluate using integral images.
///
/// # Structure
///
/// A Haar feature consists of:
/// - Multiple weighted rectangular regions (positive and negative weights)
/// - A decision threshold for splitting responses
/// - Left and right values (branch weights in the boosted tree)
pub struct HaarFeature {
    /// Rectangles and their weights (positive/negative contributions)
    pub rects: Vec<(Rect, f32)>, // (rectangle, weight)
    /// Feature response threshold for decision splitting
    pub threshold: f32,
    /// Weight if feature response is below threshold
    pub left_val: f32,
    /// Weight if feature response is above or equal to threshold
    pub right_val: f32,
}

impl HaarCascade {
    /// Detect objects in image using Haar cascade classifier
    ///
    /// Performs multi-scale sliding window detection by evaluating the cascade
    /// classifier at different image scales. Fast rejections occur at early stages
    /// to avoid evaluating expensive later stages.
    ///
    /// # Algorithm Details
    ///
    /// 1. **Integral Image**: Pre-computed for O(1) rectangle sum queries
    /// 2. **Multi-scale Evaluation**: Slides a detection window at increasing scales
    /// 3. **Cascade Evaluation**: Each window evaluated through all stages
    /// 4. **Early Rejection**: Windows failing any stage are immediately rejected
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image (any dimensions)
    ///   - Image should be large enough to contain target objects
    ///   - Recommend minimum dimension ≥ cascade size
    ///
    /// * `scale_factor` - Multiplicative scale increase per iteration (typically 1.05-1.4)
    ///   - Smaller values (1.05) → slower but more detections
    ///   - Larger values (1.4) → faster but may miss detections
    ///   - Typical recommended: 1.1
    ///
    /// * `_min_neighbors` - Minimum neighbors for grouping (reserved for future use)
    ///   - Currently unused; kept for API compatibility
    ///   - Future versions may implement non-maximal suppression
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Rect>)` - Detected object bounding rectangles in image coordinates
    ///   - Each rectangle represents one detection
    ///   - Coordinates: (x, y) = top-left, (w, h) = width/height
    ///   - May contain overlapping detections (no grouping applied yet)
    ///
    /// * `Err(Error)` - If computation fails
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if:
    /// - Integral image tensor allocation fails
    /// - Tensor data access fails
    ///
    /// # Computational Complexity
    ///
    /// - Integral Image: O(H × W)
    /// - Detection: O(H × W × log_scale(H,W) × S × F)
    ///   - S = number of stages
    ///   - F = average features per stage
    /// - Total: Fast in practice due to early rejections
    ///
    /// # Notes
    ///
    /// - Windows are evaluated on integer pixel boundaries
    /// - Step size increases with scale to reduce redundant evaluations
    /// - No grouping/suppression of overlapping detections
    /// - Cascade trained on specific object class (e.g., frontal faces)
    pub fn detect(
        &self,
        image: &GrayImage,
        scale_factor: f32,
        _min_neighbors: u32,
    ) -> Result<Vec<Rect>> {
        // 1. Compute integral image
        let integral = compute_integral_image(image)?;

        let mut detections = Vec::new();
        let (img_w, img_h) = image.dimensions();

        let mut scale = 1.0f32;
        while (scale * self.size.0 as f32) < img_w as f32
            && (scale * self.size.1 as f32) < img_h as f32
        {
            let win_w = (self.size.0 as f32 * scale) as u32;
            let win_h = (self.size.1 as f32 * scale) as u32;
            let step = (scale * 2.0).max(1.0) as u32;

            for y in (0..img_h - win_h).step_by(step as usize) {
                for x in (0..img_w - win_w).step_by(step as usize) {
                    if self.evaluate_window(&integral, x, y, scale)? {
                        detections.push(Rect::new(x as f32, y as f32, win_w as f32, win_h as f32));
                    }
                }
            }
            scale *= scale_factor;
        }

        // 2. Group detections (min_neighbors) - simplified
        Ok(detections)
    }

    fn evaluate_window(&self, integral: &Tensor<u32>, x: u32, y: u32, scale: f32) -> Result<bool> {
        for stage in &self.stages {
            let mut stage_sum = 0.0f32;
            for feature in &stage.features {
                let feature_sum = feature.evaluate(integral, x, y, scale)?;
                stage_sum += if feature_sum < feature.threshold * scale * scale {
                    feature.left_val
                } else {
                    feature.right_val
                };
            }
            if stage_sum < stage.threshold {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl HaarFeature {
    fn evaluate(&self, integral: &Tensor<u32>, ox: u32, oy: u32, scale: f32) -> Result<f32> {
        let mut sum = 0.0f32;
        for (r, weight) in &self.rects {
            let rx = ox + (r.x * scale) as u32;
            let ry = oy + (r.y * scale) as u32;
            let rw = (r.w * scale) as u32;
            let rh = (r.h * scale) as u32;

            sum += get_rect_sum(integral, rx, ry, rw, rh)? as f32 * weight;
        }
        Ok(sum)
    }
}

/// Compute 2D cumulative sum (integral) image for O(1) rectangular region queries
///
/// The integral image `I(x,y)` equals the sum of all pixels in rectangle from
/// (0,0) to (x,y). Any rectangular region sum can then be computed in O(1) time
/// using three lookups: `I(x1,y1) + I(x0,y0) - I(x1,y0) - I(x0,y1)`.
///
/// # Arguments
///
/// * `src` - Input grayscale image
///
/// # Returns
///
/// * `Ok(Tensor<u32>)` - Integral image with dimensions (1, h+1, w+1)
///   - Extra row/column at index 0 for boundary handling
/// * `Err(MemoryError)` - If tensor allocation fails
///
/// # Algorithm
///
/// Uses 2D prefix sum: `I[y][x] = I[y-1][x] + I[y][x-1] - I[y-1][x-1] + pixel[y][x]`
fn compute_integral_image(src: &GrayImage) -> Result<Tensor<u32>> {
    let (w, h) = src.dimensions();
    let mut integral = vec![0u32; (w as usize + 1) * (h as usize + 1)];
    let src_raw = src.as_raw();

    for y in 0..h as usize {
        let mut row_sum = 0u32;
        for x in 0..w as usize {
            row_sum += src_raw[y * w as usize + x] as u32;
            let idx = (y + 1) * (w as usize + 1) + (x + 1);
            integral[idx] = integral[idx - (w as usize + 1)] + row_sum;
        }
    }

    Tensor::from_vec(
        integral,
        cv_core::TensorShape::new(1, h as usize + 1, w as usize + 1),
    )
    .map_err(|_| Error::MemoryError("Failed to create integral image tensor".to_string()))
}

/// Compute sum of pixels in axis-aligned rectangle using integral image
///
/// Efficiently computes rectangular region sum in O(1) time using
/// precomputed integral image with four lookups.
///
/// # Arguments
///
/// * `integral` - Precomputed integral image
/// * `x`, `y` - Top-left corner in original image coordinates
/// * `w`, `h` - Rectangle width and height
///
/// # Returns
///
/// Sum of all pixel values in the specified rectangle
fn get_rect_sum(integral: &Tensor<u32>, x: u32, y: u32, w: u32, h: u32) -> Result<u32> {
    let iw = integral.shape.width;
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x + w) as usize;
    let y1 = (y + h) as usize;

    let data = integral
        .as_slice()
        .map_err(|_| Error::MemoryError("Failed to get integral slice".to_string()))?;
    Ok(data[y1 * iw + x1] + data[y0 * iw + x0] - data[y1 * iw + x0] - data[y0 * iw + x1])
}
mod haar_test;
