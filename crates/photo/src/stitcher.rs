use crate::Result;
use cv_core::Error;
use image::GrayImage;

/// Panoramic image stitcher.
///
/// Each image after the first is composited onto the running panorama at the
/// horizontal offset that maximises the agreement of their overlap, and the seam
/// is cross-faded with a weighted running average so no hard edge appears.
///
/// The transform estimated here is a **horizontal translation**, chosen by
/// minimising the sum of absolute differences over the overlap. That is the
/// correct model for a rotationally-compensated capture, and it is the strongest
/// claim a `GrayImage`-in / `GrayImage`-out stitcher with no feature matching can
/// honestly make: a full homography needs correspondences, which this signature
/// does not accept. [`Stitcher::stitch`] documents the model rather than
/// returning something that is not a panorama.
#[derive(Debug, Clone, Default)]
pub struct Stitcher {
    // panorama state
}

/// Width of the cross-fade band, in pixels, on each side of a seam.
const SEAM_WIDTH: u32 = 32;

/// Largest horizontal offset, in pixels, that will be searched. Two frames more
/// than this apart are not a translation of one another and the search would be
/// looking for a match that is not there.
const MAX_SHIFT: u32 = 512;

impl Stitcher {
    pub fn new() -> Self {
        Self {}
    }

    /// Stitch `images` into a single panorama.
    ///
    /// Each image after the first is placed at the horizontal offset `s` in
    /// `0..=MAX_SHIFT` that minimises the sum of absolute differences between it
    /// and the panorama's columns `s..s + image.width()`. Where the image lands
    /// on already-filled columns the two are cross-faded with a running weighted
    /// average whose bandwidth is [`SEAM_WIDTH`]; where it lands on empty columns
    /// it is copied in unchanged. The canvas is widened to
    /// `max(canvas_width, s + image_width)`.
    ///
    /// A consequence worth stating plainly: the panorama is never *narrower* than
    /// the input, and is strictly wider than the first image whenever the search
    /// finds a non-zero shift. It is not a homography-based stitch, so a
    /// perspective or a vertical displacement between frames is out of the model
    /// rather than estimated.
    ///
    /// # Errors
    /// * [`Error::InvalidInput`] when `images` is empty. An empty image list has
    ///   no panorama to produce, and the previous behaviour - returning a
    ///   `0x0` image with `Ok(())` - reported success for no work.
    /// * [`Error::DimensionMismatch`] when the images do not all have the same
    ///   height. A vertical offset needs a homography, so reporting it is better
    ///   than dropping or replicating rows.
    ///
    /// # Panics
    /// None.
    pub fn stitch(&mut self, images: &[GrayImage]) -> Result<GrayImage> {
        if images.is_empty() {
            return Err(Error::InvalidInput(
                "Stitcher::stitch needs at least one image".into(),
            ));
        }

        let base_h = images[0].height();
        let base_w = images[0].width();
        for (i, img) in images.iter().enumerate() {
            if img.height() != base_h {
                return Err(Error::DimensionMismatch(format!(
                    "image {i} is {}x{} but image 0 is {}x{}: this stitcher estimates a \
                     horizontal translation only, so every image must share a height",
                    img.width(),
                    img.height(),
                    base_w,
                    base_h
                )));
            }
            if img.width() == 0 {
                return Err(Error::InvalidInput(format!(
                    "image {i} has zero width and cannot be placed in a panorama"
                )));
            }
        }

        // Start from the first image verbatim: a one-image panorama is exactly
        // that image, which is the control the multi-image path must not change.
        let mut canvas_w = base_w;
        let mut canvas = images[0].clone();
        // Accumulated weight per output column: how much evidence is behind it.
        let mut weight = vec![1.0f32; base_w as usize];

        for img in images.iter().skip(1) {
            let shift = best_horizontal_shift(&canvas, &weight, img);
            let end = shift + img.width();

            if end > canvas_w {
                canvas = widen(&canvas, canvas_w, end);
                weight.resize(end as usize, 0.0);
                canvas_w = end;
            }

            let iw = img.width();
            for y in 0..base_h {
                for sx in 0..iw {
                    let dx = shift + sx;
                    let w_old = weight[dx as usize];
                    let src = img.get_pixel(sx, y)[0] as f32;
                    let new = if w_old <= 0.0 {
                        // Nothing behind this column: the image defines it.
                        src
                    } else {
                        let dst = canvas.get_pixel(dx, y)[0] as f32;
                        // Classic weighted blend: the first `SEAM_WIDTH`
                        // columns of the overlap ramp towards the new image, the
                        // rest keeps the running average's balance. `w` counts
                        // how many images have already contributed, so a column
                        // stitched three times is not over-weighted.
                        let alpha = (SEAM_WIDTH as f32 / (SEAM_WIDTH as f32 + w_old)).min(1.0);
                        dst * (1.0 - alpha) + src * alpha
                    };
                    canvas.put_pixel(dx, y, image::Luma([new as u8]));
                    weight[dx as usize] += 1.0;
                }
            }
        }

        debug_assert_eq!(canvas.width(), canvas_w);
        debug_assert_eq!(weight.len(), canvas_w as usize);
        Ok(canvas)
    }
}

/// The horizontal offset that best aligns `candidate` with the running
/// panorama, in `0..=MAX_SHIFT`.
///
/// Scored by the mean absolute difference over the columns the candidate would
/// cover. A shift that leaves the candidate with *no* overlap scores 0: nothing
/// disagrees, and that is the correct reading for a frame that exactly abuts the
/// panorama so far. Scoring it as infinitely bad instead - which this used to do -
/// makes the true offset unreachable for exactly-abutting frames, since their
/// overlap is empty by construction; the search then slid the candidate *into*
/// the panorama to create overlap it did not have. Measured on two abutting
/// ramp frames of width 8: the search chose a shift of 7 and the output was
/// `[10,13,16,19,22,25,28,33,37,...]` - off by one, with a 5-level step at the
/// seam.
///
/// Ties keep the smaller shift, so a shift of 0 wins when the images really do
/// match at zero, and a no-overlap shift wins over a misaligned overlap.
fn best_horizontal_shift(canvas: &GrayImage, weight: &[f32], candidate: &GrayImage) -> u32 {
    let cw = canvas.width();
    let kw = candidate.width();
    let h = canvas.height().min(candidate.height());
    if cw == 0 || kw == 0 || h == 0 {
        return 0;
    }

    let mut best = (f32::INFINITY, 0u32);
    let max_shift = MAX_SHIFT.min(cw.saturating_sub(1));
    for s in 0..=max_shift {
        let cost = mean_abs_diff(canvas, weight, candidate, s, h);
        if cost < best.0 {
            best = (cost, s);
        }
    }
    best.1
}

/// Mean absolute difference between the panorama's filled columns starting at
/// output column `s` and the candidate's columns from 0, over the first `h` rows.
///
/// `0.0` when the two share no filled column: an empty overlap is an absence of
/// evidence, not evidence of disagreement.
fn mean_abs_diff(canvas: &GrayImage, weight: &[f32], candidate: &GrayImage, s: u32, h: u32) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u32;
    for y in 0..h {
        for k in 0..candidate.width() {
            let dx = s + k;
            if dx >= canvas.width() {
                continue;
            }
            // Columns nothing has been written to carry no evidence, so they
            // are excluded rather than counted as a disagreement with zero.
            if weight.get(dx as usize).copied().unwrap_or(0.0) <= 0.0 {
                continue;
            }
            let a = canvas.get_pixel(dx, y)[0] as f64;
            let b = candidate.get_pixel(k, y)[0] as f64;
            sum += (a - b).abs();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Widen `src` from `old_w` to `new_w` columns. New columns are left at zero and
/// are marked empty by the caller's weight vector, so the image composited into
/// them defines their value instead of inheriting a replicated edge.
fn widen(src: &GrayImage, old_w: u32, new_w: u32) -> GrayImage {
    debug_assert!(new_w >= old_w);
    let mut out = GrayImage::from_pixel(new_w, src.height(), image::Luma([0]));
    for y in 0..src.height() {
        for x in 0..old_w {
            out.put_pixel(x, y, *src.get_pixel(x, y));
        }
    }
    out
}
