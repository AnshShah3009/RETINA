//! CPU template matching (`cv::matchTemplate` methods) in `f64`.
//!
//! Per output pixel one dot-product pass over the template window; the window
//! sums `sum(I)` and `sum(I^2)` that the normalised and coefficient methods
//! need come from `f64` integral images in O(1). Coefficient methods use the
//! identity `sum((I - mean_I)(T - mean_T)) = sum(I * (T - mean_T))` (the
//! centred template sums to zero), so they need no per-window mean pass.
//!
//! Everything accumulates in `f64`: with integer-valued pixels (the usual
//! u8-derived input) every sum is exact, and the normalised scores are then
//! correct to a few `f64` ulps. The previous `f32` accumulation drifted far
//! enough that `CcoeffNormed` exceeded 1.0 at an exact match.
//!
//! # Several channels
//!
//! Image and template are planar CHW with the same channel count. As in
//! `cv::matchTemplate`, the score map has one channel and every term is
//! summed over the channels: the correlation, the window's `sum(I^2)`, and
//! for the coefficient methods the per-channel centring (`T_c - mean(T_c)`,
//! and the window variance `sum_c (sum(I_c^2) - sum(I_c)^2 / n)`). Only plane
//! 0 used to be read, against a template mean taken over every plane.
//!
//! # Normalisation
//!
//! The normalised methods follow OpenCV's `common_matchTemplate`
//! (`templmatch.cpp`), degenerate cases included:
//!
//! * the denominator is `sqrt(diff2) * |T|`, where `diff2` is the window's
//!   `sum(I^2)` (less `sum(I_c)^2 / n` per channel for `CcoeffNormed`) and
//!   `|T|` the template's norm (centred per channel for `CcoeffNormed`); a
//!   `diff2` of at most `min(0.5, 10 * FLT_EPSILON * sum(I^2))` counts as 0;
//! * `|num| < den` gives `num / den`, `|num| < 1.125 * den` gives `+-1`, and
//!   anything else (a zero denominator included) 0, or 1 for `SqDiffNormed`;
//! * under `CcoeffNormed` a template whose channels are all constant scores
//!   1 in every window.
//!
//! The previous rule (`num / den` when `den > T::EPSILON`, else 0, clamped
//! to `[-1, 1]` for the correlations) gave 0 where OpenCV gives 1 for a
//! constant template or a black window, and let `SqDiffNormed` run far
//! above 1 (199 for a bright window and a dim template; OpenCV reports 1).

use crate::context::TemplateMatchMethod;
use rayon::prelude::*;

/// `sum(a[i] * b[i])` with four independent accumulators (vectorisable).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `sum((a[i] - b[i])^2)` with four independent accumulators.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..4 {
            let d = x[k] - y[k];
            acc[k] += d * d;
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        tail += d * d;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// The normalised methods.
pub fn is_normed(method: TemplateMatchMethod) -> bool {
    matches!(
        method,
        TemplateMatchMethod::SqDiffNormed
            | TemplateMatchMethod::CcorrNormed
            | TemplateMatchMethod::CcoeffNormed
    )
}

/// Template-side terms of `cv::matchTemplate`, shared by the CPU loop and
/// the GPU kernel (which uploads `corr_template` and `norm`).
pub struct TemplateTerms {
    /// The template to correlate with, every channel: as given for the
    /// squared-difference and correlation methods, centred per channel for
    /// the coefficient methods.
    pub corr_template: Vec<f64>,
    /// `|T|` of the normalised methods: `sqrt(sum(T^2))`, or for
    /// `CcoeffNormed` `sqrt(sum_c sum((T_c - mean(T_c))^2))`.
    pub norm: f64,
    /// `CcoeffNormed` of a template whose channels are all constant: OpenCV
    /// scores every window 1 (`templNorm < DBL_EPSILON`).
    pub constant_ccoeff_normed: bool,
}

impl TemplateTerms {
    /// `tmpl` holds `channels` planes of `plane > 0` values.
    pub fn new(tmpl: &[f64], channels: usize, plane: usize, method: TemplateMatchMethod) -> Self {
        let tmpl = &tmpl[..channels * plane];
        let n = plane as f64;
        let ccoeff = matches!(
            method,
            TemplateMatchMethod::Ccoeff | TemplateMatchMethod::CcoeffNormed
        );
        let mut corr_template = Vec::with_capacity(tmpl.len());
        // `n * sum_c var(T_c)`.
        let mut centred_sq = 0.0;
        for t in tmpl.chunks(plane) {
            let mean = t.iter().sum::<f64>() / n;
            centred_sq += t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
            if ccoeff {
                corr_template.extend(t.iter().map(|v| v - mean));
            } else {
                corr_template.extend_from_slice(t);
            }
        }
        let (norm, constant_ccoeff_normed) = if method == TemplateMatchMethod::CcoeffNormed {
            // OpenCV: `templNorm = sum_c sdv_c^2 < DBL_EPSILON`.
            (centred_sq.sqrt(), centred_sq / n < f64::EPSILON)
        } else {
            (tmpl.iter().map(|v| v * v).sum::<f64>().sqrt(), false)
        };
        Self {
            corr_template,
            norm,
            constant_ccoeff_normed,
        }
    }
}

/// OpenCV's normalised score from its numerator, the window's `sum(I^2)`
/// over every channel, `diff2` (the same, less `sum(I_c)^2 / n` per channel
/// for `CcoeffNormed`) and the template norm. See the module documentation;
/// `match_template.wgsl` repeats it in `f32`.
#[inline]
pub fn normalise(
    num: f64,
    wnd_sum2: f64,
    diff2: f64,
    t_norm: f64,
    method: TemplateMatchMethod,
) -> f64 {
    let diff2 = diff2.max(0.0);
    let den = if diff2 <= (10.0 * f32::EPSILON as f64 * wnd_sum2).min(0.5) {
        0.0
    } else {
        diff2.sqrt() * t_norm
    };
    if num.abs() < den {
        num / den
    } else if num.abs() < den * 1.125 {
        if num > 0.0 { 1.0 } else { -1.0 }
    } else if method == TemplateMatchMethod::SqDiffNormed {
        1.0
    } else {
        0.0
    }
}

/// Score map of size `(ih - th + 1) x (iw - tw + 1)`, row-major, of an image
/// and a template of `channels` planar channels each (`channels * ih * iw`
/// and `channels * th * tw` values).
///
/// An empty image or template (no taps, no channels, or slices shorter than
/// their sizes) scores no window: the map is empty. `cv::matchTemplate`
/// rejects both, and the backends return `InvalidInput` before getting here.
#[allow(clippy::too_many_arguments)]
pub fn match_template_f64(
    img: &[f64],
    iw: usize,
    ih: usize,
    tmpl: &[f64],
    tw: usize,
    th: usize,
    channels: usize,
    method: TemplateMatchMethod,
) -> Vec<f64> {
    assert!(tw <= iw && th <= ih);
    let (i_plane, t_plane) = (iw * ih, tw * th);
    if t_plane == 0
        || channels == 0
        || img.len() < channels * i_plane
        || tmpl.len() < channels * t_plane
    {
        return Vec::new();
    }
    let (ow, oh) = (iw - tw + 1, ih - th + 1);
    let n = t_plane as f64;
    let terms = TemplateTerms::new(tmpl, channels, t_plane, method);
    if terms.constant_ccoeff_normed {
        return vec![1.0; ow * oh];
    }

    // Integral images of I and I^2 per channel, (ih + 1) x (iw + 1) each,
    // only for the normalised methods.
    let normed = is_normed(method);
    let stride = iw + 1;
    let table = (ih + 1) * stride;
    let (integral, integral_sq) = if normed {
        let mut s = vec![0.0f64; channels * table];
        let mut s2 = vec![0.0f64; channels * table];
        for (c, (s, s2)) in s.chunks_mut(table).zip(s2.chunks_mut(table)).enumerate() {
            let src = &img[c * i_plane..(c + 1) * i_plane];
            for y in 0..ih {
                let (mut row, mut row2) = (0.0, 0.0);
                for x in 0..iw {
                    let v = src[y * iw + x];
                    row += v;
                    row2 += v * v;
                    s[(y + 1) * stride + x + 1] = s[y * stride + x + 1] + row;
                    s2[(y + 1) * stride + x + 1] = s2[y * stride + x + 1] + row2;
                }
            }
        }
        (s, s2)
    } else {
        (Vec::new(), Vec::new())
    };
    let window = |table: &[f64], x: usize, y: usize| -> f64 {
        table[(y + th) * stride + x + tw]
            - table[y * stride + x + tw]
            - table[(y + th) * stride + x]
            + table[y * stride + x]
    };

    let corr_template = &terms.corr_template;
    let mut out = vec![0.0f64; ow * oh];
    out.par_chunks_mut(ow).enumerate().for_each(|(y, row_out)| {
        for (x, o) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for c in 0..channels {
                let img = &img[c * i_plane..(c + 1) * i_plane];
                let tmpl = &corr_template[c * t_plane..(c + 1) * t_plane];
                for ty in 0..th {
                    let irow = &img[(y + ty) * iw + x..(y + ty) * iw + x + tw];
                    let trow = &tmpl[ty * tw..(ty + 1) * tw];
                    acc += match method {
                        TemplateMatchMethod::SqDiff | TemplateMatchMethod::SqDiffNormed => {
                            sq_dist(irow, trow)
                        }
                        _ => dot(irow, trow),
                    };
                }
            }
            *o = if normed {
                let (mut wnd_sum2, mut wnd_mean2) = (0.0, 0.0);
                for c in 0..channels {
                    wnd_sum2 += window(&integral_sq[c * table..(c + 1) * table], x, y);
                    if method == TemplateMatchMethod::CcoeffNormed {
                        let s = window(&integral[c * table..(c + 1) * table], x, y);
                        wnd_mean2 += s * s / n;
                    }
                }
                normalise(acc, wnd_sum2, wnd_sum2 - wnd_mean2, terms.norm, method)
            } else {
                acc
            };
        }
    });
    out
}
