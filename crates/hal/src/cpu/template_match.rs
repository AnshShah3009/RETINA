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
//! enough that `CcoeffNormed` exceeded 1.0 at an exact match; the normalised
//! correlations are now also clamped to `[-1, 1]`.

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

/// Score map of size `(ih - th + 1) x (iw - tw + 1)`, row-major.
///
/// `eps` is the caller's "zero denominator" threshold (the element type's
/// `EPSILON`): a normalised score whose denominator is not above it is 0.
#[allow(clippy::too_many_arguments)]
pub fn match_template_f64(
    img: &[f64],
    iw: usize,
    ih: usize,
    tmpl: &[f64],
    tw: usize,
    th: usize,
    method: TemplateMatchMethod,
    eps: f64,
) -> Vec<f64> {
    assert!(tw <= iw && th <= ih);
    let (ow, oh) = (iw - tw + 1, ih - th + 1);
    let n = (tw * th) as f64;

    let t_sum: f64 = tmpl.iter().sum();
    let t_sq_sum: f64 = tmpl.iter().map(|v| v * v).sum();
    let t_mean = if tw * th > 0 { t_sum / n } else { 0.0 };
    let t_centered: Vec<f64> = tmpl.iter().map(|v| v - t_mean).collect();
    let t_norm = t_centered.iter().map(|v| v * v).sum::<f64>().sqrt();

    // Integral images of I and I^2, (ih + 1) x (iw + 1), only when a method
    // needs window sums.
    let needs_window_sums = matches!(
        method,
        TemplateMatchMethod::SqDiffNormed
            | TemplateMatchMethod::CcorrNormed
            | TemplateMatchMethod::CcoeffNormed
    );
    let stride = iw + 1;
    let (integral, integral_sq) = if needs_window_sums {
        let mut s = vec![0.0f64; (ih + 1) * stride];
        let mut s2 = vec![0.0f64; (ih + 1) * stride];
        for y in 0..ih {
            let (mut row, mut row2) = (0.0, 0.0);
            for x in 0..iw {
                let v = img[y * iw + x];
                row += v;
                row2 += v * v;
                s[(y + 1) * stride + x + 1] = s[y * stride + x + 1] + row;
                s2[(y + 1) * stride + x + 1] = s2[y * stride + x + 1] + row2;
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

    let corr_template: &[f64] = match method {
        TemplateMatchMethod::Ccoeff | TemplateMatchMethod::CcoeffNormed => &t_centered,
        _ => tmpl,
    };

    let mut out = vec![0.0f64; ow * oh];
    out.par_chunks_mut(ow).enumerate().for_each(|(y, row_out)| {
        for (x, o) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for ty in 0..th {
                let irow = &img[(y + ty) * iw + x..(y + ty) * iw + x + tw];
                let trow = &corr_template[ty * tw..(ty + 1) * tw];
                acc += match method {
                    TemplateMatchMethod::SqDiff | TemplateMatchMethod::SqDiffNormed => {
                        sq_dist(irow, trow)
                    }
                    _ => dot(irow, trow),
                };
            }
            *o = match method {
                TemplateMatchMethod::SqDiff
                | TemplateMatchMethod::Ccorr
                | TemplateMatchMethod::Ccoeff => acc,
                TemplateMatchMethod::SqDiffNormed | TemplateMatchMethod::CcorrNormed => {
                    let i_sq = window(&integral_sq, x, y).max(0.0);
                    let denom = (i_sq * t_sq_sum).sqrt();
                    if denom > eps {
                        let r = acc / denom;
                        if method == TemplateMatchMethod::CcorrNormed {
                            r.clamp(-1.0, 1.0)
                        } else {
                            r
                        }
                    } else {
                        0.0
                    }
                }
                TemplateMatchMethod::CcoeffNormed => {
                    let s = window(&integral, x, y);
                    let var = (window(&integral_sq, x, y) - s * s / n).max(0.0);
                    let denom = var.sqrt() * t_norm;
                    if denom > eps {
                        (acc / denom).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    }
                }
            };
        }
    });
    out
}
