//! Bilinear resize of a planar CHW tensor.
//!
//! A multichannel `Tensor` is planar: `TensorShape::new(c, h, w)` stores
//! element `(ch, y, x)` at `ch * h * w + y * w + x` (`Tensor::index`,
//! `Tensor::from_image_rgb`, and every other multichannel kernel in the
//! backend). Resize treats each plane as an independent image.
//!
//! The sampling positions depend only on the axis lengths, so they are worked
//! out once per output column and once per output row ([`bilinear_taps`]).

use cv_core::Float;

/// The two source indices and weights one output index on one axis samples.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tap<T> {
    pub i0: usize,
    pub i1: usize,
    /// `1 - d`, rounded to `T`.
    pub w0: T,
    /// `d`, the fractional distance from `i0` towards `i1`.
    pub w1: T,
}

/// Half-pixel taps for resampling an axis of `src` samples to `dst`.
///
/// OpenCV `resize` semantics: output index `x` samples
/// `(x + 0.5) * src / dst - 0.5`. Positions below 0 clamp to the edge; the
/// upper-bound index clamp is in integer space so rounding cannot index out of
/// bounds. Empty when `dst == 0`; callers must not ask for taps when
/// `src == 0` and `dst > 0` (there is nothing to sample).
pub(crate) fn bilinear_taps<T: Float>(src: usize, dst: usize) -> Vec<Tap<T>> {
    let src_f = T::from_f32(src as f32);
    let dst_f = T::from_f32(dst.max(1) as f32);
    let last = src.saturating_sub(1);
    (0..dst)
        .map(|x| {
            let v = (T::from_f32(x as f32) + T::from_f32(0.5)) * src_f / dst_f - T::from_f32(0.5);
            let f = if v < T::ZERO { T::ZERO } else { v };
            let i0 = (f.to_f32() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            let d = if i0 == i1 {
                T::ZERO
            } else {
                let d = f - T::from_f32(i0 as f32);
                if d < T::ZERO {
                    T::ZERO
                } else if d > T::ONE {
                    T::ONE
                } else {
                    d
                }
            };
            Tap {
                i0,
                i1,
                w0: T::ONE - d,
                w1: d,
            }
        })
        .collect()
}

/// Bilinear resize of the planar CHW image `src` (`c` planes of `h x w`) into
/// `dst` (`c` planes of `nh x nw`), computed in `T`.
///
/// Each output is `(v00 * w0x + v10 * w1x) * w0y + (v01 * w0x + v11 * w1x) * w1y`,
/// evaluated left to right in `T`. An empty source with a non-empty
/// destination leaves `dst` untouched (the caller zero-fills it).
pub(crate) fn resize_bilinear_planar<T: Float>(
    src: &[T],
    dst: &mut [T],
    c: usize,
    (h, w): (usize, usize),
    (nh, nw): (usize, usize),
) {
    use rayon::prelude::*;

    if c == 0 || nh == 0 || nw == 0 || h == 0 || w == 0 {
        return;
    }
    let xs = bilinear_taps::<T>(w, nw);
    let ys = bilinear_taps::<T>(h, nh);

    // One chunk per output row of one plane: row `r` is row `r % nh` of plane
    // `r / nh`.
    dst[..c * nh * nw]
        .par_chunks_mut(nw)
        .enumerate()
        .for_each(|(r, row_out)| {
            let (ch, y) = (r / nh, r % nh);
            let plane = &src[ch * h * w..(ch + 1) * h * w];
            let ty = ys[y];
            let row0 = &plane[ty.i0 * w..(ty.i0 + 1) * w];
            let row1 = &plane[ty.i1 * w..(ty.i1 + 1) * w];
            for (out, tx) in row_out.iter_mut().zip(&xs) {
                let v0 = row0[tx.i0] * tx.w0 + row0[tx.i1] * tx.w1;
                let v1 = row1[tx.i0] * tx.w0 + row1[tx.i1] * tx.w1;
                *out = v0 * ty.w0 + v1 * ty.w1;
            }
        });
}
