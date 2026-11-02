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
///
/// The position is computed in `T` when `T` is at least as wide as f32, and
/// in f32 for the 16-bit types, whose own arithmetic cannot hold it:
/// `(x + 0.5) * src` overflows f16 above 65504 (so a 512-wide source resized
/// to 256 sampled its last column from output 128 on), and bf16 cannot count
/// past 256. Only the weight `d` is then rounded to `T`; `1 - d` is computed
/// in `T`, as the interpolation is.
pub(crate) fn bilinear_taps<T: Float>(src: usize, dst: usize) -> Vec<Tap<T>> {
    let tap = |(i0, i1, d): (usize, usize, T)| Tap {
        i0,
        i1,
        w0: T::ONE - d,
        w1: d,
    };
    if std::mem::size_of::<T>() < std::mem::size_of::<f32>() {
        positions::<f32>(src, dst)
            .into_iter()
            .map(|(i0, i1, d)| tap((i0, i1, T::from_f32(d))))
            .collect()
    } else {
        positions::<T>(src, dst).into_iter().map(tap).collect()
    }
}

/// `(i0, i1, d)` per output index, computed in `C`.
fn positions<C: Float>(src: usize, dst: usize) -> Vec<(usize, usize, C)> {
    let src_f = C::from_f32(src as f32);
    let dst_f = C::from_f32(dst.max(1) as f32);
    let last = src.saturating_sub(1);
    (0..dst)
        .map(|x| {
            let v = (C::from_f32(x as f32) + C::from_f32(0.5)) * src_f / dst_f - C::from_f32(0.5);
            let f = if v < C::ZERO { C::ZERO } else { v };
            let i0 = (f.to_f32() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            let d = if i0 == i1 {
                C::ZERO
            } else {
                let d = f - C::from_f32(i0 as f32);
                if d < C::ZERO {
                    C::ZERO
                } else if d > C::ONE {
                    C::ONE
                } else {
                    d
                }
            };
            (i0, i1, d)
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
