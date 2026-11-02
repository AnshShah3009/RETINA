//! Row-parallel Triangle-filter resize for the ORB pyramid.
//!
//! [`resize_triangle`] produces exactly the bytes of
//! `image::imageops::resize(image, w, h, FilterType::Triangle)` for a
//! [`GrayImage`]. That function is serial and generic over four-channel
//! pixels: on a 1080p frame the seven ORB pyramid levels took ~200 ms of a
//! ~245 ms `Orb::detect`, because each level is resized straight from the
//! full-resolution base and the Triangle support widens with the downscale
//! ratio. The arithmetic here is the same, step for step, in `f32`:
//!
//! * the per-output tap window and its normalised weights are computed with
//!   the same expressions (`(o + 0.5) * ratio`, the `floor`/`ceil` bounds and
//!   their clamps, `1 - |x|` weights summed in tap order, then each divided by
//!   that sum);
//! * the vertical pass accumulates `0.0 + p0 * w0 + p1 * w1 + ...` for every
//!   pixel in tap order into an `f32` intermediate, as `vertical_sample` does;
//! * the horizontal pass accumulates the intermediate the same way, clamps to
//!   `[0, 255]` with the same comparisons and rounds half away from zero, as
//!   `horizontal_sample` and its `FloatNearest` conversion do.
//!
//! Only the loop order changes (rows in parallel, and the vertical pass walks a
//! row of output pixels per tap), which never changes any single pixel's
//! sequence of floating-point operations, so every output byte is identical.

use image::GrayImage;
use rayon::prelude::*;

/// One output coordinate's source window: taps `left..left + weights.len()`.
struct Taps {
    left: usize,
    weights: Vec<f32>,
}

/// The `image` crate's triangle kernel.
#[inline]
fn triangle_kernel(x: f32) -> f32 {
    if x.abs() < 1.0 { 1.0 - x.abs() } else { 0.0 }
}

/// Tap windows and normalised weights for resampling `src` samples to `dst`,
/// as `image`'s `vertical_sample` / `horizontal_sample` compute them for the
/// Triangle filter (support 1.0).
fn taps(src: u32, dst: u32) -> Vec<Taps> {
    let ratio = src as f32 / dst as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let src_support = 1.0f32 * sratio;
    (0..dst)
        .map(|o| {
            let input = (o as f32 + 0.5) * ratio;
            let left = ((input - src_support).floor() as i64).clamp(0, src as i64 - 1);
            let right = ((input + src_support).ceil() as i64).clamp(left + 1, src as i64);
            let input = input - 0.5;
            let mut weights = Vec::with_capacity((right - left) as usize);
            let mut sum = 0.0f32;
            for i in left..right {
                let w = triangle_kernel((i as f32 - input) / sratio);
                weights.push(w);
                sum += w;
            }
            for w in weights.iter_mut() {
                *w /= sum;
            }
            Taps {
                left: left as usize,
                weights,
            }
        })
        .collect()
}

/// `image::imageops::resize(image, nwidth, nheight, FilterType::Triangle)`,
/// byte for byte, with the rows of both passes computed in parallel.
pub(crate) fn resize_triangle(image: &GrayImage, nwidth: u32, nheight: u32) -> GrayImage {
    let (width, height) = image.dimensions();
    // `image` returns a zeroed buffer for an empty source, a copy for an
    // unchanged size, and (via its two passes) a zeroed buffer of the new size
    // when either new dimension is zero.
    if width == 0 || height == 0 || nwidth == 0 || nheight == 0 {
        return GrayImage::new(nwidth, nheight);
    }
    if (nwidth, nheight) == (width, height) {
        return image.clone();
    }

    let (w, nw) = (width as usize, nwidth as usize);
    let src = image.as_raw();

    let rows = taps(height, nheight);
    let mut tmp = vec![0.0f32; w * nheight as usize];
    tmp.par_chunks_mut(w)
        .zip(rows.par_iter())
        .for_each(|(out, tap)| {
            for (i, &wt) in tap.weights.iter().enumerate() {
                let row = &src[(tap.left + i) * w..][..w];
                for (o, &p) in out.iter_mut().zip(row) {
                    *o += p as f32 * wt;
                }
            }
        });

    let cols = taps(width, nwidth);
    let mut out = vec![0u8; nw * nheight as usize];
    out.par_chunks_mut(nw)
        .zip(tmp.par_chunks(w))
        .for_each(|(dst, row)| {
            for (d, tap) in dst.iter_mut().zip(&cols) {
                let mut t = 0.0f32;
                for (&p, &wt) in row[tap.left..][..tap.weights.len()]
                    .iter()
                    .zip(&tap.weights)
                {
                    t += p * wt;
                }
                let t = if t < 0.0 {
                    0.0
                } else if t > 255.0 {
                    255.0
                } else {
                    t
                };
                *d = t.round() as u8;
            }
        });

    GrayImage::from_raw(nwidth, nheight, out).expect("buffer matches the output size")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use image::imageops::{FilterType, resize};

    fn noise(w: u32, h: u32, seed: u64) -> GrayImage {
        let mut state = seed | 1;
        GrayImage::from_fn(w, h, |x, y| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Mix smooth structure with noise so both flat and busy regions
            // (and the 0 / 255 extremes) are exercised.
            let base = ((x * 7 + y * 3) % 256) as u64;
            Luma([if state % 5 == 0 {
                (state >> 8) as u8
            } else {
                base as u8
            }])
        })
    }

    #[test]
    fn matches_image_resize_triangle_byte_for_byte() {
        let mut checked = 0;
        for (i, &(w, h)) in [
            (1, 1),
            (1, 7),
            (9, 1),
            (17, 13),
            (64, 48),
            (161, 97),
            (640, 480),
        ]
        .iter()
        .enumerate()
        {
            let img = noise(w, h, 0x9e37_79b9 + i as u64);
            let mut sizes = vec![(w, h), (w * 2, h + 3), (1, 1), (0, h), (w, 0)];
            // The ORB pyramid's sizes for scale factors 1.2, 1.5 and 2.0.
            for factor in [1.2f32, 1.5, 2.0] {
                let mut s = 1.0f32;
                for _ in 0..8 {
                    sizes.push(((w as f32 / s) as u32, (h as f32 / s) as u32));
                    s *= factor;
                }
            }
            sizes.push(((w / 3).max(1), (h * 2 / 3).max(1)));
            for (nw, nh) in sizes {
                let expect = resize(&img, nw, nh, FilterType::Triangle);
                let got = resize_triangle(&img, nw, nh);
                assert_eq!(
                    got.dimensions(),
                    expect.dimensions(),
                    "{w}x{h} -> {nw}x{nh}"
                );
                assert!(
                    got.as_raw() == expect.as_raw(),
                    "{w}x{h} -> {nw}x{nh} differs from image::imageops::resize"
                );
                checked += 1;
            }
        }
        let empty = GrayImage::new(0, 0);
        assert_eq!(
            resize_triangle(&empty, 5, 4).as_raw(),
            resize(&empty, 5, 4, FilterType::Triangle).as_raw()
        );
        assert!(checked > 100);
    }
}
