//! Sub-pixel corner refinement: OpenCV's `cornerSubPix`.
//!
//! OpenCV keeps `cornerSubPix` in imgproc; `cv-calib3d` re-exports these
//! items for the chessboard and calibration code that uses them.

use image::GrayImage;
use nalgebra::Point2;
use rayon::prelude::*;

use crate::Result;

/// Stopping rule of [`corner_subpix_with`] (OpenCV's `TermCriteria` with
/// `MAX_ITER` and/or `EPS`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubpixCriteria {
    /// Iteration limit (`MAX_ITER`), clamped to `1..=100`; `None` means
    /// OpenCV's cap of 100.
    pub max_iters: Option<usize>,
    /// Stop once an iteration moves the corner by at most this many pixels
    /// (`EPS`); `None` means never (only the iteration limit applies).
    pub epsilon: Option<f64>,
}

/// Refine corner positions to sub-pixel accuracy: OpenCV's
/// `cornerSubPix(image, corners, Size(win_radius, win_radius), Size(-1, -1),
/// TermCriteria(MAX_ITER | EPS, max_iters, eps))`.
///
/// Each corner `q` is moved to the point that minimises
/// `sum w(p) * (grad I(p) . (q - p))^2` over a `(2 * win_radius + 1)^2`
/// window centred on the current estimate, where the weight `w` is a
/// Gaussian over the window (OpenCV's mask). Pixel centres are at integer
/// coordinates. See [`corner_subpix_with`] for the details and the full set
/// of OpenCV's parameters.
///
/// # Errors
///
/// `win_radius` must be at least 1.
pub fn corner_subpix(
    image: &GrayImage,
    corners: &mut [Point2<f64>],
    win_radius: usize,
    max_iters: usize,
    eps: f64,
) -> Result<()> {
    corner_subpix_with(
        image,
        corners,
        (win_radius, win_radius),
        None,
        SubpixCriteria {
            max_iters: Some(max_iters),
            epsilon: Some(eps),
        },
    )
}

/// OpenCV 4.13's `cornerSubPix(image, corners, win, zero_zone, criteria)`.
///
/// `win` is the half-size `(width, height)` of the search window and
/// `zero_zone` the half-size of a dead zone in its middle whose pixels get
/// no weight (`None` is OpenCV's `Size(-1, -1)`; a zone not smaller than the
/// window is ignored, as in OpenCV). The window is sampled bilinearly around
/// the current estimate (OpenCV's `getRectSubPix`, replicating the border),
/// gradients are central differences, and each pixel is weighted by
/// `exp(-(dx / win.0)^2) * exp(-(dy / win.1)^2)`.
///
/// Iteration stops after `criteria.max_iters`, when a step moves the corner
/// by at most `criteria.epsilon`, when the gradient matrix is singular, or
/// when the next estimate would leave the image (the last estimate inside
/// is kept). A corner that ends up more than the window half-size away from
/// its start (on either axis) is reset to its start, and so is a corner
/// that does not lie inside the image (which OpenCV rejects). Like OpenCV,
/// the computation is done in `f32`, so the refined coordinates are `f32`
/// values.
///
/// # Errors
///
/// Both window half-sizes must be at least 1.
pub fn corner_subpix_with(
    image: &GrayImage,
    corners: &mut [Point2<f64>],
    win: (usize, usize),
    zero_zone: Option<(usize, usize)>,
    criteria: SubpixCriteria,
) -> Result<()> {
    const MAX_ITERS: usize = 100;
    if win.0 == 0 || win.1 == 0 {
        return Err(cv_core::Error::AlgorithmError(
            "corner_subpix window half-size must be >= 1".to_string(),
        ));
    }
    let max_iters = criteria
        .max_iters
        .map_or(MAX_ITERS, |n| n.clamp(1, MAX_ITERS));
    let eps = criteria.epsilon.map_or(0.0, |e| e.max(0.0));
    let eps = eps * eps;
    let (win_w, win_h) = (2 * win.0 + 1, 2 * win.1 + 1);

    let mut mask = vec![0f32; win_w * win_h];
    for i in 0..win_h {
        let y = (i as f32 - win.1 as f32) / win.1 as f32;
        let vy = (-y * y).exp();
        for j in 0..win_w {
            let x = (j as f32 - win.0 as f32) / win.0 as f32;
            mask[i * win_w + j] = vy * (-x * x).exp();
        }
    }
    if let Some((zw, zh)) = zero_zone
        && zw * 2 + 1 < win_w
        && zh * 2 + 1 < win_h
    {
        for i in win.1 - zh..=win.1 + zh {
            for j in win.0 - zw..=win.0 + zw {
                mask[i * win_w + j] = 0.0;
            }
        }
    }

    let (cols, rows) = (image.width() as f32, image.height() as f32);
    let inside = |x: f32, y: f32| 0.0 <= x && x < cols && 0.0 <= y && y < rows;
    let src = image.as_raw();
    let size = (image.width() as usize, image.height() as usize);
    let (pw, ph) = (win_w + 2, win_h + 2);

    corners.par_iter_mut().for_each(|p| {
        let c_t = (p.x as f32, p.y as f32);
        if !inside(c_t.0, c_t.1) {
            return;
        }
        let mut c_i = c_t;
        let mut buf = vec![0f32; pw * ph];
        let mut iter = 0;
        loop {
            get_rect_sub_pix_u8_f32(src, size, &mut buf, (pw, ph), c_i);
            let (mut a, mut b, mut c, mut bb1, mut bb2) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..win_h {
                let py = i as f64 - win.1 as f64;
                let row = (i + 1) * pw + 1;
                for j in 0..win_w {
                    let m = mask[i * win_w + j] as f64;
                    let k = row + j;
                    let tgx = (buf[k + 1] - buf[k - 1]) as f64;
                    let tgy = (buf[k + pw] - buf[k - pw]) as f64;
                    let gxx = tgx * tgx * m;
                    let gxy = tgx * tgy * m;
                    let gyy = tgy * tgy * m;
                    let px = j as f64 - win.0 as f64;
                    a += gxx;
                    b += gxy;
                    c += gyy;
                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }
            let det = a * c - b * b;
            if det.abs() <= f64::EPSILON * f64::EPSILON {
                break;
            }
            let scale = 1.0 / det;
            let next = (
                (c_i.0 as f64 + c * scale * bb1 - b * scale * bb2) as f32,
                (c_i.1 as f64 - b * scale * bb1 + a * scale * bb2) as f32,
            );
            let (ex, ey) = (next.0 - c_i.0, next.1 - c_i.1);
            let err = (ex * ex + ey * ey) as f64;
            if !inside(next.0, next.1) {
                break;
            }
            c_i = next;
            iter += 1;
            if iter >= max_iters || err <= eps {
                break;
            }
        }
        if (c_i.0 - c_t.0).abs() > win.0 as f32 || (c_i.1 - c_t.1).abs() > win.1 as f32 {
            c_i = c_t;
        }
        p.x = c_i.0 as f64;
        p.y = c_i.1 as f64;
    });
    Ok(())
}

/// OpenCV's `getRectSubPix` from an 8-bit image into an `f32` patch of
/// `size` centred on `center` (non-IPP path, `getRectSubPix_8u32f`):
/// bilinear sampling, the border replicated.
fn get_rect_sub_pix_u8_f32(
    src: &[u8],
    (src_w, src_h): (usize, usize),
    dst: &mut [f32],
    (win_w, win_h): (usize, usize),
    center: (f32, f32),
) {
    let cx = center.0 - (win_w - 1) as f32 * 0.5;
    let cy = center.1 - (win_h - 1) as f32 * 0.5;
    let ipx = cx.floor() as i64;
    let ipy = cy.floor() as i64;
    let (sw, sh) = (src_w as i64, src_h as i64);
    let (ww, wh) = (win_w as i64, win_h as i64);
    let px = |x: usize, y: usize| src[y * src_w + x] as f32;

    if 0 <= ipx && ipx + ww < sw && 0 <= ipy && ipy + wh < sh {
        let mut a = cx - ipx as f32;
        let b = cy - ipy as f32;
        a = a.max(0.0001);
        let a12 = a * (1.0 - b);
        let a22 = a * b;
        let b1 = 1.0 - b;
        let b2 = b;
        let s = (1.0 - a as f64) / a as f64;
        let (x0, y0) = (ipx as usize, ipy as usize);
        for r in 0..win_h {
            let y = y0 + r;
            let mut prev = (1.0 - a) * (b1 * px(x0, y) + b2 * px(x0, y + 1));
            for j in 0..win_w {
                let t = a12 * px(x0 + j + 1, y) + a22 * px(x0 + j + 1, y + 1);
                dst[r * win_w + j] = prev + t;
                prev = (t as f64 * s) as f32;
            }
        }
        return;
    }

    // getRectSubPix_Cn_<uchar, float, float> with adjustRect.
    let a = cx - ipx as f32;
    let b = cy - ipy as f32;
    let a11 = (1.0 - a) * (1.0 - b);
    let a12 = a * (1.0 - b);
    let a21 = (1.0 - a) * b;
    let a22 = a * b;
    let b1 = 1.0 - b;
    let b2 = b;

    // adjustRect: `src` points at (ipx, ipy) clamped, offset by -rect.x;
    // rows/columns outside the image repeat the edge.
    let mut base_x = 0i64;
    let mut base_y = 0i64;
    let (rx, rw);
    if ipx >= 0 {
        base_x += ipx;
        rx = 0;
    } else {
        rx = (-ipx).min(ww);
    }
    if ipx < sw - ww {
        rw = ww;
    } else {
        let mut w = sw - ipx - 1;
        if w < 0 {
            base_x += w;
            w = 0;
        }
        rw = w;
    }
    let ry;
    let rh;
    if ipy >= 0 {
        base_y += ipy;
        ry = 0;
    } else {
        ry = -ipy;
    }
    if ipy < sh - wh {
        rh = wh;
    } else {
        let mut h = sh - ipy - 1;
        if h < 0 {
            base_y += h;
            h = 0;
        }
        rh = h;
    }
    base_x -= rx;
    // `src` row pointer as a row index; `src + x` reads column base_x + x.
    let at = |row: i64, x: i64| src[(row * sw + base_x + x) as usize] as f32;
    let mut row = base_y;
    for i in 0..wh {
        let row2 = if i < ry || i >= rh { row } else { row + 1 };
        let s0 = at(row, rx) * b1 + at(row2, rx) * b2;
        for j in 0..rx {
            dst[(i * ww + j) as usize] = s0;
        }
        let s0 = at(row, rw) * b1 + at(row2, rw) * b2;
        for j in rw..ww {
            dst[(i * ww + j) as usize] = s0;
        }
        for j in rx..rw {
            dst[(i * ww + j) as usize] =
                at(row, j) * a11 + at(row, j + 1) * a12 + at(row2, j) * a21 + at(row2, j + 1) * a22;
        }
        if i < rh {
            row = row2;
        }
    }
}
