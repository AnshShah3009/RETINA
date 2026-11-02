use image::GrayImage;
use nalgebra::{Matrix2, Point2, SymmetricEigen, Vector2};
use rayon::prelude::*;

use crate::Result;

/// Detect chessboard corners in an image.
///
/// This function locates the corners of a chessboard pattern in the given image.
/// It uses Harris corner detection followed by grid assignment to find the exact positions
/// of the chessboard corners.
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `pattern_size` - Expected pattern size as (cols, rows)
///
/// # Returns
///
/// A vector of detected corner positions, or an error if the pattern cannot be found.
pub fn find_chessboard_corners(
    image: &GrayImage,
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let need = cols * rows;
    if cols < 2 || rows < 2 {
        return Err(cv_core::Error::AlgorithmError(
            "pattern_size must be at least (2,2)".to_string(),
        ));
    }
    if image.width() < 8 || image.height() < 8 {
        return Err(cv_core::Error::AlgorithmError(
            "image too small for chessboard detection".to_string(),
        ));
    }

    let (response, width, height) = harris_response(image, 0.04, 1);
    let max_r = response
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    if max_r <= 0.0 {
        return Err(cv_core::Error::AlgorithmError(
            "no chessboard-like corners found".to_string(),
        ));
    }
    let threshold = max_r * 0.01;
    let mut cands = non_max_suppression_response(&response, width, height, threshold);
    if cands.len() < need {
        return Err(cv_core::Error::AlgorithmError(format!(
            "insufficient corner candidates: found {}, need {need}",
            cands.len()
        )));
    }

    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate((need * 10).max(need));
    let mut ordered = assign_grid_points(&cands, pattern_size)?;
    corner_subpix(image, &mut ordered, 3, 25, 1e-3)?;
    Ok(ordered)
}

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

/// Compute Harris corner response for corner detection.
///
/// This function calculates the Harris corner response at each pixel in the image.
/// The Harris response is a measure of how likely a pixel is to be a corner.
fn harris_response(image: &GrayImage, k: f64, win_radius: usize) -> (Vec<f64>, usize, usize) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut ix = vec![0.0f64; width * height];
    let mut iy = vec![0.0f64; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = image.get_pixel((x + 1) as u32, y as u32)[0] as f64
                - image.get_pixel((x - 1) as u32, y as u32)[0] as f64;
            let gy = image.get_pixel(x as u32, (y + 1) as u32)[0] as f64
                - image.get_pixel(x as u32, (y - 1) as u32)[0] as f64;
            ix[y * width + x] = gx * 0.5;
            iy[y * width + x] = gy * 0.5;
        }
    }

    let mut resp = vec![0.0f64; width * height];
    let r = win_radius as i32;
    for y in win_radius..(height - win_radius) {
        for x in win_radius..(width - win_radius) {
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            let mut syy = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let xx = (x as i32 + dx) as usize;
                    let yy = (y as i32 + dy) as usize;
                    let gx = ix[yy * width + xx];
                    let gy = iy[yy * width + xx];
                    sxx += gx * gx;
                    sxy += gx * gy;
                    syy += gy * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            resp[y * width + x] = det - k * trace * trace;
        }
    }
    (resp, width, height)
}

/// Perform non-maximum suppression on the corner response.
///
/// This function suppresses non-maximal responses, keeping only the locally maximum
/// corner responses above the given threshold.
fn non_max_suppression_response(
    response: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let r = response[y * width + x];
            if r <= threshold {
                continue;
            }
            let mut is_max = true;
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    if (xx != x || yy != y) && response[yy * width + xx] > r {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            if is_max {
                out.push((x as f64, y as f64, r));
            }
        }
    }
    out
}

/// Assign detected corner candidates to a regular grid.
///
/// This function takes a set of corner candidates and assigns them to the expected
/// positions on the chessboard grid, handling the 2D spatial organization of the pattern.
fn assign_grid_points(
    candidates: &[(f64, f64, f64)],
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let points: Vec<Vector2<f64>> = candidates
        .iter()
        .map(|(x, y, _)| Vector2::new(*x, *y))
        .collect();
    if points.len() < cols * rows {
        return Err(cv_core::Error::AlgorithmError(
            "not enough candidates to assign grid".to_string(),
        ));
    }

    let mean = points.iter().fold(Vector2::zeros(), |acc, p| acc + p) / points.len() as f64;
    let mut cov = Matrix2::<f64>::zeros();
    for p in &points {
        let d = p - mean;
        cov += d * d.transpose();
    }
    cov /= points.len() as f64;
    let eig = SymmetricEigen::new(cov);
    let (i0, i1) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
    let e0 = eig.eigenvectors.column(i0).into_owned();
    let e1 = eig.eigenvectors.column(i1).into_owned();

    let mut uv = Vec::with_capacity(points.len());
    for p in &points {
        let d = p - mean;
        uv.push((d.dot(&e0), d.dot(&e1)));
    }
    let u_vals: Vec<f64> = uv.iter().map(|(u, _)| *u).collect();
    let v_vals: Vec<f64> = uv.iter().map(|(_, v)| *v).collect();
    let mut u_centers = kmeans_1d(&u_vals, cols, 30);
    let mut v_centers = kmeans_1d(&v_vals, rows, 30);
    u_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = vec![false; points.len()];
    let mut out = Vec::with_capacity(cols * rows);
    for vc in &v_centers {
        for uc in &u_centers {
            let mut best = None;
            let mut best_cost = f64::INFINITY;
            for (i, (u, v)) in uv.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let du = u - uc;
                let dv = v - vc;
                let cost = du * du + dv * dv;
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(i);
                }
            }
            let idx = best.ok_or_else(|| {
                cv_core::Error::AlgorithmError(
                    "failed to assign all chessboard corners".to_string(),
                )
            })?;
            used[idx] = true;
            out.push(Point2::new(points[idx][0], points[idx][1]));
        }
    }

    // Consistency gate: the assignment is only trustworthy when cluster
    // centers form a roughly uniform lattice. A previous revision returned
    // scrambled assignments (median corner error ~70 px on synthetic boards)
    // when Harris+NMS candidates included board-silhouette responses; down-
    // stream calibration then consumed garbage correspondences. Fail loudly
    // instead. The 2x bound tolerates moderate perspective foreshortening.
    let mut spacings_u: Vec<f64> = u_centers
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|s| *s > 1e-9)
        .collect();
    let mut spacings_v: Vec<f64> = v_centers
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|s| *s > 1e-9)
        .collect();
    spacings_u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    spacings_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let spacing = [
        spacings_u.first().copied().unwrap_or(0.0),
        spacings_u.last().copied().unwrap_or(0.0),
        spacings_v.first().copied().unwrap_or(0.0),
        spacings_v.last().copied().unwrap_or(0.0),
    ];
    // Non-degenerate grid: distinct clusters along both axes.
    if spacing[1] <= 1e-9 || spacing[3] <= 1e-9 {
        return Err(cv_core::Error::AlgorithmError(
            "chessboard grid collapsed: corners do not form a regular lattice".to_string(),
        ));
    }
    // Spacing uniformity within 35% across each axis (real boards are
    // regular; junk candidate sets produce wildly uneven k-means centers).
    let ru = if spacing[0] > 1e-9 {
        spacing[1] / spacing[0]
    } else {
        f64::INFINITY
    };
    let rv = if spacing[2] > 1e-9 {
        spacing[3] / spacing[2]
    } else {
        f64::INFINITY
    };
    if !(0.5..=2.0).contains(&ru) || !(0.5..=2.0).contains(&rv) {
        return Err(cv_core::Error::AlgorithmError(format!(
            "chessboard corner assignment failed spacing-uniformity check \
             (u ratio {:.2}, v ratio {:.2}); input likely contains no usable board",
            ru, rv
        )));
    }

    Ok(out)
}

/// 1D k-means clustering for grid coordinate assignment.
///
/// This function clusters 1D values into k clusters, useful for identifying
/// the regular grid structure along one dimension.
fn kmeans_1d(values: &[f64], k: usize, iters: usize) -> Vec<f64> {
    let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if k == 1 || (max_v - min_v).abs() < 1e-12 {
        return vec![0.5 * (min_v + max_v); k];
    }

    let mut centers = (0..k)
        .map(|i| min_v + (i as f64) * (max_v - min_v) / (k as f64 - 1.0))
        .collect::<Vec<_>>();

    for _ in 0..iters {
        let mut sums = vec![0.0f64; k];
        let mut cnts = vec![0usize; k];
        for &v in values {
            let mut bi = 0usize;
            let mut bd = (v - centers[0]).abs();
            for (i, &c) in centers.iter().enumerate().skip(1) {
                let d = (v - c).abs();
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            sums[bi] += v;
            cnts[bi] += 1;
        }
        for i in 0..k {
            if cnts[i] > 0 {
                centers[i] = sums[i] / cnts[i] as f64;
            }
        }
    }
    centers
}
