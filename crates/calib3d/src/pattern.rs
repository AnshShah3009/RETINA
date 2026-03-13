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
        return Err(cv_core::Error::CalibrationError(
            "pattern_size must be at least (2,2)".to_string(),
        ));
    }
    if image.width() < 8 || image.height() < 8 {
        return Err(cv_core::Error::CalibrationError(
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
        return Err(cv_core::Error::CalibrationError(
            "no chessboard-like corners found".to_string(),
        ));
    }
    let threshold = max_r * 0.01;
    let mut cands = non_max_suppression_response(&response, width, height, threshold);
    if cands.len() < need {
        return Err(cv_core::Error::CalibrationError(format!(
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

/// Refine corner positions to sub-pixel accuracy.
///
/// This function improves the accuracy of corner positions by fitting a local intensity
/// distribution around each corner to sub-pixel precision using an iterative process.
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `corners` - Mutable slice of corner positions to be refined
/// * `win_radius` - Radius of the window around each corner to consider (must be >= 1)
/// * `max_iters` - Maximum number of iterations for refinement
/// * `eps` - Convergence threshold: refinement stops when the corner shift is less than this value
///
/// # Returns
///
/// Ok(()) if successful, or an error if parameters are invalid.
pub fn corner_subpix(
    image: &GrayImage,
    corners: &mut [Point2<f64>],
    win_radius: usize,
    max_iters: usize,
    eps: f64,
) -> Result<()> {
    if win_radius == 0 {
        return Err(cv_core::Error::CalibrationError(
            "win_radius must be >= 1".to_string(),
        ));
    }
    let w = image.width() as i32;
    let h = image.height() as i32;
    corners.par_iter_mut().for_each(|p| {
        let mut x = p.x;
        let mut y = p.y;
        for _ in 0..max_iters {
            let mut sw = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let cx = x.round() as i32;
            let cy = y.round() as i32;
            for dy in -(win_radius as i32)..=(win_radius as i32) {
                for dx in -(win_radius as i32)..=(win_radius as i32) {
                    let xx = cx + dx;
                    let yy = cy + dy;
                    if xx <= 0 || yy <= 0 || xx >= w - 1 || yy >= h - 1 {
                        continue;
                    }
                    let gx = (image.get_pixel((xx + 1) as u32, yy as u32)[0] as f64
                        - image.get_pixel((xx - 1) as u32, yy as u32)[0] as f64)
                        * 0.5;
                    let gy = (image.get_pixel(xx as u32, (yy + 1) as u32)[0] as f64
                        - image.get_pixel(xx as u32, (yy - 1) as u32)[0] as f64)
                        * 0.5;
                    let wgt = (gx * gx + gy * gy).sqrt();
                    if wgt <= 1e-9 {
                        continue;
                    }
                    sw += wgt;
                    sx += wgt * xx as f64;
                    sy += wgt * yy as f64;
                }
            }
            if sw <= 1e-9 {
                break;
            }
            let nx = sx / sw;
            let ny = sy / sw;
            let shift = ((nx - x) * (nx - x) + (ny - y) * (ny - y)).sqrt();
            x = nx;
            y = ny;
            if shift < eps {
                break;
            }
        }
        p.x = x.clamp(0.0, (image.width() - 1) as f64);
        p.y = y.clamp(0.0, (image.height() - 1) as f64);
    });
    Ok(())
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
        return Err(cv_core::Error::CalibrationError(
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
                cv_core::Error::CalibrationError(
                    "failed to assign all chessboard corners".to_string(),
                )
            })?;
            used[idx] = true;
            out.push(Point2::new(points[idx][0], points[idx][1]));
        }
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
