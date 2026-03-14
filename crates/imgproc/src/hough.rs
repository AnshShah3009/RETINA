use crate::canny;
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::TensorToGpu;
use cv_runtime::orchestrator::{scheduler, ResourceGroup};
use image::GrayImage;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Hough Line result (rho, theta)
#[derive(Debug, Clone, Copy)]
pub struct Line {
    pub rho: f32,
    pub theta: f32,
    pub score: u32,
}

/// Hough Line Segment (x1, y1, x2, y2)
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Compute signed f32 Sobel gradients directly from pixel data.
/// Returns (gx, gy) as Vec<f32> with true signed values (not clamped to u8).
fn sobel_f32(src: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let raw = src.as_raw();
    let mut gx = vec![0.0f32; width * height];
    let mut gy = vec![0.0f32; width * height];

    // Sobel 3x3 kernels:
    // Gx: [-1 0 1; -2 0 2; -1 0 1]
    // Gy: [-1 -2 -1; 0 0 0; 1 2 1]
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let p00 = raw[(y - 1) * width + (x - 1)] as f32;
            let p01 = raw[(y - 1) * width + x] as f32;
            let p02 = raw[(y - 1) * width + (x + 1)] as f32;
            let p10 = raw[y * width + (x - 1)] as f32;
            let p12 = raw[y * width + (x + 1)] as f32;
            let p20 = raw[(y + 1) * width + (x - 1)] as f32;
            let p21 = raw[(y + 1) * width + x] as f32;
            let p22 = raw[(y + 1) * width + (x + 1)] as f32;

            gx[y * width + x] = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            gy[y * width + x] = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;
        }
    }

    (gx, gy)
}

/// Non-maximum suppression for circles: sort by score descending, suppress any
/// circle whose center is within `min_dist` of a previously-accepted circle.
fn nms_circles(circles: &mut Vec<cv_core::HoughCircle>, min_dist: f32) {
    circles.sort_by(|a, b| b.score.cmp(&a.score));
    let min_dist_sq = min_dist * min_dist;
    let mut accepted = Vec::with_capacity(circles.len());
    for c in circles.iter() {
        let dominated = accepted.iter().any(|a: &cv_core::HoughCircle| {
            let dx = c.cx - a.cx;
            let dy = c.cy - a.cy;
            dx * dx + dy * dy < min_dist_sq
        });
        if !dominated {
            accepted.push(*c);
        }
    }
    *circles = accepted;
}

pub fn hough_circles(
    src: &GrayImage,
    min_radius: f32,
    max_radius: f32,
    threshold: u32,
) -> Vec<cv_core::HoughCircle> {
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            return hough_circles_ctx(src, min_radius, max_radius, threshold, &group);
        }
    }
    // Fallback: dummy group or sequential
    let edges = canny(src, 50, 150);
    let (gx_f32, gy_f32) = sobel_f32(src);
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut all_circles = Vec::new();
    for r in (min_radius as i32)..=(max_radius as i32) {
        let r_f = r as f32;
        let mut acc = vec![0u32; width * height];
        for i in 0..edges.as_raw().len() {
            if edges.as_raw()[i] > 0 {
                let ex = (i % width) as f32;
                let ey = (i / width) as f32;
                let dx = gx_f32[i];
                let dy = gy_f32[i];
                let angle = dy.atan2(dx);
                for sign in [-1.0, 1.0] {
                    let cx = (ex + sign * r_f * angle.cos()) as i32;
                    let cy = (ey + sign * r_f * angle.sin()) as i32;
                    if cx >= 0 && cx < width as i32 && cy >= 0 && cy < height as i32 {
                        acc[cy as usize * width + cx as usize] += 1;
                    }
                }
            }
        }
        for (i, &count) in acc.iter().enumerate() {
            if count >= threshold {
                all_circles.push(cv_core::HoughCircle::new(
                    (i % width) as f32,
                    (i / width) as f32,
                    r_f,
                    count,
                ));
            }
        }
    }
    nms_circles(&mut all_circles, 10.0);
    all_circles
}

/// Optimized Hough Circle Transform using gradient direction
pub fn hough_circles_ctx(
    src: &GrayImage,
    min_radius: f32,
    max_radius: f32,
    threshold: u32,
    group: &ResourceGroup,
) -> Vec<cv_core::HoughCircle> {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(res) = hough_circles_gpu(gpu, src, min_radius, max_radius, threshold) {
            return res;
        }
    }

    let edges = canny(src, 50, 150);
    let (gx_f32, gy_f32) = sobel_f32(src);

    let width = src.width() as usize;
    let height = src.height() as usize;

    let mut all_circles = Vec::new();

    for r in (min_radius as i32)..=(max_radius as i32) {
        let r_f = r as f32;
        let acc_atomic: Vec<AtomicU32> = std::iter::repeat_with(|| AtomicU32::new(0))
            .take(width * height)
            .collect();

        group.run(|| {
            edges.as_raw().par_iter().enumerate().for_each(|(i, &e)| {
                if e > 0 {
                    let ex = (i % width) as i32;
                    let ey = (i / width) as i32;

                    let dx = gx_f32[i];
                    let dy = gy_f32[i];
                    let angle = dy.atan2(dx);

                    for sign in [-1.0, 1.0] {
                        let cx = ex as f32 + sign * r_f * angle.cos();
                        let cy = ey as f32 + sign * r_f * angle.sin();

                        if cx >= 0.0 && cx < width as f32 && cy >= 0.0 && cy < height as f32 {
                            let idx = (cy as usize) * width + (cx as usize);
                            acc_atomic[idx].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        });

        for (i, atomic_count) in acc_atomic.into_iter().enumerate() {
            let count = atomic_count.into_inner();
            if count >= threshold {
                all_circles.push(cv_core::HoughCircle::new(
                    (i % width) as f32,
                    (i / width) as f32,
                    r_f,
                    count,
                ));
            }
        }
    }

    nms_circles(&mut all_circles, 10.0);
    all_circles
}

fn hough_circles_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    min_radius: f32,
    max_radius: f32,
    threshold: u32,
) -> cv_hal::Result<Vec<cv_core::HoughCircle>> {
    use cv_hal::context::ComputeContext;
    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor
        .to_gpu_ctx(gpu)
        .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    gpu.hough_circles(&input_gpu, min_radius, max_radius, threshold)
}

pub fn hough_lines(src: &GrayImage, rho_res: f32, theta_res: f32, threshold: u32) -> Vec<Line> {
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            return hough_lines_ctx(src, rho_res, theta_res, threshold, &group);
        }
    }
    // Deep fallback
    Vec::new()
}

pub fn hough_lines_ctx(
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    group: &ResourceGroup,
) -> Vec<Line> {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(res) = hough_lines_gpu(gpu, src, rho_res, theta_res, threshold) {
            return res;
        }
    }

    let edges = canny(src, 50, 150);
    let width = src.width();
    let height = src.height();

    let max_rho = (width as f32 * width as f32 + height as f32 * height as f32).sqrt();
    let num_rho = (2.0 * max_rho / rho_res).ceil() as usize + 1;
    let num_theta = (std::f32::consts::PI / theta_res) as usize;

    let acc_atomic: Vec<AtomicU32> = std::iter::repeat_with(|| AtomicU32::new(0))
        .take(num_rho * num_theta)
        .collect();

    group.run(|| {
        edges.as_raw().par_iter().enumerate().for_each(|(i, &e)| {
            if e > 0 {
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;

                for t_idx in 0..num_theta {
                    let theta = t_idx as f32 * theta_res;
                    let rho = x * theta.cos() + y * theta.sin();
                    let rho_idx = ((rho + max_rho) / rho_res) as usize;

                    if rho_idx < num_rho {
                        acc_atomic[rho_idx * num_theta + t_idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
    });

    // Collect accumulator into a plain Vec for NMS
    let acc: Vec<u32> = acc_atomic
        .into_iter()
        .map(|a| a.into_inner())
        .collect();

    // Local-maximum suppression: only keep cells that are strictly greater
    // than all 8 neighbors in a 3x3 window in (rho, theta) accumulator space.
    let mut lines = Vec::new();
    for r_idx in 0..num_rho {
        for t_idx in 0..num_theta {
            let score = acc[r_idx * num_theta + t_idx];
            if score < threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for dr in [-1i32, 0, 1] {
                for dt in [-1i32, 0, 1] {
                    if dr == 0 && dt == 0 {
                        continue;
                    }
                    let nr = r_idx as i32 + dr;
                    let nt = t_idx as i32 + dt;
                    if nr >= 0 && nr < num_rho as i32 && nt >= 0 && nt < num_theta as i32 {
                        if acc[nr as usize * num_theta + nt as usize] >= score {
                            is_max = false;
                            break 'nms;
                        }
                    }
                }
            }
            if is_max {
                lines.push(Line {
                    rho: r_idx as f32 * rho_res - max_rho,
                    theta: t_idx as f32 * theta_res,
                    score,
                });
            }
        }
    }

    lines
}

fn hough_lines_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
) -> cv_hal::Result<Vec<Line>> {
    use cv_hal::context::ComputeContext;
    let edges = canny(src, 50, 150);
    let edges_f32: Vec<f32> = edges.as_raw().iter().map(|&p| p as f32).collect();

    let input_tensor = cv_core::CpuTensor::from_vec(
        edges_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor
        .to_gpu_ctx(gpu)
        .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    // hough_lines threshold must be u32, but underlying GPU kernel takes f32 or u32 depending on signature
    // In context.rs: fn hough_lines(..., threshold: u32)
    let res_hal = gpu.hough_lines(&input_gpu, rho_res, theta_res, threshold)?;

    Ok(res_hal
        .into_iter()
        .map(|l| Line {
            rho: l.rho,
            theta: l.theta,
            score: l.score,
        })
        .collect())
}

/// Unvote a single pixel from the accumulator: for each theta bin, compute which
/// rho bin it would have voted for and decrement.
fn ppht_unvote(
    x: usize,
    y: usize,
    acc: &mut [u32],
    num_rho: usize,
    num_theta: usize,
    rho_res: f32,
    theta_res: f32,
    max_rho: f32,
) {
    for t_idx in 0..num_theta {
        let theta = t_idx as f32 * theta_res;
        let rho = x as f32 * theta.cos() + y as f32 * theta.sin();
        let rho_idx = ((rho + max_rho) / rho_res) as usize;
        if rho_idx < num_rho {
            let aidx = rho_idx * num_theta + t_idx;
            acc[aidx] = acc[aidx].saturating_sub(1);
        }
    }
}

/// Progressive Probabilistic Hough Transform for line segment detection
///
/// Implements the Matas et al. 2000 PPHT algorithm. The key invariant is that
/// the accumulator only contains votes from unprocessed edge points. When a
/// line is detected, all supporting pixels along the segment are removed and
/// their votes decremented.
#[allow(clippy::needless_range_loop)]
pub fn hough_lines_p(
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    min_line_length: f32,
    max_line_gap: f32,
) -> Vec<LineSegment> {
    let edges = canny(src, 50, 150);
    let width = src.width() as usize;
    let height = src.height() as usize;

    let mut edge_points = Vec::new();
    let edge_data = edges.as_raw();
    for i in 0..edge_data.len() {
        if edge_data[i] > 0 {
            edge_points.push((i % width, i / width));
        }
    }

    if edge_points.is_empty() {
        return Vec::new();
    }

    let max_rho = ((width * width + height * height) as f32).sqrt();
    let num_rho = (2.0 * max_rho / rho_res).ceil() as usize + 1;
    let num_theta = (std::f32::consts::PI / theta_res) as usize;

    let mut acc = vec![0u32; num_rho * num_theta];
    let mut line_segments = Vec::new();
    let mut rng = rand::rng();

    // Track which pixels are still active (not yet processed).
    let mut active = vec![false; width * height];
    for &(x, y) in &edge_points {
        active[y * width + x] = true;
    }

    while !edge_points.is_empty() {
        // 1. Pick a random edge point
        let idx = rng.random_range(0..edge_points.len());
        let (x, y) = edge_points.swap_remove(idx);

        // Skip if already processed
        if !active[y * width + x] {
            continue;
        }

        // 2. Vote this point into the accumulator, and check each cell immediately.
        let mut found_t_idx = 0;
        let mut found_line = false;

        for t_idx in 0..num_theta {
            let theta = t_idx as f32 * theta_res;
            let rho = x as f32 * theta.cos() + y as f32 * theta.sin();
            let rho_idx = ((rho + max_rho) / rho_res) as usize;

            if rho_idx < num_rho {
                let aidx = rho_idx * num_theta + t_idx;
                acc[aidx] += 1;
                // Check threshold immediately after incrementing
                if acc[aidx] >= threshold {
                    found_t_idx = t_idx;
                    found_line = true;
                    // Finish voting all theta bins before proceeding
                }
            }
        }

        // 3. If any bin crossed threshold, extract the line segment
        if found_line {
            let theta = found_t_idx as f32 * theta_res;

            // Direction along the line (perpendicular to rho direction)
            let line_dx = -theta.sin();
            let line_dy = theta.cos();

            let mut p1 = (x as f32, y as f32);
            let mut p2 = (x as f32, y as f32);

            // Trace in both directions to find endpoints
            for dir in [-1.0f32, 1.0] {
                let mut last_valid = (x as f32, y as f32);
                let mut gap = 0.0;
                let mut dist = 1.0;

                while gap <= max_line_gap {
                    let cur_x = (x as f32 + dir * dist * line_dx).round() as i32;
                    let cur_y = (y as f32 + dir * dist * line_dy).round() as i32;

                    if cur_x >= 0
                        && cur_x < width as i32
                        && cur_y >= 0
                        && cur_y < height as i32
                    {
                        let pix_idx = cur_y as usize * width + cur_x as usize;
                        if active[pix_idx] {
                            last_valid = (cur_x as f32, cur_y as f32);
                            gap = 0.0;
                        } else {
                            gap += 1.0;
                        }
                    } else {
                        break;
                    }
                    dist += 1.0;
                }

                if dir < 0.0 {
                    p1 = last_valid;
                } else {
                    p2 = last_valid;
                }
            }

            let len = ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt();

            if len >= min_line_length {
                line_segments.push(LineSegment {
                    x1: p1.0,
                    y1: p1.1,
                    x2: p2.0,
                    y2: p2.1,
                });
            }

            // Whether the segment met min_line_length or not, walk the full
            // segment between p1 and p2 and remove all active edge pixels on it.
            let seg_len = len.max(1.0);
            let steps = seg_len.ceil() as usize;
            let sx = (p2.0 - p1.0) / seg_len;
            let sy = (p2.1 - p1.1) / seg_len;

            for s in 0..=steps {
                let px = (p1.0 + s as f32 * sx).round() as i32;
                let py = (p1.1 + s as f32 * sy).round() as i32;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let pi = py as usize * width + px as usize;
                    if active[pi] {
                        active[pi] = false;
                        // Unvote this pixel's contributions from the accumulator
                        ppht_unvote(
                            px as usize, py as usize, &mut acc, num_rho, num_theta,
                            rho_res, theta_res, max_rho,
                        );
                    }
                }
            }
        } else {
            // No line found: unvote this seed point's contributions to keep
            // the accumulator reflecting only unprocessed edge points.
            active[y * width + x] = false;
            ppht_unvote(x, y, &mut acc, num_rho, num_theta, rho_res, theta_res, max_rho);
        }
    }

    line_segments
}
