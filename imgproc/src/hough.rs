use crate::{canny, sobel};
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
    let (gx, gy) = sobel(src);
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
                let dx = gx.as_raw()[i] as f32 - 128.0;
                let dy = gy.as_raw()[i] as f32 - 128.0;
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
    let (gx, gy) = sobel(src);

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

                    let dx = gx.as_raw()[i] as f32 - 128.0;
                    let dy = gy.as_raw()[i] as f32 - 128.0;
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
    let num_rho = (max_rho / rho_res) as usize * 2;
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

    let mut lines = Vec::new();
    for (i, atomic_count) in acc_atomic.into_iter().enumerate() {
        let r_idx = i / num_theta;
        let t_idx = i % num_theta;
        let score = atomic_count.into_inner();
        if score >= threshold {
            lines.push(Line {
                rho: r_idx as f32 * rho_res - max_rho,
                theta: t_idx as f32 * theta_res,
                score,
            });
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

/// Progressive Probabilistic Hough Transform for line segment detection
pub fn hough_lines_p(
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    min_line_length: f32,
    max_line_gap: f32,
) -> Vec<LineSegment> {
    let edges = canny(src, 50, 150);
    let width = src.width();
    let height = src.height();

    let mut edge_points = Vec::new();
    let edge_data = edges.as_raw();
    for i in 0..edge_data.len() {
        if edge_data[i] > 0 {
            edge_points.push((i % width as usize, i / width as usize));
        }
    }

    if edge_points.is_empty() {
        return Vec::new();
    }

    let max_rho = (width as f32 * width as f32 + height as f32 * height as f32).sqrt();
    let num_rho = (max_rho / rho_res) as usize * 2;
    let num_theta = (std::f32::consts::PI / theta_res) as usize;

    let mut acc = vec![0u32; num_rho * num_theta];
    let mut line_segments = Vec::new();
    let mut rng = rand::rng();

    let mut processed_edges = vec![false; edge_data.len()];

    while !edge_points.is_empty() {
        // 1. Pick a random edge point
        let idx = rng.random_range(0..edge_points.len());
        let (x, y) = edge_points.swap_remove(idx);

        if processed_edges[y * width as usize + x] {
            continue;
        }

        // 2. Voting
        let mut best_score = 0;
        let mut best_rho_idx = 0;
        let mut best_t_idx = 0;

        for t_idx in 0..num_theta {
            let theta = t_idx as f32 * theta_res;
            let rho = x as f32 * theta.cos() + y as f32 * theta.sin();
            let rho_idx = ((rho + max_rho) / rho_res) as usize;

            if rho_idx < num_rho {
                let aidx = rho_idx * num_theta + t_idx;
                acc[aidx] += 1;
                if acc[aidx] > best_score {
                    best_score = acc[aidx];
                    best_rho_idx = rho_idx;
                    best_t_idx = t_idx;
                }
            }
        }

        // 3. Check if we found a line
        if best_score >= threshold {
            let theta = best_t_idx as f32 * theta_res;
            let _rho = best_rho_idx as f32 * rho_res - max_rho;

            // 4. Trace the line to find the segment
            let dx = -theta.sin();
            let dy = theta.cos();

            let mut p1 = (x as f32, y as f32);
            let mut p2 = (x as f32, y as f32);

            // Search in both directions
            for dir in [-1.0, 1.0] {
                let mut last_valid = (x as f32, y as f32);
                let mut gap = 0.0;
                let mut dist = 1.0;

                while gap <= max_line_gap {
                    let cur_x = (x as f32 + dir * dist * dx).round() as i32;
                    let cur_y = (y as f32 + dir * dist * dy).round() as i32;

                    if cur_x >= 0 && cur_x < width as i32 && cur_y >= 0 && cur_y < height as i32 {
                        let pix_idx = cur_y as usize * width as usize + cur_x as usize;
                        if edge_data[pix_idx] > 0 {
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

                // 5. Remove processed points from accumulator and edge list
                // (Simplified: just mark them as processed)
                // In real PPHT, we'd iterate along the segment and decrement votes
                processed_edges[y * width as usize + x] = true;
            }
        }
    }

    line_segments
}
