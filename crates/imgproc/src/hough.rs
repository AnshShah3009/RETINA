use crate::canny;
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::TensorToGpu;
use cv_runtime::orchestrator::{scheduler, ResourceGroup};
use image::GrayImage;
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

    // A 3x3 Sobel window needs at least one interior row/column. Without this
    // guard `1..(height - 1)`/`1..(width - 1)` underflow on zero-sized images
    // (debug panic, wrap-then-OOB in release).
    if width < 3 || height < 3 {
        return (gx, gy);
    }

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

/// Gradient Hough circle transform on a **grayscale** image.
///
/// Like `cv2.HoughCircles(.., HOUGH_GRADIENT, ..)`, and unlike
/// [`hough_lines`] / [`hough_lines_p`], this takes the intensity image and
/// runs [`canny`] internally (low 50, high 150); the Sobel gradient direction
/// of the intensity image steers the votes. The parameters and the voting
/// scheme are this crate's own (one accumulator per integer radius,
/// `threshold` votes, centres closer than 10 px suppressed), not OpenCV's.
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

/// [`hough_circles`] with an explicit resource group. Takes a grayscale image
/// and runs [`canny`] internally.
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

/// Validate the accumulator resolutions of the Hough line transforms.
///
/// `rho_res` and `theta_res` are caller-supplied `f32`s that used to be used
/// unchecked:
/// * `theta_res == 0.0` made `num_theta = (PI / 0.0) as usize == usize::MAX`, and
///   the subsequent `num_rho * num_theta` allocation overflowed (panic).
/// * `rho_res == 0.0` made `num_rho` enormous in one path and, in the other,
///   produced `Ok(vec![])` - no lines and no error - where the same image with
///   `rho_res = 1.0` produced 5 lines.
/// * Non-finite values (`NaN`, `inf`) saturate the same casts to nonsense
///   accumulator sizes.
///
/// # Returns
/// `Err` naming the offending resolution when it is not finite and positive.
pub fn validate_hough_resolutions(rho_res: f32, theta_res: f32) -> crate::Result<()> {
    if !(rho_res.is_finite() && rho_res > 0.0) {
        return Err(crate::ImgprocError::InvalidInput(format!(
            "hough: rho_res must be finite and > 0, got {}. \
             A non-positive rho_res yields an empty result with no error (or an overflowing \
             accumulator size)",
            rho_res
        )));
    }
    if !(theta_res.is_finite() && theta_res > 0.0) {
        return Err(crate::ImgprocError::InvalidInput(format!(
            "hough: theta_res must be finite and > 0, got {}. \
             theta_res == 0.0 makes num_theta == usize::MAX and overflows the accumulator \
             allocation",
            theta_res
        )));
    }
    Ok(())
}

/// `cvRound` on a float: round half to even, saturating.
#[inline]
fn cv_round(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// `cv::RNG` (multiply-with-carry), so the probabilistic transform visits
/// edge points in the same order as OpenCV.
struct CvRng(u64);

impl CvRng {
    fn next(&mut self) -> u32 {
        self.0 = (self.0 as u32 as u64)
            .wrapping_mul(4_164_903_690)
            .wrapping_add(self.0 >> 32);
        self.0 as u32
    }

    fn uniform(&mut self, a: i32, b: i32) -> i32 {
        if a == b {
            a
        } else {
            (self.next() % (b - a) as u32) as i32 + a
        }
    }
}

/// OpenCV `HoughLinesStandard` accumulator layout and trig tables.
struct StandardLayout {
    num_angle: usize,
    num_rho: usize,
    rho_offset: i32,
    /// `cos(ang) / rho`, `sin(ang) / rho` as OpenCV's `createTrigTable`
    /// computes them (angle accumulated in f32, product in f64).
    tab_cos: Vec<f32>,
    tab_sin: Vec<f32>,
}

impl StandardLayout {
    fn new(w: usize, h: usize, rho: f32, theta: f32) -> Self {
        let irho = 1.0f32 / rho;
        let mut num_angle = ((std::f64::consts::PI) / theta as f64).floor() as usize + 1;
        if num_angle > 1
            && (std::f64::consts::PI - ((num_angle - 1) as f32 * theta) as f64).abs()
                < (theta / 2.0) as f64
        {
            num_angle -= 1;
        }
        let num_rho = cv_round(((w + h) * 2 + 1) as f32 / rho).max(1) as usize;
        let rho_offset = ((num_rho - 1) / 2) as i32;

        // createTrigTable: the angle is accumulated in f32.
        let mut tab_cos = Vec::with_capacity(num_angle);
        let mut tab_sin = Vec::with_capacity(num_angle);
        let mut ang = 0.0f32;
        for _ in 0..num_angle {
            tab_sin.push(((ang as f64).sin() * irho as f64) as f32);
            tab_cos.push(((ang as f64).cos() * irho as f64) as f32);
            ang += theta;
        }
        Self {
            num_angle,
            num_rho,
            rho_offset,
            tab_cos,
            tab_sin,
        }
    }

    /// Accumulator index of edge pixel (x, y) at angle n, if in range.
    #[inline]
    fn index(&self, x: usize, y: usize, n: usize) -> Option<usize> {
        let r = cv_round(x as f32 * self.tab_cos[n] + y as f32 * self.tab_sin[n]) + self.rho_offset;
        (r >= 0 && (r as usize) < self.num_rho).then(|| n * self.num_rho + r as usize)
    }

    fn accumulate(&self, edges: &GrayImage) -> Vec<u32> {
        let w = edges.width() as usize;
        let h = edges.height() as usize;
        let src = edges.as_raw();
        let len = self.num_angle * self.num_rho;
        (0..h)
            .into_par_iter()
            .fold(
                || vec![0u32; len],
                |mut acc, y| {
                    for x in 0..w {
                        if src[y * w + x] == 0 {
                            continue;
                        }
                        for n in 0..self.num_angle {
                            if let Some(i) = self.index(x, y, n) {
                                acc[i] += 1;
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0u32; len],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    }

    /// findLocalMaximums + hough_cmp_gt ordering.
    fn lines(&self, acc: &[u32], rho: f32, theta: f32, threshold: u32) -> Vec<Line> {
        let (num_rho, num_angle) = (self.num_rho, self.num_angle);
        // `>` the lower and `>=` the upper neighbour in rho and theta;
        // out-of-range neighbours are 0. Scan order is rho-major as in
        // OpenCV, but ties are broken by accumulator index (theta-major).
        let at = |n: usize, r: usize| acc[n * num_rho + r];
        let mut peaks: Vec<(u32, usize, usize)> = Vec::new();
        for r in 0..num_rho {
            for n in 0..num_angle {
                let v = at(n, r);
                if v <= threshold {
                    continue;
                }
                let r_lo = if r > 0 { at(n, r - 1) } else { 0 };
                let r_hi = if r + 1 < num_rho { at(n, r + 1) } else { 0 };
                let n_lo = if n > 0 { at(n - 1, r) } else { 0 };
                let n_hi = if n + 1 < num_angle { at(n + 1, r) } else { 0 };
                if v > r_lo && v >= r_hi && v > n_lo && v >= n_hi {
                    peaks.push((v, n, r));
                }
            }
        }
        // hough_cmp_gt: more votes first, then smaller accumulator index.
        peaks.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        peaks
            .into_iter()
            .map(|(score, n, r)| Line {
                // Bin r holds rho = (r - offset) * rho_res. OpenCV 4.13 reports
                // that value also when num_rho is even (where the textbook
                // `(r - (num_rho - 1) * 0.5) * rho` would be half a bin off).
                rho: (r as i32 - self.rho_offset) as f32 * rho,
                theta: n as f32 * theta,
                score,
            })
            .collect()
    }
}

/// OpenCV `HoughLinesStandard` on a binary edge map (nonzero = edge).
fn hough_lines_cpu(edges: &GrayImage, rho: f32, theta: f32, threshold: u32) -> Vec<Line> {
    let w = edges.width() as usize;
    let h = edges.height() as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let layout = StandardLayout::new(w, h, rho, theta);
    let acc = layout.accumulate(edges);
    layout.lines(&acc, rho, theta, threshold)
}

/// Standard Hough line transform, `cv2.HoughLines(edges, rho_res, theta_res,
/// threshold)`.
///
/// `edges` is a binary edge map: every nonzero pixel votes. Run [`canny`] (or
/// any other edge detector) first; this function does not. (It used to run
/// `canny(src, 50, 150)` internally, which turned an edge map into the edges
/// of its edges.)
///
/// Accumulator, peak test and ordering follow OpenCV's `HoughLinesStandard`:
/// `round(((w + h) * 2 + 1) / rho_res)` rho bins, bin `k` centred on
/// `(k - (num_rho - 1) / 2) * rho_res`, `floor(pi / theta_res) + 1` angles minus a duplicate of 0, a
/// line is a bin with **more than** `threshold` votes that beats its rho and
/// theta neighbours, strongest first. `Line::score` is the vote count
/// (`cv2.HoughLinesWithAccumulator`).
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] if `rho_res` or `theta_res` is
/// not finite and positive.
pub fn hough_lines(
    edges: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
) -> crate::Result<Vec<Line>> {
    validate_hough_resolutions(rho_res, theta_res)?;
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            return hough_lines_ctx(edges, rho_res, theta_res, threshold, &group);
        }
    }
    Ok(hough_lines_cpu(edges, rho_res, theta_res, threshold))
}

/// [`hough_lines`] with an explicit resource group. On a GPU group the votes
/// are cast by the `hough.wgsl` kernel ([`hough_lines_gpu`]), which yields
/// the same accumulator - and therefore the same lines - as the CPU path; on
/// any GPU error the CPU path runs instead.
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] if `rho_res` or `theta_res` is
/// not finite and positive.
pub fn hough_lines_ctx(
    edges: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    group: &ResourceGroup,
) -> crate::Result<Vec<Line>> {
    validate_hough_resolutions(rho_res, theta_res)?;
    if edges.width() == 0 || edges.height() == 0 {
        return Ok(Vec::new());
    }
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(res) = hough_lines_gpu(gpu, edges, rho_res, theta_res, threshold) {
            return Ok(res);
        }
    }
    Ok(group.run(|| hough_lines_cpu(edges, rho_res, theta_res, threshold)))
}

/// [`hough_lines`] with the voting on `gpu`, without a CPU fallback.
///
/// The GPU casts every vote whose bin is unambiguous in f32; the few that lie
/// within rounding noise of a bin boundary are cast on the host with the CPU
/// formula, so the accumulator is bit-identical to the CPU one. Peak finding
/// and ordering are the CPU code. The result equals [`hough_lines`] on the
/// CPU exactly. Resolutions are not validated here.
pub fn hough_lines_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    edges: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
) -> cv_hal::Result<Vec<Line>> {
    let w = edges.width() as usize;
    let h = edges.height() as usize;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let layout = StandardLayout::new(w, h, rho_res, theta_res);
    let exact = |x: usize, y: usize, n: usize| layout.index(x, y, n);
    let acc = cv_hal::gpu_kernels::hough::hough_accumulate_u8(
        gpu,
        edges.as_raw(),
        w,
        h,
        &layout.tab_cos,
        &layout.tab_sin,
        layout.num_rho,
        layout.rho_offset,
        &exact,
    )?;
    Ok(layout.lines(&acc, rho_res, theta_res, threshold))
}

/// Progressive probabilistic Hough transform, `cv2.HoughLinesP(edges,
/// rho_res, theta_res, threshold, min_line_length, max_line_gap)`.
///
/// `edges` is a binary edge map: every nonzero pixel is a candidate. Run
/// [`canny`] first if needed; this function does not (it used to run
/// `canny(src, 50, 150)` internally).
///
/// A port of OpenCV's `HoughLinesProbabilistic`, including its `cv::RNG`
/// seeded with `u64::MAX`, so the segments and their order are the ones
/// OpenCV returns: points are drawn at random, a point whose best bin
/// reaches `threshold` votes seeds a walk along that line in both directions
/// (fixed-point, stopping after more than `max_line_gap` consecutive missing
/// pixels), the walked pixels are removed, and the segment is kept if its x or
/// y extent is at least `min_line_length`. `min_line_length` and
/// `max_line_gap` are rounded to integers as OpenCV does. Endpoints are
/// integer pixel coordinates.
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] if `rho_res` or `theta_res` is
/// not finite and positive.
pub fn hough_lines_p(
    edges: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    min_line_length: f32,
    max_line_gap: f32,
) -> crate::Result<Vec<LineSegment>> {
    validate_hough_resolutions(rho_res, theta_res)?;
    let width = edges.width() as usize;
    let height = edges.height() as usize;
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let threshold = threshold.min(i32::MAX as u32) as i32;
    let line_length = cv_round(min_line_length);
    let line_gap = cv_round(max_line_gap);

    let irho = 1.0f32 / rho_res;
    let num_angle = (std::f64::consts::PI / theta_res as f64).floor().max(1.0) as usize;
    let num_rho = cv_round(((width + height) * 2 + 1) as f32 / rho_res).max(1) as usize;
    let rho_offset = ((num_rho - 1) / 2) as i32;
    let mut trig = Vec::with_capacity(num_angle * 2);
    for n in 0..num_angle {
        let a = n as f64 * theta_res as f64;
        trig.push((a.cos() * irho as f64) as f32);
        trig.push((a.sin() * irho as f64) as f32);
    }
    let rbin = |x: i32, y: i32, n: usize| -> usize {
        let r = cv_round(x as f32 * trig[2 * n] + y as f32 * trig[2 * n + 1]) + rho_offset;
        r.clamp(0, num_rho as i32 - 1) as usize
    };

    let mut accum = vec![0i32; num_angle * num_rho];
    let mut mask = vec![0u8; width * height];
    let mut nzloc: Vec<(i32, i32)> = Vec::new();
    for (i, &v) in edges.as_raw().iter().enumerate() {
        if v != 0 {
            mask[i] = 1;
            nzloc.push(((i % width) as i32, (i / width) as i32));
        }
    }

    const SHIFT: i32 = 16;
    let mut rng = CvRng(u64::MAX);
    let mut lines = Vec::new();
    let mut count = nzloc.len() as i32;
    while count > 0 {
        let idx = rng.uniform(0, count) as usize;
        let (j, i) = nzloc[idx];
        nzloc[idx] = nzloc[count as usize - 1];
        count -= 1;

        if mask[i as usize * width + j as usize] == 0 {
            continue;
        }

        let mut max_val = threshold - 1;
        let mut max_n = 0usize;
        for n in 0..num_angle {
            let cell = &mut accum[n * num_rho + rbin(j, i, n)];
            *cell += 1;
            if max_val < *cell {
                max_val = *cell;
                max_n = n;
            }
        }
        if max_val < threshold {
            continue;
        }

        let a = -trig[2 * max_n + 1];
        let b = trig[2 * max_n];
        let (mut x0, mut y0) = (j, i);
        let (xflag, dx0, dy0);
        if a.abs() > b.abs() {
            xflag = true;
            dx0 = if a > 0.0 { 1 } else { -1 };
            dy0 = cv_round(b * (1 << SHIFT) as f32 / a.abs());
            y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
        } else {
            xflag = false;
            dy0 = if b > 0.0 { 1 } else { -1 };
            dx0 = cv_round(a * (1 << SHIFT) as f32 / b.abs());
            x0 = (x0 << SHIFT) + (1 << (SHIFT - 1));
        }
        let pixel = |x: i32, y: i32| -> (i32, i32) {
            if xflag {
                (x, y >> SHIFT)
            } else {
                (x >> SHIFT, y)
            }
        };

        let mut line_end = [(j, i); 2];
        for (k, end) in line_end.iter_mut().enumerate() {
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };
            let (mut x, mut y) = (x0, y0);
            let mut gap = 0;
            loop {
                let (j1, i1) = pixel(x, y);
                if j1 < 0 || j1 >= width as i32 || i1 < 0 || i1 >= height as i32 {
                    break;
                }
                if mask[i1 as usize * width + j1 as usize] != 0 {
                    gap = 0;
                    *end = (j1, i1);
                } else {
                    gap += 1;
                    if gap > line_gap {
                        break;
                    }
                }
                x += dx;
                y += dy;
            }
        }

        let good_line = (line_end[1].0 - line_end[0].0).abs() >= line_length
            || (line_end[1].1 - line_end[0].1).abs() >= line_length;

        for (k, end) in line_end.iter().enumerate() {
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };
            let (mut x, mut y) = (x0, y0);
            loop {
                let (j1, i1) = pixel(x, y);
                let m = i1 as usize * width + j1 as usize;
                if mask[m] != 0 {
                    if good_line {
                        for n in 0..num_angle {
                            accum[n * num_rho + rbin(j1, i1, n)] -= 1;
                        }
                    }
                    mask[m] = 0;
                }
                if (j1, i1) == *end {
                    break;
                }
                x += dx;
                y += dy;
            }
        }

        if good_line {
            lines.push(LineSegment {
                x1: line_end[0].0 as f32,
                y1: line_end[0].1 as f32,
                x2: line_end[1].0 as f32,
                y2: line_end[1].1 as f32,
            });
        }
    }

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn sobel_f32_handles_tiny_images() {
        // Regression: `1..(height - 1)` underflowed for zero-sized inputs.
        for (w, h) in [(0u32, 0u32), (1, 1), (2, 1), (1, 2), (2, 2)] {
            let img = GrayImage::new(w, h);
            let (gx, gy) = sobel_f32(&img);
            assert_eq!(gx.len(), (w * h) as usize);
            assert_eq!(gy.len(), (w * h) as usize);
            assert!(gx.iter().all(|&v| v == 0.0));
            assert!(gy.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_hough_lines_detects_horizontal() {
        // Draw a white horizontal line at row 32 in a 64x64 black image
        let mut img = GrayImage::new(64, 64);
        for x in 0..64 {
            img.put_pixel(x, 32, Luma([255]));
        }
        let lines =
            hough_lines(&img, 1.0, std::f32::consts::PI / 180.0, 10).expect("valid resolutions");
        assert!(!lines.is_empty(), "Hough should detect the horizontal line");
        // A horizontal line y=32 has theta near PI/2 (90 degrees)
        let half_pi = std::f32::consts::FRAC_PI_2;
        let has_horizontal = lines.iter().any(|l| (l.theta - half_pi).abs() < 0.15); // within ~8.6 degrees
        assert!(
            has_horizontal,
            "At least one detected line should have theta near PI/2 (horizontal). \
             Detected thetas: {:?}",
            lines.iter().map(|l| l.theta).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hough_circles_detects_circle() {
        // Draw a filled white circle (radius 20, center 50,50) in a 100x100
        // black image. The Canny detector inside hough_circles will find the
        // sharp boundary between the filled disk and the black background.
        let size = 100u32;
        let mut img = GrayImage::new(size, size);
        let cx = 50.0f32;
        let cy = 50.0f32;
        let radius = 20.0f32;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() <= radius {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }
        // Use a low threshold so that even moderate accumulator peaks are
        // picked up.  The radius search range brackets the true radius.
        let circles = hough_circles(&img, 15.0, 25.0, 8);
        assert!(!circles.is_empty(), "Hough should detect the drawn circle");
        let close_circle = circles.iter().any(|c| {
            let dx = c.cx - cx;
            let dy = c.cy - cy;
            (dx * dx + dy * dy).sqrt() < 5.0
        });
        assert!(
            close_circle,
            "Detected circle center should be within 5px of (50, 50). \
             Detected: {:?}",
            circles
                .iter()
                .map(|c| (c.cx, c.cy, c.r))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hough_lines_p_detects_segment() {
        // A white rectangle on black; hough_lines_p wants an edge map, so
        // run Canny first.
        let mut img = GrayImage::new(200, 200);
        for y in 80..120 {
            for x in 20..180 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let img = canny(&img, 50, 150);

        let segments = hough_lines_p(
            &img,
            1.0,                          // rho_res
            std::f32::consts::PI / 180.0, // theta_res (1 degree)
            3,                            // threshold
            10.0,                         // min_line_length
            20.0,                         // max_line_gap
        )
        .expect("valid resolutions");

        assert!(
            !segments.is_empty(),
            "PPHT should detect at least one segment on the rectangle edges"
        );

        // Verify the longest segment has reasonable length
        let longest = segments
            .iter()
            .max_by(|a, b| {
                let la = (a.x2 - a.x1).hypot(a.y2 - a.y1);
                let lb = (b.x2 - b.x1).hypot(b.y2 - b.y1);
                la.partial_cmp(&lb).unwrap()
            })
            .unwrap();
        let seg_len = (longest.x2 - longest.x1).hypot(longest.y2 - longest.y1);
        assert!(
            seg_len >= 10.0,
            "Longest segment length {:.1} should be >= 10px",
            seg_len
        );
    }
}
