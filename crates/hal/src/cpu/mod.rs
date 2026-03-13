use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, Result};
use cv_core::{storage::Storage, Float, Tensor, TensorShape};
use rayon::prelude::*;
use wide::*;

pub mod simd;

#[derive(Clone, Debug)]
pub struct CpuBackend {
    device_id: DeviceId,
    num_threads: usize,
    simd_available: bool,
}

impl CpuBackend {
    pub fn new() -> Option<Self> {
        let num_threads = std::env::var("RUSTCV_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(rayon::current_num_threads);

        Some(Self {
            device_id: DeviceId(0),
            num_threads,
            simd_available: true,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

#[allow(clippy::needless_range_loop)]
pub fn gaussian_kernel_1d<T: Float>(sigma: T, size: usize) -> Vec<T> {
    let mut kernel = vec![T::ZERO; size];
    let radius = size / 2;
    let mut sum = T::ZERO;
    let two = T::from_f32(2.0);
    for i in 0..size {
        let x = T::from_f32(i as f32 - radius as f32);
        kernel[i] = (-(x * x) / (two * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for i in 0..size {
        kernel[i] /= sum;
    }
    kernel
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn supports(&self, capability: Capability) -> bool {
        match capability {
            Capability::Compute => true,
            Capability::Simd => self.simd_available,
            Capability::TensorCore => false,
            Capability::RayTracing => false,
        }
    }

    fn queue(&self, _queue_type: QueueType) -> QueueId {
        QueueId(0)
    }

    fn preferred_queue(&self) -> QueueType {
        QueueType::Compute
    }
}

impl CpuBackend {
    #[allow(clippy::needless_range_loop)]
    fn convolve_separable<T: Float + bytemuck::Pod + std::fmt::Debug>(
        &self,
        src: &[T],
        dst: &mut [T],
        w: usize,
        h: usize,
        kx: &[T],
        ky: &[T],
    ) {
        let rx = kx.len() / 2;
        let ry = ky.len() / 2;

        let pool = cv_core::BufferPool::global();
        let required_bytes = w * h * std::mem::size_of::<T>();
        let mut intermediate_vec = pool.get(required_bytes);

        if intermediate_vec.capacity() < required_bytes {
            eprintln!(
                "Warning: Buffer pool returned insufficient buffer for separable convolution (capacity {} < required {})",
                intermediate_vec.capacity(), required_bytes
            );
            return;
        }

        intermediate_vec.resize(required_bytes, 0);
        let intermediate: &mut [T] =
            bytemuck::cast_slice_mut(&mut intermediate_vec[..required_bytes]);

        // Horizontal pass
        intermediate
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row_inter)| {
                let row_src = &src[y * w..(y + 1) * w];
                for x in 0..w {
                    let mut sum = T::ZERO;
                    for i in 0..kx.len() {
                        let sx = (x as isize + i as isize - rx as isize).clamp(0, w as isize - 1)
                            as usize;
                        sum += row_src[sx] * kx[i];
                    }
                    row_inter[x] = sum;
                }
            });

        // Vertical pass
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_dst)| {
            for x in 0..w {
                let mut sum = T::ZERO;
                for j in 0..ky.len() {
                    let sy =
                        (y as isize + j as isize - ry as isize).clamp(0, h as isize - 1) as usize;
                    sum += intermediate[sy * w + x] * ky[j];
                }
                row_dst[x] = sum;
            }
        });

        pool.return_buffer(intermediate_vec);
    }
}

impl ComputeContext for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    fn last_submission_index(&self) -> crate::SubmissionIndex {
        crate::SubmissionIndex(0)
    }

    #[allow(clippy::needless_range_loop)]
    fn convolve_2d<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        kernel: &Tensor<T, S>,
        _border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let k_data = kernel
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Kernel not on CPU".into()))?;

        let (h, w) = input.shape.hw();
        let (kh, kw) = kernel.shape.hw();

        let mut output_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        // Naive implementation for now, but parallelized
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let mut sum = T::ZERO;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sy = (y as isize + ky as isize - (kh / 2) as isize)
                            .clamp(0, h as isize - 1) as usize;
                        let sx = (x as isize + kx as isize - (kw / 2) as isize)
                            .clamp(0, w as isize - 1) as usize;
                        sum += src[sy * w + sx] * k_data[ky * kw + kx];
                    }
                }
                row[x] = sum;
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn sobel<
        T: Float + bytemuck::Pod + std::fmt::Debug + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>)> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();

        let mut gx_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let mut gy_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let gx_slice = gx_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Gx output not on CPU".into()))?;
        let gy_slice = gy_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Gy output not on CPU".into()))?;

        // Sobel is separable:
        // Gx = [1 2 1]^T * [-1 0 1]
        // Gy = [-1 0 1]^T * [1 2 1]
        let (kx_deriv, kx_smooth) = match ksize {
            3 => (
                vec![T::from_f32(-1.0), T::ZERO, T::ONE],
                vec![T::ONE, T::from_f32(2.0), T::ONE],
            ),
            5 => (
                vec![
                    T::from_f32(-1.0),
                    T::from_f32(-2.0),
                    T::ZERO,
                    T::from_f32(2.0),
                    T::ONE,
                ],
                vec![
                    T::ONE,
                    T::from_f32(4.0),
                    T::from_f32(6.0),
                    T::from_f32(4.0),
                    T::ONE,
                ],
            ),
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "Sobel ksize={} not supported on CPU",
                    ksize
                )))
            }
        };

        if dx > 0 {
            self.convolve_separable(src, gx_slice, w, h, &kx_deriv, &kx_smooth);
        }
        if dy > 0 {
            self.convolve_separable(src, gy_slice, w, h, &kx_smooth, &kx_deriv);
        }

        Ok((
            Tensor {
                storage: gx_storage,
                shape: input.shape,
                dtype: input.dtype,
                _phantom: std::marker::PhantomData,
            },
            Tensor {
                storage: gy_storage,
                shape: input.shape,
                dtype: input.dtype,
                _phantom: std::marker::PhantomData,
            },
        ))
    }

    fn canny<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        low_threshold: T,
        high_threshold: T,
    ) -> Result<Tensor<T, S>> {
        let (h, w) = input.shape.hw();
        if h < 3 || w < 3 {
            return Err(crate::Error::InvalidInput(
                "Canny requires image at least 3x3".into(),
            ));
        }

        // Step 1: Gaussian blur (sigma=1.4, ksize=5)
        let blurred = self.gaussian_blur(input, T::from_f32(1.4), 5)?;

        // Step 2: Sobel gradients
        let (gx_tensor, gy_tensor) = self.sobel(&blurred, 1, 1, 3)?;
        let gx = gx_tensor
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Gx not on CPU".into()))?;
        let gy = gy_tensor
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Gy not on CPU".into()))?;

        // Step 3: Compute magnitude and direction
        let len = h * w;
        let mut magnitude = vec![T::ZERO; len];
        let mut direction = vec![0u8; len]; // 0=horiz, 1=diag45, 2=vert, 3=diag135

        for i in 0..len {
            let mx = gx[i].to_f32();
            let my = gy[i].to_f32();
            magnitude[i] = T::from_f32((mx * mx + my * my).sqrt());

            // Quantize angle to 4 directions
            let angle = my.atan2(mx).to_degrees();
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };
            direction[i] = if !(22.5..157.5).contains(&angle) {
                0 // horizontal edge -> suppress vertically
            } else if angle < 67.5 {
                1 // 45-degree
            } else if angle < 112.5 {
                2 // vertical edge -> suppress horizontally
            } else {
                3 // 135-degree
            };
        }

        // Step 4: Non-maximum suppression
        let mut nms_out = vec![T::ZERO; len];
        nms_out
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let idx = y * w + x;
                    let mag = magnitude[idx];
                    let (n1, n2) = match direction[idx] {
                        0 => (magnitude[idx - 1], magnitude[idx + 1]), // horizontal
                        1 => (
                            magnitude[(y - 1) * w + x + 1],
                            magnitude[(y + 1) * w + x - 1],
                        ), // 45
                        2 => (magnitude[(y - 1) * w + x], magnitude[(y + 1) * w + x]), // vertical
                        _ => (
                            magnitude[(y - 1) * w + x - 1],
                            magnitude[(y + 1) * w + x + 1],
                        ), // 135
                    };
                    if mag >= n1 && mag >= n2 {
                        row_out[x] = mag;
                    }
                }
            });

        // Step 5: Hysteresis thresholding
        // Mark strong and weak edges
        let mut edge_map = vec![0u8; len]; // 0=none, 1=weak, 2=strong
        for i in 0..len {
            if nms_out[i] >= high_threshold {
                edge_map[i] = 2;
            } else if nms_out[i] >= low_threshold {
                edge_map[i] = 1;
            }
        }

        // Trace weak edges connected to strong edges using iterative flood fill
        let mut changed = true;
        while changed {
            changed = false;
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let idx = y * w + x;
                    if edge_map[idx] != 1 {
                        continue;
                    }
                    // Check 8-connected neighbors for strong edge
                    let has_strong = edge_map[(y - 1) * w + x - 1] == 2
                        || edge_map[(y - 1) * w + x] == 2
                        || edge_map[(y - 1) * w + x + 1] == 2
                        || edge_map[y * w + x - 1] == 2
                        || edge_map[y * w + x + 1] == 2
                        || edge_map[(y + 1) * w + x - 1] == 2
                        || edge_map[(y + 1) * w + x] == 2
                        || edge_map[(y + 1) * w + x + 1] == 2;
                    if has_strong {
                        edge_map[idx] = 2;
                        changed = true;
                    }
                }
            }
        }

        // Build output: strong edges = ONE, rest = ZERO
        let mut output_storage = S::new(len, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        for i in 0..len {
            if edge_map[i] == 2 {
                dst[i] = T::ONE;
            }
        }

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, h, w),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn hough_lines<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        rho: T,
        theta: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();

        let rho_f = rho.to_f32();
        let theta_f = theta.to_f32();

        // Accumulator dimensions
        let diag = ((w * w + h * h) as f32).sqrt();
        let num_rho = ((2.0 * diag) / rho_f).ceil() as usize + 1;
        let num_theta = (std::f32::consts::PI / theta_f).ceil() as usize;

        // Precompute sin/cos tables
        let cos_table: Vec<f32> = (0..num_theta).map(|t| (t as f32 * theta_f).cos()).collect();
        let sin_table: Vec<f32> = (0..num_theta).map(|t| (t as f32 * theta_f).sin()).collect();

        let rho_offset = diag; // rho can be negative, shift by diagonal

        // Accumulate votes (parallel per row, then merge)
        let row_accums: Vec<Vec<u32>> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut acc = vec![0u32; num_rho * num_theta];
                for x in 0..w {
                    if src[y * w + x] > T::ZERO {
                        let xf = x as f32;
                        let yf = y as f32;
                        for t in 0..num_theta {
                            let r = xf * cos_table[t] + yf * sin_table[t];
                            let r_idx = ((r + rho_offset) / rho_f).round() as usize;
                            if r_idx < num_rho {
                                acc[t * num_rho + r_idx] += 1;
                            }
                        }
                    }
                }
                acc
            })
            .collect();

        // Merge accumulators
        let mut accumulator = vec![0u32; num_rho * num_theta];
        for row_acc in &row_accums {
            for i in 0..accumulator.len() {
                accumulator[i] += row_acc[i];
            }
        }

        // Extract peaks above threshold
        let mut lines = Vec::new();
        for t in 0..num_theta {
            for r in 0..num_rho {
                let votes = accumulator[t * num_rho + r];
                if votes >= threshold {
                    let rho_val = r as f32 * rho_f - rho_offset;
                    let theta_val = t as f32 * theta_f;
                    lines.push(cv_core::HoughLine {
                        rho: rho_val,
                        theta: theta_val,
                        score: votes,
                    });
                }
            }
        }

        Ok(lines)
    }

    fn hough_circles<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        min_radius: T,
        max_radius: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();

        let min_r = min_radius.to_f32().max(1.0) as usize;
        let max_r = max_radius.to_f32() as usize;
        if min_r >= max_r {
            return Err(crate::Error::InvalidInput(
                "hough_circles: min_radius must be less than max_radius".into(),
            ));
        }

        // Step 1: Compute gradients for direction voting
        // We need Sobel, which requires bytemuck::Pod + Debug. Since we're in a generic
        // context, compute simple central differences directly.
        let mut gx = vec![0.0f32; h * w];
        let mut gy = vec![0.0f32; h * w];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                gx[y * w + x] = src[y * w + x + 1].to_f32() - src[y * w + x - 1].to_f32();
                gy[y * w + x] = src[(y + 1) * w + x].to_f32() - src[(y - 1) * w + x].to_f32();
            }
        }

        // Step 2: Accumulate center votes along gradient direction
        let mut center_acc = vec![0u32; h * w];

        // Collect edge pixels
        let edge_pixels: Vec<(usize, usize)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| src[y * w + x] > T::ZERO)
            .collect();

        for &(x, y) in &edge_pixels {
            let dx = gx[y * w + x];
            let dy = gy[y * w + x];
            let mag = (dx * dx + dy * dy).sqrt();
            if mag < 1e-6 {
                continue;
            }
            let nx = dx / mag;
            let ny = dy / mag;

            // Vote along positive and negative gradient direction
            for sign in &[1.0f32, -1.0f32] {
                for r in min_r..=max_r {
                    let cx = (x as f32 + sign * nx * r as f32).round() as i32;
                    let cy = (y as f32 + sign * ny * r as f32).round() as i32;
                    if cx >= 0 && cx < w as i32 && cy >= 0 && cy < h as i32 {
                        center_acc[cy as usize * w + cx as usize] += 1;
                    }
                }
            }
        }

        // Step 3: Find candidate centers above a center threshold
        // Use threshold/2 for centers (since final threshold applies to radius accumulation)
        let center_thresh = (threshold / 2).max(1);
        let mut candidates: Vec<(usize, usize, u32)> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let votes = center_acc[y * w + x];
                if votes >= center_thresh {
                    // Simple 3x3 NMS for centers
                    let mut is_max = true;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let ny = y as i32 + dy;
                            let nx = x as i32 + dx;
                            if ny >= 0
                                && ny < h as i32
                                && nx >= 0
                                && nx < w as i32
                                && center_acc[ny as usize * w + nx as usize] > votes
                            {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }
                    if is_max {
                        candidates.push((x, y, votes));
                    }
                }
            }
        }

        // Step 4: For each candidate center, accumulate radius votes
        let num_radii = max_r - min_r + 1;
        let circles: Vec<cv_core::HoughCircle> = candidates
            .par_iter()
            .filter_map(|&(cx, cy, _)| {
                let mut radius_acc = vec![0u32; num_radii];

                for &(ex, ey) in &edge_pixels {
                    let dx = ex as f32 - cx as f32;
                    let dy = ey as f32 - cy as f32;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let r_idx = dist.round() as usize;
                    if r_idx >= min_r && r_idx <= max_r {
                        radius_acc[r_idx - min_r] += 1;
                    }
                }

                // Find best radius
                let mut best_votes = 0u32;
                let mut best_r = 0usize;
                for (i, &v) in radius_acc.iter().enumerate() {
                    if v > best_votes {
                        best_votes = v;
                        best_r = i + min_r;
                    }
                }

                if best_votes >= threshold {
                    Some(cv_core::HoughCircle {
                        cx: cx as f32,
                        cy: cy as f32,
                        r: best_r as f32,
                        score: best_votes,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(circles)
    }

    fn match_template<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        template: &Tensor<T, S>,
        method: TemplateMatchMethod,
    ) -> Result<Tensor<T, OS>> {
        let img = image
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Image not on CPU".into()))?;
        let tmpl = template
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Template not on CPU".into()))?;

        let (img_h, img_w) = image.shape.hw();
        let (tmpl_h, tmpl_w) = template.shape.hw();

        if tmpl_h > img_h || tmpl_w > img_w {
            return Err(crate::Error::InvalidInput(
                "Template larger than image".into(),
            ));
        }

        let out_h = img_h - tmpl_h + 1;
        let out_w = img_w - tmpl_w + 1;

        let mut output_storage =
            OS::new(out_h * out_w, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        // Precompute template statistics for normalized / coefficient methods
        let mut tmpl_sum = T::ZERO;
        let mut tmpl_sq_sum = T::ZERO;
        let tmpl_len_f = T::from_f32((tmpl_h * tmpl_w) as f32);
        for ty in 0..tmpl_h {
            for tx in 0..tmpl_w {
                let tv = tmpl[ty * tmpl_w + tx];
                tmpl_sum += tv;
                tmpl_sq_sum += tv * tv;
            }
        }
        let tmpl_mean = tmpl_sum / tmpl_len_f;
        let tmpl_var = tmpl_sq_sum - tmpl_sum * tmpl_sum / tmpl_len_f;
        let tmpl_norm = tmpl_var.sqrt();

        let needs_mean = matches!(
            method,
            TemplateMatchMethod::Ccoeff | TemplateMatchMethod::CcoeffNormed
        );

        dst.par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(y, row_out)| {
                for x in 0..out_w {
                    let mut sum_sq_diff = T::ZERO;
                    let mut sum_ccorr = T::ZERO;
                    let mut patch_sum = T::ZERO;
                    let mut patch_sq_sum = T::ZERO;
                    let mut sum_ccoeff = T::ZERO;

                    // Compute patch mean first if needed for Ccoeff methods
                    let patch_mean = if needs_mean {
                        let mut s = T::ZERO;
                        for ty in 0..tmpl_h {
                            for tx in 0..tmpl_w {
                                s += img[(y + ty) * img_w + (x + tx)];
                            }
                        }
                        s / tmpl_len_f
                    } else {
                        T::ZERO
                    };

                    for ty in 0..tmpl_h {
                        for tx in 0..tmpl_w {
                            let iv = img[(y + ty) * img_w + (x + tx)];
                            let tv = tmpl[ty * tmpl_w + tx];
                            let diff = iv - tv;
                            sum_sq_diff += diff * diff;
                            sum_ccorr += iv * tv;
                            patch_sum += iv;
                            patch_sq_sum += iv * iv;
                            if needs_mean {
                                sum_ccoeff += (iv - patch_mean) * (tv - tmpl_mean);
                            }
                        }
                    }

                    row_out[x] = match method {
                        TemplateMatchMethod::SqDiff => sum_sq_diff,
                        TemplateMatchMethod::SqDiffNormed => {
                            let denom = (patch_sq_sum * tmpl_sq_sum).sqrt();
                            if denom > T::EPSILON {
                                sum_sq_diff / denom
                            } else {
                                T::ZERO
                            }
                        }
                        TemplateMatchMethod::Ccorr => sum_ccorr,
                        TemplateMatchMethod::CcorrNormed => {
                            let denom = (patch_sq_sum * tmpl_sq_sum).sqrt();
                            if denom > T::EPSILON {
                                sum_ccorr / denom
                            } else {
                                T::ZERO
                            }
                        }
                        TemplateMatchMethod::Ccoeff => sum_ccoeff,
                        TemplateMatchMethod::CcoeffNormed => {
                            let patch_var = patch_sq_sum - patch_sum * patch_sum / tmpl_len_f;
                            let denom = patch_var.sqrt() * tmpl_norm;
                            if denom > T::EPSILON {
                                sum_ccoeff / denom
                            } else {
                                T::ZERO
                            }
                        }
                    };
                }
            });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, out_h, out_w),
            dtype: image.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn detect_objects<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _threshold: T,
    ) -> Result<Vec<cv_core::Detection>> {
        Err(crate::Error::NotSupported(
            "CPU detect_objects not yet implemented".into(),
        ))
    }

    #[allow(clippy::needless_range_loop)]
    fn stereo_match<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        left: &Tensor<T, S>,
        right: &Tensor<T, S>,
        params: &StereoMatchParams,
    ) -> Result<Tensor<T, OS>> {
        let left_data = left
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Left image not on CPU".into()))?;
        let right_data = right
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Right image not on CPU".into()))?;

        let (h, w) = left.shape.hw();
        let (rh, rw) = right.shape.hw();
        if h != rh || w != rw {
            return Err(crate::Error::InvalidInput(
                "Left and right images must have the same dimensions".into(),
            ));
        }

        let block_size = params.block_size;
        let half_block = block_size / 2;
        let min_disp = params.min_disparity;
        let num_disp = params.num_disparities;

        let mut output_storage = OS::new(h * w, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        // Block matching with SAD (Sum of Absolute Differences)
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            if y < half_block || y + half_block >= h {
                return;
            }
            for x in 0..w {
                if x < half_block || x + half_block >= w {
                    continue;
                }

                let mut best_sad = T::from_f32(f32::MAX);
                let mut best_disp = T::ZERO;

                for d in 0..num_disp {
                    let disp = min_disp + d;
                    let rx = x as i32 - disp;
                    if rx < half_block as i32 || rx + half_block as i32 >= w as i32 {
                        continue;
                    }
                    let rx = rx as usize;

                    let mut sad = T::ZERO;
                    for by in 0..block_size {
                        let sy = y - half_block + by;
                        for bx in 0..block_size {
                            let lx = x - half_block + bx;
                            let rrx = rx - half_block + bx;
                            let lv = left_data[sy * w + lx];
                            let rv = right_data[sy * w + rrx];
                            sad += (lv - rv).abs();
                        }
                    }

                    if sad < best_sad {
                        best_sad = sad;
                        best_disp = T::from_f32(disp as f32);
                    }
                }

                row_out[x] = best_disp;
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, h, w),
            dtype: left.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn triangulate_points<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _proj_left: &[[T; 4]; 3],
        _proj_right: &[[T; 4]; 3],
        _points_left: &Tensor<T, S>,
        _points_right: &Tensor<T, S>,
    ) -> Result<Tensor<T, OS>> {
        Err(crate::Error::NotSupported(
            "CPU triangulate_points not yet implemented in HAL".into(),
        ))
    }

    fn find_chessboard_corners<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _image: &Tensor<T, S>,
        _pattern_size: (usize, usize),
    ) -> Result<Vec<cv_core::KeyPoint>> {
        Err(crate::Error::NotSupported(
            "CPU find_chessboard_corners not yet implemented in HAL".into(),
        ))
    }

    fn dispatch<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _name: &str,
        _buffers: &[&Tensor<u8, S>],
        _uniforms: &[u8],
        _workgroups: (u32, u32, u32),
    ) -> Result<()> {
        Err(crate::Error::NotSupported(
            "Generic dispatch not supported on CPU".into(),
        ))
    }

    fn threshold<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        thresh: T,
        max_value: T,
        typ: ThresholdType,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let len = src.len();

        let mut output_storage = S::new(len, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() && self.simd_available {
            let src_f32: &[f32] = bytemuck::cast_slice(src);
            let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
            let thresh_f32 = thresh.to_f32();
            let max_f32 = max_value.to_f32();

            let thresh_v = f32x8::splat(thresh_f32);
            let max_v = f32x8::splat(max_f32);
            let zero_v = f32x8::ZERO;

            dst_f32
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, dst_chunk)| {
                    let offset = chunk_idx * 4096;
                    let src_chunk = &src_f32[offset..offset + dst_chunk.len()];

                    let n = dst_chunk.len();
                    for i in (0..n).step_by(8) {
                        if i + 8 <= n {
                            let s_v = f32x8::new(src_chunk[i..i + 8].try_into().unwrap());
                            let res = match typ {
                                ThresholdType::Binary => s_v.cmp_gt(thresh_v).blend(max_v, zero_v),
                                ThresholdType::BinaryInv => {
                                    s_v.cmp_gt(thresh_v).blend(zero_v, max_v)
                                }
                                ThresholdType::Trunc => s_v.min(thresh_v),
                                ThresholdType::ToZero => s_v.cmp_gt(thresh_v).blend(s_v, zero_v),
                                ThresholdType::ToZeroInv => s_v.cmp_gt(thresh_v).blend(zero_v, s_v),
                            };
                            dst_chunk[i..i + 8].copy_from_slice(&res.to_array());
                        } else {
                            for j in i..n {
                                let value = src_chunk[j];
                                dst_chunk[j] = match typ {
                                    ThresholdType::Binary => {
                                        if value > thresh_f32 {
                                            max_f32
                                        } else {
                                            0.0
                                        }
                                    }
                                    ThresholdType::BinaryInv => {
                                        if value > thresh_f32 {
                                            0.0
                                        } else {
                                            max_f32
                                        }
                                    }
                                    ThresholdType::Trunc => value.min(thresh_f32),
                                    ThresholdType::ToZero => {
                                        if value > thresh_f32 {
                                            value
                                        } else {
                                            0.0
                                        }
                                    }
                                    ThresholdType::ToZeroInv => {
                                        if value > thresh_f32 {
                                            0.0
                                        } else {
                                            value
                                        }
                                    }
                                };
                            }
                        }
                    }
                });
        } else {
            // Scalar fallback for other types or when SIMD is disabled
            dst.par_iter_mut().enumerate().for_each(|(i, val)| {
                let value = src[i];
                *val = match typ {
                    ThresholdType::Binary => {
                        if value > thresh {
                            max_value
                        } else {
                            T::ZERO
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if value > thresh {
                            T::ZERO
                        } else {
                            max_value
                        }
                    }
                    ThresholdType::Trunc => {
                        if value > thresh {
                            thresh
                        } else {
                            value
                        }
                    }
                    ThresholdType::ToZero => {
                        if value > thresh {
                            value
                        } else {
                            T::ZERO
                        }
                    }
                    ThresholdType::ToZeroInv => {
                        if value > thresh {
                            T::ZERO
                        } else {
                            value
                        }
                    }
                };
            });
        }

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        let src_data = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let k_data = kernel
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Kernel not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let (kh, kw) = kernel.shape.hw();
        let (cx, cy) = (kw / 2, kh / 2);

        if iterations == 0 {
            return Ok(input.clone());
        }

        let mut current = src_data.to_vec();
        let mut next = vec![0u8; src_data.len()];

        for _ in 0..iterations {
            let src = &current;
            next.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                let y = y as isize;
                let mut x = 0;

                // SIMD path (32 pixels at a time)
                while x + 32 <= w {
                    let mut res_v = if typ == MorphologyType::Erode {
                        u8x32::splat(255)
                    } else {
                        u8x32::ZERO
                    };

                    for ky in 0..kh {
                        let sy = (y + ky as isize - cy as isize).clamp(0, h as isize - 1) as usize;
                        let row_src = &src[sy * w..(sy + 1) * w];

                        for kx in 0..kw {
                            if k_data[ky * kw + kx] == 0 {
                                continue;
                            }
                            let sx_base = x as isize + kx as isize - cx as isize;

                            // Load 32 bytes with clamping
                            let mut bytes = [0u8; 32];
                            for i in 0..32 {
                                let sx = (sx_base + i as isize).clamp(0, w as isize - 1) as usize;
                                bytes[i] = row_src[sx];
                            }

                            let v = u8x32::from(bytes);
                            if typ == MorphologyType::Erode {
                                res_v = res_v.min(v);
                            } else {
                                res_v = res_v.max(v);
                            }
                        }
                    }

                    let res_arr: [u8; 32] = res_v.into();
                    row_out[x..x + 32].copy_from_slice(&res_arr);
                    x += 32;
                }

                // Scalar tail
                for cx_idx in x..w {
                    let mut val = if typ == MorphologyType::Erode {
                        255u8
                    } else {
                        0u8
                    };
                    for ky in 0..kh {
                        let sy = (y + ky as isize - cy as isize).clamp(0, h as isize - 1) as usize;
                        for kx in 0..kw {
                            if k_data[ky * kw + kx] == 0 {
                                continue;
                            }
                            let sx = (cx_idx as isize + kx as isize - cx as isize)
                                .clamp(0, w as isize - 1)
                                as usize;
                            let v = src[sy * w + sx];
                            if typ == MorphologyType::Erode {
                                val = val.min(v);
                            } else {
                                val = val.max(v);
                            }
                        }
                    }
                    row_out[cx_idx] = val;
                }
            });
            std::mem::swap(&mut current, &mut next);
        }

        let result = Tensor {
            storage: S::from_vec(current).map_err(crate::Error::MemoryError)?,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        };

        Ok(result)
    }

    #[allow(clippy::needless_range_loop)]
    fn warp<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        matrix: &[[T; 3]; 3],
        new_shape: (usize, usize),
        _typ: WarpType,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let (nw, nh) = (new_shape.0, new_shape.1);
        let mut output_storage = S::new(nw * nh, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(nw).enumerate().for_each(|(y, row_out)| {
            for x in 0..nw {
                let fx = T::from_f32(x as f32);
                let fy = T::from_f32(y as f32);

                let sw = matrix[2][0] * fx + matrix[2][1] * fy + matrix[2][2];
                let sx = (matrix[0][0] * fx + matrix[0][1] * fy + matrix[0][2]) / sw;
                let sy = (matrix[1][0] * fx + matrix[1][1] * fy + matrix[1][2]) / sw;

                if sx >= T::ZERO
                    && sx < T::from_f32((w - 1) as f32)
                    && sy >= T::ZERO
                    && sy < T::from_f32((h - 1) as f32)
                {
                    // Bilinear interpolation
                    let x0 = sx.to_f32() as usize;
                    let y0 = sy.to_f32() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let dx = sx - T::from_f32(x0 as f32);
                    let dy = sy - T::from_f32(y0 as f32);
                    let one = T::ONE;

                    let v00 = src[y0 * w + x0];
                    let v10 = src[y0 * w + x1];
                    let v01 = src[y1 * w + x0];
                    let v11 = src[y1 * w + x1];

                    let val = v00 * (one - dx) * (one - dy)
                        + v10 * dx * (one - dy)
                        + v01 * (one - dx) * dy
                        + v11 * dx * dy;
                    row_out[x] = val;
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, nh, nw),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn nms<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        window_size: usize,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut output_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        let r = (window_size / 2) as isize;

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                let val = src[y * w + x];
                if val < threshold {
                    continue;
                }

                let mut is_max = true;
                for j in -r..=r {
                    for i in -r..=r {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        let sy = y as isize + j;
                        let sx = x as isize + i;
                        if sy >= 0
                            && sy < h as isize
                            && sx >= 0
                            && sx < w as isize
                            && src[sy as usize * w + sx as usize] > val
                        {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }
                if is_max {
                    row_out[x] = val;
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn nms_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        let data = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        // Based on common patterns, TensorShape is (channels, height, width).
        // Let's assume input is (N, 5).
        let rows = input.shape.height;
        let cols = input.shape.width;
        if cols != 5 {
            return Err(crate::Error::InvalidInput(
                "NMS Boxes expects (N, 5) tensor".into(),
            ));
        }

        let mut boxes: Vec<(usize, f32)> = (0..rows)
            .map(|i| {
                (i, data[i * 5 + 4].to_f32()) // (index, score)
            })
            .collect();

        // Sort by score descending
        boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; rows];

        for i in 0..boxes.len() {
            let (idx1, _) = boxes[i];
            if suppressed[idx1] {
                continue;
            }

            kept.push(idx1);
            let b1 = cv_core::Rect::new(
                data[idx1 * 5].to_f32(),
                data[idx1 * 5 + 1].to_f32(),
                (data[idx1 * 5 + 2] - data[idx1 * 5]).to_f32(),
                (data[idx1 * 5 + 3] - data[idx1 * 5 + 1]).to_f32(),
            );

            for j in (i + 1)..boxes.len() {
                let (idx2, _) = boxes[j];
                if suppressed[idx2] {
                    continue;
                }

                let b2 = cv_core::Rect::new(
                    data[idx2 * 5].to_f32(),
                    data[idx2 * 5 + 1].to_f32(),
                    (data[idx2 * 5 + 2] - data[idx2 * 5]).to_f32(),
                    (data[idx2 * 5 + 3] - data[idx2 * 5 + 1]).to_f32(),
                );
                if b1.iou(&b2) > iou_threshold.to_f32() {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    #[allow(clippy::needless_range_loop)]
    fn nms_rotated_boxes<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        let data = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let rows = input.shape.height;
        let cols = input.shape.width;
        if cols != 6 {
            return Err(crate::Error::InvalidInput(
                "NMS Rotated Boxes expects (N, 6) tensor".into(),
            ));
        }

        let mut boxes: Vec<(usize, T)> = (0..rows)
            .map(|i| {
                (i, data[i * 6 + 5]) // (index, score)
            })
            .collect();

        boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; rows];

        for i in 0..boxes.len() {
            let (idx1, _) = boxes[i];
            if suppressed[idx1] {
                continue;
            }

            kept.push(idx1);
            let r1 = cv_core::RotatedRect::new(
                data[idx1 * 6].to_f32(),
                data[idx1 * 6 + 1].to_f32(),
                data[idx1 * 6 + 2].to_f32(),
                data[idx1 * 6 + 3].to_f32(),
                data[idx1 * 6 + 4].to_f32(),
            );

            for j in (i + 1)..boxes.len() {
                let (idx2, _) = boxes[j];
                if suppressed[idx2] {
                    continue;
                }

                let r2 = cv_core::RotatedRect::new(
                    data[idx2 * 6].to_f32(),
                    data[idx2 * 6 + 1].to_f32(),
                    data[idx2 * 6 + 2].to_f32(),
                    data[idx2 * 6 + 3].to_f32(),
                    data[idx2 * 6 + 4].to_f32(),
                );

                if cv_core::rotated_iou(&r1, &r2) > iou_threshold.to_f32() {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    #[allow(clippy::needless_range_loop)]
    fn nms_polygons<T: Float + 'static>(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[T],
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        let n = polygons.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if scores.len() != n {
            return Err(crate::Error::InvalidInput(
                "Scores length must match polygons length".into(),
            ));
        }

        let mut items: Vec<(usize, T)> = (0..n).map(|i| (i, scores[i])).collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; n];

        for i in 0..n {
            let (idx1, _) = items[i];
            if suppressed[idx1] {
                continue;
            }

            kept.push(idx1);
            let p1 = &polygons[idx1];

            for j in (i + 1)..n {
                let (idx2, _) = items[j];
                if suppressed[idx2] {
                    continue;
                }

                let p2 = &polygons[idx2];
                if cv_core::polygon_iou(p1, p2) > iou_threshold.to_f32() {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    fn pointcloud_transform<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        transform: &[[T; 4]; 4],
    ) -> Result<Tensor<T, S>> {
        let src = points
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Points not on CPU".into()))?;
        let num_points = points.shape.height;
        let mut output_storage =
            S::new(num_points * 4, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(4)
            .enumerate()
            .for_each(|(i, point_out)| {
                let px = src[i * 4];
                let py = src[i * 4 + 1];
                let pz = src[i * 4 + 2];
                let pw = src[i * 4 + 3];

                point_out[0] = transform[0][0] * px
                    + transform[0][1] * py
                    + transform[0][2] * pz
                    + transform[0][3] * pw;
                point_out[1] = transform[1][0] * px
                    + transform[1][1] * py
                    + transform[1][2] * pz
                    + transform[1][3] * pw;
                point_out[2] = transform[2][0] * px
                    + transform[2][1] * py
                    + transform[2][2] * pz
                    + transform[2][3] * pw;
                point_out[3] = transform[3][0] * px
                    + transform[3][1] * py
                    + transform[3][2] * pz
                    + transform[3][3] * pw;
            });

        Ok(Tensor {
            storage: output_storage,
            shape: points.shape,
            dtype: points.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn pointcloud_normals<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<T, S>> {
        use nalgebra::{Matrix3, Vector3};

        let src = points
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Points not on CPU".into()))?;

        // Special case for f32 since the implementation uses rstar and specific eigensolver
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let src_f32: &[f32] = bytemuck::cast_slice(src);
            let num_points = points.shape.height;
            let mut normals_storage =
                S::new(num_points * 4, T::ZERO).map_err(crate::Error::MemoryError)?;
            let dst = normals_storage
                .as_mut_slice()
                .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
            let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

            let k = k_neighbors as usize;
            let pts: Vec<Vector3<f32>> = (0..num_points)
                .map(|i| Vector3::new(src_f32[i * 4], src_f32[i * 4 + 1], src_f32[i * 4 + 2]))
                .collect();

            // Use a KD-tree for efficient search
            let tree = rstar::RTree::bulk_load(
                (0..num_points)
                    .map(|i| {
                        [
                            src_f32[i * 4],
                            src_f32[i * 4 + 1],
                            src_f32[i * 4 + 2],
                            i as f32,
                        ]
                    })
                    .collect::<Vec<[f32; 4]>>(),
            );

            dst_f32
                .par_chunks_mut(4)
                .enumerate()
                .for_each(|(i, normal_out)| {
                    let p = pts[i];

                    // Find neighbors
                    let neighbors = tree
                        .nearest_neighbor_iter(&[p.x, p.y, p.z, 0.0])
                        .take(k)
                        .map(|neighbor| Vector3::new(neighbor[0], neighbor[1], neighbor[2]))
                        .collect::<Vec<_>>();

                    if neighbors.len() < 3 {
                        normal_out[0] = 0.0;
                        normal_out[1] = 0.0;
                        normal_out[2] = 1.0;
                        normal_out[3] = 0.0;
                        return;
                    }

                    // Compute centroid
                    let centroid = neighbors.iter().sum::<Vector3<f32>>() / neighbors.len() as f32;

                    // Compute covariance matrix
                    let mut cov = Matrix3::zeros();
                    for q in neighbors {
                        let diff = q - centroid;
                        cov += diff * diff.transpose();
                    }

                    // Analytic minimum eigenvector — Open3D / Geometric Tools RobustEigenSymmetric3x3.
                    let max_c = {
                        let mut m = 0.0f32;
                        for r in 0..3 {
                            for c in 0..3 {
                                let v = cov[(r, c)].abs();
                                if v > m {
                                    m = v;
                                }
                            }
                        }
                        m
                    };
                    let normal_vec: Vector3<f32> = if max_c < 1e-30 {
                        Vector3::z()
                    } else {
                        let s = 1.0 / max_c;
                        let a00 = cov[(0, 0)] * s;
                        let a01 = cov[(0, 1)] * s;
                        let a02 = cov[(0, 2)] * s;
                        let a11 = cov[(1, 1)] * s;
                        let a12 = cov[(1, 2)] * s;
                        let a22 = cov[(2, 2)] * s;
                        let norm_sq = a01 * a01 + a02 * a02 + a12 * a12;
                        let q = (a00 + a11 + a22) / 3.0;
                        let b00 = a00 - q;
                        let b11 = a11 - q;
                        let b22 = a22 - q;
                        let p_val =
                            ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm_sq) / 6.0).sqrt();
                        if p_val < 1e-10 {
                            Vector3::z()
                        } else {
                            let c00 = b11 * b22 - a12 * a12;
                            let c01 = a01 * b22 - a12 * a02;
                            let c02 = a01 * a12 - b11 * a02;
                            let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p_val * p_val * p_val);
                            let half_det = (det * 0.5_f32).clamp(-1.0, 1.0);
                            let angle = half_det.acos() / 3.0;
                            const TWO_THIRDS_PI: f32 = 2.094_395_1;
                            let eval_min = q + p_val * (angle + TWO_THIRDS_PI).cos() * 2.0;
                            let r0 = Vector3::new(a00 - eval_min, a01, a02);
                            let r1 = Vector3::new(a01, a11 - eval_min, a12);
                            let r2 = Vector3::new(a02, a12, a22 - eval_min);
                            let r0xr1 = r0.cross(&r1);
                            let r0xr2 = r0.cross(&r2);
                            let r1xr2 = r1.cross(&r2);
                            let d0 = r0xr1.norm_squared();
                            let d1 = r0xr2.norm_squared();
                            let d2 = r1xr2.norm_squared();
                            let best = if d0 >= d1 && d0 >= d2 {
                                r0xr1
                            } else if d1 >= d2 {
                                r0xr2
                            } else {
                                r1xr2
                            };
                            let len = best.norm();
                            if len < 1e-10 {
                                Vector3::z()
                            } else {
                                best / len
                            }
                        }
                    };
                    // Orient toward origin (flip if pointing away).
                    let flip = normal_vec.dot(&(-p)) < 0.0;
                    normal_out[0] = if flip { -normal_vec.x } else { normal_vec.x };
                    normal_out[1] = if flip { -normal_vec.y } else { normal_vec.y };
                    normal_out[2] = if flip { -normal_vec.z } else { normal_vec.z };
                    normal_out[3] = 0.0;
                });

            Ok(Tensor {
                storage: normals_storage,
                shape: points.shape,
                dtype: points.dtype,
                _phantom: std::marker::PhantomData,
            })
        } else {
            Err(crate::Error::NotSupported(
                "pointcloud_normals currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn tsdf_integrate<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        depth_image: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4], // World-to-camera
        intrinsics: &[T; 4],
        voxel_volume: &mut Tensor<T, S>,
        voxel_size: T,
        truncation: T,
    ) -> Result<()> {
        let depth = depth_image
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Depth not on CPU".into()))?;
        let voxels = voxel_volume
            .storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Voxels not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let depth_f32: &[f32] = bytemuck::cast_slice(depth);
            let voxels_f32: &mut [f32] = bytemuck::cast_slice_mut(voxels);
            let camera_pose_f32: [[f32; 4]; 4] = [
                [
                    camera_pose[0][0].to_f32(),
                    camera_pose[0][1].to_f32(),
                    camera_pose[0][2].to_f32(),
                    camera_pose[0][3].to_f32(),
                ],
                [
                    camera_pose[1][0].to_f32(),
                    camera_pose[1][1].to_f32(),
                    camera_pose[1][2].to_f32(),
                    camera_pose[1][3].to_f32(),
                ],
                [
                    camera_pose[2][0].to_f32(),
                    camera_pose[2][1].to_f32(),
                    camera_pose[2][2].to_f32(),
                    camera_pose[2][3].to_f32(),
                ],
                [
                    camera_pose[3][0].to_f32(),
                    camera_pose[3][1].to_f32(),
                    camera_pose[3][2].to_f32(),
                    camera_pose[3][3].to_f32(),
                ],
            ];
            let intrinsics_f32: [f32; 4] = [
                intrinsics[0].to_f32(),
                intrinsics[1].to_f32(),
                intrinsics[2].to_f32(),
                intrinsics[3].to_f32(),
            ];
            let voxel_size_f32 = voxel_size.to_f32();
            let truncation_f32 = truncation.to_f32();

            let (img_h, img_w) = depth_image.shape.hw();
            let (vx, vy, _vz) = (
                voxel_volume.shape.width,
                voxel_volume.shape.height,
                voxel_volume.shape.channels,
            );

            let fx = intrinsics_f32[0];
            let fy = intrinsics_f32[1];
            let cx = intrinsics_f32[2];
            let cy = intrinsics_f32[3];

            voxels_f32
                .par_chunks_mut(vx * vy * 2)
                .enumerate()
                .for_each(|(z_idx, plane)| {
                    for y_idx in 0..vy {
                        for x_idx in 0..vx {
                            let p_world = [
                                (x_idx as f32 + 0.5) * voxel_size_f32,
                                (y_idx as f32 + 0.5) * voxel_size_f32,
                                (z_idx as f32 + 0.5) * voxel_size_f32,
                            ];

                            let px = camera_pose_f32[0][0] * p_world[0]
                                + camera_pose_f32[0][1] * p_world[1]
                                + camera_pose_f32[0][2] * p_world[2]
                                + camera_pose_f32[0][3];
                            let py = camera_pose_f32[1][0] * p_world[0]
                                + camera_pose_f32[1][1] * p_world[1]
                                + camera_pose_f32[1][2] * p_world[2]
                                + camera_pose_f32[1][3];
                            let pz = camera_pose_f32[2][0] * p_world[0]
                                + camera_pose_f32[2][1] * p_world[1]
                                + camera_pose_f32[2][2] * p_world[2]
                                + camera_pose_f32[2][3];

                            if pz <= 0.0 {
                                continue;
                            }

                            let u = (px * fx / pz + cx).round() as i32;
                            let v = (py * fy / pz + cy).round() as i32;

                            if u < 0 || u >= img_w as i32 || v < 0 || v >= img_h as i32 {
                                continue;
                            }

                            let d = depth_f32[v as usize * img_w + u as usize];
                            if d <= 0.0 || d > 10.0 {
                                continue;
                            }

                            let dist = d - pz;
                            if dist < -truncation_f32 {
                                continue;
                            }

                            let tsdf_val = (dist / truncation_f32).clamp(-1.0, 1.0);
                            let v_idx = (y_idx * vx + x_idx) * 2;

                            let old_v = plane[v_idx];
                            let old_w = plane[v_idx + 1];

                            let new_w = (old_w + 1.0).min(50.0);
                            plane[v_idx] = (old_v * old_w + tsdf_val) / new_w;
                            plane[v_idx + 1] = new_w;
                        }
                    }
                });

            Ok(())
        } else {
            Err(crate::Error::NotSupported(
                "tsdf_integrate currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn tsdf_raycast<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _tsdf_volume: &Tensor<T, S>,
        _camera_pose: &[[T; 4]; 4],
        _intrinsics: &[T; 4],
        _image_size: (u32, u32),
        _depth_range: (T, T),
        _voxel_size: T,
        _truncation: T,
    ) -> Result<Tensor<T, S>> {
        Err(crate::Error::NotSupported(
            "CPU tsdf_raycast not yet implemented".into(),
        ))
    }

    fn tsdf_extract_mesh<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _tsdf_volume: &Tensor<T, S>,
        _voxel_size: T,
        _iso_level: T,
        _max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        Err(crate::Error::NotSupported(
            "CPU tsdf_extract_mesh not yet implemented".into(),
        ))
    }

    fn optical_flow_lk<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        prev_pyramid: &[Tensor<T, S>],
        next_pyramid: &[Tensor<T, S>],
        points: &[[T; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> Result<Vec<[T; 2]>> {
        if prev_pyramid.is_empty() || next_pyramid.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Optical flow requires non-empty pyramids".into(),
            ));
        }

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Single level implementation for CPU fallback
            let prev = &prev_pyramid[0];
            let next = &next_pyramid[0];
            let prev_data = prev
                .storage
                .as_slice()
                .ok_or_else(|| crate::Error::MemoryError("Prev not on CPU".into()))?;
            let next_data = next
                .storage
                .as_slice()
                .ok_or_else(|| crate::Error::MemoryError("Next not on CPU".into()))?;

            let prev_data_f32: &[f32] = bytemuck::cast_slice(prev_data);
            let next_data_f32: &[f32] = bytemuck::cast_slice(next_data);
            let points_f32: &[[f32; 2]] = bytemuck::cast_slice(points);

            let (h, w) = prev.shape.hw();

            let win = window_size as i32;
            let half_win = win / 2;

            let results: Vec<[f32; 2]> = points_f32
                .par_iter()
                .map(|&pt| {
                    let mut u = pt[0];
                    let mut v = pt[1];

                    for _ in 0..max_iters {
                        let ix = u.round() as i32;
                        let iy = v.round() as i32;

                        if ix - half_win < 0
                            || ix + half_win >= w as i32
                            || iy - half_win < 0
                            || iy + half_win >= h as i32
                        {
                            break;
                        }

                        let mut g: nalgebra::Matrix2<f32> = nalgebra::Matrix2::zeros();
                        let mut b: nalgebra::Vector2<f32> = nalgebra::Vector2::zeros();

                        for dy in -half_win..=half_win {
                            for dx in -half_win..=half_win {
                                let x = ix + dx;
                                let y = iy + dy;
                                let idx = (y * w as i32 + x) as usize;

                                // Spatial gradient on prev image
                                let i_x = (prev_data_f32[idx + 1] - prev_data_f32[idx - 1]) * 0.5;
                                let i_y = (prev_data_f32[idx + w] - prev_data_f32[idx - w]) * 0.5;

                                // Temporal gradient
                                let next_val = get_val_cpu(next_data_f32, w, h, x, y);
                                let i_t = next_val - prev_data_f32[idx];

                                g[(0, 0)] += i_x * i_x;
                                g[(0, 1)] += i_x * i_y;
                                g[(1, 0)] += i_x * i_y;
                                g[(1, 1)] += i_y * i_y;

                                b[0] -= i_x * i_t;
                                b[1] -= i_y * i_t;
                            }
                        }

                        if let Some(delta) = g.try_inverse().map(|inv| inv * b) {
                            u += delta[0];
                            v += delta[1];
                            if delta.norm_squared() < 0.01 {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    [u, v]
                })
                .collect();

            Ok(bytemuck::cast_vec(results))
        } else {
            Err(crate::Error::NotSupported(
                "optical_flow_lk currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn dense_icp_step<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        source_depth: &Tensor<T, S>,
        target_data: &Tensor<T, S>,
        intrinsics: &[T; 4],
        initial_guess: &nalgebra::Matrix4<T>,
        max_dist: T,
        max_angle: T,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        let src_depth = source_depth
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Source depth not on CPU".into()))?;
        let tgt_data = target_data
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Target data not on CPU".into()))?;

        let (src_h, src_w) = source_depth.shape.hw();

        // target_data is expected to be (h, w) with 7 channels: depth, vx, vy, vz, nx, ny, nz
        // or alternatively (h, w*7) — we handle both via total element count
        let tgt_h = target_data.shape.height;
        let tgt_w = target_data.shape.width;
        let tgt_channels = target_data.shape.channels;
        let tgt_stride = tgt_w * tgt_channels; // elements per row in target

        let fx = intrinsics[0].to_f32();
        let fy = intrinsics[1].to_f32();
        let cx = intrinsics[2].to_f32();
        let cy = intrinsics[3].to_f32();
        let fx_inv = 1.0 / fx;
        let fy_inv = 1.0 / fy;

        let max_dist_f = max_dist.to_f32();
        let max_angle_f = max_angle.to_f32();
        let cos_max_angle = max_angle_f.cos();

        // Convert initial_guess to f32
        let mut tf = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                tf[i][j] = initial_guess[(i, j)].to_f32();
            }
        }

        // Parallel accumulation using fold + reduce
        let (jtj_upper, jtr_acc) = (0..src_h)
            .into_par_iter()
            .fold(
                || ([0.0f32; 21], [0.0f32; 6]),
                |(mut jtj_upper, mut jtr), y| {
                    for x in 0..src_w {
                        let d_src = src_depth[y * src_w + x].to_f32();
                        if d_src <= 0.0 || !d_src.is_finite() {
                            continue;
                        }

                        // Back-project source pixel to 3D
                        let sx = d_src * (x as f32 - cx) * fx_inv;
                        let sy = d_src * (y as f32 - cy) * fy_inv;
                        let sz = d_src;

                        // Transform source point
                        let tx = tf[0][0] * sx + tf[0][1] * sy + tf[0][2] * sz + tf[0][3];
                        let ty_p = tf[1][0] * sx + tf[1][1] * sy + tf[1][2] * sz + tf[1][3];
                        let tz = tf[2][0] * sx + tf[2][1] * sy + tf[2][2] * sz + tf[2][3];

                        if tz <= 0.0 {
                            continue;
                        }

                        // Project transformed point into target image
                        let u = tx * fx / tz + cx;
                        let v = ty_p * fy / tz + cy;
                        let iu = u.round() as i32;
                        let iv = v.round() as i32;

                        if iu < 0 || iu >= tgt_w as i32 || iv < 0 || iv >= tgt_h as i32 {
                            continue;
                        }

                        // Read target vertex and normal
                        // Layout: 7 channels per pixel [depth, vx, vy, vz, nx, ny, nz]
                        let tgt_base = iv as usize * tgt_stride + iu as usize * tgt_channels;
                        if tgt_base + 6 >= tgt_data.len() {
                            continue;
                        }

                        let tgt_d = tgt_data[tgt_base].to_f32();
                        if tgt_d <= 0.0 || !tgt_d.is_finite() {
                            continue;
                        }

                        let tgt_vx = tgt_data[tgt_base + 1].to_f32();
                        let tgt_vy = tgt_data[tgt_base + 2].to_f32();
                        let tgt_vz = tgt_data[tgt_base + 3].to_f32();
                        let nx = tgt_data[tgt_base + 4].to_f32();
                        let ny = tgt_data[tgt_base + 5].to_f32();
                        let nz = tgt_data[tgt_base + 6].to_f32();

                        let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
                        if n_len < 1e-6 {
                            continue;
                        }

                        // Distance check
                        let dx = tx - tgt_vx;
                        let dy = ty_p - tgt_vy;
                        let dz = tz - tgt_vz;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > max_dist_f {
                            continue;
                        }

                        // Normal angle check: dot product between transform's Z-axis and target normal
                        let rz_x = tf[0][2];
                        let rz_y = tf[1][2];
                        let rz_z = tf[2][2];
                        let cos_angle = (rz_x * nx + rz_y * ny + rz_z * nz).abs() / n_len;
                        if cos_angle < cos_max_angle {
                            continue;
                        }

                        // Point-to-plane residual: (p_transformed - p_target) . n
                        let residual = dx * nx + dy * ny + dz * nz;

                        // Jacobian for point-to-plane ICP (rotation, translation)
                        // J = [n^T, (p_transformed x n)^T]
                        // where rotation part uses cross product of transformed point with normal
                        let cross_x = ty_p * nz - tz * ny;
                        let cross_y = tz * nx - tx * nz;
                        let cross_z = tx * ny - ty_p * nx;

                        let j = [nx, ny, nz, cross_x, cross_y, cross_z];

                        // Accumulate upper triangle of JtJ
                        let mut idx = 0;
                        for row in 0..6 {
                            for col in row..6 {
                                jtj_upper[idx] += j[row] * j[col];
                                idx += 1;
                            }
                        }
                        // Accumulate Jtr
                        for row in 0..6 {
                            jtr[row] += j[row] * residual;
                        }
                    }

                    (jtj_upper, jtr)
                },
            )
            .reduce(
                || ([0.0f32; 21], [0.0f32; 6]),
                |(mut a_jtj, mut a_jtr), (b_jtj, b_jtr)| {
                    for i in 0..21 {
                        a_jtj[i] += b_jtj[i];
                    }
                    for i in 0..6 {
                        a_jtr[i] += b_jtr[i];
                    }
                    (a_jtj, a_jtr)
                },
            );

        // Convert to nalgebra types
        let mut jtj = nalgebra::Matrix6::<T>::zeros();
        let mut idx = 0;
        for row in 0..6 {
            for col in row..6 {
                let v = T::from_f32(jtj_upper[idx]);
                jtj[(row, col)] = v;
                jtj[(col, row)] = v;
                idx += 1;
            }
        }

        let mut jtr_out = nalgebra::Vector6::<T>::zeros();
        for i in 0..6 {
            jtr_out[i] = T::from_f32(jtr_acc[i]);
        }

        Ok((jtj, jtr_out))
    }

    #[allow(clippy::needless_range_loop)]
    fn cvt_color<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        code: ColorConversion,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;

        match code {
            ColorConversion::RgbToGray | ColorConversion::BgrToGray => {
                if c != 3 {
                    return Err(crate::Error::InvalidInput(
                        "RgbToGray requires 3 channels".into(),
                    ));
                }
                let mut output_storage =
                    S::new(h * w, T::ZERO).map_err(crate::Error::MemoryError)?;
                let dst = output_storage
                    .as_mut_slice()
                    .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

                let wr = T::from_f32(0.299);
                let wg = T::from_f32(0.587);
                let wb = T::from_f32(0.114);

                dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                    for x in 0..w {
                        let base = (y * w + x) * 3;
                        let (r, g, b) = if code == ColorConversion::RgbToGray {
                            (src[base], src[base + 1], src[base + 2])
                        } else {
                            (src[base + 2], src[base + 1], src[base])
                        };
                        // Standard luminance formula
                        row_out[x] = wr * r + wg * g + wb * b;
                    }
                });

                Ok(Tensor {
                    storage: output_storage,
                    shape: TensorShape::new(1, h, w),
                    dtype: input.dtype,
                    _phantom: std::marker::PhantomData,
                })
            }
            ColorConversion::GrayToRgb => {
                if c != 1 {
                    return Err(crate::Error::InvalidInput(
                        "GrayToRgb requires 1 channel".into(),
                    ));
                }
                let mut output_storage =
                    S::new(h * w * 3, T::ZERO).map_err(crate::Error::MemoryError)?;
                let dst = output_storage
                    .as_mut_slice()
                    .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

                dst.par_chunks_mut(w * 3)
                    .enumerate()
                    .for_each(|(y, row_out)| {
                        for x in 0..w {
                            let val = src[y * w + x];
                            row_out[x * 3] = val;
                            row_out[x * 3 + 1] = val;
                            row_out[x * 3 + 2] = val;
                        }
                    });

                Ok(Tensor {
                    storage: output_storage,
                    shape: TensorShape::new(3, h, w),
                    dtype: input.dtype,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    fn resize<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;
        let (nw, nh) = new_shape;

        let mut output_storage = S::new(nw * nh * c, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let src_width_f = T::from_f32(w as f32 - 1.0);
        let src_height_f = T::from_f32(h as f32 - 1.0);
        let dst_width_f = T::from_f32(nw as f32 - 1.0);
        let dst_height_f = T::from_f32(nh as f32 - 1.0);

        dst.par_chunks_mut(nw * c)
            .enumerate()
            .for_each(|(y, row_out)| {
                for x in 0..nw {
                    let fx = T::from_f32(x as f32) * src_width_f / dst_width_f;
                    let fy = T::from_f32(y as f32) * src_height_f / dst_height_f;

                    let x0 = fx.to_f32() as usize;
                    let y0 = fy.to_f32() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let y1 = (y0 + 1).min(h - 1);

                    let dx = fx - T::from_f32(x0 as f32);
                    let dy = fy - T::from_f32(y0 as f32);

                    for ch in 0..c {
                        let v00 = src[(y0 * w + x0) * c + ch];
                        let v10 = src[(y0 * w + x1) * c + ch];
                        let v01 = src[(y1 * w + x0) * c + ch];
                        let v11 = src[(y1 * w + x1) * c + ch];

                        let v0 = v00 * (T::ONE - dx) + v10 * dx;
                        let v1 = v01 * (T::ONE - dx) + v11 * dx;
                        let v = v0 * (T::ONE - dy) + v1 * dy;

                        row_out[x * c + ch] = v;
                    }
                }
            });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(c, nh, nw),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn bilateral_filter<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        d: i32,
        sigma_color: T,
        sigma_space: T,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;
        if c != 1 {
            return Err(crate::Error::NotSupported(
                "Bilateral filter currently only for grayscale".into(),
            ));
        }

        let mut output_storage = S::new(h * w, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let radius = if d <= 0 {
            (sigma_space.to_f32() * 1.5).ceil() as i32
        } else {
            d / 2
        };
        let color_coeff = T::from_f32(-0.5) / (sigma_color * sigma_color);
        let space_coeff = T::from_f32(-0.5) / (sigma_space * sigma_space);

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                let center_val = src[y * w + x];
                let mut sum = T::ZERO;
                let mut norm = T::ZERO;

                for j in -radius..=radius {
                    for i in -radius..=radius {
                        let sy = (y as i32 + j).clamp(0, h as i32 - 1) as usize;
                        let sx = (x as i32 + i).clamp(0, w as i32 - 1) as usize;

                        let val = src[sy * w + sx];
                        let dist_sq = T::from_f32((i * i + j * j) as f32);
                        let range_sq = (val - center_val) * (val - center_val);

                        let weight = (dist_sq * space_coeff + range_sq * color_coeff).exp();
                        sum += val * weight;
                        norm += weight;
                    }
                }
                row_out[x] = sum / norm;
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn fast_detect<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        non_max_suppression: bool,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut output_storage = S::new(h * w, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            if y < 3 || y >= h - 3 {
                return;
            }
            for x in 3..w - 3 {
                let p = src[y * w + x];
                let high = p + threshold;
                let low = p - threshold;

                // Full 16-pixel circle
                let offsets = [
                    (0, -3),
                    (1, -3),
                    (2, -2),
                    (3, -1),
                    (3, 0),
                    (3, 1),
                    (2, 2),
                    (1, 3),
                    (0, 3),
                    (-1, 3),
                    (-2, 2),
                    (-3, 1),
                    (-3, 0),
                    (-3, -1),
                    (-2, -2),
                    (-1, -3),
                ];

                let mut vals = [T::ZERO; 16];
                for (i, &(dx, dy)) in offsets.iter().enumerate() {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    vals[i] = src[ny * w + nx];
                }

                if has_9_contiguous_generic(&vals, high, low) {
                    let mut score = T::ZERO;
                    for &v in &vals {
                        let diff = if v > p { v - p } else { p - v };
                        score += diff;
                    }
                    row_out[x] = score / T::from_f32(16.0);
                }
            }
        });

        if non_max_suppression {
            // Non-max suppression (3x3 neighborhood)
            let scores = dst.to_vec();
            dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                if y < 1 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let s = scores[y * w + x];
                    if s == T::ZERO {
                        continue;
                    }

                    let mut is_max = true;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let neighbor_s =
                                scores[((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize];
                            if neighbor_s > s
                                || (neighbor_s == s && (dy > 0 || (dy == 0 && dx > 0)))
                            {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }

                    if !is_max {
                        row_out[x] = T::ZERO;
                    }
                }
            });
        }

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, h, w),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn gaussian_blur<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        sigma: T,
        k_size: usize,
    ) -> Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;

        let mut output_storage = S::new(h * w * c, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let kernel = gaussian_kernel_1d(sigma, k_size);
        let rx = kernel.len() / 2;
        let ry = kernel.len() / 2;
        let kx = &kernel;
        let ky = &kernel;

        let mut intermediate = vec![T::ZERO; w * h * c];

        // Horizontal pass
        intermediate
            .par_chunks_mut(w * c)
            .enumerate()
            .for_each(|(y, row_inter)| {
                let row_src = &src[y * w * c..(y + 1) * w * c];
                for x in 0..w {
                    for ch in 0..c {
                        let mut sum = T::ZERO;
                        for i in 0..kx.len() {
                            let sx = (x as isize + i as isize - rx as isize)
                                .clamp(0, w as isize - 1)
                                as usize;
                            sum += row_src[sx * c + ch] * kx[i];
                        }
                        row_inter[x * c + ch] = sum;
                    }
                }
            });

        // Vertical pass
        dst.par_chunks_mut(w * c)
            .enumerate()
            .for_each(|(y, row_dst)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut sum = T::ZERO;
                        for j in 0..ky.len() {
                            let sy = (y as isize + j as isize - ry as isize)
                                .clamp(0, h as isize - 1)
                                as usize;
                            sum += intermediate[(sy * w + x) * c + ch] * ky[j];
                        }
                        row_dst[x * c + ch] = sum;
                    }
                }
            });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn pyramid_down<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        let (h, w) = input.shape.hw();
        let nw = w / 2;
        let nh = h / 2;
        let c = input.shape.channels;

        // Gaussian blur first
        let blurred = self.gaussian_blur(input, T::from_f32(1.0), 5)?;

        // Then downsample
        let mut output_storage = S::new(nh * nw * c, T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        let src_blurred = blurred.storage.as_slice().unwrap();

        dst.par_chunks_mut(nw * c)
            .enumerate()
            .for_each(|(y, row_out)| {
                for x in 0..nw {
                    for ch in 0..c {
                        row_out[x * c + ch] = src_blurred[((y * 2) * w + (x * 2)) * c + ch];
                    }
                }
            });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(c, nh, nw),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn subtract<
        T: Float + 'static + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        let src_a = a
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("A not on CPU".into()))?;
        let src_b = b
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("B not on CPU".into()))?;

        let mut output_storage =
            S::new(a.shape.len(), T::zeroed()).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        // This requires T to support subtraction. Since we use Pod + Debug,
        // we might need to restrict T or use dynamic dispatch if needed.
        // For SIFT, T is usually f32.

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a_f32: &[f32] = bytemuck::cast_slice(src_a);
            let b_f32: &[f32] = bytemuck::cast_slice(src_b);
            let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

            dst_f32.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = a_f32[i] - b_f32[i];
            });
        } else if TypeId::of::<T>() == TypeId::of::<u8>() {
            let a_u8: &[u8] = bytemuck::cast_slice(src_a);
            let b_u8: &[u8] = bytemuck::cast_slice(src_b);
            let dst_u8: &mut [u8] = bytemuck::cast_slice_mut(dst);

            dst_u8.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = a_u8[i].saturating_sub(b_u8[i]);
            });
        } else {
            return Err(crate::Error::NotSupported(
                "Subtraction not implemented for this type".into(),
            ));
        }

        Ok(Tensor {
            storage: output_storage,
            shape: a.shape,
            dtype: a.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        let q_slice = query
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Query not on CPU".into()))?;
        let t_slice = train
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Train not on CPU".into()))?;

        let q_len = query.shape.height;
        let t_len = train.shape.height;
        let d_size = query.shape.width; // Desc size in bytes

        let matches: Vec<cv_core::FeatureMatch> = (0..q_len)
            .into_par_iter()
            .filter_map(|qi| {
                let q_desc = &q_slice[qi * d_size..(qi + 1) * d_size];
                let mut best_dist = u32::MAX;
                let mut second_best = u32::MAX;
                let mut best_idx = 0;

                for ti in 0..t_len {
                    let t_desc = &t_slice[ti * d_size..(ti + 1) * d_size];
                    let dist = hamming_dist(q_desc, t_desc);

                    if dist < best_dist {
                        second_best = best_dist;
                        best_dist = dist;
                        best_idx = ti;
                    } else if dist < second_best {
                        second_best = dist;
                    }
                }

                if best_dist as f32 <= ratio_threshold * second_best as f32 {
                    Some(cv_core::FeatureMatch::new(
                        qi as i32,
                        best_idx as i32,
                        best_dist as f32,
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok(cv_core::Matches {
            matches,
            mask: None,
        })
    }

    fn sift_extrema<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        dog_prev: &Tensor<T, S>,
        dog_curr: &Tensor<T, S>,
        dog_next: &Tensor<T, S>,
        threshold: T,
        edge_threshold: T,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        let (h, w) = dog_curr.shape.hw();
        let prev = dog_prev
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Prev not on CPU".into()))?;
        let curr = dog_curr
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Curr not on CPU".into()))?;
        let next = dog_next
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Next not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let prev_f32: &[f32] = bytemuck::cast_slice(prev);
            let curr_f32: &[f32] = bytemuck::cast_slice(curr);
            let next_f32: &[f32] = bytemuck::cast_slice(next);
            let threshold_f32 = threshold.to_f32();
            let edge_threshold_f32 = edge_threshold.to_f32();

            let mut dst = vec![0u8; h * w];

            dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                if y < 1 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let val = curr_f32[y * w + x];
                    if val.abs() <= threshold_f32 {
                        continue;
                    }

                    let mut is_max = true;
                    let mut is_min = true;

                    for ds in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if ds == 0 && dx == 0 && dy == 0 {
                                    continue;
                                }
                                let neighbor_val = match ds {
                                    -1 => {
                                        prev_f32[(y as i32 + dy) as usize * w
                                            + (x as i32 + dx) as usize]
                                    }
                                    0 => {
                                        curr_f32[(y as i32 + dy) as usize * w
                                            + (x as i32 + dx) as usize]
                                    }
                                    1 => {
                                        next_f32[(y as i32 + dy) as usize * w
                                            + (x as i32 + dx) as usize]
                                    }
                                    _ => 0.0,
                                };
                                if neighbor_val >= val {
                                    is_max = false;
                                }
                                if neighbor_val <= val {
                                    is_min = false;
                                }
                            }
                        }
                    }

                    if is_max || is_min {
                        let dxx = curr_f32[y * w + x + 1] + curr_f32[y * w + x - 1] - 2.0 * val;
                        let dyy = curr_f32[(y + 1) * w + x] + curr_f32[(y - 1) * w + x] - 2.0 * val;
                        let dxy = (curr_f32[(y + 1) * w + x + 1]
                            - curr_f32[(y + 1) * w + x - 1]
                            - curr_f32[(y - 1) * w + x + 1]
                            + curr_f32[(y - 1) * w + x - 1])
                            / 4.0;

                        let tr = dxx + dyy;
                        let det = dxx * dyy - dxy * dxy;
                        let r = edge_threshold_f32;
                        if det > 0.0 && (tr * tr) / det < (r + 1.0) * (r + 1.0) / r {
                            row_out[x] = 1;
                        }
                    }
                }
            });

            Tensor::from_vec(dst, dog_curr.shape)
                .map_err(|e| crate::Error::RuntimeError(e.to_string()))
        } else {
            Err(crate::Error::NotSupported(
                "sift_extrema currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn compute_sift_descriptors<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        let (h, w) = image.shape.hw();
        let src = image
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Image not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let src_f32: &[f32] = bytemuck::cast_slice(src);

            let descs: Vec<cv_core::Descriptor> = keypoints
                .keypoints
                .par_iter()
                .map(|kp| {
                    let cx = kp.x.to_f32();
                    let cy = kp.y.to_f32();
                    let size = kp.size.to_f32();
                    let angle_rad = kp.angle.to_f32() * std::f32::consts::PI / 180.0;

                    let cos_a = angle_rad.cos();
                    let sin_a = angle_rad.sin();

                    let mut hist = [0.0f32; 128];
                    let bin_width = size * 3.0;
                    let radius = (bin_width * 2.0) as i32;

                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let rx = (dx as f32 * cos_a + dy as f32 * sin_a) / bin_width;
                            let ry = (-dx as f32 * sin_a + dy as f32 * cos_a) / bin_width;

                            let r_bin_x = rx + 1.5;
                            let r_bin_y = ry + 1.5;

                            if r_bin_x > -1.0 && r_bin_x < 4.0 && r_bin_y > -1.0 && r_bin_y < 4.0 {
                                let x = (cx + dx as f32) as i32;
                                let y = (cy + dy as f32) as i32;

                                let g_x = get_val_cpu(src_f32, w, h, x + 1, y)
                                    - get_val_cpu(src_f32, w, h, x - 1, y);
                                let g_y = get_val_cpu(src_f32, w, h, x, y + 1)
                                    - get_val_cpu(src_f32, w, h, x, y - 1);
                                let mag = (g_x * g_x + g_y * g_y).sqrt();
                                let mut ori = g_y.atan2(g_x) - angle_rad;
                                while ori < 0.0 {
                                    ori += 2.0 * std::f32::consts::PI;
                                }
                                let o_bin = ori * 8.0 / (2.0 * std::f32::consts::PI);

                                let ix = r_bin_x.floor() as i32;
                                let iy = r_bin_y.floor() as i32;
                                let io = o_bin.floor() as i32;

                                if (0..4).contains(&ix) && (0..4).contains(&iy) {
                                    let bin_idx = (iy * 4 + ix) * 8 + (io % 8);
                                    hist[bin_idx as usize] += mag;
                                }
                            }
                        }
                    }

                    let mut norm_sq = 0.0;
                    for v in &hist {
                        norm_sq += v * v;
                    }
                    let norm_inv = 1.0 / (norm_sq.sqrt() + 1e-7);

                    let data: Vec<u8> = hist
                        .iter()
                        .map(|&v| {
                            let norm_v = (v * norm_inv).min(0.2);
                            ((norm_v * 512.0).min(255.0)) as u8
                        })
                        .collect();

                    cv_core::Descriptor::new(data, *kp)
                })
                .collect();

            Ok(cv_core::Descriptors { descriptors: descs })
        } else {
            Err(crate::Error::NotSupported(
                "compute_sift_descriptors currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn icp_correspondences<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        src: &Tensor<T, S>,
        tgt: &Tensor<T, S>,
        max_dist: T,
    ) -> Result<Vec<(usize, usize, T)>> {
        let src_points = src
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Src not on CPU".into()))?;
        let tgt_points = tgt
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Tgt not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let src_f32: &[f32] = bytemuck::cast_slice(src_points);
            let tgt_f32: &[f32] = bytemuck::cast_slice(tgt_points);
            let max_dist_f32 = max_dist.to_f32();

            let num_src = src.shape.height;
            let num_tgt = tgt.shape.height;
            let max_dist_sq = max_dist_f32 * max_dist_f32;

            let correspondences: Vec<(usize, usize, T)> = (0..num_src)
                .into_par_iter()
                .filter_map(|si| {
                    let p_s = [src_f32[si * 4], src_f32[si * 4 + 1], src_f32[si * 4 + 2]];
                    let mut min_dist_sq = f32::MAX;
                    let mut best_ti = 0;
                    let mut found = false;

                    for ti in 0..num_tgt {
                        let dx = p_s[0] - tgt_f32[ti * 4];
                        let dy = p_s[1] - tgt_f32[ti * 4 + 1];
                        let dz = p_s[2] - tgt_f32[ti * 4 + 2];
                        let d2 = dx * dx + dy * dy + dz * dz;

                        if d2 < min_dist_sq {
                            min_dist_sq = d2;
                            best_ti = ti;
                            found = true;
                        }
                    }

                    if found && min_dist_sq <= max_dist_sq {
                        Some((si, best_ti, T::from_f32(min_dist_sq.sqrt())))
                    } else {
                        None
                    }
                })
                .collect::<Vec<(usize, usize, T)>>();

            Ok(correspondences)
        } else {
            Err(crate::Error::NotSupported(
                "icp_correspondences currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn icp_accumulate<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        source: &Tensor<T, S>,
        target: &Tensor<T, S>,
        target_normals: &Tensor<T, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<T>,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        let src_slice = source
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Src not on CPU".into()))?;
        let tgt_slice = target
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Tgt not on CPU".into()))?;
        let norm_slice = target_normals
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Normals not on CPU".into()))?;

        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Safety: we verified T == f32 via TypeId above.
            let src_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(src_slice.as_ptr() as *const f32, src_slice.len())
            };
            let tgt_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(tgt_slice.as_ptr() as *const f32, tgt_slice.len())
            };
            let norm_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(norm_slice.as_ptr() as *const f32, norm_slice.len())
            };
            let mut transform_f32 = nalgebra::Matrix4::<f32>::zeros();
            for i in 0..16 {
                transform_f32[i] = transform[i].to_f32();
            }

            let mut ata = nalgebra::Matrix6::<f32>::zeros();
            let mut atb = nalgebra::Vector6::<f32>::zeros();

            for &(src_idx, tgt_idx) in correspondences {
                let src_idx = src_idx as usize;
                let tgt_idx = tgt_idx as usize;

                let p_src = nalgebra::Point3::new(
                    src_f32[src_idx * 4],
                    src_f32[src_idx * 4 + 1],
                    src_f32[src_idx * 4 + 2],
                );
                let p_tgt = nalgebra::Point3::new(
                    tgt_f32[tgt_idx * 4],
                    tgt_f32[tgt_idx * 4 + 1],
                    tgt_f32[tgt_idx * 4 + 2],
                );
                let n_tgt = nalgebra::Vector3::new(
                    norm_f32[tgt_idx * 4],
                    norm_f32[tgt_idx * 4 + 1],
                    norm_f32[tgt_idx * 4 + 2],
                );

                let p_trans = transform_f32.transform_point(&p_src);
                let diff = p_trans - p_tgt;
                let residual = diff.dot(&n_tgt);

                let cross = p_trans.coords.cross(&n_tgt);
                let jacobian =
                    nalgebra::Vector6::new(n_tgt.x, n_tgt.y, n_tgt.z, cross.x, cross.y, cross.z);

                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
            }

            let mut ata_t = nalgebra::Matrix6::<T>::zeros();
            for i in 0..36 {
                ata_t[i] = T::from_f32(ata[i]);
            }
            let mut atb_t = nalgebra::Vector6::<T>::zeros();
            for i in 0..6 {
                atb_t[i] = T::from_f32(atb[i]);
            }
            Ok((ata_t, atb_t))
        } else {
            Err(crate::Error::NotSupported(
                "icp_accumulate currently only supports f32 on CPU".into(),
            ))
        }
    }

    fn akaze_diffusion<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        k: T,
        tau: T,
    ) -> crate::Result<Tensor<T, S>> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut output_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let dst = output_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let k2 = k * k;

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                let center = src[y * w + x];

                let n = src[if y > 0 { (y - 1) * w + x } else { y * w + x }];
                let s = src[if y < h - 1 {
                    (y + 1) * w + x
                } else {
                    y * w + x
                }];
                let west = src[if x > 0 { y * w + (x - 1) } else { y * w + x }];
                let east = src[if x < w - 1 {
                    y * w + (x + 1)
                } else {
                    y * w + x
                }];

                let grad_n = n - center;
                let grad_s = s - center;
                let grad_w = west - center;
                let grad_e = east - center;

                let g_n = T::ONE / (T::ONE + grad_n * grad_n / k2);
                let g_s = T::ONE / (T::ONE + grad_s * grad_s / k2);
                let g_w = T::ONE / (T::ONE + grad_w * grad_w / k2);
                let g_e = T::ONE / (T::ONE + grad_e * grad_e / k2);

                row_out[x] =
                    center + tau * (g_n * grad_n + g_s * grad_s + g_w * grad_w + g_e * grad_e);
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn akaze_derivatives<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> crate::Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();

        let mut lx_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let mut ly_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;
        let mut ldet_storage =
            S::new(input.shape.len(), T::ZERO).map_err(crate::Error::MemoryError)?;

        let lx_slice = lx_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        let ly_slice = ly_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        let ldet_slice = ldet_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let get_val = |x: i32, y: i32| -> T {
            let cx = x.clamp(0, w as i32 - 1) as usize;
            let cy = y.clamp(0, h as i32 - 1) as usize;
            src[cy * w + cx]
        };

        lx_slice
            .par_chunks_mut(w)
            .zip(ly_slice.par_chunks_mut(w))
            .zip(ldet_slice.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((row_lx, row_ly), row_ldet))| {
                let y = y as i32;
                for x in 0..w {
                    let x = x as i32;
                    let lx = (get_val(x + 1, y - 1)
                        + T::from_f32(3.0) * get_val(x + 1, y)
                        + get_val(x + 1, y + 1))
                        - (get_val(x - 1, y - 1)
                            + T::from_f32(3.0) * get_val(x - 1, y)
                            + get_val(x - 1, y + 1));

                    let ly = (get_val(x - 1, y + 1)
                        + T::from_f32(3.0) * get_val(x, y + 1)
                        + get_val(x + 1, y + 1))
                        - (get_val(x - 1, y - 1)
                            + T::from_f32(3.0) * get_val(x, y - 1)
                            + get_val(x + 1, y - 1));

                    row_lx[x as usize] = lx / T::from_f32(32.0);
                    row_ly[x as usize] = ly / T::from_f32(32.0);

                    let lxx =
                        get_val(x + 1, y) + get_val(x - 1, y) - T::from_f32(2.0) * get_val(x, y);
                    let lyy =
                        get_val(x, y + 1) + get_val(x, y - 1) - T::from_f32(2.0) * get_val(x, y);
                    let lxy = (get_val(x + 1, y + 1) + get_val(x - 1, y - 1)
                        - get_val(x - 1, y + 1)
                        - get_val(x + 1, y - 1))
                        / T::from_f32(4.0);

                    row_ldet[x as usize] = lxx * lyy - lxy * lxy;
                }
            });

        Ok((
            Tensor {
                storage: lx_storage,
                shape: input.shape,
                dtype: input.dtype,
                _phantom: std::marker::PhantomData,
            },
            Tensor {
                storage: ly_storage,
                shape: input.shape,
                dtype: input.dtype,
                _phantom: std::marker::PhantomData,
            },
            Tensor {
                storage: ldet_storage,
                shape: input.shape,
                dtype: input.dtype,
                _phantom: std::marker::PhantomData,
            },
        ))
    }

    fn akaze_contrast_k<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> crate::Result<T> {
        let src = input
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();

        // Sample gradients
        let mut mags: Vec<T> = (1..h - 1)
            .into_par_iter()
            .flat_map(|y| {
                let mut row_mags = Vec::with_capacity(w);
                for x in 1..w - 1 {
                    let lx = src[y * w + x + 1] - src[y * w + x - 1];
                    let ly = src[(y + 1) * w + x] - src[(y - 1) * w + x];
                    row_mags.push((lx * lx + ly * ly).sqrt());
                }
                row_mags
            })
            .collect();

        if mags.is_empty() {
            return Ok(T::from_f32(0.03));
        }

        mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (mags.len() as f32 * 0.7) as usize;
        Ok(mags[idx.min(mags.len() - 1)])
    }

    fn spmv<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[T],
        x: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        let x_slice = x
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("X not on CPU".into()))?;
        let rows = row_ptr.len() - 1;
        let mut y_storage = S::new(rows, T::ZERO).map_err(crate::Error::MemoryError)?;
        let y_slice = y_storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        y_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;
            let mut sum = T::ZERO;
            for j in start..end {
                sum += values[j] * x_slice[col_indices[j] as usize];
            }
            *val = sum;
        });

        Ok(Tensor {
            storage: y_storage,
            shape: cv_core::TensorShape::new(1, rows, 1),
            dtype: x.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn mog2_update<T: Float + 'static, S1: Storage<T> + 'static, S2: Storage<u32> + 'static>(
        &self,
        frame: &Tensor<T, S1>,
        model: &mut Tensor<T, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params<T>,
    ) -> crate::Result<()> {
        let frame_data = frame
            .storage
            .as_slice()
            .ok_or_else(|| crate::Error::MemoryError("Frame not on CPU".into()))?;
        let model_data: &mut [T] = model
            .storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Model not on CPU".into()))?;
        let mask_data: &mut [u32] = mask
            .storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::MemoryError("Mask not on CPU".into()))?;

        let width = params.width as usize;
        let _height = params.height as usize;
        let n_mixtures = params.n_mixtures as usize;
        let alpha = params.alpha;
        let var_threshold = params.var_threshold;
        let background_ratio = params.background_ratio;
        let var_init = params.var_init;
        let var_min = params.var_min;
        let var_max = params.var_max;

        mask_data
            .par_chunks_mut(width)
            .zip(model_data.par_chunks_mut(width * n_mixtures * 3))
            .enumerate()
            .for_each(|(y, (mask_row, model_row))| {
                for x in 0..width {
                    let pixel = frame_data[y * width + x];
                    let pix_model = &mut model_row[x * n_mixtures * 3..(x + 1) * n_mixtures * 3];

                    let mut fit_idx = None;
                    let mut foreground = true;
                    let mut total_weight = T::ZERO;

                    for m in 0..n_mixtures {
                        let m_base = m * 3;
                        let weight = pix_model[m_base];
                        let mean = pix_model[m_base + 1];
                        let var = pix_model[m_base + 2];

                        if weight < T::from_f32(1e-5) {
                            continue;
                        }

                        let diff = pixel - mean;
                        if diff * diff < var_threshold * var {
                            fit_idx = Some(m);
                            if total_weight < background_ratio {
                                foreground = false;
                            }
                            break;
                        }
                        total_weight += weight;
                    }

                    mask_row[x] = if foreground { 255u32 } else { 0u32 };

                    if let Some(idx) = fit_idx {
                        for m in 0..n_mixtures {
                            let m_base = m * 3;
                            if m == idx {
                                let w_val = pix_model[m_base];
                                let alpha_m = alpha / w_val.max(T::from_f32(1e-5));
                                pix_model[m_base] += alpha * (T::ONE - w_val);
                                let diff = pixel - pix_model[m_base + 1];
                                pix_model[m_base + 1] += alpha_m * diff;
                                let new_var = pix_model[m_base + 2]
                                    + alpha_m * (diff * diff - pix_model[m_base + 2]);
                                pix_model[m_base + 2] = new_var.clamp(var_min, var_max);
                            } else {
                                pix_model[m_base] *= T::ONE - alpha;
                            }
                        }
                    } else {
                        let mut min_w_idx = 0;
                        let mut min_w = T::from_f32(2.0);
                        for m in 0..n_mixtures {
                            if pix_model[m * 3] < min_w {
                                min_w = pix_model[m * 3];
                                min_w_idx = m;
                            }
                        }
                        let m_base = min_w_idx * 3;
                        pix_model[m_base] = alpha;
                        pix_model[m_base + 1] = pixel;
                        pix_model[m_base + 2] = var_init;
                    }
                }
            });
        Ok(())
    }
}

fn get_val_cpu<T: Float>(src: &[T], w: usize, h: usize, x: i32, y: i32) -> T {
    let cx = x.clamp(0, w as i32 - 1) as usize;
    let cy = y.clamp(0, h as i32 - 1) as usize;
    src[cy * w + cx]
}

#[allow(clippy::needless_range_loop)]
fn has_9_contiguous_generic<T: Float>(vals: &[T; 16], high: T, low: T) -> bool {
    let mut b_mask = 0u32;
    let mut d_mask = 0u32;
    for i in 0..16 {
        if vals[i] > high {
            b_mask |= 1 << i;
        }
        if vals[i] < low {
            d_mask |= 1 << i;
        }
    }

    let b_mask_ext = b_mask | (b_mask << 16);
    let d_mask_ext = d_mask | (d_mask << 16);

    for i in 0..16 {
        if (b_mask_ext >> i) & 0x1FF == 0x1FF {
            return true;
        }
        if (d_mask_ext >> i) & 0x1FF == 0x1FF {
            return true;
        }
    }
    false
}

fn hamming_dist(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

impl CpuBackend {
    pub fn is_available() -> bool {
        true
    }
}
