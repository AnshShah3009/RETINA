//! AKAZE (Accelerated-KAZE) feature detector and descriptor
//!
//! AKAZE is a fast multi-scale feature detector and descriptor that uses
//! non-linear diffusion scale-space.

#![allow(deprecated)]

use crate::descriptor::{Descriptor, Descriptors};
use crate::Result;
use cv_core::{storage::Storage, CpuTensor, Error, KeyPoint, KeyPoints, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::{TensorCast, TensorToCpu, TensorToGpu};
use rayon::prelude::*;

/// AKAZE feature detection and extraction parameters
///
/// Configuration for the AKAZE algorithm controlling scale-space parameters and
/// the non-linear diffusion method for scale-space construction.
///
/// # Parameters
///
/// * `n_octaves` - Number of octaves (pyramid levels) in the scale space
/// * `n_sublevels` - Number of sub-levels within each octave
/// * `threshold` - Detector response threshold for keypoint selection
/// * `diffusivity` - Type of non-linear diffusion to apply
///
/// # Default Values
///
/// The default parameters use 4 octaves, 4 sublevels per octave, a threshold of 0.001,
/// and Perona-Malik type 2 diffusivity, providing a good balance between computational
/// cost and feature detection quality.
#[derive(Debug, Clone)]
pub struct AkazeParams {
    /// Number of octaves (scale-space levels)
    pub n_octaves: usize,
    /// Number of sublevels within each octave
    pub n_sublevels: usize,
    /// Detector response threshold for selecting keypoints
    pub threshold: f32,
    /// Type of non-linear diffusivity for scale-space construction
    pub diffusivity: DiffusivityType,
}

/// Non-linear diffusion types for AKAZE scale-space construction
///
/// Different diffusivity functions model how edge-aware blurring proceeds:
/// - **Perona-Malik 1**: Edge-preserving diffusion with sharp edge detection
/// - **Perona-Malik 2**: Improved version with better contrast preservation
/// - **Weickert**: Uses flow-based diffusion with improved edge preservation
/// - **Charbonnier**: Smooth penalty function for more stable evolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusivityType {
    /// Perona-Malik type 1 diffusion
    PeronaMalik1,
    /// Perona-Malik type 2 diffusion (improved)
    PeronaMalik2,
    /// Weickert edge-aware diffusion
    Weickert,
    /// Charbonnier smooth diffusion
    Charbonnier,
}

impl Default for AkazeParams {
    fn default() -> Self {
        Self {
            n_octaves: 4,
            n_sublevels: 4,
            threshold: 0.001,
            diffusivity: DiffusivityType::PeronaMalik2,
        }
    }
}

/// AKAZE (Accelerated-KAZE) feature detector
///
/// Provides fast multi-scale feature detection and descriptor extraction using
/// accelerated non-linear diffusion for scale-space construction.
///
/// # Algorithm Overview
///
/// AKAZE builds a non-linear diffusion scale-space where each octave is constructed
/// by applying iterative nonlinear diffusion filters. Features are detected as
/// extrema in the determinant of the Hessian matrix across scales.
///
/// # Example
///
/// ```no_run
/// # use cv_features::akaze::{Akaze, AkazeParams};
/// # use cv_hal::cpu::CpuBackend;
/// # use cv_hal::compute::ComputeDevice;
/// # use cv_core::TensorShape;
/// # let cpu = CpuBackend::new().unwrap();
/// # let device = ComputeDevice::Cpu(&cpu);
/// let params = AkazeParams::default();
/// let akaze = Akaze::new(params);
/// // akaze.detect_ctx(&device, &image_tensor);
/// ```
pub struct Akaze {
    params: AkazeParams,
}

impl Akaze {
    /// Create a new AKAZE detector with the given parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Configuration parameters controlling scale-space and detection behavior
    ///
    /// # Returns
    ///
    /// A new AKAZE detector instance ready for feature detection
    pub fn new(params: AkazeParams) -> Self {
        Self { params }
    }

    /// Detect keypoints without computing descriptors
    ///
    /// Efficiently detects AKAZE keypoints by building a non-linear diffusion scale-space
    /// and finding local extrema in the Hessian determinant.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device (CPU or GPU) to execute detection on
    /// * `image` - Input grayscale image tensor (u8 format, values 0-255)
    ///
    /// # Returns
    ///
    /// * `Ok(KeyPoints)` - Detected keypoints with position, scale, and response
    /// * `Err(FeatureError)` - If scale-space construction or extrema finding fails
    ///
    /// # Errors
    ///
    /// May return errors if:
    /// - GPU operations fail (memory allocation, transfer, computation)
    /// - Tensor operations fail (shape mismatch, memory access)
    /// - Image dimensions are invalid
    pub fn detect_ctx<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> Result<KeyPoints>
    where
        Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>,
    {
        let evolution = self.create_scale_space(ctx, image)?;
        let keypoints = self.find_extrema(&evolution)?;
        Ok(KeyPoints { keypoints })
    }

    /// Detect keypoints and compute M-SURF descriptors
    ///
    /// Full AKAZE pipeline that detects keypoints and extracts M-SURF (Modified SURF)
    /// descriptors for each detected feature.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device (CPU or GPU) for execution
    /// * `image` - Input grayscale image (u8, 0-255 value range)
    ///
    /// # Returns
    ///
    /// * `Ok((KeyPoints, Descriptors))` - Detected keypoints and their 64-bit M-SURF descriptors
    /// * `Err(FeatureError)` - If detection, scale-space, or descriptor computation fails
    ///
    /// # Errors
    ///
    /// May fail with same errors as [`detect_ctx`] plus:
    /// - Descriptor computation failures (accessing image derivatives, normalization)
    ///
    /// # Output Format
    ///
    /// Each descriptor is 64 bytes (8 4D vectors: dx, dy, |dx|, |dy| for 4x4 regions),
    /// normalized to zero mean and unit variance.
    pub fn detect_and_compute_ctx<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> Result<(KeyPoints, Descriptors)>
    where
        Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>,
    {
        let evolution = self.create_scale_space(ctx, image)?;
        let keypoints = self.find_extrema(&evolution)?;
        let descriptors = self.compute_descriptors(&evolution, &keypoints)?;
        Ok((KeyPoints { keypoints }, descriptors))
    }

    fn create_scale_space<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> Result<Vec<EvolutionLevel>>
    where
        Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>,
    {
        let mut evolution = Vec::new();
        let mut t = 0.0f32;

        match ctx {
            ComputeDevice::Gpu(gpu) => {
                let current_u8 = ctx.gaussian_blur(image, 1.0, 5).map_err(|e| {
                    Error::FeatureError(format!("Gaussian blur failed on GPU: {}", e))
                })?;
                let gpu_u8 = <Tensor<u8, S> as TensorToGpu<u8>>::to_gpu_ctx(&current_u8, gpu)
                    .map_err(|e| {
                        Error::FeatureError(format!("Failed to transfer tensor to GPU: {}", e))
                    })?;
                let mut current_f32 = <Tensor<u8, _> as TensorCast>::to_f32_ctx(&gpu_u8, gpu)
                    .map_err(|e| {
                        Error::FeatureError(format!("Failed to cast tensor to f32: {}", e))
                    })?;
                let k = ctx.akaze_contrast_k(&current_f32).unwrap_or(0.03);

                for o in 0..self.params.n_octaves {
                    for s in 0..self.params.n_sublevels {
                        let sigma = (2.0f32)
                            .powf((o as f32) + (s as f32) / (self.params.n_sublevels as f32));
                        let esigma = sigma * sigma / 2.0;
                        let dt = esigma - t;

                        if dt > 0.0 {
                            let n_steps = self.get_fed_steps(dt);
                            let step_tau = dt / (n_steps as f32);
                            for _ in 0..n_steps {
                                current_f32 = gpu
                                    .akaze_diffusion(&current_f32, k, step_tau)
                                    .map_err(|e| {
                                        Error::FeatureError(format!(
                                            "Diffusion failed on GPU: {}",
                                            e
                                        ))
                                    })?;
                            }
                            t = esigma;
                        }

                        let (lx, ly, ldet) = gpu.akaze_derivatives(&current_f32).map_err(|e| {
                            Error::FeatureError(format!("Derivatives failed on GPU: {}", e))
                        })?;

                        evolution.push(EvolutionLevel {
                            image: current_f32.to_cpu_ctx(gpu).map_err(|e| {
                                Error::FeatureError(format!(
                                    "Failed to transfer image to CPU: {}",
                                    e
                                ))
                            })?,
                            lx: lx.to_cpu_ctx(gpu).map_err(|e| {
                                Error::FeatureError(format!("Failed to transfer lx to CPU: {}", e))
                            })?,
                            ly: ly.to_cpu_ctx(gpu).map_err(|e| {
                                Error::FeatureError(format!("Failed to transfer ly to CPU: {}", e))
                            })?,
                            ldet: ldet.to_cpu_ctx(gpu).map_err(|e| {
                                Error::FeatureError(format!(
                                    "Failed to transfer ldet to CPU: {}",
                                    e
                                ))
                            })?,
                            sigma,
                            octave: o,
                        });
                    }
                }
            }
            ComputeDevice::Cpu(cpu) => {
                let cpu_u8 = image.to_cpu().map_err(|e| {
                    Error::FeatureError(format!("Failed to download image to CPU: {}", e))
                })?;
                let current_u8 = cpu
                    .gaussian_blur(&cpu_u8, 1.0, 5)
                    .map_err(|e| Error::FeatureError(format!("Gaussian blur failed: {}", e)))?;
                let slice = current_u8
                    .as_slice()
                    .map_err(|e| Error::FeatureError(format!("Failed to get slice: {}", e)))?;
                let data: Vec<f32> = slice.iter().map(|&v| v as f32 / 255.0).collect();
                let mut current_f32 = CpuTensor::from_vec(data, current_u8.shape)
                    .map_err(|e| Error::FeatureError(format!("Failed to create tensor: {}", e)))?;

                let k = cpu.akaze_contrast_k(&current_f32).unwrap_or(0.03);

                for o in 0..self.params.n_octaves {
                    for s in 0..self.params.n_sublevels {
                        let sigma = (2.0f32)
                            .powf((o as f32) + (s as f32) / (self.params.n_sublevels as f32));
                        let esigma = sigma * sigma / 2.0;
                        let dt = esigma - t;

                        if dt > 0.0 {
                            let n_steps = self.get_fed_steps(dt);
                            let step_tau = dt / (n_steps as f32);
                            for _ in 0..n_steps {
                                current_f32 =
                                    cpu.akaze_diffusion(&current_f32, k, step_tau).map_err(
                                        |e| Error::FeatureError(format!("Diffusion failed: {}", e)),
                                    )?;
                            }
                            t = esigma;
                        }

                        let (lx, ly, ldet) = cpu.akaze_derivatives(&current_f32).map_err(|e| {
                            Error::FeatureError(format!("Derivatives failed: {}", e))
                        })?;

                        evolution.push(EvolutionLevel {
                            image: current_f32.clone(),
                            lx,
                            ly,
                            ldet,
                            sigma,
                            octave: o,
                        });
                    }
                }
            }
            ComputeDevice::Mlx(_) => {
                return Err(Error::FeatureError(
                    "AKAZE evolution not implemented for MLX backend".into(),
                ));
            }
        }

        Ok(evolution)
    }

    fn get_fed_steps(&self, dt: f32) -> usize {
        let tau_max = 0.25;
        ((dt / tau_max).sqrt().ceil() as usize).max(1)
    }

    fn find_extrema(&self, evolution: &[EvolutionLevel]) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();
        let threshold = self.params.threshold;

        for curr in evolution {
            let (h, w) = curr.ldet.shape.hw();
            let det_slice = curr
                .ldet
                .as_slice()
                .map_err(|e| Error::FeatureError(format!("Failed to get det slice: {}", e)))?;

            let mut level_kps: Vec<KeyPoint> = (1..h - 1)
                .into_par_iter()
                .flat_map(|y| {
                    let mut row_kps = Vec::new();
                    for x in 1..w - 1 {
                        let val = det_slice[y * w + x];
                        if val > threshold {
                            let mut is_max = true;
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    if dx == 0 && dy == 0 {
                                        continue;
                                    }
                                    if det_slice
                                        [(y as i32 + dy) as usize * w + (x as i32 + dx) as usize]
                                        >= val
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
                                row_kps.push(
                                    KeyPoint::new(x as f64, y as f64)
                                        .with_size(curr.sigma as f64 * 2.0)
                                        .with_response(val as f64)
                                        .with_octave(curr.octave as i32),
                                );
                            }
                        }
                    }
                    row_kps
                })
                .collect();

            keypoints.append(&mut level_kps);
        }

        Ok(keypoints)
    }

    fn compute_descriptors(
        &self,
        evolution: &[EvolutionLevel],
        keypoints: &[KeyPoint],
    ) -> Result<Descriptors> {
        let mut descriptors = Vec::with_capacity(keypoints.len());

        for kp in keypoints {
            let level_idx = kp.octave as usize * self.params.n_sublevels
                + (kp.size.log2() as usize % self.params.n_sublevels);
            let level = &evolution[level_idx.min(evolution.len() - 1)];

            if let Some(desc) = self.compute_msurf_descriptor(level, kp)? {
                descriptors.push(desc);
            }
        }

        Ok(Descriptors { descriptors })
    }

    fn compute_msurf_descriptor(
        &self,
        level: &EvolutionLevel,
        kp: &KeyPoint,
    ) -> Result<Option<Descriptor>> {
        let (h, w) = level.image.shape.hw();
        let lx_slice = level
            .lx
            .as_slice()
            .map_err(|e| Error::FeatureError(format!("Failed to get lx slice: {}", e)))?;
        let ly_slice = level
            .ly
            .as_slice()
            .map_err(|e| Error::FeatureError(format!("Failed to get ly slice: {}", e)))?;

        let mut desc = vec![0u8; 64];
        let mut float_desc = [0.0f32; 64];

        let x_center = kp.x as f32;
        let y_center = kp.y as f32;
        let s = kp.size as f32;

        for i in 0..4 {
            for j in 0..4 {
                let mut sum_dx = 0.0;
                let mut sum_dy = 0.0;
                let mut sum_abs_dx = 0.0;
                let mut sum_abs_dy = 0.0;

                for x_sample in 0..5 {
                    for y_sample in 0..5 {
                        let rx = x_center + (i as f32 - 2.0 + x_sample as f32 * 0.2) * s;
                        let ry = y_center + (j as f32 - 2.0 + y_sample as f32 * 0.2) * s;

                        if rx >= 0.0 && rx < (w - 1) as f32 && ry >= 0.0 && ry < (h - 1) as f32 {
                            let idx = ry as usize * w + rx as usize;
                            let dx = lx_slice[idx];
                            let dy = ly_slice[idx];
                            sum_dx += dx;
                            sum_dy += dy;
                            sum_abs_dx += dx.abs();
                            sum_abs_dy += dy.abs();
                        }
                    }
                }

                let base = (i * 4 + j) * 4;
                float_desc[base] = sum_dx;
                float_desc[base + 1] = sum_dy;
                float_desc[base + 2] = sum_abs_dx;
                float_desc[base + 3] = sum_abs_dy;
            }
        }

        let mut norm_sq = 0.0;
        for &v in &float_desc {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt() + 1e-7;
        for i in 0..64 {
            desc[i] = ((float_desc[i] / norm) * 128.0 + 128.0).clamp(0.0, 255.0) as u8;
        }

        Ok(Some(Descriptor::new(desc, *kp)))
    }
}

/// Single evolution level in the AKAZE non-linear diffusion scale-space
///
/// Represents the state at one scale in the image pyramid, storing the current
/// image and its spatial derivatives needed for keypoint detection.
struct EvolutionLevel {
    /// Image at this evolution level (f32 format, normalized to [0,1])
    pub image: CpuTensor<f32>,
    /// Partial derivative in x direction (Sobel Lx)
    pub lx: CpuTensor<f32>,
    /// Partial derivative in y direction (Sobel Ly)
    pub ly: CpuTensor<f32>,
    /// Determinant of Hessian matrix at this level
    pub ldet: CpuTensor<f32>,
    /// Effective scale (sigma) parameter for this level
    pub sigma: f32,
    /// Octave (pyramid level) index
    pub octave: usize,
}

#[allow(dead_code)]
fn to_cpu_f32<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
    ctx: &ComputeDevice,
    tensor: &Tensor<f32, S>,
) -> crate::Result<CpuTensor<f32>> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::TensorToCpu;

    if let Some(gpu_storage) = tensor.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
        let gpu_tensor = Tensor {
            storage: gpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        match ctx {
            ComputeDevice::Gpu(gpu) => gpu_tensor
                .to_cpu_ctx(gpu)
                .map_err(|e| Error::FeatureError(format!("Download from GPU failed: {}", e))),
            _ => Err(Error::FeatureError(
                "GpuStorage with non-GPU context".into(),
            )),
        }
    } else if let Some(cpu_storage) = tensor.storage.as_any().downcast_ref::<CpuStorage<f32>>() {
        let cpu_tensor = Tensor {
            storage: cpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        Ok(cpu_tensor)
    } else {
        Err(Error::FeatureError("Unsupported storage type".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{storage::CpuStorage, TensorShape};
    use cv_hal::cpu::CpuBackend;

    fn create_test_image() -> Tensor<u8, CpuStorage<u8>> {
        let size = 128usize;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                if ((x / 16) + (y / 16)) % 2 == 0 {
                    data[y * size + x] = 255;
                }
            }
        }
        Tensor::from_vec(data, TensorShape::new(1, size, size)).unwrap()
    }

    #[test]
    fn test_akaze_detect() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let tensor = create_test_image();
        let akaze = Akaze::new(AkazeParams::default());
        let kps = akaze
            .detect_ctx(&device, &tensor)
            .expect("detect_ctx failed");
        println!("Detected {} AKAZE keypoints", kps.len());
        assert!(kps.len() > 0);
    }
}
