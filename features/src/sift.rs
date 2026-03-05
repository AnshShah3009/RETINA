#![allow(deprecated)]

use cv_core::{storage::Storage, Descriptors, Error, KeyPoint, KeyPoints, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::TensorToCpu;
use nalgebra::{Matrix3, Vector3};

/// SIFT (Scale-Invariant Feature Transform) feature detector and descriptor
///
/// Implements Lowe's Scale-Invariant Feature Transform for robust feature detection,
/// extraction, and matching. SIFT features are invariant to scale, rotation, and
/// illumination changes.
///
/// # Algorithm Overview
///
/// SIFT builds a Gaussian scale-space pyramid and detects extrema in the
/// Difference-of-Gaussians (DoG) pyramid. Each detected keypoint is refined
/// with sub-pixel accuracy and assigned a descriptor based on local gradients.
///
/// # Parameters
///
/// * `n_octaves` - Number of scale octaves in the pyramid (typically 4-5)
/// * `n_layers` - Number of blur levels per octave (typically 3-5)
/// * `sigma` - Initial Gaussian blur standard deviation
/// * `contrast_threshold` - Minimum contrast for keypoint acceptance
/// * `edge_threshold` - Ratio threshold to reject edge-like features
pub struct Sift {
    /// Number of octaves (pyramid levels) in the scale space
    pub n_octaves: usize,
    /// Number of Gaussian blur layers per octave
    pub n_layers: usize,
    /// Initial Gaussian blur sigma parameter
    pub sigma: f32,
    /// Minimum contrast threshold for keypoint selection (0-1 range)
    pub contrast_threshold: f32,
    /// Edge response threshold for rejecting edge features (typically 10-15)
    pub edge_threshold: f32,
}

impl Default for Sift {
    /// Default SIFT parameters with 4 octaves, 3 layers, and balanced thresholds
    ///
    /// Uses conservative settings: sigma=1.6, contrast_threshold=0.04, edge_threshold=10.0
    fn default() -> Self {
        Self {
            n_octaves: 4,
            n_layers: 3,
            sigma: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
        }
    }
}

impl Sift {
    /// Create a new SIFT detector with default parameters
    ///
    /// Equivalent to `Sift::default()`. Provides balanced feature detection
    /// with 4 octaves and 3 layers per octave.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build SIFT Gaussian scale-space pyramid
    ///
    /// Constructs a multi-octave Gaussian scale-space by iteratively blurring
    /// and downsampling the image. Each octave contains multiple blur levels
    /// at progressively increasing sigma values.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device (CPU or GPU) for Gaussian filtering
    /// * `image` - Input grayscale image (u8 format, 0-255 range)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Vec<Tensor<u8, S>>>>` - Pyramid[octave][layer] with `n_octaves` x `n_layers+3` layers
    /// * `Err(FeatureError)` - If blur or resize operations fail
    ///
    /// # Errors
    ///
    /// May return errors if:
    /// - GPU operations fail (memory transfer, computation)
    /// - Image dimensions are invalid or too small
    /// - Tensor operations fail
    ///
    /// # Scale Relationship
    ///
    /// Between octaves, the image is downsampled 2x. Within each octave,
    /// sigma increases by factor `k = 2^(1/n_layers)`.
    pub fn build_scale_space<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> crate::Result<Vec<Vec<Tensor<u8, S>>>> {
        let mut pyramid = Vec::with_capacity(self.n_octaves);
        let mut current_base = image.clone();

        for octave in 0..self.n_octaves {
            let mut layers = Vec::with_capacity(self.n_layers + 3);

            let mut sig = self.sigma;
            let base_layer = ctx
                .gaussian_blur(&current_base, sig, 7)
                .map_err(|e| Error::FeatureError(format!("Gaussian blur failed: {}", e)))?;
            layers.push(base_layer);

            let k = 2.0f32.powf(1.0 / self.n_layers as f32);
            for _ in 1..(self.n_layers + 3) {
                let prev = layers
                    .last()
                    .ok_or_else(|| Error::FeatureError("Layer stack is empty".into()))?;
                let sig_prev = sig;
                sig *= k;
                let sig_total = (sig * sig - sig_prev * sig_prev).sqrt();

                let blurred = ctx
                    .gaussian_blur(prev, sig_total, 7)
                    .map_err(|e| Error::FeatureError(format!("Gaussian blur failed: {}", e)))?;
                layers.push(blurred);
            }
            pyramid.push(layers);

            if octave < self.n_octaves - 1 {
                let to_sample = &pyramid[octave][self.n_layers];
                let (h, w) = to_sample.shape.hw();
                current_base = ctx
                    .resize(to_sample, (w / 2, h / 2))
                    .map_err(|e| Error::FeatureError(format!("Resize failed: {}", e)))?;
            }
        }
        Ok(pyramid)
    }

    /// Compute Difference-of-Gaussians (DoG) pyramid for keypoint detection
    ///
    /// Computes the DoG by subtracting successive Gaussian layers in each octave.
    /// The DoG pyramid is used for detecting local extrema that correspond to
    /// potential keypoints.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device for tensor operations
    /// * `gaussian_pyramid` - Gaussian pyramid from [`build_scale_space`]
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Vec<Tensor<f32, CpuStorage<f32>>>>>` - DoG pyramid[octave][layer-1]
    /// * `Err(FeatureError)` - If tensor operations fail
    ///
    /// # Output Shape
    ///
    /// For input with n_layers per octave, produces (n_layers-1) DoG layers per octave.
    /// All DoG tensors are converted to f32 CPU storage for subsequent processing.
    pub fn compute_dog<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        gaussian_pyramid: &[Vec<Tensor<u8, S>>],
    ) -> crate::Result<Vec<Vec<Tensor<f32, cv_core::storage::CpuStorage<f32>>>>> {
        let mut dog_pyramid = Vec::with_capacity(gaussian_pyramid.len());

        for octave_layers in gaussian_pyramid {
            let mut dog_layers = Vec::with_capacity(octave_layers.len() - 1);
            for i in 0..(octave_layers.len() - 1) {
                let a_f32 = convert_to_f32_cpu(ctx, &octave_layers[i + 1])?;
                let b_f32 = convert_to_f32_cpu(ctx, &octave_layers[i])?;

                let diff = ctx
                    .subtract(&a_f32, &b_f32)
                    .map_err(|e| Error::FeatureError(format!("Subtraction failed: {}", e)))?;
                dog_layers.push(diff);
            }
            dog_pyramid.push(dog_layers);
        }
        Ok(dog_pyramid)
    }

    /// Detect SIFT keypoints with sub-pixel refinement
    ///
    /// Complete SIFT keypoint detection pipeline:
    /// 1. Build Gaussian scale-space
    /// 2. Compute Difference-of-Gaussians
    /// 3. Find DoG extrema
    /// 4. Refine locations to sub-pixel accuracy
    /// 5. Filter by contrast and edge-response
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device (CPU or GPU)
    /// * `image` - Input grayscale image (u8, 0-255 range)
    ///
    /// # Returns
    ///
    /// * `Ok((KeyPoints, Gaussian Pyramid))` - Detected keypoints and the scale-space for descriptor computation
    /// * `Err(FeatureError)` - If any stage fails
    ///
    /// # Errors
    ///
    /// May fail if:
    /// - Scale-space construction fails
    /// - DoG computation fails
    /// - Extrema detection fails
    /// - Sub-pixel refinement fails
    ///
    /// # Output Properties
    ///
    /// Each keypoint includes:
    /// - Position (x, y)
    /// - Scale (size) derived from octave and layer
    /// - Octave and layer indices
    /// - Contrast-weighted response value
    pub fn detect_and_refine<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> crate::Result<(KeyPoints, Vec<Vec<Tensor<u8, S>>>)> {
        let gaussian_pyramid = self.build_scale_space(ctx, image)?;
        let dog_pyramid = self.compute_dog(ctx, &gaussian_pyramid)?;

        let mut all_kps = Vec::new();

        for octave in 0..self.n_octaves {
            let dog_layers = &dog_pyramid[octave];
            let (h, w) = dog_layers[0].shape.hw();

            for s in 1..self.n_layers + 1 {
                let prev = &dog_layers[s - 1];
                let curr = &dog_layers[s];
                let next = &dog_layers[s + 1];

                let candidates = ctx
                    .sift_extrema(
                        prev,
                        curr,
                        next,
                        self.contrast_threshold,
                        self.edge_threshold,
                    )
                    .map_err(|e| {
                        Error::FeatureError(format!("SIFT extrema detection failed: {}", e))
                    })?;
                let cand_slice = match candidates.storage.as_slice() {
                    Some(slice) => slice,
                    None => {
                        eprintln!("Warning: Failed to get candidate slice");
                        continue;
                    }
                };

                for y in 1..h - 1 {
                    for x in 1..w - 1 {
                        if cand_slice[y * w + x] > 0 {
                            if let Some(refined) = refine_point(
                                dog_layers,
                                s,
                                x,
                                y,
                                self.n_layers,
                                self.contrast_threshold,
                            ) {
                                let scale = 2.0f64.powi(octave as i32);
                                let kp = KeyPoint::new(refined.x * scale, refined.y * scale)
                                    .with_size(refined.size * scale)
                                    .with_octave(octave as i32)
                                    .with_response(refined.response);
                                all_kps.push(kp);
                            }
                        }
                    }
                }
            }
        }

        Ok((KeyPoints { keypoints: all_kps }, gaussian_pyramid))
    }

    /// Detect SIFT keypoints and compute 128-dimensional descriptors
    ///
    /// Complete SIFT feature extraction pipeline:
    /// 1. Detect and refine keypoints using DoG extrema
    /// 2. Compute 128-dimensional orientation histograms for each keypoint
    /// 3. Normalize descriptors for rotation invariance
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compute device (CPU or GPU)
    /// * `image` - Input grayscale image (u8, values 0-255)
    ///
    /// # Returns
    ///
    /// * `Ok((KeyPoints, Descriptors))` - Detected features and their 128-element descriptors
    /// * `Err(FeatureError)` - If detection or descriptor computation fails
    ///
    /// # Errors
    ///
    /// May fail if:
    /// - Keypoint detection fails (see [`detect_and_refine`])
    /// - Descriptor computation fails (GPU/CPU operations)
    /// - Device operations fail
    ///
    /// # Descriptor Format
    ///
    /// Each descriptor is 128 bytes (u8 values 0-255), representing a normalized
    /// 4x4x8 histogram of image gradients around the keypoint. The descriptor is
    /// invariant to rotation and scale within the detected feature scale.
    ///
    /// # Performance Note
    ///
    /// Uses GPU when available (faster), falls back to CPU. MLX backend is not
    /// currently supported for descriptor computation.
    pub fn detect_and_compute<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> crate::Result<(KeyPoints, Descriptors)> {
        let (keypoints, gaussian_pyramid) = self.detect_and_refine(ctx, image)?;

        let mut all_descriptors = Vec::new();
        let mut valid_keypoints = Vec::new();

        for octave in 0..self.n_octaves {
            let octave_kps: Vec<KeyPoint> = keypoints
                .keypoints
                .iter()
                .filter(|kp| kp.octave == octave as i32)
                .map(|kp| {
                    let scale = 1.0 / 2.0f64.powi(octave as i32);
                    KeyPoint::new(kp.x * scale, kp.y * scale)
                        .with_size(kp.size * scale)
                        .with_octave(octave as i32)
                        .with_response(kp.response)
                })
                .collect();

            if octave_kps.is_empty() {
                continue;
            }

            match ctx {
                ComputeDevice::Cpu(cpu) => {
                    let octave_img = convert_to_f32_cpu(ctx, &gaussian_pyramid[octave][0])?;
                    let descs = cpu
                        .compute_sift_descriptors(
                            &octave_img,
                            &KeyPoints {
                                keypoints: octave_kps,
                            },
                        )
                        .map_err(|e| {
                            Error::FeatureError(format!(
                                "SIFT descriptor computation failed: {}",
                                e
                            ))
                        })?;
                    for d in descs.descriptors {
                        let mut restored_kp = d.keypoint;
                        let scale = 2.0f64.powi(octave as i32);
                        restored_kp.x *= scale;
                        restored_kp.y *= scale;
                        restored_kp.size *= scale;
                        all_descriptors.push(cv_core::Descriptor::new(d.data, restored_kp));
                        valid_keypoints.push(restored_kp);
                    }
                }
                ComputeDevice::Gpu(gpu) => {
                    use cv_hal::tensor_ext::TensorToGpu;
                    let f32_cpu = convert_to_f32_cpu(ctx, &gaussian_pyramid[octave][0])?;
                    let octave_img_gpu = f32_cpu.to_gpu_ctx(gpu).map_err(|e| {
                        Error::FeatureError(format!("Failed to upload image to GPU: {}", e))
                    })?;
                    let descs = gpu
                        .compute_sift_descriptors(
                            &octave_img_gpu,
                            &KeyPoints {
                                keypoints: octave_kps,
                            },
                        )
                        .map_err(|e| {
                            Error::FeatureError(format!(
                                "GPU SIFT descriptor computation failed: {}",
                                e
                            ))
                        })?;
                    for d in descs.descriptors {
                        let mut restored_kp = d.keypoint;
                        let scale = 2.0f64.powi(octave as i32);
                        restored_kp.x *= scale;
                        restored_kp.y *= scale;
                        restored_kp.size *= scale;
                        all_descriptors.push(cv_core::Descriptor::new(d.data, restored_kp));
                        valid_keypoints.push(restored_kp);
                    }
                }
                ComputeDevice::Mlx(_) => {
                    eprintln!("Warning: MLX backend not supported for SIFT descriptors, skipping descriptor computation");
                    break;
                }
            }
        }

        Ok((
            KeyPoints {
                keypoints: valid_keypoints,
            },
            Descriptors {
                descriptors: all_descriptors,
            },
        ))
    }

    /// Detect SIFT keypoints using the best available device
    ///
    /// High-level convenience function that automatically selects the best compute
    /// device (GPU or CPU) for keypoint detection via the runtime scheduler.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image (u8, 0-255 range)
    ///
    /// # Returns
    ///
    /// * `Ok(KeyPoints)` - Detected SIFT features
    /// * `Err(FeatureError)` - If device selection or detection fails
    ///
    /// # Errors
    ///
    /// May fail if:
    /// - Device selection fails (returns CPU fallback)
    /// - Keypoint detection fails
    pub fn detect<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        image: &Tensor<u8, S>,
    ) -> crate::Result<KeyPoints> {
        let runner = match cv_runtime::scheduler()
            .and_then(|s| s.best_gpu_or_cpu_for(cv_runtime::orchestrator::WorkloadHint::Throughput))
        {
            Ok(group) => cv_runtime::RuntimeRunner::Group(group),
            Err(_) => cv_runtime::default_runner().unwrap_or_else(|_| {
                cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
            }),
        };
        match runner.device() {
            Ok(device) => {
                let (kps, _) = self.detect_and_refine(&device, image)?;
                Ok(kps)
            }
            Err(e) => Err(Error::FeatureError(format!("Failed to get device: {}", e))),
        }
    }

    /// Detect keypoints and compute descriptors using the best available device
    ///
    /// High-level convenience function that automatically selects the best compute
    /// device (GPU or CPU) for the full SIFT pipeline via the runtime scheduler.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image (u8 format, 0-255 value range)
    ///
    /// # Returns
    ///
    /// * `Ok((KeyPoints, Descriptors))` - Detected features and 128-dimensional descriptors
    /// * `Err(FeatureError)` - If device selection or extraction fails
    ///
    /// # Errors
    ///
    /// May fail if:
    /// - Runtime scheduler fails to find a device (falls back to CPU)
    /// - Feature detection fails
    /// - Descriptor computation fails
    ///
    /// # Performance Note
    ///
    /// This method uses the cv-runtime scheduler to select between GPU and CPU
    /// based on current system load and availability.
    pub fn compute<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        image: &Tensor<u8, S>,
    ) -> crate::Result<(KeyPoints, Descriptors)> {
        let runner = match cv_runtime::scheduler()
            .and_then(|s| s.best_gpu_or_cpu_for(cv_runtime::orchestrator::WorkloadHint::Throughput))
        {
            Ok(group) => cv_runtime::RuntimeRunner::Group(group),
            Err(_) => cv_runtime::default_runner().unwrap_or_else(|_| {
                cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
            }),
        };
        match runner.device() {
            Ok(device) => self.detect_and_compute(&device, image),
            Err(e) => Err(Error::FeatureError(format!("Failed to get device: {}", e))),
        }
    }
}

fn refine_point(
    dog_layers: &[Tensor<f32, cv_core::storage::CpuStorage<f32>>],
    s: usize,
    x: usize,
    y: usize,
    n_layers: usize,
    contrast_threshold: f32,
) -> Option<KeyPoint> {
    // Validate input bounds before accessing arrays
    let (h, w) = dog_layers[0].shape.hw();
    if x < 1 || x >= w - 1 || y < 1 || y >= h - 1 || s < 1 || s >= n_layers {
        return None;
    }

    let mut curr_s = s;
    let mut curr_x = x;
    let mut curr_y = y;
    let mut offset = Vector3::zeros();

    for _ in 0..5 {
        let (h, w) = dog_layers[0].shape.hw();

        let d_s = dog_layers[curr_s].storage.as_slice()?;
        let d_prev = dog_layers[curr_s - 1].storage.as_slice()?;
        let d_next = dog_layers[curr_s + 1].storage.as_slice()?;

        let dx = (d_s[curr_y * w + curr_x + 1] - d_s[curr_y * w + curr_x - 1]) / 2.0;
        let dy = (d_s[(curr_y + 1) * w + curr_x] - d_s[(curr_y - 1) * w + curr_x]) / 2.0;
        let ds = (d_next[curr_y * w + curr_x] - d_prev[curr_y * w + curr_x]) / 2.0;
        let grad = Vector3::new(dx, dy, ds);

        let v = d_s[curr_y * w + curr_x];
        let dxx = d_s[curr_y * w + curr_x + 1] + d_s[curr_y * w + curr_x - 1] - 2.0 * v;
        let dyy = d_s[(curr_y + 1) * w + curr_x] + d_s[(curr_y - 1) * w + curr_x] - 2.0 * v;
        let dss = d_next[curr_y * w + curr_x] + d_prev[curr_y * w + curr_x] - 2.0 * v;

        let dxy = (d_s[(curr_y + 1) * w + curr_x + 1]
            - d_s[(curr_y + 1) * w + curr_x - 1]
            - d_s[(curr_y - 1) * w + curr_x + 1]
            + d_s[(curr_y - 1) * w + curr_x - 1])
            / 4.0;
        let dxs = (d_next[curr_y * w + curr_x + 1]
            - d_next[curr_y * w + curr_x - 1]
            - d_prev[curr_y * w + curr_x + 1]
            + d_prev[curr_y * w + curr_x - 1])
            / 4.0;
        let dys = (d_next[(curr_y + 1) * w + curr_x]
            - d_next[(curr_y - 1) * w + curr_x]
            - d_prev[(curr_y + 1) * w + curr_x]
            + d_prev[(curr_y - 1) * w + curr_x])
            / 4.0;

        let hessian = Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

        if let Some(inv_h) = hessian.try_inverse() {
            offset = -inv_h * grad;
        } else {
            return None;
        }

        if offset.x.abs() < 0.5 && offset.y.abs() < 0.5 && offset.z.abs() < 0.5 {
            break;
        }

        curr_x = (curr_x as f32 + offset.x.round()) as usize;
        curr_y = (curr_y as f32 + offset.y.round()) as usize;
        curr_s = (curr_s as f32 + offset.z.round()) as usize;

        if curr_s < 1
            || curr_s > n_layers
            || curr_x < 1
            || curr_x >= w - 1
            || curr_y < 1
            || curr_y >= h - 1
        {
            return None;
        }
    }

    let d_final = match dog_layers[curr_s].storage.as_slice() {
        Some(slice) => slice[curr_y * dog_layers[0].shape.width + curr_x],
        None => return None,
    };
    let contrast = d_final + 0.5 * offset.dot(&Vector3::new(0.0, 0.0, 0.0));
    if contrast.abs() < contrast_threshold / n_layers as f32 {
        return None;
    }

    Some(
        KeyPoint::new(
            curr_x as f64 + offset.x as f64,
            curr_y as f64 + offset.y as f64,
        )
        .with_response(contrast.abs() as f64)
        .with_size(1.6 * 2.0f64.powf((curr_s as f64 + offset.z as f64) / n_layers as f64)),
    )
}

fn convert_to_f32_cpu<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    ctx: &ComputeDevice,
    input: &Tensor<u8, S>,
) -> crate::Result<Tensor<f32, cv_core::storage::CpuStorage<f32>>> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    let cpu_u8 = if let Some(gpu_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>()
    {
        let input_gpu = Tensor {
            storage: gpu_storage.clone(),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        };
        let gpu_ctx = match ctx {
            ComputeDevice::Gpu(g) => g,
            _ => {
                return Err(Error::FeatureError(
                    "GpuStorage requires GPU context".into(),
                ))
            }
        };
        input_gpu
            .to_cpu_ctx(gpu_ctx)
            .map_err(|e| Error::FeatureError(format!("GPU download failed: {}", e)))?
    } else if let Some(cpu_storage) = input.storage.as_any().downcast_ref::<CpuStorage<u8>>() {
        let input_cpu = Tensor {
            storage: cpu_storage.clone(),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        };
        input_cpu.clone()
    } else {
        return Err(Error::FeatureError("Unsupported storage type".into()));
    };

    let slice_u8 = cpu_u8
        .storage
        .as_slice()
        .ok_or_else(|| Error::FeatureError("Failed to get u8 slice".into()))?;
    let data_f32: Vec<f32> = slice_u8.iter().map(|&v| v as f32 / 255.0).collect();

    Tensor::from_vec(data_f32, input.shape)
        .map_err(|e| Error::FeatureError(format!("Failed to create f32 tensor: {}", e)))
}

/// Detect SIFT keypoints using an optional GPU compute context.
///
/// Falls back to CPU detection when no GPU context is available.
pub fn sift_detect_ctx<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    ctx: &ComputeDevice,
    image: &Tensor<u8, S>,
    params: &Sift,
) -> crate::Result<KeyPoints> {
    let (kps, _) = params.detect_and_refine(ctx, image)?;
    Ok(kps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{storage::CpuStorage, Result, TensorShape};
    use cv_hal::cpu::CpuBackend;

    fn create_test_image() -> Result<Tensor<u8, CpuStorage<u8>>> {
        let size = 128usize;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                if ((x / 16) + (y / 16)) % 2 == 0 {
                    data[y * size + x] = 255;
                }
            }
        }
        Tensor::from_vec(data, TensorShape::new(1, size, size))
            .map_err(|e| Error::FeatureError(format!("Failed to create test image: {}", e)))
    }

    #[test]
    fn test_sift_detect_and_refine() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let tensor = create_test_image().unwrap();
        let sift = Sift::new();
        let (kps, _) = sift.detect_and_refine(&device, &tensor).unwrap();
        println!("Detected {} refined SIFT keypoints", kps.len());
        assert!(kps.len() > 0);
    }

    #[test]
    fn test_sift_full_pipeline() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let tensor = create_test_image().unwrap();
        let sift = Sift::new();
        let (kps, descs) = sift.detect_and_compute(&device, &tensor).unwrap();
        println!(
            "Extracted {} SIFT keypoints and {} descriptors",
            kps.len(),
            descs.len()
        );
        assert_eq!(kps.len(), descs.len());
        if !descs.descriptors.is_empty() {
            assert_eq!(descs.descriptors[0].data.len(), 128);
        }
    }
}
