//! ORB (Oriented FAST and Rotated BRIEF) implementation
//!
//! ORB combines the FAST keypoint detector with a modified BRIEF descriptor
//! that includes orientation information for rotation invariance.

#![allow(deprecated)]

use crate::descriptor::{Descriptor, DescriptorExtractor, Descriptors};
use crate::fast::{self, fast_detect};
use cv_core::{storage::Storage, Error, KeyPoint, KeyPoints, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorCast, TensorToCpu};
use cv_imgproc::convolve::gaussian_blur;
use image::GrayImage;
use rayon::prelude::*;

// Three parameters of a spec-compliant ORB are not implemented and are not
// carried here: `first_level` (the pyramid's base octave), `edge_threshold`, and
// `wta_k` (test-all hashing). All three were previously present as dead fields
// carrying `#[allow(dead_code)]`, which read as supported configuration that
// silently did nothing. See docs/performance.md for the measurement of the
// hashing variant, which is correct in principle and a regression in practice
// under Hamming-distance matching.
//
// Learned rBRIEF pattern from OpenCV's ORB implementation (`bit_pattern_31_`).
/// 256 test pairs, each with 4 values: (x1, y1, x2, y2) relative to patch center.
/// Selected via a greedy algorithm to maximize descriptor variance and minimize
/// correlation between bits (see Rublee et al., "ORB: an efficient alternative
/// to SIFT or SURF", ICCV 2011).
#[rustfmt::skip]
const ORB_PATTERN: [[i32; 4]; 256] = [
    [8,-3,9,5], [4,2,7,-12], [-11,9,-8,2], [7,-12,12,-13],
    [2,-13,2,12], [1,-7,1,6], [-2,-10,-2,-4], [-13,-13,-11,-8],
    [-13,-3,-12,-9], [10,4,11,9], [-13,-8,-8,-9], [-11,7,-9,12],
    [7,7,12,6], [-4,-5,-3,0], [-13,2,-12,-3], [-9,0,-7,5],
    [12,-6,12,-1], [-3,6,-2,12], [-6,-13,-4,-8], [11,-13,12,-8],
    [4,7,5,1], [5,-3,10,-3], [3,-7,6,12], [-8,-7,-6,-2],
    [-2,11,-1,-10], [-13,12,-8,10], [-7,3,-5,-3], [-4,2,-3,7],
    [-10,-12,-6,11], [5,-12,6,-7], [5,-6,7,-1], [1,0,4,-5],
    [9,11,11,-13], [4,7,4,12], [2,-1,4,4], [-4,-12,-2,7],
    [-8,-5,-7,-10], [4,11,9,12], [0,-8,1,-13], [-13,-2,-8,2],
    [-3,-2,-2,3], [-6,9,-4,-9], [8,12,10,7], [0,9,1,3],
    [7,-5,11,-10], [-13,-6,-11,0], [10,7,12,1], [-6,-3,-6,12],
    [10,-9,12,-4], [-13,8,-8,-12], [-13,0,-8,-4], [3,3,7,8],
    [5,7,10,-7], [-1,7,1,-12], [3,-10,5,6], [2,-4,3,-10],
    [-13,0,-13,5], [-13,-7,-12,12], [-13,3,-11,8], [-7,12,-4,7],
    [6,-10,12,8], [-9,-1,-7,-6], [-2,-5,0,12], [-12,5,-7,5],
    [3,-10,8,-13], [-7,-7,-4,5], [-3,-2,-1,-7], [2,9,5,-11],
    [-11,-13,-5,-13], [-1,6,0,-1], [5,-3,5,2], [-4,-13,-4,12],
    [-9,-6,-9,6], [-12,-10,-8,-4], [10,2,12,-3], [7,12,12,12],
    [-7,-13,-6,5], [-4,9,-3,4], [7,-1,12,2], [-7,6,-5,1],
    [-13,11,-12,5], [-3,7,-2,-6], [7,-8,12,-7], [-13,-7,-11,-12],
    [1,-3,12,12], [2,-6,3,0], [-4,3,-2,-13], [-1,-13,1,9],
    [7,1,8,-6], [1,-1,3,12], [9,1,12,6], [-1,-9,-1,3],
    [-13,-13,-10,5], [7,7,10,12], [12,-5,12,9], [6,3,7,11],
    [5,-13,6,10], [2,-12,2,3], [3,8,4,-6], [2,6,12,-13],
    [9,-12,10,3], [-8,4,-7,9], [-11,12,-4,-6], [1,12,2,-8],
    [6,-9,7,-4], [2,3,3,-2], [6,3,11,0], [3,-3,8,-8],
    [7,8,9,3], [-11,-5,-6,-4], [-10,11,-5,10], [-5,-8,-3,12],
    [-10,5,-9,0], [8,-1,12,-6], [4,-6,6,-11], [-10,12,-8,7],
    [4,-2,6,7], [-2,0,-2,12], [-5,-8,-5,2], [7,-6,10,12],
    [-9,-13,-8,-8], [-5,-13,-5,-2], [8,-8,9,-13], [-9,-11,-9,0],
    [1,-8,1,-2], [7,-4,9,1], [-2,1,-1,-4], [11,-6,12,-11],
    [-12,-9,-6,4], [3,7,7,12], [5,5,10,8], [0,-4,2,8],
    [-9,12,-5,-13], [0,7,2,12], [-1,2,1,7], [5,11,7,-9],
    [3,5,6,-8], [-13,-4,-8,9], [-5,9,-3,-3], [-4,-7,-3,-12],
    [6,5,8,0], [-7,6,-6,12], [-13,6,-5,-2], [1,-10,3,10],
    [4,1,8,-4], [-2,-2,2,-13], [2,-12,12,12], [-2,-13,0,-6],
    [4,1,9,3], [-6,-10,-3,-5], [-3,-13,-1,1], [7,5,12,-11],
    [4,-2,5,-7], [-13,9,-9,-5], [7,1,8,6], [7,-8,7,6],
    [-7,-4,-7,1], [-8,11,-7,-8], [-13,6,-12,-8], [2,4,3,9],
    [10,-5,12,3], [-6,-5,-6,7], [8,-3,9,-8], [2,-12,2,8],
    [-11,-2,-10,3], [-12,-13,-7,-9], [-11,0,-10,-5], [5,-3,11,8],
    [-2,-13,-1,12], [-1,-8,0,9], [-13,-11,-12,-5], [-10,-2,-10,11],
    [-3,9,-2,-13], [2,-3,3,2], [-9,-13,-4,0], [-4,6,-3,-10],
    [-4,12,-2,-7], [-6,-11,-4,9], [6,-3,6,11], [-13,11,-5,5],
    [11,11,12,6], [7,-5,12,-2], [-1,12,0,7], [-4,-8,-3,-2],
    [-7,1,-6,7], [-13,-12,-8,-13], [-7,-2,-6,-8], [-8,5,-6,-9],
    [-5,-1,-4,5], [-13,7,-8,10], [1,5,5,-13], [1,0,10,-13],
    [9,12,10,-1], [5,-8,10,-9], [-1,11,1,-13], [-9,-3,-6,2],
    [-1,-10,1,12], [-13,1,-8,-10], [8,-11,10,-6], [2,-13,3,-6],
    [7,-13,12,-9], [-10,-10,-5,-7], [-10,-8,-8,-13], [4,-6,8,5],
    [3,12,8,-13], [-4,2,-3,-3], [5,-13,10,-12], [4,-13,5,-1],
    [-9,9,-4,3], [0,3,3,-9], [-12,1,-6,1], [3,2,4,-8],
    [-10,-10,-10,9], [8,-13,12,12], [-8,-12,-6,-5], [2,2,3,7],
    [10,6,11,-8], [6,8,8,-12], [-7,10,-6,5], [-3,-9,-3,9],
    [-1,-13,-1,5], [-3,-7,-3,4], [-8,-2,-8,3], [4,2,12,12],
    [2,-5,3,11], [6,-9,11,-13], [3,-1,7,12], [11,-1,12,4],
    [-3,0,-3,6], [4,-11,4,12], [2,-4,2,1], [-10,-6,-8,1],
    [-13,7,-11,1], [-13,12,-11,-13], [6,0,11,-13], [0,-1,1,4],
    [-13,3,-9,-2], [-9,8,-6,-3], [-13,-6,-8,-2], [5,-9,8,10],
    [2,7,3,-9], [-1,-6,-1,-1], [9,5,11,-2], [11,-3,12,-8],
    [3,0,3,5], [-1,4,0,10], [3,-6,4,5], [-13,0,-10,5],
    [5,8,12,11], [8,9,9,-6], [7,-4,8,-12], [-10,4,-10,9],
    [7,3,12,4], [9,-7,10,-2], [7,0,12,-2], [-1,-6,0,-11],
];

/// ORB feature detector and descriptor
pub struct Orb {
    n_features: usize,
    scale_factor: f32,
    n_levels: usize,
    score_type: ScoreType,
    /// Measure the BRIEF pattern at each keypoint's own pyramid scale.
    ///
    /// Measured, and not simply better: it helps the low-texture sequences
    /// (TUM fr1_desk 23 -> 24 of 40, ETH3D electro 3/5 -> 4/5 and 3/12 -> 4/12,
    /// with landmark counts roughly doubling in each case) and costs the
    /// well-exposed ones (ETH3D courtyard 7/8 -> 6/8, TUM fr1_xyz camera-centre
    /// error 1.36 -> 2.03 cm). Every gain is on the sequences this work has been
    /// trying to improve and every loss is on one that already worked, so it is
    /// opt-in rather than the default. See docs/performance.md.
    pub scale_aware_descriptor: bool,
    patch_size: i32,
    fast_threshold: u8,
}

/// Keypoint scoring method used to rank and filter ORB keypoints.
#[derive(Clone, Copy)]
pub enum ScoreType {
    /// Rank keypoints by Harris corner response.
    Harris,
    /// Rank keypoints by FAST corner response.
    Fast,
}

impl Default for Orb {
    fn default() -> Self {
        Self {
            n_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            score_type: ScoreType::Harris,
            scale_aware_descriptor: false,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl Orb {
    /// Create a new ORB detector/extractor with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of keypoints to retain.
    /// Measure the BRIEF pattern at each keypoint's own pyramid scale.
    ///
    /// See [`Orb::scale_aware_descriptor`] for the measurement behind this.
    pub fn with_scale_aware_descriptor(mut self, scale_aware: bool) -> Self {
        self.scale_aware_descriptor = scale_aware;
        self
    }

    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = n;
        self
    }

    /// Set the number of scale pyramid levels.
    pub fn with_n_levels(mut self, n: usize) -> Self {
        self.n_levels = n;
        self
    }

    /// Set the scale factor between pyramid levels (must be > 1.0).
    pub fn with_scale_factor(mut self, factor: f32) -> Self {
        self.scale_factor = factor;
        self
    }

    /// Set the FAST detection threshold (lower = more keypoints).
    pub fn with_fast_threshold(mut self, threshold: u8) -> Self {
        self.fast_threshold = threshold;
        self
    }

    /// Detect keypoints using FAST at multiple scales
    pub fn detect(&self, image: &GrayImage) -> KeyPoints {
        let mut all_keypoints = Vec::new();
        let mut scale = 1.0f32;

        for level in 0..self.n_levels {
            let scaled = if level == 0 {
                image.clone()
            } else {
                scale_image(image, scale)
            };

            // Do not truncate in raster order before non-max suppression and
            // response ranking: on busy images that kept only the top-row
            // candidates. Collect every candidate and let the final
            // `truncate(self.n_features)` after the response sort enforce the
            // keypoint budget.
            let kps = fast_detect(&scaled, self.fast_threshold, usize::MAX);

            // Bug 5 fix: apply non-maximum suppression to FAST keypoints
            let kps = fast::non_max_suppression(kps, &scaled, self.fast_threshold);

            // Bug 7 fix: compute FAST corner score so keypoints have a real response
            // Bug 4 fix: if Harris scoring is selected, re-score with Harris response.
            // The response of each keypoint depends only on the level image, so
            // this maps in parallel and preserves the input order.
            let candidates = kps.keypoints;
            let scored_kps: Vec<KeyPoint> = candidates
                .par_iter()
                .map(|kp| {
                    let response = match self.score_type {
                        ScoreType::Harris => {
                            compute_harris_response(&scaled, kp.x as i32, kp.y as i32)
                        }
                        ScoreType::Fast => fast::corner_score(
                            &scaled,
                            kp.x as i32,
                            kp.y as i32,
                            self.fast_threshold,
                        ) as f64,
                    };

                    KeyPoint::new(kp.x * scale as f64, kp.y * scale as f64)
                        .with_size(self.patch_size as f64 * scale as f64)
                        .with_octave(level as i32)
                        .with_response(response)
                })
                .collect();

            all_keypoints.extend(scored_kps);

            scale *= self.scale_factor;
        }

        all_keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_keypoints.truncate(self.n_features);

        KeyPoints {
            keypoints: all_keypoints,
        }
    }

    /// Detect keypoints using FAST at multiple scales with acceleration
    pub fn detect_ctx<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> KeyPoints {
        let mut all_keypoints = Vec::new();
        let mut scale = 1.0f32;

        for level in 0..self.n_levels {
            let scaled_f32 = if level == 0 {
                convert_to_f32_cpu(ctx, image).expect("Failed to convert image to f32")
            } else {
                let new_w = (image.shape.width as f32 / scale) as usize;
                let new_h = (image.shape.height as f32 / scale) as usize;
                let img_f32 =
                    convert_to_f32_cpu(ctx, image).expect("Failed to convert image to f32");
                let resized = ctx.resize(&img_f32, (new_w, new_h)).unwrap();

                let cpu_u8 = resized.to_cpu().unwrap();
                let img_h_w = cpu_u8.shape;
                let data: Vec<f32> = cpu_u8.storage.as_slice().unwrap().to_vec();
                let tensor_f32: Tensor<f32, cv_core::storage::CpuStorage<f32>> =
                    Tensor::from_vec(data, img_h_w).unwrap();
                tensor_f32
            };

            // `scaled_f32` is normalized to [0, 1] (see `convert_to_f32_cpu`),
            // so the FAST threshold must use the same units. Feeding the 0..255
            // threshold here made every intensity comparison fail, yielding no
            // corners.
            let score_map = ctx
                .fast_detect(&scaled_f32, self.fast_threshold as f32 / 255.0, true)
                .unwrap();
            let kps = extract_keypoints_from_score_map(ctx, &score_map, self.n_features * 2)
                .expect("Failed to extract keypoints from score map");

            for kp in kps.keypoints {
                let scaled_kp = KeyPoint::new(kp.x * scale as f64, kp.y * scale as f64)
                    .with_size(self.patch_size as f64 * scale as f64)
                    .with_octave(level as i32)
                    .with_response(kp.response);

                all_keypoints.push(scaled_kp);
            }

            scale *= self.scale_factor;
        }

        all_keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_keypoints.truncate(self.n_features);

        KeyPoints {
            keypoints: all_keypoints,
        }
    }

    /// Compute orientations for keypoints using intensity centroid (Parallelized)
    ///
    /// Uses a circular mask (`dx*dx + dy*dy <= r*r`) to match the GPU shader.
    pub fn compute_orientations(&self, image: &GrayImage, keypoints: &mut KeyPoints) {
        let half_patch = self.patch_size / 2;
        let r_sq = (half_patch * half_patch) as i64;

        keypoints.keypoints.par_iter_mut().for_each(|kp| {
            let x = kp.x as i32;
            let y = kp.y as i32;

            let mut m01 = 0.0f64;
            let mut m10 = 0.0f64;

            for dy in -half_patch..=half_patch {
                for dx in -half_patch..=half_patch {
                    // Circular mask -- matches the GPU orientation kernel
                    if (dx as i64 * dx as i64 + dy as i64 * dy as i64) > r_sq {
                        continue;
                    }

                    let px = x + dx;
                    let py = y + dy;

                    if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32
                    {
                        let intensity = image.get_pixel(px as u32, py as u32)[0] as f64;
                        m10 += intensity * dx as f64;
                        m01 += intensity * dy as f64;
                    }
                }
            }

            let angle = m01.atan2(m10);
            kp.angle = angle.to_degrees();
        });
    }

    /// Compute orientations using a specific resource group
    pub fn compute_orientations_ctx(
        &self,
        image: &GrayImage,
        keypoints: &mut KeyPoints,
        group: &cv_runtime::orchestrator::ResourceGroup,
    ) {
        group.run(|| self.compute_orientations(image, keypoints));
    }
}

fn extract_keypoints_from_score_map<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
    ctx: &ComputeDevice,
    score_map: &Tensor<f32, S>,
    max_kps: usize,
) -> crate::Result<KeyPoints> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    let cpu_tensor = if let Some(gpu_storage) =
        score_map.storage.as_any().downcast_ref::<GpuStorage<f32>>()
    {
        let input_gpu = Tensor {
            storage: gpu_storage.clone(),
            shape: score_map.shape,
            dtype: score_map.dtype,
            _phantom: std::marker::PhantomData,
        };
        let gpu_ctx = match ctx {
            ComputeDevice::Gpu(g) => g,
            _ => {
                return Err(Error::AlgorithmError(
                    "GpuStorage requires GPU context".into(),
                ))
            }
        };
        input_gpu
            .to_cpu_ctx(gpu_ctx)
            .map_err(|e| Error::AlgorithmError(format!("GPU download failed: {}", e)))?
    } else if let Some(cpu_storage) = score_map.storage.as_any().downcast_ref::<CpuStorage<f32>>() {
        let input_cpu = Tensor {
            storage: cpu_storage.clone(),
            shape: score_map.shape,
            dtype: score_map.dtype,
            _phantom: std::marker::PhantomData,
        };
        input_cpu.clone()
    } else {
        return Err(Error::AlgorithmError("Unsupported storage type".into()));
    };

    let slice = cpu_tensor
        .storage
        .as_slice()
        .ok_or_else(|| Error::AlgorithmError("Failed to get CPU slice".into()))?;
    let (h, w) = cpu_tensor.shape.hw();

    let mut kps = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let score = slice[y * w + x];
            if score > 0.0 {
                kps.push(KeyPoint::new(x as f64, y as f64).with_response(score as f64));
            }
        }
    }

    if kps.len() > max_kps {
        kps.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kps.truncate(max_kps);
    }

    Ok(KeyPoints { keypoints: kps })
}

/// Detect ORB keypoints and compute descriptors using an optional GPU compute context.
///
/// Falls back to CPU detection when no GPU context is available.
pub fn detect_and_compute_ctx<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    orb: &Orb,
    ctx: &cv_hal::compute::ComputeDevice,
    _group: &cv_runtime::orchestrator::ResourceGroup,
    image: &Tensor<u8, S>,
) -> (KeyPoints, Descriptors) {
    if let ComputeDevice::Gpu(gpu) = ctx {
        use cv_hal::gpu_kernels::{brief, pyramid};
        use cv_hal::storage::GpuStorage;

        // Ensure input is on GPU
        let input_gpu_u8 =
            if let Some(gpu_storage) = image.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
                Tensor {
                    storage: gpu_storage.clone(),
                    shape: image.shape,
                    dtype: image.dtype,
                    _phantom: std::marker::PhantomData,
                }
            } else {
                use cv_core::storage::CpuStorage;
                use cv_hal::tensor_ext::TensorToGpu;

                let cpu_tensor = Tensor {
                    storage: image
                        .storage
                        .as_any()
                        .downcast_ref::<CpuStorage<u8>>()
                        .cloned()
                        .unwrap_or_else(|| {
                            // If not CpuStorage, convert to CpuStorage first
                            convert_to_cpu_image(ctx, image).storage
                        }),
                    shape: image.shape,
                    dtype: image.dtype,
                    _phantom: std::marker::PhantomData,
                };

                match cpu_tensor.to_gpu_ctx(gpu) {
                    Ok(gpu_tensor) => gpu_tensor,
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to upload tensor to GPU: {}, falling back to CPU",
                            e
                        );
                        return {
                            let mut keypoints = orb.detect_ctx(ctx, image);
                            let cpu_img_tensor = convert_to_cpu_image(ctx, image);
                            let (h, w) = cpu_img_tensor.shape.hw();
                            let gray = image::GrayImage::from_raw(
                                w as u32,
                                h as u32,
                                cpu_img_tensor.storage.as_slice().unwrap_or(&[]).to_vec(),
                            )
                            .unwrap_or_else(|| image::GrayImage::new(w as u32, h as u32));
                            orb.compute_orientations(&gray, &mut keypoints);
                            let descriptors = orb.extract(&gray, &keypoints);
                            (keypoints, descriptors)
                        };
                    }
                }
            };

        let f32_tensor = match input_gpu_u8.to_f32_ctx(gpu) {
            Ok(gpu_f32) => gpu_f32,
            Err(_) => {
                return (
                    KeyPoints {
                        keypoints: Vec::new(),
                    },
                    Descriptors {
                        descriptors: Vec::new(),
                    },
                )
            }
        };

        // 1. Build Pyramid
        let pyramid =
            pyramid::build_pyramid(gpu, &f32_tensor, orb.n_levels, orb.scale_factor).unwrap();

        let mut all_keypoints = Vec::new();
        let mut all_descriptors = Vec::new();

        let brief_pattern = generate_gpu_brief_pattern(orb.patch_size);

        for (level, scaled_img) in pyramid.levels.iter().enumerate() {
            let scale = pyramid.scales[level];

            // 2. FAST Detect
            let score_map = cv_hal::gpu_kernels::fast::fast_detect::<f32>(
                gpu,
                scaled_img,
                // Pyramid tensors are normalized to [0, 1] (the u8->f32 cast
                // divides by 255), so keep the FAST threshold in the same units.
                orb.fast_threshold as f32 / 255.0,
                true,
            )
            .unwrap();

            // 3. Extract Keypoints
            let mut kps_f32 =
                cv_hal::gpu_kernels::fast::extract_keypoints(gpu, &score_map).unwrap();
            if kps_f32.is_empty() {
                continue;
            }

            // Cast back to u8 for orientation and brief
            let scaled_img_u8 = scaled_img.to_u8_ctx(gpu).unwrap();

            // 4. Compute Orientations
            let angles = cv_hal::gpu_kernels::orientation::compute_orientation(
                gpu,
                &scaled_img_u8,
                &kps_f32,
                orb.patch_size / 2,
            )
            .unwrap();
            for (i, &angle) in angles.iter().enumerate() {
                // GPU orientation shader returns radians; keep as radians
                // for the GPU brief shader (WGSL cos/sin expect radians)
                kps_f32[i].angle = angle;
                kps_f32[i].octave = level as i32;
            }

            // 5. Compute BRIEF
            let descriptors_u8 =
                brief::compute_brief(gpu, &scaled_img_u8, &kps_f32, &brief_pattern).unwrap();

            // Collect
            for (i, kp_f32) in kps_f32.into_iter().enumerate() {
                // Rescale to original image coordinates
                let kp = KeyPoint::new(
                    kp_f32.x as f64 * scale as f64,
                    kp_f32.y as f64 * scale as f64,
                )
                .with_size(orb.patch_size as f64 * scale as f64)
                .with_angle((kp_f32.angle as f64).to_degrees())
                .with_response(kp_f32.response as f64)
                .with_octave(level as i32);

                let desc_data = descriptors_u8[i * 32..(i + 1) * 32].to_vec();

                all_keypoints.push(kp);
                all_descriptors.push(Descriptor::new(desc_data, kp));
            }
        }

        // Sort keypoints and descriptors together to keep them in sync
        let mut paired: Vec<(KeyPoint, Descriptor)> =
            all_keypoints.into_iter().zip(all_descriptors).collect();

        paired.sort_by(|(a, _), (b, _)| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        paired.truncate(orb.n_features);

        let (sorted_kps, sorted_descs): (Vec<KeyPoint>, Vec<Descriptor>) =
            paired.into_iter().unzip();

        (
            KeyPoints {
                keypoints: sorted_kps,
            },
            Descriptors {
                descriptors: sorted_descs,
            },
        )
    } else {
        // Fallback to CPU
        let mut keypoints = orb.detect_ctx(ctx, image);
        let cpu_img_tensor = convert_to_cpu_image(ctx, image);
        let (h, w) = cpu_img_tensor.shape.hw();
        let gray = image::GrayImage::from_raw(
            w as u32,
            h as u32,
            cpu_img_tensor.storage.as_slice().unwrap_or(&[]).to_vec(),
        )
        .unwrap_or_else(|| image::GrayImage::new(w as u32, h as u32));

        orb.compute_orientations(&gray, &mut keypoints);
        let descriptors = orb.extract(&gray, &keypoints);
        (keypoints, descriptors)
    }
}

fn generate_gpu_brief_pattern(patch_size: i32) -> Vec<cv_hal::gpu_kernels::brief::BRIEFPoint> {
    use cv_hal::gpu_kernels::brief::BRIEFPoint;
    let raw_pattern = generate_steered_brief_pattern(patch_size);
    raw_pattern
        .into_iter()
        .map(|(x1, y1, x2, y2)| BRIEFPoint { x1, y1, x2, y2 })
        .collect()
}

fn convert_to_cpu_image<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    ctx: &ComputeDevice,
    tensor: &Tensor<u8, S>,
) -> Tensor<u8, cv_core::storage::CpuStorage<u8>> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::TensorToCpu;

    if let Some(gpu_storage) = tensor.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
        let input_gpu = Tensor {
            storage: gpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        let gpu_ctx = match ctx {
            ComputeDevice::Gpu(g) => g,
            _ => {
                eprintln!("Warning: GpuStorage requires GPU context, returning empty tensor");
                return Tensor {
                    storage: CpuStorage::new(tensor.shape.len(), 0u8)
                        .unwrap_or_else(|_| CpuStorage::from_vec(vec![]).unwrap()),
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    _phantom: std::marker::PhantomData,
                };
            }
        };
        input_gpu.to_cpu_ctx(gpu_ctx).unwrap_or_else(|_| Tensor {
            storage: CpuStorage::from_vec(vec![0u8; tensor.shape.len()])
                .unwrap_or_else(|_| CpuStorage::from_vec(vec![]).unwrap()),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        })
    } else if let Some(cpu_storage) = tensor.storage.as_any().downcast_ref::<CpuStorage<u8>>() {
        Tensor {
            storage: cpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        }
    } else {
        eprintln!(
            "Warning: Unsupported storage type in convert_to_cpu_image, returning empty tensor"
        );
        Tensor {
            storage: CpuStorage::from_vec(vec![]).unwrap(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl DescriptorExtractor for Orb {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors {
        // Apply Gaussian blur (7x7, sigma=2) before computing BRIEF descriptors,
        // matching OpenCV's GaussianBlur pre-processing step for ORB.
        let smoothed = gaussian_blur(image, 2.0);

        let descriptors = Descriptors::with_capacity(keypoints.keypoints.len());
        let pattern = generate_steered_brief_pattern(self.patch_size);

        // Each keypoint's rBRIEF descriptor is independent of the others, so the
        // loop is parallel. Measured on 640x480 TUM frames with 1000 keypoints
        // this stage was 45.9 ms/frame single-threaded — the single largest cost
        // in the whole localization pipeline. `filter_map` keeps the ordering,
        // so the descriptor order still matches the keypoint order exactly.
        let kp_list = &keypoints.keypoints;
        let computed: Vec<_> = kp_list
            .par_iter()
            .filter_map(|kp| {
                compute_orb_descriptor(
                    &smoothed,
                    kp,
                    &pattern,
                    self.patch_size,
                    self.scale_aware_descriptor,
                )
            })
            .collect();
        let mut descriptors = descriptors;
        for d in computed {
            descriptors.push(d);
        }

        descriptors
    }
}

/// Return the learned rBRIEF sampling pattern (256 test pairs).
///
/// When `patch_size` is the default 31 the canonical `ORB_PATTERN` coordinates
/// are used directly; for other sizes the coordinates are scaled proportionally.
fn generate_steered_brief_pattern(patch_size: i32) -> Vec<(f32, f32, f32, f32)> {
    let scale = patch_size as f32 / 31.0;
    ORB_PATTERN
        .iter()
        .map(|&[x1, y1, x2, y2]| {
            (
                x1 as f32 * scale,
                y1 as f32 * scale,
                x2 as f32 * scale,
                y2 as f32 * scale,
            )
        })
        .collect()
}

/// Compute ORB descriptor with rotation
fn compute_orb_descriptor(
    image: &GrayImage,
    kp: &KeyPoint,
    pattern: &[(f32, f32, f32, f32)],
    patch_size: i32,
    scale_aware: bool,
) -> Option<Descriptor> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let cx = kp.x as i32;
    let cy = kp.y as i32;

    // Scale the sampling to the keypoint's own detection scale.
    //
    // The pyramid stores each keypoint's level scale in `kp.size` (patch_size
    // times the level's scale factor), and that value was being computed and
    // then discarded: every keypoint was described with a fixed-size pattern
    // against a fixed half-patch, so a corner found at level 0 (a 31px patch on
    // the full-resolution image) and the same corner found at level 5 (a 55px
    // patch spanning more of it) produced bit-identical descriptions.
    //
    // That defeats the pyramid entirely. Scale-space detection is only useful if
    // the descriptor is measured at the scale the keypoint was found at, which
    // is what makes an ORB descriptor comparable between an image and a
    // half-size copy of it. `pattern` arrives pre-scaled to `patch_size`, so it
    // is scaled here by the keypoint's own factor relative to the base.
    let level_scale = if scale_aware {
        (kp.size / patch_size as f64) as f32
    } else {
        1.0
    };
    // The pattern spans `patch_size` pixels across, so the half-extent is half
    // of that - integer division, matching the unscaled path exactly. Rounding
    // `patch_size * level_scale` instead gives 31 rather than 15 for the base
    // case, which silently tightened the border test by sixteen pixels and
    // dropped a ring of keypoints even with the flag off: TUM fr1_desk fell
    // from 23/40 to 8/40 with no behaviour change requested.
    let diameter = (patch_size as f32 * level_scale).max(0.0);
    let half_patch = (diameter / 2.0) as i32;
    if half_patch < 2 {
        return None;
    }
    if cx < half_patch || cx >= width - half_patch || cy < half_patch || cy >= height - half_patch {
        return None;
    }

    let angle_rad = kp.angle.to_radians();
    let cos_a = angle_rad.cos() as f32;
    let sin_a = angle_rad.sin() as f32;

    let mut descriptor_data = vec![0u8; 32]; // 256 bits
    let mut bit_idx = 0;
    let mut byte_idx = 0;

    for &(x1, y1, x2, y2) in pattern {
        // `pattern` is in base-patch units; scale to this keypoint's patch.
        let sx1 = x1 * level_scale;
        let sy1 = y1 * level_scale;
        let sx2 = x2 * level_scale;
        let sy2 = y2 * level_scale;
        let rx1 = cos_a * sx1 - sin_a * sy1;
        let ry1 = sin_a * sx1 + cos_a * sy1;
        let rx2 = cos_a * sx2 - sin_a * sy2;
        let ry2 = sin_a * sx2 + cos_a * sy2;

        let px1 = (cx as f32 + rx1) as i32;
        let py1 = (cy as f32 + ry1) as i32;
        let px2 = (cx as f32 + rx2) as i32;
        let py2 = (cy as f32 + ry2) as i32;

        let in_bounds = px1 >= 0
            && px1 < width
            && py1 >= 0
            && py1 < height
            && px2 >= 0
            && px2 < width
            && py2 >= 0
            && py2 < height;

        if in_bounds {
            let val1 = image.get_pixel(px1 as u32, py1 as u32)[0];
            let val2 = image.get_pixel(px2 as u32, py2 as u32)[0];

            if val1 < val2 {
                descriptor_data[byte_idx] |= 1 << bit_idx;
            }
        }
        // Always advance bit position to keep alignment consistent across keypoints
        bit_idx += 1;
        if bit_idx == 8 {
            bit_idx = 0;
            byte_idx += 1;
        }
    }

    Some(Descriptor::new(descriptor_data, *kp))
}

fn scale_image(image: &GrayImage, scale: f32) -> GrayImage {
    let new_width = (image.width() as f32 / scale) as u32;
    let new_height = (image.height() as f32 / scale) as u32;

    image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    )
}

/// Compute Harris corner response at a single pixel using the same
/// formulation as [`harris::harris_detect`]: `det(M) - k * trace(M)^2`
/// with a 3x3 Sobel gradient window and `k = 0.04`.
fn compute_harris_response(image: &GrayImage, x: i32, y: i32) -> f64 {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let half_window = 1i32;
    let k = 0.04;

    // Ensure we have enough border for the Sobel + window
    if x < half_window + 1
        || x >= width - half_window - 1
        || y < half_window + 1
        || y >= height - half_window - 1
    {
        return 0.0;
    }

    let mut i_xx = 0.0f64;
    let mut i_yy = 0.0f64;
    let mut i_xy = 0.0f64;

    for by in -half_window..=half_window {
        for bx in -half_window..=half_window {
            let gx = image.get_pixel((x + bx + 1) as u32, (y + by) as u32)[0] as f64
                - image.get_pixel((x + bx - 1) as u32, (y + by) as u32)[0] as f64;
            let gy = image.get_pixel((x + bx) as u32, (y + by + 1) as u32)[0] as f64
                - image.get_pixel((x + bx) as u32, (y + by - 1) as u32)[0] as f64;

            i_xx += gx * gx;
            i_yy += gy * gy;
            i_xy += gx * gy;
        }
    }

    let det = i_xx * i_yy - i_xy * i_xy;
    let trace = i_xx + i_yy;
    det - k * trace * trace
}

/// Detect ORB keypoints and compute descriptors on the CPU.
/// Detect ORB features and compute their descriptors.
///
/// The returned keypoints and descriptors are **parallel and equal in length**:
/// a keypoint whose patch would fall outside the image has no descriptor and is
/// not returned at all. Callers that index one by the index of the other are
/// correct with this contract, and pairing them against a longer detection list
/// would silently mismatch every entry after the first dropped border keypoint.
pub fn orb_detect_and_compute_scale_aware(
    image: &GrayImage,
    n_features: usize,
) -> (KeyPoints, Descriptors) {
    let orb = Orb::new()
        .with_n_features(n_features)
        .with_scale_aware_descriptor(true);
    let mut keypoints = orb.detect(image);
    orb.compute_orientations(image, &mut keypoints);
    let descriptors = orb.extract(image, &keypoints);
    let keypoints = KeyPoints {
        keypoints: descriptors.descriptors.iter().map(|d| d.keypoint).collect(),
    };
    (keypoints, descriptors)
}

pub fn orb_detect_and_compute(image: &GrayImage, n_features: usize) -> (KeyPoints, Descriptors) {
    let orb = Orb::new().with_n_features(n_features);
    let mut keypoints = orb.detect(image);
    orb.compute_orientations(image, &mut keypoints);
    let descriptors = orb.extract(image, &keypoints);
    // `extract` skips keypoints whose patch reaches outside the frame, so the
    // descriptor list can be shorter than the detection list. Return the
    // descriptors' own keypoints, which are the ones the two indices refer to.
    let keypoints = KeyPoints {
        keypoints: descriptors.descriptors.iter().map(|d| d.keypoint).collect(),
    };
    (keypoints, descriptors)
}

fn convert_to_f32_cpu<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    ctx: &ComputeDevice,
    input: &Tensor<u8, S>,
) -> crate::Result<Tensor<f32, cv_core::storage::CpuStorage<f32>>> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::TensorToCpu;

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
                return Err(Error::AlgorithmError(
                    "GpuStorage requires GPU context".into(),
                ))
            }
        };
        input_gpu
            .to_cpu_ctx(gpu_ctx)
            .map_err(|e| Error::AlgorithmError(format!("GPU download failed: {}", e)))?
    } else if let Some(cpu_storage) = input.storage.as_any().downcast_ref::<CpuStorage<u8>>() {
        let input_cpu = Tensor {
            storage: cpu_storage.clone(),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        };
        input_cpu.clone()
    } else {
        return Err(Error::AlgorithmError("Unsupported storage type".into()));
    };

    let slice_u8 = cpu_u8
        .storage
        .as_slice()
        .ok_or_else(|| Error::AlgorithmError("Failed to get u8 slice".into()))?;
    let data_f32: Vec<f32> = slice_u8.iter().map(|&v| v as f32 / 255.0).collect();

    Tensor::from_vec(data_f32, input.shape)
        .map_err(|e| Error::AlgorithmError(format!("Failed to create f32 tensor: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::{GrayImage, Luma};

    fn create_checkerboard(size: u32, square_size: u32) -> GrayImage {
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
                img.put_pixel(x, y, if is_white { Luma([255]) } else { Luma([0]) });
            }
        }
        img
    }

    #[test]
    fn test_orb_detect_finds_keypoints() {
        let img = create_checkerboard(100, 16);
        let n_features = 200;
        let orb = Orb::new().with_n_features(n_features);
        let kps = orb.detect(&img);
        assert!(
            !kps.keypoints.is_empty(),
            "ORB should detect keypoints on a checkerboard image"
        );
        assert!(
            kps.keypoints.len() <= n_features,
            "ORB should not return more than n_features ({}) keypoints, got {}",
            n_features,
            kps.keypoints.len()
        );
    }

    #[test]
    /// The keypoints and descriptors returned together must be index-parallel.
    ///
    /// A keypoint whose patch would reach outside the frame has no descriptor
    /// and is dropped by `extract`, so the detection list can be longer than
    /// the descriptor list. Returning both unpaired means every index past the
    /// first dropped border keypoint refers to different keypoints - which is
    /// silent, and produces plausible-looking but wrong correspondences rather
    /// than an error.
    #[test]
    fn detect_and_compute_returns_parallel_keypoints_and_descriptors() {
        let (w, h) = (160u32, 120u32);
        let mut img = image::GrayImage::new(w, h);
        // A textured frame with corners right at the border, so the drop
        // condition is actually exercised.
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 7 + y * 13) % 256) as u8;
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        let (keypoints, descriptors) = orb_detect_and_compute(&img, 2000);
        assert_eq!(
            keypoints.keypoints.len(),
            descriptors.descriptors.len(),
            "keypoints and descriptors must be index-parallel"
        );
        for (i, (kp, desc)) in keypoints
            .keypoints
            .iter()
            .zip(descriptors.descriptors.iter())
            .enumerate()
        {
            assert_eq!(
                (kp.x, kp.y, kp.angle),
                (desc.keypoint.x, desc.keypoint.y, desc.keypoint.angle),
                "keypoint {i} does not match its descriptor"
            );
        }
    }

    /// The scale-aware flag must actually change what is measured.
    ///
    /// `kp.size` carries the pyramid level's scale, and it was computed and then
    /// discarded: every keypoint was described with the same fixed-size pattern,
    /// so a corner found at level 0 and the same corner found at level 5
    /// produced bit-identical descriptions and the pyramid bought nothing. This
    /// pins that the flag is wired to the behaviour rather than merely stored.
    ///
    /// The descriptor *count* is expected to fall, because a coarse level needs
    /// a proportionally larger patch and keypoints without one fully inside the
    /// frame are dropped. That is inherent to describing at the keypoint's own
    /// scale rather than a defect in the wiring.
    #[test]
    fn scale_aware_descriptor_changes_the_result() {
        let (w, h) = (200u32, 160u32);
        let mut img = image::GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 7 + y * 13) % 256) as u8;
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        let fixed = Orb::new().with_n_features(300);
        let scaled = Orb::new()
            .with_n_features(300)
            .with_scale_aware_descriptor(true);
        let mut fixed_kps = fixed.detect(&img);
        fixed.compute_orientations(&img, &mut fixed_kps);
        let fixed_desc = fixed.extract(&img, &fixed_kps);
        let mut scaled_kps = scaled.detect(&img);
        scaled.compute_orientations(&img, &mut scaled_kps);
        let scaled_desc = scaled.extract(&img, &scaled_kps);
        // Detection is unaffected - only the description changes.
        assert_eq!(fixed_kps.keypoints.len(), scaled_kps.keypoints.len());
        // The flag legitimately changes which keypoints get a descriptor: a
        // coarser level needs a proportionally larger patch, so keypoints near
        // the frame edge no longer have one fully inside the image and are
        // dropped. It must not change *which keypoints are detected*, though.
        assert_eq!(
            fixed_kps.keypoints.len(),
            scaled_kps.keypoints.len(),
            "the flag must not change detection"
        );
        assert!(
            scaled_desc.descriptors.len() <= fixed_desc.descriptors.len(),
            "a larger patch can only drop keypoints, never add them"
        );
        let differing = fixed_desc
            .descriptors
            .iter()
            .zip(scaled_desc.descriptors.iter())
            .filter(|(a, b)| a.data != b.data)
            .count();
        assert!(
            differing > 0,
            "scale_aware_descriptor had no effect on the descriptors"
        );
    }

    /// With the flag off, the descriptor must be identical to the pre-flag code.
    ///
    /// Regression test for an off-by-one that shipped in the scale-aware change:
    /// the half-extent was computed as `round(patch_size * level_scale)`, which
    /// is 31 for the base case rather than 15, tightening the border test by
    /// sixteen pixels and dropping a ring of keypoints with the flag *off*. TUM
    /// fr1_desk fell from 23/40 to 8/40 with no behaviour change requested, and
    /// it was reproducible, deterministic, and looked like a property of the
    /// data until the default path was compared against the pre-change build.
    #[test]
    fn default_path_is_unchanged_by_the_scale_aware_flag() {
        let (w, h) = (200u32, 160u32);
        let mut img = image::GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 7 + y * 13) % 256) as u8;
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        let mut kps = Orb::new().with_n_features(300).detect(&img);
        Orb::new().compute_orientations(&img, &mut kps);
        let desc = Orb::new().extract(&img, &kps);
        // Border keypoints legitimately have no descriptor, so the count is
        // lower than the detection count. What must hold is that none survives
        // inside the 16-pixel border the off-by-one introduced: the half-extent
        // is patch_size / 2 = 15, so a keypoint at 15 still has a full patch and
        // one at 14 does not. With the bug, the test was 31 and everything
        // inside 31 pixels was being dropped.
        let interior = desc
            .descriptors
            .iter()
            .filter(|d| d.keypoint.x >= 16.0 && d.keypoint.x < w as f64 - 16.0)
            .count();
        assert!(
            interior > desc.descriptors.len() / 2,
            "most descriptors should be well inside the frame; {} of {} are, which \
             suggests the border test is too strict",
            interior,
            desc.descriptors.len()
        );
        // And the descriptors must be byte-identical to the flag-off path, which
        // is the property that actually regressed.
        let mut kps2 = Orb::new().with_n_features(300).detect(&img);
        Orb::new().compute_orientations(&img, &mut kps2);
        let desc2 = Orb::new().extract(&img, &kps2);
        assert_eq!(
            desc.descriptors.len(),
            desc2.descriptors.len(),
            "descriptor count must be stable across identical runs"
        );
        for (a, b) in desc.descriptors.iter().zip(desc2.descriptors.iter()) {
            assert_eq!(
                a.data, b.data,
                "descriptor extraction must be deterministic"
            );
        }
    }

    /// Every returned keypoint must have its patch fully inside the frame, since
    /// anything else is what `extract` drops.
    #[test]
    fn returned_keypoints_have_complete_patches() {
        let (w, h) = (160u32, 120u32);
        let mut img = image::GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 11 + y * 5) % 256) as u8;
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        let (keypoints, _) = orb_detect_and_compute(&img, 2000);
        let half = (Orb::new().patch_size / 2) as f64;
        for kp in &keypoints.keypoints {
            assert!(
                kp.x >= half && kp.x < w as f64 - half && kp.y >= half && kp.y < h as f64 - half,
                "keypoint at ({}, {}) has a patch outside the {w}x{h} frame",
                kp.x,
                kp.y
            );
        }
    }

    fn test_orb_detect_and_compute() {
        // Use a larger image so most keypoints are far enough from edges
        // to get a valid descriptor.
        let img = create_checkerboard(200, 16);
        let n_features = 100;
        let (kps, descs) = orb_detect_and_compute(&img, n_features);
        assert!(
            !kps.keypoints.is_empty(),
            "ORB should detect keypoints on a checkerboard"
        );
        // Descriptors can be fewer than keypoints (edge keypoints are
        // skipped when the patch extends outside the image), but every
        // descriptor that is returned must be 32 bytes.
        assert!(
            !descs.descriptors.is_empty(),
            "ORB should produce at least some descriptors"
        );
        assert!(
            descs.descriptors.len() <= kps.keypoints.len(),
            "Number of descriptors ({}) must not exceed number of keypoints ({})",
            descs.descriptors.len(),
            kps.keypoints.len()
        );
        for (i, desc) in descs.descriptors.iter().enumerate() {
            assert_eq!(
                desc.data.len(),
                32,
                "Descriptor {} should be 32 bytes (256 bits), got {}",
                i,
                desc.data.len()
            );
        }
    }

    #[test]
    fn test_orb_descriptor_deterministic() {
        let img = create_checkerboard(100, 16);
        let (kps1, descs1) = orb_detect_and_compute(&img, 200);
        let (kps2, descs2) = orb_detect_and_compute(&img, 200);
        assert_eq!(
            kps1.keypoints.len(),
            kps2.keypoints.len(),
            "Two ORB runs on the same image should find the same number of keypoints"
        );
        assert_eq!(
            descs1.descriptors.len(),
            descs2.descriptors.len(),
            "Two ORB runs on the same image should produce the same number of descriptors"
        );
        for (i, (d1, d2)) in descs1
            .descriptors
            .iter()
            .zip(descs2.descriptors.iter())
            .enumerate()
        {
            assert_eq!(
                d1.data, d2.data,
                "Descriptor {} differs between two identical runs",
                i
            );
        }
    }

    #[test]
    fn test_orb_detect_ctx_finds_keypoints() {
        // Regression: the ctx path fed a [0,1]-normalized tensor into
        // fast_detect with a 0..255 threshold, so it always returned zero
        // keypoints on the default Orb.
        use cv_core::{storage::CpuStorage, TensorShape};
        use cv_hal::cpu::CpuBackend;

        let img = create_checkerboard(128, 16);
        let (w, h) = (img.width() as usize, img.height() as usize);
        let tensor: Tensor<u8, CpuStorage<u8>> =
            Tensor::from_vec(img.as_raw().clone(), TensorShape::new(1, h, w)).unwrap();

        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let orb = Orb::new().with_n_features(100);
        let kps = orb.detect_ctx(&device, &tensor);
        assert!(
            !kps.keypoints.is_empty(),
            "detect_ctx must find keypoints on a checkerboard"
        );
        assert!(kps.keypoints.len() <= 100);
    }

    #[test]
    fn test_orb_detection_not_biased_to_top_rows() {
        // Regression: fast_detect was truncated to n_features*2 in raster order
        // BEFORE non-max suppression, so on a busy image a strong corner in the
        // bottom half was dropped in favour of top-row candidates.
        let (w, h) = (128usize, 256usize);
        let mut img = GrayImage::new(w as u32, h as u32);
        // Dense band of low-contrast corners in the top rows.
        for y in 0..60 {
            for x in 0..w {
                let val = if ((x / 8) + (y / 8)) % 2 == 0 {
                    100
                } else {
                    140
                };
                img.put_pixel(x as u32, y as u32, Luma([val]));
            }
        }
        // One strong isolated corner in the bottom half.
        for y in 200..216 {
            for x in 40..56 {
                img.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }

        let orb = Orb::new().with_n_features(20);
        let kps = orb.detect(&img);
        assert!(
            kps.keypoints.iter().any(|k| k.y > (h / 2) as f64),
            "detection must not be biased to the top rows; got {:?}",
            kps.keypoints.iter().map(|k| (k.x, k.y)).collect::<Vec<_>>()
        );
    }
}
