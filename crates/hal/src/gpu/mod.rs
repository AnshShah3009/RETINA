use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, DeviceId, SubmissionIndex};
use cv_core::{storage::Storage, Tensor};
use futures::executor::block_on;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{Backends, Device, Instance, PowerPreference, Queue, RequestAdapterOptions};

static GLOBAL_CONTEXT: OnceLock<crate::Result<GpuContext>> = OnceLock::new();
static NEXT_GPU_ID: AtomicU32 = AtomicU32::new(1);

/// Shared GPU Context containing Device and Queue.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    gpu_index: u32,
    is_unified: bool,
    pipeline_cache: Arc<std::sync::Mutex<std::collections::HashMap<String, wgpu::ComputePipeline>>>,
    last_submission: Arc<AtomicU64>,
}

impl GpuContext {
    /// Returns true if the GPU and CPU share the same physical memory (unified memory).
    pub fn is_unified_memory(&self) -> bool {
        self.is_unified
    }

    /// Safely downcasts a GpuStorage result to the requested generic storage S.
    /// Uses TypeId checks to avoid allocations when S is GpuStorage.
    #[allow(dead_code)]
    pub(crate) fn downcast_storage<
        T: Clone + Copy + std::fmt::Debug + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
    >(
        &self,
        result_gpu: Tensor<T, crate::storage::GpuStorage<T>>,
    ) -> crate::Result<Tensor<T, S>> {
        use std::marker::PhantomData;

        // Always use downcasting for safety.
        // The performance cost is negligible compared to GPU execution time.
        let storage_any = result_gpu.storage.as_any();

        if let Some(storage_s) = storage_any.downcast_ref::<S>() {
            Ok(Tensor {
                storage: storage_s.clone(),
                shape: result_gpu.shape,
                dtype: result_gpu.dtype,
                _phantom: PhantomData,
            })
        } else {
            Err(crate::Error::InvalidInput(
                "Failed to downcast GPU result to original storage type".into(),
            ))
        }
    }

    /// Get shader source for a named kernel.
    pub fn get_kernel_source(&self, name: &str) -> crate::Result<String> {
        let source = match name {
            "gaussian_blur" => include_str!("../../shaders/gaussian_blur_separable.wgsl"),
            "sobel" => include_str!("../../shaders/sobel.wgsl"),
            "sobel_f32" => include_str!("../../shaders/sobel_f32.wgsl"),
            "threshold" => include_str!("../../shaders/threshold.wgsl"),
            "threshold_f32" => include_str!("../../shaders/threshold_f32.wgsl"),
            "resize" => include_str!("../../shaders/resize.wgsl"),
            "resize_f32" => include_str!("../../shaders/resize_f32.wgsl"),
            "warp" => include_str!("../../shaders/warp.wgsl"),
            "morphology" => include_str!("../../shaders/morphology.wgsl"),
            "convolve_2d" => include_str!("../../shaders/convolve_2d.wgsl"),
            "color_cvt" => include_str!("../../shaders/color_cvt.wgsl"),
            "color_cvt_f32" => include_str!("../../shaders/color_cvt_f32.wgsl"),
            "bilateral" => include_str!("../../shaders/bilateral.wgsl"),
            "bilateral_f32" => include_str!("../../shaders/bilateral_f32.wgsl"),
            "fast" => include_str!("../../shaders/fast.wgsl"),
            "fast_f32" => include_str!("../../shaders/fast_f32.wgsl"),
            "fast_nms" => include_str!("../../shaders/fast_nms.wgsl"),
            "fast_nms_f32" => include_str!("../../shaders/fast_nms_f32.wgsl"),
            "canny" => include_str!("../../shaders/canny.wgsl"),
            "nms" => include_str!("../../shaders/nms.wgsl"),
            "matching" => include_str!("../../shaders/matching.wgsl"),
            "match_template" => include_str!("../../shaders/match_template.wgsl"),
            "stereo_match" => include_str!("../../shaders/stereo_match.wgsl"),
            "hough" => include_str!("../../shaders/hough.wgsl"),
            "hough_f32" => include_str!("../../shaders/hough_f32.wgsl"),
            "hough_circles" => include_str!("../../shaders/hough_circles.wgsl"),
            "subtract" => include_str!("../../shaders/subtract.wgsl"),
            "spmv" => include_str!("../../shaders/spmv.wgsl"),
            "vector_ops" => include_str!("../../shaders/vector_ops.wgsl"),
            "cast" => include_str!("../../shaders/cast.wgsl"),
            "lucas_kanade" => include_str!("../../shaders/lucas_kanade.wgsl"),
            "pointcloud_transform" => include_str!("../../shaders/pointcloud_transform.wgsl"),
            "remap" => include_str!("../../shaders/remap.wgsl"),
            "undistort" => include_str!("../../shaders/undistort.wgsl"),
            "lbvh_build" => include_str!("../../shaders/lbvh_build.wgsl"),
            "sift_extrema" => include_str!("../../shaders/sift_extrema.wgsl"),
            "sift_orientation" => include_str!("../../shaders/sift_orientation.wgsl"),
            "sift_descriptor" => include_str!("../../shaders/sift_descriptor.wgsl"),
            "akaze_derivatives" => include_str!("../../shaders/akaze_derivatives.wgsl"),
            "akaze_diffusion" => include_str!("../../shaders/akaze_diffusion.wgsl"),
            "akaze_contrast" => include_str!("../../shaders/akaze_contrast.wgsl"),
            "icp_dense" => include_str!("../../shaders/icp_dense.wgsl"),
            "icp_correspondence" => include_str!("../../shaders/icp_correspondence.wgsl"),
            "icp_accumulate" => include_str!("../../shaders/icp_accumulate.wgsl"),
            "icp_reduce" => include_str!("../../shaders/icp_reduce.wgsl"),
            "tsdf_raycast" => include_str!("../../shaders/tsdf_raycast.wgsl"),
            "marching_cubes_count" => include_str!("../../shaders/marching_cubes_count.wgsl"),
            "marching_cubes_emit" => include_str!("../../shaders/marching_cubes_emit.wgsl"),
            "marching_cubes" => include_str!("../../shaders/marching_cubes.wgsl"),
            "mog2_update" => include_str!("../../shaders/mog2_update.wgsl"),
            "iou_matrix" => include_str!("../../shaders/iou_matrix.wgsl"),
            "matrix_multiply" => include_str!("../../shaders/matrix_multiply.wgsl"),
            _ => {
                return Err(crate::Error::InvalidInput(format!(
                    "Unknown kernel name: {}",
                    name
                )))
            }
        };
        Ok(source.to_string())
    }
}


include!("compute_context_impl.rs");


impl GpuContext {
    /// Compute dominant orientations for SIFT keypoints on GPU.
    ///
    /// Returns a Vec<f32> of orientation angles in degrees [0, 360) for each keypoint.
    pub fn compute_sift_orientations(
        &self,
        image: &Tensor<f32, crate::storage::GpuStorage<f32>>,
        keypoints: &[cv_core::KeyPoint],
    ) -> crate::Result<Vec<f32>> {
        if keypoints.is_empty() {
            return Ok(Vec::new());
        }

        let num_kps = keypoints.len();

        // Upload keypoints as vec4<f32> [x, y, size, octave]
        let kp_data: Vec<[f32; 4]> = keypoints
            .iter()
            .map(|kp| [kp.x as f32, kp.y as f32, kp.size as f32, kp.octave as f32])
            .collect();

        let kp_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SIFT Orientation Keypoints"),
                contents: bytemuck::cast_slice(&kp_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer: one f32 per keypoint
        let out_byte_size = (num_kps * 4) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SIFT Orientation Output"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            image.shape.width as u32,
            image.shape.height as u32,
            num_kps as u32,
            0u32,
        ];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SIFT Orientation Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader_source = include_str!("../../shaders/sift_orientation.wgsl");
        let pipeline = self.create_compute_pipeline(shader_source, "main");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SIFT Orientation Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let x = (num_kps as u32).div_ceil(64);
            pass.dispatch_workgroups(x, 1, 1);
        }
        self.submit(encoder);

        let orientations: Vec<f32> =
            pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                self.device.clone(),
                &self.queue,
                &output_buffer,
                0,
                out_byte_size as usize,
            ))?;

        Ok(orientations)
    }

    /// Get the global GPU context. Returns an error if not yet initialized.
    pub fn global() -> crate::Result<&'static GpuContext> {
        GLOBAL_CONTEXT
            .get()
            .ok_or_else(|| {
                crate::Error::InitError(
                    "GPU Context not initialized. Call init_global() first.".into(),
                )
            })?
            .as_ref()
            .map_err(|e| crate::Error::InitError(e.to_string()))
    }

    /// Initialize the global GPU context asynchronously.
    pub async fn init_global() -> crate::Result<&'static GpuContext> {
        let res = GLOBAL_CONTEXT.get_or_init(|| block_on(Self::new_async()));

        res.as_ref()
            .map_err(|e| crate::Error::InitError(e.to_string()))
    }

    /// Initialize a new GPU context (synchronous wrapper).
    pub fn new() -> crate::Result<Self> {
        block_on(Self::new_async())
    }

    /// Initialize a new GPU context asynchronously.
    pub async fn new_async() -> crate::Result<Self> {
        Self::new_with_policy(PowerPreference::HighPerformance).await
    }

    pub async fn new_with_policy(preference: PowerPreference) -> crate::Result<Self> {
        // Create instance with conservative flags to avoid driver panics on integrated GPUs
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default()
                .difference(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                crate::Error::InitError(format!("Failed to find a suitable GPU adapter: {}", e))
            })?;

        Self::from_adapter(adapter).await
    }

    pub async fn from_adapter(adapter: wgpu::Adapter) -> crate::Result<Self> {
        // Request TIMESTAMP_QUERY if the adapter supports it (for GPU profiling)
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        // Request device with increased limits for large point clouds
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 256 * 1024 * 1024,
            max_buffer_size: 256 * 1024 * 1024,
            ..wgpu::Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("CV-HAL Device"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| crate::Error::InitError(format!("Failed to create GPU device: {}", e)))?;

        let info = adapter.get_info();
        let is_unified = info.device_type == wgpu::DeviceType::IntegratedGpu
            || (info.backend == wgpu::Backend::Metal && info.name.contains("Apple"));

        let gpu_index = NEXT_GPU_ID.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            gpu_index,
            is_unified,
            pipeline_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            last_submission: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Check if a GPU is available.
    pub fn is_available() -> bool {
        block_on(Self::is_available_async())
    }

    /// Check if a GPU is available asynchronously.
    pub async fn is_available_async() -> bool {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        !instance
            .enumerate_adapters(Backends::all())
            .await
            .is_empty()
    }

    /// Enumerate all available adapters.
    pub async fn enumerate_adapters() -> Vec<wgpu::Adapter> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        instance.enumerate_adapters(Backends::all()).await
    }

    /// Get reference to device (convenience method)
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get Arc to device
    pub fn device_arc(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Get Arc to queue
    pub fn queue_arc(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    /// Estimate usable GPU memory in megabytes.
    ///
    /// wgpu does not expose total VRAM directly. This uses `max_buffer_size` from
    /// the device limits as a conservative lower bound (typically 25-50% of actual
    /// VRAM). Override with `CV_GPU_MEMORY_MB` env var for precise control.
    pub fn estimated_memory_mb(&self) -> u32 {
        if let Ok(val) = std::env::var("CV_GPU_MEMORY_MB") {
            if let Ok(mb) = val.parse::<u32>() {
                return mb;
            }
        }
        let max_buffer = self.device.limits().max_buffer_size;
        // max_buffer_size is the largest single allocation, typically 256MB-2GB.
        // Total VRAM is usually 2-4x this value. We report max_buffer_size as a
        // conservative usable estimate to prevent overcommit.
        (max_buffer / (1024 * 1024)) as u32
    }

    /// Submit a command encoder (convenience method)
    pub fn submit(&self, encoder: wgpu::CommandEncoder) -> SubmissionIndex {
        let index = self.last_submission.fetch_add(1, Ordering::SeqCst) + 1;
        self.queue.submit(std::iter::once(encoder.finish()));
        SubmissionIndex(index)
    }

    /// Create a simplified compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let cache_key = format!("{}:{}", shader_source, entry_point);

        // Try to get from cache, handling mutex poison gracefully
        let cache_result = self.pipeline_cache.lock();
        match cache_result {
            Ok(cache) => {
                if let Some(pipeline) = cache.get(&cache_key) {
                    return pipeline.clone();
                }
            }
            Err(_) => {
                // Cache is poisoned, continue without cached pipeline
            }
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        // Try to insert into cache
        match self.pipeline_cache.lock() {
            Ok(mut cache) => {
                cache.insert(cache_key, pipeline.clone());
            }
            Err(_) => {
                // Cache is poisoned, pipeline not cached
            }
        }

        pipeline
    }

    /// Get a pooled buffer.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        crate::gpu_kernels::buffer_utils::global_pool().get(&self.device, size, usage)
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        crate::gpu_kernels::buffer_utils::global_pool().return_buffer(&self.device, buffer, usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new();
        match ctx {
            Ok(c) => println!("GPU Context created: {:?}", c.device),
            Err(e) => println!("GPU initialization failed (expected on some CI): {}", e),
        }
    }

    #[test]
    fn test_gaussian_blur_gpu_parity() {
        let gpu = if let Ok(g) = GpuContext::global() {
            g
        } else {
            return;
        };
        let cpu = crate::cpu::CpuBackend::new().unwrap();

        let width = 64usize;
        let height = 64usize;
        let mut data = vec![0f32; width * height];
        for i in 0..data.len() {
            data[i] = (i % 255) as f32;
        }

        let shape = cv_core::TensorShape::new(1, height, width);
        let tensor_cpu = cv_core::CpuTensor::from_vec(data, shape).unwrap();

        // GPU execution
        use crate::tensor_ext::{TensorToCpu, TensorToGpu};
        let tensor_gpu = tensor_cpu.to_gpu_ctx(gpu).unwrap();
        let blurred_gpu = gpu.gaussian_blur(&tensor_gpu, 1.5, 7).unwrap();
        let res_gpu = blurred_gpu.to_cpu_ctx(gpu).unwrap();

        // CPU execution
        let res_cpu = cpu.gaussian_blur(&tensor_cpu, 1.5, 7).unwrap();

        // Check equality
        let slice_gpu = res_gpu.as_slice().unwrap();
        let slice_cpu = res_cpu.as_slice().unwrap();

        let mut diff_count = 0;
        for i in 0..slice_gpu.len() {
            if (slice_gpu[i] as i32 - slice_cpu[i] as i32).abs() > 1 {
                diff_count += 1;
            }
        }

        assert!(
            diff_count < (width * height) / 100,
            "Too many differences between GPU and CPU blur: {}",
            diff_count
        );
    }
}
