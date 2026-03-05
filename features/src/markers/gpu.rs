//! GPU-accelerated marker detection using wgpu compute shaders.
//!
//! This module provides GPU acceleration for the inner loop of marker detection:
//! - Binary grid sampling from candidate bounding boxes
//! - Bitmask generation and rotation
//! - Dictionary matching with confidence scoring
//!
//! The CPU candidate finder still runs on the host; only the grid sampling
//! and bit decoding runs on the GPU.

#![allow(deprecated)]

use crate::Result;
use bytemuck::{Pod, Zeroable};
use cv_core::Error;
use cv_hal::gpu_utils;
use image::GrayImage;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

/// Global flag to enable/disable GPU marker detection at runtime.
static USE_GPU: AtomicBool = AtomicBool::new(true);

/// Enable or disable GPU marker detection globally.
pub fn use_gpu(enabled: bool) {
    USE_GPU.store(enabled, Ordering::Relaxed);
}

/// Check if GPU marker detection is enabled.
pub fn is_gpu_enabled() -> bool {
    USE_GPU.load(Ordering::Relaxed)
}

/// Candidate bounding rectangle passed to GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCandidate {
    /// Left edge of the candidate bounding box (inclusive).
    pub min_x: u32,
    /// Top edge of the candidate bounding box (inclusive).
    pub min_y: u32,
    /// Right edge of the candidate bounding box (inclusive).
    pub max_x: u32,
    /// Bottom edge of the candidate bounding box (inclusive).
    pub max_y: u32,
    /// Total grid dimension (payload + 2 × border bits).
    pub grid_size: u32,
    /// Number of payload bits per row/column.
    pub payload_bits: u32,
}

/// Result from GPU marker scan.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMarkerResult {
    /// Lower 32 bits of the sampled marker bitmask.
    pub bitmask: u32,
    /// Upper 32 bits of the sampled marker bitmask (for payloads > 32 bits).
    pub bitmask_high: u32,
    /// Index of the best-matching code in the dictionary.
    pub best_id: u32,
    /// Clockwise rotation (0–3) applied to match the best code.
    pub rotation: u32,
    /// Match confidence score (lower Hamming distance → higher confidence).
    pub confidence: f32,
    /// Detection status: 0 = invalid, 1 = valid, 2 = border check failed, 3 = no dictionary match.
    pub status: u32,
}

impl GpuMarkerResult {
    /// Check if this result represents a valid marker detection.
    pub fn is_valid(&self) -> bool {
        self.status == 1
    }

    /// Check if border validation failed.
    pub fn border_failed(&self) -> bool {
        self.status == 2
    }

    /// Get the 64-bit payload bitmask.
    pub fn payload(&self) -> u64 {
        (self.bitmask as u64) | ((self.bitmask_high as u64) << 32)
    }
}

/// GPU shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuParams {
    image_width: u32,
    image_height: u32,
    num_candidates: u32,
    dict_size: u32,
    border_bits: u32,
    threshold: u32,
    max_hamming: u32,
    _padding: u32,
}

/// GPU context for marker detection shader.
pub struct MarkerGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MarkerGpuContext {
    /// Create a new GPU context for marker detection.
    ///
    /// This compiles the shader and initializes the GPU device.
    /// Returns `None` if no suitable GPU is available.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();

        // Try to get any available adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Marker GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .ok()?;

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Marker Bind Group Layout"),
            entries: &[
                // Image texture (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // Candidates buffer (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results buffer (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform (binding 3)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Dictionary buffer (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Marker Scan Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/marker_scan.wgsl"
            ))),
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Marker Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Marker Scan Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Run marker candidate scanning on the GPU.
    ///
    /// # Arguments
    /// * `image` - Grayscale input image
    /// * `candidates` - Candidate bounding boxes from CPU finder
    /// * `dictionary` - Marker dictionary codes (pairs of low/high u32 for each code)
    /// * `border_bits` - Number of border cells (typically 1)
    /// * `max_hamming` - Maximum hamming distance for valid match (0 for exact, 1 for apriltag)
    ///
    /// # Returns
    /// Vector of results, one per candidate
    pub fn run_candidate_scan(
        &self,
        image: &GrayImage,
        candidates: &[GpuCandidate],
        dictionary: &[u64],
        border_bits: u32,
        max_hamming: u32,
    ) -> Result<Vec<GpuMarkerResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let width = image.width();
        let height = image.height();

        // Check GPU memory budget from environment
        let gpu_budget = gpu_utils::read_gpu_max_bytes_from_env()
            .map_err(|e| Error::FeatureError(format!("GPU memory config error: {}", e)))?;

        // Estimate memory usage for this operation
        // Image texture (f32): width * height * 4 bytes
        // Candidates buffer: num_candidates * 24 bytes
        // Results buffer: num_candidates * 24 bytes
        // Dictionary buffer: dict_size * 8 bytes
        // Params buffer: 32 bytes
        let image_memory = gpu_utils::estimate_image_buffer_size(width, height, 4);
        let candidates_memory = candidates.len() * std::mem::size_of::<GpuCandidate>();
        let results_memory = candidates.len() * std::mem::size_of::<GpuMarkerResult>();
        let dictionary_memory = dictionary.len() * std::mem::size_of::<u64>();
        let total_memory =
            image_memory + candidates_memory + results_memory + dictionary_memory + 32;

        // Check if operation fits in budget
        if !gpu_utils::fits_in_budget(total_memory, gpu_budget) {
            return Err(Error::FeatureError(format!(
                "GPU marker detection requires {}MB but budget is {}MB (set RUSTCV_GPU_MAX_BYTES to increase)",
                total_memory / 1024 / 1024,
                gpu_budget.unwrap_or(0) / 1024 / 1024
            )));
        }

        // Create image texture
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Convert image to f32 normalized format
        let image_data: Vec<f32> = image.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("Marker Image Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&image_data),
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create candidates buffer
        let candidates_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Candidates Buffer"),
                contents: bytemuck::cast_slice(candidates),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create results buffer
        let results_size = (candidates.len() * std::mem::size_of::<GpuMarkerResult>()) as u64;
        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = GpuParams {
            image_width: width,
            image_height: height,
            num_candidates: candidates.len() as u32,
            dict_size: dictionary.len() as u32,
            border_bits,
            threshold: 128,
            max_hamming,
            _padding: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create dictionary buffer (convert u64 to pairs of u32)
        let dict_data: Vec<u32> = dictionary
            .iter()
            .flat_map(|&code| [(code & 0xFFFFFFFF) as u32, (code >> 32) as u32])
            .collect();

        let dictionary_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dictionary Buffer"),
                contents: bytemuck::cast_slice(&dict_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Marker Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dictionary_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Marker Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Marker Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(candidates.len() as u32, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, results_size);

        let index = self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });

        rx.recv()
            .map_err(|e| Error::FeatureError(format!("GPU sync failed: {}", e)))?
            .map_err(|e| Error::FeatureError(format!("GPU buffer map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuMarkerResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// Check if GPU marker detection is available on this system.
pub fn gpu_available() -> bool {
    // Try to create a context - if it succeeds, GPU is available
    MarkerGpuContext::new().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_candidate_struct_size() {
        // Ensure struct sizes match shader expectations
        assert_eq!(std::mem::size_of::<GpuCandidate>(), 24);
        assert_eq!(std::mem::size_of::<GpuMarkerResult>(), 24);
        assert_eq!(std::mem::size_of::<GpuParams>(), 32);
    }

    #[test]
    fn test_gpu_availability_check() {
        // This test just checks that the availability check doesn't panic
        let _ = gpu_available();
    }
}
