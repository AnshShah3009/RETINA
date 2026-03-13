//! GPU-accelerated stereo processing
//!
//! Provides GPU implementations of stereo matching algorithms using wgpu.

#![allow(deprecated)]

use crate::{DisparityMap, Result, StereoError};
use cv_core::Error;
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_utils::{estimate_image_buffer_size, fits_in_budget, read_gpu_max_bytes_from_env};
use image::GrayImage;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated stereo matcher
pub struct GpuStereoMatcher {
    ctx: Arc<GpuContext>,
    stereo_pipeline: wgpu::ComputePipeline,
    params_bind_group_layout: wgpu::BindGroupLayout,
    algorithm: GpuStereoAlgorithm,
}

/// Stereo matching algorithm type
#[derive(Clone, Copy)]
pub enum GpuStereoAlgorithm {
    BlockMatching { block_size: u32 },
    Sgm,
}

impl GpuStereoMatcher {
    pub async fn new(algorithm: GpuStereoAlgorithm) -> Result<Self> {
        let global_ctx = GpuContext::global()
            .map_err(|_| Error::InvalidInput("GPU not available or not initialized".to_string()))?;

        let ctx = Arc::new(global_ctx.clone());

        // Create bind group layout for stereo parameters
        let params_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Stereo Params Bind Group Layout"),
                    entries: &[
                        // Left image
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
                        // Right image
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        // Disparity output
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        // Parameters uniform
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
                    ],
                });

        // Create shader module
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Stereo Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "shaders/stereo.wgsl"
                ))),
            });

        // Create compute pipeline
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Stereo Pipeline Layout"),
                bind_group_layouts: &[&params_bind_group_layout],
                immediate_size: 0,
            });

        let stereo_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Stereo Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Ok(Self {
            ctx,
            stereo_pipeline,
            params_bind_group_layout,
            algorithm,
        })
    }

    pub fn compute_disparity(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        min_disparity: i32,
        max_disparity: i32,
    ) -> Result<DisparityMap> {
        if left.width() != right.width() || left.height() != right.height() {
            return Err(Error::DimensionMismatch(
                "Left and right images must have the same dimensions".to_string(),
            ));
        }

        let max_bytes = read_gpu_max_bytes_from_env().unwrap_or(None);

        if let Some(max_bytes) = max_bytes {
            let estimated = estimate_gpu_bytes(left.width(), left.height());
            if estimated > max_bytes {
                return self.compute_disparity_batched(
                    left,
                    right,
                    min_disparity,
                    max_disparity,
                    max_bytes,
                );
            }
        }

        self.compute_disparity_single(left, right, min_disparity, max_disparity)
    }

    fn compute_disparity_single(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        min_disparity: i32,
        max_disparity: i32,
    ) -> Result<DisparityMap> {
        let width = left.width();
        let height = left.height();

        // Create textures
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Create input textures
        let left_texture = self.create_input_texture(left, texture_size);
        let right_texture = self.create_input_texture(right, texture_size);

        // Create output texture for disparity
        let disparity_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Disparity Output"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Create parameter buffer
        let params = StereoParamsGPU {
            width,
            height,
            min_disparity: min_disparity as u32,
            max_disparity: max_disparity as u32,
            block_size: self.block_size(),
        };

        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Stereo Parameters"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Stereo Bind Group"),
                layout: &self.params_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &left_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &right_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &disparity_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create command encoder and compute pass
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stereo Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Stereo Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.stereo_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch compute shader
            let workgroup_size = 16;
            let dispatch_x = (width + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (height + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Copy result to buffer
        let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
        let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &disparity_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
            },
            texture_size,
        );

        let index = self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });

        let data = buffer_slice.get_mapped_range();
        let disparity_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_buffer.unmap();

        // Create disparity map
        let mut disparity = DisparityMap::new(width, height, min_disparity, max_disparity);
        disparity.data = disparity_data;

        Ok(disparity)
    }

    fn compute_disparity_batched(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        min_disparity: i32,
        max_disparity: i32,
        max_bytes: usize,
    ) -> Result<DisparityMap> {
        let width = left.width();
        let height = left.height();
        let half_block = self.block_size() / 2;
        let per_row_bytes = estimate_gpu_bytes(width, 1);

        if max_bytes < per_row_bytes {
            return Err(Error::InvalidInput(format!(
                "RUSTCV_GPU_MAX_BYTES={} is too small; need at least {} bytes for one row",
                max_bytes, per_row_bytes
            )));
        }

        let max_input_rows = (max_bytes / per_row_bytes) as u32;
        if max_input_rows < (2 * half_block + 1) {
            return Err(Error::InvalidInput(format!(
                "RUSTCV_GPU_MAX_BYTES={} too small for block_size={} (requires at least {} rows)",
                max_bytes,
                self.block_size(),
                2 * half_block + 1
            )));
        }

        // Process output rows in chunks while preserving halo for block matching near tile edges.
        let max_output_rows = (max_input_rows - 2 * half_block).max(1);
        let mut disparity = DisparityMap::new(width, height, min_disparity, max_disparity);
        let width_usize = width as usize;
        let mut output_start = 0u32;

        while output_start < height {
            let output_end = (output_start + max_output_rows).min(height);
            let input_start = output_start.saturating_sub(half_block);
            let input_end = (output_end + half_block).min(height);

            let left_tile = extract_rows(left, input_start, input_end)?;
            let right_tile = extract_rows(right, input_start, input_end)?;
            let tile = self.compute_disparity_single(
                &left_tile,
                &right_tile,
                min_disparity,
                max_disparity,
            )?;

            let local_start = (output_start - input_start) as usize;
            let local_end = (output_end - input_start) as usize;
            for local_row in local_start..local_end {
                let global_row = output_start as usize + (local_row - local_start);
                let src = &tile.data[local_row * width_usize..(local_row + 1) * width_usize];
                let dst =
                    &mut disparity.data[global_row * width_usize..(global_row + 1) * width_usize];
                dst.copy_from_slice(src);
            }

            output_start = output_end;
        }

        Ok(disparity)
    }

    fn create_input_texture(&self, image: &GrayImage, size: wgpu::Extent3d) -> wgpu::Texture {
        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input Image"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload image data
        self.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width()),
                rows_per_image: Some(image.height()),
            },
            size,
        );

        texture
    }

    fn block_size(&self) -> u32 {
        match self.algorithm {
            GpuStereoAlgorithm::BlockMatching { block_size } => block_size,
            GpuStereoAlgorithm::Sgm => 11,
        }
    }
}

fn estimate_gpu_bytes(width: u32, height: u32) -> usize {
    // Approx: left r8 + right r8 + disparity r32 + readback buffer r32.
    // Add fixed overhead so guard errs on the safe side.
    let pixels = width as usize * height as usize;
    pixels * 10 + 4096
}

fn extract_rows(image: &GrayImage, start_row: u32, end_row: u32) -> Result<GrayImage> {
    if start_row >= end_row || end_row > image.height() {
        return Err(Error::InvalidInput(format!(
            "Invalid row range [{start_row}, {end_row}) for image height {}",
            image.height()
        )));
    }

    let width = image.width() as usize;
    let start_idx = start_row as usize * width;
    let end_idx = end_row as usize * width;
    let data = image.as_raw()[start_idx..end_idx].to_vec();
    GrayImage::from_raw(image.width(), end_row - start_row, data)
        .ok_or_else(|| Error::InvalidInput("Failed to build temporary image tile".to_string()))
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StereoParamsGPU {
    width: u32,
    height: u32,
    min_disparity: u32,
    max_disparity: u32,
    block_size: u32,
}

/// Check if GPU acceleration is available
pub async fn is_gpu_available() -> bool {
    GpuContext::is_available_async().await
}

/// Enumerate adapters visible to wgpu on this machine.
pub async fn enumerate_adapters() -> Vec<wgpu::AdapterInfo> {
    GpuContext::enumerate_adapters()
        .await
        .into_iter()
        .map(|adapter| adapter.get_info())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 64u32;
        let height = 64u32;

        let mut left = GrayImage::new(width, height);
        let mut right = GrayImage::new(width, height);

        // Create pattern with positive disparity
        // x_right = x_left - disparity
        // so right(x) = left(x + disparity)
        for y in 0..height {
            for x in 0..width {
                let pattern = ((x / 8) % 2) * 200;
                left.put_pixel(x, y, Luma([pattern as u8]));

                let sx = (x + 5).min(width - 1);
                let right_pattern = ((sx / 8) % 2) * 200;
                right.put_pixel(x, y, Luma([right_pattern as u8]));
            }
        }

        (left, right)
    }

    #[tokio::test]
    async fn test_gpu_availability() {
        let available = is_gpu_available().await;
        println!("GPU available: {}", available);

        let adapters = enumerate_adapters().await;
        for info in &adapters {
            println!(
                "Adapter: name='{}', type={:?}, backend={:?}",
                info.name, info.device_type, info.backend
            );
        }
        let integrated = adapters
            .iter()
            .any(|a| a.device_type == wgpu::DeviceType::IntegratedGpu);
        println!("Integrated GPU visible: {}", integrated);
    }

    #[tokio::test]
    async fn test_gpu_block_matching() {
        if !is_gpu_available().await {
            println!("Skipping GPU functional test (no GPU available)");
            return;
        }

        let (left, right) = create_test_stereo_pair();

        let matcher = GpuStereoMatcher::new(GpuStereoAlgorithm::BlockMatching { block_size: 7 })
            .await
            .unwrap();
        let disparity = matcher.compute_disparity(&left, &right, 0, 10).unwrap();

        assert_eq!(disparity.width, 64);
        assert_eq!(disparity.height, 64);

        // Center should have disparity approx 5 (from create_test_stereo_pair)
        let d = disparity.get(32, 32);
        println!("GPU Disparity at center: {}", d);
        assert!((d - 5.0).abs() < 1.0);
    }
}
