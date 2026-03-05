use crate::gpu::GpuContext;
use crate::gpu_kernels::buffer_utils;
use crate::storage::GpuStorage;
use cv_core::{CpuStorage, Tensor};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Extension trait for transferring data to the GPU.
pub trait TensorToGpu<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> {
    /// Uploads the tensor to the GPU using the global context.
    fn to_gpu(&self) -> crate::Result<Tensor<T, GpuStorage<T>>>;
    /// Uploads the tensor to the GPU using a specific context.
    fn to_gpu_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, GpuStorage<T>>>;
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> TensorToGpu<T>
    for Tensor<T, CpuStorage<T>>
{
    fn to_gpu(&self) -> crate::Result<Tensor<T, GpuStorage<T>>> {
        let ctx = GpuContext::global()?;
        self.to_gpu_ctx(ctx)
    }

    fn to_gpu_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, GpuStorage<T>>> {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor Upload"),
                contents: bytemuck::cast_slice(self.as_slice()?),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(buffer), self.shape.len()),
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        })
    }
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> TensorToGpu<T>
    for Tensor<T, GpuStorage<T>>
{
    fn to_gpu(&self) -> crate::Result<Tensor<T, GpuStorage<T>>> {
        Ok(self.clone())
    }
    fn to_gpu_ctx(&self, _ctx: &GpuContext) -> crate::Result<Tensor<T, GpuStorage<T>>> {
        Ok(self.clone())
    }
}

/// Extension trait for transferring data back to the CPU.
#[allow(async_fn_in_trait)]
pub trait TensorToCpu<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug + Sync + Send> {
    /// Downloads the tensor from the GPU using the global context.
    fn to_cpu(&self) -> crate::Result<Tensor<T, CpuStorage<T>>>;
    /// Downloads the tensor from the GPU using a specific context.
    fn to_cpu_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>>;

    /// ASYNC: Downloads the tensor from the GPU using the global context.
    async fn to_cpu_async(&self) -> crate::Result<Tensor<T, CpuStorage<T>>>;
    /// ASYNC: Downloads the tensor from the GPU using a specific context.
    async fn to_cpu_ctx_async(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>>;
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug + Sync + Send> TensorToCpu<T>
    for Tensor<T, GpuStorage<T>>
{
    fn to_cpu(&self) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        let ctx = GpuContext::global()?;
        self.to_cpu_ctx(ctx)
    }

    fn to_cpu_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        let byte_size = self.storage.len * std::mem::size_of::<T>();

        let data: Vec<T> = pollster::block_on(buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            self.storage.buffer(),
            0,
            byte_size,
        ))?;

        Tensor::from_vec(data, self.shape).map_err(|e| crate::Error::RuntimeError(e.to_string()))
    }

    async fn to_cpu_async(&self) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        let ctx = GpuContext::global()?;
        self.to_cpu_ctx_async(ctx).await
    }

    async fn to_cpu_ctx_async(&self, ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        let byte_size = self.storage.len * std::mem::size_of::<T>();

        let data: Vec<T> = buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            self.storage.buffer(),
            0,
            byte_size,
        )
        .await?;

        Tensor::from_vec(data, self.shape).map_err(|e| crate::Error::RuntimeError(e.to_string()))
    }
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug + Sync + Send> TensorToCpu<T>
    for Tensor<T, CpuStorage<T>>
{
    fn to_cpu(&self) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        Ok(self.clone())
    }
    fn to_cpu_ctx(&self, _ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        Ok(self.clone())
    }

    async fn to_cpu_async(&self) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        Ok(self.clone())
    }
    async fn to_cpu_ctx_async(&self, _ctx: &GpuContext) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        Ok(self.clone())
    }
}

/// Extension trait for type casting on GPU
pub trait TensorCast {
    fn to_f32_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<f32, GpuStorage<f32>>>;
    fn to_u8_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<u8, GpuStorage<u8>>>;
}

impl TensorCast for Tensor<u8, GpuStorage<u8>> {
    fn to_f32_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<f32, GpuStorage<f32>>> {
        let size = self.shape.len();
        let out_byte_size = (size * 4) as u64;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cast u8->f32 Output"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_source = include_str!("../shaders/cast.wgsl");
        let pipeline = ctx.create_compute_pipeline(shader_source, "u8_to_f32");

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cast u8->f32 Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.submit(encoder);

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(output_buffer), size),
            shape: self.shape,
            dtype: cv_core::DataType::F32,
            _phantom: PhantomData,
        })
    }

    fn to_u8_ctx(&self, _ctx: &GpuContext) -> crate::Result<Tensor<u8, GpuStorage<u8>>> {
        Ok(self.clone())
    }
}

impl TensorCast for Tensor<f32, GpuStorage<f32>> {
    fn to_f32_ctx(&self, _ctx: &GpuContext) -> crate::Result<Tensor<f32, GpuStorage<f32>>> {
        Ok(self.clone())
    }

    fn to_u8_ctx(&self, ctx: &GpuContext) -> crate::Result<Tensor<u8, GpuStorage<u8>>> {
        let size = self.shape.len();
        let out_byte_size = ((size + 3) & !3) as u64; // Pack 4 bytes per u32
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cast f32->u8 Output"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_source = include_str!("../shaders/cast.wgsl");
        let pipeline = ctx.create_compute_pipeline(shader_source, "f32_to_u8");

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cast f32->u8 Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                // Note: Bindings 2 and 3 in shader
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 / 4).div_ceil(256);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        ctx.submit(encoder);

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(output_buffer), size),
            shape: self.shape,
            dtype: cv_core::DataType::U8,
            _phantom: PhantomData,
        })
    }
}
