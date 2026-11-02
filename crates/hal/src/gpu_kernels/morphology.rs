use crate::Result;
use crate::context::MorphologyType;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use cv_core::Tensor;
use cv_core::storage::Storage;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MorphParams {
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    typ: u32,
    /// Pixels in the whole tensor, every plane.
    len: u32,
}

/// Erode, dilate, open or close a u8 image on the GPU.
///
/// Bit-identical to the CPU backend: anchor at `(kw / 2, kh / 2)`, replicate
/// border, every nonzero mask byte a live tap, and each CHW plane processed on
/// its own (see `shaders/morphology.wgsl`). Both tensors stay on the device;
/// the mask is bound as packed bytes, not read back.
pub fn morphology(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    typ: MorphologyType,
    kernel: &Tensor<u8, GpuStorage<u8>>,
    iterations: u32,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let len = input.shape.len();
    if iterations == 0 || len == 0 {
        return Ok(input.clone());
    }

    let (h, w) = input.shape.hw();
    let (kh, kw) = kernel.shape.hw();
    if kernel.storage.len() < kh * kw {
        return Err(crate::Error::InvalidInput(format!(
            "morphology kernel shape is {kh}x{kw} but its storage holds {} bytes",
            kernel.storage.len()
        )));
    }
    if input.storage.len() < len {
        return Err(crate::Error::InvalidInput(format!(
            "morphology input shape needs {len} bytes but its storage holds {}",
            input.storage.len()
        )));
    }
    let len_u32 = u32::try_from(len).map_err(|_| {
        crate::Error::InvalidInput(format!("morphology input of {len} pixels exceeds u32"))
    })?;
    let byte_size = (len.div_ceil(4) * 4) as u64;

    // Handle Open/Close
    if typ == MorphologyType::Open || typ == MorphologyType::Close {
        let (op1, op2) = if typ == MorphologyType::Open {
            (MorphologyType::Erode, MorphologyType::Dilate)
        } else {
            (MorphologyType::Dilate, MorphologyType::Erode)
        };
        let tmp = morphology(ctx, input, op1, kernel, iterations)?;
        return morphology(ctx, &tmp, op2, kernel, iterations);
    }

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    // We need an output buffer. If iterations > 1, we also need a temp buffer for ping-pong.
    let output_buffer = ctx.get_buffer(byte_size, usages);
    let temp_buffer: Option<wgpu::Buffer> = if iterations > 1 {
        Some(ctx.get_buffer(byte_size, usages))
    } else {
        None
    };

    // The mask is bound where it already lives. An empty mask (`kw` or `kh`
    // 0) is never read, but the binding still needs a nonempty buffer.
    let empty_mask;
    let k_buf: &wgpu::Buffer = if kernel.storage.buffer().size() >= 4 {
        kernel.storage.buffer()
    } else {
        empty_mask = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Morph Empty Mask"),
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::STORAGE,
            });
        &empty_mask
    };

    let params = MorphParams {
        width: w as u32,
        height: h as u32,
        kw: kw as u32,
        kh: kh as u32,
        typ: if typ == MorphologyType::Erode { 0 } else { 1 },
        len: len_u32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Morph Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/morphology.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Ping-pong state
    // We don't want to modify original input.buffer.
    // So for iteration 0: Input -> Output.
    // If iterations > 1:
    // Iteration 1: Output -> Temp.
    // Iteration 2: Temp -> Output.
    // ...

    let mut current_src: &wgpu::Buffer = input.storage.buffer();
    let mut current_dst: &wgpu::Buffer = &output_buffer;

    for i in 0..iterations {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Morph Pass"),
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Morph Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: current_dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y, z) = super::dispatch_1d(len_u32.div_ceil(4), 64);
            pass.dispatch_workgroups(x, y, z);
        }
        ctx.submit(encoder);

        // Swap for next iteration
        if i + 1 < iterations {
            if i == 0 {
                current_src = &output_buffer;
                current_dst = temp_buffer.as_ref().ok_or_else(|| {
                    crate::Error::KernelError(
                        "temp_buffer missing for multi-iteration morphology".into(),
                    )
                })?;
            } else {
                std::mem::swap(&mut current_src, &mut current_dst);
            }
        }
    }

    // Return the unused buffer to the pool and wrap the result buffer
    let result_handle = if std::ptr::eq(current_dst, &output_buffer) {
        if let Some(tb) = temp_buffer {
            ctx.return_buffer(tb, usages);
        }
        output_buffer
    } else {
        ctx.return_buffer(output_buffer, usages);
        temp_buffer.ok_or_else(|| {
            crate::Error::KernelError("temp_buffer lost in morphology swap".into())
        })?
    };

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(result_handle), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
