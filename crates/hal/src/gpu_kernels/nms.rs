use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NmsParams {
    num_boxes: u32,
    threshold: f32,
}

pub fn nms_boxes(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    iou_threshold: f32,
) -> Result<Vec<usize>> {
    let num_boxes = input.shape.height;
    if num_boxes == 0 {
        return Ok(Vec::new());
    }
    if num_boxes > 4096 {
        return Err(crate::Error::NotSupported(
            "GPU NMS currently limited to 4096 boxes".into(),
        ));
    }

    let byte_size = (num_boxes * num_boxes * 4) as u64;
    let iou_matrix_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("IOU Matrix"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = NmsParams {
        num_boxes: num_boxes as u32,
        threshold: iou_threshold,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NMS Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/iou_matrix.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NMS Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: iou_matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NMS Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg = (num_boxes as u32).div_ceil(16);
        pass.dispatch_workgroups(wg, wg, 1);
    }
    ctx.submit(encoder);

    // Download IOU matrix
    let matrix_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &iou_matrix_buffer,
        0,
        byte_size as usize,
    ))?;

    // Greedy selection on CPU. Match the CPU backend by processing boxes in
    // descending score order (boxes are laid out [x1, y1, x2, y2, score]).
    let box_bytes = (num_boxes * 5 * 4) as usize;
    let box_data: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        input.storage.buffer(),
        0,
        box_bytes,
    ))?;

    let mut order: Vec<usize> = (0..num_boxes).collect();
    order.sort_by(|&a, &b| {
        box_data[b * 5 + 4]
            .partial_cmp(&box_data[a * 5 + 4])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept = Vec::new();
    let mut suppressed = vec![false; num_boxes];

    for ii in 0..order.len() {
        let i = order[ii];
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        for jj in (ii + 1)..order.len() {
            let j = order[jj];
            if suppressed[j] {
                continue;
            }
            // The IOU matrix only stores the upper triangle (row < col).
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            if matrix_data[lo * num_boxes + hi] != 0 {
                suppressed[j] = true;
            }
        }
    }

    Ok(kept)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NmsPixelParams {
    width: u32,
    height: u32,
    threshold: f32,
    window_radius: i32,
}

pub fn nms_pixel(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    threshold: f32,
    window_size: usize,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU NMS currently only for single-channel".into(),
        ));
    }

    let out_len = w * h;
    let byte_size = (out_len * 4) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let params = NmsPixelParams {
        width: w as u32,
        height: h as u32,
        threshold,
        window_radius: (window_size / 2) as i32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NMS Pixel Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/nms.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NMS Pixel Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NMS Pixel Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (w as u32).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    use std::marker::PhantomData;

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::tensor_ext::TensorToGpu;

    #[test]
    fn nms_boxes_selects_by_descending_score() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // no adapter available
        };

        // Box 0 and box 1 overlap heavily, but box 0 has the lower score, so
        // box 0 (not box 1) must be suppressed. Box 2 does not overlap.
        let data = vec![
            10.0f32, 10.0, 50.0, 50.0, 0.5, // box 0 (low score)
            12.0, 12.0, 52.0, 52.0, 0.9, // box 1 (high score)
            100.0, 100.0, 150.0, 150.0, 0.7, // box 2 (disjoint)
        ];
        let input = cv_core::CpuTensor::<f32>::from_vec(data, cv_core::TensorShape::new(1, 3, 5))
            .unwrap()
            .to_gpu_ctx(&ctx)
            .unwrap();

        let kept = nms_boxes(&ctx, &input, 0.5).unwrap();
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&1), "high-score box must survive: {kept:?}");
        assert!(kept.contains(&2), "disjoint box must survive: {kept:?}");
        assert!(
            !kept.contains(&0),
            "low-score overlapping box suppressed: {kept:?}"
        );
    }
}
