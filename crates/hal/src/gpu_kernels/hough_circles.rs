use crate::gpu::GpuContext;
use crate::Result;
use cv_core::{Float, HoughCircle};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CircleParams {
    width: u32,
    height: u32,
    min_radius: f32,
    max_radius: f32,
    num_radii: u32,
    edge_threshold: f32,
}

pub fn hough_circles<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    min_radius: T,
    max_radius: T,
    threshold: u32,
) -> Result<Vec<HoughCircle>> {
    let _precision = crate::gpu_kernels::shader_template::precision_for_type::<T>()?;

    let (h, w) = input.shape.hw();
    let min_radius_f32 = min_radius.to_f32();
    let max_radius_f32 = max_radius.to_f32();

    let num_radii = (max_radius_f32 - min_radius_f32).ceil() as u32 + 1;
    let acc_len = (num_radii * h as u32 * w as u32) as usize;
    let acc_byte_size = (acc_len * 4) as u64;

    // Check memory limit (e.g. 512MB)
    if acc_byte_size > 512 * 1024 * 1024 {
        return Err(crate::Error::MemoryError(
            "Hough Circle accumulator too large. Reduce radius range.".into(),
        ));
    }

    // Accumulator buffer
    let accumulator_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hough Circle Accumulator"),
        size: acc_byte_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = CircleParams {
        width: w as u32,
        height: h as u32,
        min_radius: min_radius_f32,
        max_radius: max_radius_f32,
        num_radii,
        edge_threshold: 50.0, // Default edge threshold
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hough Circle Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/hough_circles.wgsl");
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hough Circle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let vote_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hough Circle Vote"),
            layout: None,
            module: &shader_module,
            entry_point: Some("vote"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Hough Circle BG"),
        layout: &vote_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accumulator_buffer.as_entire_binding(),
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
            label: Some("Hough Circle Dispatch"),
        });

    // Clear accumulator
    encoder.clear_buffer(&accumulator_buffer, 0, None);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hough Circle Vote"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&vote_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (w as u32).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    ctx.submit(encoder);

    // Download accumulator
    let acc_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &accumulator_buffer,
        0,
        acc_byte_size as usize,
    ))?;

    // Peak extraction on CPU
    let mut circles = Vec::new();
    let img_size = (w * h) as u32;

    for i in 0..num_radii {
        let r = min_radius_f32 + i as f32;
        let slice_offset = i * img_size;

        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let val = acc_data[(slice_offset + y * w as u32 + x) as usize];
                if val >= threshold {
                    let mut is_local_max = true;
                    'outer: for dr in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 && dr == 0 {
                                    continue;
                                }

                                let nr = i as i32 + dr;
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;

                                if nr >= 0
                                    && nr < num_radii as i32
                                    && nx >= 0
                                    && nx < w as i32
                                    && ny >= 0
                                    && ny < h as i32
                                {
                                    let n_val = acc_data[((nr as u32 * img_size)
                                        + (ny as u32 * w as u32)
                                        + nx as u32)
                                        as usize];
                                    if n_val > val {
                                        is_local_max = false;
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }

                    if is_local_max {
                        circles.push(HoughCircle::new(x as f32, y as f32, r, val));
                    }
                }
            }
        }
    }

    Ok(circles)
}
