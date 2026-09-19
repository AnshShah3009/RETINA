use crate::gpu::GpuContext;
use crate::Result;
use cv_core::{Float, HoughLine};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HoughParams {
    width: u32,
    height: u32,
    num_rho: u32,
    num_theta: u32,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
}

pub fn hough_lines<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    rho_res: T,
    theta_res: T,
    threshold: u32,
) -> Result<Vec<HoughLine>> {
    // Only f32 WGSL shader available
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Hough Lines GPU kernel only supports f32".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let rho_res_f32 = rho_res.to_f32();
    let theta_res_f32 = theta_res.to_f32();

    // Calculate accumulator size
    let max_rho = (w as f32 * w as f32 + h as f32 * h as f32).sqrt();
    let num_rho = (2.0 * max_rho / rho_res_f32).ceil() as u32;
    let num_theta = (std::f32::consts::PI / theta_res_f32).ceil() as u32;
    let acc_len = (num_rho * num_theta) as usize;
    let acc_byte_size = (acc_len * 4) as u64;

    // Accumulator buffer
    let accumulator_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hough Accumulator"),
        size: acc_byte_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = HoughParams {
        width: w as u32,
        height: h as u32,
        num_rho,
        num_theta,
        rho_res: rho_res_f32,
        theta_res: theta_res_f32,
        threshold,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hough Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Use the f32 shader variant: reads array<f32> directly instead of unpacking
    // u8 bytes from array<u32>. The dispatcher binds f32 storage buffers.
    let shader_source = include_str!("../../shaders/hough_f32.wgsl");
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hough Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let vote_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hough Vote"),
            layout: None,
            module: &shader_module,
            entry_point: Some("vote"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Hough BG"),
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
            label: Some("Hough Dispatch"),
        });

    // Clear accumulator first
    encoder.clear_buffer(&accumulator_buffer, 0, None);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hough Vote"),
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
    let mut lines = Vec::new();
    for r in 0..num_rho {
        for t in 0..num_theta {
            let val = acc_data[(r * num_theta + t) as usize];
            if val >= threshold {
                let mut is_local_max = true;
                for dr in -1..=1 {
                    for dt in -1..=1 {
                        if dr == 0 && dt == 0 {
                            continue;
                        }
                        let nr = r as i32 + dr;
                        let nt = t as i32 + dt;
                        if nr >= 0
                            && nr < num_rho as i32
                            && nt >= 0
                            && nt < num_theta as i32
                            && acc_data[(nr as u32 * num_theta + nt as u32) as usize] > val
                        {
                            is_local_max = false;
                            break;
                        }
                    }
                    if !is_local_max {
                        break;
                    }
                }

                if is_local_max {
                    let rho = (r as f32 - num_rho as f32 / 2.0) * rho_res_f32;
                    let theta = t as f32 * theta_res_f32;
                    lines.push(HoughLine::new(rho, theta, val));
                }
            }
        }
    }

    Ok(lines)
}
