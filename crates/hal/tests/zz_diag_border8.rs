// READ-ONLY DIAGNOSTIC #8. Does the GPU execute the WGSL as written?
//
// Dispatches the REAL shaders/convolve_2d.wgsl unmodified, on an input that forces an
// out-of-range storage index for `Reflect` (coord -1 -> index -1). wgpu/naga validates
// out-of-bounds *constant* indices at pipeline creation; if the driver silently
// clamps them instead of erroring, that is the defect, because the CPU and the shader
// source both say this must read the constant 0.0.
use wgpu::util::DeviceExt;

const W: usize = 37;
const H: usize = 33;

#[allow(dead_code)]
#[test]
fn oob_index_probe() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    println!("adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        println!("UNCAUGHT GPU ERROR: {e}");
    }));

    // A 17x7 impulse at the centre: any border mode reads it exactly once from the
    // interior, so the correct answer is kernel[2][2] * 1.0 = 0.140625.
    let w = 17usize;
    let h = 7usize;
    let mut inp = vec![0.0f32; w * h];
    inp[3 * w + 8] = 1.0;
    let mut kern = vec![0.0f32; 25];
    kern[2 * 5 + 2] = 1.0;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&inp),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let kern_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kernel"),
        contents: bytemuck::cast_slice(&kern),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let src = include_str!("../shaders/convolve_2d.wgsl");
    for (label, mode) in [("Replicate(1)", 1u32), ("Reflect(2)", 2u32)] {
        let mut p = Vec::new();
        p.extend_from_slice(&(w as u32).to_ne_bytes());
        p.extend_from_slice(&(h as u32).to_ne_bytes());
        p.extend_from_slice(&5u32.to_ne_bytes());
        p.extend_from_slice(&5u32.to_ne_bytes());
        p.extend_from_slice(&mode.to_ne_bytes());
        p.extend_from_slice(&0.0f32.to_ne_bytes());
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &p,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("p"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kern_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, (w * h * 4) as u64);
        queue.submit([enc.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        read_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        let out: Vec<f32> = {
            let data = read_buf.slice(..).get_mapped_range();
            let v = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            v
        };
        read_buf.unmap();
        // corner (0,0): reflect must use the constant 0.0 because index -1 is OOB
        println!(
            "{label}: out[0,0] = {}   (correct = 0.0, clamped-read = 0.140625)",
            out[0]
        );
        println!(
            "{label}: out[3,8] = {}   (centre, should be 0.140625)",
            out[3 * w + 8]
        );
    }
}
