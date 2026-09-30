// MINIMAL REPRODUCTION of the convolve_2d GPU/CPU divergence.
//
// Defect: in shaders/convolve_2d.wgsl, get_input_index() returns a SINGLE scalar for a
// 2-D sample, `iy * w + ix`. The CPU (cpu/border.rs map_border_coord) returns
// Option<(ix, iy)> = None when a Constant-border sample is out of range, and the CPU
// loop then substitutes the constant. The WGSL only has an out-of-range escape
// (`return -1`) inside the `border_mode == 0u32` branch, so for Reflect/Reflect101/Wrap
// an out-of-range coordinate silently becomes a WRONG IN-BOUNDS index instead.
//
// For Reflect that out-of-range coordinate is reached on every pixel of the top and
// left edges: at y=0, ky=0 gives iy = -1, and reflect(-1, h) = h-1, so the index is
// (h-1)*w + ix -- a completely different pixel, near the bottom of the image.
//
// The "clamped" out-of-range value below is not the GPU clamping an address; it is this
// aliasing. Correct `Reflect` on this input would have produced 0.0 at (0,0).
use wgpu::util::DeviceExt;

const W: usize = 37;
const H: usize = 33;

fn get_input_index_buggy(x: i32, y: i32, w: i32, h: i32, mode: u32) -> i32 {
    let mut ix = x;
    let mut iy = y;
    if mode == 0u32 {
        if ix < 0 || ix >= w || iy < 0 || iy >= h {
            return -1;
        }
    } else if mode == 1u32 {
        ix = ix.clamp(0, w - 1);
        iy = iy.clamp(0, h - 1);
    } else if mode == 2u32 {
        if w == 1 {
            ix = 0;
        } else {
            let period = 2 * w;
            ix = ix % period;
            if ix < 0 {
                ix += period;
            }
            if ix >= w {
                ix = period - ix - 1;
            }
        }
        if h == 1 {
            iy = 0;
        } else {
            let period = 2 * h;
            iy = iy % period;
            if iy < 0 {
                iy += period;
            }
            if iy >= h {
                iy = period - iy - 1;
            }
        }
    } else if mode == 3u32 {
        ix = ix % w;
        if ix < 0 {
            ix += w;
        }
        iy = iy % h;
        if iy < 0 {
            iy += h;
        }
    } else if mode == 4u32 {
        if w == 1 {
            ix = 0;
        } else {
            let period = 2 * w - 2;
            ix = ix % period;
            if ix < 0 {
                ix += period;
            }
            if ix >= w {
                ix = period - ix;
            }
        }
        if h == 1 {
            iy = 0;
        } else {
            let period = 2 * h - 2;
            iy = iy % period;
            if iy < 0 {
                iy += period;
            }
            if iy >= h {
                iy = period - iy;
            }
        }
    }
    iy * w + ix
}

// What the CPU does: Option<(ix, iy)>, None -> constant fill.
fn map_border_coord_buggy_free(x: i32, y: i32, w: i32, h: i32, mode: u32) -> Option<(i32, i32)> {
    let f = |c: i32, n: i32| -> Option<i32> {
        if n <= 0 {
            return None;
        }
        match mode {
            0 => {
                if c < 0 || c >= n {
                    None
                } else {
                    Some(c)
                }
            }
            1 => Some(c.clamp(0, n - 1)),
            3 => Some(c.rem_euclid(n)),
            2 => {
                if n == 1 {
                    return Some(0);
                }
                let p = 2 * n;
                let mut t = c.rem_euclid(p);
                if t >= n {
                    t = p - t - 1;
                }
                Some(t)
            }
            _ => {
                if n == 1 {
                    return Some(0);
                }
                let p = 2 * n - 2;
                let mut t = c.rem_euclid(p);
                if t >= n {
                    t = p - t;
                }
                Some(t)
            }
        }
    };
    match (f(x, w), f(y, h)) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    let src = include_str!("../shaders/convolve_2d.wgsl");
    let g1 = [0.0625f32, 0.25, 0.375, 0.25, 0.0625];
    let mut kern = vec![0.0f32; 25];
    for r in 0..5 {
        for c in 0..5 {
            kern[r * 5 + c] = g1[r] * g1[c];
        }
    }
    let vals: Vec<f32> = (0..H * W).map(|i| i as f32).collect();
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let kern_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&kern),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (H * W * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (H * W * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    for (label, mode) in [
        ("Constant(0)", 0u32),
        ("Replicate", 1u32),
        ("Reflect", 2u32),
        ("Wrap", 3u32),
        ("Reflect101", 4u32),
    ] {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&(W as u32).to_ne_bytes());
        p.extend_from_slice(&(H as u32).to_ne_bytes());
        p.extend_from_slice(&5u32.to_ne_bytes());
        p.extend_from_slice(&5u32.to_ne_bytes());
        p.extend_from_slice(&mode.to_ne_bytes());
        p.extend_from_slice(&0.0f32.to_ne_bytes());
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &p,
            usage: wgpu::BufferUsages::UNIFORM,
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
            pass.dispatch_workgroups(W.div_ceil(16) as u32, H.div_ceil(16) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, (H * W * 4) as u64);
        queue.submit([enc.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        let gpu: Vec<f32> = {
            read_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            let d = read_buf.slice(..).get_mapped_range();
            let v = bytemuck::cast_slice(&d).to_vec();
            drop(d);
            v
        };
        read_buf.unmap();

        let cpu = cpu_model(&vals, &kern, mode);
        let nd = cpu
            .iter()
            .zip(&gpu)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4)
            .count();
        let worst = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Also report what the WGSL's scalar index would do, evaluated in Rust.
        let bug = rust_model_of_shader(&vals, &kern, mode, W as i32, H as i32);
        let nd_bug = cpu
            .iter()
            .zip(&bug)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4)
            .count();
        let worst_bug = cpu
            .iter()
            .zip(&bug)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{label:>12}: GPU vs CPU ndiff={nd:>5} worst={worst:>9.4}   |   shader-as-written (rust model) ndiff={nd_bug:>5} worst={worst_bug:>9.4}");
    }
}

fn cpu_model(src: &[f32], k: &[f32], mode: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; W * H];
    for y in 0..H {
        for x in 0..W {
            let mut s = 0.0f32;
            for ky in 0..5 {
                for kx in 0..5 {
                    let v = match map_border_coord_buggy_free(
                        x as i32 + kx as i32 - 2,
                        y as i32 + ky as i32 - 2,
                        W as i32,
                        H as i32,
                        mode,
                    ) {
                        Some((a, b)) => src[b as usize * W + a as usize],
                        None => 0.0,
                    };
                    s += v * k[ky * 5 + kx];
                }
            }
            out[y * W + x] = s;
        }
    }
    out
}

fn rust_model_of_shader(src: &[f32], k: &[f32], mode: u32, w: i32, h: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; W * H];
    for y in 0..H {
        for x in 0..W {
            let mut s = 0.0f32;
            for ky in 0..5 {
                for kx in 0..5 {
                    let i = get_input_index_buggy(
                        x as i32 + kx as i32 - 2,
                        y as i32 + ky as i32 - 2,
                        w,
                        h,
                        mode,
                    );
                    let v = if i >= 0 { src[i as usize] } else { 0.0 };
                    s += v * k[ky * 5 + kx];
                }
            }
            out[y * W + x] = s;
        }
    }
    out
}
