//! A/B timing of naga's integer division polyfill, on vs off.
//!
//! naga wraps every integer `/` and `%` in a guard that swaps a zero divisor
//! (and `INT_MIN / -1`) for 1. wgpu 30 can drop the guard per shader module
//! with `ShaderRuntimeChecks { int_div_checks: false, .. }`, which is sound
//! only once every divisor is proven nonzero - see
//! `docs/wgsl_int_div_audit.md`. This harness answers whether that is worth
//! doing: it builds each kernel twice from the same WGSL, once per setting,
//! binds identical buffers, and times both with GPU timestamp queries,
//! alternating the order every round so drift and clocks hit both sides
//! equally. It reports the median per-dispatch time of each.
//!
//! The kernels are the integer-heavy ones in the audit (border folding with a
//! runtime `%` per tap, linear-index `%`/`/` per pixel, patch-index `%`/`/`
//! per sample), plus controls with no runtime division and a synthetic loop
//! that does nothing but divide.
//!
//! Ignored by default: it needs a GPU with `TIMESTAMP_QUERY` and takes a few
//! seconds. Run it with
//!
//! ```text
//! cargo test --release -p cv-hal --test int_div_checks_timing -- --ignored --nocapture
//! ```
//!
//! `ROUNDS=<n>` sets the number of samples per side (default 41).

use cv_hal::gpu::GpuContext;
use wgpu::naga;
use wgpu::util::DeviceExt;

const W: u32 = 3840;
const H: u32 = 2160;
/// Dispatches recorded back to back inside one timed pass.
const DISPATCHES_PER_SAMPLE: u32 = 10;

struct Case {
    name: &'static str,
    wgsl: String,
    entry: &'static str,
    /// The uniform block, as u32 words (f32 fields are unused by these cases).
    uniform: Vec<u32>,
    groups: (u32, u32, u32),
}

const SYNTHETIC: &str = r#"
struct P { d: u32, n: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    var acc = g.x;
    for (var i = 0u; i < p.n; i++) {
        acc = acc + (acc ^ i) % (p.d + i) + acc / (p.d + (i & 7u));
    }
    o[g.x] = acc;
}
"#;

fn cases() -> Vec<Case> {
    let read =
        |p: &str| std::fs::read_to_string(format!("{}/{p}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let gauss = read("shaders/gaussian_blur_separable.wgsl");
    let conv = read("shaders/convolve_2d.wgsl");
    let canny = read("shaders/canny.wgsl");
    let image_groups = (W.div_ceil(16), H.div_ceil(16), 1);
    let linear = |n: u32| {
        let g = n.div_ceil(256);
        (g.min(65535), g.div_ceil(65535), 1)
    };
    let case = |name, wgsl: &String, entry, uniform: &[u32], groups| Case {
        name,
        wgsl: wgsl.clone(),
        entry,
        uniform: uniform.to_vec(),
        groups,
    };
    vec![
        case(
            "synthetic u32 % and / loop, 64 iters",
            &SYNTHETIC.to_string(),
            "main",
            &[7, 64, 0, 0],
            (65535, 1, 1),
        ),
        // Params: width, height, kernel_size, is_vertical, border_mode, channels.
        case(
            "gaussian 4K k31 horiz Reflect101",
            &gauss,
            "main",
            &[W, H, 31, 0, 4, 1],
            image_groups,
        ),
        case(
            "gaussian 4K k31 vert Wrap",
            &gauss,
            "main",
            &[W, H, 31, 1, 3, 1],
            image_groups,
        ),
        case(
            "gaussian 4K k31 horiz Replicate (no %)",
            &gauss,
            "main",
            &[W, H, 31, 0, 1, 1],
            image_groups,
        ),
        // Params: width, height, kernel_width, kernel_height, border_mode.
        case(
            "convolve_2d 4K 7x7 Reflect",
            &conv,
            "main",
            &[W, H, 7, 7, 2, 0],
            image_groups,
        ),
        case(
            "convolve_2d 4K 7x7 Wrap",
            &conv,
            "main",
            &[W, H, 7, 7, 3, 0],
            image_groups,
        ),
        // Params: src_w, src_h, dst_w, dst_h, interpolation (2 = cubic), border (2 = wrap).
        case(
            "remap 4K u8 bicubic Wrap",
            &read("shaders/remap.wgsl"),
            "main",
            &[W, H, W, H, 2, 2, 0],
            ((W / 4).div_ceil(16), H.div_ceil(16), 1),
        ),
        // Params: width, height, low, high, l2, tg22.
        case(
            "canny sobel 4K",
            &canny,
            "sobel",
            &[W, H, 50, 150, 0, 13573, 0, 0],
            linear(W * H),
        ),
        case(
            "canny nms 4K",
            &canny,
            "nms",
            &[W, H, 50, 150, 0, 13573, 0, 0],
            linear(W * H / 4),
        ),
        // Params: width, height, num_rho, num_angle, rho_offset, ambiguous_capacity.
        case(
            "hough vote 1080p, 180 angles",
            &read("shaders/hough.wgsl"),
            "vote",
            &[1920, 1080, 4400, 180, 2200, 1024, 0, 0],
            linear(1920 * 1080),
        ),
        // Params: src_w, src_h, dst_w, dst_h, channels.
        case(
            "resize u8 4K -> 1080p (literal /4u)",
            &read("shaders/resize.wgsl"),
            "main",
            &[W, H, 1920, 1080, 1, 0, 0, 0],
            ((1920 / 4u32).div_ceil(16), 1080u32.div_ceil(16), 1),
        ),
        case(
            "pyramid_down 4K (no integer division)",
            &read("shaders/pyramid_down.wgsl"),
            "main",
            &[W, H, W / 2, H / 2, 1, 0, 0, 0],
            ((W / 2).div_ceil(16), (H / 2).div_ceil(16), 1),
        ),
        // Params: width, height, num_keypoints, radius.
        case(
            "ORB orientation r15, 20k keypoints",
            &read("src/gpu_kernels/orientation.wgsl"),
            "main",
            &[W, H, 20000, 15],
            (20000, 1, 1),
        ),
    ]
}

fn module(device: &wgpu::Device, wgsl: &str, int_div_checks: bool) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    };
    if int_div_checks {
        return device.create_shader_module(desc);
    }
    let checks = wgpu::ShaderRuntimeChecks {
        int_div_checks: false,
        ..wgpu::ShaderRuntimeChecks::checked()
    };
    // SAFETY: only the integer-division guard is dropped. Every case binds
    // nonzero divisors (the uniforms above, and the synthetic loop's
    // `p.d + i` with d = 7), so no division by zero or INT_MIN / -1 occurs.
    unsafe { device.create_shader_module_trusted(desc, checks) }
}

/// Storage and uniform bindings `entry` reaches, as `(binding, is_uniform)`.
fn bindings(wgsl: &str, entry: &str) -> Vec<(u32, bool)> {
    let module = naga::front::wgsl::parse_str(wgsl).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let ep = module
        .entry_points
        .iter()
        .position(|e| e.name == entry)
        .unwrap();
    let ep_info = info.get_entry_point(ep);
    module
        .global_variables
        .iter()
        .filter(|(h, _)| !ep_info[*h].is_empty())
        .filter_map(|(_, g)| {
            let b = g.binding.as_ref()?;
            Some((b.binding, g.space == naga::AddressSpace::Uniform))
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Interquartile range as a percentage of the median.
fn iqr_pct(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    (sorted[n * 3 / 4] - sorted[n / 4]) / sorted[n / 2] * 100.0
}

#[test]
#[ignore = "GPU timing harness; run with --ignored --nocapture"]
fn int_div_checks_on_vs_off() {
    let ctx = match pollster::block_on(GpuContext::init_global()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("skipping: no adapter ({e})");
            return;
        }
    };
    let device = &ctx.device;
    if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        println!("skipping: adapter has no TIMESTAMP_QUERY");
        return;
    }
    let rounds: usize = std::env::var("ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(41);
    let tick_ns = ctx.queue.get_timestamp_period() as f64;

    // One deterministic pattern in [-100, 4000) for every storage binding:
    // in range for images, negative and out-of-bounds for remap coordinates,
    // so border folding actually runs.
    let pattern: Vec<f32> = (0..(W * H * 4 + 1024))
        .map(|i| (i.wrapping_mul(2654435761) >> 8) as f32 % 4100.0 - 100.0)
        .collect();

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    println!(
        "{:<40} {:>11} {:>11} {:>7}   IQR checked / unchecked",
        "kernel (median us per dispatch)", "checked", "unchecked", "delta"
    );
    for case in cases() {
        let mut uniform: Vec<u8> = bytemuck::cast_slice(&case.uniform).to_vec();
        uniform.resize(uniform.len().div_ceil(16) * 16, 0);
        let buffers: Vec<(u32, wgpu::Buffer)> = bindings(&case.wgsl, case.entry)
            .into_iter()
            .map(|(binding, is_uniform)| {
                let (contents, usage) = if is_uniform {
                    (&uniform[..], wgpu::BufferUsages::UNIFORM)
                } else {
                    (bytemuck::cast_slice(&pattern), wgpu::BufferUsages::STORAGE)
                };
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage,
                });
                (binding, buf)
            })
            .collect();

        // Index 0 = checks on (the default), 1 = checks off.
        let variants: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)> = [true, false]
            .into_iter()
            .map(|checks| {
                let module = module(device, &case.wgsl, checks);
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some(case.entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let entries: Vec<wgpu::BindGroupEntry> = buffers
                    .iter()
                    .map(|(binding, buf)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buf.as_entire_binding(),
                    })
                    .collect();
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                (pipeline, bind_group)
            })
            .collect();

        let mut samples = [Vec::new(), Vec::new()];
        // Two warm-up rounds, then alternate which side goes first.
        for round in 0..rounds + 2 {
            for slot in 0..2 {
                let v = if round % 2 == 0 { slot } else { 1 - slot };
                let (pipeline, bind_group) = &variants[v];
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &query_set,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    for _ in 0..DISPATCHES_PER_SAMPLE {
                        pass.dispatch_workgroups(case.groups.0, case.groups.1, case.groups.2);
                    }
                }
                enc.resolve_query_set(&query_set, 0..2, &resolve, 0);
                enc.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
                ctx.queue.submit([enc.finish()]);
                readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let ticks = {
                    let view = readback.slice(..).get_mapped_range().unwrap();
                    let ts: &[u64] = bytemuck::cast_slice(&view);
                    ts[1] - ts[0]
                };
                readback.unmap();
                if round >= 2 {
                    samples[v].push(ticks as f64 * tick_ns / 1000.0 / DISPATCHES_PER_SAMPLE as f64);
                }
            }
        }
        let on = median(&mut samples[0]);
        let off = median(&mut samples[1]);
        println!(
            "{:<40} {:>11.2} {:>11.2} {:>+6.1}%   {:.1}% / {:.1}%",
            case.name,
            on,
            off,
            (off - on) / on * 100.0,
            iqr_pct(&samples[0]),
            iqr_pct(&samples[1]),
        );
    }
}
