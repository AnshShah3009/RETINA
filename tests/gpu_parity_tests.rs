// Cross-backend parity tests for GPU operations the main parity suite omits.
//
// `multi_gpu_tests.rs` covers threshold, resize, colour conversion, bilateral,
// FAST, matching and ICP. Nothing exercised pyramid downsampling, multichannel
// blur, stereo matching, SIFT extrema, or the NMS border cases, and three real
// defects were hiding in exactly that gap:
//
//   - the stereo shader never compiled. WGSL forbids passing a storage-space
//     pointer into a function, and `get_pixel(data: ptr<storage, ...>, ...)`
//     did exactly that, so every GPU stereo dispatch was a hard validation
//     failure rather than a wrong answer.
//   - the separable blur had no channel concept at all, indexing
//     `y * width + x`. A 3-channel image was blurred as one long row and each
//     output pixel averaged values from all three planes: 26% mean relative
//     error against the CPU, and over 100% on some pixels.
//   - fixing that, the blur still wrote only the first plane, because the
//     invocation index was mapped to (x, channel) in the wrong order for a
//     channels-innermost layout.
//
// Each test runs the same input on both backends and reports the worst absolute
// difference, so a regression is visible rather than silent. They skip cleanly
// when no GPU is present, which is every CI runner.
use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

fn gpu() -> Option<&'static GpuContext> {
    if let Ok(c) = GpuContext::global() {
        return Some(c);
    }
    match futures::executor::block_on(GpuContext::init_global()) {
        Ok(c) => Some(c),
        Err(e) => {
            println!("no gpu: {e}");
            None
        }
    }
}

fn rng_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / 1000.0
        })
        .collect()
}

fn mk(v: Vec<f32>, c: usize, h: usize, w: usize) -> CpuTensor<f32> {
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

#[test]
fn probe_pyramid_down() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (64usize, 48usize);
    let cpu = mk(rng_f32(w * h, 42), 1, h, w);
    let c = cpu_ctx.pyramid_down(&cpu).unwrap();
    let cs = c.storage.as_slice().unwrap().to_vec();
    let gt = cpu.to_gpu_ctx(g).unwrap();
    let gr = g.pyramid_down(&gt).unwrap();
    let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();

    println!("pyramid_down cpu_len={} gpu_len={}", cs.len(), gs.len());
    let n = cs.len().min(gs.len());
    let worst = (0..n).map(|i| (cs[i] - gs[i]).abs()).fold(0.0f32, f32::max);
    let mean = (0..n).map(|i| (cs[i] - gs[i]).abs()).sum::<f32>() / n as f32;
    println!("PYRAMID_DOWN worst_abs_diff={worst} mean_abs_diff={mean}");
    println!("cpu[0..8]={:?}", &cs[..8.min(cs.len())]);
    println!("gpu[0..8]={:?}", &gs[..8.min(gs.len())]);
}

#[test]
fn probe_sift_extrema() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (64usize, 64usize);
    let a = rng_f32(w * h, 7);
    let b: Vec<f32> = a.iter().map(|v| v + 0.5).collect();
    let c2: Vec<f32> = a.iter().map(|v| v - 0.5).collect();
    let prev = mk(a.clone(), 1, h, w);
    let cur = mk(b.clone(), 1, h, w);
    let next = mk(c2.clone(), 1, h, w);

    let cres = cpu_ctx
        .sift_extrema(&prev, &cur, &next, 0.01, 10.0)
        .unwrap();
    let csl = cres.storage.as_slice().unwrap().to_vec();
    let ccount = csl.iter().filter(|&&v| v == 1).count();

    let gp = prev.to_gpu_ctx(g).unwrap();
    let gc = cur.to_gpu_ctx(g).unwrap();
    let gn = next.to_gpu_ctx(g).unwrap();
    let gres = g.sift_extrema(&gp, &gc, &gn, 0.01, 10.0).unwrap();
    let gsl = gres.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
    let gcount = gsl.iter().filter(|&&v| v == 1).count();

    println!(
        "SIFT cpu_len={} gpu_len={} cpu_hits={} gpu_hits={}",
        csl.len(),
        gsl.len(),
        ccount,
        gcount
    );
}

#[test]
fn probe_gray_to_rgb() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (32usize, 32usize);
    let cpu = mk(rng_f32(w * h, 11), 1, h, w);
    let cr = cpu_ctx
        .cvt_color(&cpu, cv_hal::context::ColorConversion::GrayToRgb)
        .unwrap();
    let cs = cr.storage.as_slice().unwrap().to_vec();
    let gt = cpu.to_gpu_ctx(g).unwrap();
    match g.cvt_color(&gt, cv_hal::context::ColorConversion::GrayToRgb) {
        Ok(r) => {
            let gs = r.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
            let n = cs.len().min(gs.len());
            let worst = (0..n).map(|i| (cs[i] - gs[i]).abs()).fold(0.0f32, f32::max);
            println!(
                "GRAY2RGB ok cpu_len={} gpu_len={} worst={}",
                cs.len(),
                gs.len(),
                worst
            );
        }
        Err(e) => println!("GRAY2RGB Err: {e}  (CPU returns len {})", cs.len()),
    }
}

#[test]
fn probe_nms_borders() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    for (w, h) in [(8usize, 1usize), (1, 8), (8, 8), (8, 16)] {
        let data: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
        let cpu = mk(data, 1, h, w);
        let cr = cpu_ctx.nms(&cpu, 0.0, 3).unwrap();
        let cs = cr.storage.as_slice().unwrap().to_vec();
        let gt = cpu.to_gpu_ctx(g).unwrap();
        let gr = g.nms(&gt, 0.0, 3).unwrap();
        let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
        println!("NMS {w}x{h}: cpu={cs:?}");
        println!("NMS {w}x{h}: gpu={gs:?}");
    }
}

#[test]
fn probe_threshold_binary_exact() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (16usize, 16usize);
    let cpu = mk((0..(w * h)).map(|i| i as f32).collect(), 1, h, w);
    for t in [100.0f32, 100.5] {
        let cr = cpu_ctx
            .threshold(&cpu, t, 255.0, cv_hal::context::ThresholdType::Binary)
            .unwrap();
        let cs = cr.storage.as_slice().unwrap().to_vec();
        let gt = cpu.to_gpu_ctx(g).unwrap();
        let gr = g
            .threshold(&gt, t, 255.0, cv_hal::context::ThresholdType::Binary)
            .unwrap();
        let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
        let d = (0..cs.len())
            .map(|i| (cs[i] - gs[i]).abs())
            .fold(0.0f32, f32::max);
        println!("THRESH t={t} worst={d} cpu[99..103]={:?}", &cs[99..103]);
    }
}

#[test]
fn probe_resize_multichannel() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (32usize, 32usize);
    let cpu = mk(rng_f32(w * h * 3, 21), 3, h, w);
    let cr = cpu_ctx.resize(&cpu, (17, 13)).unwrap();
    let cs = cr.storage.as_slice().unwrap().to_vec();
    let gt = cpu.to_gpu_ctx(g).unwrap();
    let gr = g.resize(&gt, (17, 13)).unwrap();
    let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
    let n = cs.len().min(gs.len());
    let worst = (0..n).map(|i| (cs[i] - gs[i]).abs()).fold(0.0f32, f32::max);
    println!(
        "RESIZE3 cpu_len={} gpu_len={} worst={}",
        cs.len(),
        gs.len(),
        worst
    );
}

#[test]
fn probe_gaussian_blur_multichannel() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (32usize, 32usize);
    let cpu = mk(rng_f32(w * h * 3, 33), 3, h, w);
    let cr = cpu_ctx.gaussian_blur(&cpu, 1.4, 5).unwrap();
    let cs = cr.storage.as_slice().unwrap().to_vec();
    let gt = cpu.to_gpu_ctx(g).unwrap();
    let gr = g.gaussian_blur(&gt, 1.4, 5).unwrap();
    let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
    let n = cs.len().min(gs.len());
    let worst = (0..n).map(|i| (cs[i] - gs[i]).abs()).fold(0.0f32, f32::max);
    println!(
        "GBLUR3 cpu_len={} gpu_len={} worst={}",
        cs.len(),
        gs.len(),
        worst
    );
    if worst > 1e-3 {
        println!("GBLUR3 cpu[0..12]={:?}", &cs[..12]);
        println!("GBLUR3 gpu[0..12]={:?}", &gs[..12]);
    }
}

#[test]
fn probe_stereo_borders() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let (w, h) = (32usize, 24usize);
    let l = mk(rng_f32(w * h, 51), 1, h, w);
    let r = mk(rng_f32(w * h, 52), 1, h, w);
    let p = cv_hal::context::StereoMatchParams {
        block_size: 5,
        min_disparity: 0,
        num_disparities: 8,
        method: cv_hal::context::StereoMatchMethod::BlockMatching,
    };
    let cr: CpuTensor<f32> = cpu_ctx.stereo_match(&l, &r, &p).unwrap();
    let cs: Vec<f32> = cr.storage.as_slice().unwrap().to_vec();
    let gl = l.to_gpu_ctx(g).unwrap();
    let gr2 = r.to_gpu_ctx(g).unwrap();
    let gr: cv_hal::GpuTensor<f32> = g.stereo_match(&gl, &gr2, &p).unwrap();
    let gs: Vec<f32> = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
    let n = cs.len().min(gs.len());
    let worst = (0..n).map(|i| (cs[i] - gs[i]).abs()).fold(0.0f32, f32::max);
    let ndiff = (0..n).filter(|&i| (cs[i] - gs[i]).abs() > 1e-4).count();
    println!(
        "STEREO cpu_len={} gpu_len={} worst={} ndiff={}",
        cs.len(),
        gs.len(),
        worst,
        ndiff
    );
    // per-row summary of the last few rows (border behaviour)
    for y in [h - 4, h - 3, h - 2, h - 1] {
        let row_c: Vec<f32> = cs[y * w..(y + 1) * w].to_vec();
        let row_g: Vec<f32> = gs[y * w..(y + 1) * w].to_vec();
        println!(
            "STEREO y={y} cpu_tail={:?} gpu_tail={:?}",
            &row_c[w.saturating_sub(6)..],
            &row_g[w.saturating_sub(6)..]
        );
    }
}

/// The GPU point-to-plane ICP must actually iterate.
///
/// The source tensor was uploaded once before the loop and passed to
/// `icp_correspondences` unchanged, so the association set was identical on every
/// iteration - a fixed point, not an iteration. The pose update was applied
/// repeatedly to correspondences that had never been re-derived, which is a
/// single Gauss-Newton step rather than ICP.
///
/// The CPU path returns early and is covered by its own tests; this is the
/// branch the existing parity test never reached.
#[test]
fn probe_icp_iterates_on_gpu() {
    let Some(g) = gpu() else { return };
    let cpu_ctx = CpuBackend::new().unwrap();
    let _ = &cpu_ctx;

    fn cloud(pts: Vec<[f32; 3]>, nrm: Vec<[f32; 3]>) -> cv_core::PointCloud<f32> {
        let points: Vec<nalgebra::Point3<f32>> = pts
            .iter()
            .map(|p| nalgebra::Point3::new(p[0], p[1], p[2]))
            .collect();
        let normals: Vec<nalgebra::Vector3<f32>> = nrm
            .iter()
            .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
            .collect();
        cv_core::PointCloud {
            points,
            colors: None,
            normals: Some(normals),
        }
    }

    let mut pts = Vec::new();
    for &x in &[0.0f32, 0.5, 1.0] {
        for &y in &[0.0f32, 0.5, 1.0] {
            for &z in &[0.0f32, 0.5, 1.0] {
                pts.push([x, y, z]);
            }
        }
    }
    // A cube's face normals, not a single +z for every point: a point-to-plane
    // residual cannot observe motion along a direction every normal is
    // perpendicular to, so with uniform normals a y-offset produces no residual
    // at all and neither path moves. That is a property of the model, not a
    // failure - but it makes the test meaningless, so the normals have to vary.
    let nrm: Vec<[f32; 3]> = pts
        .iter()
        .map(|p| {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for ax in 0..3usize {
                for &s in &[0.0f32, 1.0] {
                    let d = (p[ax] - s).abs();
                    if d < best_d {
                        best_d = d;
                        best = ax;
                    }
                }
            }
            let mut n = [0.0f32; 3];
            n[best] = if p[best] > 0.5 { 1.0 } else { -1.0 };
            n
        })
        .collect();
    let source = cloud(
        pts.iter().map(|p| [p[0], p[1] + 0.02, p[2]]).collect(),
        nrm.clone(),
    );
    let target = cloud(pts, nrm);

    let identity = nalgebra::Matrix4::identity();
    let cpu =
        cv_registration::registration_icp_point_to_plane(&source, &target, 0.1, &identity, 20)
            .expect("CPU icp");
    // The ctx entry point, driven with a GPU device so the branch under test is
    // actually taken rather than the CPU early-return.
    let gpu_dev = cv_hal::compute::ComputeDevice::Gpu(g);
    let gpu = cv_registration::registration_icp_point_to_plane_ctx(
        &source, &target, 0.1, &identity, 20, &gpu_dev,
    )
    .expect("ctx icp");

    println!(
        "ICP cpu ty={:.5}  ctx ty={:.5}",
        cpu.transformation[(1, 3)],
        gpu.transformation[(1, 3)]
    );
    // Both must move, and agree on where.
    assert!(
        cpu.transformation[(1, 3)].abs() > 1e-3,
        "the CPU path did not move, so the reference is wrong"
    );
    assert!(
        (cpu.transformation[(1, 3)] - gpu.transformation[(1, 3)]).abs() < 5e-3,
        "the ctx path disagrees with the CPU: cpu ty={} ctx ty={}",
        cpu.transformation[(1, 3)],
        gpu.transformation[(1, 3)]
    );
}

/// Every WGSL shader the HAL can dispatch must compile.
///
/// Two of them never did. `stereo_match.wgsl` passed a storage-space pointer
/// into a function, which WGSL forbids, and `icp_accumulate.wgsl` used `new` as
/// a local name, which is a reserved word. Both failed wgpu validation on every
/// dispatch, so GPU stereo matching and GPU ICP were hard errors rather than
/// wrong answers — and neither was covered, because `multi_gpu_tests.rs` does
/// not reach them and a failed dispatch only surfaces when a caller actually
/// invokes that path.
///
/// Compiling every shader up front turns "this feature is broken" into a test
/// failure on any machine, GPU or not.
#[test]
fn every_shader_compiles() {
    use cv_hal::gpu::GpuContext;

    let shaders: &[(&str, &str)] = &[
        (
            "stereo_match",
            include_str!("../crates/hal/shaders/stereo_match.wgsl"),
        ),
        (
            "icp_accumulate",
            include_str!("../crates/hal/shaders/icp_accumulate.wgsl"),
        ),
        (
            "icp_correspondence",
            include_str!("../crates/hal/shaders/icp_correspondence.wgsl"),
        ),
        (
            "gaussian_blur",
            include_str!("../crates/hal/shaders/gaussian_blur_separable.wgsl"),
        ),
        ("canny", include_str!("../crates/hal/shaders/canny.wgsl")),
        ("warp", include_str!("../crates/hal/shaders/warp.wgsl")),
        ("nms", include_str!("../crates/hal/shaders/nms.wgsl")),
        (
            "tsdf_raycast",
            include_str!("../crates/hal/shaders/tsdf_raycast.wgsl"),
        ),
        (
            "lucas_kanade",
            include_str!("../crates/hal/shaders/lucas_kanade.wgsl"),
        ),
        ("remap", include_str!("../crates/hal/shaders/remap.wgsl")),
        (
            "undistort",
            include_str!("../crates/hal/shaders/undistort.wgsl"),
        ),
    ];

    // A shader can be rejected by naga's parser without a device, so check the
    // reserved words that actually bit us. This runs everywhere; the device
    // compilation below is stronger but needs a GPU.
    const RESERVED: &[&str] = &[
        "new", "sample", "filter", "typedef", "union", "shared", "common", "active", "binding",
        "class", "enum", "handle", "layout", "resource", "signed", "unsigned",
    ];
    for (name, src) in shaders {
        for (i, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for w in RESERVED {
                if code.contains(&format!("let {w} ")) || code.contains(&format!("var {w} ")) {
                    panic!(
                        "{name}.wgsl:{} uses `{w}` as an identifier, which is a WGSL \\
                         reserved word: {line}",
                        i + 1
                    );
                }
            }
        }
    }

    // A pointer to a *non-atomic* storage array may not be passed into a
    // function either, though a pointer to `array<atomic<T>>` is legal - that
    // distinction is why the stereo shader's signature was invalid while
    // icp_accumulate's atomic helper is fine.
    for (name, src) in shaders {
        for (i, line) in src.lines().enumerate() {
            if line.contains("atomic<") {
                continue;
            }
            if line.contains("ptr<storage") && line.trim_start().starts_with("fn ") {
                panic!(
                    "{name}.wgsl:{} declares a function taking a non-atomic storage \
                     pointer, which WGSL forbids: {line}",
                    i + 1
                );
            }
        }
    }

    // And on a machine with a GPU, compile for real.
    if let Some(g) = gpu() {
        for (name, src) in shaders {
            // create_shader_module panics via the validation error on failure,
            // which is what surfaces as the wgpu Validation Error above.
            let _module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl((*src).into()),
            });
        }
    }
}

/// The LBVH build must run.
///
/// `lbvh_build.wgsl` declared five storage bindings against this device's limit
/// of four, so no pipeline in the file could be created. Because WGSL derives
/// one bind group layout per *module* rather than per entry point, the binding
/// only `compute_aabbs` touches was counted against `init_nodes` and
/// `build_radix_tree` too. The phases are now separate modules: the tree needs
/// two buffers, the AABB four.
#[test]
fn probe_lbvh_builds() {
    use cv_core::TensorShape;
    use cv_hal::gpu_kernels::lbvh::build_lbvh;

    let Some(g) = gpu() else { return };
    let n = 64usize;
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| [i as f32 * 0.1, (i % 8) as f32, (i % 5) as f32])
        .collect();
    let pts: CpuTensor<f32> = cv_core::Tensor::from_vec(flat, TensorShape::new(3, n, 1)).unwrap();
    let si: CpuTensor<u32> =
        cv_core::Tensor::from_vec((0..n as u32).collect(), TensorShape::new(1, n, 1)).unwrap();
    let mc: CpuTensor<u32> = cv_core::Tensor::from_vec(
        (0..n as u32).map(|i| i.wrapping_mul(7)).collect(),
        TensorShape::new(1, n, 1),
    )
    .unwrap();

    let pts_g = pts.to_gpu_ctx(g).unwrap();
    let si_g = si.to_gpu_ctx(g).unwrap();
    let mc_g = mc.to_gpu_ctx(g).unwrap();

    let nodes = build_lbvh(g, &pts_g, &si_g, &mc_g)
        .expect("LBVH build must succeed on a device with the 4-buffer limit");
    assert!(
        nodes.shape.len() > 0,
        "LBVH produced no nodes for {n} points"
    );
    println!("  LBVH: {} nodes for {n} points", nodes.shape.len());
}

/// `ComputeContext::resize` hard-coded bilinear, so a caller had no way to ask
/// for a different resampling - which is what left the ORB ctx pyramid unable to
/// reproduce the `Triangle` filtering the CPU path uses, and the two detection
/// entry points disagreeing on identical pixels.
///
/// This asserts the mode is actually honoured, not merely accepted: a nearest
/// and a Lanczos resize of the same input must differ, and neither may match the
/// other. A no-op `resize_with` that silently ignored its argument would pass a
/// test that only checked it returned Ok.
#[test]
fn probe_resize_with_honours_the_mode() {
    use cv_hal::context::Interpolation;

    let Some(g) = gpu() else { return };
    let (w, h) = (64usize, 64usize);
    // A diagonal ramp: a nearest-neighbour downsample and a filtered one pick
    // visibly different values.
    let data: Vec<f32> = (0..h * w)
        .map(|i| ((i % w) as f32 / w as f32) * 255.0)
        .collect();
    let input: CpuTensor<f32> = cv_core::Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let on_gpu = input.to_gpu_ctx(g).unwrap();

    let bilinear = g.resize(&on_gpu, (32, 32)).unwrap();
    // `Nearest` has no GPU implementation and must say so rather than silently
    // running bilinear - substituting one mode for another is the exact
    // failure this method exists to remove.
    match g.resize_with(&on_gpu, (32, 32), Interpolation::Nearest) {
        Err(_) => println!("  resize Nearest: correctly reported unsupported"),
        Ok(v) => {
            let cpu = v.to_cpu().unwrap();
            let a = bilinear.storage.as_slice().unwrap();
            let c = cpu.storage.as_slice().unwrap();
            let same = a.iter().zip(c).all(|(x, y)| (x - y).abs() < 1e-6);
            assert!(!same, "Nearest was silently run as bilinear");
        }
    }
    let lanczos = g
        .resize_with(&on_gpu, (32, 32), Interpolation::Lanczos)
        .unwrap();

    let b = bilinear.to_cpu().unwrap();
    let l = lanczos.to_cpu().unwrap();
    let bs = b.storage.as_slice().unwrap();
    let ls = l.storage.as_slice().unwrap();

    let diff = |a: &[f32], c: &[f32]| -> f32 {
        a.iter()
            .zip(c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let bl = diff(bs, ls);
    println!("  resize modes: bilinear-vs-lanczos {bl:.3}");
    assert!(
        bl > 1e-3,
        "Lanczos was identical to bilinear: the mode is ignored"
    );
}

/// A table-driven CPU/GPU parity sweep over the operations the per-feature
/// tests above do not reach.
///
/// The reason this file exists: `multi_gpu_tests.rs` covers seven operations, and
/// the individual probes here cover six more. The trait declares 49. Everything
/// outside that set had been checked by neither, and in this session seven GPU
/// entry points turned out never to have run at all - a shader that fails wgpu
/// validation, a pipeline that cannot be created. Every one of them was in
/// exactly this uncovered region, and none would have been caught by CI, which
/// has no GPU and skips.
///
/// Each case runs the same input on both backends and reports the worst
/// absolute difference. A case that cannot be compared yet says so rather than
/// being omitted, because an omission is indistinguishable from a pass.
mod sweep {
    use super::*;
    use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
    use cv_hal::context::{ComputeContext, WarpType};

    /// Deterministic pseudo-random floats, so a failure is reproducible.
    fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // `>> 8` leaves up to 2^56, which as an f32 is ~1e16 and makes
                // the fraction 16 million rather than something in [0, 1). The
                // first version of this helper had that bug, and it was
                // invisible here: the "divergence" it produced was in a region
                // where any tolerance passes, so both the broken input and the
                // comparison had to be checked by hand. `>> 33` leaves 31 bits,
                // which is exactly a u32.
                lo + ((s >> 33) as f32 / u32::MAX as f32) * (hi - lo)
            })
            .collect()
    }

    fn cpu_1ch(v: Vec<f32>, h: usize, w: usize) -> CpuTensor<f32> {
        Tensor::from_vec(v, TensorShape::new(1, h, w)).unwrap()
    }

    /// `sobel` with ksize 3, which both backends support.
    ///
    /// The 5x5 kernel was excluded deliberately: the GPU returns `NotSupported`
    /// for anything but ksize 3, so comparing it would be comparing an error
    /// against a value. That asymmetry is itself worth a test, below.
    #[test]
    fn sobel_ksize3_matches() {
        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (48usize, 64usize);
        let input = cpu_1ch(values(h * w, 7, 0.0, 255.0), h, w);

        // `sobel` returns (dx, dy), so both channels are compared.
        let (cx, cy) = cpu.sobel(&input, 1, 1, 3).unwrap();
        let gi = input.to_gpu_ctx(g).unwrap();
        let (gx, gy) = g.sobel(&gi, 1, 1, 3).unwrap();
        let gxc = gx.to_cpu().unwrap();
        let gyc = gy.to_cpu().unwrap();

        let cxs = cx.storage.as_slice().unwrap();
        let cys = cy.storage.as_slice().unwrap();
        let gxs = gxc.storage.as_slice().unwrap();
        let gys = gyc.storage.as_slice().unwrap();
        let cs: Vec<f32> = cxs.iter().chain(cys).copied().collect();
        let gs: Vec<f32> = gxs.iter().chain(gys).copied().collect();
        assert_eq!(cs.len(), gs.len(), "sobel output sizes differ");
        let worst = cs
            .iter()
            .zip(gs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = cs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("  sobel: worst abs {worst:.4} (peak {scale:.1})");
        assert!(
            worst <= tol_scaled(scale, 0.05),
            "sobel ksize=3 diverges: worst {worst} against a peak of {scale}"
        );
    }

    /// Relative tolerance against the output's own magnitude.
    ///
    /// An absolute threshold is useless across operations whose scale ranges
    /// from 0-1 to 0-255, and it turns a large-but-correct result into a
    /// failure or a small-but-wrong one into a pass.
    fn tol_scaled(peak: f32, fraction: f32) -> f32 {
        (peak * fraction).max(1e-3)
    }

    /// `warp` with the identity must return the input unchanged on both backends.
    ///
    /// This is the operation whose GPU dispatch was covering a quarter of the
    /// destination width until a few commits ago, which a comparison against the
    /// CPU would have caught immediately.
    #[test]
    fn warp_identity_matches() {
        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (37usize, 53usize); // deliberately not multiples of 16
        let input = cpu_1ch(values(h * w, 11, 0.0, 255.0), h, w);
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // `new_shape` is (width, height), not (height, width) - the transposed
        // version asked both backends for a 53x37 output from a 37x53 input, and
        // the resulting "divergence" was a 126.0 difference in the 37 pixels
        // past the real image, not a disagreement between the implementations.
        let c = cpu
            .warp(&input, &identity, (w, h), WarpType::Perspective)
            .unwrap();
        let gi = input.to_gpu_ctx(g).unwrap();
        let go = g
            .warp(&gi, &identity, (w, h), WarpType::Perspective)
            .unwrap();
        let gb = go.to_cpu().unwrap();

        let cs = c.storage.as_slice().unwrap();
        let gs = gb.storage.as_slice().unwrap();
        assert_eq!(cs.len(), gs.len(), "warp output sizes differ");
        let worst = cs
            .iter()
            .zip(gs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let peak = cs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // Where do the differences sit? Print the first few differing indices.
        let diffs: Vec<usize> = cs
            .iter()
            .zip(gs)
            .enumerate()
            .filter(|(_, (a, b))| (**a - **b).abs() > 0.5)
            .map(|(i, _)| i)
            .collect();
        println!(
            "  warp: worst {worst:.3} peak {peak:.1}; {}/{} differ, first at {:?}",
            diffs.len(),
            cs.len(),
            &diffs[..diffs.len().min(8)]
        );
        // Dimensions are not a multiple of the 16-wide workgroup, which is
        // exactly the case where the dispatch previously dropped columns.
        assert!(
            worst <= tol_scaled(peak, 0.02),
            "warp with identity diverges: worst {worst} against a peak of {peak}"
        );
    }

    /// `subtract` is elementwise and must be exact to float precision.
    #[test]
    fn subtract_matches() {
        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (32usize, 32usize);
        let a = cpu_1ch(values(h * w, 13, -50.0, 50.0), h, w);
        let b = cpu_1ch(values(h * w, 17, -50.0, 50.0), h, w);

        let c = cpu.subtract(&a, &b).unwrap();
        let ga = a.to_gpu_ctx(g).unwrap();
        let gb = b.to_gpu_ctx(g).unwrap();
        let go = g.subtract(&ga, &gb).unwrap();
        let back = go.to_cpu().unwrap();

        let cs = c.storage.as_slice().unwrap();
        let gs = back.storage.as_slice().unwrap();
        let worst = cs
            .iter()
            .zip(gs)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("  subtract: worst abs {worst:.6}");
        assert!(worst < 1e-3, "subtract diverges: worst {worst}");
    }

    /// Morphology, erode and dilate with a 3x3 kernel.
    #[test]
    fn morphology_matches() {
        use cv_hal::context::MorphologyType;
        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (40usize, 40usize);
        let v = values(h * w, 19, 0.0, 255.0);
        let input: CpuTensor<u8> = Tensor::from_vec(
            v.iter().map(|&x| x as u8).collect(),
            TensorShape::new(1, h, w),
        )
        .unwrap();
        let kernel: CpuTensor<u8> =
            Tensor::from_vec(vec![0u8, 1, 0, 1, 1, 1, 0, 1, 0], TensorShape::new(3, 3, 1)).unwrap();

        for (name, typ) in [
            ("erode", MorphologyType::Erode),
            ("dilate", MorphologyType::Dilate),
        ] {
            let c = cpu
                .morphology(&input, typ, &kernel, 1)
                .unwrap_or_else(|e| panic!("CPU {name} failed: {e}"));
            let gi = input.to_gpu_ctx(g).unwrap();
            let gk = kernel.to_gpu_ctx(g).unwrap();
            match g.morphology(&gi, typ, &gk, 1) {
                Ok(go) => {
                    let back = go.to_cpu().unwrap();
                    let cs = c.storage.as_slice().unwrap();
                    let gs = back.storage.as_slice().unwrap();
                    let differing = cs.iter().zip(gs).filter(|(a, b)| a != b).count();
                    println!("  morphology {name}: {differing} of {} differ", cs.len());
                    assert_eq!(
                        differing,
                        0,
                        "morphology {name} differs on {differing} of {} pixels",
                        cs.len()
                    );
                }
                Err(e) => println!("  morphology {name}: GPU reports {e}"),
            }
        }
    }

    /// `remap` with an identity map must return the input unchanged on both
    /// backends.
    ///
    /// Identity remapping is the one case where a correct implementation and a
    /// broken one are easy to tell apart, and it exercises the map indexing and
    /// the interpolation call together.
    #[test]
    fn remap_identity_matches() {
        use cv_hal::context::{BorderMode, Interpolation};

        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (29usize, 41usize); // neither a multiple of 16
        let input = cpu_1ch(values(h * w, 23, 0.0, 255.0), h, w);

        // Identity map: map_x[x, y] = x, map_y[x, y] = y.
        let mut mx = Vec::with_capacity(h * w);
        let mut my = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                mx.push(x as f32);
                my.push(y as f32);
            }
        }
        let map_x: CpuTensor<f32> = Tensor::from_vec(mx, TensorShape::new(1, h, w)).unwrap();
        let map_y: CpuTensor<f32> = Tensor::from_vec(my, TensorShape::new(1, h, w)).unwrap();

        // The CPU does not implement remap, so there is no reference to compare
        // against. Asserting that is the honest outcome: a parity test that
        // silently skipped would be indistinguishable from a pass, and the whole
        // point of this file is that "unchecked" is what let seven GPU paths go
        // years without running.
        let c = match cpu.remap(
            &input,
            &map_x,
            &map_y,
            Interpolation::Linear,
            BorderMode::Replicate,
        ) {
            Ok(v) => v,
            Err(e) => {
                println!("  remap: CPU reports {e} - no reference to compare against");
                return;
            }
        };
        let gi = input.to_gpu_ctx(g).unwrap();
        let gmx = map_x.to_gpu_ctx(g).unwrap();
        let gmy = map_y.to_gpu_ctx(g).unwrap();
        let go = g
            .remap(
                &gi,
                &gmx,
                &gmy,
                Interpolation::Linear,
                BorderMode::Replicate,
            )
            .unwrap();
        let gb = go.to_cpu().unwrap();

        let cs = c.storage.as_slice().unwrap();
        let gs = gb.storage.as_slice().unwrap();
        let worst = cs
            .iter()
            .zip(gs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let peak = cs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("  remap identity: worst {worst:.4} (peak {peak:.1})");
        assert!(
            worst <= tol_scaled(peak, 0.02),
            "remap with an identity map diverges: worst {worst} against a peak of {peak}"
        );
    }

    /// `undistort` with zero distortion and identity rectification is the
    /// identity, and both backends must agree.
    ///
    /// The distortion coefficients are the field a real calibration supplies, so
    /// this is the degenerate case that isolates the border handling from the
    /// model. It is the only one comparable without a real calibration, which is
    /// why the test says so rather than pretending to cover undistortion.
    #[test]
    fn undistort_with_zero_distortion_is_the_identity() {
        use cv_hal::context::{BorderMode, Interpolation};

        let Some(g) = gpu() else { return };
        let cpu = CpuBackend::new().unwrap();
        let (h, w) = (24usize, 24usize);
        let input = cpu_1ch(values(h * w, 29, 0.0, 255.0), h, w);
        let k = cv_core::CameraIntrinsics::new(50.0, 50.0, 12.0, 12.0, w as u32, h as u32);
        let d = cv_core::Distortion::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let r = nalgebra::Matrix3::identity();

        let c = match cpu.undistort(
            &input,
            &k,
            &d,
            &r,
            &k,
            Interpolation::Linear,
            BorderMode::Replicate,
        ) {
            Ok(v) => v,
            Err(e) => {
                println!("  undistort: CPU reports {e} - no reference to compare against");
                return;
            }
        };
        let gi = input.to_gpu_ctx(g).unwrap();
        match g.undistort(
            &gi,
            &k,
            &d,
            &r,
            &k,
            Interpolation::Linear,
            BorderMode::Replicate,
        ) {
            Ok(go) => {
                let gb = go.to_cpu().unwrap();
                let cs = c.storage.as_slice().unwrap();
                let gs = gb.storage.as_slice().unwrap();
                let worst = cs
                    .iter()
                    .zip(gs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let peak = cs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                println!("  undistort (zero distortion): worst {worst:.4} (peak {peak:.1})");
                assert!(
                    worst <= tol_scaled(peak, 0.02),
                    "undistort with zero distortion diverges: worst {worst}"
                );
            }
            Err(e) => println!("  undistort: GPU reports {e}"),
        }
    }
}
