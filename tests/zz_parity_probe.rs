// TEMPORARY parity probe - DELETE BEFORE FINISHING
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
    let cr = cpu_ctx.stereo_match(&l, &r, &p).unwrap();
    let cs = cr.storage.as_slice().unwrap().to_vec();
    let gl = l.to_gpu_ctx(g).unwrap();
    let gr2 = r.to_gpu_ctx(g).unwrap();
    let gr = g.stereo_match(&gl, &gr2, &p).unwrap();
    let gs = gr.to_cpu().unwrap().storage.as_slice().unwrap().to_vec();
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
