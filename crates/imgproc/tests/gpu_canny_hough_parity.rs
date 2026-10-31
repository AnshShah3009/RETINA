//! GPU `canny` / `hough_lines` must return exactly what the CPU paths return
//! (which match OpenCV 4.13 - see canny_opencv_parity.rs and
//! hough_opencv_parity.rs).
//!
//! The `*_ctx` entry points try the GPU first on a GPU runner/group and fall
//! back to the CPU on error, so a GPU-runner result that equals the CPU result
//! could also mean the GPU path silently failed. Each test therefore also
//! calls the GPU kernel directly. Tests skip when no adapter is available.

use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_imgproc::hough::{hough_lines, hough_lines_ctx};
use cv_imgproc::{canny_with_options_ctx, CannyOptions};
use cv_runtime::orchestrator::{ResourceGroup, RuntimeRunner};
use image::GrayImage;
use std::sync::{Arc, OnceLock};

fn gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match pollster::block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU canny/hough parity test (no adapter): {e}");
            None
        }
    }
}

fn gpu_runner(gpu: &GpuContext) -> RuntimeRunner {
    RuntimeRunner::Sync(gpu.device_id())
}

fn cpu_runner() -> RuntimeRunner {
    RuntimeRunner::Sync(cv_runtime::registry().unwrap().default_cpu().id())
}

fn gpu_group(gpu: &GpuContext) -> Arc<ResourceGroup> {
    static GROUP: OnceLock<Arc<ResourceGroup>> = OnceLock::new();
    GROUP
        .get_or_init(|| {
            cv_runtime::scheduler()
                .unwrap()
                .create_group_with_device(
                    "gpu_canny_hough_parity",
                    1,
                    None,
                    cv_runtime::GroupPolicy::default(),
                    gpu.device_id(),
                )
                .unwrap()
        })
        .clone()
}

// ---- fixtures (same images as the OpenCV parity tests) ----------------------

fn rand_img(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut s = seed;
    GrayImage::from_fn(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        image::Luma([(s >> 24) as u8])
    })
}

fn vstep() -> GrayImage {
    GrayImage::from_fn(32, 24, |x, _| image::Luma([if x < 16 { 50 } else { 200 }]))
}

fn rect() -> GrayImage {
    GrayImage::from_fn(40, 30, |x, y| {
        image::Luma([if (8..30).contains(&x) && (6..22).contains(&y) {
            180
        } else {
            30
        }])
    })
}

fn disc() -> GrayImage {
    GrayImage::from_fn(40, 40, |x, y| {
        let (dx, dy) = (x as i32 - 20, y as i32 - 20);
        image::Luma([if dx * dx + dy * dy <= 121 { 220 } else { 40 }])
    })
}

fn diag() -> GrayImage {
    GrayImage::from_fn(32, 32, |x, y| {
        image::Luma([if x + y < 30 { 60 } else { 190 }])
    })
}

/// Larger smooth-plus-noise scene: blobs, ramps and texture, so every NMS
/// branch, long hysteresis chains and many ties occur.
fn scene(w: u32, h: u32) -> GrayImage {
    let n = rand_img(w, h, 77);
    GrayImage::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 0.25 * xf + 0.1 * yf;
        if ((xf - 150.0).powi(2) + (yf - 120.0).powi(2)) < 70.0f32.powi(2) {
            v += 90.0;
        }
        if (x / 40 + y / 40) % 2 == 0 {
            v += 35.0;
        }
        v += 30.0 * ((xf * 0.07).sin() * (yf * 0.05).cos());
        v += (n.get_pixel(x, y)[0] as f32 - 128.0) * 0.15;
        image::Luma([v.clamp(0.0, 255.0) as u8])
    })
}

fn hline() -> GrayImage {
    GrayImage::from_fn(100, 100, |_, y| {
        image::Luma([if y == 50 { 255 } else { 0 }])
    })
}

fn multi() -> GrayImage {
    let n = rand_img(80, 60, 9);
    GrayImage::from_fn(80, 60, |x, y| {
        let on = x == y || x == 20 || y == (x + 20) / 2 || n.get_pixel(x, y)[0] > 250;
        image::Luma([if on { 255 } else { 0 }])
    })
}

fn dashed() -> GrayImage {
    GrayImage::from_fn(120, 90, |x, y| {
        let on = (y == 30 && x % 12 < 8) || x + y == 100;
        image::Luma([if on { 255 } else { 0 }])
    })
}

fn theta(div: f64) -> f32 {
    (std::f64::consts::PI / div) as f32
}

// ---- canny -----------------------------------------------------------------

fn assert_same_mask(name: &str, gpu: &GrayImage, cpu: &GrayImage) {
    assert_eq!(gpu.dimensions(), cpu.dimensions(), "{name}: size");
    let w = gpu.width() as usize;
    let diffs: Vec<(usize, usize, u8, u8)> = gpu
        .as_raw()
        .iter()
        .zip(cpu.as_raw())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i % w, i / w, *a, *b))
        .collect();
    let g = gpu.as_raw().iter().filter(|&&v| v != 0).count();
    let c = cpu.as_raw().iter().filter(|&&v| v != 0).count();
    assert!(
        diffs.is_empty(),
        "{name}: {} pixels differ (gpu {g} edge px, cpu {c}); first (x, y, gpu, cpu): {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(8)]
    );
}

fn check_canny(gpu: &GpuContext, name: &str, img: &GrayImage, lo: f32, hi: f32, o: CannyOptions) {
    let cpu = canny_with_options_ctx(img, lo, hi, &o, &cpu_runner());
    let via_ctx = canny_with_options_ctx(img, lo, hi, &o, &gpu_runner(gpu));
    assert_same_mask(&format!("{name} (ctx)"), &via_ctx, &cpu);
    if o.blur_sigma.is_none() {
        // The kernel itself, so a silent CPU fallback cannot pass the test.
        let direct = cv_hal::gpu_kernels::canny::canny_u8(
            gpu,
            img.as_raw(),
            img.width() as usize,
            img.height() as usize,
            lo as f64,
            hi as f64,
            o.l2_gradient,
        )
        .expect("GPU canny kernel");
        let direct = GrayImage::from_raw(img.width(), img.height(), direct).unwrap();
        assert_same_mask(&format!("{name} (kernel)"), &direct, &cpu);
    }
}

const L1: CannyOptions = CannyOptions {
    l2_gradient: false,
    blur_sigma: None,
};
const L2: CannyOptions = CannyOptions {
    l2_gradient: true,
    blur_sigma: None,
};

#[test]
fn gpu_canny_matches_cpu_on_the_opencv_fixtures() {
    let Some(gpu) = gpu() else { return };
    check_canny(gpu, "vstep", &vstep(), 50.0, 150.0, L1);
    check_canny(gpu, "rect", &rect(), 50.0, 150.0, L1);
    check_canny(gpu, "disc", &disc(), 50.0, 150.0, L1);
    check_canny(gpu, "diag", &diag(), 50.0, 150.0, L1);
    check_canny(gpu, "noise", &rand_img(40, 30, 5), 300.0, 600.0, L1);
    check_canny(gpu, "rect swapped", &rect(), 150.0, 50.0, L1);
}

#[test]
fn gpu_canny_l2_matches_cpu_on_the_opencv_fixtures() {
    let Some(gpu) = gpu() else { return };
    check_canny(gpu, "disc L2", &disc(), 50.0, 150.0, L2);
    check_canny(gpu, "noise L2", &rand_img(40, 30, 5), 200.0, 400.0, L2);
}

#[test]
fn gpu_canny_matches_cpu_on_a_large_scene() {
    let Some(gpu) = gpu() else { return };
    let img = scene(641, 479);
    for &(lo, hi) in &[(10.0, 30.0), (40.0, 120.0), (100.0, 250.0), (0.0, 0.0)] {
        check_canny(gpu, &format!("scene L1 {lo}/{hi}"), &img, lo, hi, L1);
        check_canny(gpu, &format!("scene L2 {lo}/{hi}"), &img, lo, hi, L2);
    }
    check_canny(
        gpu,
        "noise 300x200",
        &rand_img(300, 200, 3),
        200.0,
        500.0,
        L1,
    );
}

#[test]
fn gpu_canny_with_pre_blur_matches_cpu() {
    let Some(gpu) = gpu() else { return };
    let blur = CannyOptions {
        l2_gradient: false,
        blur_sigma: Some(1.0),
    };
    check_canny(gpu, "vstep blur", &vstep(), 10.0, 30.0, blur);
    check_canny(gpu, "scene blur", &scene(200, 150), 20.0, 60.0, blur);
}

#[test]
fn gpu_canny_tiny_images_match_cpu() {
    let Some(gpu) = gpu() else { return };
    for (w, h) in [(1, 1), (1, 7), (5, 1), (2, 2), (3, 3)] {
        let img = rand_img(w, h, w * 31 + h);
        check_canny(gpu, &format!("{w}x{h}"), &img, 20.0, 60.0, L1);
    }
}

// ---- hough -----------------------------------------------------------------

type L = (f32, f32, u32);

fn lines(v: Vec<cv_imgproc::hough::Line>) -> Vec<L> {
    v.iter().map(|l| (l.rho, l.theta, l.score)).collect()
}

fn check_hough(gpu: &GpuContext, name: &str, img: &GrayImage, rho: f32, th: f32, thr: u32) {
    let cpu = lines(hough_lines_ctx(img, rho, th, thr, &cpu_group()).unwrap());
    let via_ctx = lines(hough_lines_ctx(img, rho, th, thr, &gpu_group(gpu)).unwrap());
    assert_eq!(via_ctx, cpu, "{name}: GPU hough_lines_ctx differs from CPU");
    let direct = lines(
        cv_imgproc::hough::hough_lines_gpu(gpu, img, rho, th, thr).expect("GPU hough kernel"),
    );
    assert_eq!(direct, cpu, "{name}: GPU hough kernel differs from CPU");
}

fn cpu_group() -> Arc<ResourceGroup> {
    static GROUP: OnceLock<Arc<ResourceGroup>> = OnceLock::new();
    GROUP
        .get_or_init(|| {
            let cpu = cv_runtime::registry().unwrap().default_cpu().id();
            cv_runtime::scheduler()
                .unwrap()
                .create_group_with_device(
                    "cpu_canny_hough_parity",
                    2,
                    None,
                    cv_runtime::GroupPolicy::default(),
                    cpu,
                )
                .unwrap()
        })
        .clone()
}

#[test]
fn gpu_hough_lines_match_cpu_on_the_opencv_fixtures() {
    let Some(gpu) = gpu() else { return };
    check_hough(gpu, "hline", &hline(), 1.0, theta(180.0), 50);
    check_hough(gpu, "multi", &multi(), 1.0, theta(180.0), 25);
    check_hough(gpu, "multi coarse", &multi(), 2.0, theta(90.0), 25);
    check_hough(gpu, "dashed", &dashed(), 1.0, theta(180.0), 40);
}

#[test]
fn gpu_hough_lines_match_cpu_on_canny_edges_of_a_scene() {
    let Some(gpu) = gpu() else { return };
    let img = scene(641, 479);
    let edges = canny_with_options_ctx(&img, 40.0, 120.0, &L1, &cpu_runner());
    for &(rho, div, thr) in &[
        (1.0f32, 180.0, 80u32),
        (1.0, 360.0, 60),
        (0.5, 720.0, 40),
        (3.0, 45.0, 100),
        (1.5, 101.0, 70),
    ] {
        check_hough(
            gpu,
            &format!("scene rho {rho} pi/{div} thr {thr}"),
            &edges,
            rho,
            theta(div),
            thr,
        );
    }
    // Default entry point too (whatever the default group is).
    let _ = hough_lines(&edges, 1.0, theta(180.0), 80).unwrap();
}
