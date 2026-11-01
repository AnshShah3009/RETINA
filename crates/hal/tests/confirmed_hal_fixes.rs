//! Confirmed HAL defects. Each test is the behaviour the matching CPU
//! implementation (imgproc / video / features) already has, or the layout the
//! tensor type documents.

use cv_core::TensorShape;
use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_hal::context::{ColorConversion, ComputeContext, MorphologyType};
use cv_hal::cpu::CpuBackend;
use cv_hal::cpu::hough_standard::HoughLayout;
use cv_hal::cpu::simd::rgb_to_gray_simd;

fn cpu() -> CpuBackend {
    CpuBackend::new().unwrap()
}

fn f32_tensor(data: Vec<f32>, c: usize, h: usize, w: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data, TensorShape::new(c, h, w)).unwrap()
}

fn u8_tensor(data: Vec<u8>, c: usize, h: usize, w: usize) -> Tensor<u8, CpuStorage<u8>> {
    Tensor::from_vec(data, TensorShape::new(c, h, w)).unwrap()
}

/// CHW `(C=2, H=1, W=2)` bytes `[0, 100, 0, 0]`: `index(1,0,0) = 2` is 0.
/// An interleaved read treats channel 1 of pixel 0 as 100.
#[test]
fn gaussian_blur_reads_planar_chw() {
    let cpu = cpu();
    let img = f32_tensor(vec![0.0, 100.0, 0.0, 0.0], 2, 1, 2);
    let out = cpu.gaussian_blur(&img, 1.0, 3).unwrap();
    let d = out.as_slice().unwrap();
    assert_eq!(d.len(), 4);
    // Channel 1 is the constant 0 plane. A blur must leave it at 0.
    assert!(
        d[2].abs() < 1e-5 && d[3].abs() < 1e-5,
        "channel 1 of pixel 0 must be read as 0, got plane {:?}",
        &d[2..4]
    );
    // Channel 0 is [0, 100]. The blur has to see that 100, which an
    // interleaved reader never puts in plane 0.
    assert!(
        d[0] > 1.0 && d[1] > 1.0,
        "channel 0 must be the blur of [0, 100], got {:?}",
        &d[0..2]
    );
}

/// OpenCV `pyrDown`: `((w+1)/2, (h+1)/2)`, not floor division.
#[test]
fn pyramid_down_3x3_is_2x2() {
    let cpu = cpu();
    let img = f32_tensor(vec![10.0; 9], 1, 3, 3);
    let out = cpu.pyramid_down(&img).unwrap();
    assert_eq!(out.shape.width, 2, "3 wide must come back 2, not 1");
    assert_eq!(out.shape.height, 2, "3 tall must come back 2, not 1");
    for (i, v) in out.as_slice().unwrap().iter().enumerate() {
        assert!(
            (v - 10.0).abs() < 1e-4,
            "constant 10 must survive pyrDown, pixel {i} is {v}"
        );
    }
}

/// Column `x=4` on an 8×8 edge map, `rho=1`, `threshold=8`: eight votes is
/// not *more than* the threshold, so OpenCV returns no line.
#[test]
fn hough_lines_requires_votes_strictly_above_threshold() {
    let cpu = cpu();
    let mut data = vec![0.0f32; 64];
    for y in 0..8 {
        data[y * 8 + 4] = 255.0;
    }
    let img = f32_tensor(data, 1, 8, 8);
    let lines = cpu
        .hough_lines(&img, 1.0, std::f32::consts::PI / 180.0, 8)
        .unwrap();
    assert!(
        lines.is_empty(),
        "8 votes at threshold 8 must be dropped, got {lines:?}"
    );
}

/// `num_rho` uses OpenCV `cvRound` (half to even), not half away from zero.
/// `(1+1)*2+1 = 5`, `5/2 = 2.5` → 2.
#[test]
fn hough_rho_bin_count_rounds_half_to_even() {
    let layout = HoughLayout::new(1, 1, 2.0, 0.5);
    assert_eq!(layout.num_rho, 2, "2.5 rho bins must round half to even");
}

/// Query `[0]`, train `[[0], [0]]`, ratio 0.8: the second-best distance is 0,
/// which is ambiguous and must not be kept.
#[test]
fn match_descriptors_rejects_zero_second_best() {
    let cpu = cpu();
    let q = u8_tensor(vec![0], 1, 1, 1);
    let t = u8_tensor(vec![0, 0], 1, 2, 1);
    let m = cpu.match_descriptors(&q, &t, 0.8).unwrap();
    assert!(
        m.matches.is_empty(),
        "identical train descriptors are ambiguous, got {:?}",
        m.matches
    );
}

/// A point at `(0, 0)` with window 21 does not fit. It must not come back as
/// a successful stationary track.
#[test]
fn lk_rejects_point_whose_window_does_not_fit() {
    let cpu = cpu();
    let img = f32_tensor((0..64 * 64).map(|i| (i % 17) as f32).collect(), 1, 64, 64);
    let out = cpu
        .optical_flow_lk(&[img.clone()], &[img], &[[0.0, 0.0]], 21, 5)
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(
        !out[0][0].is_finite() || !out[0][1].is_finite(),
        "window that does not fit must not be reported as a track, got {:?}",
        out[0]
    );
}

/// Left-neighbour-only dilate, offset `(-1, 0)`, row `[1, 2, 3]`.
/// Replicate border returns `[1, 1, 2]`, not the dilate identity `[0, 1, 2]`.
#[test]
fn morphology_replicate_samples_the_edge_pixel() {
    let cpu = cpu();
    // Anchor is `kw/2`. A 3×1 mask with only the left tap set is offset -1.
    let img = u8_tensor(vec![1, 2, 3], 1, 1, 3);
    let kernel = u8_tensor(vec![1, 0, 0], 1, 1, 3);
    let out = cpu
        .morphology(&img, MorphologyType::Dilate, &kernel, 1)
        .unwrap();
    assert_eq!(out.as_slice().unwrap(), &[1, 1, 2]);
}

/// Bytes `[0, 100, 0, 0, 0, 0]` as CHW are R=`[0, 100]`, G=0, B=0.
/// Pixel 0 is black. An interleaved read treats pixel 0 as green 100.
#[test]
fn cvt_color_rgb_to_gray_is_planar_chw() {
    let cpu = cpu();
    let img = f32_tensor(vec![0.0, 100.0, 0.0, 0.0, 0.0, 0.0], 3, 1, 2);
    let out = cpu.cvt_color(&img, ColorConversion::RgbToGray).unwrap();
    let d = out.as_slice().unwrap();
    assert!(d[0].abs() < 1e-4, "pixel 0 gray is {}, want 0", d[0]);
    assert!(
        (d[1] - 29.9).abs() < 0.05,
        "pixel 1 gray is {}, want 0.299*100",
        d[1]
    );
}

/// Integer intensities must match `exp(space + range)` bit for bit. The
/// spatial term and the 256-entry range term are hoisted; the exp is not
/// factored, because `exp(a)*exp(b)` is a different f32.
#[test]
fn bilateral_integer_image_matches_the_direct_exp() {
    let cpu = cpu();
    let src: Vec<f32> = (0..16).map(|i| (i * 17 % 256) as f32).collect();
    let out = cpu
        .bilateral_filter(&f32_tensor(src.clone(), 1, 4, 4), 3, 10.0, 10.0)
        .unwrap();
    let got = out.as_slice().unwrap();
    let (h, w) = (4usize, 4usize);
    let radius = 1i32;
    let color_coeff = -0.5f32 / (10.0 * 10.0);
    let space_coeff = color_coeff;
    for y in 0..h {
        for x in 0..w {
            let center = src[y * w + x];
            let mut sum = 0.0f32;
            let mut norm = 0.0f32;
            for j in -radius..=radius {
                for i in -radius..=radius {
                    let sy = (y as i32 + j).clamp(0, h as i32 - 1) as usize;
                    let sx = (x as i32 + i).clamp(0, w as i32 - 1) as usize;
                    let val = src[sy * w + sx];
                    let dist_sq = (i * i + j * j) as f32;
                    let range_sq = (val - center) * (val - center);
                    let weight = (dist_sq * space_coeff + range_sq * color_coeff).exp();
                    sum += val * weight;
                    norm += weight;
                }
            }
            let want = sum / norm;
            let g = got[y * w + x];
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "pixel {x},{y}: got {g} want {want}"
            );
        }
    }
}

/// Parse the shaders this change edits. Skips when no GPU adapter is present.
#[test]
fn changed_shaders_parse_when_a_device_is_present() {
    use cv_hal::gpu::GpuContext;
    let gpu = if let Ok(ctx) = GpuContext::global() {
        ctx
    } else {
        match pollster::block_on(GpuContext::init_global()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("no GPU adapter, skipping shader parse: {e}");
                return;
            }
        }
    };
    for (name, src) in [
        (
            "farneback_poly_expand.wgsl",
            include_str!("../shaders/farneback_poly_expand.wgsl"),
        ),
        (
            "farneback_update_flow.wgsl",
            include_str!("../shaders/farneback_update_flow.wgsl"),
        ),
        (
            "pyramid_down.wgsl",
            include_str!("../shaders/pyramid_down.wgsl"),
        ),
        (
            "gaussian_blur_separable.wgsl",
            include_str!("../shaders/gaussian_blur_separable.wgsl"),
        ),
        (
            "lucas_kanade.wgsl",
            include_str!("../shaders/lucas_kanade.wgsl"),
        ),
        (
            "tsdf_raycast.wgsl",
            include_str!("../shaders/tsdf_raycast.wgsl"),
        ),
        (
            "color_cvt_f32.wgsl",
            include_str!("../shaders/color_cvt_f32.wgsl"),
        ),
    ] {
        gpu.create_compute_pipeline(src, "main");
        println!("{name} parsed");
    }
}

/// `(0, 1, 0)` is `0.587` and must round to 1, not truncate to 0.
#[test]
fn rgb_to_gray_rounds_green_unit_to_one() {
    let mut gray = [0u8; 1];
    rgb_to_gray_simd(&[0, 1, 0], &mut gray);
    assert_eq!(gray[0], 1);
    let mut rgb = vec![0u8; 24];
    rgb[1] = 1;
    let mut gray8 = vec![0u8; 8];
    rgb_to_gray_simd(&rgb, &mut gray8);
    assert_eq!(gray8[0], 1);
}
