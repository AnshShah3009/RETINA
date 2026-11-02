//! `resize` treats a multichannel tensor as planar CHW, like every other op.
//!
//! `TensorShape::new(c, h, w)` stores element `(ch, y, x)` at
//! `ch * h * w + y * w + x`: `Tensor::index`, `Tensor::from_image_rgb`,
//! `gaussian_blur`, `cvt_color`, `warp`, `morphology`, `pyramid_down` and the
//! `photo` and `dnn` crates all read it that way. CPU `resize` and the GPU
//! multichannel shader used to index `(y * w + x) * c + ch` (interleaved HWC).
//! They agreed with each other, so the CPU/GPU parity probe passed, while a
//! resized RGB tensor came back with its planes scrambled for every other op:
//! a constant-plane image (R = 0, G = 100, B = 200) returned planes holding a
//! mix of all three values.
//!
//! The tests below would have caught that through the public API alone:
//! constant planes must stay constant, each plane must resize exactly as a
//! one-channel tensor holding that plane, and `resize` composed with
//! `cvt_color` and `gaussian_blur` must commute with doing it plane by plane.
//! The GPU half skips without an adapter.

use cv_core::storage::{CpuStorage, Storage};
use cv_core::{Tensor, TensorShape};
use cv_hal::context::{ColorConversion, ComputeContext, Interpolation};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

type Cpu = Tensor<f32, CpuStorage<f32>>;

fn cpu() -> CpuBackend {
    CpuBackend::new().unwrap()
}

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("skipping: no GPU adapter");
            None
        }
    }
}

fn tensor(data: Vec<f32>, c: usize, h: usize, w: usize) -> Cpu {
    Tensor::from_vec(data, TensorShape::new(c, h, w)).unwrap()
}

/// Planes of `c` constants, CHW.
fn constant_planes(values: &[f32], h: usize, w: usize) -> Cpu {
    let data = values
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, h * w))
        .collect();
    tensor(data, values.len(), h, w)
}

/// Three planes with different structure: a horizontal ramp, a vertical ramp
/// and a checkerboard, so a plane read from the wrong place is obvious.
fn structured_rgb(h: usize, w: usize) -> Cpu {
    let mut data = Vec::with_capacity(3 * h * w);
    for _ in 0..h {
        for x in 0..w {
            data.push(x as f32 / w as f32);
        }
    }
    for y in 0..h {
        for _ in 0..w {
            data.push(y as f32 / h as f32);
        }
    }
    for y in 0..h {
        for x in 0..w {
            data.push(if (x / 2 + y / 2) % 2 == 0 { 0.1 } else { 0.9 });
        }
    }
    tensor(data, 3, h, w)
}

fn plane(t: &Cpu, ch: usize) -> Cpu {
    let (h, w) = t.shape.hw();
    let d = t.as_slice().unwrap();
    tensor(d[ch * h * w..(ch + 1) * h * w].to_vec(), 1, h, w)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_constant_planes(out: &Cpu, values: &[f32], tol: f32, what: &str) {
    let (h, w) = out.shape.hw();
    assert_eq!(out.shape.channels, values.len(), "{what}: channels");
    let d = out.as_slice().unwrap();
    for (ch, &v) in values.iter().enumerate() {
        let p = &d[ch * h * w..(ch + 1) * h * w];
        let worst = p.iter().map(|x| (x - v).abs()).fold(0.0, f32::max);
        assert!(
            worst <= tol,
            "{what}: plane {ch} must stay {v}, worst deviation {worst} (plane starts {:?})",
            &p[..p.len().min(6)]
        );
    }
}

const SIZES: [((usize, usize), (usize, usize)); 4] = [
    // (src h, w) -> (dst w, h): shrink, enlarge, odd sizes, one row.
    ((16, 12), (5, 7)),
    ((5, 7), (13, 11)),
    ((9, 9), (9, 4)),
    ((1, 6), (3, 1)),
];

// ------------------------------------------------------------------ CPU

#[test]
fn cpu_resize_keeps_constant_planes_constant() {
    let cpu = cpu();
    for ((h, w), (nw, nh)) in SIZES {
        let input = constant_planes(&[0.0, 100.0, 200.0], h, w);
        let out = cpu.resize(&input, (nw, nh)).unwrap();
        assert_eq!(out.shape, TensorShape::new(3, nh, nw));
        assert_constant_planes(&out, &[0.0, 100.0, 200.0], 0.0, "CPU");
    }
}

#[test]
fn cpu_resize_of_each_plane_is_the_one_channel_resize() {
    let cpu = cpu();
    for ((h, w), (nw, nh)) in SIZES {
        let input = structured_rgb(h, w);
        let out = cpu.resize(&input, (nw, nh)).unwrap();
        for ch in 0..3 {
            let alone = cpu.resize(&plane(&input, ch), (nw, nh)).unwrap();
            assert_eq!(
                plane(&out, ch).as_slice().unwrap(),
                alone.as_slice().unwrap(),
                "CPU plane {ch}, {w}x{h} -> {nw}x{nh}"
            );
        }
    }
}

/// `resize` then `cvt_color` equals `cvt_color` then `resize`: both are linear
/// in the pixel values and resize weights every plane the same way. With the
/// interleaved indexing the left side mixed planes before conversion.
#[test]
fn cpu_resize_commutes_with_cvt_color() {
    let cpu = cpu();
    for ((h, w), (nw, nh)) in SIZES {
        let rgb = structured_rgb(h, w);
        let a = cpu
            .cvt_color(
                &cpu.resize(&rgb, (nw, nh)).unwrap(),
                ColorConversion::RgbToGray,
            )
            .unwrap();
        let b = cpu
            .resize(
                &cpu.cvt_color(&rgb, ColorConversion::RgbToGray).unwrap(),
                (nw, nh),
            )
            .unwrap();
        let d = max_diff(a.as_slice().unwrap(), b.as_slice().unwrap());
        assert!(d < 1e-5, "CPU {w}x{h} -> {nw}x{nh}: differ by {d}");
    }
}

/// An RGB image loaded through `from_image_rgb` (CHW) and resized must keep
/// each colour in its own plane: a red left half and a blue right half give
/// no green anywhere and a gray image that is 0.299 / 0.114 at the far edges.
#[test]
fn cpu_resized_rgb_image_converts_to_the_right_gray() {
    let cpu = cpu();
    let (w, h) = (8usize, 6usize);
    let mut bytes = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(if x < w / 2 {
                &[255, 0, 0]
            } else {
                &[0, 0, 255]
            });
        }
    }
    let rgb = Tensor::<f32>::from_image_rgb(&bytes, w, h).unwrap();
    let out = cpu.resize(&rgb, (4, 3)).unwrap();
    let green = plane(&out, 1);
    assert!(
        green.as_slice().unwrap().iter().all(|&v| v == 0.0),
        "green plane must stay 0, got {:?}",
        green.as_slice().unwrap()
    );
    let gray = cpu.cvt_color(&out, ColorConversion::RgbToGray).unwrap();
    let g = gray.as_slice().unwrap();
    for y in 0..3 {
        assert!((g[y * 4] - 0.299).abs() < 1e-5, "left edge {:?}", g);
        assert!((g[y * 4 + 3] - 0.114).abs() < 1e-5, "right edge {:?}", g);
    }
}

/// `gaussian_blur` reads planar CHW (see `confirmed_hal_fixes`). Blurring a
/// resized RGB tensor must equal blurring each resized plane on its own.
#[test]
fn cpu_resize_then_gaussian_blur_is_per_plane() {
    let cpu = cpu();
    let rgb = structured_rgb(16, 12);
    let blurred = cpu
        .gaussian_blur(&cpu.resize(&rgb, (9, 7)).unwrap(), 1.2, 5)
        .unwrap();
    for ch in 0..3 {
        let alone = cpu
            .gaussian_blur(&cpu.resize(&plane(&rgb, ch), (9, 7)).unwrap(), 1.2, 5)
            .unwrap();
        let d = max_diff(
            plane(&blurred, ch).as_slice().unwrap(),
            alone.as_slice().unwrap(),
        );
        assert!(d < 1e-6, "plane {ch} differs by {d}");
    }
}

// ------------------------------------------------------------------ GPU

fn gpu_resize(g: &GpuContext, input: &Cpu, nw: usize, nh: usize, interp: Interpolation) -> Cpu {
    let on_gpu = input.to_gpu_ctx(g).unwrap();
    g.resize_with(&on_gpu, (nw, nh), interp)
        .unwrap()
        .to_cpu_ctx(g)
        .unwrap()
}

#[test]
fn gpu_resize_keeps_constant_planes_constant() {
    let Some(g) = gpu() else { return };
    for interp in [Interpolation::Linear, Interpolation::Lanczos] {
        for ((h, w), (nw, nh)) in SIZES {
            let input = constant_planes(&[0.0, 100.0, 200.0], h, w);
            let out = gpu_resize(g, &input, nw, nh, interp);
            assert_eq!(out.shape, TensorShape::new(3, nh, nw));
            // Lanczos normalises by the weight sum, so a constant comes back
            // to within rounding.
            assert_constant_planes(&out, &[0.0, 100.0, 200.0], 1e-3, &format!("GPU {interp:?}"));
        }
    }
}

#[test]
fn gpu_resize_of_each_plane_is_the_one_channel_resize() {
    let Some(g) = gpu() else { return };
    for interp in [Interpolation::Linear, Interpolation::Lanczos] {
        for ((h, w), (nw, nh)) in SIZES {
            let input = structured_rgb(h, w);
            let out = gpu_resize(g, &input, nw, nh, interp);
            for ch in 0..3 {
                let alone = gpu_resize(g, &plane(&input, ch), nw, nh, interp);
                assert_eq!(
                    plane(&out, ch).as_slice().unwrap(),
                    alone.as_slice().unwrap(),
                    "GPU {interp:?} plane {ch}, {w}x{h} -> {nw}x{nh}"
                );
            }
        }
    }
}

/// The GPU bilinear resize agrees with the CPU one on a multichannel tensor
/// (the existing parity probe uses random data, which cannot tell the
/// layouts apart when both backends share the same mistake).
#[test]
fn gpu_bilinear_resize_matches_cpu_per_plane() {
    let Some(g) = gpu() else { return };
    let cpu = cpu();
    for ((h, w), (nw, nh)) in SIZES {
        let input = structured_rgb(h, w);
        let c = cpu.resize(&input, (nw, nh)).unwrap();
        let gr = gpu_resize(g, &input, nw, nh, Interpolation::Linear);
        let d = max_diff(c.as_slice().unwrap(), gr.as_slice().unwrap());
        assert!(d < 1e-5, "{w}x{h} -> {nw}x{nh}: CPU and GPU differ by {d}");
    }
}

#[test]
fn gpu_resize_commutes_with_cvt_color_and_matches_cpu() {
    let Some(g) = gpu() else { return };
    let cpu = cpu();
    for ((h, w), (nw, nh)) in SIZES {
        let rgb = structured_rgb(h, w);
        let rgb_gpu = rgb.to_gpu_ctx(g).unwrap();
        let a = g
            .cvt_color(
                &g.resize(&rgb_gpu, (nw, nh)).unwrap(),
                ColorConversion::RgbToGray,
            )
            .unwrap()
            .to_cpu_ctx(g)
            .unwrap();
        let b = g
            .resize(
                &g.cvt_color(&rgb_gpu, ColorConversion::RgbToGray).unwrap(),
                (nw, nh),
            )
            .unwrap()
            .to_cpu_ctx(g)
            .unwrap();
        let d = max_diff(a.as_slice().unwrap(), b.as_slice().unwrap());
        assert!(d < 1e-5, "GPU {w}x{h} -> {nw}x{nh}: orders differ by {d}");

        let c = cpu
            .cvt_color(
                &cpu.resize(&rgb, (nw, nh)).unwrap(),
                ColorConversion::RgbToGray,
            )
            .unwrap();
        let d = max_diff(a.as_slice().unwrap(), c.as_slice().unwrap());
        assert!(d < 1e-5, "GPU vs CPU {w}x{h} -> {nw}x{nh}: differ by {d}");
    }
}

#[test]
fn gpu_resize_then_gaussian_blur_is_per_plane() {
    let Some(g) = gpu() else { return };
    let rgb = structured_rgb(16, 12);
    let blur_of_resize = |t: &Cpu| -> Cpu {
        let on = t.to_gpu_ctx(g).unwrap();
        g.gaussian_blur(&g.resize(&on, (9, 7)).unwrap(), 1.2, 5)
            .unwrap()
            .to_cpu_ctx(g)
            .unwrap()
    };
    let blurred = blur_of_resize(&rgb);
    for ch in 0..3 {
        let alone = blur_of_resize(&plane(&rgb, ch));
        let d = max_diff(
            plane(&blurred, ch).as_slice().unwrap(),
            alone.as_slice().unwrap(),
        );
        assert!(d < 1e-6, "plane {ch} differs by {d}");
    }
}

// ------------------------------------------------------------ empty sizes

/// A zero target is an empty tensor of the target shape; an empty source
/// with a non-empty target is zeros of the target shape. `cv_imgproc::resize`
/// does the same for an empty image. The CPU used to panic on both
/// (`par_chunks_mut(0)`, and indexing an empty source).
#[test]
fn empty_sizes_do_not_panic_and_agree() {
    let cpu = cpu();
    let g = gpu();
    let cases: &[((usize, usize, usize), (usize, usize))] = &[
        // (c, h, w) -> (nw, nh)
        ((3, 0, 0), (5, 4)),
        ((3, 0, 7), (5, 4)),
        ((3, 7, 0), (5, 4)),
        ((3, 5, 5), (0, 4)),
        ((3, 5, 5), (4, 0)),
        ((3, 5, 5), (0, 0)),
        ((1, 0, 0), (0, 0)),
        ((0, 5, 5), (3, 3)),
    ];
    for &((c, h, w), (nw, nh)) in cases {
        let input = tensor(vec![1.0; c * h * w], c, h, w);
        let want_shape = TensorShape::new(c, nh, nw);
        let want = vec![0.0f32; want_shape.len()];
        let what = format!("({c},{h},{w}) -> {nw}x{nh}");

        let out = cpu.resize(&input, (nw, nh)).unwrap();
        assert_eq!(out.shape, want_shape, "CPU {what}");
        assert_eq!(out.as_slice().unwrap(), &want[..], "CPU {what}");

        let Some(g) = g else { continue };
        let on_gpu = input.to_gpu_ctx(g).unwrap();
        for interp in [Interpolation::Linear, Interpolation::Lanczos] {
            let out = g.resize_with(&on_gpu, (nw, nh), interp).unwrap();
            assert_eq!(out.shape, want_shape, "GPU {interp:?} {what}");
            if want.is_empty() {
                assert_eq!(out.storage.len(), 0, "GPU {interp:?} {what}");
            } else {
                let back = out.to_cpu_ctx(g).unwrap();
                assert_eq!(back.as_slice().unwrap(), &want[..], "GPU {interp:?} {what}");
            }
        }
    }
}
