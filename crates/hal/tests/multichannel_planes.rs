//! Multi-channel (planar CHW) images through every image `ComputeContext`
//! entry, on the CPU and GPU backends.
//!
//! Several CPU operations read only the first `h * w` values of their input,
//! plane 0, while labelling the output with the input's channel count:
//! `sobel` returned plane 0's gradients followed by copies of plane 0's last
//! row, `hough_lines` voted plane 0 only, `sift_extrema` failed with a
//! `RuntimeError` (shape mismatch), and the AKAZE and SIFT helpers sampled
//! plane 0 or read across plane boundaries. Each operation now has one rule
//! for several channels, the same on both backends:
//!
//! * **Per plane**, where `cv::` processes each channel independently
//!   (`filter2D`, `threshold`, `Sobel`, `GaussianBlur`, `pyrDown`, `resize`,
//!   `erode`/`dilate`, `warpAffine`, `subtract`): plane `k` of the output is
//!   the operation applied to plane `k` alone.
//! * **`InvalidInput`**, where the operation is single-channel by definition
//!   or `cv::` requires one channel (`HoughLines`, `HoughCircles`, `StereoBM`,
//!   SIFT's DoG extrema and descriptors, AKAZE's diffusion, MOG2's model, LK
//!   on a single-channel pyramid, the non-maximum suppression and FAST
//!   response maps), or where OpenCV's multi-channel variant is not per plane
//!   and is not implemented here (`Canny`, which takes the strongest channel's
//!   gradient). `bilateral_filter` returns `NotSupported` on both backends
//!   (OpenCV's colour version weights by the summed channel distance).
//!
//! The GPU may answer `NotSupported` only for operations it does not
//! implement at all (`detect_objects`, `find_chessboard_corners`).

use cv_core::storage::Storage;
use cv_core::{CpuTensor, KeyPoint, KeyPoints, Tensor, TensorShape};
use cv_hal::context::{
    BorderMode, ComputeContext, DiffusivityType, Mog2Params, MorphologyType, StereoMatchMethod,
    StereoMatchParams, ThresholdType, WarpType,
};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Out = cv_hal::Result<Vec<CpuTensor<f32>>>;

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("no GPU adapter: checking the CPU backend only");
            None
        }
    }
}

const C: usize = 3;
const H: usize = 14;
const W: usize = 17;

/// Distinct integer-valued planes in 0..=255.
fn image() -> CpuTensor<f32> {
    let mut v = Vec::with_capacity(C * H * W);
    for c in 0..C {
        for y in 0..H {
            for x in 0..W {
                v.push(((c * 83 + y * 37 + x * 59 + (x * y * 13) % 31 + c * x * 7) % 256) as f32);
            }
        }
    }
    Tensor::from_vec(v, TensorShape::new(C, H, W)).unwrap()
}

fn plane<T: bytemuck::Pod + std::fmt::Debug + Send + Sync + Clone + Copy>(
    t: &CpuTensor<T>,
    k: usize,
) -> CpuTensor<T> {
    let (h, w) = t.shape.hw();
    let d = t.storage.as_slice().unwrap();
    Tensor::from_vec(
        d[k * h * w..(k + 1) * h * w].to_vec(),
        TensorShape::new(1, h, w),
    )
    .unwrap()
}

fn data(t: &CpuTensor<f32>) -> &[f32] {
    t.storage.as_slice().unwrap()
}

fn down(g: &GpuContext, t: &Tensor<f32, cv_hal::storage::GpuStorage<f32>>) -> CpuTensor<f32> {
    t.to_cpu_ctx(g).unwrap()
}

fn kind(e: &cv_hal::Error) -> String {
    let d = format!("{e:?}");
    d.split('(').next().unwrap_or(&d).to_string()
}

/// Runs `f` and turns a panic into an error string.
fn run<R>(f: impl FnOnce() -> cv_hal::Result<R>) -> Result<R, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Err(format!("Err {}", kind(&e))),
        Err(_) => Err("PANIC".into()),
    }
}

/// Plane `k` of every output of `three` (the operation on the 3-channel
/// image) must equal the output of `one(k)` (the operation on plane `k`
/// alone). Returns the 3-channel outputs.
fn per_plane(
    failures: &mut Vec<String>,
    what: &str,
    three: impl FnOnce() -> Out,
    one: impl Fn(usize) -> Out,
    tol: f32,
) -> Option<Vec<CpuTensor<f32>>> {
    let outs = match run(three) {
        Ok(o) => o,
        Err(e) => {
            failures.push(format!("{what}: 3 channels gave {e}"));
            return None;
        }
    };
    for (j, out) in outs.iter().enumerate() {
        if out.shape.channels != C {
            failures.push(format!("{what}: output {j} has shape {:?}", out.shape));
            return None;
        }
    }
    for k in 0..C {
        let singles = match run(|| one(k)) {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{what}: plane {k} alone gave {e}"));
                return None;
            }
        };
        for (j, (out, single)) in outs.iter().zip(&singles).enumerate() {
            let got = plane(out, k);
            let worst = data(&got)
                .iter()
                .zip(data(single))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if got.shape != single.shape || !(worst <= tol) {
                failures.push(format!(
                    "{what}: output {j} plane {k} differs from the plane alone \
                     (max |diff| {worst}, shapes {:?} / {:?})",
                    got.shape, single.shape
                ));
            }
        }
    }
    Some(outs)
}

/// The GPU's 3-channel outputs must agree with the CPU's.
fn agree(
    failures: &mut Vec<String>,
    what: &str,
    cpu: &[CpuTensor<f32>],
    gpu: &[CpuTensor<f32>],
    tol: f32,
) {
    for (j, (c, g)) in cpu.iter().zip(gpu).enumerate() {
        let worst = data(c)
            .iter()
            .zip(data(g))
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0f32, f32::max);
        if c.shape != g.shape || !(worst <= tol) {
            failures.push(format!(
                "{what}: GPU output {j} differs from the CPU (max rel diff {worst}, shapes {:?} / {:?})",
                c.shape, g.shape
            ));
        }
    }
}

fn report(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{} failures:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn per_plane_operations_process_every_channel() {
    let cpu = CpuBackend::new().unwrap();
    let img = image();
    let k3 = Tensor::from_vec(
        vec![1.0f32, 2.0, -1.0, 0.0, 3.0, 1.0, -2.0, 1.0, 0.5],
        TensorShape::new(1, 3, 3),
    )
    .unwrap();
    let m = [[0.9f32, 0.1, 1.5], [-0.05, 1.1, 0.5], [0.0, 0.0, 1.0]];
    let g = gpu();
    let mut f = Vec::new();

    // Each case: name, CPU op, GPU op (None where the GPU has no such
    // variant), tolerance between the GPU and the CPU.
    type CpuOp<'a> = Box<dyn Fn(&CpuTensor<f32>) -> Out + 'a>;
    type GpuOp<'a> = Box<dyn Fn(&GpuContext, &CpuTensor<f32>) -> Out + 'a>;
    let cases: Vec<(&str, CpuOp, Option<GpuOp>, f32)> = vec![
        (
            "convolve_2d",
            Box::new(|t| Ok(vec![cpu.convolve_2d(t, &k3, BorderMode::Replicate)?])),
            Some(Box::new(|g, t| {
                let o =
                    g.convolve_2d(&t.to_gpu_ctx(g)?, &k3.to_gpu_ctx(g)?, BorderMode::Replicate)?;
                Ok(vec![down(g, &o)])
            })),
            1e-5,
        ),
        (
            "threshold",
            Box::new(|t| {
                Ok(vec![cpu.threshold(
                    t,
                    100.0,
                    255.0,
                    ThresholdType::Binary,
                )?])
            }),
            Some(Box::new(|g, t| {
                let o = g.threshold(&t.to_gpu_ctx(g)?, 100.0, 255.0, ThresholdType::Binary)?;
                Ok(vec![down(g, &o)])
            })),
            0.0,
        ),
        (
            "sobel dx=1 dy=1 ksize=3",
            Box::new(|t| {
                let (a, b) = cpu.sobel(t, 1, 1, 3)?;
                Ok(vec![a, b])
            }),
            Some(Box::new(|g, t| {
                let (a, b) = g.sobel(&t.to_gpu_ctx(g)?, 1, 1, 3)?;
                Ok(vec![down(g, &a), down(g, &b)])
            })),
            1e-6,
        ),
        (
            "sobel dx=1 dy=1 ksize=5",
            Box::new(|t| {
                let (a, b) = cpu.sobel(t, 1, 1, 5)?;
                Ok(vec![a, b])
            }),
            None,
            0.0,
        ),
        (
            "gaussian_blur",
            Box::new(|t| Ok(vec![cpu.gaussian_blur(t, 1.2, 5)?])),
            Some(Box::new(|g, t| {
                Ok(vec![down(g, &g.gaussian_blur(&t.to_gpu_ctx(g)?, 1.2, 5)?)])
            })),
            1e-5,
        ),
        (
            "pyramid_down",
            Box::new(|t| Ok(vec![cpu.pyramid_down(t)?])),
            Some(Box::new(|g, t| {
                Ok(vec![down(g, &g.pyramid_down(&t.to_gpu_ctx(g)?)?)])
            })),
            1e-5,
        ),
        (
            "resize",
            Box::new(|t| Ok(vec![cpu.resize(t, (9, 23))?])),
            Some(Box::new(|g, t| {
                Ok(vec![down(g, &g.resize(&t.to_gpu_ctx(g)?, (9, 23))?)])
            })),
            1e-5,
        ),
        (
            "warp",
            Box::new(|t| Ok(vec![cpu.warp(t, &m, (15, 12), WarpType::Affine)?])),
            Some(Box::new(|g, t| {
                let o = g.warp(&t.to_gpu_ctx(g)?, &m, (15, 12), WarpType::Affine)?;
                Ok(vec![down(g, &o)])
            })),
            1e-4,
        ),
        (
            "subtract",
            Box::new(|t| {
                let b = cpu.threshold(t, 128.0, 7.0, ThresholdType::Binary)?;
                Ok(vec![cpu.subtract(t, &b)?])
            }),
            Some(Box::new(|g, t| {
                let b = cpu.threshold(t, 128.0, 7.0, ThresholdType::Binary)?;
                Ok(vec![down(
                    g,
                    &g.subtract(&t.to_gpu_ctx(g)?, &b.to_gpu_ctx(g)?)?,
                )])
            })),
            0.0,
        ),
    ];

    for (what, cpu_op, gpu_op, tol) in &cases {
        let c3 = per_plane(
            &mut f,
            &format!("CPU {what}"),
            || cpu_op(&img),
            |k| cpu_op(&plane(&img, k)),
            0.0,
        );
        if let (Some(g), Some(gpu_op)) = (g, gpu_op) {
            let g3 = per_plane(
                &mut f,
                &format!("GPU {what}"),
                || gpu_op(g, &img),
                |k| gpu_op(g, &plane(&img, k)),
                0.0,
            );
            if let (Some(c3), Some(g3)) = (c3, g3) {
                agree(&mut f, what, &c3, &g3, *tol);
            }
        }
    }

    // morphology works on u8.
    let img8: CpuTensor<u8> =
        Tensor::from_vec(data(&img).iter().map(|&v| v as u8).collect(), img.shape).unwrap();
    let k8: CpuTensor<u8> =
        Tensor::from_vec(vec![0, 1, 0, 1, 1, 1, 0, 1, 1], TensorShape::new(1, 3, 3)).unwrap();
    for typ in [MorphologyType::Erode, MorphologyType::Close] {
        let to_f = |t: CpuTensor<u8>| -> CpuTensor<f32> {
            Tensor::from_vec(
                t.storage
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
                t.shape,
            )
            .unwrap()
        };
        let cpu_op = |t: &CpuTensor<u8>| -> Out { Ok(vec![to_f(cpu.morphology(t, typ, &k8, 2)?)]) };
        let c3 = per_plane(
            &mut f,
            &format!("CPU morphology {typ:?}"),
            || cpu_op(&img8),
            |k| cpu_op(&plane(&img8, k)),
            0.0,
        );
        if let Some(g) = g {
            let gpu_op = |t: &CpuTensor<u8>| -> Out {
                let o = g.morphology(&t.to_gpu_ctx(g)?, typ, &k8.to_gpu_ctx(g)?, 2)?;
                Ok(vec![to_f(o.to_cpu_ctx(g)?)])
            };
            let g3 = per_plane(
                &mut f,
                &format!("GPU morphology {typ:?}"),
                || gpu_op(&img8),
                |k| gpu_op(&plane(&img8, k)),
                0.0,
            );
            if let (Some(c3), Some(g3)) = (c3, g3) {
                agree(&mut f, &format!("morphology {typ:?}"), &c3, &g3, 0.0);
            }
        }
    }
    report(f);
}

/// `sobel` on three planes used to leave planes 1 and 2 as copies of plane
/// 0's last gradient row on the CPU, and as zeros on the GPU. Spelled out
/// separately from the sweep above so the failure names the values.
#[test]
fn sobel_fills_every_plane() {
    let cpu = CpuBackend::new().unwrap();
    let img = image();
    let (gx, gy) = cpu.sobel(&img, 1, 1, 3).unwrap();
    for k in 0..C {
        let (ax, ay) = cpu.sobel(&plane(&img, k), 1, 1, 3).unwrap();
        assert_eq!(data(&plane(&gx, k)), data(&ax), "CPU gx plane {k}");
        assert_eq!(data(&plane(&gy, k)), data(&ay), "CPU gy plane {k}");
    }
    let Some(g) = gpu() else { return };
    let (gx, gy) = g.sobel(&img.to_gpu_ctx(g).unwrap(), 1, 1, 3).unwrap();
    let (gx, gy) = (down(g, &gx), down(g, &gy));
    for k in 0..C {
        let (ax, ay) = cpu.sobel(&plane(&img, k), 1, 1, 3).unwrap();
        assert_eq!(data(&plane(&gx, k)), data(&ax), "GPU gx plane {k}");
        assert_eq!(data(&plane(&gy, k)), data(&ay), "GPU gy plane {k}");
    }
}

#[test]
fn single_channel_operations_reject_several_channels() {
    let cpu = CpuBackend::new().unwrap();
    let img = image();
    let g = gpu();
    let mut f = Vec::new();

    // (name, CPU outcome, GPU outcome, expected kind, GPU may say NotSupported)
    let mut check = |what: &str,
                     c: Result<(), String>,
                     gk: Option<Result<(), String>>,
                     want: &str,
                     gpu_unsupported: bool| {
        let c = c.err().unwrap_or_else(|| "Ok".into());
        if c != want {
            f.push(format!("CPU {what}: {c}, want {want}"));
        }
        if let Some(gk) = gk {
            let gk = gk.err().unwrap_or_else(|| "Ok".into());
            if gk != want && !(gpu_unsupported && gk == "Err NotSupported") {
                f.push(format!("GPU {what}: {gk}, want {want}"));
            }
        }
    };
    let invalid = "Err InvalidInput";
    let gt = |g: &GpuContext| img.to_gpu_ctx(g).unwrap();

    check(
        "canny",
        run(|| cpu.canny(&img, 10.0, 30.0).map(drop)),
        g.map(|g| run(|| g.canny(&gt(g), 10.0, 30.0).map(drop))),
        invalid,
        false,
    );
    check(
        "hough_lines",
        run(|| cpu.hough_lines(&img, 1.0, 0.05, 5).map(drop)),
        g.map(|g| run(|| g.hough_lines(&gt(g), 1.0, 0.05, 5).map(drop))),
        invalid,
        false,
    );
    check(
        "hough_circles",
        run(|| cpu.hough_circles(&img, 2.0, 5.0, 3).map(drop)),
        g.map(|g| run(|| g.hough_circles(&gt(g), 2.0, 5.0, 3).map(drop))),
        invalid,
        false,
    );
    check(
        "nms",
        run(|| cpu.nms(&img, 10.0, 3).map(drop)),
        g.map(|g| run(|| g.nms(&gt(g), 10.0, 3).map(drop))),
        invalid,
        false,
    );
    for nms in [false, true] {
        check(
            &format!("fast_detect nms {nms}"),
            run(|| cpu.fast_detect(&img, 20.0, nms).map(drop)),
            g.map(|g| run(|| g.fast_detect(&gt(g), 20.0, nms).map(drop))),
            invalid,
            false,
        );
    }
    check(
        "sift_extrema",
        run(|| cpu.sift_extrema(&img, &img, &img, 0.5, 10.0).map(drop)),
        g.map(|g| run(|| g.sift_extrema(&gt(g), &gt(g), &gt(g), 0.5, 10.0).map(drop))),
        invalid,
        false,
    );
    let kps = KeyPoints {
        keypoints: vec![KeyPoint::new(8.0, 7.0).with_size(2.0)],
    };
    check(
        "compute_sift_descriptors",
        run(|| cpu.compute_sift_descriptors(&img, &kps).map(drop)),
        g.map(|g| run(|| g.compute_sift_descriptors(&gt(g), &kps).map(drop))),
        invalid,
        false,
    );
    check(
        "akaze_diffusion",
        run(|| {
            cpu.akaze_diffusion(&img, 5.0, 0.25, DiffusivityType::PeronaMalik2)
                .map(drop)
        }),
        g.map(|g| {
            run(|| {
                g.akaze_diffusion(&gt(g), 5.0, 0.25, DiffusivityType::PeronaMalik2)
                    .map(drop)
            })
        }),
        invalid,
        false,
    );
    check(
        "akaze_derivatives",
        run(|| cpu.akaze_derivatives(&img).map(drop)),
        g.map(|g| run(|| g.akaze_derivatives(&gt(g)).map(drop))),
        invalid,
        false,
    );
    check(
        "akaze_contrast_k",
        run(|| cpu.akaze_contrast_k(&img).map(drop)),
        g.map(|g| run(|| g.akaze_contrast_k(&gt(g)).map(drop))),
        invalid,
        false,
    );
    let sp = StereoMatchParams {
        method: StereoMatchMethod::BlockMatching,
        min_disparity: 0,
        num_disparities: 16,
        block_size: 5,
    };
    check(
        "stereo_match",
        run(|| {
            cpu.stereo_match::<f32, _, cv_core::storage::CpuStorage<f32>>(&img, &img, &sp)
                .map(drop)
        }),
        g.map(|g| {
            run(|| {
                g.stereo_match::<f32, _, cv_hal::storage::GpuStorage<f32>>(&gt(g), &gt(g), &sp)
                    .map(drop)
            })
        }),
        invalid,
        false,
    );
    check(
        "optical_flow_lk",
        run(|| {
            cpu.optical_flow_lk(&[img.clone()], &[img.clone()], &[[8.0, 7.0]], 5, 10)
                .map(drop)
        }),
        g.map(|g| {
            run(|| {
                g.optical_flow_lk(&[gt(g)], &[gt(g)], &[[8.0, 7.0]], 5, 10)
                    .map(drop)
            })
        }),
        invalid,
        false,
    );
    check(
        "find_chessboard_corners",
        run(|| cpu.find_chessboard_corners(&img, (3, 3)).map(drop)),
        g.map(|g| run(|| g.find_chessboard_corners(&gt(g), (3, 3)).map(drop))),
        invalid,
        true,
    );
    // Large enough for the 24x24 detector window.
    let big: CpuTensor<f32> =
        Tensor::from_vec(vec![1.0; 3 * 30 * 30], TensorShape::new(3, 30, 30)).unwrap();
    check(
        "detect_objects",
        run(|| cpu.detect_objects(&big, 0.5).map(drop)),
        g.map(|g| run(|| g.detect_objects(&big.to_gpu_ctx(g)?, 0.5).map(drop))),
        invalid,
        true,
    );
    check(
        "bilateral_filter",
        run(|| cpu.bilateral_filter(&img, 5, 10.0, 2.0).map(drop)),
        g.map(|g| run(|| g.bilateral_filter(&gt(g), 5, 10.0, 2.0).map(drop))),
        "Err NotSupported",
        false,
    );
    let mp = Mog2Params::<f32> {
        width: W as u32,
        height: H as u32,
        n_mixtures: 3,
        alpha: 0.01,
        var_threshold: 16.0,
        background_ratio: 0.9,
        var_init: 15.0,
        var_min: 4.0,
        var_max: 75.0,
        adaptive_k: 0,
        max_components: 3,
        _padding: [0],
    };
    check(
        "mog2_update",
        run(|| {
            let mut model: CpuTensor<f32> =
                Tensor::from_vec(vec![0.0; H * W * 9], TensorShape::new(1, 1, H * W * 9)).unwrap();
            let mut mask: CpuTensor<u32> = Tensor::from_vec(vec![0; C * H * W], img.shape).unwrap();
            cpu.mog2_update(&img, &mut model, &mut mask, &mp)
        }),
        g.map(|g| {
            run(|| {
                let model: CpuTensor<f32> =
                    Tensor::from_vec(vec![0.0; H * W * 9], TensorShape::new(1, 1, H * W * 9))
                        .unwrap();
                let mask: CpuTensor<u32> = Tensor::from_vec(vec![0; C * H * W], img.shape).unwrap();
                let (mut model, mut mask) = (model.to_gpu_ctx(g)?, mask.to_gpu_ctx(g)?);
                g.mog2_update(&gt(g), &mut model, &mut mask, &mp)
            })
        }),
        invalid,
        false,
    );
    report(f);
}

/// `sift_extrema` on a single channel still finds the extremum; a level of
/// three planes is rejected as invalid input rather than failing deep in
/// tensor construction with a `RuntimeError`.
#[test]
fn sift_extrema_single_channel_still_works() {
    let cpu = CpuBackend::new().unwrap();
    let mut curr = vec![0.0f32; 49];
    curr[3 * 7 + 3] = 5.0;
    let flat = || Tensor::from_vec(vec![0.0f32; 49], TensorShape::new(1, 7, 7)).unwrap();
    let curr: CpuTensor<f32> = Tensor::from_vec(curr, TensorShape::new(1, 7, 7)).unwrap();
    let m = cpu
        .sift_extrema(&flat(), &curr, &flat(), 0.5, 10.0)
        .unwrap();
    assert_eq!(m.shape, TensorShape::new(1, 7, 7));
    assert_eq!(
        m.storage
            .as_slice()
            .unwrap()
            .iter()
            .filter(|&&v| v != 0)
            .count(),
        1
    );
    assert_eq!(m.storage.as_slice().unwrap()[3 * 7 + 3], 1);
    if let Some(g) = gpu() {
        let up = |t: &CpuTensor<f32>| t.to_gpu_ctx(g).unwrap();
        let gm = g
            .sift_extrema(&up(&flat()), &up(&curr), &up(&flat()), 0.5, 10.0)
            .unwrap();
        assert_eq!(gm.shape, m.shape);
        assert_eq!(
            gm.storage.as_slice().unwrap(),
            m.storage.as_slice().unwrap()
        );
    }
}

/// A sample a hair outside the source edge is the edge pixel, on both
/// backends. With an exact `[0, w - 1]` bounds test the outcome depended on the
/// last ulp of the mapped coordinate: macOS CI (Metal, fused multiply-add) got
/// -7.45e-9 for `-0.05 * 10 + 0.5` and wrote 0 where the CPU, rounding step by
/// step, got 0.0 and sampled the pixel. A translation of -1e-6 puts every
/// column of the first destination column just outside the source.
#[test]
fn warp_samples_an_edge_coordinate_off_by_an_ulp() {
    let cpu = CpuBackend::new().unwrap();
    let img = image();
    let m = [[1.0f32, 0.0, -1e-6], [0.0, 1.0, -1e-6], [0.0, 0.0, 1.0]];
    let out = cpu.warp(&img, &m, (W, H), WarpType::Affine).unwrap();
    let (o, i) = (data(&out), data(&img));
    for c in 0..C {
        let p = c * H * W;
        assert_eq!(
            o[p], i[p],
            "plane {c}: the (0, 0) sample at -1e-6 px fell outside"
        );
    }
    if let Some(g) = gpu() {
        let go = g
            .warp(&img.to_gpu_ctx(g).unwrap(), &m, (W, H), WarpType::Affine)
            .unwrap();
        let go = down(g, &go);
        for (k, (a, b)) in data(&go).iter().zip(o).enumerate() {
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "GPU sample {k}: {a} vs CPU {b}"
            );
        }
    }
}
