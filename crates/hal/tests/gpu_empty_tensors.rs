//! Empty tensors through every `ComputeContext` entry, on the CPU and GPU.
//!
//! Each case runs the same empty input (a zero width, height or channel
//! count, or a point/box/descriptor set with no rows) on both backends. A
//! case fails if either backend panics, or if the two disagree: they must
//! return an equal result (shape and data, or value) or an error of the same
//! kind. Without a GPU adapter the CPU side still runs and must not panic.
//! `EMPTY_TENSORS_VERBOSE=1` prints every case's outcome.
//!
//! A tensor with a zero dimension uploads as a zero-size buffer, which wgpu
//! refuses to bind (its default error handler panics), so every GPU kernel
//! returns before binding one. On the CPU the usual failure was
//! `par_chunks_mut(0)` ("chunk_size must not be zero"), `h - 1` underflow, or
//! indexing a `(0, h, w)` tensor's empty storage by its height and width.
//!
//! # What an empty input returns
//!
//! For an empty `cv::Mat` most OpenCV functions raise (`!_src.empty()`). Here
//! an operation whose output is naturally empty returns it; one whose output
//! cannot be computed is an error. One rule per operation, on both backends:
//!
//! * **Empty tensor of the output shape** - an image in, an image of the
//!   input's size out: `convolve_2d` (image), `threshold`, `sobel` (both
//!   gradients), `canny`, `gaussian_blur`, `bilateral_filter`, `nms`,
//!   `subtract`, `morphology`, `akaze_diffusion`, `akaze_derivatives`,
//!   `stereo_match`, `sift_extrema`, `fast_detect`, `cvt_color` (the target's
//!   channel count), `pointcloud_transform`, `pointcloud_normals`, `spmv`
//!   (no rows). Also every operation asked for an empty output: `warp`,
//!   `remap`, `undistort`, `tsdf_raycast` and `resize` to a zero size.
//!   Channel-count checks come first, so `canny`, `nms`, `fast_detect`,
//!   `stereo_match`, `bilateral_filter` and `cvt_color` reject a wrong channel
//!   count even when the image is empty (the GPU says `NotSupported` where it
//!   only implements one channel count; see `GPU_UNSUPPORTED`).
//! * **Empty list** - nothing to find: `hough_lines`, `hough_circles`,
//!   `detect_objects`, `nms_boxes`, `nms_rotated_boxes`, `nms_polygons`,
//!   `icp_correspondences` (either cloud empty), `match_descriptors` (either
//!   set empty), `optical_flow_lk` with no points, `compute_sift_descriptors`
//!   with no keypoints, `tsdf_extract_mesh`.
//! * **Unchanged state / neutral value** - `tsdf_integrate` of an empty depth
//!   image leaves the volume as it was; `mog2_update` of an empty frame is a
//!   no-op; `dense_icp_step` and `icp_accumulate` return a zero system;
//!   `akaze_contrast_k` returns OpenCV's 0.03 fallback; `tsdf_raycast` of an
//!   empty volume returns zeros (every ray misses).
//! * **`InvalidInput`** - the output is not empty but there is nothing to
//!   compute it from, which OpenCV also rejects: `convolve_2d` with an empty
//!   kernel (`!_kernel.empty()`); `match_template` with an empty image or
//!   template; `warp` of an empty source to any size (`src.cols > 0 &&
//!   src.rows > 0`); `remap` / `undistort` of an empty image to a non-empty
//!   output; `compute_sift_descriptors` of keypoints on an empty image;
//!   `pyramid_down` of a zero width or height (`!_src.empty()`; a plane with
//!   no channels halves to an empty `(0, h/2, w/2)`); `find_chessboard_corners`;
//!   `optical_flow_lk` with an empty pyramid level; `tsdf_integrate` into an
//!   empty volume; `triangulate_points` with no points (the GPU does not
//!   triangulate at all). `subtract` of an empty and a non-empty tensor is a
//!   size mismatch. `fast_detect` and `nms` reject a channel count other than
//!   one, empty or not (`fast_detect` used to label its `h * w` scores with
//!   the input's channel count: 16 zeros as `(0, 4, 4)`).
//! * **OpenCV's default** - `morphology` with an empty structuring element
//!   uses a 3x3 rectangle, as `cv::erode` / `cv::dilate` do for `noArray()`.
//! * **`resize`** of an empty source to a non-empty size returns zeros of the
//!   target size, the rule `cv_imgproc::resize` follows; it is not this
//!   test's to change, only to hold the backends to.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, KeyPoint, KeyPoints, Tensor, TensorShape};
use cv_hal::context::{
    BorderMode, ColorConversion, ComputeContext, DiffusivityType, Interpolation, Mog2Params,
    MorphologyType, StereoMatchMethod, StereoMatchParams, TemplateMatchMethod, ThresholdType,
    WarpType,
};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::GpuStorage;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};

type GT<T> = Tensor<T, GpuStorage<T>>;

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("no GPU adapter: checking the CPU backend only");
            None
        }
    }
}

/// Shapes with no elements, as (channels, height, width).
const EMPTY: [(usize, usize, usize); 5] = [(1, 0, 0), (1, 0, 8), (1, 8, 0), (3, 0, 8), (0, 4, 4)];

fn cpu_t<T: Clone + Copy + bytemuck::Pod + Debug + Send + Sync>(
    v: Vec<T>,
    (c, h, w): (usize, usize, usize),
) -> CpuTensor<T> {
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

fn empty<T: Clone + Copy + bytemuck::Pod + Debug + Send + Sync>(
    s: (usize, usize, usize),
) -> CpuTensor<T> {
    cpu_t(Vec::new(), s)
}

/// The error's variant, which both backends must agree on; the messages
/// differ.
fn kind(e: &cv_hal::Error) -> String {
    let d = format!("{e:?}");
    d.split('(').next().unwrap_or(&d).to_string()
}

fn outcome(f: impl FnOnce() -> cv_hal::Result<String>) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(s)) => format!("Ok {s}"),
        Ok(Err(e)) => format!("Err {}", kind(&e)),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            format!("PANIC {msg}")
        }
    }
}

fn show<T: Clone + Copy + bytemuck::Pod + Debug + Send + Sync>(t: &CpuTensor<T>) -> String {
    format!("{:?} {:?}", t.shape, t.storage.as_slice().unwrap())
}

fn show_gpu<T: Clone + Copy + bytemuck::Pod + Debug + Send + Sync>(
    g: &GpuContext,
    t: &GT<T>,
) -> cv_hal::Result<String> {
    Ok(show(&t.to_cpu_ctx(g)?))
}

/// Operations (or variants) the GPU backend does not implement for any
/// input; for these an empty input may get the same `NotSupported` as a
/// non-empty one.
const GPU_UNSUPPORTED: &[&str] = &[
    "cvt_color GrayToRgb",
    "detect_objects",
    "find_chessboard_corners",
    "triangulate_points",
    // Single-channel only on the GPU: three channels, or none.
    "fast_detect nms false (3",
    "fast_detect nms true (3",
    "fast_detect nms false (0",
    "fast_detect nms true (0",
    "nms (3",
    "nms (0",
    "warp input (3",
    "warp input (0",
];

struct Report {
    gpu: Option<&'static GpuContext>,
    verbose: bool,
    cases: usize,
    panics: usize,
    failures: Vec<String>,
}

impl Report {
    fn new(gpu: Option<&'static GpuContext>) -> Self {
        Self {
            gpu,
            verbose: std::env::var_os("EMPTY_TENSORS_VERBOSE").is_some(),
            cases: 0,
            panics: 0,
            failures: Vec::new(),
        }
    }

    /// Neither backend may panic, and the GPU must return what the CPU does:
    /// an equal value, or an error of the same kind.
    fn case(
        &mut self,
        what: String,
        cpu: impl FnOnce() -> cv_hal::Result<String>,
        gpu: impl FnOnce(&GpuContext) -> cv_hal::Result<String>,
    ) {
        self.cases += 1;
        let c = outcome(cpu);
        let g = self.gpu.map(|ctx| outcome(|| gpu(ctx)));
        let cut = |x: &str| x.chars().take(160).collect::<String>();
        if self.verbose {
            let g = g.as_deref().map_or("-".to_string(), cut);
            println!("{what}\n    CPU {}\n    GPU {g}", cut(&c));
        }
        let panicked = |x: &str| x.starts_with("PANIC");
        self.panics += usize::from(panicked(&c)) + usize::from(g.as_deref().is_some_and(panicked));
        let ok = !panicked(&c)
            && match &g {
                None => true,
                Some(g) if panicked(g) => false,
                Some(g) if g == "Err NotSupported" => {
                    c == *g || GPU_UNSUPPORTED.iter().any(|p| what.starts_with(p))
                }
                Some(g) => c == *g,
            };
        if !ok {
            let g = g.as_deref().map_or("-".to_string(), cut);
            self.failures
                .push(format!("{what}:\n      CPU {}\n      GPU {g}", cut(&c)));
        }
    }

    /// For an operation the CPU backend does not implement: the GPU outcome
    /// must be `want`.
    fn gpu_only(
        &mut self,
        what: String,
        want: &str,
        gpu: impl FnOnce(&GpuContext) -> cv_hal::Result<String>,
    ) {
        let Some(ctx) = self.gpu else { return };
        self.cases += 1;
        let g = outcome(|| gpu(ctx));
        if self.verbose {
            println!("{what}\n    GPU {g}");
        }
        self.panics += usize::from(g.starts_with("PANIC"));
        if g != want {
            self.failures
                .push(format!("{what}:\n      want {want}\n      GPU  {g}"));
        }
    }
}

#[test]
fn empty_tensors_match_the_cpu_on_every_gpu_entry() {
    let cpu = CpuBackend::new().unwrap();
    let mut r = Report::new(gpu());

    // Transfers.
    for s in EMPTY {
        r.case(
            format!("to_gpu + to_cpu {s:?}"),
            || Ok(show(&empty::<f32>(s))),
            |g| show_gpu(g, &empty::<f32>(s).to_gpu_ctx(g)?),
        );
    }

    for s in EMPTY {
        let t = empty::<f32>(s);
        let gt = |g: &GpuContext| t.to_gpu_ctx(g).unwrap();
        let k3 = cpu_t(vec![1.0f32 / 9.0; 9], (1, 3, 3));
        let img = cpu_t(
            (0..64).map(|i| (i % 7) as f32 * 0.25).collect::<Vec<f32>>(),
            (1, 8, 8),
        );

        r.case(
            format!("convolve_2d {s:?}"),
            || Ok(show(&cpu.convolve_2d(&t, &k3, BorderMode::Replicate)?)),
            |g| {
                let out = g.convolve_2d(&gt(g), &k3.to_gpu_ctx(g)?, BorderMode::Replicate)?;
                show_gpu(g, &out)
            },
        );
        r.case(
            format!("convolve_2d kernel {s:?}"),
            || {
                let k = empty::<f32>(s);
                Ok(show(&cpu.convolve_2d(&img, &k, BorderMode::Replicate)?))
            },
            |g| {
                let k = empty::<f32>(s).to_gpu_ctx(g)?;
                show_gpu(
                    g,
                    &g.convolve_2d(&img.to_gpu_ctx(g)?, &k, BorderMode::Replicate)?,
                )
            },
        );
        for typ in [ThresholdType::Binary, ThresholdType::Trunc] {
            r.case(
                format!("threshold {typ:?} {s:?}"),
                || Ok(show(&cpu.threshold(&t, 0.5, 1.0, typ)?)),
                |g| show_gpu(g, &g.threshold(&gt(g), 0.5, 1.0, typ)?),
            );
        }
        r.case(
            format!("sobel {s:?}"),
            || {
                let (a, b) = cpu.sobel(&t, 1, 1, 3)?;
                Ok(format!("{} {}", show(&a), show(&b)))
            },
            |g| {
                let (a, b) = g.sobel(&gt(g), 1, 1, 3)?;
                Ok(format!("{} {}", show_gpu(g, &a)?, show_gpu(g, &b)?))
            },
        );
        r.case(
            format!("canny {s:?}"),
            || Ok(show(&cpu.canny(&t, 0.1, 0.3)?)),
            |g| show_gpu(g, &g.canny(&gt(g), 0.1, 0.3)?),
        );
        r.case(
            format!("hough_lines {s:?}"),
            || Ok(format!("{:?}", cpu.hough_lines(&t, 1.0, 0.1, 1)?)),
            |g| Ok(format!("{:?}", g.hough_lines(&gt(g), 1.0, 0.1, 1)?)),
        );
        r.case(
            format!("hough_circles {s:?}"),
            || Ok(format!("{:?}", cpu.hough_circles(&t, 2.0, 5.0, 1)?)),
            |g| Ok(format!("{:?}", g.hough_circles(&gt(g), 2.0, 5.0, 1)?)),
        );
        let tmpl = cpu_t(vec![0.5f32; 4], (1, 2, 2));
        r.case(
            format!("match_template image {s:?}"),
            || {
                let o: CpuTensor<f32> =
                    cpu.match_template(&t, &tmpl, TemplateMatchMethod::SqDiff)?;
                Ok(show(&o))
            },
            |g| {
                let o: GT<f32> =
                    g.match_template(&gt(g), &tmpl.to_gpu_ctx(g)?, TemplateMatchMethod::SqDiff)?;
                show_gpu(g, &o)
            },
        );
        for method in [
            TemplateMatchMethod::SqDiff,
            TemplateMatchMethod::SqDiffNormed,
            TemplateMatchMethod::Ccorr,
            TemplateMatchMethod::CcorrNormed,
            TemplateMatchMethod::Ccoeff,
            TemplateMatchMethod::CcoeffNormed,
        ] {
            r.case(
                format!("match_template template {method:?} {s:?}"),
                || {
                    let o: CpuTensor<f32> = cpu.match_template(&img, &t, method)?;
                    Ok(show(&o))
                },
                |g| {
                    let o: GT<f32> = g.match_template(&img.to_gpu_ctx(g)?, &gt(g), method)?;
                    show_gpu(g, &o)
                },
            );
        }
        let m = [[1.0f32, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]];
        r.case(
            format!("warp input {s:?}"),
            || Ok(show(&cpu.warp(&t, &m, (4, 4), WarpType::Affine)?)),
            |g| show_gpu(g, &g.warp(&gt(g), &m, (4, 4), WarpType::Affine)?),
        );
        r.case(
            format!("warp to empty from {s:?}"),
            || {
                Ok(show(&cpu.warp(
                    &img,
                    &m,
                    (s.2, s.1),
                    WarpType::Perspective,
                )?))
            },
            |g| {
                let o = g.warp(&img.to_gpu_ctx(g)?, &m, (s.2, s.1), WarpType::Perspective)?;
                show_gpu(g, &o)
            },
        );
        r.case(
            format!("nms {s:?}"),
            || Ok(show(&cpu.nms(&t, 0.1, 3)?)),
            |g| show_gpu(g, &g.nms(&gt(g), 0.1, 3)?),
        );
        r.case(
            format!("cvt_color RgbToGray {s:?}"),
            || Ok(show(&cpu.cvt_color(&t, ColorConversion::RgbToGray)?)),
            |g| show_gpu(g, &g.cvt_color(&gt(g), ColorConversion::RgbToGray)?),
        );
        r.case(
            format!("cvt_color GrayToRgb {s:?}"),
            || Ok(show(&cpu.cvt_color(&t, ColorConversion::GrayToRgb)?)),
            |g| show_gpu(g, &g.cvt_color(&gt(g), ColorConversion::GrayToRgb)?),
        );
        r.case(
            format!("pyramid_down {s:?}"),
            || Ok(show(&cpu.pyramid_down(&t)?)),
            |g| show_gpu(g, &g.pyramid_down(&gt(g))?),
        );
        r.case(
            format!("bilateral_filter {s:?}"),
            || Ok(show(&cpu.bilateral_filter(&t, 5, 0.1, 2.0)?)),
            |g| show_gpu(g, &g.bilateral_filter(&gt(g), 5, 0.1, 2.0)?),
        );
        for nms in [false, true] {
            r.case(
                format!("fast_detect nms {nms} {s:?}"),
                || Ok(show(&cpu.fast_detect(&t, 0.1, nms)?)),
                |g| show_gpu(g, &g.fast_detect(&gt(g), 0.1, nms)?),
            );
        }
        r.case(
            format!("gaussian_blur {s:?}"),
            || Ok(show(&cpu.gaussian_blur(&t, 1.0, 5)?)),
            |g| show_gpu(g, &g.gaussian_blur(&gt(g), 1.0, 5)?),
        );
        r.case(
            format!("subtract {s:?}"),
            || Ok(show(&cpu.subtract(&t, &t)?)),
            |g| show_gpu(g, &g.subtract(&gt(g), &gt(g))?),
        );
        r.case(
            format!("sift_extrema {s:?}"),
            || Ok(show(&cpu.sift_extrema(&t, &t, &t, 0.01, 10.0)?)),
            |g| Ok(show(&g.sift_extrema(&gt(g), &gt(g), &gt(g), 0.01, 10.0)?)),
        );
        let kps = KeyPoints {
            keypoints: vec![KeyPoint::new(1.0, 1.0).with_size(1.6)],
        };
        r.case(
            format!("compute_sift_descriptors {s:?}"),
            || Ok(format!("{:?}", cpu.compute_sift_descriptors(&t, &kps)?)),
            |g| Ok(format!("{:?}", g.compute_sift_descriptors(&gt(g), &kps)?)),
        );
        r.case(
            format!("compute_sift_descriptors no keypoints, image {s:?}"),
            || {
                Ok(format!(
                    "{:?}",
                    cpu.compute_sift_descriptors(&t, &KeyPoints::default())?
                ))
            },
            |g| {
                Ok(format!(
                    "{:?}",
                    g.compute_sift_descriptors(&gt(g), &KeyPoints::default())?
                ))
            },
        );
        r.case(
            format!("akaze_diffusion {s:?}"),
            || {
                Ok(show(&cpu.akaze_diffusion(
                    &t,
                    0.1,
                    0.25,
                    DiffusivityType::PeronaMalik2,
                )?))
            },
            |g| {
                let o = g.akaze_diffusion(&gt(g), 0.1, 0.25, DiffusivityType::PeronaMalik2)?;
                show_gpu(g, &o)
            },
        );
        r.case(
            format!("akaze_derivatives {s:?}"),
            || {
                let (a, b, c) = cpu.akaze_derivatives(&t)?;
                Ok(format!("{} {} {}", show(&a), show(&b), show(&c)))
            },
            |g| {
                let (a, b, c) = g.akaze_derivatives(&gt(g))?;
                Ok(format!(
                    "{} {} {}",
                    show_gpu(g, &a)?,
                    show_gpu(g, &b)?,
                    show_gpu(g, &c)?
                ))
            },
        );
        r.case(
            format!("akaze_contrast_k {s:?}"),
            || Ok(format!("{:?}", cpu.akaze_contrast_k(&t)?)),
            |g| Ok(format!("{:?}", g.akaze_contrast_k(&gt(g))?)),
        );
        let sp = StereoMatchParams {
            method: StereoMatchMethod::BlockMatching,
            min_disparity: 0,
            num_disparities: 16,
            block_size: 5,
        };
        r.case(
            format!("stereo_match {s:?}"),
            || {
                let o: CpuTensor<f32> = cpu.stereo_match(&t, &t, &sp)?;
                Ok(show(&o))
            },
            |g| {
                let o: GT<f32> = g.stereo_match(&gt(g), &gt(g), &sp)?;
                show_gpu(g, &o)
            },
        );
        r.case(
            format!("find_chessboard_corners {s:?}"),
            || Ok(format!("{:?}", cpu.find_chessboard_corners(&t, (3, 3))?)),
            |g| Ok(format!("{:?}", g.find_chessboard_corners(&gt(g), (3, 3))?)),
        );
        r.case(
            format!("detect_objects {s:?}"),
            || Ok(format!("{:?}", cpu.detect_objects(&t, 0.5)?)),
            |g| Ok(format!("{:?}", g.detect_objects(&gt(g), 0.5)?)),
        );
        // remap and undistort are GPU-only (the CPU backend returns
        // NotSupported), so the expected outcomes are spelled out: an empty
        // output is an empty tensor, and an empty image with a non-empty
        // output is InvalidInput, there being no pixel to sample. The GPU
        // takes single-channel images only.
        let single = s.0 == 1;
        let empty_out =
            |c: usize, h: usize, w: usize| format!("Ok {:?} []", TensorShape::new(c, h, w));
        let unsupported = "Err NotSupported".to_string();
        if s.1 * s.2 == 0 {
            let map: CpuTensor<f32> = empty((1, s.1, s.2));
            r.gpu_only(
                format!("remap empty map {s:?}"),
                &empty_out(1, s.1, s.2),
                |g| {
                    let gm = map.to_gpu_ctx(g)?;
                    let o = g.remap(
                        &img.to_gpu_ctx(g)?,
                        &gm,
                        &gm,
                        Interpolation::Linear,
                        BorderMode::Constant(7.0),
                    )?;
                    show_gpu(g, &o)
                },
            );
        }
        let map4: CpuTensor<f32> = cpu_t(vec![0.5; 16], (1, 4, 4));
        for interp in [Interpolation::Nearest, Interpolation::Linear] {
            r.gpu_only(
                format!("remap empty input {interp:?} {s:?}"),
                if single {
                    "Err InvalidInput"
                } else {
                    &unsupported
                },
                |g| {
                    let gm = map4.to_gpu_ctx(g)?;
                    let o = g.remap(&gt(g), &gm, &gm, interp, BorderMode::Constant(7.0))?;
                    show_gpu(g, &o)
                },
            );
        }
        let dist = cv_core::Distortion::new(0.1, 0.0, 0.0, 0.0, 0.0);
        let rect = nalgebra::Matrix3::<f64>::identity();
        for (dw, dh) in [(s.2, s.1), (4, 4), (0, 4)] {
            let intr = cv_core::CameraIntrinsics::new(4.0, 4.0, 2.0, 2.0, dw as u32, dh as u32);
            let want = if !single {
                unsupported.clone()
            } else if dw * dh == 0 {
                empty_out(1, dh, dw)
            } else {
                "Err InvalidInput".to_string()
            };
            r.gpu_only(format!("undistort {s:?} to {dw}x{dh}"), &want, |g| {
                let o = g.undistort(
                    &gt(g),
                    &intr,
                    &dist,
                    &rect,
                    &intr,
                    Interpolation::Linear,
                    BorderMode::Constant(0.0),
                )?;
                show_gpu(g, &o)
            });
        }
        // tsdf_raycast to an empty image.
        let pose = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let vol = cpu_t(vec![1.0f32; 8 * 8 * 16], (16, 8, 8));
        r.case(
            format!("tsdf_raycast to {}x{}", s.2, s.1),
            || {
                Ok(show(&cpu.tsdf_raycast(
                    &vol,
                    &pose,
                    &[4.0, 4.0, 2.0, 2.0],
                    (s.2 as u32, s.1 as u32),
                    (0.1, 2.0),
                    0.05,
                    0.1,
                )?))
            },
            |g| {
                let o = g.tsdf_raycast(
                    &vol.to_gpu_ctx(g)?,
                    &pose,
                    &[4.0, 4.0, 2.0, 2.0],
                    (s.2 as u32, s.1 as u32),
                    (0.1, 2.0),
                    0.05,
                    0.1,
                )?;
                show_gpu(g, &o)
            },
        );
        r.case(
            format!("tsdf_raycast empty volume {s:?}"),
            || {
                Ok(show(&cpu.tsdf_raycast(
                    &t,
                    &pose,
                    &[4.0, 4.0, 2.0, 2.0],
                    (4, 4),
                    (0.1, 2.0),
                    0.05,
                    0.1,
                )?))
            },
            |g| {
                let o = g.tsdf_raycast(
                    &gt(g),
                    &pose,
                    &[4.0, 4.0, 2.0, 2.0],
                    (4, 4),
                    (0.1, 2.0),
                    0.05,
                    0.1,
                )?;
                show_gpu(g, &o)
            },
        );
        r.case(
            format!("tsdf_extract_mesh {s:?}"),
            || {
                Ok(format!(
                    "{:?}",
                    cpu.tsdf_extract_mesh(&t, 0.05, 0.0, 100)?.len()
                ))
            },
            |g| {
                Ok(format!(
                    "{:?}",
                    g.tsdf_extract_mesh(&gt(g), 0.05, 0.0, 100)?.len()
                ))
            },
        );
        r.case(
            format!("tsdf_integrate empty depth {s:?}"),
            || {
                let mut v = vol.clone();
                cpu.tsdf_integrate(&t, &pose, &[4.0, 4.0, 2.0, 2.0], &mut v, 0.05, 0.1)?;
                Ok(show(&v))
            },
            |g| {
                let mut v = vol.to_gpu_ctx(g)?;
                g.tsdf_integrate(&gt(g), &pose, &[4.0, 4.0, 2.0, 2.0], &mut v, 0.05, 0.1)?;
                show_gpu(g, &v)
            },
        );
        r.case(
            format!("tsdf_integrate empty volume {s:?}"),
            || {
                let mut v = t.clone();
                cpu.tsdf_integrate(&img, &pose, &[4.0, 4.0, 2.0, 2.0], &mut v, 0.05, 0.1)?;
                Ok(show(&v))
            },
            |g| {
                let mut v = gt(g);
                g.tsdf_integrate(
                    &img.to_gpu_ctx(g)?,
                    &pose,
                    &[4.0, 4.0, 2.0, 2.0],
                    &mut v,
                    0.05,
                    0.1,
                )?;
                show_gpu(g, &v)
            },
        );
        r.case(
            format!("optical_flow_lk empty images {s:?}"),
            || {
                Ok(format!(
                    "{:?}",
                    cpu.optical_flow_lk(&[t.clone()], &[t.clone()], &[[1.0, 1.0]], 5, 10)?
                ))
            },
            |g| {
                Ok(format!(
                    "{:?}",
                    g.optical_flow_lk(&[gt(g)], &[gt(g)], &[[1.0, 1.0]], 5, 10)?
                ))
            },
        );
        let mp = Mog2Params::<f32> {
            width: s.2 as u32,
            height: s.1 as u32,
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
        r.case(
            format!("mog2_update {s:?}"),
            || {
                let mut model = empty::<f32>(s);
                let mut mask: CpuTensor<u32> = empty(s);
                cpu.mog2_update(&t, &mut model, &mut mask, &mp)?;
                Ok(format!("{} {}", show(&model), show(&mask)))
            },
            |g| {
                let mut model = gt(g);
                let mut mask: GT<u32> = empty::<u32>(s).to_gpu_ctx(g)?;
                g.mog2_update(&gt(g), &mut model, &mut mask, &mp)?;
                Ok(format!("{} {}", show_gpu(g, &model)?, show_gpu(g, &mask)?))
            },
        );

        r.case(
            format!("resize {s:?} to 4x4"),
            || Ok(show(&cpu.resize(&t, (4, 4))?)),
            |g| show_gpu(g, &g.resize(&gt(g), (4, 4))?),
        );
        r.case(
            format!("resize 8x8 to {}x{}", s.2, s.1),
            || Ok(show(&cpu.resize(&img, (s.2, s.1))?)),
            |g| show_gpu(g, &g.resize(&img.to_gpu_ctx(g)?, (s.2, s.1))?),
        );

        // u8 operations.
        let t8 = empty::<u8>(s);
        let k8 = cpu_t(vec![1u8; 9], (1, 3, 3));
        for typ in [MorphologyType::Erode, MorphologyType::Close] {
            r.case(
                format!("morphology {typ:?} {s:?}"),
                || Ok(show(&cpu.morphology(&t8, typ, &k8, 1)?)),
                |g| {
                    let o = g.morphology(&t8.to_gpu_ctx(g)?, typ, &k8.to_gpu_ctx(g)?, 1)?;
                    show_gpu(g, &o)
                },
            );
        }
    }

    // Point and descriptor sets with no rows.
    let pts0 = || empty::<f32>((1, 0, 3));
    let pts4 = cpu_t(
        vec![
            0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        (1, 4, 3),
    );
    let pose = [
        [1.0f32, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    r.case(
        "pointcloud_transform no points".into(),
        || Ok(show(&cpu.pointcloud_transform(&pts0(), &pose)?)),
        |g| show_gpu(g, &g.pointcloud_transform(&pts0().to_gpu_ctx(g)?, &pose)?),
    );
    r.case(
        "pointcloud_normals no points".into(),
        || Ok(show(&cpu.pointcloud_normals(&pts0(), 8)?)),
        |g| show_gpu(g, &g.pointcloud_normals(&pts0().to_gpu_ctx(g)?, 8)?),
    );
    for (what, src, tgt) in [
        ("no source", pts0(), pts4.clone()),
        ("no target", pts4.clone(), pts0()),
        ("neither", pts0(), pts0()),
    ] {
        r.case(
            format!("icp_correspondences {what}"),
            || Ok(format!("{:?}", cpu.icp_correspondences(&src, &tgt, 1.0)?)),
            |g| {
                let o = g.icp_correspondences(&src.to_gpu_ctx(g)?, &tgt.to_gpu_ctx(g)?, 1.0)?;
                Ok(format!("{o:?}"))
            },
        );
    }
    let ident = nalgebra::Matrix4::<f32>::identity();
    r.case(
        "icp_accumulate no points".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.icp_accumulate(&pts0(), &pts0(), &pts0(), &[], &ident)?
            ))
        },
        |g| {
            let p = pts0().to_gpu_ctx(g)?;
            Ok(format!("{:?}", g.icp_accumulate(&p, &p, &p, &[], &ident)?))
        },
    );
    r.case(
        "icp_accumulate no correspondences".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.icp_accumulate(&pts4, &pts4, &pts4, &[], &ident)?
            ))
        },
        |g| {
            let p = pts4.to_gpu_ctx(g)?;
            Ok(format!("{:?}", g.icp_accumulate(&p, &p, &p, &[], &ident)?))
        },
    );
    for s in EMPTY {
        let d = empty::<f32>(s);
        r.case(
            format!("dense_icp_step {s:?}"),
            || {
                Ok(format!(
                    "{:?}",
                    cpu.dense_icp_step(&d, &d, &[4.0, 4.0, 2.0, 2.0], &ident, 0.1, 0.5)?
                ))
            },
            |g| {
                let gd = d.to_gpu_ctx(g)?;
                Ok(format!(
                    "{:?}",
                    g.dense_icp_step(&gd, &gd, &[4.0, 4.0, 2.0, 2.0], &ident, 0.1, 0.5)?
                ))
            },
        );
    }
    let boxes0 = || empty::<f32>((1, 0, 5));
    r.case(
        "nms_boxes no boxes".into(),
        || Ok(format!("{:?}", cpu.nms_boxes(&boxes0(), 0.5)?)),
        |g| Ok(format!("{:?}", g.nms_boxes(&boxes0().to_gpu_ctx(g)?, 0.5)?)),
    );
    let rboxes0 = || empty::<f32>((1, 0, 6));
    r.case(
        "nms_rotated_boxes no boxes".into(),
        || Ok(format!("{:?}", cpu.nms_rotated_boxes(&rboxes0(), 0.5)?)),
        |g| {
            Ok(format!(
                "{:?}",
                g.nms_rotated_boxes(&rboxes0().to_gpu_ctx(g)?, 0.5)?
            ))
        },
    );
    r.case(
        "nms_polygons no polygons".into(),
        || Ok(format!("{:?}", cpu.nms_polygons::<f32>(&[], &[], 0.5)?)),
        |g| Ok(format!("{:?}", g.nms_polygons::<f32>(&[], &[], 0.5)?)),
    );
    let pts2 = || empty::<f32>((1, 0, 2));
    let proj = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    r.case(
        "triangulate_points no points".into(),
        || {
            let o: CpuTensor<f32> = cpu.triangulate_points(&proj, &proj, &pts2(), &pts2())?;
            Ok(show(&o))
        },
        |g| {
            let p = pts2().to_gpu_ctx(g)?;
            let o: GT<f32> = g.triangulate_points(&proj, &proj, &p, &p)?;
            show_gpu(g, &o)
        },
    );
    r.case(
        "optical_flow_lk no points".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.optical_flow_lk(&[pts4.clone()], &[pts4.clone()], &[], 5, 10)?
            ))
        },
        |g| {
            let p = pts4.to_gpu_ctx(g)?;
            Ok(format!(
                "{:?}",
                g.optical_flow_lk(&[p.clone()], &[p], &[], 5, 10)?
            ))
        },
    );
    r.case(
        "spmv empty x".into(),
        || {
            let x = empty::<f32>((1, 1, 0));
            Ok(show(&cpu.spmv(&[0u32], &[], &[], &x)?))
        },
        |g| {
            let x = empty::<f32>((1, 1, 0)).to_gpu_ctx(g)?;
            show_gpu(g, &g.spmv(&[0u32], &[], &[], &x)?)
        },
    );
    r.case(
        "spmv rows without entries".into(),
        || {
            let x = cpu_t(vec![1.0f32; 3], (1, 1, 3));
            Ok(show(&cpu.spmv(&[0u32, 0, 0], &[], &[], &x)?))
        },
        |g| {
            let x = cpu_t(vec![1.0f32; 3], (1, 1, 3)).to_gpu_ctx(g)?;
            show_gpu(g, &g.spmv(&[0u32, 0, 0], &[], &[], &x)?)
        },
    );
    for (q, t) in [
        ((1, 0, 32), (1, 4, 32)),
        ((1, 4, 32), (1, 0, 32)),
        ((1, 0, 32), (1, 0, 32)),
    ] {
        let qd = if q.1 == 0 {
            empty::<u8>(q)
        } else {
            cpu_t(vec![7u8; 128], q)
        };
        let td = if t.1 == 0 {
            empty::<u8>(t)
        } else {
            cpu_t(vec![9u8; 128], t)
        };
        r.case(
            format!("match_descriptors {q:?} vs {t:?}"),
            || {
                Ok(format!(
                    "{:?}",
                    cpu.match_descriptors(&qd, &td, 0.8)?.matches
                ))
            },
            |g| {
                let m = g.match_descriptors(&qd.to_gpu_ctx(g)?, &td.to_gpu_ctx(g)?, 0.8)?;
                Ok(format!("{:?}", m.matches))
            },
        );
    }

    // Sets whose rows are there but whose channel count is zero: no data,
    // although the height says otherwise. Code that counted rows by height
    // indexed the empty storage.
    let none = |w: usize| empty::<f32>((0, 4, w));
    r.case(
        "pointcloud_transform no channels".into(),
        || Ok(show(&cpu.pointcloud_transform(&none(4), &pose)?)),
        |g| show_gpu(g, &g.pointcloud_transform(&none(4).to_gpu_ctx(g)?, &pose)?),
    );
    r.case(
        "pointcloud_normals no channels".into(),
        || Ok(show(&cpu.pointcloud_normals(&none(4), 8)?)),
        |g| show_gpu(g, &g.pointcloud_normals(&none(4).to_gpu_ctx(g)?, 8)?),
    );
    r.case(
        "icp_correspondences no channels".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.icp_correspondences(&none(4), &pts4, 1.0)?
            ))
        },
        |g| {
            let o = g.icp_correspondences(&none(4).to_gpu_ctx(g)?, &pts4.to_gpu_ctx(g)?, 1.0)?;
            Ok(format!("{o:?}"))
        },
    );
    r.case(
        "icp_accumulate correspondences into empty clouds".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.icp_accumulate(&none(4), &none(4), &none(4), &[(0, 0), (3, 1)], &ident)?
            ))
        },
        |g| {
            let p = none(4).to_gpu_ctx(g)?;
            Ok(format!(
                "{:?}",
                g.icp_accumulate(&p, &p, &p, &[(0, 0), (3, 1)], &ident)?
            ))
        },
    );
    r.case(
        "icp_accumulate no normals".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.icp_accumulate(&pts4, &pts4, &pts0(), &[(0, 0), (1, 1)], &ident)?
            ))
        },
        |g| {
            let p = pts4.to_gpu_ctx(g)?;
            let n = pts0().to_gpu_ctx(g)?;
            Ok(format!(
                "{:?}",
                g.icp_accumulate(&p, &p, &n, &[(0, 0), (1, 1)], &ident)?
            ))
        },
    );
    r.case(
        "dense_icp_step empty target".into(),
        || {
            let d = cpu_t(vec![1.0f32; 16], (1, 4, 4));
            let e = empty::<f32>((7, 0, 4));
            Ok(format!(
                "{:?}",
                cpu.dense_icp_step(&d, &e, &[4.0, 4.0, 2.0, 2.0], &ident, 0.1, 0.5)?
            ))
        },
        |g| {
            let d = cpu_t(vec![1.0f32; 16], (1, 4, 4)).to_gpu_ctx(g)?;
            let e = empty::<f32>((7, 0, 4)).to_gpu_ctx(g)?;
            Ok(format!(
                "{:?}",
                g.dense_icp_step(&d, &e, &[4.0, 4.0, 2.0, 2.0], &ident, 0.1, 0.5)?
            ))
        },
    );
    r.case(
        "nms_boxes no channels".into(),
        || Ok(format!("{:?}", cpu.nms_boxes(&none(5), 0.5)?)),
        |g| Ok(format!("{:?}", g.nms_boxes(&none(5).to_gpu_ctx(g)?, 0.5)?)),
    );
    r.case(
        "nms_rotated_boxes no channels".into(),
        || Ok(format!("{:?}", cpu.nms_rotated_boxes(&none(6), 0.5)?)),
        |g| {
            Ok(format!(
                "{:?}",
                g.nms_rotated_boxes(&none(6).to_gpu_ctx(g)?, 0.5)?
            ))
        },
    );
    let pointless = vec![cv_core::Polygon::new(Vec::new()); 2];
    r.case(
        "nms_polygons polygons without points".into(),
        || {
            Ok(format!(
                "{:?}",
                cpu.nms_polygons(&pointless, &[0.9f32, 0.8], 0.5)?
            ))
        },
        |g| {
            Ok(format!(
                "{:?}",
                g.nms_polygons(&pointless, &[0.9f32, 0.8], 0.5)?
            ))
        },
    );
    r.case(
        "triangulate_points no channels".into(),
        || {
            let o: CpuTensor<f32> = cpu.triangulate_points(&proj, &proj, &none(2), &none(2))?;
            Ok(show(&o))
        },
        |g| {
            let p = none(2).to_gpu_ctx(g)?;
            let o: GT<f32> = g.triangulate_points(&proj, &proj, &p, &p)?;
            show_gpu(g, &o)
        },
    );
    let some = cpu_t(vec![5u8; 4 * 32], (1, 4, 32));
    let no_bytes = empty::<u8>((0, 4, 32));
    for (what, q, t) in [("query", &no_bytes, &some), ("train", &some, &no_bytes)] {
        r.case(
            format!("match_descriptors no channels in the {what} set"),
            || Ok(format!("{:?}", cpu.match_descriptors(q, t, 0.8)?.matches)),
            |g| {
                let m = g.match_descriptors(&q.to_gpu_ctx(g)?, &t.to_gpu_ctx(g)?, 0.8)?;
                Ok(format!("{:?}", m.matches))
            },
        );
    }
    // Second operands that are empty while the first is not.
    let img8 = cpu_t(vec![9u8; 64], (1, 8, 8));
    for s in EMPTY {
        r.case(
            format!("morphology Dilate kernel {s:?}"),
            || {
                Ok(show(&cpu.morphology(
                    &img8,
                    MorphologyType::Dilate,
                    &empty(s),
                    1,
                )?))
            },
            |g| {
                let o = g.morphology(
                    &img8.to_gpu_ctx(g)?,
                    MorphologyType::Dilate,
                    &empty::<u8>(s).to_gpu_ctx(g)?,
                    1,
                )?;
                show_gpu(g, &o)
            },
        );
        let a = cpu_t(vec![1.0f32; 64], (1, 8, 8));
        r.case(
            format!("subtract non-empty minus {s:?}"),
            || Ok(show(&cpu.subtract(&a, &empty(s))?)),
            |g| {
                show_gpu(
                    g,
                    &g.subtract(&a.to_gpu_ctx(g)?, &empty::<f32>(s).to_gpu_ctx(g)?)?,
                )
            },
        );
    }

    println!(
        "{} empty-input cases, {} panics, {} failures",
        r.cases,
        r.panics,
        r.failures.len()
    );
    assert!(
        r.failures.is_empty(),
        "{} of {} empty-input cases panic or differ between the backends:\n  {}",
        r.failures.len(),
        r.cases,
        r.failures.join("\n  ")
    );
}
