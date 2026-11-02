//! `match_template` on multi-channel images, and OpenCV's normalisation of
//! the `*_NORMED` methods, on the CPU and GPU backends, against
//! `cv2.matchTemplate` (OpenCV 4.13).
//!
//! `cv::matchTemplate` sums every term over the channels: the correlation,
//! the window's `sum(I^2)`, and for the coefficient methods the per-channel
//! centring. The CPU backend read plane 0 only; the GPU kernel did too, and
//! computed `SqDiff` and `Ccorr` only - every other method returned zeros.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, TemplateMatchMethod};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::GpuStorage;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

include!("fixtures/match_template_3ch_cv2.rs");

const METHODS: [TemplateMatchMethod; 6] = [
    TemplateMatchMethod::SqDiff,
    TemplateMatchMethod::SqDiffNormed,
    TemplateMatchMethod::Ccorr,
    TemplateMatchMethod::CcorrNormed,
    TemplateMatchMethod::Ccoeff,
    TemplateMatchMethod::CcoeffNormed,
];

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("no GPU adapter: checking the CPU backend only");
            None
        }
    }
}

fn img_v(c: usize, y: usize, x: usize) -> f32 {
    ((c * 83 + y * 37 + x * 59 + (x * y * 13) % 31 + c * x * 7) % 256) as f32
}

fn tmpl_v(c: usize, y: usize, x: usize) -> f32 {
    ((c * 47 + y * 91 + x * 23 + (x + 2 * y + c) % 5 * 17) % 256) as f32
}

fn planar(c: usize, h: usize, w: usize, f: impl Fn(usize, usize, usize) -> f32) -> CpuTensor<f32> {
    let mut v = Vec::with_capacity(c * h * w);
    for k in 0..c {
        for y in 0..h {
            for x in 0..w {
                v.push(f(k, y, x));
            }
        }
    }
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

fn cpu_match(
    img: &CpuTensor<f32>,
    t: &CpuTensor<f32>,
    m: TemplateMatchMethod,
) -> cv_hal::Result<CpuTensor<f32>> {
    CpuBackend::new().unwrap().match_template(img, t, m)
}

fn gpu_match(
    g: &GpuContext,
    img: &CpuTensor<f32>,
    t: &CpuTensor<f32>,
    m: TemplateMatchMethod,
) -> cv_hal::Result<CpuTensor<f32>> {
    let o: Tensor<f32, GpuStorage<f32>> =
        g.match_template(&img.to_gpu_ctx(g)?, &t.to_gpu_ctx(g)?, m)?;
    o.to_cpu_ctx(g)
}

/// cv2 computes the float32 correlation by DFT, so its raw sums carry f32
/// rounding of the ~1e6 totals; the GPU accumulates in f32 too.
fn tolerance(m: TemplateMatchMethod) -> f64 {
    match m {
        TemplateMatchMethod::SqDiff | TemplateMatchMethod::Ccorr => 0.5,
        TemplateMatchMethod::Ccoeff => 0.5,
        _ => 2e-6,
    }
}

fn check(what: &str, got: &CpuTensor<f32>, want: &[f64], tol: f64, failures: &mut Vec<String>) {
    if got.shape != TensorShape::new(1, 5, 6) {
        failures.push(format!("{what}: shape {:?}", got.shape));
        return;
    }
    let got = got.storage.as_slice().unwrap();
    let (i, worst) = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w).abs())
        .enumerate()
        .fold((0, 0.0f64), |a, (i, d)| if d > a.1 { (i, d) } else { a });
    if !(worst <= tol) {
        failures.push(format!(
            "{what}: max |diff| {worst} at #{i} (got {}, cv2 {})",
            got[i], want[i]
        ));
    }
}

#[test]
fn three_channel_match_template_is_cv2s() {
    let img = planar(3, 7, 9, img_v);
    let t = planar(3, 3, 4, tmpl_v);
    let g = gpu();
    let mut failures = Vec::new();
    for (m, want) in METHODS.iter().zip(&CV2_3CH) {
        match cpu_match(&img, &t, *m) {
            Ok(o) => check(
                &format!("CPU {m:?}"),
                &o,
                want,
                tolerance(*m),
                &mut failures,
            ),
            Err(e) => failures.push(format!("CPU {m:?}: {e:?}")),
        }
        if let Some(g) = g {
            // The GPU accumulates in f32: allow a few f32 ulps of the totals.
            let tol = match m {
                TemplateMatchMethod::SqDiff
                | TemplateMatchMethod::Ccorr
                | TemplateMatchMethod::Ccoeff => 1.0,
                _ => 2e-6,
            };
            match gpu_match(g, &img, &t, *m) {
                Ok(o) => check(&format!("GPU {m:?}"), &o, want, tol, &mut failures),
                Err(e) => failures.push(format!("GPU {m:?}: {e:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The degenerate windows of the normalised methods, single channel, with
/// the values `cv2.matchTemplate` returns for them.
#[test]
fn normalised_methods_follow_cv2_in_degenerate_windows() {
    let g = gpu();
    let mut failures = Vec::new();
    let mut expect = |what: &str,
                      img: &CpuTensor<f32>,
                      t: &CpuTensor<f32>,
                      m: TemplateMatchMethod,
                      first: &[f32]| {
        let mut outs = vec![("CPU", cpu_match(img, t, m))];
        if let Some(g) = g {
            outs.push(("GPU", gpu_match(g, img, t, m)));
        }
        for (backend, o) in outs {
            match o {
                Ok(o) => {
                    let got = &o.storage.as_slice().unwrap()[..first.len()];
                    if got.iter().zip(first).any(|(a, b)| (a - b).abs() > 1e-6) {
                        failures.push(format!("{backend} {what} {m:?}: {got:?}, cv2 {first:?}"));
                    }
                }
                Err(e) => failures.push(format!("{backend} {what} {m:?}: {e:?}")),
            }
        }
    };

    // A constant template has no variance: `CcoeffNormed` is 1 everywhere
    // (OpenCV: `templNorm < DBL_EPSILON`).
    let ramp = planar(1, 5, 6, |_, y, x| (y * 6 + x) as f32);
    let flat = planar(1, 2, 2, |_, _, _| 7.0);
    expect(
        "constant template",
        &ramp,
        &flat,
        TemplateMatchMethod::CcoeffNormed,
        &[1.0, 1.0, 1.0],
    );
    // A zero window: `SqDiffNormed` is 1, the correlations 0.
    let dark = planar(1, 5, 6, |_, y, x| if y >= 3 && x >= 3 { 5.0 } else { 0.0 });
    let t = planar(1, 2, 2, |_, y, x| (1 + 2 * y + x) as f32);
    expect(
        "zero window",
        &dark,
        &t,
        TemplateMatchMethod::SqDiffNormed,
        &[1.0, 1.0, 1.0],
    );
    expect(
        "zero window",
        &dark,
        &t,
        TemplateMatchMethod::CcorrNormed,
        &[0.0, 0.0, 0.0],
    );
    expect(
        "zero window",
        &dark,
        &t,
        TemplateMatchMethod::CcoeffNormed,
        &[0.0, 0.0, 0.0],
    );
    // A bright window and a dim template: the squared difference is far
    // above the norm product, and OpenCV reports 1, not 39801 / 300.
    let bright = planar(1, 3, 3, |_, _, _| 100.0);
    let dim = planar(1, 2, 2, |_, y, x| if y == 0 && x == 0 { 1.0 } else { 0.0 });
    expect(
        "bright window",
        &bright,
        &dim,
        TemplateMatchMethod::SqDiffNormed,
        &[1.0, 1.0, 1.0],
    );
    expect(
        "bright window",
        &bright,
        &dim,
        TemplateMatchMethod::CcorrNormed,
        &[0.5, 0.5, 0.5],
    );
    expect(
        "bright window",
        &bright,
        &dim,
        TemplateMatchMethod::CcoeffNormed,
        &[0.0, 0.0, 0.0],
    );
    // Every channel of a 3-channel template constant: `CcoeffNormed` 1,
    // `SqDiffNormed` 1 (far above the norm product), as cv2.
    let im3 = planar(3, 6, 7, img_v);
    let t3 = planar(3, 2, 3, |_, _, _| 9.0);
    expect(
        "3ch constant template",
        &im3,
        &t3,
        TemplateMatchMethod::CcoeffNormed,
        &[1.0, 1.0, 1.0],
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `cv::matchTemplate` requires the template's type, channel count
/// included, to be the image's.
#[test]
fn channel_count_mismatch_is_invalid_input() {
    let img = planar(3, 7, 9, img_v);
    let t = planar(1, 3, 4, |_, y, x| tmpl_v(0, y, x));
    for m in METHODS {
        let e = cpu_match(&img, &t, m).unwrap_err();
        assert!(
            matches!(e, cv_hal::Error::InvalidInput(_)),
            "CPU {m:?}: {e:?}"
        );
        if let Some(g) = gpu() {
            let e = gpu_match(g, &img, &t, m).unwrap_err();
            assert!(
                matches!(e, cv_hal::Error::InvalidInput(_)),
                "GPU {m:?}: {e:?}"
            );
            let e = gpu_match(g, &t, &planar(3, 2, 2, |_, _, _| 1.0), m).unwrap_err();
            assert!(
                matches!(e, cv_hal::Error::InvalidInput(_)),
                "GPU {m:?}: {e:?}"
            );
        }
    }
}

/// A larger single-channel case on both backends against each other: the
/// GPU kernel used to return zeros for every method but `SqDiff` / `Ccorr`.
#[test]
fn gpu_agrees_with_cpu_on_every_method() {
    let Some(g) = gpu() else { return };
    let img = planar(3, 40, 52, |c, y, x| {
        ((c * 31 + y * y * 7 + x * 13 + (x * y) % 17) % 256) as f32
    });
    // Cut from the image, so an exact match exists at (11, 9).
    let t = planar(3, 9, 12, |c, y, x| {
        ((c * 31 + (y + 9) * (y + 9) * 7 + (x + 11) * 13 + ((x + 11) * (y + 9)) % 17) % 256) as f32
    });
    for channels in [1, 3] {
        let (img, t) = if channels == 1 {
            let p = |s: &CpuTensor<f32>| {
                let (h, w) = s.shape.hw();
                Tensor::from_vec(
                    s.storage.as_slice().unwrap()[..h * w].to_vec(),
                    TensorShape::new(1, h, w),
                )
                .unwrap()
            };
            (p(&img), p(&t))
        } else {
            (img.clone(), t.clone())
        };
        for m in METHODS {
            let c = cpu_match(&img, &t, m).unwrap();
            let gg = gpu_match(g, &img, &t, m).unwrap();
            assert_eq!(c.shape, gg.shape);
            let (c, gg) = (
                c.storage.as_slice().unwrap(),
                gg.storage.as_slice().unwrap(),
            );
            let scale = c.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1.0);
            let normed = !matches!(
                m,
                TemplateMatchMethod::SqDiff
                    | TemplateMatchMethod::Ccorr
                    | TemplateMatchMethod::Ccoeff
            );
            let tol = if normed { 1e-5 } else { scale * 1e-6 };
            for (i, (a, b)) in c.iter().zip(gg).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{channels}ch {m:?} #{i}: CPU {a}, GPU {b}"
                );
            }
            let at = 9 * (52 - 12 + 1) + 11;
            match m {
                TemplateMatchMethod::SqDiff | TemplateMatchMethod::SqDiffNormed => {
                    assert_eq!(gg[at], 0.0, "{channels}ch {m:?} exact match")
                }
                TemplateMatchMethod::CcorrNormed | TemplateMatchMethod::CcoeffNormed => {
                    assert!(
                        (gg[at] - 1.0).abs() < 1e-6,
                        "{channels}ch {m:?} exact match {}",
                        gg[at]
                    )
                }
                _ => {}
            }
        }
    }
}
