//! `CpuBackend::match_template` now accumulates in f64 with integral-image
//! window sums. Checked against
//! * `definition`: the six `cv::TemplateMatchModes` written out directly
//!   (two-pass means, f64) - must agree to f64 rounding;
//! * `old_f32`: a transcription of the previous f32 brute-force kernel -
//!   must agree within that kernel's own f32 accumulation error;
//! * the exact-match property `CcoeffNormed == 1` (it used to exceed 1).

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{ComputeContext, TemplateMatchMethod};
use cv_hal::cpu::CpuBackend;
use std::time::Instant;

const METHODS: [TemplateMatchMethod; 6] = [
    TemplateMatchMethod::SqDiff,
    TemplateMatchMethod::SqDiffNormed,
    TemplateMatchMethod::Ccorr,
    TemplateMatchMethod::CcorrNormed,
    TemplateMatchMethod::Ccoeff,
    TemplateMatchMethod::CcoeffNormed,
];

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) % 256) as f32
        })
        .collect()
}

fn tensor(d: Vec<f32>, w: usize, h: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(d, TensorShape::new(1, h, w)).unwrap()
}

fn run(
    img: &[f32],
    iw: usize,
    ih: usize,
    t: &[f32],
    tw: usize,
    th: usize,
    m: TemplateMatchMethod,
) -> Vec<f32> {
    let out: Tensor<f32, CpuStorage<f32>> = CpuBackend::new()
        .unwrap()
        .match_template(
            &tensor(img.to_vec(), iw, ih),
            &tensor(t.to_vec(), tw, th),
            m,
        )
        .unwrap();
    out.as_slice().unwrap().to_vec()
}

/// The modes from their definitions, f64, two-pass means.
fn definition(
    img: &[f32],
    iw: usize,
    ih: usize,
    t: &[f32],
    tw: usize,
    th: usize,
    m: TemplateMatchMethod,
) -> Vec<f64> {
    let n = (tw * th) as f64;
    let tm = t.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut out = Vec::new();
    for y in 0..=ih - th {
        for x in 0..=iw - tw {
            let px = |i: usize| img[(y + i / tw) * iw + x + i % tw] as f64;
            let im = (0..tw * th).map(px).sum::<f64>() / n;
            let (mut sd, mut cc, mut co, mut ii, mut tt, mut iic, mut ttc) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for i in 0..tw * th {
                let (a, b) = (px(i), t[i] as f64);
                sd += (a - b) * (a - b);
                cc += a * b;
                co += (a - im) * (b - tm);
                ii += a * a;
                tt += b * b;
                iic += (a - im) * (a - im);
                ttc += (b - tm) * (b - tm);
            }
            let norm = |num: f64, den: f64| {
                if den > f32::EPSILON as f64 {
                    num / den
                } else {
                    0.0
                }
            };
            out.push(match m {
                TemplateMatchMethod::SqDiff => sd,
                TemplateMatchMethod::SqDiffNormed => norm(sd, (ii * tt).sqrt()),
                TemplateMatchMethod::Ccorr => cc,
                TemplateMatchMethod::CcorrNormed => norm(cc, (ii * tt).sqrt()),
                TemplateMatchMethod::Ccoeff => co,
                TemplateMatchMethod::CcoeffNormed => norm(co, iic.sqrt() * ttc.sqrt()),
            });
        }
    }
    out
}

/// The previous f32 kernel (single-precision accumulation, two passes for
/// the coefficient methods).
fn old_f32(
    img: &[f32],
    iw: usize,
    ih: usize,
    t: &[f32],
    tw: usize,
    th: usize,
    m: TemplateMatchMethod,
) -> Vec<f32> {
    let n = (tw * th) as f32;
    let (mut ts, mut tsq) = (0.0f32, 0.0f32);
    for &v in t {
        ts += v;
        tsq += v * v;
    }
    let tm = ts / n;
    let tnorm = (tsq - ts * ts / n).sqrt();
    let mut out = Vec::new();
    for y in 0..=ih - th {
        for x in 0..=iw - tw {
            let mut pm = 0.0f32;
            for ty in 0..th {
                for tx in 0..tw {
                    pm += img[(y + ty) * iw + x + tx];
                }
            }
            pm /= n;
            let (mut sd, mut cc, mut ps, mut psq, mut co) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for ty in 0..th {
                for tx in 0..tw {
                    let a = img[(y + ty) * iw + x + tx];
                    let b = t[ty * tw + tx];
                    sd += (a - b) * (a - b);
                    cc += a * b;
                    ps += a;
                    psq += a * a;
                    co += (a - pm) * (b - tm);
                }
            }
            let norm = |num: f32, den: f32| if den > f32::EPSILON { num / den } else { 0.0 };
            out.push(match m {
                TemplateMatchMethod::SqDiff => sd,
                TemplateMatchMethod::SqDiffNormed => norm(sd, (psq * tsq).sqrt()),
                TemplateMatchMethod::Ccorr => cc,
                TemplateMatchMethod::CcorrNormed => norm(cc, (psq * tsq).sqrt()),
                TemplateMatchMethod::Ccoeff => co,
                TemplateMatchMethod::CcoeffNormed => norm(co, (psq - ps * ps / n).sqrt() * tnorm),
            });
        }
    }
    out
}

fn is_normed(m: TemplateMatchMethod) -> bool {
    matches!(
        m,
        TemplateMatchMethod::SqDiffNormed
            | TemplateMatchMethod::CcorrNormed
            | TemplateMatchMethod::CcoeffNormed
    )
}

#[test]
fn f64_template_matching_agrees_with_definition_and_old_kernel() {
    for (case, &(iw, ih, tw, th)) in [
        (1, 1, 1, 1),
        (9, 7, 3, 2),
        (40, 33, 11, 9),
        (64, 48, 17, 16),
        (25, 30, 25, 1),
    ]
    .iter()
    .enumerate()
    {
        let img = noise(case as u64, iw * ih);
        // Template cut from the image so an exact match exists.
        let (ox, oy) = ((iw - tw) / 2, (ih - th) / 2);
        let t: Vec<f32> = (0..tw * th)
            .map(|i| img[(oy + i / tw) * iw + ox + i % tw])
            .collect();
        let n = (tw * th) as f64;
        for m in METHODS {
            let got = run(&img, iw, ih, &t, tw, th, m);
            let def = definition(&img, iw, ih, &t, tw, th, m);
            let old = old_f32(&img, iw, ih, &t, tw, th, m);
            assert_eq!(got.len(), def.len());
            // Raw sums are bounded by n * 255^2; normalised scores by 1-2.
            let scale = if is_normed(m) { 1.0 } else { n * 255.0 * 255.0 };
            for i in 0..got.len() {
                let g = got[i] as f64;
                // f64 vs f64 definition: only the f32 output rounding remains.
                let tol_def = scale * 1e-12 + def[i].abs() * f32::EPSILON as f64;
                assert!(
                    (g - def[i]).abs() <= tol_def,
                    "{m:?} {iw}x{ih}/{tw}x{th} #{i}: got {g}, definition {}",
                    def[i]
                );
                // vs the old f32 kernel: its n-term f32 sums carry up to
                // ~n * 2^-24 relative error of the term magnitudes (scale);
                // normalised scores divide two such sums.
                let tol_old = if is_normed(m) {
                    4.0 * (n + 2.0) * f32::EPSILON as f64
                } else {
                    (n + 2.0) * f32::EPSILON as f64 * scale
                };
                if !(is_normed(m) && def[i] == 0.0) {
                    assert!(
                        (g - old[i] as f64).abs() <= tol_old,
                        "{m:?} {iw}x{ih}/{tw}x{th} #{i}: got {g}, old f32 kernel {}",
                        old[i]
                    );
                }
            }
            if is_normed(m) && m != TemplateMatchMethod::SqDiffNormed {
                assert!(
                    got.iter().all(|v| (-1.0..=1.0).contains(v)),
                    "{m:?} left [-1, 1]"
                );
            }
            let at = oy * (iw - tw + 1) + ox;
            if m == TemplateMatchMethod::CcoeffNormed && tw * th > 1 {
                assert!(
                    got[at] >= 1.0 - 1e-6 && got[at] <= 1.0,
                    "exact match scored {}",
                    got[at]
                );
            }
            if m == TemplateMatchMethod::SqDiff {
                assert_eq!(got[at], 0.0, "exact match SqDiff not 0");
            }
        }
    }
}

/// `cargo test --release -p cv-hal --test template_match_f64 -- --ignored --nocapture`
#[test]
#[ignore]
fn template_match_timing() {
    let (iw, ih, tw, th) = (640usize, 480usize, 32usize, 32usize);
    let img = tensor(noise(1, iw * ih), iw, ih);
    let t = tensor(noise(2, tw * th), tw, th);
    let cpu = CpuBackend::new().unwrap();
    for m in METHODS {
        let _: Tensor<f32, CpuStorage<f32>> = cpu.match_template(&img, &t, m).unwrap();
        let reps = 3;
        let start = Instant::now();
        for _ in 0..reps {
            let o: Tensor<f32, CpuStorage<f32>> = cpu.match_template(&img, &t, m).unwrap();
            std::hint::black_box(o);
        }
        println!(
            "match_template 640x480 / 32x32 {m:?}: {:8.2} ms",
            start.elapsed().as_secs_f64() / reps as f64 * 1e3
        );
    }
}
