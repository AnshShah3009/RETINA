//! `CpuBackend::morphology`: rectangular masks above 5x5 take the separable
//! `cpu::morph_rect` path and other masks the per-tap loop (whose 32-wide
//! block now loads interior rows as slices). Both must return exactly the
//! bytes of the per-tap definition: out-of-image taps replicate the edge pixel.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{ComputeContext, MorphologyType};
use cv_hal::cpu::CpuBackend;
use std::time::Instant;

fn u8_tensor(data: &[u8], w: usize, h: usize, c: usize) -> Tensor<u8, CpuStorage<u8>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()
}

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn per_tap(
    src: &[u8],
    w: usize,
    h: usize,
    c: usize,
    mask: &[u8],
    kw: usize,
    kh: usize,
    typ: MorphologyType,
) -> Vec<u8> {
    let identity = if typ == MorphologyType::Erode {
        255u8
    } else {
        0
    };
    let (cx, cy) = (kw / 2, kh / 2);
    let mut out = vec![0u8; src.len()];
    for p in 0..c {
        let plane = &src[p * w * h..(p + 1) * w * h];
        for y in 0..h {
            for x in 0..w {
                let mut v = identity;
                for ky in 0..kh {
                    for kx in 0..kw {
                        if mask[ky * kw + kx] == 0 {
                            continue;
                        }
                        let sy = y as isize + ky as isize - cy as isize;
                        let sx = x as isize + kx as isize - cx as isize;
                        let t = if w == 0 || h == 0 {
                            identity
                        } else {
                            let sy = sy.clamp(0, h as isize - 1) as usize;
                            let sx = sx.clamp(0, w as isize - 1) as usize;
                            plane[sy * w + sx]
                        };
                        v = if typ == MorphologyType::Erode {
                            v.min(t)
                        } else {
                            v.max(t)
                        };
                    }
                }
                out[p * w * h + y * w + x] = v;
            }
        }
    }
    out
}

#[test]
fn morphology_matches_per_tap_definition_for_rect_and_general_masks() {
    let cpu = CpuBackend::new().unwrap();
    let typs = [
        MorphologyType::Erode,
        MorphologyType::Dilate,
        MorphologyType::Open,
        MorphologyType::Close,
    ];
    let mut masks: Vec<(Vec<u8>, usize, usize)> =
        [(1, 1), (3, 3), (2, 5), (7, 1), (1, 4), (17, 3), (21, 21)]
            .iter()
            .map(|&(kw, kh)| (vec![1u8; kw * kh], kw, kh))
            .collect();
    // Non-rectangular: cross, sparse random, all-zero.
    masks.push((vec![0, 1, 0, 1, 1, 1, 0, 1, 0], 3, 3));
    masks.push((noise(9, 35).iter().map(|v| v & 1).collect(), 7, 5));
    masks.push((vec![0u8; 9], 3, 3));

    for (i, &(w, h, c)) in [
        (1, 1, 1),
        (5, 1, 1),
        (1, 6, 1),
        (31, 7, 1),
        (33, 9, 2),
        (100, 13, 1),
        (64, 20, 3),
    ]
    .iter()
    .enumerate()
    {
        let src = noise(i as u64, w * h * c);
        for (mask, kw, kh) in &masks {
            let k = u8_tensor(mask, *kw, *kh, 1);
            for &typ in &typs {
                for iters in [1u32, 2] {
                    let got = cpu
                        .morphology(&u8_tensor(&src, w, h, c), typ, &k, iters)
                        .unwrap();
                    let mut want = src.clone();
                    for _ in 0..iters {
                        want = per_tap(&want, w, h, c, mask, *kw, *kh, typ);
                    }
                    assert_eq!(
                        got.as_slice().unwrap(),
                        &want[..],
                        "{w}x{h}x{c} {typ:?} iters={iters} mask {kw}x{kh} {mask:?}"
                    );
                }
            }
        }
    }
}

/// `cargo test --release -p cv-hal --test morph_rect_separable -- --ignored --nocapture`
#[test]
#[ignore]
fn morphology_timing() {
    let cpu = CpuBackend::new().unwrap();
    let (w, h) = (1920usize, 1080usize);
    let img = u8_tensor(&noise(1, w * h), w, h, 1);
    let reps = 5;
    let time = |mask: &[u8], k: usize| {
        let kt = u8_tensor(mask, k, k, 1);
        cpu.morphology(&img, MorphologyType::Erode, &kt, 1).unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(cpu.morphology(&img, MorphologyType::Erode, &kt, 1).unwrap());
        }
        t.elapsed().as_secs_f64() / reps as f64 * 1e3
    };
    for k in [3usize, 5, 7, 15, 31] {
        let rect = vec![1u8; k * k];
        // Same mask with one corner tap cleared: forces the per-tap loop with
        // one tap less work, the closest stand-in for the old rectangle cost.
        let mut general = rect.clone();
        general[0] = 0;
        let t_rect = time(&rect, k);
        let t_general = time(&general, k);
        println!(
            "hal erode 1920x1080 {k:2}x{k:<2}: one tap cleared {t_general:8.2} ms, full rect {t_rect:6.2} ms ({:.0}x)",
            t_general / t_rect
        );
    }
}
