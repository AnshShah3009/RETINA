//! Argument validation that has to agree between the CPU and GPU backends:
//! `mog2_update`'s parameters against its frame and buffers, `nms_boxes`'s
//! box layout, and `resize` of an empty source.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::{ComputeContext, Interpolation, Mog2Params};
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn gpu() -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init_global()) {
        Ok(g) => Some(g),
        Err(_) => {
            println!("no GPU adapter: checking the CPU backend only");
            None
        }
    }
}

fn t<T: bytemuck::Pod + std::fmt::Debug + Send + Sync + Clone + Copy>(
    v: Vec<T>,
    c: usize,
    h: usize,
    w: usize,
) -> CpuTensor<T> {
    Tensor::from_vec(v, TensorShape::new(c, h, w)).unwrap()
}

/// `Ok`, the error's variant, or `PANIC`.
fn outcome<R>(f: impl FnOnce() -> cv_hal::Result<R>) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(_)) => "Ok".into(),
        Ok(Err(e)) => {
            let d = format!("{e:?}");
            d.split('(').next().unwrap_or(&d).to_string()
        }
        Err(_) => "PANIC".into(),
    }
}

const W: usize = 6;
const H: usize = 4;

fn params(
    width: u32,
    height: u32,
    n_mixtures: u32,
    adaptive: bool,
    max_components: u32,
) -> Mog2Params<f32> {
    Mog2Params {
        width,
        height,
        n_mixtures,
        alpha: 0.05,
        var_threshold: 16.0,
        background_ratio: 0.9,
        var_init: 0.01,
        var_min: 0.001,
        var_max: 0.2,
        adaptive_k: u32::from(adaptive),
        max_components,
        _padding: [0],
    }
}

fn stride(p: &Mog2Params<f32>) -> usize {
    if p.adaptive_k != 0 {
        p.max_components as usize * 3 + 1
    } else {
        p.n_mixtures as usize * 3
    }
}

/// Runs one MOG2 update on both backends; returns (CPU, GPU) outcomes and,
/// when both succeed, the masks.
fn mog2(
    p: &Mog2Params<f32>,
    model_len: usize,
    mask_len: usize,
) -> (String, Option<String>, Option<(Vec<u32>, Vec<u32>)>) {
    let frame = t((0..H * W).map(|i| (i % 5) as f32 * 0.2).collect(), 1, H, W);
    let model0: Vec<f32> = (0..model_len)
        .map(|i| match i % 3 {
            0 => 0.5,
            1 => 0.4,
            _ => 0.01,
        })
        .collect();
    let mut cm = t(model0.clone(), 1, 1, model_len);
    let mut cmask = t(vec![0u32; mask_len], 1, 1, mask_len);
    let cpu = CpuBackend::new().unwrap();
    let c = outcome(|| cpu.mog2_update(&frame, &mut cm, &mut cmask, p));
    let Some(g) = gpu() else {
        return (c, None, None);
    };
    let mut gmask_out = None;
    let gres = outcome(|| {
        let mut gm = t(model0.clone(), 1, 1, model_len).to_gpu_ctx(g)?;
        let mut gmask = t(vec![0u32; mask_len], 1, 1, mask_len).to_gpu_ctx(g)?;
        g.mog2_update(&frame.to_gpu_ctx(g)?, &mut gm, &mut gmask, p)?;
        gmask_out = Some(gmask.to_cpu_ctx(g)?.storage.as_slice().unwrap().to_vec());
        Ok(())
    });
    let masks = gmask_out.map(|gm| (cmask.storage.as_slice().unwrap().to_vec(), gm));
    (c, Some(gres), masks)
}

#[test]
fn mog2_rejects_parameters_that_do_not_fit_the_frame() {
    let fixed = params(W as u32, H as u32, 3, false, 3);
    let adaptive = params(W as u32, H as u32, 3, true, 4);
    let px = W * H;
    let cases: Vec<(&str, Mog2Params<f32>, usize, usize)> = vec![
        ("width 0", params(0, H as u32, 3, false, 3), px * 9, px),
        ("height 0", params(W as u32, 0, 3, false, 3), px * 9, px),
        (
            "no mixtures",
            params(W as u32, H as u32, 0, false, 3),
            px * 9,
            px,
        ),
        (
            "adaptive, no components",
            params(W as u32, H as u32, 3, true, 0),
            px * 9,
            px,
        ),
        (
            "width too large",
            params(W as u32 + 1, H as u32, 3, false, 3),
            (px + H) * 9,
            px + H,
        ),
        (
            "width too small",
            params(W as u32 - 1, H as u32, 3, false, 3),
            px * 9,
            px,
        ),
        (
            "height too large",
            params(W as u32, H as u32 + 1, 3, false, 3),
            (px + W) * 9,
            px + W,
        ),
        ("model too short", fixed, px * stride(&fixed) - 1, px),
        ("mask too short", fixed, px * stride(&fixed), px - 1),
        (
            "adaptive model too short",
            adaptive,
            px * stride(&adaptive) - 1,
            px,
        ),
    ];
    let mut failures = Vec::new();
    for (what, p, model_len, mask_len) in cases {
        let (c, g, _) = mog2(&p, model_len, mask_len);
        if c != "InvalidInput" {
            failures.push(format!("CPU {what}: {c}"));
        }
        if let Some(g) = g
            && g != "InvalidInput"
        {
            failures.push(format!("GPU {what}: {g}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mog2_valid_parameters_still_update_on_both_backends() {
    for p in [
        params(W as u32, H as u32, 3, false, 3),
        params(W as u32, H as u32, 3, true, 4),
    ] {
        let (c, g, masks) = mog2(&p, W * H * stride(&p), W * H);
        assert_eq!(c, "Ok", "{p:?}");
        if let Some(g) = g {
            assert_eq!(g, "Ok", "{p:?}");
            let (cm, gm) = masks.unwrap();
            assert_eq!(cm, gm, "CPU and GPU masks differ for {p:?}");
        }
    }
}

#[test]
fn nms_boxes_requires_five_columns_on_both_backends() {
    let cpu = CpuBackend::new().unwrap();
    let g = gpu();
    for (c, h, w) in [(1, 3, 4), (1, 3, 6), (1, 0, 4), (1, 2, 1)] {
        let boxes = t(vec![1.0f32; c * h * w], c, h, w);
        let co = outcome(|| cpu.nms_boxes(&boxes, 0.5));
        assert_eq!(co, "InvalidInput", "CPU ({c}, {h}, {w})");
        if let Some(g) = g {
            let go = outcome(|| g.nms_boxes(&boxes.to_gpu_ctx(g)?, 0.5));
            assert_eq!(go, "InvalidInput", "GPU ({c}, {h}, {w})");
        }
    }
    // The layout the CPU accepts still works on the GPU.
    let boxes = t(
        vec![
            0.0f32, 0.0, 10.0, 10.0, 0.9, 1.0, 1.0, 11.0, 11.0, 0.8, 50.0, 50.0, 60.0, 60.0, 0.7,
        ],
        1,
        3,
        5,
    );
    let want = cpu.nms_boxes(&boxes, 0.5).unwrap();
    assert_eq!(want, vec![0, 2]);
    if let Some(g) = g {
        assert_eq!(
            g.nms_boxes(&boxes.to_gpu_ctx(g).unwrap(), 0.5).unwrap(),
            want
        );
    }
}

/// `cv::resize` asserts `!ssize.empty()`: an empty source has no pixel to
/// sample for a non-empty target. An empty target is still an empty tensor.
#[test]
fn resize_of_an_empty_source_to_a_nonempty_size_is_invalid_input() {
    let cpu = CpuBackend::new().unwrap();
    let g = gpu();
    for (c, h, w) in [(1, 0, 0), (1, 0, 8), (1, 8, 0), (3, 0, 8)] {
        let src = t(Vec::<f32>::new(), c, h, w);
        assert_eq!(
            outcome(|| cpu.resize(&src, (4, 4))),
            "InvalidInput",
            "CPU ({c}, {h}, {w})"
        );
        let to_empty = cpu.resize(&src, (0, 4)).unwrap();
        assert_eq!(to_empty.shape, TensorShape::new(c, 4, 0));
        if let Some(g) = g {
            let gs = || src.to_gpu_ctx(g).unwrap();
            assert_eq!(
                outcome(|| g.resize(&gs(), (4, 4))),
                "InvalidInput",
                "GPU ({c}, {h}, {w})"
            );
            for i in [Interpolation::Linear, Interpolation::Lanczos] {
                assert_eq!(
                    outcome(|| g.resize_with(&gs(), (4, 4), i)),
                    "InvalidInput",
                    "GPU resize_with {i:?} ({c}, {h}, {w})"
                );
            }
            let o = g.resize(&gs(), (0, 4)).unwrap().to_cpu_ctx(g).unwrap();
            assert_eq!(o.shape, TensorShape::new(c, 4, 0));
        }
    }
}
