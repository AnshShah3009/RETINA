//! `cv-hal`'s CPU `morphology` and `cv-imgproc`'s morphology must agree on the
//! same image with the same structuring element.
//!
//! `MorphologyType::Open` and `MorphologyType::Close` are *composed* operations:
//! open is erode-then-dilate, close is dilate-then-erode. `cv-imgproc` exposes
//! exactly that as [`opening`] / [`closing`] (and as the `opening` / `closing`
//! loops over `erode_with_border_into` / `dilate_with_border_into`), and
//! `cv-hal`'s GPU kernel dispatches them as that composition too
//! (`gpu_kernels/morphology.rs`, the `Open || Close` branch before the shader).
//!
//! The HAL CPU backend had no such branch: `is_max` was derived as
//! `typ != MorphologyType::Erode`, so **Open and Close both ran a plain
//! dilate**. On a lone bright pixel — the smallest input where open and dilate
//! disagree at all — open must delete the pixel (erosion has no neighbour to
//! keep it), and the CPU backend returned the 3x3 spread dilate produces.
//!
//! The image and the kernel are deliberately tiny: one pixel, one number, no
//! hypothesis about phase or padding.
//!
//! `cv-imgproc`'s own `morphology_ctx` had the mirror defect on its CPU
//! fallback: the `match typ` there listed `Dilate` and `Erode` and let `_` fall
//! through to `GrayImage::new`, so on a CPU group `Open` and `Close` returned
//! an all-black image. That is pinned here too, against `opening`/`closing`.

#![forbid(unsafe_code)]

use cv_core::storage::{CpuStorage, Storage};
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::{ComputeContext, MorphologyType};
use cv_hal::cpu::CpuBackend;
use cv_imgproc::{
    create_morph_kernel, dilate_with_border_into, erode_with_border_into, BorderMode, MorphShape,
};
use image::GrayImage;

fn kernel_tensor(kernel: &[(i32, i32)]) -> (Tensor<u8, CpuStorage<u8>>, usize, usize) {
    let (min_x, max_x, min_y, max_y) = kernel.iter().fold((0, 0, 0, 0), |(a, b, c, d), &(x, y)| {
        (a.min(x), b.max(x), c.min(y), d.max(y))
    });
    let kw = (max_x - min_x + 1) as usize;
    let kh = (max_y - min_y + 1) as usize;
    let mut data = vec![0u8; kw * kh];
    for &(x, y) in kernel {
        data[(y - min_y) as usize * kw + (x - min_x) as usize] = 1;
    }
    (
        Tensor::from_vec(data, TensorShape::new(1, kh, kw)).unwrap(),
        kw,
        kh,
    )
}

fn u8_tensor(data: &[u8], w: usize, h: usize) -> Tensor<u8, CpuStorage<u8>> {
    Tensor::from_vec(data.to_vec(), TensorShape::new(1, h, w)).unwrap()
}

/// `cv-imgproc`'s CPU composition of the same operation, with the same
/// replicate border `cv-hal` uses.
fn imgproc_compose(
    src: &GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    open: bool,
) -> GrayImage {
    let mut a = GrayImage::new(src.width(), src.height());
    let mut b = GrayImage::new(src.width(), src.height());
    if open {
        erode_with_border_into(src, &mut a, kernel, iterations, BorderMode::Replicate);
        dilate_with_border_into(&a, &mut b, kernel, iterations, BorderMode::Replicate);
    } else {
        dilate_with_border_into(src, &mut a, kernel, iterations, BorderMode::Replicate);
        erode_with_border_into(&a, &mut b, kernel, iterations, BorderMode::Replicate);
    }
    b
}

#[test]
fn hal_cpu_open_and_close_match_imgproc_on_the_same_image() {
    let cpu = CpuBackend::new().unwrap();

    // One bright pixel on black: open must remove it, and close must keep it.
    let (w, h) = (9usize, 7usize);
    let mut lone = vec![0u8; w * h];
    lone[(h / 2) * w + w / 2] = 255;

    // A 2x2 block: open keeps it (it survives a 3x3 erosion only partly), close
    // fills around it — the case where the two compositions differ from each
    // other and from dilate in different directions.
    let mut block = vec![0u8; w * h];
    for y in 3..5 {
        for x in 4..6 {
            block[y * w + x] = 180;
        }
    }

    let images: [(&str, &[u8], usize, usize); 2] =
        [("lone pixel", &lone, w, h), ("2x2 block", &block, w, h)];

    for (name, data, iw, ih) in images {
        let src = GrayImage::from_raw(iw as u32, ih as u32, data.to_vec()).unwrap();
        for &(ksize, shape_name) in &[(3u32, "3x3"), (7, "7x7")] {
            let kernel = create_morph_kernel(MorphShape::Rectangle, ksize, ksize);
            let (kt, kw, kh) = kernel_tensor(&kernel);
            assert_eq!((kw, kh), (ksize as usize, ksize as usize));

            for &iterations in &[1u32, 2] {
                for &(typ, open) in &[(MorphologyType::Open, true), (MorphologyType::Close, false)]
                {
                    let got = cpu
                        .morphology(&u8_tensor(data, iw, ih), typ, &kt, iterations)
                        .unwrap()
                        .storage
                        .as_slice()
                        .unwrap()
                        .to_vec();
                    let want = imgproc_compose(&src, &kernel, iterations, open);
                    let want: Vec<u8> = want.as_raw().clone();
                    let diffs: Vec<(usize, u8, u8)> = got
                        .iter()
                        .zip(&want)
                        .enumerate()
                        .filter(|(_, (g, w))| g != w)
                        .map(|(i, (g, w))| (i, *g, *w))
                        .collect();
                    assert!(
                        diffs.is_empty(),
                        "{name} {shape_name} iters={iterations} {typ:?}: \
                         cv-hal CPU and cv-imgproc disagree on {} of {} pixels; \
                         first (index, hal, imgproc): {:?}\n\
                         hal     = {got:?}\nimgproc = {want:?}",
                        diffs.len(),
                        got.len(),
                        &diffs[..diffs.len().min(6)]
                    );
                }
            }
        }
    }

    // The smallest possible statement of the defect: open of a lone pixel is
    // black, and the HAL must not report the dilate that keeps it lit.
    let kernel = create_morph_kernel(MorphShape::Rectangle, 3, 3);
    let (kt, _, _) = kernel_tensor(&kernel);
    let got = cpu
        .morphology(&u8_tensor(&lone, w, h), MorphologyType::Open, &kt, 1)
        .unwrap()
        .storage
        .as_slice()
        .unwrap()
        .to_vec();
    assert!(
        got.iter().all(|&v| v == 0),
        "opening a lone 255 pixel must delete it, got {got:?}"
    );
}

/// `cv-imgproc`'s `morphology_ctx` on a **CPU** group must agree with
/// `opening` / `closing` on the same image.
///
/// Its CPU fallback matched only `Dilate` and `Erode` and let `Open` and
/// `Close` fall into `_ => {}`, leaving `GrayImage::new` — a fully black image
/// for every input, on the very backend that is supposed to be the reference.
#[test]
fn imgproc_morphology_ctx_open_and_close_agree_with_opening_and_closing() {
    let cpu_id = cv_runtime::registry().unwrap().default_cpu().id();
    let group = cv_runtime::scheduler()
        .unwrap()
        .create_group_with_device(
            "imgproc_open_close_parity",
            1,
            None,
            cv_runtime::GroupPolicy::default(),
            cpu_id,
        )
        .unwrap();

    // A 2x2 bright block on black, 3x3 rectangle: open keeps part of it, close
    // keeps all of it and grows it. Both are non-empty, so a blank result is
    // distinguishable from either composition.
    let (w, h) = (9u32, 7u32);
    let src = GrayImage::from_fn(w, h, |x, y| {
        image::Luma([if (4..6).contains(&x) && (3..5).contains(&y) {
            180
        } else {
            20
        }])
    });

    for &ksize in &[3u32, 7] {
        let kernel = create_morph_kernel(MorphShape::Rectangle, ksize, ksize);
        for &iterations in &[1u32, 2] {
            for &(typ, open) in &[(MorphologyType::Open, true), (MorphologyType::Close, false)] {
                let got = cv_imgproc::morphology_ctx(
                    &src,
                    typ,
                    &kernel,
                    iterations,
                    BorderMode::Replicate,
                    &group,
                );
                let want = imgproc_compose(&src, &kernel, iterations, open);
                assert_eq!(
                    got.as_raw(),
                    want.as_raw(),
                    "cv-imgproc morphology_ctx {ksize}x{ksize} iters={iterations} {typ:?} \
                     differs from the composition it names"
                );
            }
        }
    }
}
