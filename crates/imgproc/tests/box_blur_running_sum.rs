//! `box_blur` CPU path: the O(1)-per-pixel running-sum implementation must
//! return exactly the bytes of the dense O(size^2) replicate-border
//! convolution it replaced (`convolve_with_border_into_ctx` + `box_kernel`).

use cv_imgproc::{
    box_blur_cpu_into, box_blur_ctx, box_kernel, convolve_with_border_into_ctx, BorderMode,
};
use cv_runtime::RuntimeRunner;
use image::GrayImage;
use std::time::Instant;

fn cpu_runner() -> RuntimeRunner {
    RuntimeRunner::Sync(cv_runtime::registry().unwrap().default_cpu().id())
}

fn noise(w: u32, h: u32, seed: u64) -> GrayImage {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    GrayImage::from_fn(w, h, |x, y| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Mix smooth structure with noise so window means hit every residue.
        let v = ((s >> 33) as u32 % 256 + x * 3 + y * 5) % 256;
        image::Luma([v as u8])
    })
}

fn dense(img: &GrayImage, size: usize) -> GrayImage {
    let mut out = GrayImage::new(img.width(), img.height());
    convolve_with_border_into_ctx(
        img,
        &mut out,
        &box_kernel(size),
        BorderMode::Replicate,
        &cpu_runner(),
    );
    out
}

#[test]
fn running_sum_box_blur_is_byte_identical_to_dense_convolution() {
    let shapes = [(1, 1), (1, 7), (7, 1), (2, 3), (17, 9), (64, 33), (97, 61)];
    for (i, &(w, h)) in shapes.iter().enumerate() {
        let img = noise(w, h, i as u64);
        for size in [
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 21, 31, 32, 40,
        ] {
            let mut fast = GrayImage::new(w, h);
            box_blur_cpu_into(&img, &mut fast, size);
            let reference = dense(&img, size);
            assert_eq!(
                fast.as_raw(),
                reference.as_raw(),
                "box_blur size {size} on {w}x{h} differs from the dense loop"
            );
            // The public CPU entry point routes to the same code.
            assert_eq!(
                box_blur_ctx(&img, size, &cpu_runner()).as_raw(),
                reference.as_raw()
            );
        }
    }
    // Flat images make every window an exact tie candidate for even sizes.
    for v in [0u8, 1, 127, 128, 254, 255] {
        let img = GrayImage::from_pixel(23, 19, image::Luma([v]));
        for size in [2usize, 4, 6, 8, 12] {
            let mut fast = GrayImage::new(23, 19);
            box_blur_cpu_into(&img, &mut fast, size);
            assert_eq!(fast.as_raw(), dense(&img, size).as_raw());
        }
    }
}

/// `cargo test --release -p cv-imgproc --test box_blur_running_sum -- --ignored --nocapture`
#[test]
#[ignore]
fn box_blur_timing() {
    let img = noise(1920, 1080, 7);
    for size in [3usize, 4, 7, 15, 31] {
        let reps = 5;
        let mut fast = GrayImage::new(1920, 1080);
        box_blur_cpu_into(&img, &mut fast, size);
        let t = Instant::now();
        for _ in 0..reps {
            box_blur_cpu_into(&img, &mut fast, size);
        }
        let t_fast = t.elapsed().as_secs_f64() / reps as f64;
        let reference = dense(&img, size);
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(dense(&img, size));
        }
        let t_dense = t.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(fast.as_raw(), reference.as_raw());
        println!(
            "box_blur 1920x1080 size {size:2}: dense {:8.2} ms, running-sum {:6.2} ms ({:.0}x)",
            t_dense * 1e3,
            t_fast * 1e3,
            t_dense / t_fast
        );
    }
}
