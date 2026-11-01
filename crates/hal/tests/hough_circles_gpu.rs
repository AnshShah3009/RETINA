//! The GPU Hough-circle kernel binds the edge map as `array<f32>`, but its
//! `get_pixel` still unpacked four bytes per word the way it did when the
//! binding was `array<u32>`. On a float buffer the low byte of a whole-number
//! pixel is 0x00, so every pixel read as 0, the Sobel magnitude stayed below
//! the edge threshold and nothing voted. As shipped the shader does not even
//! validate (`ShiftRight` on an f32), so `hough_circles` panicked before it
//! could return an (empty) accumulator.
//! Skips when no adapter is available.

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::TensorToGpu;
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU hough circles test (no adapter): {}", e);
            None
        }
    }
}

fn disk(w: usize, h: usize, cx: f32, cy: f32, radius: f32) -> CpuTensor<f32> {
    let data: Vec<f32> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                if d <= radius { 255.0 } else { 0.0 }
            })
        })
        .collect();
    Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap()
}

#[test]
fn gpu_hough_circles_see_a_single_disk() {
    let Some(gpu) = require_gpu() else { return };

    // A filled disk so the boundary produces a Sobel response (the centre tap
    // is not part of the kernel, so a one-pixel ring reads as flat). The vote
    // threshold is low because the discrete gradient spreads the centre peak
    // over a few cells; the point of the test is that any vote survives.
    let (w, h) = (65usize, 65usize);
    let (cx, cy, radius) = (32.0f32, 32.0f32, 20.0f32);
    let image = disk(w, h, cx, cy, radius);
    let device_image = image.to_gpu_ctx(gpu).unwrap();

    let circles = gpu
        .hough_circles(&device_image, 15.0f32, 25.0f32, 8u32)
        .unwrap();
    assert!(
        !circles.is_empty(),
        "the disk's edge pixels cast no votes at all"
    );

    let found = circles
        .iter()
        .any(|c| (c.cx - cx).powi(2) + (c.cy - cy).powi(2) <= 4.0 && (c.r - radius).abs() <= 0.5);
    assert!(
        found,
        "expected a circle at ({cx}, {cy}) r={radius}, got {:?}",
        circles
    );
}
