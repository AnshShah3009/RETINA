//! Regression tests for validated-parameter defects (audit batch).
//!
//! Each test pairs a CONTROL assertion (an input that already worked before the
//! fix, so the test cannot pass vacuously) with the assertion that the defect is
//! gone.
//!
//! Covered defects:
//! 1. `local_threshold` Sauvola with `r = 0` returned an all-zero image.
//! 2. `sobel_ex` silently substituted a 3x3 kernel for any unsupported `ksize`.
//! 3. `distance_transform` accepted multi-channel tensors and ignored every
//!    channel past the first.
//! 4. `hough_lines` / `hough_lines_p` did not check `rho_res` / `theta_res`.

use cv_core::{CpuTensor, TensorShape};
use cv_imgproc::convolve::BorderMode;
use cv_imgproc::distance_transform::{
    distance_transform, distance_transform_with_labels, DistanceType,
};
use cv_imgproc::edges::sobel_ex;
use cv_imgproc::hough::{hough_lines, hough_lines_p};
use cv_imgproc::local_threshold::{local_threshold, LocalThresholdMethod};
use cv_imgproc::threshold::ThresholdType;
use image::{GrayImage, Luma};

// ---------------------------------------------------------------------------
// 1. local_threshold: Sauvola r = 0 produced an all-black image
// ---------------------------------------------------------------------------

fn uniform_image(v: u8, size: u32) -> GrayImage {
    GrayImage::from_pixel(size, size, Luma([v]))
}

#[test]
fn defect01_sauvola_r_zero_is_rejected_instead_of_returning_black() {
    let img = uniform_image(100, 16);

    // CONTROL: a valid Sauvola r still produces the all-white result it did
    // before the fix. On a uniform image every pixel equals the local mean and
    // std_dev == 0, so thresh = mean * (1 + k * (0/128 - 1)) = mean * (1 - k),
    // which is < mean for k > 0 -> every pixel is "above threshold" -> white.
    let good = local_threshold(
        &img,
        255,
        LocalThresholdMethod::Sauvola,
        ThresholdType::Binary,
        5,
        0.2,
        128.0,
    )
    .expect("r = 128.0 is a valid Sauvola dynamic range");
    assert!(
        good.as_raw().iter().all(|&v| v == 255),
        "CONTROL: r = 128.0 must keep producing an all-white image, got {:?}",
        good.as_raw()
    );

    // DEFECT: r = 0.0 makes std_dev / r = 0/0 = NaN and +inf for non-zero
    // std_dev, so thresh became NaN/+inf and every pixel was thresholded away.
    let bad = local_threshold(
        &img,
        255,
        LocalThresholdMethod::Sauvola,
        ThresholdType::Binary,
        5,
        0.2,
        0.0,
    );
    assert!(
        bad.is_err(),
        "Sauvola with r = 0.0 must return an error, got Ok(all-black: {:?})",
        bad.map(|i| i.as_raw().to_vec())
    );
    let msg = bad.unwrap_err().to_string();
    assert!(
        msg.contains("r"),
        "error message should name the offending parameter r, got: {msg}"
    );
    println!("defect01 error message: {msg}");

    // Other non-positive / non-finite values of r are equally degenerate.
    for bad_r in [f32::NEG_INFINITY, f32::NAN, -1.0] {
        assert!(
            local_threshold(
                &img,
                255,
                LocalThresholdMethod::Sauvola,
                ThresholdType::Binary,
                5,
                0.2,
                bad_r,
            )
            .is_err(),
            "Sauvola with r = {bad_r} must be rejected"
        );
    }

    // CONTROL: Niblack does not use r at all, so r = 0.0 stays legal there.
    let niblack = local_threshold(
        &img,
        255,
        LocalThresholdMethod::Niblack,
        ThresholdType::Binary,
        5,
        0.2,
        0.0,
    )
    .expect("Niblack does not use r");
    assert_eq!(niblack.width(), 16);
    assert_eq!(niblack.height(), 16);
}

#[test]
fn defect01_block_size_zero_is_rejected() {
    let img = uniform_image(100, 8);
    assert!(local_threshold(
        &img,
        255,
        LocalThresholdMethod::Niblack,
        ThresholdType::Binary,
        0,
        0.2,
        128.0,
    )
    .is_err());
}

// ---------------------------------------------------------------------------
// 2. sobel_ex: unsupported ksize silently became 3x3
// ---------------------------------------------------------------------------

fn step_image(size: u32) -> GrayImage {
    let mut img = GrayImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            img.put_pixel(x, y, Luma([if x < size / 2 { 0u8 } else { 255u8 }]));
        }
    }
    img
}

#[test]
fn defect02_sobel_ex_unsupported_ksize_is_rejected() {
    let img = step_image(32);

    // CONTROL: ksize = 3 works exactly as before.
    let k3 =
        sobel_ex(&img, 1, 0, 3, 1.0, 0.0, BorderMode::Replicate).expect("ksize = 3 is supported");
    assert_eq!(k3.width(), 32);
    assert_eq!(k3.height(), 32);
    assert!(
        k3.as_raw().iter().any(|&v| v > 0),
        "CONTROL: the 3x3 Sobel must still find the vertical edge"
    );

    // DEFECT: ksize = 4 produced output bit-identical to ksize = 3 with no error
    // and no warning, so a caller could not tell an unsupported size apart from
    // a 3-tap request.
    let k4 = sobel_ex(&img, 1, 0, 4, 1.0, 0.0, BorderMode::Replicate);
    assert!(
        k4.is_err(),
        "ksize = 4 has no separable Sobel kernel and must be rejected, got Ok(...)"
    );
    println!("defect02 error message: {}", k4.unwrap_err());

    for bad in [0usize, 1, 2, 4, 6, 8, 9, 100] {
        assert!(
            sobel_ex(&img, 1, 0, bad, 1.0, 0.0, BorderMode::Replicate).is_err(),
            "ksize = {bad} must be rejected"
        );
    }

    // CONTROL: 5 and 7 remain supported and still differ from each other.
    let k5 = sobel_ex(&img, 1, 0, 5, 1.0, 0.0, BorderMode::Replicate).expect("ksize = 5");
    let k7 = sobel_ex(&img, 1, 0, 7, 1.0, 0.0, BorderMode::Replicate).expect("ksize = 7");
    assert_ne!(k5.as_raw(), k3.as_raw(), "ksize 5 must differ from ksize 3");
    assert_ne!(k7.as_raw(), k3.as_raw(), "ksize 7 must differ from ksize 3");
}

// ---------------------------------------------------------------------------
// 3. distance_transform: multi-channel input was silently reduced to plane 0
// ---------------------------------------------------------------------------

#[test]
fn defect03_distance_transform_rejects_multichannel_input() {
    // 4x4 with a single background pixel at (0, 0): the distance map is known
    // exactly and is easy to reason about.
    let plane = {
        let mut d = vec![1.0f32; 16];
        d[0] = 0.0;
        d
    };

    // CONTROL: a (1, 4, 4) tensor works, as before.
    let single = CpuTensor::from_vec(plane.clone(), TensorShape::new(1, 4, 4)).unwrap();
    let dist = distance_transform(&single, DistanceType::L1).expect("1-channel is valid");
    assert_eq!(dist.shape.channels, 1);
    let d = dist.as_slice().unwrap();
    assert_eq!(d[0], 0.0);
    // L1 distance from (1,1) to the background pixel at (0,0) is |1-0| + |1-0|
    // = 2, not 5.
    assert_eq!(d[5], 2.0, "Manhattan distance from (1,1) to (0,0)");
    assert_eq!(d[15], 6.0, "Manhattan distance from (3,3) to (0,0)");

    // DEFECT: a (3, 4, 4) input where channel 0 is all-foreground and channel 1
    // is all-background used to return Ok, computed from the red plane alone
    // (the whole buffer was indexed as y * width + x), with an output that
    // still claimed channels: 1.
    let mut three_chan = vec![1.0f32; 48];
    for i in 16..32 {
        three_chan[i] = 0.0; // channel 1: all background
    }
    let multi = CpuTensor::from_vec(three_chan, TensorShape::new(3, 4, 4)).unwrap();
    assert_eq!(multi.shape.channels, 3);

    let res = distance_transform(&multi, DistanceType::L1);
    assert!(
        res.is_err(),
        "a 3-channel input must be rejected; previously it silently used channel 0 only \
         (all-foreground) and returned Ok"
    );
    println!("defect03 error message: {}", res.unwrap_err());

    // Same for the labels variant.
    let res = distance_transform_with_labels(&multi, DistanceType::L1);
    assert!(
        res.is_err(),
        "distance_transform_with_labels must reject multi-channel input too"
    );

    // CONTROL: single-channel with labels still works.
    let (dmap, lmap) =
        distance_transform_with_labels(&single, DistanceType::L1).expect("1-channel is valid");
    assert_eq!(dmap.shape.channels, 1);
    assert_eq!(lmap.as_slice().unwrap()[15], 0);
}

// ---------------------------------------------------------------------------
// 4. hough_lines / hough_lines_p: unchecked rho_res / theta_res
// ---------------------------------------------------------------------------

fn line_image() -> GrayImage {
    let mut img = GrayImage::new(64, 64);
    for x in 0..64 {
        img.put_pixel(x, 32, Luma([255]));
    }
    img
}

#[test]
fn defect04_hough_lines_rejects_bad_resolutions() {
    let img = line_image();
    const THETA: f32 = std::f32::consts::PI / 180.0;

    // CONTROL: rho_res = 1.0 / theta_res = 1 degree returns 5 lines on this image.
    let lines = hough_lines(&img, 1.0, THETA, 10).expect("valid resolutions");
    assert_eq!(
        lines.len(),
        5,
        "CONTROL: rho_res = 1.0 must keep detecting the 5 line segments of the drawn line"
    );

    // DEFECT (a): theta_res = 0.0 -> num_theta = (PI / 0) as usize == usize::MAX,
    // and num_rho * num_theta overflowed the accumulator allocation (panic).
    let res = hough_lines(&img, 1.0, 0.0, 10);
    assert!(res.is_err(), "theta_res = 0.0 must be rejected, not panic");
    println!("defect04a error message: {}", res.unwrap_err());

    // DEFECT (b): rho_res = 0.0 used to return Ok(vec![]) - no lines and no
    // error - where the same image with rho_res = 1.0 returns 5 lines.
    let res = hough_lines(&img, 0.0, THETA, 10);
    assert!(
        res.is_err(),
        "rho_res = 0.0 must be rejected; it previously returned Ok(empty vec)"
    );
    println!("defect04b error message: {}", res.unwrap_err());

    for (rho, theta) in [
        (0.0, 0.0),
        (-1.0, THETA),
        (1.0, -0.1),
        (f32::NAN, THETA),
        (1.0, f32::NAN),
        (f32::INFINITY, THETA),
        (1.0, f32::INFINITY),
    ] {
        assert!(
            hough_lines(&img, rho, theta, 10).is_err(),
            "rho_res = {rho}, theta_res = {theta} must be rejected"
        );
    }
}

#[test]
fn defect04_hough_lines_p_rejects_bad_resolutions() {
    let mut img = GrayImage::new(200, 120);
    for y in 80..120 {
        for x in 20..180 {
            img.put_pixel(x, y, Luma([255]));
        }
    }
    const THETA: f32 = std::f32::consts::PI / 180.0;

    // CONTROL: valid resolutions detect the rectangle edges.
    let segments = hough_lines_p(&img, 1.0, THETA, 3, 10.0, 20.0).expect("valid resolutions");
    assert!(
        !segments.is_empty(),
        "CONTROL: PPHT with rho_res = 1.0 must still find segments"
    );

    // DEFECT: theta_res = 0.0 panicked with an overflowing accumulator
    // allocation; rho_res = 0.0 silently returned an empty list.
    let res = hough_lines_p(&img, 1.0, 0.0, 3, 10.0, 20.0);
    assert!(res.is_err(), "theta_res = 0.0 must be rejected, not panic");
    println!("defect04c error message: {}", res.unwrap_err());

    let res = hough_lines_p(&img, 0.0, THETA, 3, 10.0, 20.0);
    assert!(
        res.is_err(),
        "rho_res = 0.0 must be rejected; it previously returned Ok(empty vec)"
    );
    println!("defect04d error message: {}", res.unwrap_err());
}
