//! BRIEF steers its sampling pattern by `KeyPoint::angle`, which is in
//! **degrees** across the workspace (ORB stores `atan2(..).to_degrees()`,
//! OpenCV's convention), with `-1.0` meaning "no orientation".
//!
//! `compute_brief` fed the angle straight to `cos`/`sin` as radians and
//! treated *every* negative angle as unset. ORB's `atan2` yields negative
//! angles for about half of all keypoints, so those got an unrotated
//! descriptor, and the rest were rotated by `angle` radians - 57x too far.
//! Measured on a random texture: hamming(0 deg, 360 deg) = 124 bits of 256.

use cv_core::{KeyPoint, KeyPoints};
use cv_features::{BriefDescriptor, DescriptorExtractor};
use image::GrayImage;

fn texture() -> GrayImage {
    let mut s: u32 = 12345;
    GrayImage::from_fn(64, 64, |_, _| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        image::Luma([(s >> 16) as u8])
    })
}

fn describe(img: &GrayImage, angle: f64) -> Vec<u8> {
    let mut kps = KeyPoints::new();
    kps.push(KeyPoint::new(32.0, 32.0).with_angle(angle));
    let d = BriefDescriptor::new(32, 31).extract(img, &kps);
    d.descriptors[0].data.clone()
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

#[test]
fn angles_are_degrees_and_only_minus_one_means_unset() {
    let img = texture();
    // Same orientation, different spelling: descriptors must agree up to
    // the odd bit flipped by bilinear rounding at the pattern's edge.
    assert!(hamming(&describe(&img, 0.0), &describe(&img, 360.0)) <= 4);
    assert!(hamming(&describe(&img, -90.0), &describe(&img, 270.0)) <= 4);
    // A real negative orientation is not "unset".
    assert!(hamming(&describe(&img, -90.0), &describe(&img, -1.0)) > 32);
    // Unset is the unrotated pattern.
    assert_eq!(describe(&img, -1.0), describe(&img, 0.0));
}
