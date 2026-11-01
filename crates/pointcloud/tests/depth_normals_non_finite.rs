//! A depth sample without a position must not leak into its neighbours' normals.
//!
//! `compute_normals_from_depth` rejected a sample with `d <= 0.0`. NaN fails
//! every comparison, so a NaN depth slipped past that guard and was
//! back-projected to `[NaN, NaN, NaN]`, while an infinite depth produced
//! `inf - inf` in the horizontal/vertical difference. Either way the four
//! pixels that use the bad sample as a neighbour got a NaN normal: one bad
//! sample took out four surrounding pixels of a depth map.
//!
//! The bad pixel itself is unaffected by its own depth - its normal is built
//! from its four neighbours - so it still gets a real normal.

use cv_pointcloud::compute_normals_from_depth;
use nalgebra::Vector3;

const W: usize = 5;
const H: usize = 5;
const BAD: usize = 2 * W + 2; // centre pixel

fn flat_depth_with(bad_value: f32) -> Vec<f32> {
    let mut depth = vec![1.0f32; W * H];
    depth[BAD] = bad_value;
    depth
}

/// The pixels whose four-neighbour stencil contains the bad centre pixel.
const POISONED: [usize; 4] = [1 * W + 2, 3 * W + 2, 2 * W + 1, 2 * W + 3];

fn assert_only_the_expected_pixels_were_affected(depth: Vec<f32>) {
    let normals = compute_normals_from_depth(&depth, W, H, 5.0, 5.0, 2.0, 2.0);
    assert_eq!(normals.len(), W * H);

    for (i, n) in normals.iter().enumerate() {
        assert!(
            n.iter().all(|v| v.is_finite()),
            "pixel {i} ({}, {}) has a non-finite normal {n:?}",
            i % W,
            i / W
        );
    }

    // The four pixels that would have used the bad sample fall back to the
    // documented default instead of receiving a NaN.
    for i in POISONED {
        assert_eq!(
            normals[i],
            Vector3::new(0.0, 0.0, 1.0),
            "pixel {i} ({}, {}) next to the bad sample",
            i % W,
            i / W
        );
    }

    // The bad pixel itself is computed from its four good neighbours, so it
    // still sees the plane and faces the camera.
    assert!(
        normals[BAD].z < -0.99,
        "the bad pixel should still face the camera, got {:?}",
        normals[BAD]
    );

    // Every other interior pixel is untouched flat plane facing the camera.
    for (i, n) in normals.iter().enumerate() {
        let (px, py) = (i % W, i / W);
        let interior = px != 0 && px + 1 < W && py != 0 && py + 1 < H;
        if !interior || i == BAD || POISONED.contains(&i) {
            continue;
        }
        assert!(
            n.z < -0.99,
            "pixel {i} ({px}, {py}) should face the camera, got {n:?}"
        );
    }
}

#[test]
fn a_nan_depth_does_not_poison_neighbouring_normals() {
    assert_only_the_expected_pixels_were_affected(flat_depth_with(f32::NAN));
}

#[test]
fn an_infinite_depth_does_not_poison_neighbouring_normals() {
    assert_only_the_expected_pixels_were_affected(flat_depth_with(f32::INFINITY));
}

#[test]
fn a_negative_depth_still_yields_the_default_normal() {
    // Control for the existing contract: a non-positive depth was already
    // rejected and must stay that way.
    assert_only_the_expected_pixels_were_affected(flat_depth_with(-1.0));
}
