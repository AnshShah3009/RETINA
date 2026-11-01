//! The nearest-background label map must agree, pixel for pixel, with a
//! brute-force "scan every background pixel, keep the first on a tie" oracle.
//!
//! The label pass used to *be* that brute-force scan, so the oracle is a
//! transcription of the old code rather than a second opinion derived from the
//! new one. The replacement (chamfer propagation for L1/Chessboard, a separable
//! Euclidean transform over `(squared distance, label)` keys for L2) is
//! linear, and these tests pin down that optimisation changes nothing
//! observable - in particular the tie-break: the smallest flat index wins.

use cv_core::{CpuTensor, TensorShape};
use cv_imgproc::distance_transform::{distance_transform_with_labels, DistanceType};

fn make_binary(data: Vec<f32>, width: usize, height: usize) -> CpuTensor<f32> {
    CpuTensor::from_vec(data, TensorShape::new(1, height, width)).unwrap()
}

/// The pre-optimisation label scan: for every pixel, walk the background pixels
/// in row-major order and keep the first one that improves on a strict `<`.
fn brute_force_labels(src: &[f32], width: usize, height: usize, dt: DistanceType) -> Vec<i32> {
    let mut bg: Vec<(usize, usize, i32)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if src[y * width + x] == 0.0 {
                bg.push((x, y, (y * width + x) as i32));
            }
        }
    }
    if bg.is_empty() {
        return vec![-1; width * height];
    }
    (0..width * height)
        .map(|idx| {
            let (px, py) = (idx % width, idx / width);
            if src[idx] == 0.0 {
                return idx as i32;
            }
            let mut best = f64::MAX;
            let mut label = -1;
            for &(bx, by, bi) in &bg {
                let dx = px as f64 - bx as f64;
                let dy = py as f64 - by as f64;
                let d = match dt {
                    DistanceType::L1 => dx.abs() + dy.abs(),
                    DistanceType::Chessboard => dx.abs().max(dy.abs()),
                    DistanceType::L2 => dx * dx + dy * dy,
                };
                if d < best {
                    best = d;
                    label = bi;
                }
            }
            label
        })
        .collect()
}

const ALL: [DistanceType; 3] = [DistanceType::L1, DistanceType::Chessboard, DistanceType::L2];

fn check_against_oracle(data: Vec<f32>, width: usize, height: usize, what: &str) {
    let src = data.clone();
    let image = make_binary(data, width, height);
    for dt in ALL {
        let (_, labels) = distance_transform_with_labels(&image, dt).unwrap();
        let got = labels.as_slice().unwrap();
        let want = brute_force_labels(&src, width, height, dt);
        assert_eq!(
            got,
            want.as_slice(),
            "{what}: labels disagree with the brute-force oracle for {dt:?}"
        );
    }
}

/// A hand-built case with several background pixels and an exact tie: in a 7x7
/// image the zeros at (1,3) and (5,3) are both two steps from (3,3) under every
/// metric, while the other zeros are strictly farther. The midpoint must take
/// the left zero - the smaller flat index (22, not 26) - and not the globally
/// smallest label either, which is 0 at (0,0).
#[test]
fn tie_goes_to_the_smallest_flat_index() {
    let (w, h) = (7usize, 7usize);
    let mut data = vec![1.0f32; w * h];
    let zeros = [(1usize, 3usize), (5, 3), (0, 0), (6, 6)];
    for &(x, y) in &zeros {
        data[y * w + x] = 0.0;
    }
    let (left, right) = ((3 * w + 1) as i32, (3 * w + 5) as i32); // 22, 26
    let image = make_binary(data, w, h);

    for dt in ALL {
        let (_, labels) = distance_transform_with_labels(&image, dt).unwrap();
        let l = labels.as_slice().unwrap();
        for &(x, y) in &zeros {
            assert_eq!(
                l[y * w + x],
                (y * w + x) as i32,
                "{dt:?}: a background pixel labels itself"
            );
        }
        assert_eq!(
            l[3 * w + 3],
            left,
            "{dt:?}: the midpoint of the two inner zeros must tie to the lower index"
        );
        // Same for the pixel below the midpoint: only the two inner zeros are
        // at distance 3 there under all three metrics.
        assert_eq!(l[4 * w + 3], left, "{dt:?}: below the midpoint");
        // And to the right of the midpoint the other zero wins, so the scan is
        // not merely "always pick the smallest index".
        assert_eq!(l[3 * w + 4], right, "{dt:?}: right of the midpoint");
    }
}

/// A vertical tie whose winner is the *second* background pixel in row-major
/// order among the two nearest, and which is neither the first background pixel
/// in the image nor the last: zeros at (3,1) and (3,5) bracket (3,3), with a
/// farther zero at (0,0). Keeping the last-improving candidate would yield 38
/// and keeping the smallest index would yield 0; the answer is 10.
#[test]
fn vertical_tie_between_two_background_pixels() {
    let (w, h) = (7usize, 7usize);
    let mut data = vec![1.0f32; w * h];
    for &(x, y) in &[(3usize, 1usize), (3, 5), (0, 0)] {
        data[y * w + x] = 0.0;
    }
    let image = make_binary(data, w, h);

    for dt in ALL {
        let (_, labels) = distance_transform_with_labels(&image, dt).unwrap();
        let l = labels.as_slice().unwrap();
        assert_eq!(
            l[3 * w + 3],
            10,
            "{dt:?}: the tie at (3,3) belongs to the smaller of the two nearest indices"
        );
    }
}

/// Several background pixels, no ties: a 7x5 image with a scattering of zeros
/// and a deterministic foreground pattern.
#[test]
fn several_background_pixels_match_oracle() {
    let (w, h) = (7usize, 5usize);
    let mut data = vec![1.0f32; w * h];
    for (x, y) in [(0, 0), (3, 1), (6, 4), (1, 3)] {
        data[y * w + x] = 0.0;
    }
    check_against_oracle(data, w, h, "several background pixels");
}

/// Randomised images, including degenerate sizes (a single row / column, a
/// 1x1 image, all-background and all-foreground) where the raster scans have
/// no neighbours to trade with.
#[test]
fn randomised_images_match_oracle() {
    let mut state = 0x1234_5678u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };

    let sizes = [
        (1usize, 1usize),
        (1, 9),
        (9, 1),
        (2, 2),
        (5, 4),
        (11, 7),
        (13, 13),
        (23, 19),
        (29, 17),
    ];
    for &(w, h) in &sizes {
        for density in [0u32, 1, 3, 7] {
            // density 0 => no background at all, density 7 => mostly background.
            let mut data = vec![1.0f32; w * h];
            for v in data.iter_mut() {
                if next() % 8 < density {
                    *v = 0.0;
                }
            }
            check_against_oracle(data, w, h, &format!("random {w}x{h} density {density}"));
        }
    }
}

/// Tie-dense patterns: a lattice of zeros puts whole rows and columns exactly
/// equidistant from two background pixels, so the tie-break decides most of
/// the image. These are the cases a raster scan is most likely to get wrong.
#[test]
fn tie_dense_lattices_match_oracle() {
    let cases = [
        (37usize, 23usize, 4usize, 2usize),
        (31, 31, 3, 3),
        (40, 17, 1, 5),
        (17, 40, 6, 1),
        (33, 29, 2, 2),
        (64, 64, 8, 8),
    ];
    for &(w, h, px, py) in &cases {
        let mut data = vec![1.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if x % px == 0 || y % py == 0 {
                    data[y * w + x] = 0.0;
                }
            }
        }
        check_against_oracle(data, w, h, &format!("lattice {w}x{h} step {px},{py}"));
    }
}

/// The old label scan ran in parallel over pixels but was a pure function of
/// them; the new one is sequential. Nothing else about the shape of the output
/// may change, so the distance map must still line up with the labels.
#[test]
fn labels_agree_with_distance_map_on_a_larger_image() {
    let (w, h) = (64usize, 48usize);
    let mut data = vec![1.0f32; w * h];
    for y in (3..h).step_by(11) {
        for x in (5..w).step_by(13) {
            data[y * w + x] = 0.0;
        }
    }
    let image = make_binary(data.clone(), w, h);
    for dt in ALL {
        let (dist, labels) = distance_transform_with_labels(&image, dt).unwrap();
        let d = dist.as_slice().unwrap();
        let l = labels.as_slice().unwrap();
        for idx in 0..w * h {
            let target = l[idx] as usize;
            assert!(target < w * h, "{dt:?}: label {idx} out of range");
            let (px, py) = ((idx % w) as f64, (idx / w) as f64);
            let (bx, by) = ((target % w) as f64, (target / w) as f64);
            let expected = match dt {
                DistanceType::L1 => (px - bx).abs() + (py - by).abs(),
                DistanceType::Chessboard => (px - bx).abs().max((py - by).abs()),
                DistanceType::L2 => ((px - bx).powi(2) + (py - by).powi(2)).sqrt(),
            };
            assert!(
                (d[idx] as f64 - expected).abs() < 1e-4,
                "{dt:?}: pixel {idx} distance {} but its label {target} is at {expected}",
                d[idx]
            );
        }
        let want = brute_force_labels(&data, w, h, dt);
        assert_eq!(l, want.as_slice(), "{dt:?}: 64x48 labels");
    }
}
