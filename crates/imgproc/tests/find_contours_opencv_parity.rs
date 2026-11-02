//! `find_contours_with` against `cv2.findContours` (OpenCV 4.13) in every
//! retrieval mode (`RETR_LIST`, `RETR_EXTERNAL`, `RETR_CCOMP`, `RETR_TREE`)
//! and both chain approximations (`CHAIN_APPROX_NONE`, `CHAIN_APPROX_SIMPLE`).
//!
//! The previous implementation only started a hole border on a pixel no
//! outer border had visited, so the inner border of a ring one or two pixels
//! thick was never traced; it also returned contours in raster order, had no
//! hierarchy and no modes. OpenCV 4.13 (`contours_new.cpp`) pads the image
//! with a background frame, so borders on the image edge are traced too;
//! RETR_LIST lists contours in the reverse of the order the raster scan met
//! them, and the tree modes list the hierarchy in pre-order.
//!
//! Each case image is built here by a generator that the reference script
//! reproduces exactly (an LCG for the random ones). The reference value for
//! each (case, mode) is a 32-bit FNV-1a digest of OpenCV's contours (count,
//! then each contour's length and points) followed by its hierarchy, so a
//! match means identical contours, in identical order, with identical start
//! points, directions and hierarchy.

use cv_core::{CpuTensor, TensorShape};
use cv_imgproc::{
    ContourApproximation, ContourHierarchy, ContourRetrieval, ContourWithHierarchy, find_contours,
    find_contours_with,
};

const MODES: [(ContourRetrieval, ContourApproximation); 8] = [
    (ContourRetrieval::List, ContourApproximation::None),
    (ContourRetrieval::List, ContourApproximation::Simple),
    (ContourRetrieval::External, ContourApproximation::None),
    (ContourRetrieval::External, ContourApproximation::Simple),
    (ContourRetrieval::CComp, ContourApproximation::None),
    (ContourRetrieval::CComp, ContourApproximation::Simple),
    (ContourRetrieval::Tree, ContourApproximation::None),
    (ContourRetrieval::Tree, ContourApproximation::Simple),
];

enum Gen {
    /// Each pixel foreground when the next LCG byte is below `density`.
    Noise {
        w: usize,
        h: usize,
        seed: u32,
        density: u8,
    },
    /// `Noise` on a grid of `cell` x `cell` blocks.
    Blocks {
        w: usize,
        h: usize,
        seed: u32,
        cell: usize,
        density: u8,
    },
    /// Foreground where at least `thr` pixels of the 3x3 neighbourhood of
    /// `Noise` are.
    Smooth {
        w: usize,
        h: usize,
        seed: u32,
        density: u8,
        thr: u32,
    },
    /// Rectangles `[x0, x1) x [y0, y1)` filled with a value, in order.
    Rects {
        w: usize,
        h: usize,
        ops: &'static [(usize, usize, usize, usize, u8)],
    },
    /// An `n` x `n` grid of 3 x 3 one-pixel rings, `pitch` apart, inside a
    /// one-pixel frame ring.
    RingGrid { n: usize, pitch: usize },
}

fn lcg(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn noise(w: usize, h: usize, seed: u32, density: u8) -> Vec<u8> {
    let mut s = seed;
    (0..w * h)
        .map(|_| {
            s = lcg(s);
            if ((s >> 24) as u8) < density { 255 } else { 0 }
        })
        .collect()
}

fn fill(img: &mut [u8], w: usize, ops: &[(usize, usize, usize, usize, u8)]) {
    for &(x0, y0, x1, y1, v) in ops {
        for y in y0..y1 {
            img[y * w + x0..y * w + x1].fill(v);
        }
    }
}

fn build(g: &Gen) -> (Vec<u8>, usize, usize) {
    match *g {
        Gen::Noise {
            w,
            h,
            seed,
            density,
        } => (noise(w, h, seed, density), w, h),
        Gen::Blocks {
            w,
            h,
            seed,
            cell,
            density,
        } => {
            let gw = w.div_ceil(cell);
            let grid = noise(gw, h.div_ceil(cell), seed, density);
            let img = (0..w * h)
                .map(|i| grid[(i / w / cell) * gw + (i % w) / cell])
                .collect();
            (img, w, h)
        }
        Gen::Smooth {
            w,
            h,
            seed,
            density,
            thr,
        } => {
            let n = noise(w, h, seed, density);
            let mut img = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut c = 0;
                    for yy in y.saturating_sub(1)..(y + 2).min(h) {
                        for xx in x.saturating_sub(1)..(x + 2).min(w) {
                            c += u32::from(n[yy * w + xx] > 0);
                        }
                    }
                    img[y * w + x] = if c >= thr { 255 } else { 0 };
                }
            }
            (img, w, h)
        }
        Gen::Rects { w, h, ops } => {
            let mut img = vec![0u8; w * h];
            fill(&mut img, w, ops);
            (img, w, h)
        }
        Gen::RingGrid { n, pitch } => {
            let w = n * pitch + 4;
            let mut img = vec![0u8; w * w];
            let ring =
                |x0, y0, x1, y1| [(x0, y0, x1, y1, 255), (x0 + 1, y0 + 1, x1 - 1, y1 - 1, 0)];
            fill(&mut img, w, &ring(0, 0, w, w));
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = (2 + i * pitch, 2 + j * pitch);
                    fill(&mut img, w, &ring(x, y, x + pitch - 1, y + pitch - 1));
                }
            }
            (img, w, w)
        }
    }
}

fn tensor(img: Vec<u8>, w: usize, h: usize) -> CpuTensor<u8> {
    CpuTensor::from_vec(img, TensorShape::new(1, h, w)).unwrap()
}

fn digest(contours: &[ContourWithHierarchy], hierarchy: &[ContourHierarchy]) -> u32 {
    let mut h = 0xcbf29ce484222325u64;
    let mut feed = |v: i64| {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    };
    feed(contours.len() as i64);
    for c in contours {
        feed(c.points.len() as i64);
        for &(x, y) in &c.points {
            feed(i64::from(x));
            feed(i64::from(y));
        }
    }
    for e in hierarchy {
        for &v in e {
            feed(i64::from(v));
        }
    }
    (h ^ (h >> 32)) as u32
}

#[test]
fn all_modes_match_opencv() {
    let mut failures = Vec::new();
    for (name, g, expected) in CASES {
        let (img, w, h) = build(g);
        let t = tensor(img, w, h);
        for (k, &(mode, method)) in MODES.iter().enumerate() {
            let (c, hcy) = find_contours_with(&t, mode, method).unwrap();
            assert_eq!(c.len(), hcy.len());
            if digest(&c, &hcy) != expected[k] {
                failures.push(format!("{name} {mode:?}/{method:?}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} (case, mode) pairs differ from cv2: {failures:?}",
        failures.len(),
        CASES.len() * MODES.len()
    );
}

/// `is_hole` agrees with the depth of the contour in the RETR_TREE
/// hierarchy (holes at odd depths), and the RETR_LIST contours are the
/// same borders.
#[test]
fn hole_flags_follow_the_tree() {
    for (name, g, _) in CASES {
        let (img, w, h) = build(g);
        let t = tensor(img, w, h);
        let (tree, hcy) =
            find_contours_with(&t, ContourRetrieval::Tree, ContourApproximation::None).unwrap();
        for (i, c) in tree.iter().enumerate() {
            let mut depth = 0;
            let mut p = hcy[i][3];
            while p >= 0 {
                depth += 1;
                p = hcy[p as usize][3];
            }
            assert_eq!(c.is_hole, depth % 2 == 1, "{name}: contour {i}");
        }
        let mut list = find_contours(&t).unwrap();
        let mut tree = tree;
        list.sort_by(|a, b| a.points.cmp(&b.points));
        tree.sort_by(|a, b| a.points.cmp(&b.points));
        assert_eq!(list, tree, "{name}");
    }
}

/// A ring one pixel thick, written out: cv2.findContours(RETR_TREE,
/// CHAIN_APPROX_SIMPLE) gives the outer border from its top-left pixel,
/// counter-clockwise on screen, and the hole border from the pixel left of
/// the hole's top-left pixel, clockwise, skipping the ring's corners.
#[test]
fn thin_ring_written_out() {
    let (img, w, h) = build(&CASES[0].1);
    let t = tensor(img, w, h);
    let (c, hcy) =
        find_contours_with(&t, ContourRetrieval::Tree, ContourApproximation::Simple).unwrap();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].points, [(2, 2), (2, 10), (11, 10), (11, 2)]);
    assert!(!c[0].is_hole);
    assert_eq!(
        c[1].points,
        [
            (2, 3),
            (3, 2),
            (10, 2),
            (11, 3),
            (11, 9),
            (10, 10),
            (3, 10),
            (2, 9)
        ]
    );
    assert!(c[1].is_hole);
    assert_eq!(hcy, [[-1, -1, 1, -1], [-1, -1, -1, 0]]);

    // RETR_LIST lists the hole first; CHAIN_APPROX_NONE keeps every pixel.
    let list = find_contours(&t).unwrap();
    assert_eq!(list.len(), 2);
    assert!(list[0].is_hole && !list[1].is_hole);
    assert_eq!((list[0].points.len(), list[0].points[0]), (30, (2, 3)));
    assert_eq!((list[1].points.len(), list[1].points[0]), (34, (2, 2)));
}

/// Pixels on the image edge are foreground like any other (OpenCV pads the
/// image with background).
#[test]
fn full_image_is_one_border_along_the_edge() {
    let t = tensor(vec![1u8; 4 * 3], 4, 3);
    let c = find_contours(&t).unwrap();
    assert_eq!(c.len(), 1);
    assert_eq!(
        c[0].points,
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 1),
            (3, 0),
            (2, 0),
            (1, 0)
        ]
    );
}

/// Only values above zero are foreground, as OpenCV's threshold at zero.
#[test]
fn float_input_thresholds_at_zero() {
    let mut data = vec![-1.0f32; 5 * 5];
    data[2 * 5 + 2] = 0.5;
    let t = CpuTensor::from_vec(data, TensorShape::new(1, 5, 5)).unwrap();
    let c = find_contours(&t).unwrap();
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].points, [(2, 2)]);
}

// Generated by cv2 4.13.0: for each case, the digests of
// cv2.findContours(img, mode, method) for the eight entries of MODES.
#[rustfmt::skip]
const CASES: &[(&str, Gen, [u32; 8])] = &[
    ("ring_t1", Gen::Rects { w: 14, h: 13, ops: &[(2, 2, 12, 11, 255), (3, 3, 11, 10, 0)] }, [0x7c53caf2, 0xd81a5cd6, 0xb1cedc8a, 0x2e00e580, 0xea5ab5e0, 0xfb5d8ca8, 0xea5ab5e0, 0xfb5d8ca8]),
    ("ring_t2", Gen::Rects { w: 14, h: 13, ops: &[(2, 2, 12, 11, 255), (4, 4, 10, 9, 0)] }, [0xf365de60, 0x3c66952d, 0xb1cedc8a, 0x2e00e580, 0x1ed95349, 0x15002a2f, 0x1ed95349, 0x15002a2f]),
    ("ring_t3", Gen::Rects { w: 14, h: 13, ops: &[(2, 2, 12, 11, 255), (5, 5, 9, 8, 0)] }, [0x0600f974, 0x2fe3817f, 0xb1cedc8a, 0x2e00e580, 0xa4db3f7f, 0xbb536e33, 0xa4db3f7f, 0xbb536e33]),
    ("frame_ring_t1", Gen::Rects { w: 10, h: 9, ops: &[(0, 0, 10, 9, 255), (1, 1, 9, 8, 0)] }, [0x2695ed9f, 0x4d591755, 0xe8e6516c, 0x61d72be0, 0x1adb8241, 0xb86950c0, 0x1adb8241, 0xb86950c0]),
    ("frame_ring_t2", Gen::Rects { w: 10, h: 9, ops: &[(0, 0, 10, 9, 255), (2, 2, 8, 7, 0)] }, [0x58a76fde, 0x75d341be, 0xe8e6516c, 0x61d72be0, 0x7a79ec05, 0xcbc88be0, 0x7a79ec05, 0xcbc88be0]),
    ("nested_three_levels", Gen::Rects { w: 20, h: 20, ops: &[(1, 1, 19, 19, 255), (3, 3, 17, 17, 0), (5, 5, 15, 15, 255), (6, 6, 14, 14, 0), (8, 8, 12, 12, 255)] }, [0xd0609259, 0xaf058326, 0x3d6a6466, 0x5277dfdf, 0xd03b7a80, 0xe4b4a541, 0x93cf8f39, 0xa1dfb147]),
    ("nested_hole_in_hole", Gen::Rects { w: 24, h: 22, ops: &[(1, 1, 23, 21, 255), (3, 3, 21, 19, 0), (5, 5, 19, 17, 255), (7, 7, 10, 10, 0), (12, 8, 17, 14, 0), (14, 10, 15, 11, 255)] }, [0xf36c7290, 0x651303fd, 0x91812d9b, 0x60aa01b4, 0x6371e37d, 0x33c2edd9, 0xa8ba0a60, 0x42382d24]),
    ("rings_touching_diagonally", Gen::Rects { w: 16, h: 16, ops: &[(1, 1, 8, 8, 255), (2, 2, 7, 7, 0), (8, 8, 15, 15, 255), (9, 9, 14, 14, 0)] }, [0xe2ddb8b0, 0x08ca5c64, 0x095b4a66, 0x42c88a63, 0xb77f3e6e, 0x2b705465, 0xb77f3e6e, 0x2b705465]),
    ("rings_sharing_an_edge", Gen::Rects { w: 18, h: 10, ops: &[(1, 1, 10, 9, 255), (2, 2, 9, 8, 0), (9, 1, 17, 9, 255), (10, 2, 16, 8, 0)] }, [0x7de0a608, 0x0d0a0e7b, 0x1bf778d4, 0x080b26b8, 0x907159a9, 0xcb34fc8a, 0x907159a9, 0xcb34fc8a]),
    ("full_image", Gen::Rects { w: 7, h: 5, ops: &[(0, 0, 7, 5, 255)] }, [0x91a8ae38, 0x1072ad31, 0x91a8ae38, 0x1072ad31, 0x91a8ae38, 0x1072ad31, 0x91a8ae38, 0x1072ad31]),
    ("empty_image", Gen::Rects { w: 7, h: 5, ops: &[] }, [0x29620a93, 0x29620a93, 0x29620a93, 0x29620a93, 0x29620a93, 0x29620a93, 0x29620a93, 0x29620a93]),
    ("single_pixel", Gen::Rects { w: 5, h: 5, ops: &[(2, 2, 3, 3, 255)] }, [0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3, 0xfa24b4a3]),
    ("corner_pixels", Gen::Rects { w: 6, h: 4, ops: &[(0, 0, 1, 1, 255), (5, 0, 6, 1, 255), (0, 3, 1, 4, 255), (5, 3, 6, 4, 255)] }, [0xea41099b, 0xea41099b, 0xea41099b, 0xea41099b, 0xea41099b, 0xea41099b, 0xea41099b, 0xea41099b]),
    ("edge_lines", Gen::Rects { w: 9, h: 7, ops: &[(0, 0, 9, 1, 255), (0, 2, 1, 7, 255), (8, 3, 9, 7, 255), (3, 6, 6, 7, 255)] }, [0x8390818e, 0x370379ce, 0x8390818e, 0x370379ce, 0x8390818e, 0x370379ce, 0x8390818e, 0x370379ce]),
    ("row_image", Gen::Rects { w: 9, h: 1, ops: &[(1, 0, 4, 1, 255), (5, 0, 9, 1, 255)] }, [0xecd731d7, 0x46696ff0, 0xecd731d7, 0x46696ff0, 0xecd731d7, 0x46696ff0, 0xecd731d7, 0x46696ff0]),
    ("column_image", Gen::Rects { w: 1, h: 8, ops: &[(0, 0, 1, 3, 255), (0, 4, 1, 5, 255)] }, [0x1fe9eb9c, 0x3f0632ee, 0x1fe9eb9c, 0x3f0632ee, 0x1fe9eb9c, 0x3f0632ee, 0x1fe9eb9c, 0x3f0632ee]),
    ("one_by_one", Gen::Rects { w: 1, h: 1, ops: &[(0, 0, 1, 1, 255)] }, [0x263b2360, 0x263b2360, 0x263b2360, 0x263b2360, 0x263b2360, 0x263b2360, 0x263b2360, 0x263b2360]),
    ("pixel_checkerboard", Gen::Rects { w: 8, h: 8, ops: &[(1, 1, 2, 2, 255), (3, 1, 4, 2, 255), (5, 1, 6, 2, 255), (2, 2, 3, 3, 255), (4, 2, 5, 3, 255), (6, 2, 7, 3, 255), (1, 3, 2, 4, 255), (3, 3, 4, 4, 255), (5, 3, 6, 4, 255), (2, 4, 3, 5, 255), (4, 4, 5, 5, 255), (6, 4, 7, 5, 255), (1, 5, 2, 6, 255), (3, 5, 4, 6, 255), (5, 5, 6, 6, 255), (2, 6, 3, 7, 255), (4, 6, 5, 7, 255), (6, 6, 7, 7, 255)] }, [0xa4ab56f7, 0xa4ab56f7, 0x1a9c5c58, 0x1a9c5c58, 0x8177febf, 0x8177febf, 0x8177febf, 0x8177febf]),
    ("c_shape", Gen::Rects { w: 12, h: 12, ops: &[(1, 1, 11, 11, 255), (3, 3, 9, 9, 0), (9, 5, 11, 7, 0)] }, [0x617ef585, 0x66bd196d, 0x617ef585, 0x66bd196d, 0x617ef585, 0x66bd196d, 0x617ef585, 0x66bd196d]),
    ("thick_ring_with_islands", Gen::Rects { w: 22, h: 18, ops: &[(0, 0, 22, 18, 255), (3, 3, 19, 15, 0), (6, 6, 7, 7, 255), (9, 5, 13, 9, 255), (10, 6, 12, 8, 0), (15, 10, 18, 13, 255)] }, [0xd1a871ff, 0x69d0e512, 0xfb9c0983, 0x3c11fbec, 0x2e453871, 0x04fb2c26, 0xe95aa414, 0xa3d09d25]),
    ("ring_grid_8", Gen::RingGrid { n: 8, pitch: 4 }, [0x57347b33, 0xaf93c093, 0xf0b6c9c2, 0xa7d8255e, 0x4111c5c9, 0x0ab4ae8a, 0x2a1d7360, 0xdab4c5fe]),
    ("ring_grid_12", Gen::RingGrid { n: 12, pitch: 4 }, [0xb91c6e46, 0x73334c10, 0xed2a9381, 0xabb14624, 0xa2ceeb30, 0x39a20e1f, 0x151da5cb, 0x19841f3c]),
    ("noise_0", Gen::Noise { w: 9, h: 29, seed: 342332860, density: 40 }, [0xc708bd10, 0x39eee3c5, 0x1cb95f90, 0x9de769ef, 0x55400e7e, 0x6bab27ab, 0x55400e7e, 0x6bab27ab]),
    ("noise_1", Gen::Noise { w: 31, h: 7, seed: 435844777, density: 170 }, [0xabf5d8fa, 0xa269aefc, 0x15aef0ed, 0xa4f10731, 0x59c0b2cc, 0x14ff0164, 0x59c0b2cc, 0x14ff0164]),
    ("noise_2", Gen::Noise { w: 38, h: 34, seed: 413979305, density: 220 }, [0x4d67b12a, 0x7e63dc90, 0xb889c994, 0x200a688c, 0x4a8b222d, 0x785b6f32, 0x4a8b222d, 0x785b6f32]),
    ("noise_3", Gen::Noise { w: 16, h: 40, seed: 875031525, density: 220 }, [0x42581034, 0x6ebf8964, 0x24b1454e, 0x46e935ef, 0x70d30662, 0xcbaf32a0, 0x70d30662, 0xcbaf32a0]),
    ("noise_4", Gen::Noise { w: 33, h: 25, seed: 427906332, density: 220 }, [0x92cea7a9, 0x4c8c2e72, 0xde4c16f5, 0xe768fa9d, 0x3d030978, 0xa96542c6, 0x3d030978, 0xa96542c6]),
    ("noise_5", Gen::Noise { w: 37, h: 8, seed: 185369214, density: 40 }, [0xf1a1a605, 0x4ae032dd, 0xf1a1a605, 0x4ae032dd, 0xf1a1a605, 0x4ae032dd, 0xf1a1a605, 0x4ae032dd]),
    ("noise_6", Gen::Noise { w: 9, h: 38, seed: 8154533, density: 40 }, [0xfc3b9b01, 0x8553a8a9, 0xfc3b9b01, 0x8553a8a9, 0xfc3b9b01, 0x8553a8a9, 0xfc3b9b01, 0x8553a8a9]),
    ("noise_7", Gen::Noise { w: 34, h: 27, seed: 149863985, density: 170 }, [0x422baa52, 0xf5ec429b, 0xd85996ff, 0xcd5dced5, 0x3ecee19c, 0x905739d4, 0x3ecee19c, 0x905739d4]),
    ("noise_8", Gen::Noise { w: 12, h: 19, seed: 212715300, density: 128 }, [0xc7a94ca9, 0x1f3c08a5, 0x098a21c4, 0xd14f9092, 0x795ea944, 0xd09a6109, 0x795ea944, 0xd09a6109]),
    ("noise_9", Gen::Noise { w: 38, h: 39, seed: 936995932, density: 40 }, [0x3f4d9d55, 0xa71e636a, 0x3fee5017, 0xdc97bbc8, 0xb89d87cf, 0xd712a845, 0xb89d87cf, 0xd712a845]),
    ("noise_10", Gen::Noise { w: 23, h: 15, seed: 582814744, density: 128 }, [0xa1863ec4, 0xa38f0800, 0xef546482, 0x3cd444e8, 0xa6370c34, 0x33b8c621, 0xa6370c34, 0x33b8c621]),
    ("noise_11", Gen::Noise { w: 12, h: 35, seed: 49533191, density: 170 }, [0x9010bb47, 0xb81d422c, 0x6daf6717, 0x82c710de, 0x15b26c34, 0xc69661cc, 0x15b26c34, 0xc69661cc]),
    ("noise_12", Gen::Noise { w: 15, h: 27, seed: 736281784, density: 220 }, [0x565f89bb, 0xac55a758, 0x96c32772, 0x2f1ec78c, 0x6ac0f6a9, 0xa766f62d, 0x6ac0f6a9, 0xa766f62d]),
    ("noise_13", Gen::Noise { w: 32, h: 13, seed: 869389208, density: 40 }, [0x79e5a509, 0x25e1047a, 0x79e5a509, 0x25e1047a, 0x79e5a509, 0x25e1047a, 0x79e5a509, 0x25e1047a]),
    ("noise_14", Gen::Noise { w: 19, h: 33, seed: 899652497, density: 128 }, [0xe05b335f, 0x3f21bd20, 0x0073fc93, 0x32dac984, 0x83d0db11, 0xc6dca72b, 0x83d0db11, 0xc6dca72b]),
    ("noise_15", Gen::Noise { w: 21, h: 5, seed: 575279556, density: 128 }, [0x61a25f70, 0x861719a3, 0x267e9883, 0x1ff6c22e, 0x8228f58b, 0xe32380aa, 0x8228f58b, 0xe32380aa]),
    ("noise_16", Gen::Noise { w: 10, h: 15, seed: 563305163, density: 90 }, [0xf5276473, 0xc7779c3c, 0xf761a85b, 0x147ca720, 0x7072ac9d, 0xb433fa1b, 0x7072ac9d, 0xb433fa1b]),
    ("noise_17", Gen::Noise { w: 21, h: 34, seed: 1069095910, density: 220 }, [0x1c3f3c87, 0x3e0dbc8f, 0xbc1b3113, 0x9f3f2ce3, 0x00787e7b, 0x1bfa41d4, 0x00787e7b, 0x1bfa41d4]),
    ("noise_18", Gen::Noise { w: 36, h: 9, seed: 197384471, density: 40 }, [0xd8c2002d, 0xd8c2002d, 0xd8c2002d, 0xd8c2002d, 0xd8c2002d, 0xd8c2002d, 0xd8c2002d, 0xd8c2002d]),
    ("noise_19", Gen::Noise { w: 13, h: 19, seed: 391251663, density: 90 }, [0x31bc6ec0, 0xdaa126cb, 0x31bc6ec0, 0xdaa126cb, 0x31bc6ec0, 0xdaa126cb, 0x31bc6ec0, 0xdaa126cb]),
    ("noise_20", Gen::Noise { w: 26, h: 16, seed: 344396989, density: 170 }, [0xfb7adc95, 0x614c3622, 0x003d8713, 0xd395e703, 0x8e2a0987, 0x450ac71f, 0x8e2a0987, 0x450ac71f]),
    ("noise_21", Gen::Noise { w: 29, h: 7, seed: 798890353, density: 128 }, [0x37e3700c, 0xe3ec1659, 0x3f08d2a1, 0x841f548c, 0x845ab875, 0xfd998b9b, 0x845ab875, 0xfd998b9b]),
    ("noise_22", Gen::Noise { w: 13, h: 23, seed: 891021917, density: 220 }, [0xfaf66776, 0x04da9c11, 0xff9cba13, 0x65e582b9, 0x6adca7de, 0x6224e3fd, 0x6adca7de, 0x6224e3fd]),
    ("noise_23", Gen::Noise { w: 23, h: 35, seed: 930801649, density: 128 }, [0x6397692d, 0x56ad9f14, 0x2ef7c0ad, 0xdad4f31a, 0x1796a6cc, 0xe343be28, 0x1796a6cc, 0xe343be28]),
    ("noise_24", Gen::Noise { w: 32, h: 40, seed: 1040413899, density: 170 }, [0xde285f6b, 0xa04f82ba, 0x7d6a14de, 0x4f74f185, 0xd763ee6e, 0xd832df13, 0xd763ee6e, 0xd832df13]),
    ("noise_25", Gen::Noise { w: 34, h: 7, seed: 102775925, density: 128 }, [0xd7e32d2b, 0x1b020450, 0x8d8d6217, 0x33cda5ca, 0x95b9f31a, 0x0c8171e2, 0x95b9f31a, 0x0c8171e2]),
    ("noise_26", Gen::Noise { w: 30, h: 9, seed: 164897375, density: 220 }, [0x9dc8ab0e, 0xc0b64c33, 0x79d20d18, 0x46c3a480, 0x9f8a66cd, 0x64891509, 0x9f8a66cd, 0x64891509]),
    ("noise_27", Gen::Noise { w: 6, h: 35, seed: 420991602, density: 128 }, [0x4eb78d90, 0xb0e44acb, 0xdfec8c23, 0x1c15eefa, 0xb96984e5, 0xd02c8d13, 0xb96984e5, 0xd02c8d13]),
    ("noise_28", Gen::Noise { w: 26, h: 29, seed: 267128312, density: 170 }, [0x0d8e6396, 0x959f038f, 0xd6130150, 0x3303e138, 0xbaccf31a, 0xf35f9fe1, 0xbaccf31a, 0xf35f9fe1]),
    ("noise_29", Gen::Noise { w: 29, h: 19, seed: 52113190, density: 40 }, [0xe7bb16f9, 0x9039cace, 0xe7bb16f9, 0x9039cace, 0xe7bb16f9, 0x9039cace, 0xe7bb16f9, 0x9039cace]),
    ("noise_30", Gen::Noise { w: 16, h: 5, seed: 685964260, density: 220 }, [0xc0994b67, 0x1b52f799, 0x18e4d430, 0x8b98446f, 0x5b232fec, 0x5f54d699, 0x5b232fec, 0x5f54d699]),
    ("noise_31", Gen::Noise { w: 27, h: 33, seed: 762192170, density: 128 }, [0xe3e326cb, 0x5da144d1, 0x69986e18, 0x642cb0b7, 0x4e9343d7, 0xcc9b31e1, 0x4e9343d7, 0xcc9b31e1]),
    ("noise_32", Gen::Noise { w: 29, h: 37, seed: 446359966, density: 128 }, [0xae12ca8c, 0x7ebf3229, 0xe6fb582b, 0x8297eb14, 0xf4ff7e7e, 0x97e6938f, 0x32004370, 0xeda00e07]),
    ("noise_33", Gen::Noise { w: 36, h: 11, seed: 924633189, density: 220 }, [0x73fe3805, 0xab0e12b2, 0xf04a0217, 0x725c2208, 0x99640148, 0xebe1d8bb, 0x99640148, 0xebe1d8bb]),
    ("noise_34", Gen::Noise { w: 24, h: 18, seed: 431741715, density: 220 }, [0x6fc6c871, 0x8cff7f8f, 0x1060cd3d, 0x03cc9087, 0xb3a429bf, 0xdde35c2e, 0xb3a429bf, 0xdde35c2e]),
    ("noise_35", Gen::Noise { w: 30, h: 38, seed: 591711884, density: 128 }, [0x47176172, 0x4d5bcdf7, 0xc2a9b9bd, 0xfc0a50b7, 0xa8c108ab, 0xad4bee49, 0xf60d3b82, 0xbf556dcd]),
    ("noise_36", Gen::Noise { w: 24, h: 24, seed: 233868861, density: 128 }, [0x282f41d3, 0x4f0f18f9, 0x08c31a35, 0xeb4135ec, 0x608e6c84, 0xee2c74f5, 0xc4319010, 0x4368c6ea]),
    ("noise_37", Gen::Noise { w: 14, h: 26, seed: 488325081, density: 90 }, [0xd64a02a3, 0xc3e0f02f, 0x59976dc8, 0x81988910, 0x299e1515, 0xac67f055, 0x299e1515, 0xac67f055]),
    ("noise_38", Gen::Noise { w: 11, h: 34, seed: 386460754, density: 40 }, [0xc76f6d03, 0x0b0dba7e, 0xc76f6d03, 0x0b0dba7e, 0xc76f6d03, 0x0b0dba7e, 0xc76f6d03, 0x0b0dba7e]),
    ("noise_39", Gen::Noise { w: 33, h: 27, seed: 112975330, density: 128 }, [0x4abe797f, 0x7489c459, 0x7d7df8e7, 0xe4854c56, 0x5dd14ef9, 0x7cd6cb42, 0xa5c19958, 0x7e132c4e]),
    ("blocks_0", Gen::Blocks { w: 37, h: 29, seed: 693426301, cell: 5, density: 128 }, [0x1e660bab, 0xf5d1fb31, 0x1e660bab, 0xf5d1fb31, 0x1e660bab, 0xf5d1fb31, 0x1e660bab, 0xf5d1fb31]),
    ("blocks_1", Gen::Blocks { w: 10, h: 12, seed: 12865424, cell: 6, density: 128 }, [0x7dcc8839, 0xd292c27a, 0x7dcc8839, 0xd292c27a, 0x7dcc8839, 0xd292c27a, 0x7dcc8839, 0xd292c27a]),
    ("blocks_2", Gen::Blocks { w: 56, h: 58, seed: 357750828, cell: 6, density: 80 }, [0xcf06cdad, 0x58615656, 0xcf06cdad, 0x58615656, 0xcf06cdad, 0x58615656, 0xcf06cdad, 0x58615656]),
    ("blocks_3", Gen::Blocks { w: 60, h: 17, seed: 699637872, cell: 4, density: 160 }, [0xbaca35a3, 0x2dc68f3a, 0x7e4ce32a, 0x9552edda, 0x3e338260, 0x2cd1be79, 0x3e338260, 0x2cd1be79]),
    ("blocks_4", Gen::Blocks { w: 36, h: 17, seed: 757326140, cell: 4, density: 128 }, [0x096e2100, 0xa2a17fd6, 0xb3108bc2, 0x1a1f4ac6, 0xddfa3242, 0x32292f91, 0xddfa3242, 0x32292f91]),
    ("blocks_5", Gen::Blocks { w: 53, h: 36, seed: 461974918, cell: 5, density: 128 }, [0x80caa670, 0xc437b6e0, 0x80caa670, 0xc437b6e0, 0x80caa670, 0xc437b6e0, 0x80caa670, 0xc437b6e0]),
    ("blocks_6", Gen::Blocks { w: 54, h: 53, seed: 887915712, cell: 5, density: 80 }, [0x2cbef93d, 0x045cb922, 0x37bf3479, 0xe24355e4, 0x67592e62, 0x63505943, 0x67592e62, 0x63505943]),
    ("blocks_7", Gen::Blocks { w: 53, h: 56, seed: 908289249, cell: 2, density: 128 }, [0x319ed1c8, 0x6012f5d5, 0x5ac5726c, 0x7d7670cd, 0xea549c39, 0x3138c435, 0xea549c39, 0x3138c435]),
    ("blocks_8", Gen::Blocks { w: 59, h: 20, seed: 936357639, cell: 6, density: 160 }, [0xcfa387a4, 0x125c7f68, 0xcfa387a4, 0x125c7f68, 0xcfa387a4, 0x125c7f68, 0xcfa387a4, 0x125c7f68]),
    ("blocks_9", Gen::Blocks { w: 54, h: 15, seed: 919754421, cell: 3, density: 80 }, [0x88a9cffb, 0x25e3af88, 0x88a9cffb, 0x25e3af88, 0x88a9cffb, 0x25e3af88, 0x88a9cffb, 0x25e3af88]),
    ("blocks_10", Gen::Blocks { w: 10, h: 34, seed: 712369847, cell: 3, density: 80 }, [0x7f7647fd, 0xe9fcac0e, 0x7f7647fd, 0xe9fcac0e, 0x7f7647fd, 0xe9fcac0e, 0x7f7647fd, 0xe9fcac0e]),
    ("blocks_11", Gen::Blocks { w: 53, h: 31, seed: 561256346, cell: 4, density: 160 }, [0x0d472165, 0x97de7e34, 0xbb6a98cc, 0x73e0bccd, 0xb2fdaf34, 0xcfec41d5, 0xb2fdaf34, 0xcfec41d5]),
    ("blocks_12", Gen::Blocks { w: 33, h: 34, seed: 406227276, cell: 4, density: 128 }, [0x12af902f, 0x6376652c, 0xadef783a, 0x834466af, 0x005688df, 0x9b4385ae, 0x005688df, 0x9b4385ae]),
    ("blocks_13", Gen::Blocks { w: 53, h: 40, seed: 419707742, cell: 3, density: 160 }, [0x160fc8e9, 0xcd273c27, 0xfdb94812, 0x3d0b50ae, 0xef31f728, 0x5b02eae0, 0xef31f728, 0x5b02eae0]),
    ("blocks_14", Gen::Blocks { w: 40, h: 53, seed: 725211498, cell: 6, density: 160 }, [0x53176f01, 0x7a6819c0, 0xa7912481, 0x2dbd78d9, 0xee056d77, 0x7fb61e44, 0xee056d77, 0x7fb61e44]),
    ("blocks_15", Gen::Blocks { w: 52, h: 29, seed: 506168660, cell: 2, density: 80 }, [0x31f46f77, 0x7d9efb06, 0x3bf68152, 0xf6281fd4, 0xeb10bd3d, 0xf17e017b, 0xeb10bd3d, 0xf17e017b]),
    ("blocks_16", Gen::Blocks { w: 47, h: 31, seed: 201026794, cell: 3, density: 80 }, [0xc2a1b984, 0x26feda70, 0xc2a1b984, 0x26feda70, 0xc2a1b984, 0x26feda70, 0xc2a1b984, 0x26feda70]),
    ("blocks_17", Gen::Blocks { w: 36, h: 51, seed: 475709905, cell: 5, density: 160 }, [0xb79d35f2, 0xfd58be6c, 0x9cae2688, 0xaa1f45bd, 0x4dca2eb0, 0x9d781db8, 0x4dca2eb0, 0x9d781db8]),
    ("blocks_18", Gen::Blocks { w: 51, h: 53, seed: 738761531, cell: 2, density: 160 }, [0x57ea6277, 0x4bf30df9, 0xac5a24ba, 0xb939d0d5, 0x655e4e62, 0x7bbf0396, 0x655e4e62, 0x7bbf0396]),
    ("blocks_19", Gen::Blocks { w: 16, h: 59, seed: 131996164, cell: 5, density: 160 }, [0x568afa8e, 0xc2693732, 0x50082b6b, 0x332f4660, 0xc3a93e95, 0xf4d20dcc, 0xc3a93e95, 0xf4d20dcc]),
    ("blocks_20", Gen::Blocks { w: 48, h: 32, seed: 886502584, cell: 2, density: 80 }, [0xfd03d671, 0x6bb61e6d, 0x0175c211, 0xfb251abd, 0xbef08ff1, 0xbee65256, 0xbef08ff1, 0xbee65256]),
    ("blocks_21", Gen::Blocks { w: 41, h: 60, seed: 323088993, cell: 3, density: 80 }, [0x5284d808, 0x658ef13e, 0x59bfd444, 0xeab44f94, 0xb86029d4, 0x7738fd19, 0xb86029d4, 0x7738fd19]),
    ("blocks_22", Gen::Blocks { w: 10, h: 32, seed: 186545564, cell: 5, density: 80 }, [0x7ee4ba0d, 0xf05816a9, 0x7ee4ba0d, 0xf05816a9, 0x7ee4ba0d, 0xf05816a9, 0x7ee4ba0d, 0xf05816a9]),
    ("blocks_23", Gen::Blocks { w: 53, h: 55, seed: 511750034, cell: 5, density: 160 }, [0xc1ad116a, 0xff73bade, 0xa254dc64, 0xc2ceacbc, 0x820ee6c4, 0x62b9a3d7, 0x820ee6c4, 0x62b9a3d7]),
    ("blocks_24", Gen::Blocks { w: 37, h: 8, seed: 302462403, cell: 2, density: 128 }, [0xb8f74c7b, 0x2bd81580, 0x5684da69, 0xb1f756e8, 0xe47c77a9, 0x80441455, 0xe47c77a9, 0x80441455]),
    ("blocks_25", Gen::Blocks { w: 27, h: 20, seed: 441879618, cell: 4, density: 128 }, [0xfae7beb3, 0xd5084b50, 0xfae7beb3, 0xd5084b50, 0xfae7beb3, 0xd5084b50, 0xfae7beb3, 0xd5084b50]),
    ("blocks_26", Gen::Blocks { w: 19, h: 60, seed: 404189746, cell: 5, density: 160 }, [0x77beccd3, 0x5acf0a2c, 0x2a5df86b, 0x07ec4bc3, 0x9d265b4c, 0xe7c7ecdf, 0x9d265b4c, 0xe7c7ecdf]),
    ("blocks_27", Gen::Blocks { w: 10, h: 13, seed: 479204197, cell: 5, density: 160 }, [0xc02fea8a, 0xc4448d88, 0xc02fea8a, 0xc4448d88, 0xc02fea8a, 0xc4448d88, 0xc02fea8a, 0xc4448d88]),
    ("blocks_28", Gen::Blocks { w: 17, h: 36, seed: 366827852, cell: 6, density: 80 }, [0x1df72035, 0xddf9b371, 0x1df72035, 0xddf9b371, 0x1df72035, 0xddf9b371, 0x1df72035, 0xddf9b371]),
    ("blocks_29", Gen::Blocks { w: 13, h: 25, seed: 436728602, cell: 6, density: 160 }, [0x260ebd60, 0x53aad70e, 0x260ebd60, 0x53aad70e, 0x260ebd60, 0x53aad70e, 0x260ebd60, 0x53aad70e]),
    ("blocks_30", Gen::Blocks { w: 34, h: 42, seed: 901301440, cell: 3, density: 80 }, [0x66f9d4c0, 0xb15f5477, 0x59f226de, 0xc3b6b667, 0x703df40d, 0xe62e85e5, 0x703df40d, 0xe62e85e5]),
    ("blocks_31", Gen::Blocks { w: 20, h: 32, seed: 929832979, cell: 6, density: 80 }, [0x0eb84b9b, 0xb276e023, 0x0eb84b9b, 0xb276e023, 0x0eb84b9b, 0xb276e023, 0x0eb84b9b, 0xb276e023]),
    ("blocks_32", Gen::Blocks { w: 28, h: 43, seed: 937848188, cell: 4, density: 128 }, [0xafd60b15, 0x29b960aa, 0x66b974ef, 0x82f49b4f, 0x0b49dde6, 0x040a52e7, 0x0b49dde6, 0x040a52e7]),
    ("blocks_33", Gen::Blocks { w: 40, h: 53, seed: 645973809, cell: 2, density: 128 }, [0x8f1a73c4, 0x2cab1f74, 0x22304f7d, 0xe0fa7d21, 0x7adffb02, 0x6b1c348f, 0x7adffb02, 0x6b1c348f]),
    ("blocks_34", Gen::Blocks { w: 60, h: 9, seed: 61022415, cell: 4, density: 80 }, [0x77dbf02d, 0x616a66bc, 0x77dbf02d, 0x616a66bc, 0x77dbf02d, 0x616a66bc, 0x77dbf02d, 0x616a66bc]),
    ("blocks_35", Gen::Blocks { w: 22, h: 60, seed: 182499955, cell: 6, density: 80 }, [0xc0db13fe, 0x13dd61c0, 0xc0db13fe, 0x13dd61c0, 0xc0db13fe, 0x13dd61c0, 0xc0db13fe, 0x13dd61c0]),
    ("blocks_36", Gen::Blocks { w: 36, h: 40, seed: 930068451, cell: 4, density: 128 }, [0x5bd07773, 0x47d17757, 0xd0903e2f, 0xa7add21a, 0xdfb5252d, 0x156b8c05, 0xdfb5252d, 0x156b8c05]),
    ("blocks_37", Gen::Blocks { w: 42, h: 39, seed: 275653448, cell: 5, density: 80 }, [0x4af90a69, 0xcd24ffb2, 0x4af90a69, 0xcd24ffb2, 0x4af90a69, 0xcd24ffb2, 0x4af90a69, 0xcd24ffb2]),
    ("blocks_38", Gen::Blocks { w: 29, h: 23, seed: 831408258, cell: 2, density: 80 }, [0x82e1ad52, 0xae2f231a, 0x82e1ad52, 0xae2f231a, 0x82e1ad52, 0xae2f231a, 0x82e1ad52, 0xae2f231a]),
    ("blocks_39", Gen::Blocks { w: 56, h: 54, seed: 859482511, cell: 4, density: 160 }, [0xad5c5ec4, 0xb68e5788, 0xf7c64bdf, 0xa35631c0, 0xfd96c9fd, 0xf086aaab, 0xfd96c9fd, 0xf086aaab]),
    ("smooth_0", Gen::Smooth { w: 38, h: 56, seed: 341000458, density: 100, thr: 5 }, [0x9d76b989, 0x163b930e, 0xe5aba48c, 0xee1a1ca4, 0x1e1e6d01, 0x1d0e75e9, 0x1e1e6d01, 0x1d0e75e9]),
    ("smooth_1", Gen::Smooth { w: 44, h: 64, seed: 465258431, density: 100, thr: 4 }, [0x92869b18, 0x7ec0c002, 0x79d8e9c6, 0xd3b86ae3, 0xbe1aed83, 0x646dcbb0, 0xbe1aed83, 0x646dcbb0]),
    ("smooth_2", Gen::Smooth { w: 15, h: 46, seed: 391513919, density: 60, thr: 3 }, [0x245bf85b, 0xb2da38cf, 0x245bf85b, 0xb2da38cf, 0x245bf85b, 0xb2da38cf, 0x245bf85b, 0xb2da38cf]),
    ("smooth_3", Gen::Smooth { w: 52, h: 21, seed: 701363653, density: 128, thr: 4 }, [0x09573ebc, 0x5298c164, 0xb2936886, 0x3e4dc3eb, 0xec5f181d, 0xcdebd006, 0x52756f6c, 0xe9fb0f2d]),
    ("smooth_4", Gen::Smooth { w: 23, h: 50, seed: 500815044, density: 100, thr: 6 }, [0x0321917c, 0xac952f6c, 0x0321917c, 0xac952f6c, 0x0321917c, 0xac952f6c, 0x0321917c, 0xac952f6c]),
    ("smooth_5", Gen::Smooth { w: 63, h: 62, seed: 810381461, density: 128, thr: 3 }, [0x76fdc867, 0x3b6b9095, 0xf6677bf4, 0xc97f6826, 0xd96efe3f, 0x5cf69052, 0xd96efe3f, 0x5cf69052]),
    ("smooth_6", Gen::Smooth { w: 16, h: 11, seed: 728424493, density: 100, thr: 4 }, [0x9b94b04e, 0xbc445967, 0x9b94b04e, 0xbc445967, 0x9b94b04e, 0xbc445967, 0x9b94b04e, 0xbc445967]),
    ("smooth_7", Gen::Smooth { w: 21, h: 15, seed: 850655571, density: 100, thr: 4 }, [0x31e9ca23, 0x20c1b58d, 0xe7265824, 0xcc24affb, 0x7c359677, 0xbd5ad52d, 0x7c359677, 0xbd5ad52d]),
    ("smooth_8", Gen::Smooth { w: 55, h: 43, seed: 384360327, density: 128, thr: 4 }, [0xce40cdcf, 0xe333272b, 0x6f51c816, 0xf8308390, 0xb8bf6a14, 0xf13ac48a, 0x4b4367b1, 0x0506df54]),
    ("smooth_9", Gen::Smooth { w: 38, h: 13, seed: 235831588, density: 128, thr: 6 }, [0xedcf4d2b, 0x265ab6e9, 0xedcf4d2b, 0x265ab6e9, 0xedcf4d2b, 0x265ab6e9, 0xedcf4d2b, 0x265ab6e9]),
    ("smooth_10", Gen::Smooth { w: 57, h: 38, seed: 1024593274, density: 60, thr: 5 }, [0xc60291c7, 0x120944cb, 0xc60291c7, 0x120944cb, 0xc60291c7, 0x120944cb, 0xc60291c7, 0x120944cb]),
    ("smooth_11", Gen::Smooth { w: 15, h: 51, seed: 37101993, density: 100, thr: 4 }, [0xa165b7de, 0x3498299c, 0xe7ac639a, 0xbb0fa112, 0x315b0100, 0x659bc53c, 0x315b0100, 0x659bc53c]),
    ("smooth_12", Gen::Smooth { w: 42, h: 37, seed: 1017797394, density: 128, thr: 6 }, [0x3fe499bd, 0x048a330c, 0x3fe499bd, 0x048a330c, 0x3fe499bd, 0x048a330c, 0x3fe499bd, 0x048a330c]),
    ("smooth_13", Gen::Smooth { w: 40, h: 49, seed: 492421044, density: 60, thr: 6 }, [0x1f734fd0, 0xe4f933d6, 0x1f734fd0, 0xe4f933d6, 0x1f734fd0, 0xe4f933d6, 0x1f734fd0, 0xe4f933d6]),
    ("smooth_14", Gen::Smooth { w: 30, h: 40, seed: 938296425, density: 60, thr: 5 }, [0x3fe63349, 0x65eeb88a, 0x3fe63349, 0x65eeb88a, 0x3fe63349, 0x65eeb88a, 0x3fe63349, 0x65eeb88a]),
    ("smooth_15", Gen::Smooth { w: 46, h: 22, seed: 799283178, density: 60, thr: 4 }, [0x06d5a6ca, 0x129f9ce5, 0x06d5a6ca, 0x129f9ce5, 0x06d5a6ca, 0x129f9ce5, 0x06d5a6ca, 0x129f9ce5]),
    ("smooth_16", Gen::Smooth { w: 62, h: 36, seed: 816922593, density: 60, thr: 4 }, [0xe5d61d24, 0xc33afead, 0xe5d61d24, 0xc33afead, 0xe5d61d24, 0xc33afead, 0xe5d61d24, 0xc33afead]),
    ("smooth_17", Gen::Smooth { w: 38, h: 58, seed: 67511827, density: 128, thr: 3 }, [0x1d0920f5, 0x1127c018, 0x7bd34183, 0xbe99ac3f, 0x6f7d4e0a, 0x8b073c58, 0x6f7d4e0a, 0x8b073c58]),
    ("smooth_18", Gen::Smooth { w: 49, h: 28, seed: 610005536, density: 100, thr: 4 }, [0x09fca933, 0xfa423530, 0x6341f3c2, 0x9398cd04, 0x0cd28f33, 0x7cdd16c8, 0x0cd28f33, 0x7cdd16c8]),
    ("smooth_19", Gen::Smooth { w: 61, h: 40, seed: 374269539, density: 60, thr: 6 }, [0x35c53b91, 0x35c53b91, 0x35c53b91, 0x35c53b91, 0x35c53b91, 0x35c53b91, 0x35c53b91, 0x35c53b91]),
    ("smooth_20", Gen::Smooth { w: 17, h: 13, seed: 946492206, density: 128, thr: 5 }, [0x65023e43, 0xc9f9ed87, 0x65023e43, 0xc9f9ed87, 0x65023e43, 0xc9f9ed87, 0x65023e43, 0xc9f9ed87]),
    ("smooth_21", Gen::Smooth { w: 34, h: 45, seed: 1029460325, density: 128, thr: 3 }, [0x0ab71d54, 0x9d1ad988, 0x54c284c0, 0x27fa3f30, 0x34b0edf9, 0xfffbc309, 0x34b0edf9, 0xfffbc309]),
    ("smooth_22", Gen::Smooth { w: 34, h: 40, seed: 571378756, density: 100, thr: 3 }, [0xc8570fd5, 0x7465cae4, 0xe5791f2c, 0x85c631d7, 0x325890f1, 0xd503bf49, 0x325890f1, 0xd503bf49]),
    ("smooth_23", Gen::Smooth { w: 18, h: 46, seed: 403499242, density: 100, thr: 6 }, [0x7cf22e3f, 0xc533d004, 0x7cf22e3f, 0xc533d004, 0x7cf22e3f, 0xc533d004, 0x7cf22e3f, 0xc533d004]),
    ("smooth_24", Gen::Smooth { w: 37, h: 27, seed: 62359722, density: 128, thr: 6 }, [0xc485cb31, 0x54a85411, 0x1e3a6696, 0xa7533f80, 0xe9eef7f3, 0xb49702dd, 0xe9eef7f3, 0xb49702dd]),
    ("smooth_25", Gen::Smooth { w: 14, h: 25, seed: 910844773, density: 60, thr: 3 }, [0x8b1fe05b, 0xffebcfc9, 0x8b1fe05b, 0xffebcfc9, 0x8b1fe05b, 0xffebcfc9, 0x8b1fe05b, 0xffebcfc9]),
    ("smooth_26", Gen::Smooth { w: 62, h: 35, seed: 300383170, density: 60, thr: 6 }, [0x399bb94d, 0x399bb94d, 0x399bb94d, 0x399bb94d, 0x399bb94d, 0x399bb94d, 0x399bb94d, 0x399bb94d]),
    ("smooth_27", Gen::Smooth { w: 49, h: 37, seed: 858727821, density: 100, thr: 3 }, [0xcbe7fee0, 0xd0101632, 0x1d31d0cb, 0x12d0ad51, 0x22da9b05, 0xcc79e110, 0x81f39633, 0x49b006aa]),
    ("smooth_28", Gen::Smooth { w: 15, h: 33, seed: 864744233, density: 128, thr: 3 }, [0x4c317e35, 0xbfa192c8, 0xff2d5b2e, 0xd7a998b6, 0x69ab4540, 0x4ede9686, 0x69ab4540, 0x4ede9686]),
    ("smooth_29", Gen::Smooth { w: 12, h: 42, seed: 330124697, density: 128, thr: 5 }, [0x382b6c34, 0x885e7bd9, 0x38ad9b69, 0x6dc239c6, 0x4ddd7786, 0x5b996be0, 0x4ddd7786, 0x5b996be0]),
    ("smooth_30", Gen::Smooth { w: 33, h: 17, seed: 782624747, density: 100, thr: 3 }, [0x8d78e7ce, 0xd7647cc7, 0x32a8f51d, 0xf1fbfebc, 0x255bdfd2, 0xf5a7b785, 0x255bdfd2, 0xf5a7b785]),
    ("smooth_31", Gen::Smooth { w: 39, h: 63, seed: 611670097, density: 100, thr: 4 }, [0x01adac74, 0x09b077f6, 0xc1b1a9fd, 0x4eeb39a4, 0x06230e38, 0xd610e6a0, 0x06230e38, 0xd610e6a0]),
    ("smooth_32", Gen::Smooth { w: 44, h: 21, seed: 1048313986, density: 100, thr: 3 }, [0x78c50904, 0x56784e7d, 0x1dc972b7, 0xf5930dcd, 0xa35880dc, 0x72f655a3, 0xa35880dc, 0x72f655a3]),
    ("smooth_33", Gen::Smooth { w: 44, h: 37, seed: 268877560, density: 128, thr: 3 }, [0xca5b1750, 0xc1d4d1fb, 0xed256b32, 0x82929303, 0xe09c3a64, 0x5648ba9e, 0xe09c3a64, 0x5648ba9e]),
    ("smooth_34", Gen::Smooth { w: 37, h: 12, seed: 366266559, density: 60, thr: 5 }, [0x33ee4d37, 0x6979b4d1, 0x33ee4d37, 0x6979b4d1, 0x33ee4d37, 0x6979b4d1, 0x33ee4d37, 0x6979b4d1]),
    ("smooth_35", Gen::Smooth { w: 41, h: 25, seed: 998591684, density: 128, thr: 3 }, [0x01008405, 0x1324adc3, 0x059a6cbf, 0xa0df63ff, 0xf53990af, 0x27f36207, 0xf53990af, 0x27f36207]),
    ("smooth_36", Gen::Smooth { w: 41, h: 51, seed: 481630829, density: 100, thr: 3 }, [0x465dd48c, 0x2099f4d4, 0x985fe598, 0x8e3625e7, 0x9d143de1, 0xff92045a, 0x41f891fb, 0x7e815e2b]),
    ("smooth_37", Gen::Smooth { w: 28, h: 52, seed: 207509296, density: 128, thr: 3 }, [0xa232ccb8, 0xcff5afad, 0x83667224, 0x251b9cf1, 0x252767f9, 0x1e147253, 0x252767f9, 0x1e147253]),
    ("smooth_38", Gen::Smooth { w: 52, h: 44, seed: 882502613, density: 100, thr: 3 }, [0x98bda7c1, 0x9bfe1c7d, 0x60fe7329, 0xcee2cfbf, 0x71bb0b45, 0x6dd12b1c, 0x43c4d44c, 0x91913a0e]),
    ("smooth_39", Gen::Smooth { w: 42, h: 47, seed: 784493961, density: 100, thr: 5 }, [0x5d90f48f, 0xed0dd9d0, 0x4f92602d, 0x2d7b6605, 0x4d46c569, 0x2bf0c023, 0x4d46c569, 0x2bf0c023]),
];
