//! `approx_poly_dp` (integer contours) and `approx::approx_poly_dp`
//! (floating-point curves) against `cv2.approxPolyDP` (OpenCV 4.13).
//!
//! The previous integer version always kept the contour's first point as a
//! vertex of a closed curve, which turns most quadrilateral contours into
//! pentagons, and measured deviations to the infinite chord line. OpenCV
//! starts a closed curve at a far-apart pair of points found from point 0,
//! measures deviations to the chord *segment* and finishes with a clean-up
//! pass over nearly collinear vertices; both functions now do the same.
//!
//! Inputs: noisy polyline curves, the cv2-identical `find_contours` borders
//! of filled convex quadrilaterals (the ArUco candidate case) and of random
//! blobs, each approximated with eight tolerances, closed and open, as
//! integers and (scaled by 1/4, exact in `f32`) as floating point. The
//! reference for each (case, variant) is a 32-bit FNV-1a digest of every
//! output polygon (length, then points, in order).

use cv_core::{CpuTensor, TensorShape};
use cv_imgproc::{Contour, approx, approx_poly_dp, find_contours};
use nalgebra::Point2;

enum Gen {
    /// A closed polyline through `verts` LCG vertices in `[0, size)^2`,
    /// `steps` points per edge, each moved by up to `noise` pixels.
    Poly {
        seed: u32,
        verts: usize,
        size: i64,
        steps: i64,
        noise: i64,
    },
    /// The borders of a filled convex polygon (clockwise on screen).
    Polygon {
        w: usize,
        h: usize,
        corners: &'static [(i64, i64)],
    },
    /// The borders of thresholded, 3x3-smoothed LCG noise.
    Smooth {
        w: usize,
        h: usize,
        seed: u32,
        density: u8,
        thr: u32,
    },
}

fn lcg(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn contours_of(img: Vec<u8>, w: usize, h: usize) -> Vec<Vec<(i32, i32)>> {
    let t = CpuTensor::from_vec(img, TensorShape::new(1, h, w)).unwrap();
    find_contours(&t)
        .unwrap()
        .into_iter()
        .map(|c| {
            c.points
                .iter()
                .map(|&(x, y)| (x as i32, y as i32))
                .collect()
        })
        .collect()
}

fn curves(g: &Gen) -> Vec<Vec<(i32, i32)>> {
    match *g {
        Gen::Poly {
            seed,
            verts,
            size,
            steps,
            noise,
        } => {
            let mut s = seed;
            let mut next = |m: i64| {
                s = lcg(s);
                i64::from(s >> 24) % m
            };
            let v: Vec<(i64, i64)> = (0..verts).map(|_| (next(size), next(size))).collect();
            let mut pts = Vec::new();
            for k in 0..verts {
                let ((x0, y0), (x1, y1)) = (v[k], v[(k + 1) % verts]);
                for t in 0..steps {
                    let mut x = x0 + ((x1 - x0) * t).div_euclid(steps);
                    let mut y = y0 + ((y1 - y0) * t).div_euclid(steps);
                    if noise > 0 {
                        x += next(2 * noise + 1) - noise;
                        y += next(2 * noise + 1) - noise;
                    }
                    pts.push((x as i32, y as i32));
                }
            }
            vec![pts]
        }
        Gen::Polygon { w, h, corners } => {
            let n = corners.len();
            let img = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as i64, (i / w) as i64);
                    let inside = (0..n).all(|k| {
                        let ((ax, ay), (bx, by)) = (corners[k], corners[(k + 1) % n]);
                        (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0
                    });
                    if inside { 255 } else { 0 }
                })
                .collect();
            contours_of(img, w, h)
        }
        Gen::Smooth {
            w,
            h,
            seed,
            density,
            thr,
        } => {
            let mut s = seed;
            let n: Vec<bool> = (0..w * h)
                .map(|_| {
                    s = lcg(s);
                    ((s >> 24) as u8) < density
                })
                .collect();
            let mut img = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut c = 0;
                    for yy in y.saturating_sub(1)..(y + 2).min(h) {
                        for xx in x.saturating_sub(1)..(x + 2).min(w) {
                            c += u32::from(n[yy * w + xx]);
                        }
                    }
                    img[y * w + x] = if c >= thr { 255 } else { 0 };
                }
            }
            contours_of(img, w, h)
        }
    }
}

fn epsilons(n: usize) -> [f64; 8] {
    let n = n as f64;
    [0.0, 0.5, 1.0, 2.0, 3.7, n * 0.03, n * 0.05, n * 0.1]
}

struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf29ce484222325)
    }
    fn feed(&mut self, v: i64) {
        self.0 ^= v as u64;
        self.0 = self.0.wrapping_mul(0x100000001b3);
    }
    fn polygon(&mut self, pts: impl ExactSizeIterator<Item = (i64, i64)>) {
        self.feed(pts.len() as i64);
        for (x, y) in pts {
            self.feed(x);
            self.feed(y);
        }
    }
    fn done(&self) -> u32 {
        (self.0 ^ (self.0 >> 32)) as u32
    }
}

/// Digests of the four variants: integer closed, integer open, float
/// closed, float open (float curves are the integer ones times 1/4).
fn digests(cs: &[Vec<(i32, i32)>]) -> [u32; 4] {
    let mut out = [0; 4];
    for (k, closed) in [true, false].into_iter().enumerate() {
        let mut d = Digest::new();
        for c in cs {
            let contour = Contour { points: c.clone() };
            for eps in epsilons(c.len()) {
                let a = approx_poly_dp(&contour, eps, closed);
                d.polygon(a.points.iter().map(|&(x, y)| (i64::from(x), i64::from(y))));
            }
        }
        out[k] = d.done();

        let mut d = Digest::new();
        for c in cs {
            let curve: Vec<Point2<f64>> = c
                .iter()
                .map(|&(x, y)| Point2::new(x as f64 * 0.25, y as f64 * 0.25))
                .collect();
            for eps in epsilons(c.len()) {
                let a = approx::approx_poly_dp(&curve, eps, closed);
                d.polygon(a.iter().map(|p| ((p.x * 4.0) as i64, (p.y * 4.0) as i64)));
            }
        }
        out[2 + k] = d.done();
    }
    out
}

#[test]
fn approx_poly_dp_matches_opencv() {
    let mut failures = Vec::new();
    for (name, g, expected) in CASES {
        let got = digests(&curves(g));
        for (k, variant) in ["int closed", "int open", "float closed", "float open"]
            .iter()
            .enumerate()
        {
            if got[k] != expected[k] {
                failures.push(format!("{name} {variant}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} (case, variant) pairs differ from cv2: {failures:?}",
        failures.len(),
        CASES.len() * 4
    );
}

/// A tilted square's border approximates to its four corners whichever
/// border pixel the contour starts on; the old version kept the start
/// pixel and returned a pentagon. Values from cv2.approxPolyDP.
#[test]
fn tilted_square_is_a_quadrilateral() {
    let corners: &'static [(i64, i64)] = &[(30, 8), (52, 30), (30, 52), (8, 30)];
    let cs = curves(&Gen::Polygon {
        w: 60,
        h: 60,
        corners,
    });
    assert_eq!(cs.len(), 1);
    let c = Contour {
        points: cs[0].clone(),
    };
    let a = approx_poly_dp(&c, c.points.len() as f64 * 0.03, true);
    assert_eq!(a.points, [(30, 8), (8, 30), (30, 52), (52, 30)]);
}

/// An open curve keeps both end points; a closed one does not have to keep
/// its first point.
#[test]
fn end_points() {
    // Square traced from the middle of its top edge.
    let mut pts = Vec::new();
    for x in 5..=10 {
        pts.push((x, 0));
    }
    for y in 1..=10 {
        pts.push((10, y));
    }
    for x in (0..10).rev() {
        pts.push((x, 10));
    }
    for y in (0..10).rev() {
        pts.push((0, y));
    }
    for x in 1..5 {
        pts.push((x, 0));
    }
    let c = Contour { points: pts };
    // cv2.approxPolyDP(c, 1.0, True) and (c, 1.0, False).
    let closed = approx_poly_dp(&c, 1.0, true);
    assert_eq!(closed.points, [(0, 0), (10, 0), (10, 10), (0, 10)]);
    let open = approx_poly_dp(&c, 1.0, false);
    assert_eq!(
        open.points,
        [(5, 0), (10, 0), (10, 10), (0, 10), (0, 0), (4, 0)]
    );
}

// Generated by cv2 4.13.0: digests of cv2.approxPolyDP over every curve of
// the case and every tolerance of `epsilons`, for the four variants of
// `digests`.
#[rustfmt::skip]
const CASES: &[(&str, Gen, [u32; 4])] = &[
    ("poly_0", Gen::Poly { seed: 142460359, verts: 6, size: 40, steps: 5, noise: 2 }, [0xdeb2ea81, 0xae49019d, 0xec9ab16d, 0xb36dd191]),
    ("poly_1", Gen::Poly { seed: 95985679, verts: 8, size: 40, steps: 26, noise: 2 }, [0xd22e1fdf, 0xdb33ff9e, 0x3e824c73, 0x5bb00e18]),
    ("poly_2", Gen::Poly { seed: 1002612076, verts: 4, size: 40, steps: 24, noise: 2 }, [0xf22aa6d1, 0x3437a8ed, 0xf68d435b, 0x25398161]),
    ("poly_3", Gen::Poly { seed: 950245731, verts: 7, size: 100, steps: 27, noise: 0 }, [0xdf01e739, 0xf38d4a5c, 0x2781a717, 0x4c807ddb]),
    ("poly_4", Gen::Poly { seed: 118509867, verts: 8, size: 16, steps: 5, noise: 0 }, [0x42191e43, 0xdb8d7135, 0x02a35f10, 0x031ac507]),
    ("poly_5", Gen::Poly { seed: 8154533, verts: 3, size: 250, steps: 13, noise: 0 }, [0x5ef89960, 0xe7d92934, 0x31e53f24, 0x578d11e9]),
    ("poly_6", Gen::Poly { seed: 922782871, verts: 8, size: 16, steps: 23, noise: 0 }, [0x9cffe909, 0xb4aadf55, 0x4a461c2b, 0xd76d5424]),
    ("poly_7", Gen::Poly { seed: 212715300, verts: 5, size: 250, steps: 27, noise: 0 }, [0xaaf9ccce, 0x0889ab35, 0x70a1b9b5, 0xbd6f8a16]),
    ("poly_8", Gen::Poly { seed: 615097663, verts: 8, size: 40, steps: 25, noise: 1 }, [0xf8308527, 0xb81f016d, 0xe18fcbf9, 0x28ac3323]),
    ("poly_9", Gen::Poly { seed: 692049935, verts: 3, size: 250, steps: 24, noise: 0 }, [0x1e3daff8, 0x0dba4243, 0x01b11765, 0x6425337b]),
    ("poly_10", Gen::Poly { seed: 1056784158, verts: 4, size: 100, steps: 12, noise: 2 }, [0x4c121ca7, 0x67b57126, 0x234a1076, 0x915aa466]),
    ("poly_11", Gen::Poly { seed: 294716235, verts: 6, size: 16, steps: 9, noise: 2 }, [0xd6966c1c, 0x85960c75, 0xe5327df8, 0x7a2c251d]),
    ("poly_12", Gen::Poly { seed: 899652497, verts: 5, size: 100, steps: 2, noise: 1 }, [0xd494ca95, 0x8296e225, 0x3ced5d58, 0xccdcc185]),
    ("poly_13", Gen::Poly { seed: 645645118, verts: 3, size: 40, steps: 10, noise: 0 }, [0xe2237b1f, 0xddaa9d7c, 0xc1663e9d, 0x0e567bbf]),
    ("poly_14", Gen::Poly { seed: 550981715, verts: 6, size: 250, steps: 18, noise: 2 }, [0x8959d3f4, 0xa11ebeed, 0x33e6ee0f, 0x0deeebc5]),
    ("poly_15", Gen::Poly { seed: 135423294, verts: 3, size: 16, steps: 21, noise: 0 }, [0x19757ff1, 0x9d9872ca, 0x56c70dc1, 0x0a924bfe]),
    ("poly_16", Gen::Poly { seed: 480283213, verts: 4, size: 40, steps: 12, noise: 0 }, [0xe9a95757, 0x1878d2c6, 0x25b75be2, 0x3ca8b855]),
    ("poly_17", Gen::Poly { seed: 344396989, verts: 8, size: 250, steps: 14, noise: 0 }, [0x1b7991c9, 0xbb3cd71a, 0x994f79ad, 0x461cdc6b]),
    ("poly_18", Gen::Poly { seed: 798890353, verts: 5, size: 40, steps: 11, noise: 2 }, [0x968bbc05, 0x8f384e06, 0x5769a0e9, 0x31e4a2b1]),
    ("poly_19", Gen::Poly { seed: 612446916, verts: 6, size: 250, steps: 13, noise: 2 }, [0xb5ed1816, 0xf60b9fc1, 0xe6201ef0, 0x9e8009cd]),
    ("poly_20", Gen::Poly { seed: 1040413899, verts: 6, size: 250, steps: 30, noise: 0 }, [0x3f15e801, 0x916f129a, 0xa5962669, 0xdeb56948]),
    ("poly_21", Gen::Poly { seed: 102775925, verts: 5, size: 250, steps: 30, noise: 0 }, [0xfe5e030c, 0x0b7700ac, 0x5f831908, 0x76713f05]),
    ("poly_22", Gen::Poly { seed: 164897375, verts: 7, size: 16, steps: 28, noise: 2 }, [0xaa6a3305, 0xc84fe5e8, 0xdc19c3aa, 0xdea031b1]),
    ("poly_23", Gen::Poly { seed: 420991602, verts: 5, size: 100, steps: 20, noise: 2 }, [0xcc0ad3df, 0x6601592a, 0x76fff838, 0x15e59eec]),
    ("poly_24", Gen::Poly { seed: 267128312, verts: 6, size: 250, steps: 24, noise: 0 }, [0xdb9009f5, 0xfb1c4b9c, 0xa5ef1ed0, 0x079d7fbd]),
    ("poly_25", Gen::Poly { seed: 52113190, verts: 3, size: 40, steps: 26, noise: 0 }, [0x937f3b43, 0x937f3b43, 0x0150305d, 0x0150305d]),
    ("poly_26", Gen::Poly { seed: 685964260, verts: 7, size: 100, steps: 20, noise: 2 }, [0xff208ff2, 0x0e612b14, 0x4fa7a718, 0x9010b64e]),
    ("poly_27", Gen::Poly { seed: 762192170, verts: 5, size: 250, steps: 18, noise: 0 }, [0x6d066245, 0x50c3473a, 0xeed2f6ba, 0x3454990a]),
    ("poly_28", Gen::Poly { seed: 727239021, verts: 6, size: 16, steps: 23, noise: 2 }, [0xca6ab8c9, 0x09a6d36a, 0x67bdee0a, 0x2862e0eb]),
    ("poly_29", Gen::Poly { seed: 667991238, verts: 8, size: 40, steps: 8, noise: 2 }, [0x256ba7ba, 0xab70abca, 0x2dff5725, 0x86562120]),
    ("poly_30", Gen::Poly { seed: 591711884, verts: 5, size: 100, steps: 11, noise: 0 }, [0x42dcdc6d, 0x8be95635, 0x5c3f9c4d, 0x932a3ba5]),
    ("poly_31", Gen::Poly { seed: 609736732, verts: 4, size: 100, steps: 28, noise: 0 }, [0x178f468b, 0x0b4d09c5, 0xb3a01f4a, 0xadea86d9]),
    ("poly_32", Gen::Poly { seed: 429138953, verts: 3, size: 250, steps: 7, noise: 0 }, [0x7cdeee3e, 0x63194531, 0xdd56fe84, 0x9c7ddafd]),
    ("poly_33", Gen::Poly { seed: 940495541, verts: 8, size: 100, steps: 3, noise: 1 }, [0xffda31f6, 0x65dbe5c2, 0xb3a63cdf, 0xa32ffc19]),
    ("poly_34", Gen::Poly { seed: 994733972, verts: 5, size: 100, steps: 17, noise: 2 }, [0x337fb234, 0x8c7000d1, 0x203dee5c, 0x1e1217b8]),
    ("poly_35", Gen::Poly { seed: 71209212, verts: 3, size: 16, steps: 19, noise: 1 }, [0x0821d70d, 0x94fb20ab, 0x85723b76, 0xd566154e]),
    ("poly_36", Gen::Poly { seed: 357750828, verts: 7, size: 40, steps: 28, noise: 0 }, [0x99c5373d, 0xf9d03436, 0x3599daea, 0x4ecefdc0]),
    ("poly_37", Gen::Poly { seed: 699637872, verts: 5, size: 250, steps: 6, noise: 1 }, [0x90588306, 0xe02bd796, 0x1924f7b3, 0x763ac555]),
    ("poly_38", Gen::Poly { seed: 614836534, verts: 6, size: 250, steps: 20, noise: 0 }, [0x0729f614, 0x225f40bd, 0xb2232bc6, 0x2a083fc1]),
    ("poly_39", Gen::Poly { seed: 857743428, verts: 6, size: 250, steps: 15, noise: 0 }, [0xd7da6276, 0x6fbacea4, 0x7dd48161, 0x989b0d79]),
    ("quad_0", Gen::Polygon { w: 61, h: 61, corners: &[(46, 17), (44, 50), (18, 39), (22, 16)] }, [0xe036a694, 0xc3049cab, 0x82795083, 0x0df39f77]),
    ("quad_1", Gen::Polygon { w: 24, h: 24, corners: &[(16, 7), (18, 15), (7, 19), (6, 9)] }, [0x3c30a1bf, 0x944f9927, 0x8edde34e, 0xadf3cd7b]),
    ("quad_2", Gen::Polygon { w: 56, h: 56, corners: &[(39, 41), (18, 40), (16, 17), (39, 19)] }, [0x5742452d, 0x6350fe3d, 0x1ff49075, 0x351fca6c]),
    ("quad_3", Gen::Polygon { w: 28, h: 28, corners: &[(24, 10), (20, 22), (5, 16), (9, 8)] }, [0xde5dbb80, 0xc212245d, 0x804b818f, 0x5b1d702d]),
    ("quad_4", Gen::Polygon { w: 39, h: 39, corners: &[(22, 16), (23, 24), (14, 22), (17, 16)] }, [0xdd2ee6fc, 0x05da9ee4, 0x9aa79465, 0x922b588f]),
    ("quad_5", Gen::Polygon { w: 61, h: 61, corners: &[(45, 18), (43, 41), (19, 40), (22, 18)] }, [0x504efa68, 0x1bf5974b, 0xe3357da5, 0x36dd0ebc]),
    ("quad_6", Gen::Polygon { w: 40, h: 40, corners: &[(28, 12), (29, 28), (9, 28), (11, 9)] }, [0x092fb1f2, 0x0c27e70d, 0x087c03a4, 0x6eb06cd9]),
    ("quad_7", Gen::Polygon { w: 41, h: 41, corners: &[(24, 17), (24, 23), (17, 23), (15, 16)] }, [0xd5ed0252, 0x0cb6f71c, 0x0d75901f, 0xb99ba469]),
    ("quad_8", Gen::Polygon { w: 35, h: 35, corners: &[(23, 14), (19, 22), (12, 20), (15, 10)] }, [0x53e87aee, 0xb3ef4c19, 0x07608f27, 0x0b4240b1]),
    ("quad_9", Gen::Polygon { w: 21, h: 21, corners: &[(15, 10), (9, 13), (7, 10), (10, 5)] }, [0x398cf9c8, 0x9288da93, 0x3b4391f3, 0x091b58b7]),
    ("quad_10", Gen::Polygon { w: 33, h: 33, corners: &[(23, 20), (11, 21), (9, 12), (21, 10)] }, [0x1e16a90a, 0x206e8743, 0xc320aaa4, 0x5b7e9f86]),
    ("quad_11", Gen::Polygon { w: 51, h: 51, corners: &[(32, 22), (28, 31), (20, 29), (22, 18)] }, [0x61836d00, 0x9b32b7b6, 0x0d8ecaeb, 0x768474e9]),
    ("quad_12", Gen::Polygon { w: 37, h: 37, corners: &[(21, 7), (27, 20), (16, 27), (8, 16)] }, [0x8fe8e6f9, 0x62c4bb3b, 0xc36e5fd5, 0x36d9f0e5]),
    ("quad_13", Gen::Polygon { w: 65, h: 65, corners: &[(51, 42), (29, 46), (18, 24), (39, 15)] }, [0x8f1f0127, 0x7087e54b, 0x8bba4656, 0xfca48dab]),
    ("quad_14", Gen::Polygon { w: 40, h: 40, corners: &[(25, 10), (30, 25), (16, 31), (11, 15)] }, [0xa120f6b0, 0x3c1cf43d, 0xca4c6293, 0x2fbae4cf]),
    ("quad_15", Gen::Polygon { w: 38, h: 38, corners: &[(25, 26), (11, 25), (13, 13), (25, 11)] }, [0x9a1f53cd, 0xd4bbb4eb, 0x5c053026, 0x4313ded8]),
    ("quad_16", Gen::Polygon { w: 16, h: 16, corners: &[(13, 7), (9, 12), (4, 11), (5, 3)] }, [0x989b8994, 0xff368bb2, 0x0858a785, 0x7a20218d]),
    ("quad_17", Gen::Polygon { w: 58, h: 58, corners: &[(51, 33), (23, 45), (6, 28), (33, 14)] }, [0xb8929571, 0x070f4a20, 0x88ccfca4, 0x9289d094]),
    ("quad_18", Gen::Polygon { w: 56, h: 56, corners: &[(34, 19), (37, 34), (22, 38), (18, 22)] }, [0x6f076b5b, 0x98fd1253, 0x0f69e8e5, 0xcd119ed7]),
    ("quad_19", Gen::Polygon { w: 24, h: 24, corners: &[(19, 5), (19, 18), (7, 18), (6, 5)] }, [0xa012992a, 0xed7b99d4, 0x5967e1d8, 0x23d93b08]),
    ("quad_20", Gen::Polygon { w: 32, h: 32, corners: &[(19, 24), (7, 21), (12, 8), (25, 13)] }, [0x97bdd5d5, 0x5d25c3ae, 0x3acefd45, 0x5b0090be]),
    ("quad_21", Gen::Polygon { w: 66, h: 66, corners: &[(55, 45), (19, 53), (16, 23), (48, 14)] }, [0xe887a6d8, 0x14350b93, 0x305b2a82, 0x1157815f]),
    ("quad_22", Gen::Polygon { w: 39, h: 39, corners: &[(32, 14), (21, 29), (10, 23), (16, 8)] }, [0xfa42c0a8, 0x7174fdac, 0xd8c37fc3, 0xbf29e7ae]),
    ("quad_23", Gen::Polygon { w: 40, h: 40, corners: &[(24, 30), (9, 26), (14, 11), (30, 15)] }, [0xca1a361f, 0x8073379c, 0x84e937fd, 0x5b898cd9]),
    ("quad_24", Gen::Polygon { w: 33, h: 33, corners: &[(20, 21), (10, 20), (10, 10), (23, 11)] }, [0x23eb130f, 0xec7267c5, 0x1fe04c9d, 0xa1b9d75b]),
    ("quad_25", Gen::Polygon { w: 40, h: 40, corners: &[(29, 26), (16, 31), (9, 17), (23, 8)] }, [0x197e88cc, 0xb08534f3, 0x1c0978b3, 0xd3fe6b8a]),
    ("quad_26", Gen::Polygon { w: 52, h: 52, corners: &[(33, 26), (26, 32), (19, 27), (26, 19)] }, [0x8dce819e, 0x6a958700, 0x6bf2b8a7, 0x944c0fa5]),
    ("quad_27", Gen::Polygon { w: 31, h: 31, corners: &[(22, 13), (17, 22), (7, 20), (13, 7)] }, [0x57b50078, 0xa9159c0e, 0xbe7fd2b2, 0x4948c361]),
    ("quad_28", Gen::Polygon { w: 41, h: 41, corners: &[(30, 10), (32, 33), (8, 26), (8, 8)] }, [0xa013579e, 0x52ac856c, 0x9ee12482, 0x421377a9]),
    ("quad_29", Gen::Polygon { w: 25, h: 25, corners: &[(19, 8), (16, 19), (5, 14), (9, 6)] }, [0x83d1d735, 0x245f405f, 0x4ebadcf3, 0x761e7d84]),
    ("quad_30", Gen::Polygon { w: 67, h: 67, corners: &[(43, 46), (19, 47), (21, 21), (47, 18)] }, [0x0eb3c3bb, 0x1490cef1, 0x91333680, 0x0829d3e1]),
    ("quad_31", Gen::Polygon { w: 50, h: 50, corners: &[(33, 28), (21, 32), (15, 19), (32, 16)] }, [0xd800f873, 0xcf068cac, 0xf54fcea2, 0xdf899e20]),
    ("quad_32", Gen::Polygon { w: 47, h: 47, corners: &[(30, 26), (22, 30), (16, 21), (26, 16)] }, [0xb0e292bd, 0x3f436752, 0xef597fb2, 0x95af96b3]),
    ("quad_33", Gen::Polygon { w: 35, h: 35, corners: &[(22, 16), (18, 20), (14, 18), (18, 13)] }, [0x472d73c2, 0xda28d300, 0xecda7c70, 0x6f51b8bc]),
    ("quad_34", Gen::Polygon { w: 35, h: 35, corners: &[(22, 25), (10, 21), (13, 10), (23, 14)] }, [0x2fd4deb3, 0xbcf4e5fe, 0xd2c00ccb, 0x0e02e8aa]),
    ("quad_35", Gen::Polygon { w: 48, h: 48, corners: &[(29, 30), (16, 28), (21, 18), (32, 21)] }, [0xcf02537a, 0xcc0a32b9, 0xc99b90f2, 0xbe908554]),
    ("quad_36", Gen::Polygon { w: 65, h: 65, corners: &[(43, 13), (49, 42), (21, 48), (17, 20)] }, [0x6e9ce6e1, 0x8d0f3667, 0xf4e07414, 0x96ccb7f3]),
    ("quad_37", Gen::Polygon { w: 66, h: 66, corners: &[(38, 47), (19, 39), (27, 21), (47, 27)] }, [0x3155245f, 0xe2ef4609, 0x133168cd, 0xee1dad52]),
    ("quad_38", Gen::Polygon { w: 38, h: 38, corners: &[(23, 25), (13, 22), (14, 13), (25, 14)] }, [0xb077b0a3, 0xfcb55fd7, 0x0984790d, 0xdfbcf626]),
    ("quad_39", Gen::Polygon { w: 26, h: 26, corners: &[(16, 9), (18, 16), (8, 18), (7, 10)] }, [0x416b2eeb, 0x20688e28, 0x8034e4ac, 0x7820d697]),
    ("blob_0", Gen::Smooth { w: 21, h: 53, seed: 987006452, density: 128, thr: 6 }, [0x5c4b578d, 0xd9dd6f23, 0x09d6a07a, 0xd7ffe914]),
    ("blob_1", Gen::Smooth { w: 34, h: 39, seed: 45198744, density: 100, thr: 6 }, [0x2d9b271e, 0xb30804a2, 0xb2ebefa2, 0x9dcab2bc]),
    ("blob_2", Gen::Smooth { w: 22, h: 36, seed: 915454847, density: 128, thr: 6 }, [0x8a693116, 0xe83242ea, 0xa062dabd, 0x670a7e23]),
    ("blob_3", Gen::Smooth { w: 58, h: 16, seed: 608313740, density: 100, thr: 4 }, [0xb216ad77, 0x90983312, 0x41af0a6a, 0xfd476896]),
    ("blob_4", Gen::Smooth { w: 13, h: 27, seed: 769815007, density: 100, thr: 5 }, [0x61cbf067, 0x59584016, 0x777b1da3, 0x6eeb0933]),
    ("blob_5", Gen::Smooth { w: 23, h: 55, seed: 177811731, density: 60, thr: 4 }, [0xff684894, 0xd503019c, 0xaa3ba558, 0x91fe5270]),
    ("blob_6", Gen::Smooth { w: 34, h: 58, seed: 634104970, density: 128, thr: 3 }, [0x2f774749, 0x80c9a6aa, 0xf734243d, 0x399690f7]),
    ("blob_7", Gen::Smooth { w: 20, h: 25, seed: 1005237786, density: 100, thr: 5 }, [0x8c0eb038, 0x74f8aef0, 0x0315a884, 0x9d086622]),
    ("blob_8", Gen::Smooth { w: 39, h: 29, seed: 800755253, density: 100, thr: 6 }, [0x12596fda, 0xbcf5dc99, 0xa172dd55, 0x525dad1d]),
    ("blob_9", Gen::Smooth { w: 32, h: 60, seed: 652141981, density: 128, thr: 3 }, [0x043c3004, 0x5faa3b76, 0x16226e0c, 0xbfa6523b]),
    ("blob_10", Gen::Smooth { w: 53, h: 43, seed: 224299339, density: 100, thr: 3 }, [0x7cbda40b, 0x7e9f04bc, 0xa146d788, 0x968c41e5]),
    ("blob_11", Gen::Smooth { w: 38, h: 54, seed: 1068731415, density: 128, thr: 4 }, [0xce9f88fd, 0x1a39c49f, 0x5ee46f57, 0x0f92d77d]),
    ("blob_12", Gen::Smooth { w: 39, h: 12, seed: 612349796, density: 128, thr: 6 }, [0x72aaa6d2, 0x894c9e75, 0x0621ba24, 0xb941d585]),
    ("blob_13", Gen::Smooth { w: 60, h: 59, seed: 704028060, density: 60, thr: 4 }, [0x9ef9f310, 0xd31ff414, 0x7497acf0, 0x03306ef3]),
    ("blob_14", Gen::Smooth { w: 18, h: 51, seed: 917689297, density: 60, thr: 6 }, [0x858763bf, 0x537f31a2, 0x4212615b, 0x537f31a2]),
    ("blob_15", Gen::Smooth { w: 24, h: 55, seed: 49029228, density: 60, thr: 5 }, [0x79cbc4b7, 0x46d4f021, 0x1d685026, 0xb1bc34e9]),
    ("blob_16", Gen::Smooth { w: 15, h: 56, seed: 326507632, density: 128, thr: 6 }, [0x4fef7a55, 0x5451663c, 0xccddeec8, 0x84f96d48]),
    ("blob_17", Gen::Smooth { w: 47, h: 46, seed: 549574755, density: 100, thr: 3 }, [0x2abffc33, 0x3545d0ad, 0x99be8322, 0x0163c543]),
    ("blob_18", Gen::Smooth { w: 21, h: 60, seed: 791632202, density: 128, thr: 5 }, [0x6b80ebf5, 0x134c1d06, 0x2380ab49, 0xe0dbda42]),
    ("blob_19", Gen::Smooth { w: 46, h: 47, seed: 222221254, density: 128, thr: 6 }, [0xb1ec4e7d, 0x56f0eb4f, 0x5fe874bd, 0xa6dbca59]),
];
