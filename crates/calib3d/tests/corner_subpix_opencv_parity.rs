//! `corner_subpix_with` against `cv2.cornerSubPix` (OpenCV 4.13).
//!
//! The previous `corner_subpix` weighted every pixel of its window equally,
//! took gradients of the integer pixels around the rounded estimate instead
//! of a bilinearly resampled window centred on it, compared the step length
//! with `<` instead of OpenCV's `<=`, had no iteration cap, no zero zone and
//! no "too far from the start" fallback. OpenCV weights the gradient terms
//! with `exp(-(dx / win)^2 - (dy / win)^2)`, resamples the window with
//! `getRectSubPix` (border replicated) every iteration and works in `f32`.
//!
//! The images are anti-aliased rotated checkerboards (some with a dark
//! square inside every white one, as on a ChArUco board), optionally
//! blurred and with LCG noise, built here by integer and IEEE arithmetic the
//! reference script reproduces exactly. Start points are lattice corners
//! moved by up to 1.5 px, some within half a pixel of the image edge.
//! Windows, zero zones and termination criteria vary per case.
//!
//! The pip build of cv2 runs Intel IPP's `getRectSubPix`, whose bilinear
//! weights round differently from OpenCV's own C++ code; `cornerSubPix` then
//! amplifies the difference at weakly conditioned corners. The port follows
//! the C++ code, so the references are cv2's results with IPP disabled
//! (`cv2.ipp.setUseIPP(False)`), as `f32` bit patterns, plus the corners
//! where default cv2 (IPP enabled) returned something else.

use cv_calib3d::{SubpixCriteria, corner_subpix_with};
use image::GrayImage;
use nalgebra::Point2;

/// Rotations with rational cosine and sine (Pythagorean triples).
const ROTS: [(f64, f64, f64); 7] = [
    (1.0, 0.0, 1.0),
    (4.0, 3.0, 5.0),
    (12.0, 5.0, 13.0),
    (15.0, 8.0, 17.0),
    (21.0, 20.0, 29.0),
    (3.0, -4.0, 5.0),
    (24.0, -7.0, 25.0),
];
const SUB: usize = 4;
/// Lattice corners -LAT..=LAT along both board axes.
const LAT: i32 = 3;

struct Case {
    w: usize,
    h: usize,
    rot: usize,
    cell: f64,
    cx: f64,
    cy: f64,
    lo: i64,
    hi: i64,
    marker: bool,
    blur: usize,
    noise: i64,
    seed: u32,
    margin: f64,
    spread: u32,
    win: (usize, usize),
    zero_zone: Option<(usize, usize)>,
    criteria: (Option<usize>, Option<f64>),
    /// cv2 with IPP disabled: x, y of every refined corner as `f32` bits.
    no_ipp: &'static [u32],
    /// The corners where default cv2 (IPP enabled) returned something else:
    /// (index, x bits, y bits).
    ipp_differs: &'static [(usize, u32, u32)],
}

fn lcg(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn render(c: &Case) -> GrayImage {
    let (a, b, r) = ROTS[c.rot];
    let (cs, sn) = (a / r, b / r);
    let (w, h) = (c.w, c.h);
    let mut img = vec![0i64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut count = 0;
            for j in 0..SUB {
                for i in 0..SUB {
                    let px = x as f64 - 0.5 + (i as f64 + 0.5) / SUB as f64 - c.cx;
                    let py = y as f64 - 0.5 + (j as f64 + 0.5) / SUB as f64 - c.cy;
                    let u = (cs * px + sn * py) / c.cell;
                    let v = (-sn * px + cs * py) / c.cell;
                    let (fu, fv) = (u.floor(), v.floor());
                    let mut white = (fu + fv).rem_euclid(2.0) == 0.0;
                    if white && c.marker {
                        let (du, dv) = (u - fu, v - fv);
                        if (0.25..0.75).contains(&du) && (0.25..0.75).contains(&dv) {
                            white = false;
                        }
                    }
                    count += i64::from(white);
                }
            }
            img[y * w + x] = c.lo + (c.hi - c.lo) * count / (SUB * SUB) as i64;
        }
    }
    for _ in 0..c.blur {
        let k = [1, 2, 1];
        let src = img.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        let yy = (y + dy).saturating_sub(1).min(h - 1);
                        let xx = (x + dx).saturating_sub(1).min(w - 1);
                        acc += k[dx] * k[dy] * src[yy * w + xx];
                    }
                }
                img[y * w + x] = (acc + 8).div_euclid(16);
            }
        }
    }
    if c.noise > 0 {
        let mut s = c.seed;
        for v in img.iter_mut() {
            s = lcg(s);
            *v = (*v + i64::from(s >> 24) % (2 * c.noise + 1) - c.noise).clamp(0, 255);
        }
    }
    GrayImage::from_raw(w as u32, h as u32, img.iter().map(|&v| v as u8).collect()).unwrap()
}

/// Lattice corners at least `margin` inside the image and the start points
/// (moved by up to `spread` quarter pixels, kept inside the image).
fn corners(c: &Case) -> (Vec<(f64, f64)>, Vec<Point2<f64>>) {
    let (a, b, r) = ROTS[c.rot];
    let (cs, sn) = (a / r, b / r);
    let (w, h) = (c.w as f64, c.h as f64);
    let mut s = c.seed ^ 0x5a5a5a5a;
    let m = 2 * c.spread + 1;
    let (mut truth, mut start) = (Vec::new(), Vec::new());
    for j in -LAT..=LAT {
        for i in -LAT..=LAT {
            let (i, j) = (f64::from(i), f64::from(j));
            let x = c.cx + c.cell * (cs * i - sn * j);
            let y = c.cy + c.cell * (sn * i + cs * j);
            if c.margin <= x && x < w - c.margin && c.margin <= y && y < h - c.margin {
                s = lcg(s);
                let ox = f64::from((s >> 24) % m) - f64::from(c.spread);
                s = lcg(s);
                let oy = f64::from((s >> 24) % m) - f64::from(c.spread);
                let (sx, sy) = (x + ox * 0.25, y + oy * 0.25);
                if 0.0 <= sx && sx < w && 0.0 <= sy && sy < h {
                    truth.push((x, y));
                    // cv2 receives float32 corners.
                    start.push(Point2::new(sx as f32 as f64, sy as f32 as f64));
                }
            }
        }
    }
    (truth, start)
}

fn refine(c: &Case) -> Vec<Point2<f64>> {
    let img = render(c);
    let (_, mut pts) = corners(c);
    let criteria = SubpixCriteria {
        max_iters: c.criteria.0,
        epsilon: c.criteria.1,
    };
    corner_subpix_with(&img, &mut pts, c.win, c.zero_zone, criteria).unwrap();
    pts
}

fn reference(bits: &[u32]) -> Vec<(f64, f64)> {
    bits.chunks(2)
        .map(|p| (f32::from_bits(p[0]) as f64, f32::from_bits(p[1]) as f64))
        .collect()
}

/// With IPP disabled, cv2 runs the C++ code the port follows: every refined
/// coordinate is the same `f32`.
#[test]
fn matches_opencv_without_ipp_exactly() {
    let (mut total, mut differ, mut max_diff) = (0, 0, 0.0f64);
    for (k, c) in CASES.iter().enumerate() {
        let got = refine(c);
        let exp = reference(c.no_ipp);
        assert_eq!(got.len(), exp.len(), "case {k}");
        for (g, e) in got.iter().zip(&exp) {
            total += 1;
            let d = (g.x - e.0).abs().max((g.y - e.1).abs());
            if d != 0.0 {
                differ += 1;
                max_diff = max_diff.max(d);
            }
        }
    }
    assert_eq!(
        differ, 0,
        "{differ} of {total} corners differ from cv2 (IPP off), max {max_diff:e} px"
    );
}

/// Against default cv2 (IPP on): within 1e-3 px except where cv2's own
/// IPP and non-IPP results are further apart than that.
#[test]
fn matches_default_opencv_within_tolerance() {
    let (mut total, mut within, mut ipp_only) = (0, 0, 0);
    for (k, c) in CASES.iter().enumerate() {
        let got = refine(c);
        let mut exp = reference(c.no_ipp);
        let no_ipp = exp.clone();
        for &(i, x, y) in c.ipp_differs {
            exp[i] = (f32::from_bits(x) as f64, f32::from_bits(y) as f64);
        }
        for (i, (g, e)) in got.iter().zip(&exp).enumerate() {
            total += 1;
            let d = (g.x - e.0).abs().max((g.y - e.1).abs());
            if d <= 1e-3 {
                within += 1;
                continue;
            }
            let n = no_ipp[i];
            let cv2_spread = (e.0 - n.0).abs().max((e.1 - n.1).abs());
            assert!(
                cv2_spread > 1e-3,
                "case {k} corner {i}: {d:e} px from cv2, which agrees with itself without IPP"
            );
            ipp_only += 1;
        }
    }
    // 698 of 700 at the time of writing. The other two are weakly
    // conditioned corners where cv2's own results with and without IPP
    // differ by more than 1e-3 px (a 5777-corner sweep gave 5761 within
    // 1e-3 of default cv2, all 5777 identical to cv2 without IPP).
    assert!(
        within + ipp_only == total && ipp_only * 50 <= total,
        "{within} {ipp_only} {total}"
    );
}

#[rustfmt::skip]
const CASES: &[Case] = &[
    Case { w: 76, h: 44, rot: 3, cell: 10.0, cx: 36.75, cy: 23.25, lo: 0, hi: 255, marker: false, blur: 1, noise: 8, seed: 1002612076, margin: 6.0, spread: 4, win: (3, 3), zero_zone: Some((1, 1)), criteria: (Some(0), None),
        no_ipp: &[0x42890c01, 0x40c88a53, 0x425bf1c2, 0x4126b839, 0x427ebae7, 0x4171880e, 0x42023055, 0x411ac302, 0x4225c911, 0x41682999, 0x4248f139, 0x419905d3, 0x426c7f75, 0x41bf61a4, 0x4287ee96, 0x41e4d4af, 0x4124f230, 0x41134744, 0x41987c05, 0x415cdf1c, 0x41df7df0, 0x41937d7d, 0x4212caac, 0x41b8c007, 0x42365766, 0x41e00c12, 0x42598513, 0x4202bda4, 0x427c8c1b, 0x4215ad7e, 0x4165d8c7, 0x41b58917, 0x41b8927a, 0x41daeed8, 0x42001525, 0x4200a276, 0x4223c484, 0x42133bf3, 0x411ced2d, 0x41fc59ee, 0x4192e185, 0x42102747],
        ipp_differs: &[(12, 0x42365766, 0x41e00c13), (15, 0x4165d8c7, 0x41b58916), (16, 0x41b8927a, 0x41daeed9)] },
    Case { w: 60, h: 52, rot: 4, cell: 20.0, cx: 28.25, cy: 24.5, lo: 0, hi: 255, marker: false, blur: 0, noise: 0, seed: 42903404, margin: 6.0, spread: 6, win: (5, 5), zero_zone: None, criteria: (None, Some(0.001)),
        no_ipp: &[0x42281372, 0x4120e1c2, 0x415be22e, 0x412b3a56, 0x41e21f68, 0x41c3fe54, 0x422acc66, 0x421929d7, 0x4167c72d, 0x421bda1f],
        ipp_differs: &[] },
    Case { w: 88, h: 78, rot: 0, cell: 10.0, cx: 46.0, cy: 37.75, lo: 30, hi: 220, marker: false, blur: 3, noise: 0, seed: 615097663, margin: 2.0, spread: 4, win: (8, 8), zero_zone: None, criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x41801c60, 0x414b7e96, 0x41cfe3e0, 0x414b7e3a, 0x42100dfc, 0x414b7def, 0x4237f1f6, 0x414b7e26, 0x42600e23, 0x414b7e7b, 0x4283f8e7, 0x414b7e9d, 0x42980684, 0x414b7eb9, 0x41800edc, 0x418d411b, 0x41cff12d, 0x418d411b, 0x42100771, 0x418d4121, 0x4237f899, 0x418d411b, 0x42600768, 0x418d411b, 0x4283fc4e, 0x418d4121, 0x4297f8cc, 0x41b5bf72, 0x417fe23d, 0x41dd4121, 0x41d00edb, 0x41dd4121, 0x420ff890, 0x41dd411c, 0x42380770, 0x41dd4121, 0x425ff8a3, 0x41dd411b, 0x428403b2, 0x41dd411b, 0x4297fc46, 0x41dd4121, 0x41800ee2, 0x4216a092, 0x41cff125, 0x4216a092, 0x42100767, 0x4216a08c, 0x4237f899, 0x4216a092, 0x4260077e, 0x4216a092, 0x4283fc52, 0x4216a08c, 0x4298035c, 0x4216a082, 0x417fe20c, 0x423ea092, 0x41d00edb, 0x423ea092, 0x420ff8a3, 0x423ea08c, 0x42380768, 0x423ea08c, 0x425ff890, 0x423ea08c, 0x428403ba, 0x423ea08c, 0x4297fc46, 0x423ea08d, 0x41800ecd, 0x4266a08c, 0x41cff147, 0x4266a08c, 0x4210077e, 0x4266a092, 0x4237f899, 0x4266a092, 0x42600767, 0x4266a08c, 0x4283fc46, 0x4266a092, 0x4298035c, 0x4266a088, 0x417ffb8f, 0x42865390, 0x41d0023a, 0x4286538f, 0x420ffee5, 0x4286538f, 0x4238011b, 0x4286538f, 0x425ffee5, 0x4286538f, 0x42840093, 0x4286538e, 0x4297ff6d, 0x4286538e],
        ipp_differs: &[(0, 0x41801c61, 0x414b7e97), (2, 0x42100dfc, 0x414b7dee), (6, 0x42980684, 0x414b7eba), (7, 0x41800edb, 0x418d411b), (20, 0x4297fc46, 0x41dd4122), (33, 0x428403ba, 0x423ea08d), (36, 0x41cff148, 0x4266a08c), (43, 0x41d0023a, 0x42865390), (45, 0x4238011b, 0x42865390)] },
    Case { w: 47, h: 70, rot: 5, cell: 8.0, cx: 25.25, cy: 34.25, lo: 90, hi: 160, marker: false, blur: 2, noise: 8, seed: 294716235, margin: 2.0, spread: 4, win: (5, 5), zero_zone: Some((1, 1)), criteria: (None, Some(0.001)),
        no_ipp: &[0x40c617a0, 0x419eeae3, 0x412c1f13, 0x41576013, 0x417badff, 0x40e538ce, 0x40523cf3, 0x42156cc2, 0x40f23536, 0x41f65bed, 0x4147c641, 0x41c53d20, 0x418b3c39, 0x41930908, 0x41aedca2, 0x413da257, 0x41d75b95, 0x40aebb28, 0x409170b8, 0x424201a8, 0x4114e4f7, 0x42295774, 0x41619d14, 0x420f1d8b, 0x419739ba, 0x41ebc7c3, 0x41bddc0f, 0x41b847c9, 0x41e469aa, 0x41841e6e, 0x4205c2b8, 0x412295f4, 0x412dff11, 0x425612ae, 0x417af676, 0x423b995f, 0x41a5546d, 0x4222f2c4, 0x41c98368, 0x4208e080, 0x41f13621, 0x41de6c23, 0x420ad55d, 0x41aa9019, 0x421eaa24, 0x4171ebe8, 0x4189881e, 0x42691361, 0x41b0af11, 0x424f9ed8, 0x41d77685, 0x423540f0, 0x41fd3399, 0x421bdd15, 0x4211db97, 0x42033044, 0x42243513, 0x41d2399f, 0x41beac26, 0x427bb353, 0x41e2c970, 0x42627e4e, 0x4204bf23, 0x4248b7ef, 0x4217eb85, 0x42304f8e, 0x422a3d0a, 0x42153cbb, 0x41ef31e1, 0x42874b6a, 0x420b01b7, 0x42754d0b, 0x421e72be, 0x425c6c32, 0x422fefc8, 0x4242de2a],
        ipp_differs: &[(0, 0x40c6179f, 0x419eeae3), (1, 0x412c1f13, 0x41576012), (2, 0x417badfe, 0x40e538cd), (3, 0x40523cf6, 0x42156cc2), (4, 0x40f23537, 0x41f65bed), (6, 0x418b3c39, 0x41930907), (9, 0x409170b9, 0x424201a8), (10, 0x4114e4f6, 0x42295774), (12, 0x419739b9, 0x41ebc7c3), (23, 0x4189881e, 0x42691362), (27, 0x4211db98, 0x42033044), (31, 0x4204bf23, 0x4248b7f0), (32, 0x4217eb85, 0x42304f8d)] },
    Case { w: 40, h: 57, rot: 2, cell: 8.0, cx: 19.25, cy: 28.5, lo: 90, hi: 160, marker: false, blur: 2, noise: 8, seed: 1069095910, margin: 0.5, spread: 6, win: (6, 2), zero_zone: None, criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x41ab353d, 0x40667a7e, 0x41e59560, 0x40d1c7ca, 0x420e653a, 0x41187c9e, 0x406c9846, 0x407079b4, 0x4132afdd, 0x40f5f1a3, 0x41982762, 0x411a7627, 0x41cc8705, 0x415c6b54, 0x42038198, 0x41877ed8, 0x40f407a6, 0x4171a7a0, 0x4175b31f, 0x41917314, 0x41b59331, 0x41a9e154, 0x41ef5f44, 0x41c2f413, 0x4212dad2, 0x41dd25f6, 0x408ac514, 0x41afcf77, 0x413a832a, 0x41caf244, 0x419f3148, 0x41e8c1ab, 0x41d67d62, 0x41fbde4b, 0x42088d16, 0x420a2a39, 0x3ff5c12b, 0x41e55e80, 0x41125919, 0x42031b40, 0x4183cae2, 0x4210283b, 0x41bab824, 0x421c763a, 0x41f58143, 0x4228c282, 0x421b4ec5, 0x42327627, 0x40b679dc, 0x4221c572, 0x415352f8, 0x422d14e3, 0x41a1e32a, 0x42398c79, 0x41dfc650, 0x4245ef3b, 0x420c90d2, 0x4252b65a, 0x40285585, 0x423be59c, 0x411a949b, 0x424a316e, 0x4186a863, 0x4255421b],
        ipp_differs: &[(3, 0x406c9848, 0x407079b4), (8, 0x40f407a5, 0x4171a7a0), (11, 0x41ef5f44, 0x41c2f414), (15, 0x419f3149, 0x41e8c1ab), (18, 0x3ff5c127, 0x41e55e80), (19, 0x41125918, 0x42031b40), (24, 0x40b679db, 0x4221c572), (29, 0x40285584, 0x423be59c), (30, 0x411a949a, 0x424a316e)] },
    Case { w: 54, h: 51, rot: 1, cell: 12.5, cx: 26.25, cy: 24.75, lo: 90, hi: 160, marker: false, blur: 0, noise: 3, seed: 713077998, margin: 2.0, spread: 4, win: (8, 8), zero_zone: Some((1, 1)), criteria: (Some(0), None),
        no_ipp: &[0x42255345, 0x409d6177, 0x424c509d, 0x4143083f, 0x41bf05fb, 0x40ea0df6, 0x42073ead, 0x416b0d00, 0x422f20cc, 0x41b1f85e, 0x40c9cc65, 0x411b29dd, 0x4181d49b, 0x418aa0f3, 0x41d22259, 0x41c595f7, 0x4210e0ef, 0x420121d1, 0x4238f4f6, 0x421fd537, 0x410cc867, 0x41d9d51d, 0x4197b72c, 0x420b3373, 0x41e4bb55, 0x4228babc, 0x412f2914, 0x42326395],
        ipp_differs: &[(7, 0x41d22279, 0x41c595f3), (8, 0x4210e102, 0x420121d1)] },
    Case { w: 79, h: 82, rot: 3, cell: 12.5, cx: 40.75, cy: 42.75, lo: 30, hi: 220, marker: false, blur: 0, noise: 0, seed: 563356369, margin: 0.5, spread: 6, win: (5, 2), zero_zone: None, criteria: (None, Some(0.001)),
        no_ipp: &[0x423db050, 0x40702f05, 0x4269c6a4, 0x411a4a3b, 0x428ae898, 0x4178ac27, 0x419b613a, 0x40430d37, 0x41f3ef90, 0x410ee30b, 0x42261e6b, 0x416c751a, 0x42523593, 0x41a55bfe, 0x427e3834, 0x41d479bf, 0x42951f00, 0x420290eb, 0x415821c4, 0x4161218c, 0x41c49ebf, 0x419f9df7, 0x420e7674, 0x41ce5ac1, 0x423aa697, 0x41fd8cd6, 0x4266b09e, 0x4216581d, 0x42895a0c, 0x422de66e, 0x40f3aa8f, 0x41c904e2, 0x4195618b, 0x41f7b207, 0x41edf378, 0x42135565, 0x42232a79, 0x422ad9ea, 0x424f2dfa, 0x424266bf, 0x427b29a5, 0x4259f533, 0x4293951f, 0x4271d1a7, 0x3f078788, 0x42168788, 0x414c8b3f, 0x4227dada, 0x41bece3b, 0x423f5806, 0x420ba66d, 0x4256e09e, 0x4237bba2, 0x426e7f66, 0x4263e505, 0x42830d55, 0x4287dd31, 0x428eda01, 0x40dc6383, 0x42540876, 0x41903233, 0x426b989f, 0x41e89b23, 0x42818f31, 0x42205837, 0x428d5963, 0x424c44c3, 0x429925ff, 0x3f9ca098, 0x427f3222, 0x4141cd53, 0x428bfbb0, 0x41b95c2f, 0x4297a787],
        ipp_differs: &[(34, 0x3f9ca096, 0x427f3222)] },
    Case { w: 59, h: 61, rot: 4, cell: 16.0, cx: 28.25, cy: 31.5, lo: 90, hi: 160, marker: false, blur: 1, noise: 0, seed: 177252621, margin: 6.0, spread: 4, win: (3, 2), zero_zone: Some((0, 0)), criteria: (Some(0), None),
        no_ipp: &[0x42493c12, 0x41066f9e, 0x41db74b8, 0x410a0c8c, 0x421e475a, 0x419d8a0e, 0x424c3bb0, 0x41f8fc58, 0x418382b2, 0x41a3d9c5, 0x41e07110, 0x41fc9fc6, 0x421f64bb, 0x422a1987, 0x424ddfd3, 0x425672e1, 0x418c359a, 0x422b6ec2, 0x41e799d1, 0x4258dcbe, 0x40c02fac, 0x425ab49a],
        ipp_differs: &[(1, 0x41db74b7, 0x410a0c8c), (6, 0x421f64ba, 0x422a1987)] },
    Case { w: 85, h: 81, rot: 3, cell: 12.5, cx: 42.5, cy: 41.5, lo: 90, hi: 160, marker: true, blur: 2, noise: 8, seed: 209826459, margin: 0.5, spread: 2, win: (8, 8), zero_zone: Some((0, 0)), criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x4249a010, 0x40a2b051, 0x426ff014, 0x4105f356, 0x428e92b5, 0x4166fa40, 0x42a0b4c7, 0x41aacfcc, 0x41a69875, 0x4017ae2e, 0x42012275, 0x40f507d9, 0x422e217a, 0x4153fb22, 0x4258488f, 0x419cc9e7, 0x4282d2b9, 0x41cb5e50, 0x4298b252, 0x41f9b505, 0x4175f7c6, 0x414d057c, 0x41d3bf63, 0x419471f6, 0x42151af8, 0x41c40957, 0x424258fd, 0x41f33feb, 0x426e67e1, 0x4211c504, 0x428cc9ef, 0x4228e047, 0x42a252b7, 0x42417d09, 0x411827c7, 0x41bed509, 0x41a5831c, 0x41f13c28, 0x41fa87a5, 0x420ed4c4, 0x42294517, 0x422467a4, 0x4257a456, 0x423ce6d7, 0x4280ec5b, 0x4253cde9, 0x4296c99c, 0x426d3846, 0x406031cc, 0x420925d2, 0x41651436, 0x42231d17, 0x41cca4a5, 0x423a241e, 0x42118032, 0x42528a82, 0x423eabef, 0x4269d99b, 0x4269a8ed, 0x428104f6, 0x428b4e62, 0x428c681b, 0x41056553, 0x424e6b59, 0x419e2d2b, 0x42668324, 0x41f66b14, 0x427df1a5, 0x4226d802, 0x428b17b5, 0x424f3da4, 0x429388f3, 0x408a11f9, 0x42778c60, 0x41638420, 0x428a1141, 0x41bc0363, 0x42947e73, 0x42108788, 0x42a1f0f1],
        ipp_differs: &[(3, 0x42a0b4c7, 0x41aacfcd), (4, 0x41a69875, 0x4017ae2f), (17, 0x411827c6, 0x41bed509), (19, 0x41fa87a4, 0x420ed4c5), (20, 0x42294517, 0x422467a5), (24, 0x406031cd, 0x420925d2), (27, 0x42118031, 0x42528a83), (36, 0x408a11fa, 0x42778c60), (38, 0x41bc0361, 0x42947e73)] },
    Case { w: 77, h: 84, rot: 3, cell: 20.0, cx: 38.5, cy: 42.75, lo: 90, hi: 160, marker: false, blur: 2, noise: 3, seed: 233868861, margin: 6.0, spread: 6, win: (4, 5), zero_zone: None, criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x42650d05, 0x40edc487, 0x414ba012, 0x40caa618, 0x41f26bbc, 0x417bd678, 0x423fc3fb, 0x41c8d427, 0x42832ac6, 0x420a353e, 0x41a6629e, 0x42058eaa, 0x4219bfc3, 0x422ab362, 0x4260b7d7, 0x4250ba96, 0x41377708, 0x424b94d6, 0x41e97224, 0x4271c8e7, 0x423a9cd3, 0x428baf1f],
        ipp_differs: &[(1, 0x414ba013, 0x40caa618), (2, 0x41f26bbb, 0x417bd677), (8, 0x41377709, 0x424b94d6), (9, 0x41e97225, 0x4271c8e6)] },
    Case { w: 52, h: 46, rot: 3, cell: 10.0, cx: 24.25, cy: 24.5, lo: 90, hi: 160, marker: false, blur: 0, noise: 3, seed: 994733972, margin: 0.5, spread: 2, win: (4, 5), zero_zone: Some((1, 1)), criteria: (None, Some(0.001)),
        no_ipp: &[0x423dd167, 0x40448393, 0x41c7d539, 0x4017727b, 0x4206706e, 0x40da7166, 0x4229df88, 0x4138c032, 0x424d894f, 0x4183cc2e, 0x403a9a3e, 0x3fe04ffc, 0x41352ba5, 0x40c86a36, 0x41a0ccb3, 0x41300d12, 0x41e7e08c, 0x41793eca, 0x4216ef68, 0x41a31d89, 0x423a40b8, 0x41c97334, 0x40d23e92, 0x4172be5c, 0x41770ec5, 0x419ddd67, 0x41c26bfc, 0x41c44b3e, 0x42043647, 0x41ea1a7a, 0x42278ca2, 0x42071e05, 0x424ae442, 0x421aca3e, 0x4004994c, 0x41be9397, 0x412b34d4, 0x41e4b7b6, 0x419c3626, 0x4205881b, 0x41e353d5, 0x4217ea8a, 0x42142aad, 0x422a29d9, 0x40c02741, 0x4215dad0, 0x4168a65b, 0x422851c6],
        ipp_differs: &[(0, 0x423dd1df, 0x40447c72), (3, 0x4229df88, 0x4138c033), (5, 0x403a9a3e, 0x3fe04ff8), (10, 0x423a40b8, 0x41c97333), (14, 0x42043647, 0x41ea1a79)] },
    Case { w: 40, h: 75, rot: 2, cell: 10.0, cx: 22.0, cy: 36.75, lo: 0, hi: 255, marker: false, blur: 2, noise: 3, seed: 966402698, margin: 2.0, spread: 6, win: (3, 3), zero_zone: Some((0, 0)), criteria: (None, Some(0.001)),
        no_ipp: &[0x41c2869c, 0x40a6dff9, 0x42063baf, 0x4111608c, 0x4013a2c7, 0x40d0bfdd, 0x413499cb, 0x41297b43, 0x41a3ac07, 0x41674ee4, 0x41ed52f6, 0x4192309a, 0x40ecc7e3, 0x419ebbcf, 0x41851250, 0x41bd3d14, 0x41cf1d9e, 0x41dbee4b, 0x420c51ce, 0x41fb1983, 0x406485ff, 0x41e74e92, 0x414c627d, 0x42039720, 0x41b06580, 0x4212fc65, 0x41fa08b6, 0x42226c10, 0x410e264b, 0x4228a049, 0x4191392e, 0x423809ba, 0x41db0e2b, 0x42473bcd, 0x4211ee4e, 0x425769e5, 0x40a1b353, 0x424d5b53, 0x41654aaf, 0x425ce084, 0x41bc2a16, 0x426c1d91, 0x4203157d, 0x427b9d41, 0x41276faa, 0x4280e1ae, 0x419cf1dd, 0x4288955f, 0x41e51905, 0x428f9b7c],
        ipp_differs: &[(4, 0x41a3ac06, 0x41674ee4), (18, 0x40a1b352, 0x424d5b53)] },
    Case { w: 76, h: 53, rot: 3, cell: 16.0, cx: 39.25, cy: 27.75, lo: 0, hi: 255, marker: false, blur: 3, noise: 0, seed: 613143833, margin: 6.0, spread: 6, win: (3, 3), zero_zone: None, criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x4288d962, 0x40e0bbe0, 0x42028f9a, 0x40c42a78, 0x423b0d3e, 0x4159c7ea, 0x4273833b, 0x41a947a3, 0x412feb89, 0x414b02de, 0x41c8f173, 0x41a229cb, 0x421d0cfe, 0x41de0316, 0x42558336, 0x420d1ad9, 0x4286fe3e, 0x422b27ca, 0x418ccf08, 0x42096dbf, 0x41fdf2b3, 0x42277780],
        ipp_differs: &[(4, 0x412feb88, 0x414b02df), (6, 0x421d0cff, 0x41de0316), (10, 0x41fdf2b4, 0x42277780)] },
    Case { w: 67, h: 48, rot: 1, cell: 8.0, cx: 34.75, cy: 24.5, lo: 0, hi: 255, marker: false, blur: 2, noise: 3, seed: 579850181, margin: 6.0, spread: 4, win: (6, 6), zero_zone: None, criteria: (Some(7), None),
        no_ipp: &[0x425e318e, 0x4121d419, 0x4217b68b, 0x40dc2071, 0x42318fcb, 0x413aded5, 0x424af2ac, 0x4183a7b1, 0x4264a6b2, 0x41aaeac1, 0x41d598c2, 0x4107eccc, 0x420493fd, 0x4155646e, 0x421e350a, 0x4190d677, 0x4237eedb, 0x41b713a2, 0x42518132, 0x41ddc893, 0x426aefca, 0x4201d73a, 0x4178b506, 0x4121bfe7, 0x41afd78f, 0x416dea7a, 0x41e2b7e6, 0x419d8851, 0x420ae4c9, 0x41c41351, 0x42248438, 0x41eaa275, 0x423e5b17, 0x42084d57, 0x4257f352, 0x421b5f7a, 0x412be366, 0x4183c41e, 0x41894bb9, 0x41aa85cc, 0x41bc8fce, 0x41d0a1f8, 0x41ef9041, 0x41f6cb84, 0x4211a946, 0x420ed359, 0x422b29cc, 0x42220e24, 0x4145ad5a, 0x41dda8b9, 0x4195b216, 0x4202148d, 0x41c956e0, 0x42155abb, 0x40f1a277, 0x42086eed, 0x415f4da5, 0x421b591f],
        ipp_differs: &[(1, 0x4217b68b, 0x40dc2070), (11, 0x4178b506, 0x4121bfe6)] },
    Case { w: 72, h: 52, rot: 5, cell: 10.0, cx: 38.0, cy: 26.5, lo: 90, hi: 160, marker: false, blur: 2, noise: 0, seed: 107316312, margin: 2.0, spread: 2, win: (3, 3), zero_zone: None, criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x410002ef, 0x418402a3, 0x41601d79, 0x4107fbe3, 0x40814d9d, 0x4219dfd0, 0x411fe855, 0x41f40144, 0x4180117f, 0x41b3fe03, 0x41aff30c, 0x4168024e, 0x41dff4b6, 0x40d005af, 0x41401dfd, 0x4231ff1b, 0x4190117f, 0x4211ff01, 0x41bfed36, 0x41e401b2, 0x41effae0, 0x41a3fcc6, 0x42100950, 0x4147fc60, 0x42280fbf, 0x40904409, 0x41d00178, 0x422a0151, 0x41fff4b4, 0x420a00f7, 0x4217f9ab, 0x41d4013d, 0x422ff9cb, 0x41940161, 0x4247f73d, 0x412803ac, 0x426082ce, 0x4032334c, 0x42076e67, 0x4240dbb7, 0x422000bc, 0x42220151, 0x4237fa5b, 0x420200b6, 0x425004ed, 0x41c3febf, 0x4267fff3, 0x4183fffe, 0x427ff881, 0x41080395, 0x423ff175, 0x4239f708, 0x4257ff3d, 0x4219feaf, 0x426ff740, 0x41f401fd, 0x4283ffd3, 0x41b40159, 0x4277f94b, 0x4232020e, 0x42878d7d, 0x421262dd],
        ipp_differs: &[(0, 0x410002fd, 0x418402a2), (1, 0x41601d78, 0x4107fbe2), (3, 0x411fe856, 0x41f40144), (5, 0x41aff30d, 0x4168024f), (6, 0x41dff4b6, 0x40d005ae), (10, 0x41effae7, 0x41a3fcc5), (12, 0x42280fc5, 0x40904407), (13, 0x41d0017e, 0x422a0151), (18, 0x426082cf, 0x4032334b), (20, 0x422000bf, 0x42220151), (23, 0x42680000, 0x41840000), (24, 0x427ff881, 0x41080396), (26, 0x4257ff41, 0x4219feaf), (27, 0x426ff741, 0x41f401fd), (28, 0x4283ffd9, 0x41b4015a)] },
    Case { w: 83, h: 82, rot: 1, cell: 16.0, cx: 42.25, cy: 39.75, lo: 90, hi: 160, marker: true, blur: 0, noise: 8, seed: 809338873, margin: 0.5, spread: 2, win: (5, 5), zero_zone: None, criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x428f153c, 0x3fb72008, 0x42435d17, 0x4092a172, 0x4275bb50, 0x41626ffc, 0x42947be8, 0x41be07fc, 0x41d231e1, 0x40f9fac0, 0x421c6988, 0x418bbe92, 0x425044c1, 0x41d76186, 0x42814b48, 0x42124878, 0x429ab40d, 0x42391e22, 0x407e0aed, 0x412d1b73, 0x41855a18, 0x41a4303f, 0x41eb9341, 0x41f127a8, 0x4228f52b, 0x421eae32, 0x425bc00e, 0x42456d65, 0x42879ae3, 0x426baeb9, 0x42a09121, 0x42891720, 0x40e2842b, 0x420564a5, 0x419f30f8, 0x422bc73b, 0x4202bae1, 0x42523a32, 0x4235eb60, 0x4278d297, 0x4269938e, 0x428f75f5, 0x428de4d8, 0x42a24e85, 0x4125007f, 0x425ed390, 0x41b8ba00, 0x4282edc7, 0x420f3c06, 0x4295c3ff, 0x3f20c179, 0x4288d1e9, 0x41550dd1, 0x429c25c2],
        ipp_differs: &[(0, 0x428f153c, 0x3fb72002), (9, 0x407e0aec, 0x412d1b73), (16, 0x40e2842a, 0x420564a5), (25, 0x3f20c17e, 0x4288d1e9)] },
    Case { w: 64, h: 81, rot: 0, cell: 16.0, cx: 30.5, cy: 40.25, lo: 30, hi: 220, marker: false, blur: 3, noise: 0, seed: 302462403, margin: 2.0, spread: 4, win: (2, 5), zero_zone: None, criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x4167df29, 0x41048970, 0x41f3f54f, 0x41048895, 0x4239f7a0, 0x41048973, 0x416818f0, 0x41c244a9, 0x41f408f2, 0x41c24452, 0x423a01d1, 0x41c242c9, 0x4168128f, 0x4221222c, 0x41f3f9b0, 0x4221222e, 0x4239f869, 0x4221225e, 0x41681df6, 0x42612251, 0x41f3f15a, 0x4261226e, 0x423a0860, 0x4261225d, 0x416810dd, 0x42909113, 0x41f3f5cb, 0x42909113, 0x4239f892, 0x4290912a],
        ipp_differs: &[(0, 0x4167df29, 0x41048971), (12, 0x416810de, 0x42909113), (14, 0x4239f893, 0x4290912a)] },
    Case { w: 51, h: 52, rot: 6, cell: 16.0, cx: 23.75, cy: 24.5, lo: 0, hi: 255, marker: false, blur: 1, noise: 8, seed: 366827852, margin: 6.0, spread: 2, win: (8, 8), zero_zone: None, criteria: (Some(7), None),
        no_ipp: &[0x419a66ff, 0x4112818f, 0x41066b46, 0x41e80459, 0x41bde2da, 0x41c3faf1, 0x421c73b2, 0x419fdbef, 0x41506dab, 0x42315c69, 0x41e1b21f, 0x421f44ea, 0x422e64f3, 0x420cf9db],
        ipp_differs: &[] },
    Case { w: 76, h: 66, rot: 4, cell: 16.0, cx: 37.25, cy: 32.75, lo: 0, hi: 255, marker: false, blur: 3, noise: 0, seed: 682740924, margin: 2.0, spread: 6, win: (6, 5), zero_zone: Some((1, 1)), criteria: (Some(0), None),
        no_ipp: &[0x426d565f, 0x41195ce5, 0x428d3126, 0x41a6d623, 0x4212823f, 0x412277f1, 0x4241179c, 0x41a977b3, 0x426f7132, 0x4200959f, 0x428e1d32, 0x422ceea2, 0x4160d9cf, 0x412a88ee, 0x41cdc27e, 0x41ad1cca, 0x4214fe7d, 0x42032228, 0x42430ee0, 0x422f1f2a, 0x4271d5e9, 0x425b68de, 0x405bb509, 0x41b30b1e, 0x416a8107, 0x4205647b, 0x41d159f7, 0x42314410, 0x42177baf, 0x425d6d40, 0x4084035f, 0x42346fd8, 0x4172a8df, 0x425faa81],
        ipp_differs: &[(8, 0x4214fe91, 0x42032226)] },
    Case { w: 44, h: 61, rot: 6, cell: 8.0, cx: 20.0, cy: 30.75, lo: 0, hi: 255, marker: true, blur: 0, noise: 0, seed: 939787337, margin: 2.0, spread: 6, win: (6, 7), zero_zone: Some((0, 0)), criteria: (None, Some(0.001)),
        no_ipp: &[0x40b163bc, 0x41039830, 0x415d748e, 0x40c446ba, 0x415f365d, 0x40c3af66, 0x41cde3bb, 0x41039088, 0x40b9f876, 0x41b446b2, 0x41a33a41, 0x41929b57, 0x41cddee5, 0x4103a7d3, 0x41cde3a5, 0x41039182, 0x420f4838, 0x41569a4a, 0x4039999a, 0x41e26666, 0x40b9f9e8, 0x41b446c8, 0x41718f4c, 0x41df43b2, 0x41f3a6a1, 0x41bc541e, 0x41f3ae18, 0x41bc50cd, 0x420f480b, 0x41569acd, 0x411852ac, 0x42181785, 0x41185944, 0x421808d1, 0x41c730b3, 0x4205d5be, 0x41c73e10, 0x4205d3d2, 0x4215cd35, 0x41f28374, 0x40cf0fb8, 0x423acd87, 0x41185906, 0x4218096f, 0x41c73053, 0x4205d5cf, 0x420b924b, 0x421a61cd, 0x420b9ad8, 0x421a5c9d, 0x41641c27, 0x4255cfd7, 0x419bbdec, 0x422dbdf1, 0x419bc43c, 0x422dbcba, 0x41ece4f7, 0x4242e871, 0x420b9abe, 0x421a5cac, 0x41641d68, 0x4255cfdb, 0x416427be, 0x4255cfdf, 0x41ecdd48, 0x4242eb86, 0x4213f08c, 0x4258cf61],
        ipp_differs: &[(1, 0x415d748f, 0x40c446ba), (2, 0x415f365c, 0x40c3af66), (5, 0x41a33a40, 0x41929b57), (6, 0x41cddee5, 0x4103a7d4), (16, 0x41185943, 0x421808d1), (21, 0x41185905, 0x42180970), (23, 0x420b924a, 0x421a61ce)] },
    Case { w: 48, h: 80, rot: 6, cell: 16.0, cx: 23.0, cy: 40.5, lo: 0, hi: 255, marker: false, blur: 0, noise: 0, seed: 859482511, margin: 0.5, spread: 6, win: (4, 4), zero_zone: Some((1, 1)), criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x41606bc4, 0x411c6e50, 0x41eb44a6, 0x40a9344e, 0x423232be, 0x3f47e452, 0x404aba8c, 0x41edf7ec, 0x41944276, 0x41c939cf, 0x42076709, 0x41a5438a, 0x40f49731, 0x42340d5c, 0x41b7fffb, 0x42220000, 0x42196d1a, 0x420ff2a4, 0x414263dd, 0x42715e3b, 0x41dbbd8a, 0x425f6318, 0x422b589d, 0x424d6090, 0x4185bde7, 0x4297643e, 0x41ffca1e, 0x428e7236, 0x423dcc4d, 0x4284ba73],
        ipp_differs: &[(2, 0x42323b52, 0x3f266ab4), (7, 0x41b80000, 0x42220000)] },
    Case { w: 62, h: 58, rot: 4, cell: 10.0, cx: 31.25, cy: 28.75, lo: 0, hi: 255, marker: false, blur: 1, noise: 0, seed: 2698843, margin: 6.0, spread: 6, win: (8, 8), zero_zone: None, criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x425016ac, 0x40e5dee3, 0x4218f511, 0x40eefa47, 0x42328959, 0x4161e7ac, 0x424ff763, 0x41abc475, 0x41c0513d, 0x41012a0c, 0x41f5815f, 0x416b5700, 0x4219b792, 0x41aabc56, 0x4234f873, 0x41e18d04, 0x42516114, 0x420fe36b, 0x411a6572, 0x4105c014, 0x41862e0f, 0x4174122d, 0x41bf711f, 0x41ac693c, 0x41f99f0f, 0x41e669ae, 0x421ae9f0, 0x42102c94, 0x423752b0, 0x4228c1f7, 0x425220cd, 0x4244ec11, 0x411cc483, 0x41ada88c, 0x4189cca4, 0x41eb1e4e, 0x41bf52bc, 0x4210d18b, 0x41ff17ee, 0x422b0b41, 0x4219f08f, 0x4246ab65, 0x411f7cf1, 0x4210605f, 0x418d81b3, 0x422d01b1, 0x41bf6c08, 0x42478190, 0x4119c052, 0x4247d01e],
        ipp_differs: &[(0, 0x42500442, 0x40e5e91a), (11, 0x41bf711f, 0x41ac693b), (13, 0x421ae9ef, 0x42102c94), (17, 0x4189cca5, 0x41eb1e4e), (19, 0x41ff17ed, 0x422b0b41), (24, 0x4119c053, 0x4247d01e)] },
    Case { w: 54, h: 60, rot: 5, cell: 20.0, cx: 29.0, cy: 31.75, lo: 90, hi: 160, marker: false, blur: 0, noise: 0, seed: 43605119, margin: 6.0, spread: 6, win: (8, 8), zero_zone: Some((1, 1)), criteria: (Some(0), None),
        no_ipp: &[0x414f98a8, 0x419dc931, 0x41883e99, 0x423ecc8d, 0x41e85047, 0x41fdb6d4, 0x42241f82, 0x417cdb80, 0x4233d827, 0x422f036c],
        ipp_differs: &[] },
    Case { w: 48, h: 51, rot: 0, cell: 16.0, cx: 25.5, cy: 24.75, lo: 90, hi: 160, marker: false, blur: 1, noise: 8, seed: 132842014, margin: 2.0, spread: 2, win: (6, 7), zero_zone: Some((1, 1)), criteria: (None, Some(0.001)),
        no_ipp: &[0x4116cf14, 0x410c56e5, 0x41cc6a71, 0x410ac319, 0x422589a8, 0x410a4115, 0x4117baa6, 0x41c63a70, 0x41cc5850, 0x41c51019, 0x4225cca5, 0x41c54b54, 0x41169b87, 0x422303e5, 0x41cb5ad8, 0x422295c6, 0x422644d1, 0x4222935d],
        ipp_differs: &[(2, 0x422589a8, 0x410a4114), (3, 0x4117baa5, 0x41c63a70)] },
    Case { w: 45, h: 81, rot: 5, cell: 20.0, cx: 20.5, cy: 41.5, lo: 0, hi: 255, marker: false, blur: 3, noise: 8, seed: 1019771629, margin: 2.0, spread: 4, win: (5, 3), zero_zone: None, criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x40933226, 0x41ea2e8a, 0x41849727, 0x41578f10, 0x41089c25, 0x4265d29c, 0x41a446e7, 0x422603c6, 0x4201e7b3, 0x41cbc6bf, 0x41c30a95, 0x428b1d3a, 0x4212206e, 0x42561543],
        ipp_differs: &[(3, 0x41a446e6, 0x422603c6)] },
    Case { w: 70, h: 67, rot: 1, cell: 12.5, cx: 34.5, cy: 35.5, lo: 30, hi: 220, marker: true, blur: 3, noise: 8, seed: 293700151, margin: 6.0, spread: 2, win: (8, 7), zero_zone: None, criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x421e8a4b, 0x41007c59, 0x4245b6f1, 0x417733d7, 0x426e2b76, 0x41b78c52, 0x41afd818, 0x4128c4fc, 0x41ffb72e, 0x418f0fd7, 0x42282225, 0x41cc2d5b, 0x424fd7ca, 0x42044e13, 0x42760a67, 0x42231f3e, 0x41674742, 0x41a2dedf, 0x41c32f15, 0x41e035fa, 0x4209c529, 0x420dfe1b, 0x4231e985, 0x422bcb53, 0x425a0924, 0x424a3a7b, 0x40ee0456, 0x41f1d94d, 0x418797cd, 0x42176ccc, 0x41d86e8a, 0x4235ef34, 0x421418ac, 0x42538927, 0x423cbcd4, 0x42717580, 0x4115716c, 0x42404a38, 0x419bfd20, 0x425dda6a],
        ipp_differs: &[(1, 0x4245b6f1, 0x417733d5), (3, 0x41afd818, 0x4128c4fd), (4, 0x41ffb72f, 0x418f0fd7)] },
    Case { w: 58, h: 58, rot: 2, cell: 10.0, cx: 30.75, cy: 28.25, lo: 0, hi: 255, marker: false, blur: 0, noise: 0, seed: 946492206, margin: 6.0, spread: 6, win: (8, 7), zero_zone: Some((1, 1)), criteria: (Some(200), Some(1e-6)),
        no_ipp: &[0x4217bbd5, 0x41177d2b, 0x42406755, 0x415e16f3, 0x40f4e0ca, 0x40f2fbd7, 0x4181740a, 0x4135fe71, 0x41c6df46, 0x416d02b9, 0x420c7592, 0x419b0d50, 0x423112fb, 0x41b930b4, 0x413df170, 0x41a28c13, 0x41adb5a2, 0x41c441ca, 0x41f98dc1, 0x41e3ecc3, 0x422168f9, 0x420172d4, 0x42420372, 0x420edc58, 0x410c185d, 0x41f00074, 0x419169bf, 0x4207e27f, 0x41d9a75a, 0x42168f0b, 0x42109e56, 0x42257306, 0x423374c8, 0x42333fb7, 0x416430fd, 0x422c9c8f, 0x41b5bc41, 0x423a19b8, 0x41fc394d, 0x4247e883],
        ipp_differs: &[(3, 0x4181740b, 0x4135fe71)] },
    Case { w: 81, h: 45, rot: 3, cell: 16.0, cx: 40.5, cy: 22.5, lo: 90, hi: 160, marker: true, blur: 1, noise: 8, seed: 849114571, margin: 2.0, spread: 2, win: (5, 5), zero_zone: Some((1, 1)), criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x423f768e, 0x4106454b, 0x4279311c, 0x417e5f8b, 0x42980c73, 0x41bcf736, 0x4142c8c5, 0x40f54d0f, 0x41d6a136, 0x416f2f09, 0x4221cbf5, 0x41b3dd26, 0x4259b38f, 0x41f26e1e, 0x428957db, 0x421698ac, 0x409e2a9e, 0x41ab33bd, 0x4196bdc3, 0x41e7aed5, 0x4205815e, 0x42121b83],
        ipp_differs: &[(5, 0x4221cbf5, 0x41b3dd25), (6, 0x4259b38f, 0x41f26e1d), (9, 0x4196bdc2, 0x41e7aed5)] },
    Case { w: 51, h: 40, rot: 6, cell: 16.0, cx: 24.5, cy: 18.75, lo: 30, hi: 220, marker: false, blur: 3, noise: 8, seed: 856605176, margin: 0.5, spread: 6, win: (8, 8), zero_zone: Some((0, 0)), criteria: (None, Some(0.001)),
        no_ipp: &[0x409498d0, 0x41025b71, 0x419dfdab, 0x4061d513, 0x411272af, 0x41ba5f60, 0x41c47b0c, 0x4196427d, 0x421f8f45, 0x41657b8c, 0x415ab5d4, 0x4219412c, 0x41eaa692, 0x420814bc, 0x42316d48, 0x41eb4ebd],
        ipp_differs: &[(1, 0x419dfdab, 0x4061d512)] },
    Case { w: 42, h: 72, rot: 1, cell: 20.0, cx: 21.0, cy: 36.75, lo: 0, hi: 255, marker: false, blur: 2, noise: 8, seed: 41398739, margin: 0.5, spread: 6, win: (5, 5), zero_zone: Some((1, 1)), criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x4188703e, 0x410c39f6, 0x4203df1f, 0x41a61fa0, 0x40a1c74a, 0x41c4f8bb, 0x41a874ac, 0x4212fad7, 0x421380c2, 0x4243403e, 0x410ff313, 0x4252dcca, 0x41c7cf64, 0x428174e0],
        ipp_differs: &[(6, 0x41c7cf65, 0x428174e0)] },
    Case { w: 86, h: 71, rot: 3, cell: 8.0, cx: 44.25, cy: 34.5, lo: 90, hi: 160, marker: true, blur: 3, noise: 0, seed: 366266559, margin: 6.0, spread: 4, win: (3, 5), zero_zone: Some((1, 1)), criteria: (Some(30), Some(0.01)),
        no_ipp: &[0x4241eb04, 0x4118eb48, 0x425e2e7e, 0x41555c17, 0x427a85df, 0x41895222, 0x428b5213, 0x41a6bb07, 0x429968cc, 0x41c48ebd, 0x41f47971, 0x4112131e, 0x4216b09f, 0x414e4ece, 0x4232ccaf, 0x418502c0, 0x424f15b0, 0x41a2cedd, 0x426b483d, 0x41c13051, 0x4283c814, 0x41deaa70, 0x4291ebd8, 0x41fd4a98, 0x41d69c13, 0x4180e48d, 0x4207759c, 0x419f08ed, 0x4223b275, 0x41bce20e, 0x423fea7c, 0x41dba506, 0x425c2274, 0x41fa31a3, 0x42788b0b, 0x420bffa4, 0x428a6ca9, 0x421b012f, 0x41b8bdc1, 0x41ba0f9e, 0x41f0d54b, 0x41d82765, 0x4214d6e6, 0x41f5e464, 0x423110e9, 0x4209ea89, 0x424d6037, 0x4218ecc5, 0x42696512, 0x4227cfcf, 0x4282e164, 0x423707c7, 0x419a8e62, 0x41f2010e, 0x41d3284f, 0x42083a6d, 0x4205b51f, 0x4216f31d, 0x4221de5f, 0x422649ca, 0x423e16dc, 0x4235ade2, 0x425a3b17, 0x4244842e, 0x4276ade4, 0x42538f14, 0x4178b584, 0x421556c6, 0x41b50c0e, 0x4224a8f0, 0x41ed467b, 0x42335d6a, 0x4212d1aa, 0x424247df, 0x422f1840, 0x42518842, 0x424b4901, 0x42606fde, 0x42677a05, 0x426f8598, 0x413c4952, 0x42316610, 0x419698fd, 0x42409915, 0x41cf29d4, 0x424f5491, 0x4203d6e8, 0x425ea45a, 0x42201f4e, 0x426de61e, 0x423cb584, 0x427d1c80],
        ipp_differs: &[(6, 0x4216b09f, 0x414e4ecf), (16, 0x425c2273, 0x41fa31a3), (29, 0x4221de5f, 0x422649c9), (33, 0x4178b585, 0x421556c6), (35, 0x41ed467b, 0x42335d69), (37, 0x422f1841, 0x42518842), (42, 0x41cf29d3, 0x424f5490), (44, 0x42201f4e, 0x426de61d)] },
    Case { w: 86, h: 80, rot: 4, cell: 8.0, cx: 44.75, cy: 39.75, lo: 30, hi: 220, marker: true, blur: 0, noise: 0, seed: 882502613, margin: 2.0, spread: 4, win: (4, 4), zero_zone: Some((0, 0)), criteria: (Some(7), None),
        no_ipp: &[0x422f9de4, 0x40b8df2e, 0x4246a59d, 0x4136091c, 0x425e014b, 0x41863653, 0x427597db, 0x41b2f392, 0x42865290, 0x41df8423, 0x42922423, 0x4205b136, 0x429d4bb7, 0x421bb95e, 0x421988eb, 0x4139cebe, 0x423113a3, 0x4187c5f3, 0x42482488, 0x41b4d173, 0x425efd13, 0x41e27b48, 0x42762ce0, 0x4206a4a2, 0x4286c0ce, 0x421d0942, 0x429272c4, 0x4232b66c, 0x42035db0, 0x418b4fd1, 0x421a4581, 0x41b75623, 0x4231e4d8, 0x41e31b3d, 0x4249436f, 0x4207c828, 0x42606306, 0x421ddc01, 0x4276f71d, 0x4233ea09, 0x42873977, 0x424a2b8e, 0x41dbe1e8, 0x41ba19e0, 0x4204c267, 0x41e59336, 0x421b5eef, 0x4208fca2, 0x4232cee3, 0x421ed199, 0x424a4544, 0x4235426c, 0x42618526, 0x424a97df, 0x42788c07, 0x42613b55, 0x41ae7b8e, 0x41e69752, 0x41dcc949, 0x4209faca, 0x4205b986, 0x4220dd49, 0x421da212, 0x4236314f, 0x42343d27, 0x424cc7cc, 0x424afd29, 0x42623537, 0x42626859, 0x427856b8, 0x4182e76e, 0x420b03ed, 0x41b1778d, 0x4221bc32, 0x41df156e, 0x4236fb8e, 0x4206b7ba, 0x424d7b15, 0x421e0f72, 0x42635b67, 0x42352f9d, 0x4279c9d3, 0x424c82f8, 0x4287aa4f, 0x412c4fae, 0x42224483, 0x4184535b, 0x42387f03, 0x41b36cf4, 0x424e2239, 0x41e228c6, 0x4264c0a7, 0x420802d3, 0x427a588c, 0x421ecf3e, 0x4288619d, 0x423639b5, 0x42933677],
        ipp_differs: &[(37, 0x41df156f, 0x4236fb8e)] },
];
