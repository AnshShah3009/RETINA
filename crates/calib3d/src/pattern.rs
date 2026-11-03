use image::GrayImage;
use nalgebra::{Matrix3, Point2, Vector2, Vector3};

use crate::Result;
pub use cv_imgproc::corner_subpix::{SubpixCriteria, corner_subpix, corner_subpix_with};

/// Detect chessboard corners in an image.
///
/// This function locates the corners of a chessboard pattern in the given image.
/// It uses Harris corner detection followed by grid assignment to find the exact positions
/// of the chessboard corners.
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `pattern_size` - Expected pattern size as (cols, rows)
///
/// # Returns
///
/// A vector of detected corner positions, or an error if the pattern cannot be found.
pub fn find_chessboard_corners(
    image: &GrayImage,
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let need = cols * rows;
    if cols < 2 || rows < 2 {
        return Err(cv_core::Error::AlgorithmError(
            "pattern_size must be at least (2,2)".to_string(),
        ));
    }
    if image.width() < 8 || image.height() < 8 {
        return Err(cv_core::Error::AlgorithmError(
            "image too small for chessboard detection".to_string(),
        ));
    }

    let (response, width, height) = harris_response(image, 0.04, 1);
    let max_r = response
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    if max_r <= 0.0 {
        return Err(cv_core::Error::AlgorithmError(
            "no chessboard-like corners found".to_string(),
        ));
    }
    let threshold = max_r * 0.01;
    let mut cands = non_max_suppression_response(&response, width, height, threshold);
    if cands.len() < need {
        return Err(cv_core::Error::AlgorithmError(format!(
            "insufficient corner candidates: found {}, need {need}",
            cands.len()
        )));
    }

    cands.sort_by(|a, b| b.2.total_cmp(&a.2));
    // Keep only X-junctions, strongest first, one per corner.
    //
    // The grid used to be assembled from the strongest Harris responses by
    // 1-D k-means along the PCA axes of *all* of them. Harris fires as
    // strongly on the board's outer L-corners as on its inner X-corners, so
    // on an ordinary board (a white margin around the squares) the clusters
    // mixed the two lattices: on a 9x6 board, axis-aligned or rotated, the
    // returned "corners" were up to 52 px from any inner corner, several were
    // the same corner twice, and the order was unrelated to the board's -
    // all returned as `Ok`.
    let mut corners: Vec<Vector2<f64>> = Vec::new();
    for &(x, y, _) in &cands {
        let p = Vector2::new(x, y);
        if corners.iter().any(|c| (c - p).norm() < 3.0) {
            continue;
        }
        if is_x_corner(image, x, y) {
            corners.push(p);
        }
    }
    let mut ordered = assign_grid_points(image, &corners, pattern_size)?;
    corner_subpix(image, &mut ordered, 3, 25, 1e-3)?;
    Ok(ordered)
}

/// Compute Harris corner response for corner detection.
///
/// This function calculates the Harris corner response at each pixel in the image.
/// The Harris response is a measure of how likely a pixel is to be a corner.
fn harris_response(image: &GrayImage, k: f64, win_radius: usize) -> (Vec<f64>, usize, usize) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut ix = vec![0.0f64; width * height];
    let mut iy = vec![0.0f64; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = image.get_pixel((x + 1) as u32, y as u32)[0] as f64
                - image.get_pixel((x - 1) as u32, y as u32)[0] as f64;
            let gy = image.get_pixel(x as u32, (y + 1) as u32)[0] as f64
                - image.get_pixel(x as u32, (y - 1) as u32)[0] as f64;
            ix[y * width + x] = gx * 0.5;
            iy[y * width + x] = gy * 0.5;
        }
    }

    let mut resp = vec![0.0f64; width * height];
    let r = win_radius as i32;
    for y in win_radius..(height - win_radius) {
        for x in win_radius..(width - win_radius) {
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            let mut syy = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let xx = (x as i32 + dx) as usize;
                    let yy = (y as i32 + dy) as usize;
                    let gx = ix[yy * width + xx];
                    let gy = iy[yy * width + xx];
                    sxx += gx * gx;
                    sxy += gx * gy;
                    syy += gy * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            resp[y * width + x] = det - k * trace * trace;
        }
    }
    (resp, width, height)
}

/// Perform non-maximum suppression on the corner response.
///
/// This function suppresses non-maximal responses, keeping only the locally maximum
/// corner responses above the given threshold.
fn non_max_suppression_response(
    response: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let r = response[y * width + x];
            if r <= threshold {
                continue;
            }
            let mut is_max = true;
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    if (xx != x || yy != y) && response[yy * width + xx] > r {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            if is_max {
                out.push((x as f64, y as f64, r));
            }
        }
    }
    out
}

/// Bilinear sample of `image` at `(x, y)`, `None` outside the image.
fn sample_bilinear(image: &GrayImage, x: f64, y: f64) -> Option<f64> {
    let (w, h) = (image.width() as f64, image.height() as f64);
    if !(x >= 0.0 && y >= 0.0 && x <= w - 1.0 && y <= h - 1.0) {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as u32, y0 as u32);
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let p = |x: u32, y: u32| image.get_pixel(x, y)[0] as f64;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// Whether `(x, y)` is a chessboard X-junction (a saddle point).
///
/// Harris responds just as strongly to the L-shaped corners on the board's
/// outer edge, to the corners of the margin and to background clutter; only
/// an *inner* chessboard corner has four alternating sectors around it. On a
/// small circle the intensity of an X-junction crosses its mid level exactly
/// four times and is point-symmetric (opposite sectors have the same colour);
/// an L-corner crosses twice.
fn is_x_corner(image: &GrayImage, x: f64, y: f64) -> bool {
    const N: usize = 24;
    for radius in [3.0f64, 4.5, 6.0] {
        let mut s = [0.0f64; N];
        let mut inside = true;
        for (i, v) in s.iter_mut().enumerate() {
            let a = i as f64 * std::f64::consts::TAU / N as f64;
            match sample_bilinear(image, x + radius * a.cos(), y + radius * a.sin()) {
                Some(value) => *v = value,
                None => {
                    inside = false;
                    break;
                }
            }
        }
        if !inside {
            continue;
        }
        let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if hi - lo < 20.0 {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let bits: Vec<bool> = s.iter().map(|v| *v > mid).collect();
        let transitions = (0..N).filter(|&i| bits[i] != bits[(i + 1) % N]).count();
        let symmetric = (0..N / 2).filter(|&i| bits[i] == bits[i + N / 2]).count();
        if transitions == 4 && symmetric >= N / 2 - 2 {
            return true;
        }
    }
    false
}

/// Indices of the convex hull of `pts`, in order (Andrew's monotone chain;
/// collinear boundary points are dropped).
fn convex_hull(pts: &[Vector2<f64>]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    idx.sort_by(|&a, &b| {
        pts[a]
            .x
            .total_cmp(&pts[b].x)
            .then(pts[a].y.total_cmp(&pts[b].y))
    });
    let cross = |o: usize, a: usize, b: usize| {
        let (oa, ob) = (pts[a] - pts[o], pts[b] - pts[o]);
        oa.x * ob.y - oa.y * ob.x
    };
    let mut hull: Vec<usize> = Vec::with_capacity(2 * idx.len());
    for pass in 0..2 {
        let start = hull.len();
        let order: Box<dyn Iterator<Item = &usize>> = if pass == 0 {
            Box::new(idx.iter())
        } else {
            Box::new(idx.iter().rev())
        };
        for &p in order {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Assign corner candidates to the `cols x rows` grid, in OpenCV's order.
///
/// The inner corners of a board seen through a pinhole camera are the image of
/// a regular lattice under a homography, so their convex hull is a
/// quadrilateral whose vertices are the four outer corners. Each assignment of
/// those four vertices to the lattice corners gives a homography that predicts
/// every other corner; a hypothesis is accepted only if every lattice position
/// has its own candidate within a fraction of the local spacing (then refit on
/// all of them and checked again). Anything else is an error rather than a
/// best guess.
///
/// Among the accepted hypotheses the order is OpenCV's: row-major, `cols`
/// corners per row, the row direction turning clockwise into the column
/// direction in the image (`y` down), and the first corner the one whose
/// inner diagonal square is black. For a board with an odd and an even number
/// of squares along its sides this fixes the order uniquely; measured against
/// `cv2.findChessboardCorners` 4.13 on boards rotated through 360 degrees, and
/// on one with inverted colours, the first corner agrees in every case.
fn assign_grid_points(
    image: &GrayImage,
    candidates: &[Vector2<f64>],
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let need = cols * rows;
    let fail = |why: &str| {
        Err(cv_core::Error::AlgorithmError(format!(
            "chessboard {cols}x{rows} not found: {why}"
        )))
    };
    if candidates.len() < need {
        return fail("too few X-corner candidates");
    }
    let hull = convex_hull(candidates);
    if hull.len() < 4 {
        return fail("candidates do not span a quadrilateral");
    }
    if hull.len() > 80 {
        return fail("candidate hull is not a board outline");
    }
    // The four hull vertices spanning the largest area are the board corners.
    let area = |q: &[usize; 4]| {
        let mut a = 0.0;
        for i in 0..4 {
            let (p, n) = (candidates[q[i]], candidates[q[(i + 1) % 4]]);
            a += p.x * n.y - n.x * p.y;
        }
        0.5 * a.abs()
    };
    let h = hull.len();
    let mut quad = [hull[0], hull[1], hull[2], hull[3]];
    let mut best_area = area(&quad);
    for a in 0..h {
        for b in a + 1..h {
            for c in b + 1..h {
                for d in c + 1..h {
                    let q = [hull[a], hull[b], hull[c], hull[d]];
                    let qa = area(&q);
                    if qa > best_area {
                        best_area = qa;
                        quad = q;
                    }
                }
            }
        }
    }

    let lattice = |i: usize| [(i % cols) as f64, (i / cols) as f64];
    let lattice_corners = [
        [0.0, 0.0],
        [(cols - 1) as f64, 0.0],
        [(cols - 1) as f64, (rows - 1) as f64],
        [0.0, (rows - 1) as f64],
    ];
    let apply = |hm: &Matrix3<f64>, p: [f64; 2]| {
        let v = hm * Vector3::new(p[0], p[1], 1.0);
        Vector2::new(v.x / v.z, v.y / v.z)
    };
    // Every lattice point must own the nearest candidate, closer than a third
    // of the local lattice spacing.
    let assign = |hm: &Matrix3<f64>| -> Option<Vec<usize>> {
        let mut used = vec![false; candidates.len()];
        let mut out = Vec::with_capacity(need);
        for i in 0..need {
            let [u, v] = lattice(i);
            let p = apply(hm, [u, v]);
            let spacing = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
                .iter()
                .map(|d| (apply(hm, [u + d[0], v + d[1]]) - p).norm())
                .fold(f64::INFINITY, f64::min);
            if !spacing.is_finite() || spacing < 2.0 {
                return None;
            }
            let (best, dist) = candidates
                .iter()
                .enumerate()
                .map(|(k, c)| (k, (c - p).norm()))
                .min_by(|a, b| a.1.total_cmp(&b.1))?;
            if used[best] || dist > spacing / 3.0 {
                return None;
            }
            used[best] = true;
            out.push(best);
        }
        Some(out)
    };
    let patch_mean = |hm: &Matrix3<f64>, centre: [f64; 2]| {
        let mut sum = 0.0;
        let mut n = 0.0;
        for dy in [-0.2, 0.0, 0.2] {
            for dx in [-0.2, 0.0, 0.2] {
                let p = apply(hm, [centre[0] + dx, centre[1] + dy]);
                if let Some(v) = sample_bilinear(image, p.x, p.y) {
                    sum += v;
                    n += 1.0;
                }
            }
        }
        if n > 0.0 { sum / n } else { f64::NAN }
    };

    // (black first corner, -(x + y) of the first corner, points)
    let mut best: Option<(bool, f64, Vec<Point2<f64>>)> = None;
    for start in 0..4 {
        for step in [1usize, 3] {
            let img_quad: Vec<[f64; 2]> = (0..4)
                .map(|k| {
                    let c = candidates[quad[(start + k * step) % 4]];
                    [c.x, c.y]
                })
                .collect();
            let Some(h0) = crate::dlt::solve_dlt_homography(&lattice_corners, &img_quad) else {
                continue;
            };
            let Some(first) = assign(&h0) else {
                continue;
            };
            // Refit on every corner and check the assignment again.
            let src: Vec<[f64; 2]> = (0..need).map(lattice).collect();
            let dst: Vec<[f64; 2]> = first
                .iter()
                .map(|&k| [candidates[k].x, candidates[k].y])
                .collect();
            let Some(h1) = crate::dlt::solve_dlt_homography(&src, &dst) else {
                continue;
            };
            let Some(order) = assign(&h1) else {
                continue;
            };
            let pts: Vec<Point2<f64>> = order
                .iter()
                .map(|&k| Point2::new(candidates[k].x, candidates[k].y))
                .collect();
            let row_dir = apply(&h1, [1.0, 0.0]) - apply(&h1, [0.0, 0.0]);
            let col_dir = apply(&h1, [0.0, 1.0]) - apply(&h1, [0.0, 0.0]);
            if row_dir.x * col_dir.y - row_dir.y * col_dir.x <= 0.0 {
                continue;
            }
            let black_first = patch_mean(&h1, [0.5, 0.5]) < patch_mean(&h1, [-0.5, 0.5]);
            let key = (black_first, -(pts[0].x + pts[0].y));
            if best.as_ref().is_none_or(|b| (key.0, key.1) > (b.0, b.1)) {
                best = Some((key.0, key.1, pts));
            }
        }
    }
    match best {
        Some((_, _, pts)) => Ok(pts),
        None => fail("the corner candidates do not form the requested grid"),
    }
}
