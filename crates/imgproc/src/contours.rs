use cv_core::CpuTensor;
use image::GrayImage;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contour {
    pub points: Vec<(i32, i32)>,
}

/// A contour with hole/outer classification from Suzuki-Abe border following.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContourWithHierarchy {
    /// Ordered boundary points as (x, y).
    pub points: Vec<(u32, u32)>,
    /// True if this contour is a hole (inner border), false if outer border.
    pub is_hole: bool,
}

/// A rotated rectangle described by center, size, and angle.
#[derive(Debug, Clone, PartialEq)]
pub struct RotatedRect {
    /// Center x-coordinate.
    pub center_x: f64,
    /// Center y-coordinate.
    pub center_y: f64,
    /// Width of the rectangle (along the rotated axis).
    pub width: f64,
    /// Height of the rectangle (along the rotated axis).
    pub height: f64,
    /// Rotation angle in radians.
    pub angle: f64,
}

/// Image moments up to 3rd order, including central moments.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    /// Spatial moment m00 (area for a filled contour).
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,
    /// Central moments.
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectedComponentStats {
    pub label: u32,
    pub area: u32,
    pub bbox: (u32, u32, u32, u32), // x, y, width, height
    pub centroid: (f64, f64),
}

const DIRS_8: [(i32, i32); 8] = [
    (1, 0),   // E
    (1, 1),   // SE
    (0, 1),   // S
    (-1, 1),  // SW
    (-1, 0),  // W
    (-1, -1), // NW
    (0, -1),  // N
    (1, -1),  // NE
];

fn in_bounds(x: i32, y: i32, w: i32, h: i32) -> bool {
    x >= 0 && y >= 0 && x < w && y < h
}

fn is_foreground(data: &[u8], w: i32, h: i32, x: i32, y: i32) -> bool {
    in_bounds(x, y, w, h) && data[(y * w + x) as usize] > 0
}

fn is_boundary(data: &[u8], w: i32, h: i32, x: i32, y: i32) -> bool {
    if !is_foreground(data, w, h, x, y) {
        return false;
    }
    for (dx, dy) in DIRS_8 {
        let nx = x + dx;
        let ny = y + dy;
        if !in_bounds(nx, ny, w, h) || !is_foreground(data, w, h, nx, ny) {
            return true;
        }
    }
    false
}

fn trace_boundary(data: &[u8], w: i32, h: i32, sx: i32, sy: i32) -> Vec<(i32, i32)> {
    let mut contour = Vec::new();
    let mut current = (sx, sy);
    let mut prev_dir = 4usize; // Start as if we came from W.
    let start = current;
    let start_prev_dir = prev_dir;
    let max_steps = (w as usize * h as usize).saturating_mul(8).max(32);

    for _ in 0..max_steps {
        contour.push(current);

        let mut found = None;
        for step in 1..=8 {
            let k = (prev_dir + step) % 8;
            let nx = current.0 + DIRS_8[k].0;
            let ny = current.1 + DIRS_8[k].1;
            if is_foreground(data, w, h, nx, ny) {
                // Backtrack direction for next step: neighbor before k in clockwise search.
                prev_dir = (k + 6) % 8;
                found = Some((nx, ny));
                break;
            }
        }

        let Some(next) = found else { break };

        if next == start && prev_dir == start_prev_dir && contour.len() > 1 {
            break;
        }
        current = next;
    }

    // Remove trivial duplicates if any.
    if contour.len() > 1 && contour.first() == contour.last() {
        contour.pop();
    }
    contour
}

/// Find external contours in a binary image (non-zero pixels are foreground).
pub fn find_external_contours(binary: &GrayImage) -> Vec<Contour> {
    let w = binary.width() as i32;
    let h = binary.height() as i32;
    let data = binary.as_raw();
    let mut visited_boundary = vec![false; (w * h) as usize];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if visited_boundary[idx] || !is_boundary(data, w, h, x, y) {
                continue;
            }
            let points = trace_boundary(data, w, h, x, y);
            if points.len() >= 3 {
                for &(px, py) in &points {
                    visited_boundary[(py * w + px) as usize] = true;
                }
                contours.push(Contour { points });
            } else {
                visited_boundary[idx] = true;
            }
        }
    }

    contours
}

/// Polygon area (shoelace). Contour should be ordered.
pub fn contour_area(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        area += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
    }
    area.abs() * 0.5
}

/// Closed-contour perimeter.
pub fn contour_perimeter(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Bounding rectangle as (x, y, width, height).
pub fn bounding_rect(contour: &Contour) -> Option<(i32, i32, u32, u32)> {
    if contour.points.is_empty() {
        return None;
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in &contour.points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some((
        min_x,
        min_y,
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
    ))
}

fn cross(o: (i32, i32), a: (i32, i32), b: (i32, i32)) -> i64 {
    let (ox, oy) = o;
    let (ax, ay) = a;
    let (bx, by) = b;
    (ax - ox) as i64 * (by - oy) as i64 - (ay - oy) as i64 * (bx - ox) as i64
}

/// Monotonic chain convex hull.
pub fn convex_hull(contour: &Contour) -> Contour {
    let mut pts: Vec<(i32, i32)> = {
        let mut set = HashSet::new();
        contour
            .points
            .iter()
            .copied()
            .filter(|p| set.insert(*p))
            .collect()
    };
    if pts.len() <= 2 {
        return Contour { points: pts };
    }
    pts.sort_unstable();

    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    Contour { points: lower }
}

fn point_line_distance(p: (i32, i32), a: (i32, i32), b: (i32, i32)) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let dx = bx - ax;
    let dy = by - ay;
    if dx == 0.0 && dy == 0.0 {
        let ex = px - ax;
        let ey = py - ay;
        return (ex * ex + ey * ey).sqrt();
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
    let proj_x = ax + t * dx;
    let proj_y = ay + t * dy;
    let ex = px - proj_x;
    let ey = py - proj_y;
    (ex * ex + ey * ey).sqrt()
}

fn rdp(points: &[(i32, i32)], epsilon: f64, out: &mut Vec<(i32, i32)>) {
    if points.len() < 2 {
        return;
    }
    let first = points[0];
    let last = points[points.len() - 1];
    let mut max_dist = 0.0f64;
    let mut idx = 0usize;
    for (i, &p) in points
        .iter()
        .enumerate()
        .skip(1)
        .take(points.len().saturating_sub(2))
    {
        let d = point_line_distance(p, first, last);
        if d > max_dist {
            max_dist = d;
            idx = i;
        }
    }

    if max_dist > epsilon && idx > 0 {
        rdp(&points[..=idx], epsilon, out);
        out.pop();
        rdp(&points[idx..], epsilon, out);
    } else {
        out.push(first);
        out.push(last);
    }
}

/// Douglas-Peucker contour approximation.
pub fn approx_poly_dp(contour: &Contour, epsilon: f64, closed: bool) -> Contour {
    let mut pts = contour.points.clone();
    if pts.len() < 3 {
        return Contour { points: pts };
    }

    if closed && pts.first() != pts.last() {
        pts.push(pts[0]);
    }

    let mut out = Vec::new();
    rdp(&pts, epsilon.max(0.0), &mut out);

    out.dedup();
    if closed && out.len() > 2 && out.first() == out.last() {
        out.pop();
    }
    Contour { points: out }
}

/// Contour perimeter with explicit open/closed option.
///
/// When `closed` is true, the distance from the last point back to the first is included.
/// When `closed` is false, only consecutive point-to-point distances are summed.
pub fn contour_perimeter_ex(contour: &Contour, closed: bool) -> f64 {
    let n = contour.points.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = 0.0f64;
    let limit = if closed { n } else { n - 1 };
    for i in 0..limit {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Check whether a contour is convex using cross-product sign consistency.
///
/// Returns true if all consecutive cross products have the same sign (or are zero).
/// Contours with fewer than 3 points are considered convex.
pub fn is_convex(contour: &Contour) -> bool {
    let n = contour.points.len();
    if n < 3 {
        return true;
    }
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let o = contour.points[i];
        let a = contour.points[(i + 1) % n];
        let b = contour.points[(i + 2) % n];
        let cp = cross(o, a, b);
        if cp > 0 {
            positive = true;
        } else if cp < 0 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

/// Compute the minimum area enclosing rotated rectangle using rotating calipers.
///
/// Returns a `RotatedRect` with center, size, and rotation angle.
/// For contours with fewer than 3 points, returns a degenerate rectangle.
pub fn min_area_rect(contour: &Contour) -> RotatedRect {
    let hull = convex_hull(contour);
    let pts = &hull.points;

    if pts.is_empty() {
        return RotatedRect {
            center_x: 0.0,
            center_y: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };
    }
    if pts.len() == 1 {
        return RotatedRect {
            center_x: pts[0].0 as f64,
            center_y: pts[0].1 as f64,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };
    }
    if pts.len() == 2 {
        let cx = (pts[0].0 + pts[1].0) as f64 / 2.0;
        let cy = (pts[0].1 + pts[1].1) as f64 / 2.0;
        let dx = (pts[1].0 - pts[0].0) as f64;
        let dy = (pts[1].1 - pts[0].1) as f64;
        let len = (dx * dx + dy * dy).sqrt();
        return RotatedRect {
            center_x: cx,
            center_y: cy,
            width: len,
            height: 0.0,
            angle: dy.atan2(dx),
        };
    }

    // Rotating calipers on convex hull to find minimum area bounding rectangle
    let n = pts.len();
    let mut best_area = f64::MAX;
    let mut best_rect = RotatedRect {
        center_x: 0.0,
        center_y: 0.0,
        width: 0.0,
        height: 0.0,
        angle: 0.0,
    };

    // For each edge of the convex hull, compute the bounding rectangle aligned to that edge
    for i in 0..n {
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let ex = (p2.0 - p1.0) as f64;
        let ey = (p2.1 - p1.1) as f64;
        let edge_len = (ex * ex + ey * ey).sqrt();
        if edge_len < 1e-12 {
            continue;
        }
        // Unit vectors along and perpendicular to edge
        let ux = ex / edge_len;
        let uy = ey / edge_len;

        // Project all hull points onto edge direction and perpendicular
        let mut min_proj = f64::MAX;
        let mut max_proj = f64::NEG_INFINITY;
        let mut min_perp = f64::MAX;
        let mut max_perp = f64::NEG_INFINITY;

        for &(px, py) in pts {
            let dx = px as f64 - p1.0 as f64;
            let dy = py as f64 - p1.1 as f64;
            let proj = dx * ux + dy * uy;
            let perp = -dx * uy + dy * ux;
            if proj < min_proj {
                min_proj = proj;
            }
            if proj > max_proj {
                max_proj = proj;
            }
            if perp < min_perp {
                min_perp = perp;
            }
            if perp > max_perp {
                max_perp = perp;
            }
        }

        let w = max_proj - min_proj;
        let h = max_perp - min_perp;
        let area = w * h;

        if area < best_area {
            best_area = area;
            // Center in the edge-aligned coordinate system, then transform back
            let mid_proj = (min_proj + max_proj) / 2.0;
            let mid_perp = (min_perp + max_perp) / 2.0;
            let cx = p1.0 as f64 + mid_proj * ux - mid_perp * uy;
            let cy = p1.1 as f64 + mid_proj * uy + mid_perp * ux;
            best_rect = RotatedRect {
                center_x: cx,
                center_y: cy,
                width: w,
                height: h,
                angle: uy.atan2(ux),
            };
        }
    }

    best_rect
}

/// Compute image moments of a contour up to 3rd order.
///
/// Uses the Green's theorem formulation for polygon moments.
/// The contour is treated as a closed polygon.
pub fn moments(contour: &Contour) -> Moments {
    let n = contour.points.len();
    let zero = Moments {
        m00: 0.0,
        m10: 0.0,
        m01: 0.0,
        m20: 0.0,
        m11: 0.0,
        m02: 0.0,
        m30: 0.0,
        m21: 0.0,
        m12: 0.0,
        m03: 0.0,
        mu20: 0.0,
        mu11: 0.0,
        mu02: 0.0,
        mu30: 0.0,
        mu21: 0.0,
        mu12: 0.0,
        mu03: 0.0,
    };

    if n < 3 {
        return zero;
    }

    // Compute raw spatial moments using Green's theorem for polygons
    let mut m00 = 0.0f64;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    let mut m20 = 0.0f64;
    let mut m11 = 0.0f64;
    let mut m02 = 0.0f64;
    let mut m30 = 0.0f64;
    let mut m21 = 0.0f64;
    let mut m12 = 0.0f64;
    let mut m03 = 0.0f64;

    for i in 0..n {
        let (xi, yi) = (contour.points[i].0 as f64, contour.points[i].1 as f64);
        let j = (i + 1) % n;
        let (xj, yj) = (contour.points[j].0 as f64, contour.points[j].1 as f64);
        let a = xi * yj - xj * yi; // cross product term

        m00 += a;
        m10 += a * (xi + xj);
        m01 += a * (yi + yj);
        m20 += a * (xi * xi + xi * xj + xj * xj);
        m11 += a * (2.0 * xi * yi + xi * yj + xj * yi + 2.0 * xj * yj);
        m02 += a * (yi * yi + yi * yj + yj * yj);
        m30 += a * (xi + xj) * (xi * xi + xj * xj);
        m21 +=
            a * (xi * xi * (3.0 * yi + yj) + 2.0 * xi * xj * (yi + yj) + xj * xj * (yi + 3.0 * yj));
        m12 +=
            a * (yi * yi * (3.0 * xi + xj) + 2.0 * yi * yj * (xi + xj) + yj * yj * (xi + 3.0 * xj));
        m03 += a * (yi + yj) * (yi * yi + yj * yj);
    }

    m00 /= 2.0;
    m10 /= 6.0;
    m01 /= 6.0;
    m20 /= 12.0;
    m11 /= 24.0;
    m02 /= 12.0;
    m30 /= 20.0;
    m21 /= 60.0;
    m12 /= 60.0;
    m03 /= 20.0;

    // Make moments positive (orientation-independent)
    if m00 < 0.0 {
        m00 = -m00;
        m10 = -m10;
        m01 = -m01;
        m20 = -m20;
        m11 = -m11;
        m02 = -m02;
        m30 = -m30;
        m21 = -m21;
        m12 = -m12;
        m03 = -m03;
    }

    // Central moments
    let x_bar = if m00.abs() > 1e-12 { m10 / m00 } else { 0.0 };
    let y_bar = if m00.abs() > 1e-12 { m01 / m00 } else { 0.0 };

    let mu20 = m20 - x_bar * m10;
    let mu11 = m11 - x_bar * m01;
    let mu02 = m02 - y_bar * m01;
    let mu30 = m30 - 3.0 * x_bar * m20 + 2.0 * x_bar * x_bar * m10;
    let mu21 = m21 - 2.0 * x_bar * m11 - y_bar * m20 + 2.0 * x_bar * x_bar * m01;
    let mu12 = m12 - 2.0 * y_bar * m11 - x_bar * m02 + 2.0 * y_bar * y_bar * m10;
    let mu03 = m03 - 3.0 * y_bar * m02 + 2.0 * y_bar * y_bar * m01;

    Moments {
        m00,
        m10,
        m01,
        m20,
        m11,
        m02,
        m30,
        m21,
        m12,
        m03,
        mu20,
        mu11,
        mu02,
        mu30,
        mu21,
        mu12,
        mu03,
    }
}

/// Find contours in a binary `CpuTensor` using simplified Suzuki-Abe border following.
///
/// Input must be a single-channel tensor. Non-zero values are foreground.
/// Returns a list of `ContourWithHierarchy` structs, each with an `is_hole` flag.
pub fn find_contours<T>(binary: &CpuTensor<T>) -> crate::Result<Vec<ContourWithHierarchy>>
where
    T: Clone + Copy + Default + PartialEq + std::fmt::Debug + Into<f64> + 'static,
{
    if binary.shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Input must be a single-channel image".into(),
        ));
    }

    let h = binary.shape.height as i32;
    let w = binary.shape.width as i32;
    let src = binary.as_slice()?;

    // Build a working copy as i32: 0 = background, 1 = foreground (unvisited)
    let total = (h * w) as usize;
    let mut img = vec![0i32; total];
    for i in 0..total {
        let v: f64 = src[i].into();
        if v != 0.0 {
            img[i] = 1;
        }
    }

    let mut contours = Vec::new();
    let mut nbd: i32 = 1; // current border sequential number

    // Suzuki-Abe simplified: scan row by row
    for y in 0..h {
        let mut lnbd: i32 = 1; // last encountered non-zero border
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let fxy = img[idx];

            // Determine border type
            let is_outer = fxy == 1 && (x == 0 || img[idx - 1] == 0);
            let is_hole = fxy >= 1 && (x == w - 1 || img[idx + 1] == 0);

            if !is_outer && !is_hole {
                if fxy != 0 && fxy != 1 {
                    lnbd = fxy.unsigned_abs() as i32;
                }
                continue;
            }

            let border_is_hole;
            let start_dir;
            if is_outer {
                nbd += 1;
                border_is_hole = false;
                start_dir = 0usize; // start search from W (left)
            } else if is_hole && fxy > 0 {
                nbd += 1;
                border_is_hole = true;
                start_dir = 4usize; // start search from E (right)
            } else {
                if fxy != 0 && fxy != 1 {
                    lnbd = fxy.unsigned_abs() as i32;
                }
                continue;
            }

            // Trace contour using Moore boundary tracing
            let mut points = Vec::new();
            let mut cx = x;
            let mut cy = y;

            // Find first neighbor
            let mut dir = start_dir;
            let mut found_first = false;
            for step in 0..8 {
                let d = (dir + step) % 8;
                let nx = cx + DIRS_8[d].0;
                let ny = cy + DIRS_8[d].1;
                if in_bounds(nx, ny, w, h) && img[(ny * w + nx) as usize] != 0 {
                    dir = d;
                    found_first = true;
                    break;
                }
            }

            if !found_first {
                // Isolated pixel
                img[idx] = -nbd;
                points.push((x as u32, y as u32));
                contours.push(ContourWithHierarchy {
                    points,
                    is_hole: border_is_hole,
                });
                lnbd = nbd;
                continue;
            }

            // Follow the border
            let start_x = cx;
            let start_y = cy;
            let start_dir_val = dir;
            let max_iter = (total * 8).max(64);
            let mut first_step = true;

            for _ in 0..max_iter {
                points.push((cx as u32, cy as u32));

                // Search for next boundary pixel clockwise from (dir+4+2)%8
                let search_start = (dir + 6) % 8; // start from direction opposite to where we came, +1 CW
                let mut next_found = false;
                let mut next_dir = 0;
                let mut nx = 0i32;
                let mut ny = 0i32;

                for step in 0..8 {
                    let d = (search_start + step) % 8;
                    let tx = cx + DIRS_8[d].0;
                    let ty = cy + DIRS_8[d].1;
                    if in_bounds(tx, ty, w, h) && img[(ty * w + tx) as usize] != 0 {
                        next_dir = d;
                        nx = tx;
                        ny = ty;
                        next_found = true;
                        break;
                    }
                }

                if !next_found {
                    break;
                }

                // Mark pixel
                let cidx = (cy * w + cx) as usize;
                if img[cidx] == 1 {
                    img[cidx] = nbd;
                } else if img[cidx] > 1 {
                    // already part of another border, mark negatively if on edge
                }

                cx = nx;
                cy = ny;
                dir = next_dir;

                if !first_step && cx == start_x && cy == start_y {
                    break;
                }
                first_step = false;
            }

            // Remove duplicate if last == first
            if points.len() > 1 && points.first() == points.last() {
                points.pop();
            }

            // Deduplicate consecutive identical points
            points.dedup();

            if !points.is_empty() {
                contours.push(ContourWithHierarchy {
                    points,
                    is_hole: border_is_hole,
                });
            }

            let _ = start_dir_val; // suppress warning
            let _ = lnbd;
            lnbd = nbd;
        }
    }

    Ok(contours)
}

/// Connected components labeling with component statistics.
///
/// Returns `(labels, num_labels, stats)` where:
/// - `labels` is row-major label map (0 is background)
/// - `num_labels` includes background label 0
/// - `stats` only contains foreground components (labels 1..)
pub fn connected_components_with_stats(
    binary: &GrayImage,
    connectivity: u8,
) -> (Vec<u32>, u32, Vec<ConnectedComponentStats>) {
    let w = binary.width() as i32;
    let h = binary.height() as i32;
    let data = binary.as_raw();
    let mut labels = vec![0u32; (w * h) as usize];
    let mut stats = Vec::new();
    let mut next_label = 1u32;

    let neigh_4: &[(i32, i32)] = &[(1, 0), (-1, 0), (0, 1), (0, -1)];
    let neigh_8: &[(i32, i32)] = &[
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let neigh = if connectivity == 4 { neigh_4 } else { neigh_8 };

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if data[idx] == 0 || labels[idx] != 0 {
                continue;
            }

            let label = next_label;
            next_label += 1;

            let mut q = VecDeque::new();
            q.push_back((x, y));
            labels[idx] = label;

            let mut area = 0u32;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut sum_x = 0f64;
            let mut sum_y = 0f64;

            while let Some((cx, cy)) = q.pop_front() {
                area += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                sum_x += cx as f64;
                sum_y += cy as f64;

                for &(dx, dy) in neigh {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if !in_bounds(nx, ny, w, h) {
                        continue;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if data[nidx] == 0 || labels[nidx] != 0 {
                        continue;
                    }
                    labels[nidx] = label;
                    q.push_back((nx, ny));
                }
            }

            let centroid = (sum_x / area as f64, sum_y / area as f64);
            let bbox = (
                min_x as u32,
                min_y as u32,
                (max_x - min_x + 1) as u32,
                (max_y - min_y + 1) as u32,
            );
            stats.push(ConnectedComponentStats {
                label,
                area,
                bbox,
                centroid,
            });
        }
    }

    (labels, next_label, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn finds_rectangle_contour_and_descriptors() {
        let mut img = GrayImage::new(32, 24);
        for y in 6..18 {
            for x in 8..22 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let contours = find_external_contours(&img);
        assert!(!contours.is_empty());
        let c = &contours[0];

        let rect = bounding_rect(c).unwrap();
        assert_eq!(rect.0, 8);
        assert_eq!(rect.1, 6);
        assert_eq!(rect.2, 14);
        assert_eq!(rect.3, 12);

        let a = contour_area(c);
        let p = contour_perimeter(c);
        assert!(a > 0.0);
        assert!(p > 0.0);
    }

    #[test]
    fn convex_hull_not_empty() {
        let c = Contour {
            points: vec![(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 1)],
        };
        let h = convex_hull(&c);
        assert!(h.points.len() >= 3);
    }

    #[test]
    fn approx_poly_dp_reduces_points() {
        let c = Contour {
            points: vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (2, 3),
                (1, 3),
                (0, 3),
                (0, 2),
                (0, 1),
            ],
        };
        let a = approx_poly_dp(&c, 0.8, true);
        assert!(a.points.len() < c.points.len());
        assert!(a.points.len() >= 4);
    }

    #[test]
    fn connected_components_stats_basic() {
        let mut img = GrayImage::new(16, 16);
        // Component 1: 3x3 block
        for y in 2..5 {
            for x in 3..6 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        // Component 2: 2x4 block
        for y in 10..14 {
            for x in 11..13 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let (_labels, num_labels, stats) = connected_components_with_stats(&img, 8);
        assert_eq!(num_labels, 3); // background + 2 components
        assert_eq!(stats.len(), 2);
        let total_area: u32 = stats.iter().map(|s| s.area).sum();
        assert_eq!(total_area, 9 + 8);
    }

    #[test]
    fn test_contour_perimeter_open_vs_closed() {
        // A simple triangle
        let c = Contour {
            points: vec![(0, 0), (3, 0), (0, 4)],
        };
        let closed = contour_perimeter_ex(&c, true);
        let open = contour_perimeter_ex(&c, false);
        // Closed should include the return edge (0,4)->(0,0) = 4.0
        // Open should not
        assert!(closed > open);
        // Open = 3 + 5 = 8, Closed = 3 + 5 + 4 = 12
        assert!((open - 8.0).abs() < 1e-9);
        assert!((closed - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_convex_square() {
        let c = Contour {
            points: vec![(0, 0), (4, 0), (4, 4), (0, 4)],
        };
        assert!(is_convex(&c));
    }

    #[test]
    fn test_is_convex_concave() {
        // L-shape is not convex
        let c = Contour {
            points: vec![(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)],
        };
        assert!(!is_convex(&c));
    }

    #[test]
    fn test_min_area_rect_axis_aligned() {
        // Axis-aligned rectangle: corners at (0,0),(10,0),(10,5),(0,5)
        let c = Contour {
            points: vec![(0, 0), (10, 0), (10, 5), (0, 5)],
        };
        let r = min_area_rect(&c);
        // Area should be 50
        assert!((r.width * r.height - 50.0).abs() < 1.0);
        // Center should be near (5, 2.5)
        assert!((r.center_x - 5.0).abs() < 0.5);
        assert!((r.center_y - 2.5).abs() < 0.5);
    }

    #[test]
    fn test_moments_rectangle() {
        // Rectangle with vertices: (0,0), (4,0), (4,3), (0,3)
        let c = Contour {
            points: vec![(0, 0), (4, 0), (4, 3), (0, 3)],
        };
        let m = moments(&c);
        // m00 = area = 12
        assert!((m.m00 - 12.0).abs() < 1e-6);
        // Centroid at (2, 1.5)
        let cx = m.m10 / m.m00;
        let cy = m.m01 / m.m00;
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_moments_triangle() {
        // Right triangle (0,0), (6,0), (0,4)
        let c = Contour {
            points: vec![(0, 0), (6, 0), (0, 4)],
        };
        let m = moments(&c);
        // Area = 0.5 * 6 * 4 = 12
        assert!((m.m00 - 12.0).abs() < 1e-6);
        // Centroid at (2, 4/3)
        let cx = m.m10 / m.m00;
        let cy = m.m01 / m.m00;
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_contours_tensor_rectangle() {
        use cv_core::TensorShape;
        // 10x10 image with a filled rectangle at (2,2)-(6,6)
        let mut data = vec![0.0f32; 100];
        for y in 2..7 {
            for x in 2..7 {
                data[y * 10 + x] = 1.0;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 10, 10)).unwrap();
        let contours = find_contours(&img).unwrap();
        assert!(!contours.is_empty());
        // Should find at least one outer contour
        let outer: Vec<_> = contours.iter().filter(|c| !c.is_hole).collect();
        assert!(!outer.is_empty());
    }

    #[test]
    fn test_find_contours_tensor_with_hole() {
        use cv_core::TensorShape;
        // 12x12 image: filled square with a hole inside
        let mut data = vec![0.0f32; 144];
        for y in 1..11 {
            for x in 1..11 {
                data[y * 12 + x] = 1.0;
            }
        }
        // Create hole: clear center
        for y in 4..8 {
            for x in 4..8 {
                data[y * 12 + x] = 0.0;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 12, 12)).unwrap();
        let contours = find_contours(&img).unwrap();
        assert!(contours.len() >= 2); // outer border + hole
        let holes: Vec<_> = contours.iter().filter(|c| c.is_hole).collect();
        assert!(!holes.is_empty(), "Should detect at least one hole contour");
    }
}
