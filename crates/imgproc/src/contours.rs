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
    let max_steps = (w as usize * h as usize).saturating_mul(8).max(32);
    // Terminate on the SECOND arrival at the start pixel regardless of entry
    // direction. The previous criterion (same entry direction required) never
    // fired for structures whose trace immediately doubles back — e.g. two
    // adjacent pixels alternate A→B→A forever, producing a multi-million
    // point garbage contour.
    let mut returned_to_start = 0usize;

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

        let Some(next) = found else {
            break;
        };

        if next == start && contour.len() > 1 {
            returned_to_start += 1;
            if returned_to_start >= 1 {
                // Loop closed: the start pixel is already the first element.
                break;
            }
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

/// OpenCV 4.13's `approxPolyDP(curve, epsilon, closed)` on integer points:
/// Ramer-Douglas-Peucker, returning a subset of the contour's points.
///
/// A closed curve is not split at its first point: the algorithm first
/// looks for two far-apart points (from point 0 to the point farthest from
/// it, then to the point farthest from that one, three times) and simplifies
/// the two arcs between them, so the output starts at one of those points.
/// An open curve keeps both end points (unless they coincide, in which case
/// it is treated as closed). A point's deviation is its distance to the
/// chord *segment*. A final pass drops vertices lying within
/// `epsilon / sqrt(2)` of the line through their neighbours when the turn is
/// not back-tracking and the neighbours differ in both `x` and `y`, exactly
/// as OpenCV does. A negative or NaN `epsilon` is treated as zero (OpenCV
/// rejects it).
pub fn approx_poly_dp(contour: &Contour, epsilon: f64, closed: bool) -> Contour {
    let pts = &contour.points;
    let keep = approx_poly_dp_indices(
        pts.len(),
        |i| (pts[i].0 as f64, pts[i].1 as f64),
        epsilon,
        closed,
    );
    Contour {
        points: keep.into_iter().map(|i| pts[i]).collect(),
    }
}

/// `approxPolyDP_` on `count` points read through `pt`, returning the
/// indices of the points kept, in output order. Coordinates and their
/// differences are taken in `f64`, which for integer points is exactly
/// OpenCV's arithmetic.
pub(crate) fn approx_poly_dp_indices(
    count: usize,
    pt: impl Fn(usize) -> (f64, f64),
    epsilon: f64,
    closed: bool,
) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let eps = epsilon.max(0.0);
    let eps = eps * eps;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut dst: Vec<usize> = Vec::with_capacity(count);
    let mut is_closed = closed;
    let mut init_iters = 3;

    if !is_closed {
        let (sx, sy) = pt(count - 1);
        let (ex, ey) = pt(0);
        if sx != ex || sy != ey {
            stack.push((0, count - 1));
        } else {
            is_closed = true;
            init_iters = 1;
        }
    }

    if is_closed {
        // 1. Approximately the two farthest points of the curve.
        let mut right_start = 0usize;
        let mut pos = 0usize;
        let mut le_eps = false;
        for _ in 0..init_iters {
            pos = (pos + right_start) % count;
            let (sx, sy) = pt(pos);
            let mut max_dist = 0.0;
            for j in 1..count {
                let (x, y) = pt((pos + j) % count);
                let (dx, dy) = (x - sx, y - sy);
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    right_start = j;
                }
            }
            le_eps = max_dist <= eps;
        }
        // 2. Seed the stack with the two arcs between them.
        if !le_eps {
            let start = pos % count;
            let far = (right_start + start) % count;
            stack.push((far, start));
            stack.push((start, far));
        } else {
            dst.push(pos);
        }
    }

    // 3. Split each slice at its point farthest from the chord segment.
    while let Some((s0, s1)) = stack.pop() {
        let (ex, ey) = pt(s1);
        let (sx, sy) = pt(s0);
        let mut pos = (s0 + 1) % count;
        let le_eps = if pos != s1 {
            let (dx, dy) = (ex - sx, ey - sy);
            let segment_len_2 = dx * dx + dy * dy;
            let mut max_dist = 0.0;
            let mut split = s0;
            while pos != s1 {
                let (x, y) = pt(pos);
                let projection = (x - sx) * dx + (y - sy) * dy;
                let dist = if projection < 0.0 {
                    ((x - sx) * (x - sx) + (y - sy) * (y - sy)) * segment_len_2
                } else if projection > segment_len_2 {
                    ((x - ex) * (x - ex) + (y - ey) * (y - ey)) * segment_len_2
                } else {
                    let d = (y - sy) * dx - (x - sx) * dy;
                    d * d
                };
                if dist > max_dist {
                    max_dist = dist;
                    split = pos;
                }
                pos = (pos + 1) % count;
            }
            if max_dist <= eps * segment_len_2 {
                true
            } else {
                stack.push((split, s1));
                stack.push((s0, split));
                false
            }
        } else {
            true
        };
        if le_eps {
            dst.push(s0);
        }
    }
    if !is_closed {
        dst.push(count - 1);
    }

    // 4. Clean-up: drop vertices on an almost straight line, reading ahead
    // in place exactly as OpenCV does.
    let is_closed = closed;
    let count = dst.len();
    let mut new_count = count;
    let open = usize::from(!is_closed);
    let mut pos = if is_closed { count - 1 } else { 0 };
    let read = |pos: &mut usize, dst: &[usize]| {
        let v = dst[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        v
    };
    let mut start = read(&mut pos, &dst);
    let mut wpos = pos;
    let mut cur = read(&mut pos, &dst);
    let mut i = open;
    while i < count - open && new_count > 2 {
        let end = read(&mut pos, &dst);
        let (sx, sy) = pt(start);
        let (px, py) = pt(cur);
        let (ex, ey) = pt(end);
        let (dx, dy) = (ex - sx, ey - sy);
        let dist = ((px - sx) * dy - (py - sy) * dx).abs();
        let successive_inner_product = (px - sx) * (ex - px) + (py - sy) * (ey - py);
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy)
            && dx != 0.0
            && dy != 0.0
            && successive_inner_product >= 0.0
        {
            new_count -= 1;
            start = end;
            dst[wpos] = end;
            wpos += 1;
            if wpos >= count {
                wpos = 0;
            }
            cur = read(&mut pos, &dst);
            i += 2;
            continue;
        }
        start = cur;
        dst[wpos] = cur;
        wpos += 1;
        if wpos >= count {
            wpos = 0;
        }
        cur = end;
        i += 1;
    }
    if !is_closed {
        dst[wpos] = cur;
    }
    dst.truncate(new_count);
    dst
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

/// Which contours [`find_contours_with`] returns and how they are linked
/// (OpenCV's `RETR_*` modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourRetrieval {
    /// Only the outermost borders (`RETR_EXTERNAL`).
    External,
    /// Every border, outer and hole, with no nesting (`RETR_LIST`).
    List,
    /// Every border in a two-level hierarchy: outer borders at the top, the
    /// holes of each component as its children (`RETR_CCOMP`).
    CComp,
    /// Every border in the full nesting tree (`RETR_TREE`).
    Tree,
}

/// How [`find_contours_with`] stores each border (OpenCV's `CHAIN_APPROX_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximation {
    /// Every border pixel (`CHAIN_APPROX_NONE`).
    None,
    /// Only the end points of horizontal, vertical and diagonal runs
    /// (`CHAIN_APPROX_SIMPLE`).
    Simple,
}

/// One entry of the contour hierarchy, OpenCV's layout:
/// `[next, previous, first_child, parent]`, indices into the contour list,
/// `-1` where there is none.
pub type ContourHierarchy = [i32; 4];

/// Every border of the foreground of a single-channel `binary` tensor, as
/// OpenCV's `findContours(image, RETR_LIST, CHAIN_APPROX_NONE)`: outer
/// borders and hole borders alike, every border pixel in order, contours in
/// OpenCV's order. Pixels greater than zero are foreground.
///
/// See [`find_contours_with`] for the other retrieval modes, the simple chain
/// approximation and the hierarchy.
pub fn find_contours<T>(binary: &CpuTensor<T>) -> crate::Result<Vec<ContourWithHierarchy>>
where
    T: Clone + Copy + Default + PartialEq + std::fmt::Debug + Into<f64> + 'static,
{
    find_contours_with(binary, ContourRetrieval::List, ContourApproximation::None)
        .map(|(contours, _)| contours)
}

/// OpenCV 4.13's `findContours(image, mode, method)` on a single-channel
/// tensor: Suzuki-Abe border following, returning the contours and their
/// hierarchy (one [`ContourHierarchy`] entry per contour) in exactly OpenCV's
/// order, with the same start point and direction for every border.
///
/// Pixels greater than zero are foreground (OpenCV thresholds the image at
/// zero). Pixels on the image edge are traced like any other: the image is
/// treated as surrounded by background. An outer border runs
/// counter-clockwise on screen from its top-left pixel; a hole border runs
/// clockwise from the pixel left of its top-left background pixel. With
/// [`ContourRetrieval::List`] and [`ContourRetrieval::External`] every
/// contour is at the top level, in the reverse of the order the raster scan
/// met them; with `CComp` and `Tree` the list is the tree in pre-order,
/// siblings again latest first.
pub fn find_contours_with<T>(
    binary: &CpuTensor<T>,
    mode: ContourRetrieval,
    method: ContourApproximation,
) -> crate::Result<(Vec<ContourWithHierarchy>, Vec<ContourHierarchy>)>
where
    T: Clone + Copy + Default + PartialEq + std::fmt::Debug + Into<f64> + 'static,
{
    if binary.shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Input must be a single-channel image".into(),
        ));
    }
    let (h, w) = (binary.shape.height, binary.shape.width);
    let src = binary.as_slice()?;
    if w == 0 || h == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let (contours, hierarchy) = suzuki::find(src, w, h, mode, method, |v| {
        let v: f64 = v.into();
        v > 0.0
    });
    let contours = contours
        .into_iter()
        .map(|(points, is_hole)| ContourWithHierarchy {
            points: points
                .into_iter()
                .map(|(x, y)| (x as u32, y as u32))
                .collect(),
            is_hole,
        })
        .collect();
    Ok((contours, hierarchy))
}

/// Port of OpenCV 4.13's `contours_new.cpp` (8-bit image path).
mod suzuki {
    use super::{ContourApproximation, ContourHierarchy, ContourRetrieval};

    /// Chain-code steps (OpenCV's `chainCodeDeltas`): 0 east, then
    /// counter-clockwise on screen (`y` down).
    const DELTAS: [(i32, i32); 8] = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    /// Label flag of a pixel whose east neighbour is background.
    const RIGHT: u8 = 0x80;
    /// Label of every border pixel in the modes without a hierarchy.
    const NEW: i8 = 2;
    const LVAL: i8 = 0x7f;

    struct Node {
        parent: i32,
        first_child: i32,
        prev: i32,
        next: i32,
        ctable_next: i32,
        is_hole: bool,
        /// Bounding box `(x, y, width, height)` in padded coordinates.
        brect: (i32, i32, i32, i32),
        /// Start pixel in padded coordinates.
        origin: (i32, i32),
        points: Vec<(i32, i32)>,
    }

    impl Node {
        fn new(is_hole: bool) -> Self {
            Node {
                parent: -1,
                first_child: -1,
                prev: -1,
                next: -1,
                ctable_next: -1,
                is_hole,
                brect: (0, 0, 0, 0),
                origin: (0, 0),
                points: Vec::new(),
            }
        }
    }

    struct Scanner {
        /// The image with a one-pixel background frame: 0 background,
        /// 1 unvisited foreground, otherwise a border label (bit 7 set:
        /// the "right" flag).
        img: Vec<i8>,
        step: i32,
        width: i32,
        height: i32,
        pt: (i32, i32),
        lnbd: (i32, i32),
        nbd: i8,
        mode: ContourRetrieval,
        direct: bool,
        nodes: Vec<Node>,
        ctable: [i32; 128],
    }

    fn delta(s: i32, step: i32) -> isize {
        let (dx, dy) = DELTAS[(s & 7) as usize];
        (dx + dy * step) as isize
    }

    fn set_right(img: &mut [i8], i: isize, nbd: i8) {
        img[i as usize] = (nbd as u8 | RIGHT) as i8;
    }

    impl Scanner {
        fn at(&self, i: isize) -> i8 {
            self.img[i as usize]
        }

        fn idx(&self, (x, y): (i32, i32)) -> isize {
            (y * self.step + x) as isize
        }

        fn is_simple(&self) -> bool {
            matches!(
                self.mode,
                ContourRetrieval::External | ContourRetrieval::List
            )
        }

        /// `icvFetchContourEx`: follow the border starting at `start`,
        /// labelling it with `nbd`, and return its points (image
        /// coordinates, starting at `origin`) and its bounding box.
        fn fetch(
            &mut self,
            start: (i32, i32),
            nbd: i8,
            is_hole: bool,
            origin: (i32, i32),
        ) -> (Vec<(i32, i32)>, (i32, i32, i32, i32)) {
            let step = self.step;
            let i0 = self.idx(start);
            let mut pt = origin;
            let mut points = Vec::new();
            // (min x, min y, max x, max y) until the end.
            let mut rect = (pt.0, pt.1, pt.0, pt.1);

            let mut s_end: i32 = if is_hole { 0 } else { 4 };
            let mut s = s_end;
            let mut i1;
            loop {
                s = (s - 1) & 7;
                i1 = i0 + delta(s, step);
                if self.at(i1) != 0 || s == s_end {
                    break;
                }
            }

            if s == s_end {
                set_right(&mut self.img, i0, nbd);
                points.push(pt);
            } else {
                let mut i3 = i0;
                let mut i4;
                let mut prev_s = s ^ 4;
                loop {
                    s_end = s;
                    s = s.min(15);
                    loop {
                        s += 1;
                        i4 = i3 + delta(s, step);
                        if self.at(i4) != 0 || s >= 15 {
                            break;
                        }
                    }
                    s &= 7;

                    if ((s - 1) as u32) < s_end as u32 {
                        set_right(&mut self.img, i3, nbd);
                    } else if self.at(i3) == 1 {
                        self.img[i3 as usize] = nbd;
                    }

                    if s != prev_s || self.direct {
                        points.push(pt);
                    }
                    if s != prev_s {
                        if pt.0 < rect.0 {
                            rect.0 = pt.0;
                        } else if pt.0 > rect.2 {
                            rect.2 = pt.0;
                        }
                        if pt.1 < rect.1 {
                            rect.1 = pt.1;
                        } else if pt.1 > rect.3 {
                            rect.3 = pt.1;
                        }
                    }
                    prev_s = s;
                    pt.0 += DELTAS[s as usize].0;
                    pt.1 += DELTAS[s as usize].1;

                    if i4 == i0 && i3 == i1 {
                        break;
                    }
                    i3 = i4;
                    s = (s + 4) & 7;
                }
            }
            let brect = (rect.0, rect.1, rect.2 - rect.0 + 1, rect.3 - rect.1 + 1);
            (points, brect)
        }

        /// `icvTraceContour`: whether following the border that starts at
        /// `start` reaches `end` as the last border met by the raster scan.
        fn trace(&self, start: (i32, i32), end: (i32, i32), is_hole: bool) -> bool {
            let step = self.step;
            let stop = self.idx(end);
            let i0 = self.idx(start);
            let s_end: i32 = if is_hole { 0 } else { 4 };
            let mut s = s_end;
            let mut i1;
            loop {
                s = (s - 1) & 7;
                i1 = i0 + delta(s, step);
                if self.at(i1) != 0 || s == s_end {
                    break;
                }
            }
            let mut i3 = i0;
            if s == s_end {
                return i3 == stop;
            }
            loop {
                s = s.min(15);
                let mut i4;
                loop {
                    s += 1;
                    i4 = i3 + delta(s, step);
                    if self.at(i4) != 0 || s >= 15 {
                        break;
                    }
                }
                if i3 == stop {
                    if (self.at(i3) as u8 & RIGHT) == 0 {
                        return true;
                    }
                    let mut t = s;
                    loop {
                        t = (t - 1) & 7;
                        if self.at(i3 + delta(t, step)) != 0 {
                            break;
                        }
                        if t == 0 {
                            return true;
                        }
                    }
                }
                if i4 == i0 && i3 == i1 {
                    return false;
                }
                i3 = i4;
                s = (s + 4) & 7;
            }
        }

        fn add_child(&mut self, parent: i32, child: i32) {
            let first = self.nodes[parent as usize].first_child;
            if first != -1 {
                self.nodes[first as usize].prev = child;
                self.nodes[child as usize].next = first;
            }
            self.nodes[parent as usize].first_child = child;
            let c = &mut self.nodes[child as usize];
            c.parent = parent;
            c.prev = -1;
        }

        fn make_contour(&mut self, nbd: &mut i8, is_hole: bool, x: i32, y: i32) -> i32 {
            let start = (x - i32::from(is_hole), y);
            // The padding frame shifts image coordinates by one.
            let origin = (start.0 - 1, start.1 - 1);
            let id = self.nodes.len() as i32;
            let mut node = Node::new(is_hole);
            if self.is_simple() {
                let (points, brect) = self.fetch(start, NEW, is_hole, origin);
                node.points = points;
                node.brect = brect;
            } else {
                let lval = *nbd;
                *nbd = nbd.wrapping_add(1) & LVAL;
                if *nbd == 0 {
                    *nbd = 3;
                }
                let (points, brect) = self.fetch(start, lval, is_hole, origin);
                node.points = points;
                node.brect = (brect.0 + 1, brect.1 + 1, brect.2, brect.3);
                node.ctable_next = self.ctable[lval as usize];
                self.ctable[lval as usize] = id;
            }
            node.origin = start;
            self.nodes.push(node);
            id
        }

        fn find_first_bounding_contour(&self, last_pos: (i32, i32), y: i32, lval: i32) -> i32 {
            let end = (last_pos.0, y);
            let mut res = -1;
            let mut cur = self.ctable[lval as usize];
            while cur != -1 {
                let node = &self.nodes[cur as usize];
                let (bx, by, bw, bh) = node.brect;
                if last_pos.0 - bx < bw && last_pos.1 - by < bh {
                    if res != -1 {
                        let r = &self.nodes[res as usize];
                        if self.trace(r.origin, end, r.is_hole) {
                            break;
                        }
                    }
                    res = cur;
                }
                cur = node.ctable_next;
            }
            res
        }

        fn contour_scan(
            &mut self,
            prev: i32,
            p: i32,
            last_pos: &mut (i32, i32),
            x: i32,
            y: i32,
        ) -> bool {
            let mut is_hole = false;
            if !(prev == 0 && p == 1) {
                if p != 0 || prev < 1 {
                    return false;
                }
                if prev & !1 != 0 {
                    last_pos.0 = x - 1;
                }
                is_hole = true;
            }
            if self.mode == ContourRetrieval::External
                && (is_hole || self.at(self.idx(*last_pos)) > 0)
            {
                return false;
            }

            let main_parent = if self.is_simple()
                || (!is_hole && self.mode == ContourRetrieval::CComp)
                || last_pos.0 <= 0
            {
                0
            } else {
                let lval = i32::from(self.at(self.idx(*last_pos)) & LVAL);
                let mut parent = self.find_first_bounding_contour(*last_pos, y, lval);
                if parent < 0 {
                    // Unreachable for a consistent labelling (OpenCV would
                    // index out of range); attach to the root instead.
                    parent = 0;
                }
                let node = &self.nodes[parent as usize];
                if node.is_hole == is_hole {
                    parent = node.parent.max(0);
                }
                parent
            };

            last_pos.0 = x - i32::from(is_hole);
            let mut nbd = self.nbd;
            let id = self.make_contour(&mut nbd, is_hole, x, y);
            self.add_child(main_parent, id);
            self.pt = (x + 1, y);
            self.nbd = nbd;
            true
        }

        fn find_next(&mut self) -> bool {
            let (mut x, mut y) = self.pt;
            let mut last_pos = self.lnbd;
            let mut prev = i32::from(self.at(self.idx((x - 1, y))));
            while y < self.height {
                let mut p;
                while x < self.width {
                    loop {
                        p = i32::from(self.at(self.idx((x, y))));
                        if p != prev {
                            break;
                        }
                        x += 1;
                        if x >= self.width {
                            break;
                        }
                    }
                    if x >= self.width {
                        break;
                    }
                    if self.contour_scan(prev, p, &mut last_pos, x, y) {
                        self.lnbd = last_pos;
                        return true;
                    }
                    prev = p;
                    if prev & !1 != 0 {
                        last_pos.0 = x;
                    }
                    x += 1;
                }
                last_pos = (0, y + 1);
                x = 1;
                prev = 0;
                y += 1;
            }
            false
        }
    }

    type Contours = Vec<(Vec<(i32, i32)>, bool)>;

    /// Contours of the `w` x `h` image `src` (row-major), foreground where
    /// `fg` holds, and their hierarchy.
    pub(super) fn find<T: Copy>(
        src: &[T],
        w: usize,
        h: usize,
        mode: ContourRetrieval,
        method: ContourApproximation,
        fg: impl Fn(T) -> bool,
    ) -> (Contours, Vec<ContourHierarchy>) {
        let (pw, ph) = (w + 2, h + 2);
        let mut img = vec![0i8; pw * ph];
        for y in 0..h {
            for x in 0..w {
                img[(y + 1) * pw + x + 1] = i8::from(fg(src[y * w + x]));
            }
        }
        let mut root = Node::new(true);
        root.brect = (0, 0, pw as i32, ph as i32);
        let mut sc = Scanner {
            img,
            step: pw as i32,
            width: pw as i32 - 1,
            height: ph as i32 - 1,
            pt: (1, 1),
            lnbd: (0, 1),
            nbd: 2,
            mode,
            direct: method == ContourApproximation::None,
            nodes: vec![root],
            ctable: [-1; 128],
        };
        while sc.find_next() {}

        // `contourTreeToResults`: pre-order, siblings in list order.
        let mut order = Vec::with_capacity(sc.nodes.len() - 1);
        let mut stack = vec![0i32];
        while let Some(idx) = stack.pop() {
            if idx != 0 {
                order.push(idx);
            }
            let mut children = Vec::new();
            let mut c = sc.nodes[idx as usize].first_child;
            while c != -1 {
                children.push(c);
                c = sc.nodes[c as usize].next;
            }
            stack.extend(children.into_iter().rev());
        }
        let mut mapping = vec![-1i32; sc.nodes.len()];
        for (i, &idx) in order.iter().enumerate() {
            mapping[idx as usize] = i as i32;
        }
        let map = |i: i32| if i < 0 { -1 } else { mapping[i as usize] };
        let hierarchy = order
            .iter()
            .map(|&idx| {
                let n = &sc.nodes[idx as usize];
                [map(n.next), map(n.prev), map(n.first_child), map(n.parent)]
            })
            .collect();
        let contours = order
            .iter()
            .map(|&idx| {
                let n = &mut sc.nodes[idx as usize];
                (std::mem::take(&mut n.points), n.is_hole)
            })
            .collect();
        (contours, hierarchy)
    }
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
    use crate::moments::{contour_area, contour_perimeter};
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

#[cfg(test)]
mod contour_regression_tests {
    use super::*;

    #[test]
    fn test_solid_rectangle_yields_single_outer_no_phantom_holes() {
        use cv_core::TensorShape;
        // Regression: every row's rightmost border pixel previously started a
        // phantom "hole" re-trace of the outer boundary.
        let mut data = vec![0.0f32; 20 * 20];
        for y in 5..15 {
            for x in 5..15 {
                data[y * 20 + x] = 1.0;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 20, 20)).unwrap();
        let contours = find_contours(&img).expect("find_contours");
        let outer: Vec<_> = contours.iter().filter(|c| !c.is_hole).collect();
        let holes: Vec<_> = contours.iter().filter(|c| c.is_hole).collect();
        assert_eq!(outer.len(), 1, "expected exactly one outer contour");
        assert!(holes.is_empty(), "solid rectangle must have no holes");
    }

    #[test]
    fn test_rectangle_with_hole_detected() {
        use cv_core::TensorShape;
        let mut data = vec![0.0f32; 21 * 21];
        for y in 3..18 {
            for x in 3..18 {
                data[y * 21 + x] = 1.0;
            }
        }
        for y in 7..13 {
            for x in 7..13 {
                data[y * 21 + x] = 0.0;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 21, 21)).unwrap();
        let contours = find_contours(&img).expect("find_contours");
        assert!(
            contours.iter().any(|c| c.is_hole),
            "inner hole border must be detected"
        );
    }

    #[test]
    fn test_two_adjacent_pixels_terminate_immediately() {
        use cv_core::TensorShape;
        // Regression: A->B->A ping-pong never met the old termination
        // condition and ran to max_steps producing a giant garbage contour.
        let mut data = vec![0.0f32; 8 * 8];
        data[3 * 8 + 3] = 1.0;
        data[3 * 8 + 4] = 1.0;
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 8, 8)).unwrap();
        let contours = find_contours(&img).unwrap();
        let total_points: usize = contours.iter().map(|c| c.points.len()).sum();
        // A 2-pixel blob's border is tiny; the old bug produced millions of
        // alternating A/B points.
        assert!(
            total_points <= 6,
            "2-pixel blob boundary must be tiny, got {}",
            total_points
        );
    }
}
