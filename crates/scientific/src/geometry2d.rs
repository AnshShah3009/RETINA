//! 2D Computational Geometry (Shapely-equivalent, zero external dependencies)
//!
//! Provides primitives, measurements, spatial predicates, boolean operations,
//! construction algorithms, a spatial index, and serialization — all implemented
//! from first principles using only `std`.

use std::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;

// ─── Primitives ──────────────────────────────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// An ordered sequence of 2D points forming a polyline.
#[derive(Debug, Clone)]
pub struct LineString {
    pub coords: Vec<Point2D>,
}

impl LineString {
    pub fn new(coords: Vec<Point2D>) -> Self {
        Self { coords }
    }

    /// Total length of the polyline.
    pub fn length(&self) -> f64 {
        self.coords.windows(2).map(|w| w[0].distance(&w[1])).sum()
    }

    /// Whether the first and last points coincide.
    pub fn is_closed(&self) -> bool {
        if self.coords.len() < 2 {
            return false;
        }
        let first = &self.coords[0];
        let last = self.coords.last().unwrap();
        (first.x - last.x).abs() < EPS && (first.y - last.y).abs() < EPS
    }
}

/// A polygon with an exterior ring and zero or more holes.
///
/// Convention: exterior ring is counter-clockwise (CCW), holes are clockwise (CW).
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Outer boundary vertices (should be closed: first == last).
    pub exterior: Vec<Point2D>,
    /// Interior holes, each a closed ring.
    pub holes: Vec<Vec<Point2D>>,
}

impl Polygon {
    pub fn new(exterior: Vec<Point2D>, holes: Vec<Vec<Point2D>>) -> Self {
        Self { exterior, holes }
    }

    /// Signed area using the shoelace formula.  Positive for CCW winding.
    pub fn signed_area(&self) -> f64 {
        ring_signed_area(&self.exterior)
    }

    /// Unsigned area (exterior minus holes).
    pub fn area(&self) -> f64 {
        let ext = ring_signed_area(&self.exterior).abs();
        let holes: f64 = self.holes.iter().map(|h| ring_signed_area(h).abs()).sum();
        ext - holes
    }

    /// Perimeter of the exterior ring plus all hole perimeters.
    pub fn perimeter(&self) -> f64 {
        let ext = ring_perimeter(&self.exterior);
        let holes: f64 = self.holes.iter().map(|h| ring_perimeter(h)).sum();
        ext + holes
    }

    /// Centroid of the exterior ring (ignoring holes for simplicity).
    pub fn centroid(&self) -> Point2D {
        ring_centroid(&self.exterior)
    }

    /// Axis-aligned bounding box: (min_x, min_y, max_x, max_y).
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        ring_bbox(&self.exterior)
    }

    /// Basic validity check: at least 3 distinct vertices and no self-intersection
    /// of the exterior ring.
    pub fn is_valid(&self) -> bool {
        if self.exterior.len() < 4 {
            // Need at least 3 distinct + closing vertex
            return false;
        }
        !ring_self_intersects(&self.exterior)
    }
}

/// A collection of points.
#[derive(Debug, Clone)]
pub struct MultiPoint(pub Vec<Point2D>);

/// A collection of line strings.
#[derive(Debug, Clone)]
pub struct MultiLineString(pub Vec<LineString>);

/// A collection of polygons.
#[derive(Debug, Clone)]
pub struct MultiPolygon(pub Vec<Polygon>);

// ─── Ring helpers ────────────────────────────────────────────────────────────

fn ring_signed_area(ring: &[Point2D]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += ring[i].x * ring[j].y;
        sum -= ring[j].x * ring[i].y;
    }
    sum * 0.5
}

fn ring_perimeter(ring: &[Point2D]) -> f64 {
    if ring.len() < 2 {
        return 0.0;
    }
    ring.windows(2).map(|w| w[0].distance(&w[1])).sum()
}

fn ring_centroid(ring: &[Point2D]) -> Point2D {
    let n = ring.len();
    if n == 0 {
        return Point2D::new(0.0, 0.0);
    }
    let a = ring_signed_area(ring);
    if a.abs() < EPS {
        // Degenerate: return average
        let sx: f64 = ring.iter().map(|p| p.x).sum();
        let sy: f64 = ring.iter().map(|p| p.y).sum();
        return Point2D::new(sx / n as f64, sy / n as f64);
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = ring[i].x * ring[j].y - ring[j].x * ring[i].y;
        cx += (ring[i].x + ring[j].x) * cross;
        cy += (ring[i].y + ring[j].y) * cross;
    }
    let factor = 1.0 / (6.0 * a);
    Point2D::new(cx * factor, cy * factor)
}

fn ring_bbox(ring: &[Point2D]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in ring {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }
    (min_x, min_y, max_x, max_y)
}

fn ring_self_intersects(ring: &[Point2D]) -> bool {
    let n = if ring.len() > 1
        && (ring[0].x - ring.last().unwrap().x).abs() < EPS
        && (ring[0].y - ring.last().unwrap().y).abs() < EPS
    {
        ring.len() - 1
    } else {
        ring.len()
    };
    if n < 4 {
        return false;
    }
    for i in 0..n {
        let i2 = (i + 1) % n;
        for j in (i + 2)..n {
            let j2 = (j + 1) % n;
            // Skip adjacent edges
            if j2 == i {
                continue;
            }
            if segments_intersect_proper(&ring[i], &ring[i2], &ring[j], &ring[j2]) {
                return true;
            }
        }
    }
    false
}

// ─── Low-level geometry helpers ──────────────────────────────────────────────

/// 2D cross product of vectors (b-a) x (c-a).
fn cross2d(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Whether point p lies on segment (a, b).
fn on_segment(a: &Point2D, p: &Point2D, b: &Point2D) -> bool {
    p.x <= a.x.max(b.x) + EPS
        && p.x >= a.x.min(b.x) - EPS
        && p.y <= a.y.max(b.y) + EPS
        && p.y >= a.y.min(b.y) - EPS
}

/// Properly-intersecting segments (excludes shared endpoints / collinear overlaps).
fn segments_intersect_proper(a1: &Point2D, a2: &Point2D, b1: &Point2D, b2: &Point2D) -> bool {
    let d1 = cross2d(b1, b2, a1);
    let d2 = cross2d(b1, b2, a2);
    let d3 = cross2d(a1, a2, b1);
    let d4 = cross2d(a1, a2, b2);

    if ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
        && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS))
    {
        return true;
    }
    false
}

// ─── Spatial Predicates ──────────────────────────────────────────────────────

/// Test whether two segments (a1-a2) and (b1-b2) intersect (including touching / collinear).
pub fn segments_intersect(a1: &Point2D, a2: &Point2D, b1: &Point2D, b2: &Point2D) -> bool {
    let d1 = cross2d(b1, b2, a1);
    let d2 = cross2d(b1, b2, a2);
    let d3 = cross2d(a1, a2, b1);
    let d4 = cross2d(a1, a2, b2);

    if ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
        && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS))
    {
        return true;
    }

    // Collinear / touching cases
    if d1.abs() <= EPS && on_segment(a1, b1, a2) {
        return true;
    }
    if d2.abs() <= EPS && on_segment(a1, b2, a2) {
        return true;
    }
    if d3.abs() <= EPS && on_segment(b1, a1, b2) {
        return true;
    }
    if d4.abs() <= EPS && on_segment(b1, a2, b2) {
        return true;
    }
    false
}

/// Compute the intersection point of two line segments, if it exists.
pub fn segment_intersection_point(
    a1: &Point2D,
    a2: &Point2D,
    b1: &Point2D,
    b2: &Point2D,
) -> Option<Point2D> {
    let dax = a2.x - a1.x;
    let day = a2.y - a1.y;
    let dbx = b2.x - b1.x;
    let dby = b2.y - b1.y;

    let denom = dax * dby - day * dbx;
    if denom.abs() < EPS {
        return None; // Parallel or collinear
    }

    let t = ((b1.x - a1.x) * dby - (b1.y - a1.y) * dbx) / denom;
    let u = ((b1.x - a1.x) * day - (b1.y - a1.y) * dax) / denom;

    if (-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u) {
        Some(Point2D::new(a1.x + t * dax, a1.y + t * day))
    } else {
        None
    }
}

/// Point-in-polygon test using the ray casting algorithm.
///
/// Returns `true` if the point is inside the polygon (accounting for holes).
pub fn point_in_polygon(point: &Point2D, polygon: &Polygon) -> bool {
    if !point_in_ring(point, &polygon.exterior) {
        return false;
    }
    // Check holes
    for hole in &polygon.holes {
        if point_in_ring(point, hole) {
            return false;
        }
    }
    true
}

fn point_in_ring(point: &Point2D, ring: &[Point2D]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = ring[i].y;
        let yj = ring[j].y;
        if ((yi > point.y) != (yj > point.y))
            && (point.x < (ring[j].x - ring[i].x) * (point.y - yi) / (yj - yi) + ring[i].x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Whether any edges of the two polygons cross, or one contains the other.
pub fn polygons_intersect(a: &Polygon, b: &Polygon) -> bool {
    // Quick AABB rejection
    let ba = a.bbox();
    let bb = b.bbox();
    if ba.2 < bb.0 || bb.2 < ba.0 || ba.3 < bb.1 || bb.3 < ba.1 {
        return false;
    }

    // Edge-edge intersection
    let ea = &a.exterior;
    let eb = &b.exterior;
    for i in 0..ea.len().saturating_sub(1) {
        for j in 0..eb.len().saturating_sub(1) {
            if segments_intersect(&ea[i], &ea[i + 1], &eb[j], &eb[j + 1]) {
                return true;
            }
        }
    }

    // Containment check: a vertex of one inside the other
    if !a.exterior.is_empty() && point_in_polygon(&a.exterior[0], b) {
        return true;
    }
    if !b.exterior.is_empty() && point_in_polygon(&b.exterior[0], a) {
        return true;
    }
    false
}

/// Whether `outer` completely contains `inner`.
pub fn polygon_contains_polygon(outer: &Polygon, inner: &Polygon) -> bool {
    for p in &inner.exterior {
        if !point_in_polygon(p, outer) {
            return false;
        }
    }
    true
}

// ─── Boolean Operations ──────────────────────────────────────────────────────

/// Sutherland-Hodgman polygon clipping: clips `subject` against `clip`.
///
/// Both polygons should be convex for correct results.  Vertices are given
/// as open rings (no repeated closing vertex).
fn sutherland_hodgman(subject: &[Point2D], clip: &[Point2D]) -> Vec<Point2D> {
    let mut output = subject.to_vec();

    let cn = clip.len();
    if cn < 2 || output.is_empty() {
        return output;
    }

    for i in 0..cn {
        if output.is_empty() {
            return output;
        }
        let edge_start = &clip[i];
        let edge_end = &clip[(i + 1) % cn];
        let input = output;
        output = Vec::new();

        let n = input.len();
        if n == 0 {
            break;
        }
        let mut s = &input[n - 1];
        for e in &input {
            let e_inside = cross2d(edge_start, edge_end, e) >= -EPS;
            let s_inside = cross2d(edge_start, edge_end, s) >= -EPS;

            if e_inside {
                if !s_inside {
                    if let Some(p) = line_intersection(edge_start, edge_end, s, e) {
                        output.push(p);
                    }
                }
                output.push(e.clone());
            } else if s_inside {
                if let Some(p) = line_intersection(edge_start, edge_end, s, e) {
                    output.push(p);
                }
            }
            s = e;
        }
    }
    output
}

/// Intersection of infinite lines through (a1,a2) and (b1,b2).
fn line_intersection(a1: &Point2D, a2: &Point2D, b1: &Point2D, b2: &Point2D) -> Option<Point2D> {
    let dax = a2.x - a1.x;
    let day = a2.y - a1.y;
    let dbx = b2.x - b1.x;
    let dby = b2.y - b1.y;
    let denom = dax * dby - day * dbx;
    if denom.abs() < EPS {
        return None;
    }
    let t = ((b1.x - a1.x) * dby - (b1.y - a1.y) * dbx) / denom;
    Some(Point2D::new(a1.x + t * dax, a1.y + t * day))
}

/// Make an open ring (remove trailing closing vertex if present).
fn open_ring(ring: &[Point2D]) -> Vec<Point2D> {
    if ring.len() >= 2 {
        let first = &ring[0];
        let last = &ring[ring.len() - 1];
        if (first.x - last.x).abs() < EPS && (first.y - last.y).abs() < EPS {
            return ring[..ring.len() - 1].to_vec();
        }
    }
    ring.to_vec()
}

/// Close a ring by appending the first vertex if not already closed.
fn close_ring(ring: &mut Vec<Point2D>) {
    if ring.len() >= 2 {
        let first = ring[0].clone();
        let last = ring.last().unwrap();
        if (first.x - last.x).abs() > EPS || (first.y - last.y).abs() > EPS {
            ring.push(first);
        }
    }
}

/// Compute the intersection of two polygons.
///
/// Uses Sutherland-Hodgman for convex polygons.  For general (non-convex) polygons
/// the result is approximate.
pub fn polygon_intersection(a: &Polygon, b: &Polygon) -> Vec<Polygon> {
    let sa = open_ring(&a.exterior);
    let sb = open_ring(&b.exterior);
    let mut result = sutherland_hodgman(&sa, &sb);
    if result.len() < 3 {
        return vec![];
    }
    close_ring(&mut result);
    vec![Polygon::new(result, vec![])]
}

/// Compute the union of two polygons.
///
/// Simplified approach: if they don't intersect, return both as separate polygons.
/// If they do, compute a combined convex hull as an approximation.
pub fn polygon_union(a: &Polygon, b: &Polygon) -> Vec<Polygon> {
    if !polygons_intersect(a, b) {
        return vec![a.clone(), b.clone()];
    }
    // Approximation for overlapping polygons: combined convex hull
    let mut all_pts: Vec<Point2D> = a.exterior.clone();
    all_pts.extend(b.exterior.clone());
    vec![convex_hull(&all_pts)]
}

/// Compute A - B (polygon difference).
///
/// Simplified: clips A against the complement of B.  For convex B this uses
/// Sutherland-Hodgman on the half-planes outside B edges.
pub fn polygon_difference(a: &Polygon, b: &Polygon) -> Vec<Polygon> {
    if !polygons_intersect(a, b) {
        return vec![a.clone()];
    }
    // Simple approach: return A with B as a hole
    let inter = polygon_intersection(a, b);
    if inter.is_empty() {
        return vec![a.clone()];
    }
    let inter_area: f64 = inter.iter().map(|p| p.area()).sum();
    let a_area = a.area();
    if (inter_area - a_area).abs() < EPS {
        // A is fully inside B
        return vec![];
    }
    // Return A with the intersection ring as a hole
    let mut holes = a.holes.clone();
    for ip in &inter {
        holes.push(ip.exterior.clone());
    }
    vec![Polygon::new(a.exterior.clone(), holes)]
}

// ─── Construction Algorithms ─────────────────────────────────────────────────

/// Convex hull using Andrew's monotone chain algorithm.  O(n log n).
pub fn convex_hull(points: &[Point2D]) -> Polygon {
    let mut pts: Vec<Point2D> = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    pts.dedup_by(|a, b| (a.x - b.x).abs() < EPS && (a.y - b.y).abs() < EPS);

    let n = pts.len();
    if n == 0 {
        return Polygon::new(vec![], vec![]);
    }
    if n == 1 {
        return Polygon::new(vec![pts[0].clone(), pts[0].clone()], vec![]);
    }
    if n == 2 {
        let mut ext = pts.clone();
        ext.push(pts[0].clone());
        return Polygon::new(ext, vec![]);
    }

    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2
            && cross2d(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= EPS
        {
            lower.pop();
        }
        lower.push(p.clone());
    }

    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2
            && cross2d(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= EPS
        {
            upper.pop();
        }
        upper.push(p.clone());
    }

    lower.pop();
    upper.pop();
    lower.append(&mut upper);
    lower.push(lower[0].clone()); // close
    Polygon::new(lower, vec![])
}

/// Offset (buffer) a polygon outward by `distance`.
///
/// Each edge is offset outward and arc segments of `segments` pieces are added
/// at convex corners.
pub fn buffer_polygon(polygon: &Polygon, distance: f64, segments: usize) -> Polygon {
    let ring = open_ring(&polygon.exterior);
    let n = ring.len();
    if n < 3 {
        return polygon.clone();
    }

    // Ensure CCW
    let area = ring_signed_area(&ring);
    let pts: Vec<Point2D> = if area < 0.0 {
        ring.into_iter().rev().collect()
    } else {
        ring
    };

    let n = pts.len();
    // Compute outward normals for each edge.
    // For a CCW polygon, the outward (right-hand) normal of edge (dx,dy) is (dy,-dx).
    let mut normals: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].x - pts[i].x;
        let dy = pts[j].y - pts[i].y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < EPS {
            normals.push((0.0, 1.0));
        } else {
            normals.push((dy / len, -dx / len));
        }
    }

    let mut result = Vec::new();
    let seg = segments.max(1);

    for i in 0..n {
        let prev_edge = if i == 0 { n - 1 } else { i - 1 };
        let curr_edge = i;
        let (nx0, ny0) = normals[prev_edge];
        let (nx1, ny1) = normals[curr_edge];

        // Turn direction at this vertex
        let curr_next = (curr_edge + 1) % n;
        let epx = pts[i].x - pts[prev_edge].x;
        let epy = pts[i].y - pts[prev_edge].y;
        let ecx = pts[curr_next].x - pts[i].x;
        let ecy = pts[curr_next].y - pts[i].y;
        let turn = epx * ecy - epy * ecx; // positive = left turn = convex for CCW

        if turn > EPS && distance > 0.0 {
            // Convex corner: insert a rounded arc
            let angle0 = ny0.atan2(nx0);
            let angle1 = ny1.atan2(nx1);
            let mut da = angle1 - angle0;
            while da > std::f64::consts::PI {
                da -= 2.0 * std::f64::consts::PI;
            }
            while da < -std::f64::consts::PI {
                da += 2.0 * std::f64::consts::PI;
            }
            for s in 0..=seg {
                let t = s as f64 / seg as f64;
                let a = angle0 + da * t;
                result.push(Point2D::new(
                    pts[i].x + distance * a.cos(),
                    pts[i].y + distance * a.sin(),
                ));
            }
        } else {
            // Reflex or straight: miter
            let a1 = Point2D::new(
                pts[prev_edge].x + nx0 * distance,
                pts[prev_edge].y + ny0 * distance,
            );
            let a2 = Point2D::new(pts[i].x + nx0 * distance, pts[i].y + ny0 * distance);
            let b1 = Point2D::new(pts[i].x + nx1 * distance, pts[i].y + ny1 * distance);
            let b2 = Point2D::new(
                pts[curr_next].x + nx1 * distance,
                pts[curr_next].y + ny1 * distance,
            );
            if let Some(miter) = line_intersection(&a1, &a2, &b1, &b2) {
                result.push(miter);
            } else {
                result.push(Point2D::new(
                    pts[i].x + nx0 * distance,
                    pts[i].y + ny0 * distance,
                ));
            }
        }
    }

    close_ring(&mut result);
    Polygon::new(result, vec![])
}

/// Douglas-Peucker line simplification.
pub fn simplify(coords: &[Point2D], tolerance: f64) -> Vec<Point2D> {
    if coords.len() <= 2 {
        return coords.to_vec();
    }
    dp_simplify(coords, tolerance)
}

#[allow(clippy::needless_range_loop)]
fn dp_simplify(coords: &[Point2D], tolerance: f64) -> Vec<Point2D> {
    let n = coords.len();
    if n <= 2 {
        return coords.to_vec();
    }

    let mut max_dist = 0.0_f64;
    let mut max_idx = 0;
    let first = &coords[0];
    let last = &coords[n - 1];

    for i in 1..n - 1 {
        let d = perpendicular_distance(&coords[i], first, last);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > tolerance {
        let left = dp_simplify(&coords[..=max_idx], tolerance);
        let right = dp_simplify(&coords[max_idx..], tolerance);
        let mut result = left;
        result.pop(); // Remove duplicate at join
        result.extend(right);
        result
    } else {
        vec![first.clone(), last.clone()]
    }
}

fn perpendicular_distance(p: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < EPS {
        return p.distance(a);
    }
    ((dy * p.x - dx * p.y + b.x * a.y - b.y * a.x).abs()) / len_sq.sqrt()
}

/// Bowyer-Watson incremental Delaunay triangulation.
///
/// Returns a list of triangles, each represented as three indices into the input
/// `points` slice.
pub fn delaunay_triangulation(points: &[Point2D]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return vec![];
    }

    // Compute bounding box and create super-triangle
    let (min_x, min_y, max_x, max_y) = {
        let mut mnx = f64::INFINITY;
        let mut mny = f64::INFINITY;
        let mut mxx = f64::NEG_INFINITY;
        let mut mxy = f64::NEG_INFINITY;
        for p in points {
            if p.x < mnx {
                mnx = p.x;
            }
            if p.y < mny {
                mny = p.y;
            }
            if p.x > mxx {
                mxx = p.x;
            }
            if p.y > mxy {
                mxy = p.y;
            }
        }
        (mnx, mny, mxx, mxy)
    };

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let d_max = dx.max(dy);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    // Super-triangle vertices (indices n, n+1, n+2)
    let st0 = Point2D::new(mid_x - 20.0 * d_max, mid_y - d_max);
    let st1 = Point2D::new(mid_x, mid_y + 20.0 * d_max);
    let st2 = Point2D::new(mid_x + 20.0 * d_max, mid_y - d_max);

    let mut all_pts: Vec<Point2D> = points.to_vec();
    all_pts.push(st0);
    all_pts.push(st1);
    all_pts.push(st2);

    // Each triangle stores 3 vertex indices
    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    // Insert each point
    for i in 0..n {
        let p = &all_pts[i];
        let mut bad_triangles = Vec::new();

        for (ti, tri) in triangles.iter().enumerate() {
            if in_circumcircle(p, &all_pts[tri[0]], &all_pts[tri[1]], &all_pts[tri[2]]) {
                bad_triangles.push(ti);
            }
        }

        // Find boundary polygon of the bad triangles
        let mut boundary = Vec::new();
        for &ti in &bad_triangles {
            let tri = &triangles[ti];
            for edge_idx in 0..3 {
                let e0 = tri[edge_idx];
                let e1 = tri[(edge_idx + 1) % 3];
                let mut shared = false;
                for &tj in &bad_triangles {
                    if tj == ti {
                        continue;
                    }
                    let other = &triangles[tj];
                    if triangle_has_edge(other, e0, e1) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    boundary.push((e0, e1));
                }
            }
        }

        // Remove bad triangles (in reverse order to keep indices valid)
        let mut bad_sorted = bad_triangles.clone();
        bad_sorted.sort_unstable();
        for &ti in bad_sorted.iter().rev() {
            triangles.swap_remove(ti);
        }

        // Create new triangles
        for (e0, e1) in &boundary {
            triangles.push([i, *e0, *e1]);
        }
    }

    // Remove triangles that share vertices with the super-triangle
    triangles.retain(|tri| tri[0] < n && tri[1] < n && tri[2] < n);

    triangles
}

fn in_circumcircle(p: &Point2D, a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    let ax = a.x - p.x;
    let ay = a.y - p.y;
    let bx = b.x - p.x;
    let by = b.y - p.y;
    let cx = c.x - p.x;
    let cy = c.y - p.y;

    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - bx * (ay * (cx * cx + cy * cy) - cy * (ax * ax + ay * ay))
        + cx * (ay * (bx * bx + by * by) - by * (ax * ax + ay * ay));

    // Positive det means inside circumcircle (for CCW triangle)
    // We need to handle both orientations
    let orient = cross2d(a, b, c);
    if orient > 0.0 {
        det > EPS
    } else {
        det < -EPS
    }
}

fn triangle_has_edge(tri: &[usize; 3], e0: usize, e1: usize) -> bool {
    for i in 0..3 {
        let a = tri[i];
        let b = tri[(i + 1) % 3];
        if (a == e0 && b == e1) || (a == e1 && b == e0) {
            return true;
        }
    }
    false
}

/// Voronoi diagram from a set of points within given bounds.
///
/// Constructs Voronoi cells by assigning each point in the bounding region to its
/// nearest input point, then building cells via perpendicular bisector half-plane
/// intersection (clipped to bounds).
///
/// Returns one polygon per input point.
pub fn voronoi_diagram(points: &[Point2D], bounds: (f64, f64, f64, f64)) -> Vec<Polygon> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }
    let (bx0, by0, bx1, by1) = bounds;
    let bounds_poly = vec![
        Point2D::new(bx0, by0),
        Point2D::new(bx1, by0),
        Point2D::new(bx1, by1),
        Point2D::new(bx0, by1),
    ];

    if n == 1 {
        let mut ring = bounds_poly;
        close_ring(&mut ring);
        return vec![Polygon::new(ring, vec![])];
    }

    // For each point i, the Voronoi cell is the intersection of the bounding box
    // with all half-planes closer to point i than to any other point j.
    // Each half-plane is defined by the perpendicular bisector of (i, j).
    let mut cells = Vec::with_capacity(n);

    for i in 0..n {
        let mut cell = bounds_poly.clone();

        for j in 0..n {
            if i == j {
                continue;
            }
            if cell.len() < 3 {
                break;
            }

            // Perpendicular bisector of points[i] and points[j]:
            // All p where (p - mid) . (j - i) <= 0   (i.e., closer to i)
            let mid = Point2D::new(
                (points[i].x + points[j].x) * 0.5,
                (points[i].y + points[j].y) * 0.5,
            );
            let nx = points[j].x - points[i].x;
            let ny = points[j].y - points[i].y;

            // Clip cell polygon by the half-plane: nx*(x-mid.x) + ny*(y-mid.y) <= 0
            cell = clip_by_halfplane(&cell, &mid, nx, ny);
        }

        if cell.len() < 3 {
            cells.push(Polygon::new(vec![], vec![]));
        } else {
            close_ring(&mut cell);
            cells.push(Polygon::new(cell, vec![]));
        }
    }

    cells
}

/// Clip a convex polygon by the half-plane: nx*(x-pt.x) + ny*(y-pt.y) <= 0.
fn clip_by_halfplane(poly: &[Point2D], pt: &Point2D, nx: f64, ny: f64) -> Vec<Point2D> {
    let n = poly.len();
    if n == 0 {
        return vec![];
    }
    let mut output = Vec::new();
    let mut s = &poly[n - 1];
    let mut s_val = nx * (s.x - pt.x) + ny * (s.y - pt.y);

    for e in poly {
        let e_val = nx * (e.x - pt.x) + ny * (e.y - pt.y);
        if e_val <= EPS {
            // e is inside
            if s_val > EPS {
                // s is outside -> crossing
                let t = s_val / (s_val - e_val);
                output.push(Point2D::new(s.x + t * (e.x - s.x), s.y + t * (e.y - s.y)));
            }
            output.push(e.clone());
        } else if s_val <= EPS {
            // s is inside, e is outside -> crossing
            let t = s_val / (s_val - e_val);
            output.push(Point2D::new(s.x + t * (e.x - s.x), s.y + t * (e.y - s.y)));
        }
        s = e;
        s_val = e_val;
    }
    output
}

// ─── Spatial Index (STR-tree) ────────────────────────────────────────────────

/// A node in the STR-tree.
#[derive(Debug)]
enum STRNode {
    Leaf {
        items: Vec<(usize, f64, f64, f64, f64)>,
    },
    Branch {
        bbox: (f64, f64, f64, f64),
        children: Vec<STRNode>,
    },
}

/// Sort-Tile-Recursive R-tree for fast spatial queries.
pub struct STRtree {
    root: STRNode,
}

impl STRtree {
    /// Build an STR-tree from items (id, min_x, min_y, max_x, max_y).
    pub fn new(items: &[(usize, f64, f64, f64, f64)]) -> Self {
        if items.is_empty() {
            return Self {
                root: STRNode::Leaf { items: vec![] },
            };
        }
        let root = str_build(items.to_vec(), 8); // node capacity 8
        Self { root }
    }

    /// Query all items whose bounding boxes intersect the given rectangle.
    pub fn query(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        let mut result = Vec::new();
        str_query(&self.root, min_x, min_y, max_x, max_y, &mut result);
        result
    }

    /// Find the `k` nearest items to point (x, y) by bounding-box center distance.
    pub fn nearest(&self, x: f64, y: f64, k: usize) -> Vec<usize> {
        let mut all = Vec::new();
        str_collect_all(&self.root, &mut all);
        all.sort_by(|a, b| {
            let da = {
                let cx = (a.1 + a.3) * 0.5;
                let cy = (a.2 + a.4) * 0.5;
                (cx - x).powi(2) + (cy - y).powi(2)
            };
            let db = {
                let cx = (b.1 + b.3) * 0.5;
                let cy = (b.2 + b.4) * 0.5;
                (cx - x).powi(2) + (cy - y).powi(2)
            };
            da.partial_cmp(&db).unwrap()
        });
        all.into_iter().take(k).map(|item| item.0).collect()
    }
}

fn str_build(mut items: Vec<(usize, f64, f64, f64, f64)>, capacity: usize) -> STRNode {
    if items.len() <= capacity {
        return STRNode::Leaf { items };
    }

    let n = items.len();
    let num_slices = ((n as f64) / (capacity as f64)).sqrt().ceil() as usize;
    let slice_size = n.div_ceil(num_slices);

    // Sort by x-center
    items.sort_by(|a, b| {
        let ca = (a.1 + a.3) * 0.5;
        let cb = (b.1 + b.3) * 0.5;
        ca.partial_cmp(&cb).unwrap()
    });

    let mut children = Vec::new();
    for slice in items.chunks(slice_size) {
        let mut s = slice.to_vec();
        // Sort slice by y-center
        s.sort_by(|a, b| {
            let ca = (a.2 + a.4) * 0.5;
            let cb = (b.2 + b.4) * 0.5;
            ca.partial_cmp(&cb).unwrap()
        });
        for chunk in s.chunks(capacity) {
            children.push(str_build(chunk.to_vec(), capacity));
        }
    }

    let bbox = str_node_bbox_all(&children);
    STRNode::Branch { bbox, children }
}

fn str_node_bbox(node: &STRNode) -> (f64, f64, f64, f64) {
    match node {
        STRNode::Leaf { items } => {
            let mut mnx = f64::INFINITY;
            let mut mny = f64::INFINITY;
            let mut mxx = f64::NEG_INFINITY;
            let mut mxy = f64::NEG_INFINITY;
            for item in items {
                if item.1 < mnx {
                    mnx = item.1;
                }
                if item.2 < mny {
                    mny = item.2;
                }
                if item.3 > mxx {
                    mxx = item.3;
                }
                if item.4 > mxy {
                    mxy = item.4;
                }
            }
            (mnx, mny, mxx, mxy)
        }
        STRNode::Branch { bbox, .. } => *bbox,
    }
}

fn str_node_bbox_all(nodes: &[STRNode]) -> (f64, f64, f64, f64) {
    let mut mnx = f64::INFINITY;
    let mut mny = f64::INFINITY;
    let mut mxx = f64::NEG_INFINITY;
    let mut mxy = f64::NEG_INFINITY;
    for node in nodes {
        let (a, b, c, d) = str_node_bbox(node);
        if a < mnx {
            mnx = a;
        }
        if b < mny {
            mny = b;
        }
        if c > mxx {
            mxx = c;
        }
        if d > mxy {
            mxy = d;
        }
    }
    (mnx, mny, mxx, mxy)
}

fn str_query(
    node: &STRNode,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    result: &mut Vec<usize>,
) {
    match node {
        STRNode::Leaf { items } => {
            for item in items {
                if item.3 >= min_x && item.1 <= max_x && item.4 >= min_y && item.2 <= max_y {
                    result.push(item.0);
                }
            }
        }
        STRNode::Branch { bbox, children } => {
            if bbox.2 < min_x || bbox.0 > max_x || bbox.3 < min_y || bbox.1 > max_y {
                return;
            }
            for child in children {
                str_query(child, min_x, min_y, max_x, max_y, result);
            }
        }
    }
}

fn str_collect_all(node: &STRNode, result: &mut Vec<(usize, f64, f64, f64, f64)>) {
    match node {
        STRNode::Leaf { items } => result.extend_from_slice(items),
        STRNode::Branch { children, .. } => {
            for child in children {
                str_collect_all(child, result);
            }
        }
    }
}

// ─── Serialization ───────────────────────────────────────────────────────────

/// Convert a polygon to WKT format.
pub fn to_wkt(polygon: &Polygon) -> String {
    let mut s = String::from("POLYGON (");
    s.push('(');
    for (i, p) in polygon.exterior.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{} {}", p.x, p.y));
    }
    s.push(')');
    for hole in &polygon.holes {
        s.push_str(", (");
        for (i, p) in hole.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{} {}", p.x, p.y));
        }
        s.push(')');
    }
    s.push(')');
    s
}

/// Parse a simple POLYGON WKT string.
pub fn from_wkt(wkt: &str) -> Result<Polygon, String> {
    let trimmed = wkt.trim();
    if !trimmed.starts_with("POLYGON") {
        return Err("Expected POLYGON".to_string());
    }

    // Strip "POLYGON" prefix and outer parens
    let rest = trimmed["POLYGON".len()..].trim();
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return Err("Expected outer parentheses".to_string());
    }
    let inner = &rest[1..rest.len() - 1];

    // Split rings by ")," pattern
    let mut rings: Vec<Vec<Point2D>> = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let chars: Vec<char> = inner.chars().collect();
    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let ring_str: String = chars[start..=i].iter().collect();
                    let ring = parse_ring(&ring_str)?;
                    rings.push(ring);
                    // Skip comma and whitespace
                    start = i + 1;
                }
            }
            _ => {}
        }
    }

    if rings.is_empty() {
        return Err("No rings found".to_string());
    }

    let exterior = rings.remove(0);
    Ok(Polygon::new(exterior, rings))
}

fn parse_ring(s: &str) -> Result<Vec<Point2D>, String> {
    let trimmed = s.trim();
    let inner = if trimmed.starts_with('(') && trimmed.ends_with(')') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let mut points = Vec::new();
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let coords: Vec<&str> = pair.split_whitespace().collect();
        if coords.len() != 2 {
            return Err(format!(
                "Expected 2 coords, got {}: '{}'",
                coords.len(),
                pair
            ));
        }
        let x: f64 = coords[0]
            .parse()
            .map_err(|_| format!("Invalid x: '{}'", coords[0]))?;
        let y: f64 = coords[1]
            .parse()
            .map_err(|_| format!("Invalid y: '{}'", coords[1]))?;
        points.push(Point2D::new(x, y));
    }
    Ok(points)
}

/// Convert a polygon to GeoJSON.
pub fn to_geojson(polygon: &Polygon) -> String {
    let mut s = String::from(r#"{"type":"Polygon","coordinates":["#);

    // Exterior ring
    s.push('[');
    for (i, p) in polygon.exterior.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{},{}]", p.x, p.y));
    }
    s.push(']');

    // Holes
    for hole in &polygon.holes {
        s.push_str(",[");
        for (i, p) in hole.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{},{}]", p.x, p.y));
        }
        s.push(']');
    }

    s.push_str("]}");
    s
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, size: f64) -> Polygon {
        Polygon::new(
            vec![
                Point2D::new(x, y),
                Point2D::new(x + size, y),
                Point2D::new(x + size, y + size),
                Point2D::new(x, y + size),
                Point2D::new(x, y),
            ],
            vec![],
        )
    }

    // ── Point in polygon ─────────────────────────────────────────────────

    #[test]
    fn test_point_in_polygon_inside() {
        let poly = square(0.0, 0.0, 10.0);
        assert!(point_in_polygon(&Point2D::new(5.0, 5.0), &poly));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let poly = square(0.0, 0.0, 10.0);
        assert!(!point_in_polygon(&Point2D::new(15.0, 5.0), &poly));
    }

    #[test]
    fn test_point_in_polygon_with_hole() {
        let poly = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(10.0, 0.0),
                Point2D::new(10.0, 10.0),
                Point2D::new(0.0, 10.0),
                Point2D::new(0.0, 0.0),
            ],
            vec![vec![
                Point2D::new(3.0, 3.0),
                Point2D::new(3.0, 7.0),
                Point2D::new(7.0, 7.0),
                Point2D::new(7.0, 3.0),
                Point2D::new(3.0, 3.0),
            ]],
        );
        // Inside exterior but inside hole -> outside
        assert!(!point_in_polygon(&Point2D::new(5.0, 5.0), &poly));
        // Inside exterior, outside hole -> inside
        assert!(point_in_polygon(&Point2D::new(1.0, 1.0), &poly));
    }

    // ── Polygon area ─────────────────────────────────────────────────────

    #[test]
    fn test_area_triangle() {
        let tri = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(4.0, 0.0),
                Point2D::new(0.0, 3.0),
                Point2D::new(0.0, 0.0),
            ],
            vec![],
        );
        assert!((tri.area() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_area_square() {
        let sq = square(0.0, 0.0, 5.0);
        assert!((sq.area() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_area_with_hole() {
        let poly = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(10.0, 0.0),
                Point2D::new(10.0, 10.0),
                Point2D::new(0.0, 10.0),
                Point2D::new(0.0, 0.0),
            ],
            vec![vec![
                Point2D::new(2.0, 2.0),
                Point2D::new(2.0, 4.0),
                Point2D::new(4.0, 4.0),
                Point2D::new(4.0, 2.0),
                Point2D::new(2.0, 2.0),
            ]],
        );
        // 100 - 4 = 96
        assert!((poly.area() - 96.0).abs() < 1e-6);
    }

    // ── Perimeter ────────────────────────────────────────────────────────

    #[test]
    fn test_perimeter_square() {
        let sq = square(0.0, 0.0, 5.0);
        assert!((sq.perimeter() - 20.0).abs() < 1e-6);
    }

    // ── Convex hull ──────────────────────────────────────────────────────

    #[test]
    fn test_convex_hull_square_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // interior point
        ];
        let hull = convex_hull(&pts);
        // Hull should have 4 distinct vertices + closing = 5
        assert_eq!(hull.exterior.len(), 5);
        assert!((hull.area() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_convex_hull_collinear() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
        ];
        let hull = convex_hull(&pts);
        // Collinear points: degenerate hull (line)
        assert!(hull.area() < 1e-6);
    }

    // ── Segment intersection ─────────────────────────────────────────────

    #[test]
    fn test_segments_crossing() {
        let a1 = Point2D::new(0.0, 0.0);
        let a2 = Point2D::new(2.0, 2.0);
        let b1 = Point2D::new(0.0, 2.0);
        let b2 = Point2D::new(2.0, 0.0);
        assert!(segments_intersect(&a1, &a2, &b1, &b2));
    }

    #[test]
    fn test_segments_parallel() {
        let a1 = Point2D::new(0.0, 0.0);
        let a2 = Point2D::new(2.0, 0.0);
        let b1 = Point2D::new(0.0, 1.0);
        let b2 = Point2D::new(2.0, 1.0);
        assert!(!segments_intersect(&a1, &a2, &b1, &b2));
    }

    #[test]
    fn test_segments_collinear_overlapping() {
        let a1 = Point2D::new(0.0, 0.0);
        let a2 = Point2D::new(2.0, 0.0);
        let b1 = Point2D::new(1.0, 0.0);
        let b2 = Point2D::new(3.0, 0.0);
        assert!(segments_intersect(&a1, &a2, &b1, &b2));
    }

    // ── Polygon intersection ─────────────────────────────────────────────

    #[test]
    fn test_polygon_intersection_overlap() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let result = polygon_intersection(&a, &b);
        assert!(!result.is_empty());
        // Intersection of (0,0)-(2,2) and (1,1)-(3,3) = (1,1)-(2,2) = area 1
        let area: f64 = result.iter().map(|p| p.area()).sum();
        assert!((area - 1.0).abs() < 0.1);
    }

    // ── Delaunay triangulation ───────────────────────────────────────────

    #[test]
    fn test_delaunay_4_points() {
        // Square vertices
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let tris = delaunay_triangulation(&pts);
        // 4 points in general position -> 2 triangles
        assert_eq!(tris.len(), 2);
        // All indices should be valid
        for tri in &tris {
            for &idx in tri {
                assert!(idx < 4);
            }
        }
    }

    #[test]
    fn test_delaunay_many_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
        ];
        let tris = delaunay_triangulation(&pts);
        // 6 points on a grid -> should produce 4 triangles
        assert!(tris.len() >= 4);
    }

    // ── Douglas-Peucker simplification ───────────────────────────────────

    #[test]
    fn test_simplify_removes_intermediate() {
        let coords = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.01),  // nearly collinear
            Point2D::new(2.0, -0.01), // nearly collinear
            Point2D::new(3.0, 0.0),
        ];
        let simplified = simplify(&coords, 0.1);
        assert_eq!(simplified.len(), 2); // only endpoints
    }

    #[test]
    fn test_simplify_keeps_significant() {
        let coords = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 5.0), // big deviation
            Point2D::new(2.0, 0.0),
        ];
        let simplified = simplify(&coords, 0.1);
        assert_eq!(simplified.len(), 3); // all kept
    }

    // ── Buffer polygon ───────────────────────────────────────────────────

    #[test]
    fn test_buffer_polygon_grows() {
        let sq = square(0.0, 0.0, 10.0);
        let buffered = buffer_polygon(&sq, 1.0, 4);
        assert!(buffered.area() > sq.area());
    }

    // ── WKT round-trip ───────────────────────────────────────────────────

    #[test]
    fn test_wkt_roundtrip() {
        let poly = square(0.0, 0.0, 1.0);
        let wkt = to_wkt(&poly);
        assert!(wkt.starts_with("POLYGON"));
        let parsed = from_wkt(&wkt).unwrap();
        assert_eq!(parsed.exterior.len(), poly.exterior.len());
        assert!((parsed.area() - poly.area()).abs() < 1e-6);
    }

    // ── GeoJSON ──────────────────────────────────────────────────────────

    #[test]
    fn test_geojson_output() {
        let poly = square(0.0, 0.0, 1.0);
        let json = to_geojson(&poly);
        assert!(json.contains("\"type\":\"Polygon\""));
        assert!(json.contains("\"coordinates\""));
    }

    // ── STRtree ──────────────────────────────────────────────────────────

    #[test]
    fn test_strtree_query() {
        let items: Vec<(usize, f64, f64, f64, f64)> = vec![
            (0, 0.0, 0.0, 1.0, 1.0),
            (1, 5.0, 5.0, 6.0, 6.0),
            (2, 10.0, 10.0, 11.0, 11.0),
        ];
        let tree = STRtree::new(&items);
        let hits = tree.query(4.0, 4.0, 7.0, 7.0);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn test_strtree_nearest() {
        let items: Vec<(usize, f64, f64, f64, f64)> = vec![
            (0, 0.0, 0.0, 1.0, 1.0),
            (1, 10.0, 10.0, 11.0, 11.0),
            (2, 100.0, 100.0, 101.0, 101.0),
        ];
        let tree = STRtree::new(&items);
        let nearest = tree.nearest(0.5, 0.5, 1);
        assert_eq!(nearest, vec![0]);
    }

    // ── Voronoi ──────────────────────────────────────────────────────────

    #[test]
    fn test_voronoi_3_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(5.0, 10.0),
        ];
        let cells = voronoi_diagram(&pts, (-10.0, -10.0, 20.0, 20.0));
        assert_eq!(cells.len(), 3);
        // Each cell should have non-zero area
        for cell in &cells {
            assert!(cell.area() > 0.0, "Voronoi cell has zero area");
        }
    }

    // ── LineString ───────────────────────────────────────────────────────

    #[test]
    fn test_linestring_length_and_closed() {
        let ls = LineString::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 4.0),
        ]);
        assert!((ls.length() - 7.0).abs() < 1e-6);
        assert!(!ls.is_closed());

        let closed = LineString::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 0.0),
        ]);
        assert!(closed.is_closed());
    }

    // ── Polygon validity ─────────────────────────────────────────────────

    #[test]
    fn test_polygon_validity() {
        let valid = square(0.0, 0.0, 5.0);
        assert!(valid.is_valid());

        // Self-intersecting (bowtie)
        let invalid = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(2.0, 2.0),
                Point2D::new(2.0, 0.0),
                Point2D::new(0.0, 2.0),
                Point2D::new(0.0, 0.0),
            ],
            vec![],
        );
        assert!(!invalid.is_valid());
    }

    // ── Polygons intersect predicate ─────────────────────────────────────

    #[test]
    fn test_polygons_intersect_predicate() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        assert!(polygons_intersect(&a, &b));

        let c = square(5.0, 5.0, 1.0);
        assert!(!polygons_intersect(&a, &c));
    }

    // ── Polygon contains polygon ─────────────────────────────────────────

    #[test]
    fn test_polygon_contains() {
        let outer = square(0.0, 0.0, 10.0);
        let inner = square(2.0, 2.0, 3.0);
        assert!(polygon_contains_polygon(&outer, &inner));
        assert!(!polygon_contains_polygon(&inner, &outer));
    }

    // ── Centroid ─────────────────────────────────────────────────────────

    #[test]
    fn test_centroid() {
        let sq = square(0.0, 0.0, 4.0);
        let c = sq.centroid();
        assert!((c.x - 2.0).abs() < 1e-6);
        assert!((c.y - 2.0).abs() < 1e-6);
    }
}
