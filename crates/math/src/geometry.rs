use cv_core::Rect;
use geo::algorithm::convex_hull::ConvexHull;
use geo::algorithm::simplify::Simplify;
use geo::{Area, BooleanOps, EuclideanDistance, Polygon};
use ndarray::Array2;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

/// Wrapper for a Polygon that implements RTreeObject.
#[derive(Debug, Clone)]
pub struct IndexedPolygon {
    pub id: usize,
    pub polygon: Polygon<f64>,
}

impl RTreeObject for IndexedPolygon {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        use geo::BoundingRect;
        let bbox = self.polygon.bounding_rect().unwrap_or_else(|| {
            geo::Rect::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 0.0, y: 0.0 })
        });

        let min = bbox.min();
        let max = bbox.max();
        AABB::from_corners([min.x, min.y], [max.x, max.y])
    }
}

impl PointDistance for IndexedPolygon {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let p = geo::Point::new(point[0], point[1]);
        // euclidean_distance returns the distance. We need squared distance.
        let d = self.polygon.euclidean_distance(&p);
        d * d
    }
}

pub struct SpatialIndex {
    tree: RTree<IndexedPolygon>,
}

impl SpatialIndex {
    pub fn new(polygons: Vec<Polygon<f64>>) -> Self {
        let indexed: Vec<IndexedPolygon> = polygons
            .into_iter()
            .enumerate()
            .map(|(id, polygon)| IndexedPolygon { id, polygon })
            .collect();
        Self {
            tree: RTree::bulk_load(indexed),
        }
    }

    /// Find all polygons that intersect with the given query envelope.
    pub fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
        self.tree
            .locate_in_envelope_intersecting(&envelope)
            .map(|ip| ip.id)
            .collect()
    }

    /// Find nearest neighbor to a point.
    pub fn nearest(&self, x: f64, y: f64) -> Option<usize> {
        self.tree.nearest_neighbor(&[x, y]).map(|ip| ip.id)
    }

    /// Find all polygons containing the given point.
    pub fn query_contains_point(&self, x: f64, y: f64) -> Vec<usize> {
        use geo::Contains;
        let point = geo::Point::new(x, y);
        let envelope = AABB::from_point([x, y]);
        self.tree
            .locate_in_envelope_intersecting(&envelope)
            .filter(|ip| ip.polygon.contains(&point))
            .map(|ip| ip.id)
            .collect()
    }

    /// Find all polygons intersecting the given polygon.
    pub fn query_intersect_polygon(&self, polygon: &Polygon<f64>) -> Vec<usize> {
        use geo::{BoundingRect, Intersects};
        if let Some(bbox) = polygon.bounding_rect() {
            let min = bbox.min();
            let max = bbox.max();
            let envelope = AABB::from_corners([min.x, min.y], [max.x, max.y]);

            self.tree
                .locate_in_envelope_intersecting(&envelope)
                .filter(|ip| ip.polygon.intersects(polygon))
                .map(|ip| ip.id)
                .collect()
        } else {
            vec![]
        }
    }
}

/// Compute the convex hull of a polygon.
pub fn convex_hull(polygon: &Polygon<f64>) -> Polygon<f64> {
    polygon.convex_hull()
}

/// Simplify a polygon using Ramer-Douglas-Peucker algorithm.
pub fn simplify(polygon: &Polygon<f64>, epsilon: f64) -> Polygon<f64> {
    polygon.simplify(&epsilon)
}

/// Buffer a polygon by a given distance.
/// Returns a MultiPolygon as buffering can create disjoint shapes or holes.
pub fn buffer(polygon: &Polygon<f64>, distance: f64) -> geo::MultiPolygon<f64> {
    geo_buffer::buffer_polygon(polygon, distance)
}

/// Vectorized IoU computation using ndarray.
/// Computes IoU between all pairs from boxes1 and boxes2.
pub fn vectorized_iou(boxes1: &[Rect], boxes2: &[Rect]) -> Array2<f32> {
    let mut ious = Array2::zeros((boxes1.len(), boxes2.len()));
    let n2 = boxes2.len();

    // Use raw data pointer or simple indexing if Array2 is not being cooperative with par_iter
    let ious_raw = ious.as_slice_mut().expect("ndarray should be contiguous");

    ious_raw
        .par_chunks_mut(n2)
        .enumerate()
        .for_each(|(i, row)| {
            let b1 = &boxes1[i];
            for (j, b2) in boxes2.iter().enumerate() {
                row[j] = b1.iou(b2);
            }
        });

    ious
}

/// Compute IoU for two polygons.
pub fn polygon_iou(p1: &Polygon<f64>, p2: &Polygon<f64>) -> f64 {
    let intersection = p1.intersection(p2);
    let intersection_area = intersection.unsigned_area();

    if intersection_area <= 1e-9 {
        return 0.0;
    }

    let union = p1.union(p2);
    let union_area = union.unsigned_area();

    if union_area <= 1e-9 {
        return 0.0;
    }

    intersection_area / union_area
}

/// Vectorized Polygon IoU.
pub fn vectorized_polygon_iou(polys1: &[Polygon<f64>], polys2: &[Polygon<f64>]) -> Array2<f64> {
    let mut ious = Array2::zeros((polys1.len(), polys2.len()));
    let n2 = polys2.len();

    let ious_raw = ious.as_slice_mut().expect("ndarray should be contiguous");

    ious_raw
        .par_chunks_mut(n2)
        .enumerate()
        .for_each(|(i, row)| {
            let p1 = &polys1[i];
            for (j, p2) in polys2.iter().enumerate() {
                row[j] = polygon_iou(p1, p2);
            }
        });

    ious
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::LineString;

    fn make_square(x: f64, y: f64, size: f64) -> Polygon<f64> {
        let coords = vec![
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y),
        ];
        let line_string = LineString::from(coords);
        Polygon::new(line_string, vec![])
    }

    #[test]
    fn test_convex_hull() {
        // Concave shape: a "C" shape or similar.
        // Or simplified: a square with a dent.
        let coords = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (5.0, 5.0), // dent
            (0.0, 10.0),
            (0.0, 0.0),
        ];
        let poly = Polygon::new(LineString::from(coords), vec![]);
        let hull = convex_hull(&poly);

        // Hull should remove the dent (5,5).
        // Area of hull should be larger than poly.
        assert!(hull.unsigned_area() > poly.unsigned_area());
    }

    #[test]
    fn test_simplify() {
        // A line with jitter.
        let coords = vec![
            (0.0, 0.0),
            (5.0, 0.1), // jitter
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ];
        let poly = Polygon::new(LineString::from(coords), vec![]);
        let simple = simplify(&poly, 0.2);

        // Should remove (5.0, 0.1)
        // Check vertex count
        assert!(simple.exterior().0.len() < poly.exterior().0.len());
    }

    #[test]
    fn test_spatial_index() {
        let polygons = vec![
            make_square(0.0, 0.0, 10.0),   // (0,0)-(10,10)
            make_square(20.0, 20.0, 10.0), // (20,20)-(30,30)
        ];
        let index = SpatialIndex::new(polygons);

        // Query intersecting (5,5). Should match index 0.
        let hits = index.query_bbox(4.0, 4.0, 6.0, 6.0);
        assert_eq!(hits, vec![0]);

        // Nearest to (5,5) -> 0
        assert_eq!(index.nearest(5.0, 5.0), Some(0));

        // Nearest to (25,25) -> 1
        assert_eq!(index.nearest(25.0, 25.0), Some(1));
    }

    #[test]
    fn test_buffer() {
        let poly = make_square(0.0, 0.0, 10.0);
        let buffered = buffer(&poly, 1.0);

        // Original area 100.
        // Buffered by 1.0 -> approx (10+2)*(10+2) = 144 (corners are rounded, so slightly less).
        // Check finding area.
        use geo::Area;
        let area = buffered.unsigned_area();
        assert!(area > 100.0);
        assert!(area < 144.0 + 1.0); // Simple check
    }

    #[test]
    fn test_predicates() {
        let polygons = vec![
            make_square(0.0, 0.0, 10.0),   // (0,0)-(10,10)
            make_square(20.0, 20.0, 10.0), // (20,20)-(30,30)
        ];
        let index = SpatialIndex::new(polygons);

        // Test Contains Point
        // (5,5) is inside poly 0
        let hits = index.query_contains_point(5.0, 5.0);
        assert_eq!(hits, vec![0]);

        // (25,25) is inside poly 1
        let hits = index.query_contains_point(25.0, 25.0);
        assert_eq!(hits, vec![1]);

        // (15,15) is nowhere
        let hits = index.query_contains_point(15.0, 15.0);
        assert!(hits.is_empty());

        // Test Intersect Polygon
        // Square at (5,5) size 2 -> (5,5)-(7,7). Fully inside 0.
        let query_poly = make_square(5.0, 5.0, 2.0);
        let hits = index.query_intersect_polygon(&query_poly);
        assert_eq!(hits, vec![0]);

        // Square overlapping (8, 8) to (12, 12). Overlaps 0?
        // 0 is (0,0)-(10,10). Yes.
        let query_poly = make_square(8.0, 8.0, 4.0);
        let hits = index.query_intersect_polygon(&query_poly);
        assert_eq!(hits, vec![0]);

        // Square covering both? (0,0) to (30,30).
        let query_poly = make_square(0.0, 0.0, 30.0);
        let mut hits = index.query_intersect_polygon(&query_poly);
        hits.sort();
        assert_eq!(hits, vec![0, 1]);
    }
}
