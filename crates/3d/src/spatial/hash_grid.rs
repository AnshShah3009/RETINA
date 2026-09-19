//! Spatial Hash Grid for O(1) amortized neighbor queries on uniform distributions.
//!
//! Inspired by NVIDIA Warp's `HashGrid`. Points are hashed into cells of fixed size,
//! then sorted by cell for cache-friendly neighbor iteration. Radius queries check
//! only the 27 neighboring cells (3×3×3 stencil).
//!
//! Best for: particle simulations, ICP with bounded correspondence distance,
//! uniform point clouds. For highly non-uniform data, prefer KDTree.

use nalgebra::Point3;

/// Spatial hash grid with sort-based construction and O(1) cell lookup.
pub struct HashGrid {
    cell_size: f32,
    inv_cell_size: f32,
    table_size: usize,
    /// Points sorted by cell hash for cache locality.
    sorted_points: Vec<Point3<f32>>,
    /// Original indices of sorted points.
    sorted_indices: Vec<usize>,
    /// First point index in sorted array for each hash bucket.
    cell_start: Vec<u32>,
    /// Number of points in each hash bucket.
    cell_count: Vec<u32>,
}

const HASH_P1: u32 = 73856093;
const HASH_P2: u32 = 19349663;
const HASH_P3: u32 = 83492791;

#[inline]
fn hash_cell(ix: i32, iy: i32, iz: i32, table_size: usize) -> usize {
    let h = (ix.wrapping_mul(HASH_P1 as i32) as u32)
        ^ (iy.wrapping_mul(HASH_P2 as i32) as u32)
        ^ (iz.wrapping_mul(HASH_P3 as i32) as u32);
    (h as usize) % table_size
}

impl HashGrid {
    /// Build a hash grid from points with the given cell size.
    ///
    /// `cell_size` should typically be set to the query radius for optimal performance.
    /// Table size is automatically chosen as the next power of 2 >= 2*n_points.
    pub fn build(points: &[Point3<f32>], cell_size: f32) -> Self {
        let n = points.len();
        let table_size = (n * 2).next_power_of_two().max(256);
        let inv_cell_size = 1.0 / cell_size;

        // Compute cell hash for each point
        let mut entries: Vec<(usize, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let ix = (p.x * inv_cell_size).floor() as i32;
                let iy = (p.y * inv_cell_size).floor() as i32;
                let iz = (p.z * inv_cell_size).floor() as i32;
                (hash_cell(ix, iy, iz, table_size), i)
            })
            .collect();

        // Sort by hash for cache-friendly access
        entries.sort_unstable_by_key(|e| e.0);

        let sorted_points: Vec<_> = entries.iter().map(|&(_, i)| points[i]).collect();
        let sorted_indices: Vec<_> = entries.iter().map(|&(_, i)| i).collect();

        // Build cell_start / cell_count arrays
        let mut cell_start = vec![u32::MAX; table_size];
        let mut cell_count = vec![0u32; table_size];

        for (pos, &(hash, _)) in entries.iter().enumerate() {
            cell_count[hash] += 1;
            if cell_start[hash] == u32::MAX {
                cell_start[hash] = pos as u32;
            }
        }

        Self {
            cell_size,
            inv_cell_size,
            table_size,
            sorted_points,
            sorted_indices,
            cell_start,
            cell_count,
        }
    }

    /// Find all points within `radius` of `query`. Returns (original_index, squared_distance).
    pub fn radius_search(&self, query: &Point3<f32>, radius: f32) -> Vec<(usize, f32)> {
        let radius_sq = radius * radius;
        let mut results = Vec::new();

        // Determine cell range to check (3×3×3 stencil)
        let qx = (query.x * self.inv_cell_size).floor() as i32;
        let qy = (query.y * self.inv_cell_size).floor() as i32;
        let qz = (query.z * self.inv_cell_size).floor() as i32;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let h = hash_cell(qx + dx, qy + dy, qz + dz, self.table_size);
                    let start = self.cell_start[h];
                    if start == u32::MAX {
                        continue;
                    }
                    let count = self.cell_count[h] as usize;
                    let start = start as usize;
                    for i in start..start + count {
                        let p = &self.sorted_points[i];
                        let dx = p.x - query.x;
                        let dy = p.y - query.y;
                        let dz = p.z - query.z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq <= radius_sq {
                            results.push((self.sorted_indices[i], dist_sq));
                        }
                    }
                }
            }
        }

        results
    }

    /// Find the nearest point to `query` within `max_radius`.
    /// Returns (original_index, point, squared_distance).
    pub fn nearest(
        &self,
        query: &Point3<f32>,
        max_radius: f32,
    ) -> Option<(usize, Point3<f32>, f32)> {
        let max_sq = max_radius * max_radius;
        let mut best: Option<(usize, Point3<f32>, f32)> = None;

        let qx = (query.x * self.inv_cell_size).floor() as i32;
        let qy = (query.y * self.inv_cell_size).floor() as i32;
        let qz = (query.z * self.inv_cell_size).floor() as i32;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let h = hash_cell(qx + dx, qy + dy, qz + dz, self.table_size);
                    let start = self.cell_start[h];
                    if start == u32::MAX {
                        continue;
                    }
                    let count = self.cell_count[h] as usize;
                    let start = start as usize;
                    for i in start..start + count {
                        let p = &self.sorted_points[i];
                        let ddx = p.x - query.x;
                        let ddy = p.y - query.y;
                        let ddz = p.z - query.z;
                        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                        if dist_sq <= max_sq {
                            let replace = match best {
                                None => true,
                                Some((_, _, bd)) => dist_sq < bd,
                            };
                            if replace {
                                best = Some((self.sorted_indices[i], *p, dist_sq));
                            }
                        }
                    }
                }
            }
        }

        best
    }

    /// Number of points in the grid.
    pub fn len(&self) -> usize {
        self.sorted_points.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.sorted_points.is_empty()
    }

    /// The cell size used for hashing.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_grid_basic() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.1, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.1, 1.0, 1.0),
            Point3::new(5.0, 5.0, 5.0),
        ];

        let grid = HashGrid::build(&points, 0.5);
        assert_eq!(grid.len(), 5);

        // Query near origin should find first two points
        let results = grid.radius_search(&Point3::new(0.05, 0.0, 0.0), 0.2);
        assert_eq!(results.len(), 2);

        // Query far away should find nothing
        let results = grid.radius_search(&Point3::new(10.0, 10.0, 10.0), 0.5);
        assert_eq!(results.len(), 0);

        // Nearest to (1.05, 1.0, 1.0) should be point 2 (1.0, 1.0, 1.0) — distance 0.05
        let nearest = grid.nearest(&Point3::new(1.05, 1.0, 1.0), 0.5);
        assert!(nearest.is_some());
        assert_eq!(nearest.unwrap().0, 2);
    }

    #[test]
    fn test_hash_grid_vs_brute_force() {
        // Generate random points and verify HashGrid matches brute-force
        let n = 1000;
        let points: Vec<_> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                Point3::new(t.cos() * 2.0, t.sin() * 2.0, t * 0.1)
            })
            .collect();

        let grid = HashGrid::build(&points, 0.3);
        let query = Point3::new(1.0, 0.0, 0.0);
        let radius = 0.3;

        let hash_results = grid.radius_search(&query, radius);

        // Brute force
        let brute_results: Vec<(usize, f32)> = points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let d = (p - query).norm_squared();
                if d <= radius * radius {
                    Some((i, d))
                } else {
                    None
                }
            })
            .collect();

        // Same count
        assert_eq!(
            hash_results.len(),
            brute_results.len(),
            "HashGrid found {} vs brute force {}",
            hash_results.len(),
            brute_results.len()
        );

        // Same indices (order may differ)
        let mut h_idx: Vec<_> = hash_results.iter().map(|r| r.0).collect();
        let mut b_idx: Vec<_> = brute_results.iter().map(|r| r.0).collect();
        h_idx.sort();
        b_idx.sort();
        assert_eq!(h_idx, b_idx);
    }
}
