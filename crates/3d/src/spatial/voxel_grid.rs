use nalgebra::Point3;
use std::collections::HashMap;

/// VoxelGrid for voxelization
pub struct VoxelGrid {
    pub origin: Point3<f32>,
    pub voxel_size: f32,
    pub grid: HashMap<(i32, i32, i32), Voxel>,
}

#[derive(Debug, Clone)]
pub struct Voxel {
    pub indices: Vec<usize>,
    pub centroid: Option<Point3<f32>>,
}

impl VoxelGrid {
    pub fn new(origin: Point3<f32>, voxel_size: f32) -> Self {
        Self {
            origin,
            voxel_size,
            grid: HashMap::new(),
        }
    }

    pub fn insert(&mut self, point: Point3<f32>, index: usize) {
        let key = self.point_to_voxel(&point);
        self.grid
            .entry(key)
            .or_insert_with(|| Voxel {
                indices: Vec::new(),
                centroid: None,
            })
            .indices
            .push(index);
    }

    pub fn point_to_voxel(&self, point: &Point3<f32>) -> (i32, i32, i32) {
        (
            ((point.x - self.origin.x) / self.voxel_size).floor() as i32,
            ((point.y - self.origin.y) / self.voxel_size).floor() as i32,
            ((point.z - self.origin.z) / self.voxel_size).floor() as i32,
        )
    }

    pub fn voxel_to_point(&self, voxel: (i32, i32, i32)) -> Point3<f32> {
        Point3::new(
            voxel.0 as f32 * self.voxel_size + self.origin.x,
            voxel.1 as f32 * self.voxel_size + self.origin.y,
            voxel.2 as f32 * self.voxel_size + self.origin.z,
        )
    }

    /// Average the points in each voxel.
    ///
    /// Indices are stored by `insert` and are not re-validated here, so a grid
    /// built from one point array and then asked to average a *different* one -
    /// or one that has since shrunk - indexed past the end and panicked. An
    /// out-of-bounds index is now skipped rather than trusted: a grid holding a
    /// stale index should lose that voxel, not abort the caller.
    ///
    /// This is the same defect class as `icp_accumulate`, which read
    /// correspondence indices into point arrays with no bounds check.
    pub fn compute_centroids(&mut self, points: &[Point3<f32>]) {
        for voxel in self.grid.values_mut() {
            if voxel.indices.is_empty() {
                continue;
            }
            let mut centroid = Point3::origin();
            let mut used = 0usize;
            for &idx in &voxel.indices {
                if let Some(p) = points.get(idx) {
                    centroid += p.coords;
                    used += 1;
                }
            }
            if used > 0 {
                voxel.centroid = Some(centroid / used as f32);
            } else {
                // Every index was stale, so this voxel has no centroid at all.
                voxel.centroid = None;
            }
        }
    }

    pub fn downsample(&self, points: &[Point3<f32>]) -> Vec<Point3<f32>> {
        self.grid
            .values()
            .filter_map(|voxel| {
                if voxel.indices.is_empty() {
                    None
                } else {
                    let mut centroid = Point3::origin();
                    let mut used = 0usize;
                    for &idx in &voxel.indices {
                        if let Some(p) = points.get(idx) {
                            centroid += p.coords;
                            used += 1;
                        }
                    }
                    if used == 0 {
                        return None;
                    }
                    Some(centroid / used as f32)
                }
            })
            .collect()
    }
}
