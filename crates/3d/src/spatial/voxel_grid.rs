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

    pub fn compute_centroids(&mut self, points: &[Point3<f32>]) {
        for voxel in self.grid.values_mut() {
            if !voxel.indices.is_empty() {
                let mut centroid = Point3::origin();
                for &idx in &voxel.indices {
                    centroid += points[idx].coords;
                }
                centroid /= voxel.indices.len() as f32;
                voxel.centroid = Some(centroid);
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
                    for &idx in &voxel.indices {
                        centroid += points[idx].coords;
                    }
                    Some(centroid / voxel.indices.len() as f32)
                }
            })
            .collect()
    }
}
