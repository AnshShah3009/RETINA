//! TSDF (Truncated Signed Distance Function) Integration
//!
//! Core algorithm for fusing multiple RGBD images into a 3D volume.
//! Based on "KinectFusion: Real-Time Dense Surface Mapping and Tracking" by Newcombe et al.

use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

use cv_core::CameraIntrinsicsF32;
use cv_runtime::orchestrator::RuntimeRunner;

/// Re-export the canonical f32 camera intrinsics from cv-core.
pub type CameraIntrinsics = CameraIntrinsicsF32;

/// Voxel block for TSDF volume
#[derive(Debug, Clone)]
pub struct VoxelBlock {
    pub coords: (i32, i32, i32),  // Block coordinates in grid
    pub tsdf: Vec<f32>,           // TSDF values (8x8x8 = 512 voxels per block)
    pub weights: Vec<f32>,        // Integration weights
    pub colors: Vec<Vector3<u8>>, // RGB colors (optional)
}

impl VoxelBlock {
    pub const BLOCK_SIZE: usize = 8;
    pub const VOXELS_PER_BLOCK: usize = 512; // 8^3

    pub fn new(coords: (i32, i32, i32)) -> Self {
        Self {
            coords,
            tsdf: vec![1.0; Self::VOXELS_PER_BLOCK], // Initialize to empty (positive)
            weights: vec![0.0; Self::VOXELS_PER_BLOCK],
            colors: vec![Vector3::new(0, 0, 0); Self::VOXELS_PER_BLOCK],
        }
    }

    /// Get linear index from local voxel coordinates
    pub fn voxel_index(local: (usize, usize, usize)) -> usize {
        local.0 + local.1 * Self::BLOCK_SIZE + local.2 * Self::BLOCK_SIZE * Self::BLOCK_SIZE
    }

    /// Get voxel value at local coordinates
    pub fn get_tsdf(&self, local: (usize, usize, usize)) -> f32 {
        self.tsdf[Self::voxel_index(local)]
    }

    /// Set voxel value
    pub fn set_voxel(
        &mut self,
        local: (usize, usize, usize),
        tsdf: f32,
        weight: f32,
        color: Vector3<u8>,
    ) {
        let idx = Self::voxel_index(local);
        self.tsdf[idx] = tsdf;
        self.weights[idx] = weight;
        self.colors[idx] = color;
    }
}

/// TSDF Volume for RGBD integration
pub struct TSDFVolume {
    voxel_size: f32,
    truncation_distance: f32,
    blocks: HashMap<(i32, i32, i32), VoxelBlock>,
    depth_scale: f32,
    max_weight: f32,
}

impl TSDFVolume {
    pub fn new(voxel_size: f32, truncation_distance: f32) -> Self {
        Self {
            voxel_size,
            truncation_distance,
            blocks: HashMap::new(),
            depth_scale: 1000.0, // Default: depth in mm
            max_weight: 100.0,
        }
    }

    pub fn with_depth_scale(mut self, scale: f32) -> Self {
        self.depth_scale = scale;
        self
    }

    /// Integrate a single RGBD frame using best available runner
    pub fn integrate(
        &mut self,
        depth_image: &[f32],
        color_image: Option<&[Vector3<u8>]>,
        intrinsics: &CameraIntrinsics,
        extrinsics: &Matrix4<f32>,
        width: usize,
        height: usize,
    ) {
        if let Ok(runner) = cv_runtime::best_runner() {
            self.integrate_ctx(
                depth_image,
                color_image,
                intrinsics,
                extrinsics,
                width,
                height,
                &runner,
            );
        }
    }

    /// Integrate a single RGBD frame with explicit context
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_ctx(
        &mut self,
        depth_image: &[f32],
        color_image: Option<&[Vector3<u8>]>,
        intrinsics: &CameraIntrinsics,
        extrinsics: &Matrix4<f32>,
        width: usize,
        height: usize,
        group: &RuntimeRunner,
    ) {
        // NOTE: A previous revision dispatched GPU integration into a freshly
        // allocated 256^3 dense volume and returned early. Without a
        // read-back/merge step into the sparse block storage, every frame was
        // integrated into a throwaway buffer and silently dropped, leaving
        // extract_mesh()/extract_point_cloud() empty on GPU machines. Until
        // merge-back exists, always use the CPU path below so data is actually
        // integrated into self.blocks.

        // CPU Fallback (Rayon)
        // Compute inverse extrinsics for projective distance
        let extrinsics_inv = extrinsics.try_inverse().unwrap_or_else(Matrix4::identity);
        let camera_origin = extrinsics.transform_point(&Point3::origin());

        let updates: Vec<_> = group.run(|| {
            (0..height)
                .into_par_iter()
                .flat_map(|v| {
                    let mut local_updates = Vec::new();

                    for u in 0..width {
                        let idx = v * width + u;
                        let depth = depth_image[idx] / self.depth_scale;

                        if depth <= 0.0 || depth > 10.0 {
                            continue;
                        }

                        // Backproject to 3D in camera space
                        let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                        let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                        let z = depth;
                        let point_camera = Point3::new(x, y, z);

                        // Transform to world space
                        let point_world = extrinsics.transform_point(&point_camera);

                        // Get color
                        let color = color_image
                            .map(|c| c[idx])
                            .unwrap_or(Vector3::new(128, 128, 128));

                        // Find affected voxels along ray
                        let ray_dir = (point_world - camera_origin).normalize();
                        let start =
                            camera_origin + ray_dir * (depth - self.truncation_distance).max(0.01);
                        let end = camera_origin + ray_dir * (depth + self.truncation_distance);

                        // March along ray and update voxels
                        let steps = ((end - start).norm() / self.voxel_size).ceil() as usize;
                        for i in 0..=steps {
                            let t = i as f32 / steps.max(1) as f32;
                            let voxel_pos = start + (end - start) * t;

                            // Calculate TSDF value using projective distance:
                            // Transform voxel back to camera frame and compare depths
                            let voxel_camera = extrinsics_inv.transform_point(&voxel_pos);
                            let sdf = voxel_camera.z - depth;

                            let tsdf = (sdf / self.truncation_distance).clamp(-1.0, 1.0);

                            local_updates.push((voxel_pos, tsdf, color));
                        }
                    }

                    local_updates
                })
                .collect()
        });

        // Apply updates
        for (pos, tsdf, color) in updates {
            self.update_voxel(pos, tsdf, color);
        }
    }

    /// Update a single voxel with new TSDF value
    fn update_voxel(&mut self, world_pos: Point3<f32>, tsdf: f32, color: Vector3<u8>) {
        // Convert to voxel coordinates
        let voxel_x = (world_pos.x / self.voxel_size).floor() as i32;
        let voxel_y = (world_pos.y / self.voxel_size).floor() as i32;
        let voxel_z = (world_pos.z / self.voxel_size).floor() as i32;

        // Calculate block coordinates
        let block_x = voxel_x.div_euclid(VoxelBlock::BLOCK_SIZE as i32);
        let block_y = voxel_y.div_euclid(VoxelBlock::BLOCK_SIZE as i32);
        let block_z = voxel_z.div_euclid(VoxelBlock::BLOCK_SIZE as i32);

        let block_coords = (block_x, block_y, block_z);

        // Calculate local voxel coordinates within block
        let local_x = voxel_x.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local_y = voxel_y.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local_z = voxel_z.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local = (local_x, local_y, local_z);

        // Get or create block
        let block = self
            .blocks
            .entry(block_coords)
            .or_insert_with(|| VoxelBlock::new(block_coords));

        // Integrate with running average
        let idx = VoxelBlock::voxel_index(local);
        let old_tsdf = block.tsdf[idx];
        let old_weight = block.weights[idx];

        let new_tsdf = (old_tsdf * old_weight + tsdf) / (old_weight + 1.0);
        let new_weight = (old_weight + 1.0).min(self.max_weight);

        // Weighted average for colors instead of overwriting
        let old_color = block.colors[idx];
        let new_color = Vector3::new(
            ((old_color.x as f32 * old_weight + color.x as f32) / (old_weight + 1.0)) as u8,
            ((old_color.y as f32 * old_weight + color.y as f32) / (old_weight + 1.0)) as u8,
            ((old_color.z as f32 * old_weight + color.z as f32) / (old_weight + 1.0)) as u8,
        );

        block.tsdf[idx] = new_tsdf;
        block.weights[idx] = new_weight;
        block.colors[idx] = new_color;
    }

    /// Extract surface mesh using Marching Cubes
    pub fn extract_mesh(&self) -> Vec<Triangle> {
        let mut triangles = Vec::new();

        for (block_coords, block) in &self.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE - 1 {
                for y in 0..VoxelBlock::BLOCK_SIZE - 1 {
                    for x in 0..VoxelBlock::BLOCK_SIZE - 1 {
                        // Get corner values
                        let corners = [
                            block.get_tsdf((x, y, z)),
                            block.get_tsdf((x + 1, y, z)),
                            block.get_tsdf((x + 1, y + 1, z)),
                            block.get_tsdf((x, y + 1, z)),
                            block.get_tsdf((x, y, z + 1)),
                            block.get_tsdf((x + 1, y, z + 1)),
                            block.get_tsdf((x + 1, y + 1, z + 1)),
                            block.get_tsdf((x, y + 1, z + 1)),
                        ];

                        // Marching cubes on this cell
                        let cell_tris =
                            marching_cubes_cell(block_coords, (x, y, z), &corners, self.voxel_size);

                        triangles.extend(cell_tris);
                    }
                }
            }
        }

        triangles
    }

    /// Extract point cloud from surface (zero-crossings)
    pub fn extract_point_cloud(&self) -> Vec<(Point3<f32>, Vector3<f32>, Vector3<u8>)> {
        let mut points = Vec::new();

        for (block_coords, block) in &self.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE {
                for y in 0..VoxelBlock::BLOCK_SIZE {
                    for x in 0..VoxelBlock::BLOCK_SIZE {
                        let local = (x, y, z);
                        let tsdf = block.get_tsdf(local);
                        let weight = block.weights[VoxelBlock::voxel_index(local)];

                        // Check if this is a surface voxel (near zero crossing)
                        if weight > 0.0 && tsdf.abs() < 0.1 {
                            let world_pos = self.voxel_to_world(*block_coords, local);
                            let normal = self.estimate_normal(*block_coords, local);
                            let color = block.colors[VoxelBlock::voxel_index(local)];

                            points.push((world_pos, normal, color));
                        }
                    }
                }
            }
        }

        points
    }

    fn voxel_to_world(
        &self,
        block_coords: (i32, i32, i32),
        local: (usize, usize, usize),
    ) -> Point3<f32> {
        let vx = block_coords.0 * VoxelBlock::BLOCK_SIZE as i32 + local.0 as i32;
        let vy = block_coords.1 * VoxelBlock::BLOCK_SIZE as i32 + local.1 as i32;
        let vz = block_coords.2 * VoxelBlock::BLOCK_SIZE as i32 + local.2 as i32;

        Point3::new(
            vx as f32 * self.voxel_size,
            vy as f32 * self.voxel_size,
            vz as f32 * self.voxel_size,
        )
    }

    fn estimate_normal(
        &self,
        block_coords: (i32, i32, i32),
        local: (usize, usize, usize),
    ) -> Vector3<f32> {
        // Simple central difference gradient
        let x = local.0;
        let y = local.1;
        let z = local.2;

        let dx = self.get_tsdf_safe(block_coords, (x + 1, y, z))
            - self.get_tsdf_safe(block_coords, (x.saturating_sub(1), y, z));
        let dy = self.get_tsdf_safe(block_coords, (x, y + 1, z))
            - self.get_tsdf_safe(block_coords, (x, y.saturating_sub(1), z));
        let dz = self.get_tsdf_safe(block_coords, (x, y, z + 1))
            - self.get_tsdf_safe(block_coords, (x, y, z.saturating_sub(1)));

        Vector3::new(dx, dy, dz).normalize()
    }

    fn get_tsdf_safe(&self, block_coords: (i32, i32, i32), local: (usize, usize, usize)) -> f32 {
        if let Some(block) = self.blocks.get(&block_coords) {
            let x = local.0.min(VoxelBlock::BLOCK_SIZE - 1);
            let y = local.1.min(VoxelBlock::BLOCK_SIZE - 1);
            let z = local.2.min(VoxelBlock::BLOCK_SIZE - 1);
            block.get_tsdf((x, y, z))
        } else {
            1.0 // Empty space
        }
    }
}

/// Triangle for mesh extraction
#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertices: [Point3<f32>; 3],
    pub normals: [Vector3<f32>; 3],
    pub colors: [Vector3<u8>; 3],
}

/// Marching Cubes edge table — bitmask of which edges are intersected for each
/// of the 256 cube configurations.
/// See: http://paulbourke.net/geometry/polygonise/
#[allow(dead_code)]
const MC_EDGE_TABLE: [i32; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x139, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x139, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

/// Maps each of the 12 cube edges to its two endpoint corner indices.
///
/// Standard marching cubes corner numbering (Paul Bourke):
///   Corner 0: (0,0,0)  Corner 1: (1,0,0)  Corner 2: (1,1,0)  Corner 3: (0,1,0)
///   Corner 4: (0,0,1)  Corner 5: (1,0,1)  Corner 6: (1,1,1)  Corner 7: (0,1,1)
const EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1], // edge 0
    [1, 2], // edge 1
    [2, 3], // edge 2
    [3, 0], // edge 3
    [4, 5], // edge 4
    [5, 6], // edge 5
    [6, 7], // edge 6
    [7, 4], // edge 7
    [0, 4], // edge 8
    [1, 5], // edge 9
    [2, 6], // edge 10
    [3, 7], // edge 11
];

/// Marching Cubes triangle table — 256 entries, each a list of edge indices
/// grouped in triples (one triple per triangle), terminated by -1.
/// From Paul Bourke's reference: http://paulbourke.net/geometry/polygonise/
#[rustfmt::skip]
const TRI_TABLE: [[i32; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [ 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [ 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [ 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [ 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [ 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [ 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [ 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [ 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [ 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [ 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [ 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [ 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [ 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [ 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [ 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [ 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [ 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [ 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [ 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [ 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [ 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [ 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1],
    [ 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1],
    [ 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [ 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [ 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [ 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [ 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [ 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [ 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [ 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [ 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [ 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [ 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [ 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [ 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [ 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [ 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [ 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [ 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [ 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [ 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [ 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [ 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [ 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [ 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [ 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [ 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [ 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [ 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [ 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [ 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [ 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [ 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [ 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [ 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [ 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [ 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [ 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [ 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [ 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [ 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [ 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [ 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [ 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

/// Linearly interpolate between two points to find the zero-crossing position.
///
/// Given two corner positions `p1` and `p2` with TSDF values `val1` and `val2`,
/// finds the point along the edge where the isosurface (at `iso_level`) crosses.
fn vertex_interp(
    iso_level: f32,
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    val1: f32,
    val2: f32,
) -> Point3<f32> {
    let eps = 1.0e-5;
    if (val1 - iso_level).abs() < eps {
        return *p1;
    }
    if (val2 - iso_level).abs() < eps {
        return *p2;
    }
    if (val1 - val2).abs() < eps {
        return *p1;
    }
    let mu = (iso_level - val1) / (val2 - val1);
    Point3::new(
        p1.x + mu * (p2.x - p1.x),
        p1.y + mu * (p2.y - p1.y),
        p1.z + mu * (p2.z - p1.z),
    )
}

/// Marching cubes on a single cell.
///
/// Standard corner ordering (Paul Bourke):
/// ```text
///        7 -------- 6
///       /|         /|
///      / |        / |
///     4 -------- 5  |
///     |  3 ------|- 2
///     | /        | /
///     |/         |/
///     0 -------- 1
/// ```
///
/// The 8 corners map to local voxel offsets:
///   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
///   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
#[allow(clippy::needless_range_loop)]
fn marching_cubes_cell(
    block_coords: &(i32, i32, i32),
    local: (usize, usize, usize),
    corners: &[f32; 8],
    voxel_size: f32,
) -> Vec<Triangle> {
    let iso_level = 0.0_f32;

    // Determine cube index from sign of each corner's TSDF value
    let mut cube_index: usize = 0;
    for i in 0..8 {
        if corners[i] < iso_level {
            cube_index |= 1 << i;
        }
    }

    // No surface in this cell
    if MC_EDGE_TABLE[cube_index] == 0 {
        return Vec::new();
    }

    // Compute world-space positions of the 8 corners
    let (lx, ly, lz) = local;
    let base_x = (block_coords.0 * VoxelBlock::BLOCK_SIZE as i32 + lx as i32) as f32 * voxel_size;
    let base_y = (block_coords.1 * VoxelBlock::BLOCK_SIZE as i32 + ly as i32) as f32 * voxel_size;
    let base_z = (block_coords.2 * VoxelBlock::BLOCK_SIZE as i32 + lz as i32) as f32 * voxel_size;

    // Corner offsets in (dx, dy, dz) matching Paul Bourke numbering
    let corner_offsets: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];

    let positions: [Point3<f32>; 8] = core::array::from_fn(|i| {
        Point3::new(
            base_x + corner_offsets[i][0] * voxel_size,
            base_y + corner_offsets[i][1] * voxel_size,
            base_z + corner_offsets[i][2] * voxel_size,
        )
    });

    // Compute interpolated vertex on each intersected edge
    let edge_bits = MC_EDGE_TABLE[cube_index];
    let mut vert_list: [Point3<f32>; 12] = [Point3::origin(); 12];

    for edge in 0..12 {
        if edge_bits & (1 << edge) != 0 {
            let [c0, c1] = EDGE_VERTICES[edge];
            vert_list[edge] = vertex_interp(
                iso_level,
                &positions[c0],
                &positions[c1],
                corners[c0],
                corners[c1],
            );
        }
    }

    // Build triangles from the triangle table
    let row = &TRI_TABLE[cube_index];
    let mut triangles = Vec::new();
    let mut i = 0;
    while i < 16 {
        if row[i] < 0 {
            break;
        }
        let v0 = vert_list[row[i] as usize];
        let v1 = vert_list[row[i + 1] as usize];
        let v2 = vert_list[row[i + 2] as usize];

        // Compute face normal from cross product
        let edge_a = v1 - v0;
        let edge_b = v2 - v0;
        let normal = edge_a.cross(&edge_b);
        let normal = if normal.norm() > 1.0e-10 {
            normal.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        triangles.push(Triangle {
            vertices: [v0, v1, v2],
            normals: [normal, normal, normal],
            colors: [
                Vector3::new(128, 128, 128),
                Vector3::new(128, 128, 128),
                Vector3::new(128, 128, 128),
            ],
        });

        i += 3;
    }

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsdf_integrate_flat_plane() {
        // Create a TsdfVolume with small voxels
        let voxel_size = 0.02; // 2cm voxels
        let truncation = 0.06; // 6cm truncation
        let mut volume = TSDFVolume::new(voxel_size, truncation);
        // Depth values already in meters (scale = 1.0)
        volume = volume.with_depth_scale(1.0);

        // Camera at origin looking down +Z
        let extrinsics = Matrix4::identity();

        let width: usize = 64;
        let height: usize = 64;

        // Camera intrinsics (simple pinhole)
        let intrinsics =
            CameraIntrinsics::new(500.0, 500.0, 32.0, 32.0, width as u32, height as u32);

        // Synthetic depth image: flat plane at z = 1.0m
        let depth_image: Vec<f32> = vec![1.0; width * height];

        // Build a RuntimeRunner for CPU fallback
        let runner = cv_runtime::best_runner()
            .unwrap_or_else(|_| cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0)));

        volume.integrate_ctx(
            &depth_image,
            None,
            &intrinsics,
            &extrinsics,
            width,
            height,
            &runner,
        );

        // The TSDF formula in integrate_ctx is: sdf = voxel_camera.z - depth
        // So: voxels IN FRONT of the surface (z < depth) => negative TSDF
        //     voxels BEHIND the surface (z > depth) => positive TSDF
        //     voxels AT the surface (z == depth)    => TSDF near 0
        //
        // Ray marching covers [depth - trunc, depth + trunc] along each ray,
        // which for the center pixel means z in ~[0.94, 1.06].

        let mut near_surface_ok = false;
        let mut front_negative = false;
        let mut behind_positive = false;

        for (block_coords, block) in &volume.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE {
                for y in 0..VoxelBlock::BLOCK_SIZE {
                    for x in 0..VoxelBlock::BLOCK_SIZE {
                        let local = (x, y, z);
                        let idx = VoxelBlock::voxel_index(local);
                        let weight = block.weights[idx];
                        if weight <= 0.0 {
                            continue;
                        }
                        let tsdf = block.tsdf[idx];
                        let world = volume.voxel_to_world(*block_coords, local);
                        let wz = world.z;

                        // Voxels near z=1.0 should have TSDF near 0
                        if (wz - 1.0).abs() < voxel_size {
                            if tsdf.abs() < 0.3 {
                                near_surface_ok = true;
                            }
                        }
                        // Just in front of surface: sdf = z_vox - 1.0 < 0 => negative
                        if wz > 0.95 && wz < 0.99 {
                            if tsdf < 0.0 {
                                front_negative = true;
                            }
                        }
                        // Just behind surface: sdf = z_vox - 1.0 > 0 => positive
                        if wz > 1.01 && wz < 1.05 {
                            if tsdf > 0.0 {
                                behind_positive = true;
                            }
                        }
                    }
                }
            }
        }

        assert!(
            near_surface_ok,
            "Expected voxels near z=1.0 to have TSDF near 0"
        );
        assert!(
            front_negative,
            "Expected voxels in front of z=1.0 surface to have negative TSDF (sdf = z_vox - depth < 0)"
        );
        assert!(
            behind_positive,
            "Expected voxels behind z=1.0 surface to have positive TSDF (sdf = z_vox - depth > 0)"
        );
    }
}
