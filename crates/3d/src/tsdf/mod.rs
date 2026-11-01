//! TSDF (Truncated Signed Distance Function) Integration
//!
//! Core algorithm for fusing multiple RGBD images into a 3D volume.
//! Based on "KinectFusion: Real-Time Dense Surface Mapping and Tracking" by Newcombe et al.

use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use cv_core::CameraIntrinsicsF32;
use cv_runtime::orchestrator::RuntimeRunner;

/// Re-export the canonical f32 camera intrinsics from cv-core.
pub type CameraIntrinsics = CameraIntrinsicsF32;

/// Depths beyond this (metres, after `depth_scale`) are treated as invalid.
const MAX_INTEGRATION_DEPTH: f32 = 10.0;

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
            // Positive is free space. The value of an unobserved voxel is never
            // meshed (its weight is zero), but +1 is the honest placeholder.
            tsdf: vec![1.0; Self::VOXELS_PER_BLOCK],
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

    /// Fold one observation into voxel `idx` as a running weighted mean.
    fn fuse(&mut self, idx: usize, tsdf: f32, color: Vector3<u8>, max_weight: f32) {
        let w = self.weights[idx];
        self.tsdf[idx] = (self.tsdf[idx] * w + tsdf) / (w + 1.0);
        let old = self.colors[idx];
        let mix = |o: u8, c: u8| ((o as f32 * w + c as f32) / (w + 1.0)) as u8;
        self.colors[idx] = Vector3::new(
            mix(old.x, color.x),
            mix(old.y, color.y),
            mix(old.z, color.z),
        );
        self.weights[idx] = (w + 1.0).min(max_weight);
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
    ) -> cv_core::Result<()> {
        // The runner failure used to be swallowed by `if let Ok(runner)`, so a
        // machine where even the CPU registry could not initialize got a silent
        // no-op: `integrate` returned as though the frame had been merged, and
        // the volume simply never changed. Dropping a frame without saying so is
        // indistinguishable from a frame that contributed nothing.
        let (runner, _) = cv_runtime::best_runner().map_err(|e| {
            cv_core::Error::AlgorithmError(format!(
                "no compute runner is available, so the frame was not integrated: {e}"
            ))
        })?;
        self.integrate_ctx(
            depth_image,
            color_image,
            intrinsics,
            extrinsics,
            width,
            height,
            &runner,
        )
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
    ) -> cv_core::Result<()> {
        // NOTE: A previous revision dispatched GPU integration into a freshly
        // allocated 256^3 dense volume and returned early. Without a
        // read-back/merge step into the sparse block storage, every frame was
        // integrated into a throwaway buffer and silently dropped, leaving
        // extract_mesh()/extract_point_cloud() empty on GPU machines. Until
        // merge-back exists, always use the CPU path below so data is actually
        // integrated into self.blocks.

        // CPU Fallback (Rayon)
        // Compute inverse extrinsics for projective distance
        // A singular `extrinsics` has no camera frame, and the identity is not a
        // neutral substitute: `extrinsics_inv` is used below to bring each world
        // voxel into camera space, so the identity means "every world point is
        // already a camera point". The projection distances - and therefore every
        // TSDF value in the volume - would be computed from the wrong position,
        // silently, with the call reporting success.
        //
        // `extrinsics` is caller-supplied (`Matrix4<f32>`, with no validation up
        // the stack), so a degenerate rotation is a caller bug that must be
        // reported rather than absorbed.
        let extrinsics_inv = extrinsics.try_inverse().ok_or_else(|| {
            cv_core::Error::InvalidInput(
                "extrinsics is singular, so the camera frame is undefined and the \
                 projective distances cannot be computed; the identity would \
                 silently integrate every frame at the wrong pose"
                    .to_string(),
            )
        })?;

        // Validate the frame up front, at the boundary where caller data enters.
        //
        // The parallel loop below allocates a `Vec`, not a `Result`, so a length
        // problem discovered inside it can only be swallowed - which is why this
        // function used to `continue` past a short buffer. That was the best
        // available option while `integrate_ctx` returned `()`, and the comment
        // here said so; it returns `Result` now, so the honest answer is
        // reachable and the workaround is unnecessary.
        //
        // A partial frame integrated without a word is the failure mode this
        // prevents: the caller sees a volume that looks successfully updated from
        // whatever part of the frame happened to fit.
        let expected = width.saturating_mul(height);
        if depth_image.len() < expected {
            return Err(cv_core::Error::InvalidInput(format!(
                "depth_image has {} samples but the frame is declared {width}x{height} \
                 ({expected}); a short buffer would otherwise integrate a partial \
                 frame and report success",
                depth_image.len()
            )));
        }
        if let Some(colors) = color_image {
            if colors.len() < expected {
                return Err(cv_core::Error::InvalidInput(format!(
                    "color_image has {} samples but the frame is declared {width}x{height} \
                     ({expected}); the missing samples would be replaced with a \
                     fabricated mid-grey",
                    colors.len()
                )));
            }
        }

        // A zero focal length makes the backprojection below divide by zero, so
        // every ray direction becomes NaN and the TSDF values are silently
        // corrupted. The fields are public, so this is constructible.
        if intrinsics.fx == 0.0 || intrinsics.fy == 0.0 {
            return Err(cv_core::Error::InvalidInput(format!(
                "intrinsics has a zero focal length (fx = {}, fy = {}); the \
                 backprojection would divide by zero and every TSDF value would be \
                 NaN",
                intrinsics.fx, intrinsics.fy
            )));
        }

        // KinectFusion projective TSDF, integrated per voxel.
        //
        // Convention: **positive is free space**. `sdf = depth - z_voxel`, where
        // `z_voxel` is the voxel centre's camera-frame z, truncated to
        // `[-trunc, +trunc]` and normalised to `[-1, 1]`. Voxels more than
        // `trunc` behind the measured surface are occluded, i.e. unobserved,
        // and are not touched at all, so their weight stays zero.
        //
        // Three defects lived here:
        //
        // - The sign was `z_voxel - depth`, negative in observed free space,
        //   while untouched voxels start at `+1` ("empty"). Where the measured
        //   band met never-observed voxels there was a sign change, and marching
        //   cubes put a phantom surface there.
        // - The band was marched `depth +/- trunc` along the *unit* ray, but
        //   `depth` is a z value: off axis the surface sits at range
        //   `depth / cos(theta)`, so the band stopped short of it.
        // - Each ray sample was written into whichever voxel it fell in, so the
        //   stored value was the distance of some point inside the voxel rather
        //   than of the voxel centre the mesh is built on.
        //
        // Integration is now two passes: find the blocks the truncation band
        // reaches (walking each pixel's ray in *z*, so off-axis rays reach
        // `z = depth`), then project every voxel centre of those blocks into the
        // image and update it from the pixel it lands on.
        let bs = VoxelBlock::BLOCK_SIZE as i32;
        let vs = self.voxel_size;
        let trunc = self.truncation_distance;
        let depth_scale = self.depth_scale;
        let max_weight = self.max_weight;
        let (fx, fy, cx, cy) = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);
        // Written to reject NaN as well as out-of-range depths.
        let valid_depth = |d: f32| d > 0.0 && d <= MAX_INTEGRATION_DEPTH;

        let touched: HashSet<(i32, i32, i32)> = group.run(|| {
            (0..height)
                .into_par_iter()
                .fold(HashSet::new, |mut set, v| {
                    for u in 0..width {
                        // Lengths were validated above, before the parallel
                        // region, so every index is in range.
                        let depth = depth_image[v * width + u] / depth_scale;
                        if !valid_depth(depth) {
                            continue;
                        }
                        // Camera-frame ray scaled so its z component is 1: the
                        // point at camera depth z is `(rx * z, ry * z, z)`.
                        let rx = (u as f32 - cx) / fx;
                        let ry = (v as f32 - cy) / fy;
                        let z0 = (depth - trunc).max(0.01);
                        let z1 = depth + trunc;
                        // Samples at most half a voxel apart in 3D.
                        let len = (rx * rx + ry * ry + 1.0).sqrt();
                        let steps = ((z1 - z0) * len / (0.5 * vs)).ceil().max(1.0) as usize;
                        for i in 0..=steps {
                            let z = z0 + (z1 - z0) * i as f32 / steps as f32;
                            let p = extrinsics.transform_point(&Point3::new(rx * z, ry * z, z));
                            set.insert((
                                ((p.x / vs).floor() as i32).div_euclid(bs),
                                ((p.y / vs).floor() as i32).div_euclid(bs),
                                ((p.z / vs).floor() as i32).div_euclid(bs),
                            ));
                        }
                    }
                    set
                })
                .reduce(HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                })
        });

        for &c in &touched {
            self.blocks.entry(c).or_insert_with(|| VoxelBlock::new(c));
        }
        let mut targets: Vec<&mut VoxelBlock> = self
            .blocks
            .iter_mut()
            .filter(|(c, _)| touched.contains(c))
            .map(|(_, b)| b)
            .collect();

        group.run(|| {
            targets.par_iter_mut().for_each(|block| {
                let n = VoxelBlock::BLOCK_SIZE;
                for lz in 0..n {
                    for ly in 0..n {
                        for lx in 0..n {
                            let centre = Point3::new(
                                ((block.coords.0 * bs + lx as i32) as f32 + 0.5) * vs,
                                ((block.coords.1 * bs + ly as i32) as f32 + 0.5) * vs,
                                ((block.coords.2 * bs + lz as i32) as f32 + 0.5) * vs,
                            );
                            let pc = extrinsics_inv.transform_point(&centre);
                            if pc.z <= 0.0 {
                                continue;
                            }
                            // Pixel `u` backprojects through its integer
                            // coordinate (see the ray walk above), so the pixel
                            // a voxel belongs to is its rounded projection.
                            let u = (fx * pc.x / pc.z + cx).round();
                            let v = (fy * pc.y / pc.z + cy).round();
                            if !(u >= 0.0 && v >= 0.0 && u < width as f32 && v < height as f32) {
                                continue;
                            }
                            let idx = v as usize * width + u as usize;
                            let depth = depth_image[idx] / depth_scale;
                            if !valid_depth(depth) {
                                continue;
                            }
                            let sdf = depth - pc.z;
                            if sdf < -trunc {
                                continue;
                            }
                            let tsdf = (sdf / trunc).min(1.0);
                            // Length-checked above, so a supplied image always
                            // has this sample; grey only when there is no colour
                            // image at all.
                            let color = color_image
                                .and_then(|c| c.get(idx).copied())
                                .unwrap_or(Vector3::new(128, 128, 128));
                            block.fuse(
                                VoxelBlock::voxel_index((lx, ly, lz)),
                                tsdf,
                                color,
                                max_weight,
                            );
                        }
                    }
                }
            })
        });

        Ok(())
    }

    /// Extract surface mesh using Marching Cubes.
    ///
    /// A cell is meshed only when all eight of its corners have been observed
    /// (non-zero weight): an unobserved voxel's value is a placeholder, and a
    /// sign change against it is not a surface.
    ///
    /// Each cell is owned by the block holding its lowest corner, and corners
    /// past that block's upper faces are read from the neighbouring blocks, so
    /// cells straddling a block seam are meshed. They used to be skipped -
    /// cells ran `0..BLOCK_SIZE - 1` within each block - which left a
    /// one-voxel crack every eight voxels.
    pub fn extract_mesh(&self) -> Vec<Triangle> {
        const CORNERS: [(usize, usize, usize); 8] = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ];
        let n = VoxelBlock::BLOCK_SIZE;
        let mut triangles = Vec::new();

        for (&(bx, by, bz), block) in &self.blocks {
            // This block and its +x / +y / +z neighbours, indexed by
            // `dx | dy << 1 | dz << 2`.
            let neighbours: [Option<&VoxelBlock>; 8] = core::array::from_fn(|i| {
                if i == 0 {
                    Some(block)
                } else {
                    let (dx, dy, dz) = ((i & 1) as i32, ((i >> 1) & 1) as i32, (i >> 2) as i32);
                    self.blocks.get(&(bx + dx, by + dy, bz + dz))
                }
            });
            // An observed voxel at block-relative coordinates in `0..=n`.
            let sample = |x: usize, y: usize, z: usize| -> Option<(f32, Vector3<u8>)> {
                let b = neighbours[(x / n) | ((y / n) << 1) | ((z / n) << 2)]?;
                let idx = VoxelBlock::voxel_index((x % n, y % n, z % n));
                (b.weights[idx] > 0.0).then(|| (b.tsdf[idx], b.colors[idx]))
            };

            for z in 0..n {
                for y in 0..n {
                    'cell: for x in 0..n {
                        let mut corners = [0.0f32; 8];
                        // Corner colours, in the same order as `corners`, so a
                        // triangle vertex can be coloured by where it lies
                        // between the two corners its edge joins.
                        let mut corner_colors = [Vector3::new(0u8, 0, 0); 8];
                        for (c, &(dx, dy, dz)) in CORNERS.iter().enumerate() {
                            match sample(x + dx, y + dy, z + dz) {
                                Some((t, col)) => {
                                    corners[c] = t;
                                    corner_colors[c] = col;
                                }
                                None => continue 'cell,
                            }
                        }

                        triangles.extend(marching_cubes_cell(
                            &(bx, by, bz),
                            (x, y, z),
                            &corners,
                            &corner_colors,
                            self.voxel_size,
                        ));
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

        // Voxel `i` spans `[i, i + 1) * voxel_size`; its value is the distance
        // at its centre.
        Point3::new(
            (vx as f32 + 0.5) * self.voxel_size,
            (vy as f32 + 0.5) * self.voxel_size,
            (vz as f32 + 0.5) * self.voxel_size,
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

        // A locally flat region has a zero gradient; guard against dividing by
        // zero so the normal stays finite (zeros) instead of becoming NaN.
        Vector3::new(dx, dy, dz)
            .try_normalize(1e-8)
            .unwrap_or(Vector3::zeros())
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
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
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
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
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
/// Where `iso_level` falls between two corner values, as a fraction in [0, 1].
///
/// The same fraction the position interpolation uses, so a vertex's colour sits
/// where the vertex sits rather than at whichever corner happens to come first.
fn interpolation_fraction(iso_level: f32, val1: f32, val2: f32) -> f32 {
    // Exactly `mu` from `vertex_interp`: the weight on the second corner, so a
    // vertex's colour is read at the same place along the edge as the vertex.
    // The former `(iso - val2) / (val1 - val2)` was `1 - mu`, the complementary
    // fraction, and coloured every vertex from the opposite end of its edge.
    let denom = val2 - val1;
    if denom.abs() < 1.0e-12 {
        0.0
    } else {
        ((iso_level - val1) / denom).clamp(0.0, 1.0)
    }
}

/// Linear blend of two u8 channel values, rounded and clamped.
fn blend_u8(a: u8, b: u8, t: f32) -> u8 {
    let v = a as f32 + (b as f32 - a as f32) * t;
    v.round().clamp(0.0, 255.0) as u8
}

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
    corner_colors: &[Vector3<u8>; 8],
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
    // Corners are voxel centres, `(i + 0.5) * voxel_size`, where the stored
    // distances were evaluated.
    let centre = |b: i32, l: usize| {
        ((b * VoxelBlock::BLOCK_SIZE as i32 + l as i32) as f32 + 0.5) * voxel_size
    };
    let base_x = centre(block_coords.0, lx);
    let base_y = centre(block_coords.1, ly);
    let base_z = centre(block_coords.2, lz);

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
    // Colour at each edge's vertex, interpolated between the two corners that
    // edge joins, exactly as the position is.
    //
    // This was hardcoded to `(128, 128, 128)` for every triangle, so
    // `Triangle::colors` was a valid-length field carrying no information - and
    // the per-voxel colours `update_voxel` stores were read nowhere. Measured:
    // 563,394 of 563,394 output colours were the default.
    let mut color_list: [Vector3<u8>; 12] = [Vector3::new(128, 128, 128); 12];

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
            let f = interpolation_fraction(iso_level, corners[c0], corners[c1]);
            color_list[edge] = Vector3::new(
                blend_u8(corner_colors[c0].x, corner_colors[c1].x, f),
                blend_u8(corner_colors[c0].y, corner_colors[c1].y, f),
                blend_u8(corner_colors[c0].z, corner_colors[c1].z, f),
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
        // Bourke's table winds each triangle so its normal faces the corners
        // *below* the iso level. Positive is free space here, so the edges are
        // taken in reverse to make the face normal point out of the surface,
        // toward the camera that observed it.
        let e = [row[i] as usize, row[i + 2] as usize, row[i + 1] as usize];
        let (v0, v1, v2) = (vert_list[e[0]], vert_list[e[1]], vert_list[e[2]]);

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
            colors: [color_list[e[0]], color_list[e[1]], color_list[e[2]]],
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
            .map(|(r, _)| r)
            .unwrap_or_else(|_| cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0)));

        volume
            .integrate_ctx(
                &depth_image,
                None,
                &intrinsics,
                &extrinsics,
                width,
                height,
                &runner,
            )
            .expect("a well-formed frame must integrate");

        // KinectFusion convention: sdf = depth - z_voxel, positive in front of
        // the surface (free space), negative behind it, evaluated at the voxel
        // centre and truncated. With the camera at the identity, a voxel's
        // camera z is its world z, so every observed voxel's value is known
        // exactly. Voxels more than `trunc` behind the plane are occluded and
        // must never have been touched.
        //
        // This test used to assert the opposite sign (negative in front), which
        // is what put a phantom surface wherever the observed band met
        // never-observed voxels initialised to +1.
        let mut observed = 0;
        let (mut front, mut behind) = (0, 0);
        for (block_coords, block) in &volume.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE {
                for y in 0..VoxelBlock::BLOCK_SIZE {
                    for x in 0..VoxelBlock::BLOCK_SIZE {
                        let local = (x, y, z);
                        let idx = VoxelBlock::voxel_index(local);
                        if block.weights[idx] <= 0.0 {
                            continue;
                        }
                        observed += 1;
                        let wz = volume.voxel_to_world(*block_coords, local).z;
                        assert!(
                            1.0 - wz >= -truncation - 1e-5,
                            "voxel centred at z = {wz} is more than the truncation \
                             behind the surface and must be unobserved"
                        );
                        let expected = ((1.0 - wz) / truncation).min(1.0);
                        let tsdf = block.tsdf[idx];
                        assert!(
                            (tsdf - expected).abs() < 1e-4,
                            "voxel centred at z = {wz}: tsdf {tsdf}, expected {expected}"
                        );
                        if wz < 1.0 {
                            front += 1;
                            assert!(tsdf > 0.0, "free space in front must be positive");
                        } else {
                            behind += 1;
                            assert!(tsdf < 0.0, "space behind must be negative");
                        }
                    }
                }
            }
        }
        assert!(observed > 0 && front > 0 && behind > 0);
    }

    #[test]
    fn test_estimate_normal_flat_region_is_finite() {
        let mut volume = TSDFVolume::new(0.02, 0.06);
        // A uniform block has a zero gradient everywhere.
        volume.blocks.insert((0, 0, 0), VoxelBlock::new((0, 0, 0)));

        let normal = volume.estimate_normal((0, 0, 0), (4, 4, 4));
        assert_eq!(normal, Vector3::zeros());
        assert!(normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite());
    }

    /// `MC_EDGE_TABLE` must agree with the sign rule that defines it.
    ///
    /// Edge `e` of a cube is crossed exactly when its two corner signs differ,
    /// so the whole 256-entry table is derivable rather than transcribed. It is
    /// transcribed - and **two entries were wrong**: indices 33 and 222 both
    /// held `0x139` where the rule gives `0x339`. Bit 9 is the crossing on edge 9
    /// (corners 1 and 5), and `TRI_TABLE[33]` references edge 9.
    ///
    /// `marching_cubes_cell` only computes `vert_list[e]` when bit `e` is set, so
    /// `vert_list[9]` kept its initialiser `Point3::origin()` and that literal
    /// world origin was emitted as a mesh vertex. Measured on a tilted plane:
    /// 167 of 187,798 triangles contained a vertex at exactly (0, 0, 0), and the
    /// first corrupted triangle spanned 1.88 world units - 188 voxels - from a
    /// one-voxel cell.
    ///
    /// Deriving the table here means a transcription error cannot survive.
    #[test]
    fn edge_table_matches_the_sign_rule() {
        // Bourke's corner and edge numbering, as used by the tables above.
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        for cfg in 0..256usize {
            let mut expected: i32 = 0;
            for (e, (a, b)) in EDGES.iter().enumerate() {
                if ((cfg >> a) & 1) != ((cfg >> b) & 1) {
                    expected |= 1 << e;
                }
            }
            assert_eq!(
                MC_EDGE_TABLE[cfg], expected,
                "MC_EDGE_TABLE[{cfg}]: the sign rule gives {expected:#06x}, the \
                 table has {:#06x}",
                MC_EDGE_TABLE[cfg]
            );
        }
    }

    /// The colour fraction must be the **same** fraction the position uses.
    ///
    /// `vertex_interp` places the vertex at `mu = (iso - val1) / (val2 - val1)`.
    /// `interpolation_fraction` computed `(iso - val2) / (val1 - val2)`, which is
    /// exactly `1 - mu` - the complementary fraction. The per-edge colour was
    /// therefore taken from the opposite end of the edge: a vertex sitting a
    /// sixth of the way from `c0` was coloured as if it sat five sixths of the
    /// way, i.e. with `c1`'s colour (and vice versa).
    #[test]
    fn marching_cubes_colour_fraction_matches_the_vertex_position() {
        // Asymmetric crossing, so the two fractions differ by far more than
        // rounding: iso = 0 between val1 = -0.2 and val2 = +1.
        let (val1, val2) = (-0.2f32, 1.0f32);
        let mu = (0.0 - val1) / (val2 - val1); // 1/6, the position fraction
        let f = interpolation_fraction(0.0, val1, val2);
        assert!(
            (f - mu).abs() < 1e-6,
            "colour fraction {f} is not the position fraction {mu}: it is its \
             complement, so the colour is read from the opposite end of the edge"
        );

        // End to end: one cube corner below the isosurface, so the cell emits a
        // single triangle whose three vertices all lie close to that corner.
        // That corner is red and every other corner is blue, so a correctly
        // placed colour is red-dominant.
        let mut corners = [1.0f32; 8];
        corners[0] = -0.2;
        let red = Vector3::new(255u8, 0, 0);
        let blue = Vector3::new(0u8, 0, 255);
        let mut corner_colors = [blue; 8];
        corner_colors[0] = red;

        let tris = marching_cubes_cell(&(0, 0, 0), (0, 0, 0), &corners, &corner_colors, 1.0);
        assert_eq!(tris.len(), 1, "configuration 1 emits exactly one triangle");
        for c in tris[0].colors {
            assert!(
                c.x > c.z,
                "vertex colour {c:?} is blue-dominant at a vertex only {mu:.3} \
                 of the way from the red corner: the blend used the complementary \
                 fraction instead of the one placing the vertex"
            );
        }
    }
}
