//! Poisson Surface Reconstruction
//!
//! Implements a simplified Poisson surface reconstruction:
//! 1. Build an adaptive octree from oriented point cloud
//! 2. Splat normals onto a regular grid to compute the vector field
//! 3. Compute divergence of the normal field
//! 4. Solve the Poisson equation using Gauss-Seidel iteration
//! 5. Extract the zero-level isosurface using marching cubes

use super::marching_cubes::{extract_isosurface, idx3};
use super::compute_bounds;
use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// Poisson Surface Reconstruction
pub fn poisson_reconstruction(
    cloud: &PointCloud,
    depth: usize,
    samples_per_node: f32,
) -> Option<TriangleMesh> {
    let normals = cloud.normals.as_ref()?;
    if normals.len() != cloud.points.len() || cloud.points.is_empty() {
        return None;
    }

    let depth = depth.min(8); // Cap depth to keep memory reasonable
    let grid_size = 1usize << depth;

    // Step 1: Compute padded bounding box
    let (bb_min, bb_max) = compute_bounds(cloud);
    let extent = bb_max - bb_min;
    let max_extent = extent.x.max(extent.y).max(extent.z);
    // Pad the bounding box by 10% on each side and make it cubic
    let padding = max_extent * 0.1;
    let center = (bb_min.coords + bb_max.coords) * 0.5;
    let half = (max_extent * 0.5) + padding;
    let origin = Point3::from(center - Vector3::new(half, half, half));
    let cube_size = half * 2.0;
    let voxel_size = cube_size / grid_size as f32;

    // Step 2: Build an adaptive octree and splat normals onto a regular grid
    // We use a regular grid for the simplified implementation. The octree is
    // used to determine which cells contain enough samples.
    let mut octree = PoissonOctree::new(origin, cube_size, depth);
    for (point, normal) in cloud.points.iter().zip(normals.iter()) {
        octree.insert(point, normal);
    }

    // Step 3: Splat the normal field onto a staggered grid
    // vx[i][j][k] stores the x-component of the vector field at face (i-1/2, j, k)
    let gs = grid_size + 1; // staggered grid is one larger
    let mut vx = vec![0.0f32; gs * grid_size * grid_size];
    let mut vy = vec![0.0f32; grid_size * gs * grid_size];
    let mut vz = vec![0.0f32; grid_size * grid_size * gs];
    let mut weight_x = vec![0.0f32; gs * grid_size * grid_size];
    let mut weight_y = vec![0.0f32; grid_size * gs * grid_size];
    let mut weight_z = vec![0.0f32; grid_size * grid_size * gs];

    let inv_voxel = 1.0 / voxel_size;

    // Minimum samples required per splat region (based on samples_per_node)
    let _min_samples = samples_per_node;

    for (point, normal) in cloud.points.iter().zip(normals.iter()) {
        // Compute continuous grid coordinates
        let gx = (point.x - origin.x) * inv_voxel;
        let gy = (point.y - origin.y) * inv_voxel;
        let gz = (point.z - origin.z) * inv_voxel;

        // Trilinear splat of the normal onto the staggered grid faces
        // X-component: staggered in x, centered at (i+0.5, j, k)
        // So the grid face x-index nearest to gx is floor(gx + 0.5)
        splat_component(
            &mut vx,
            &mut weight_x,
            gx + 0.5,
            gy,
            gz,
            normal.x,
            gs,
            grid_size,
            grid_size,
        );
        // Y-component: staggered in y
        splat_component(
            &mut vy,
            &mut weight_y,
            gx,
            gy + 0.5,
            gz,
            normal.y,
            grid_size,
            gs,
            grid_size,
        );
        // Z-component: staggered in z
        splat_component(
            &mut vz,
            &mut weight_z,
            gx,
            gy,
            gz + 0.5,
            normal.z,
            grid_size,
            grid_size,
            gs,
        );
    }

    // Normalize by weights (parallel)
    vx.par_iter_mut()
        .zip(weight_x.par_iter())
        .for_each(|(v, &w)| {
            if w > 0.0 {
                *v /= w;
            }
        });
    vy.par_iter_mut()
        .zip(weight_y.par_iter())
        .for_each(|(v, &w)| {
            if w > 0.0 {
                *v /= w;
            }
        });
    vz.par_iter_mut()
        .zip(weight_z.par_iter())
        .for_each(|(v, &w)| {
            if w > 0.0 {
                *v /= w;
            }
        });

    // Step 4: Compute divergence of the vector field on the primal grid (parallel)
    let n = grid_size;
    let mut divergence = vec![0.0f32; n * n * n];

    divergence
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let iz = row_idx / n;
            let iy = row_idx % n;
            for ix in 0..n {
                let dvx = vx[idx3(ix + 1, iy, iz, gs, n)] - vx[idx3(ix, iy, iz, gs, n)];
                let dvy = vy[idx3(ix, iy + 1, iz, n, gs)] - vy[idx3(ix, iy, iz, n, gs)];
                let dvz = vz[idx3(ix, iy, iz + 1, n, n)] - vz[idx3(ix, iy, iz, n, n)];
                row[ix] = (dvx + dvy + dvz) * inv_voxel;
            }
        });

    // Step 5: Solve Poisson equation: Laplacian(chi) = divergence
    // Using SOR (Successive Over-Relaxation) with red-black ordering
    let mut chi = vec![0.0f32; n * n * n];
    let max_iter = 100.max(n * 2);
    let h2 = voxel_size * voxel_size;
    // Optimal SOR parameter for 3D Laplacian on a cube grid
    let omega: f32 = 2.0 / (1.0 + (std::f32::consts::PI / n as f32).sin());

    for _ in 0..max_iter {
        let mut max_delta = 0.0f32;
        // Red-black sweep: two half-sweeps per iteration for better convergence
        for color in 0..2u32 {
            for iz in 0..n {
                for iy in 0..n {
                    // Start ix at the correct parity for this color
                    let start = (iz + iy + color as usize) % 2;
                    let mut ix = start;
                    while ix < n {
                        let center_idx = iz * n * n + iy * n + ix;
                        let rhs = divergence[center_idx];

                        let mut neighbor_sum = 0.0f32;
                        let mut neighbor_count = 0.0f32;

                        if ix > 0 {
                            neighbor_sum += chi[center_idx - 1];
                            neighbor_count += 1.0;
                        }
                        if ix + 1 < n {
                            neighbor_sum += chi[center_idx + 1];
                            neighbor_count += 1.0;
                        }
                        if iy > 0 {
                            neighbor_sum += chi[center_idx - n];
                            neighbor_count += 1.0;
                        }
                        if iy + 1 < n {
                            neighbor_sum += chi[center_idx + n];
                            neighbor_count += 1.0;
                        }
                        if iz > 0 {
                            neighbor_sum += chi[center_idx - n * n];
                            neighbor_count += 1.0;
                        }
                        if iz + 1 < n {
                            neighbor_sum += chi[center_idx + n * n];
                            neighbor_count += 1.0;
                        }

                        if neighbor_count > 0.0 {
                            let gs_val = (neighbor_sum - h2 * rhs) / neighbor_count;
                            let new_val = chi[center_idx] + omega * (gs_val - chi[center_idx]);
                            let delta = (new_val - chi[center_idx]).abs();
                            if delta > max_delta {
                                max_delta = delta;
                            }
                            chi[center_idx] = new_val;
                        }
                        ix += 2;
                    }
                }
            }
        }
        if max_delta < 1e-6 {
            break;
        }
    }

    // Step 6: Determine isovalue — average chi at sample positions
    let mut iso_sum = 0.0f64;
    let mut iso_count = 0u32;
    for point in &cloud.points {
        let gx = ((point.x - origin.x) * inv_voxel)
            .max(0.0)
            .min((n - 1) as f32);
        let gy = ((point.y - origin.y) * inv_voxel)
            .max(0.0)
            .min((n - 1) as f32);
        let gz = ((point.z - origin.z) * inv_voxel)
            .max(0.0)
            .min((n - 1) as f32);

        let ix = (gx as usize).min(n - 2);
        let iy = (gy as usize).min(n - 2);
        let iz = (gz as usize).min(n - 2);

        let fx = gx - ix as f32;
        let fy = gy - iy as f32;
        let fz = gz - iz as f32;

        // Trilinear interpolation of chi
        let val = trilinear_interp(&chi, ix, iy, iz, fx, fy, fz, n);
        iso_sum += val as f64;
        iso_count += 1;
    }

    let iso_value = if iso_count > 0 {
        (iso_sum / iso_count as f64) as f32
    } else {
        0.0
    };

    // Step 7: Extract isosurface using marching cubes
    let mesh = extract_isosurface(&chi, n, voxel_size, &origin, iso_value);

    if mesh.vertices.is_empty() {
        // Fallback: if marching cubes produces nothing, return an empty-but-valid mesh
        return Some(TriangleMesh::new());
    }

    Some(mesh)
}

/// Trilinear interpolation in a 3D grid
fn trilinear_interp(
    grid: &[f32],
    ix: usize,
    iy: usize,
    iz: usize,
    fx: f32,
    fy: f32,
    fz: f32,
    n: usize,
) -> f32 {
    let c000 = grid[idx3(ix, iy, iz, n, n)];
    let c100 = grid[idx3(ix + 1, iy, iz, n, n)];
    let c010 = grid[idx3(ix, iy + 1, iz, n, n)];
    let c110 = grid[idx3(ix + 1, iy + 1, iz, n, n)];
    let c001 = grid[idx3(ix, iy, iz + 1, n, n)];
    let c101 = grid[idx3(ix + 1, iy, iz + 1, n, n)];
    let c011 = grid[idx3(ix, iy + 1, iz + 1, n, n)];
    let c111 = grid[idx3(ix + 1, iy + 1, iz + 1, n, n)];

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

/// Splat a scalar value onto a 3D grid using trilinear weights
#[allow(clippy::too_many_arguments)]
fn splat_component(
    grid: &mut [f32],
    weight: &mut [f32],
    gx: f32,
    gy: f32,
    gz: f32,
    value: f32,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    let ix = gx.floor() as i32;
    let iy = gy.floor() as i32;
    let iz = gz.floor() as i32;
    let fx = gx - ix as f32;
    let fy = gy - iy as f32;
    let fz = gz - iz as f32;

    for dz in 0..2i32 {
        for dy in 0..2i32 {
            for dx in 0..2i32 {
                let cx = ix + dx;
                let cy = iy + dy;
                let cz = iz + dz;

                if cx < 0
                    || cx >= nx as i32
                    || cy < 0
                    || cy >= ny as i32
                    || cz < 0
                    || cz >= nz as i32
                {
                    continue;
                }

                let w = (if dx == 0 { 1.0 - fx } else { fx })
                    * (if dy == 0 { 1.0 - fy } else { fy })
                    * (if dz == 0 { 1.0 - fz } else { fz });

                let idx = (cz as usize) * ny * nx + (cy as usize) * nx + cx as usize;
                grid[idx] += value * w;
                weight[idx] += w;
            }
        }
    }
}

/// Adaptive octree for Poisson reconstruction
struct PoissonOctree {
    origin: Point3<f32>,
    size: f32,
    max_depth: usize,
    root: PoissonOctreeNode,
}

struct PoissonOctreeNode {
    children: Option<Box<[PoissonOctreeNode; 8]>>,
    normal_sum: Vector3<f32>,
    point_count: u32,
}

impl PoissonOctree {
    fn new(origin: Point3<f32>, size: f32, max_depth: usize) -> Self {
        Self {
            origin,
            size,
            max_depth,
            root: PoissonOctreeNode::new(),
        }
    }

    fn insert(&mut self, point: &Point3<f32>, normal: &Vector3<f32>) {
        let rel = point - self.origin;
        if rel.x < 0.0
            || rel.y < 0.0
            || rel.z < 0.0
            || rel.x > self.size
            || rel.y > self.size
            || rel.z > self.size
        {
            return; // Outside bounds
        }
        self.root
            .insert(rel.x, rel.y, rel.z, self.size, normal, 0, self.max_depth);
    }
}

impl PoissonOctreeNode {
    fn new() -> Self {
        Self {
            children: None,
            normal_sum: Vector3::zeros(),
            point_count: 0,
        }
    }

    fn insert(
        &mut self,
        rx: f32,
        ry: f32,
        rz: f32,
        size: f32,
        normal: &Vector3<f32>,
        current_depth: usize,
        max_depth: usize,
    ) {
        self.normal_sum += normal;
        self.point_count += 1;

        if current_depth >= max_depth {
            return;
        }

        let half = size * 0.5;
        let child_idx = ((if rx >= half { 1 } else { 0 })
            | (if ry >= half { 2 } else { 0 })
            | (if rz >= half { 4 } else { 0 })) as usize;

        if self.children.is_none() {
            self.children = Some(Box::new(core::array::from_fn(|_| PoissonOctreeNode::new())));
        }

        let children = self.children.as_mut().unwrap();
        children[child_idx].insert(
            rx - if rx >= half { half } else { 0.0 },
            ry - if ry >= half { half } else { 0.0 },
            rz - if rz >= half { half } else { 0.0 },
            half,
            normal,
            current_depth + 1,
            max_depth,
        );
    }
}
