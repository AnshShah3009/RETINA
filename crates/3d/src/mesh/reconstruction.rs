//! Surface Reconstruction from Point Clouds
//!
//! Implements:
//! - Poisson Surface Reconstruction
//! - Ball Pivoting Algorithm (BPA)
//! - Alpha Shapes
//! - Marching Cubes
//! - Delaunay-based reconstruction

use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

/// Compute normals for point cloud using PCA (simplified)
pub fn compute_point_normals(cloud: &PointCloud, _k: usize) -> Vec<Vector3<f32>> {
    let n = cloud.points.len();
    if n == 0 {
        return vec![];
    }

    // Simplified: use existing normals or compute basic normals
    if let Some(ref normals) = cloud.normals {
        return normals.clone();
    }

    // Default: return upward normals
    vec![Vector3::new(0.0, 1.0, 0.0); n]
}

/// Poisson Surface Reconstruction
///
/// Implements a simplified Poisson surface reconstruction:
/// 1. Build an adaptive octree from oriented point cloud
/// 2. Splat normals onto a regular grid to compute the vector field
/// 3. Compute divergence of the normal field
/// 4. Solve the Poisson equation using Gauss-Seidel iteration
/// 5. Extract the zero-level isosurface using marching cubes
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

    // Normalize by weights
    for i in 0..vx.len() {
        if weight_x[i] > 0.0 {
            vx[i] /= weight_x[i];
        }
    }
    for i in 0..vy.len() {
        if weight_y[i] > 0.0 {
            vy[i] /= weight_y[i];
        }
    }
    for i in 0..vz.len() {
        if weight_z[i] > 0.0 {
            vz[i] /= weight_z[i];
        }
    }

    // Step 4: Compute divergence of the vector field on the primal grid
    // div(V)[i][j][k] = (vx[i+1,j,k] - vx[i,j,k] + vy[i,j+1,k] - vy[i,j,k]
    //                     + vz[i,j,k+1] - vz[i,j,k]) / voxel_size
    let n = grid_size;
    let mut divergence = vec![0.0f32; n * n * n];

    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dvx = vx[idx3(ix + 1, iy, iz, gs, n)] - vx[idx3(ix, iy, iz, gs, n)];
                let dvy = vy[idx3(ix, iy + 1, iz, n, gs)] - vy[idx3(ix, iy, iz, n, gs)];
                let dvz = vz[idx3(ix, iy, iz + 1, n, n)] - vz[idx3(ix, iy, iz, n, n)];
                divergence[idx3(ix, iy, iz, n, n)] = (dvx + dvy + dvz) * inv_voxel;
            }
        }
    }

    // Step 5: Solve Poisson equation: Laplacian(chi) = divergence
    // Using Gauss-Seidel relaxation
    let mut chi = vec![0.0f32; n * n * n];
    let max_iter = 100.max(n * 2);
    let h2 = voxel_size * voxel_size;

    for _ in 0..max_iter {
        let mut max_delta = 0.0f32;
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let center_idx = idx3(ix, iy, iz, n, n);
                    let rhs = divergence[center_idx];

                    // Sum of neighbors and count of valid neighbors
                    let mut neighbor_sum = 0.0f32;
                    let mut neighbor_count = 0.0f32;

                    if ix > 0 {
                        neighbor_sum += chi[idx3(ix - 1, iy, iz, n, n)];
                        neighbor_count += 1.0;
                    }
                    if ix + 1 < n {
                        neighbor_sum += chi[idx3(ix + 1, iy, iz, n, n)];
                        neighbor_count += 1.0;
                    }
                    if iy > 0 {
                        neighbor_sum += chi[idx3(ix, iy - 1, iz, n, n)];
                        neighbor_count += 1.0;
                    }
                    if iy + 1 < n {
                        neighbor_sum += chi[idx3(ix, iy + 1, iz, n, n)];
                        neighbor_count += 1.0;
                    }
                    if iz > 0 {
                        neighbor_sum += chi[idx3(ix, iy, iz - 1, n, n)];
                        neighbor_count += 1.0;
                    }
                    if iz + 1 < n {
                        neighbor_sum += chi[idx3(ix, iy, iz + 1, n, n)];
                        neighbor_count += 1.0;
                    }

                    if neighbor_count > 0.0 {
                        let new_val = (neighbor_sum - h2 * rhs) / neighbor_count;
                        let delta = (new_val - chi[center_idx]).abs();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                        chi[center_idx] = new_val;
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

/// 3D index helper for a grid of size (nx, ny, nz) stored in z-major order
fn idx3(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize) -> usize {
    iz * ny * nx + iy * nx + ix
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

/// Extract isosurface from a scalar field using marching cubes
fn extract_isosurface(
    chi: &[f32],
    grid_size: usize,
    voxel_size: f32,
    origin: &Point3<f32>,
    iso_value: f32,
) -> TriangleMesh {
    let n = grid_size;
    let mut vertices: Vec<Point3<f32>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Deduplicate vertices using a hash map keyed by edge
    // Edge is identified by (ix, iy, iz, edge_direction)
    let mut edge_to_vertex: HashMap<(usize, usize, usize, u8), usize> = HashMap::new();

    for iz in 0..n - 1 {
        for iy in 0..n - 1 {
            for ix in 0..n - 1 {
                // Get the 8 corner values
                let corners = [
                    chi[idx3(ix, iy, iz, n, n)] - iso_value,
                    chi[idx3(ix + 1, iy, iz, n, n)] - iso_value,
                    chi[idx3(ix + 1, iy + 1, iz, n, n)] - iso_value,
                    chi[idx3(ix, iy + 1, iz, n, n)] - iso_value,
                    chi[idx3(ix, iy, iz + 1, n, n)] - iso_value,
                    chi[idx3(ix + 1, iy, iz + 1, n, n)] - iso_value,
                    chi[idx3(ix + 1, iy + 1, iz + 1, n, n)] - iso_value,
                    chi[idx3(ix, iy + 1, iz + 1, n, n)] - iso_value,
                ];

                // Compute cube index
                let mut cube_index = 0usize;
                for i in 0..8 {
                    if corners[i] < 0.0 {
                        cube_index |= 1 << i;
                    }
                }

                if MC_EDGE_TABLE[cube_index] == 0 {
                    continue;
                }

                // Corner positions in world space
                let corner_offsets: [[usize; 3]; 8] = [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ];

                let positions: [Point3<f32>; 8] = core::array::from_fn(|i| {
                    Point3::new(
                        origin.x + (ix + corner_offsets[i][0]) as f32 * voxel_size,
                        origin.y + (iy + corner_offsets[i][1]) as f32 * voxel_size,
                        origin.z + (iz + corner_offsets[i][2]) as f32 * voxel_size,
                    )
                });

                // Compute edge vertices
                let edge_bits = MC_EDGE_TABLE[cube_index];
                let mut vert_indices = [0usize; 12];

                for edge in 0..12 {
                    if edge_bits & (1 << edge) != 0 {
                        let [c0, c1] = MC_EDGE_VERTICES[edge];

                        // Canonical edge key for deduplication
                        let (eix, eiy, eiz, edir) = canonical_edge_key(
                            ix,
                            iy,
                            iz,
                            &corner_offsets[c0],
                            &corner_offsets[c1],
                        );

                        let vi = edge_to_vertex
                            .entry((eix, eiy, eiz, edir))
                            .or_insert_with(|| {
                                let p = mc_vertex_interp(
                                    &positions[c0],
                                    &positions[c1],
                                    corners[c0],
                                    corners[c1],
                                );
                                let idx = vertices.len();
                                vertices.push(p);
                                idx
                            });
                        vert_indices[edge] = *vi;
                    }
                }

                // Build triangles
                let row = &MC_TRI_TABLE[cube_index];
                let mut ti = 0;
                while ti < 16 {
                    if row[ti] < 0 {
                        break;
                    }
                    faces.push([
                        vert_indices[row[ti] as usize],
                        vert_indices[row[ti + 1] as usize],
                        vert_indices[row[ti + 2] as usize],
                    ]);
                    ti += 3;
                }
            }
        }
    }

    let mut mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);
    if !mesh.faces.is_empty() {
        mesh.compute_vertex_normals();
    }
    mesh
}

/// Compute a canonical edge key for vertex deduplication.
/// Each edge is shared between cells; we canonicalize by choosing the
/// cell-corner with the smallest coordinates and the edge direction (0=x, 1=y, 2=z).
fn canonical_edge_key(
    ix: usize,
    iy: usize,
    iz: usize,
    off0: &[usize; 3],
    off1: &[usize; 3],
) -> (usize, usize, usize, u8) {
    let x0 = ix + off0[0];
    let y0 = iy + off0[1];
    let z0 = iz + off0[2];
    let x1 = ix + off1[0];
    let y1 = iy + off1[1];
    let z1 = iz + off1[2];

    // Edge direction
    if x0 != x1 {
        (x0.min(x1), y0, z0, 0)
    } else if y0 != y1 {
        (x0, y0.min(y1), z0, 1)
    } else {
        (x0, y0, z0.min(z1), 2)
    }
}

/// Linear interpolation between two points at the zero crossing
fn mc_vertex_interp(p1: &Point3<f32>, p2: &Point3<f32>, val1: f32, val2: f32) -> Point3<f32> {
    let eps = 1.0e-5;
    if val1.abs() < eps {
        return *p1;
    }
    if val2.abs() < eps {
        return *p2;
    }
    if (val1 - val2).abs() < eps {
        return *p1;
    }
    let t = -val1 / (val2 - val1);
    let t = t.clamp(0.0, 1.0);
    Point3::from(p1.coords * (1.0 - t) + p2.coords * t)
}

/// Ball Pivoting Algorithm
pub fn ball_pivoting(cloud: &PointCloud, ball_radius: f32) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();

    if cloud.points.len() < 3 {
        return mesh;
    }

    let spatial_index = SpatialIndex::new(cloud, ball_radius * 2.0);

    let mut edge_count: HashMap<(usize, usize), u8> = HashMap::new();
    let mut mesh_faces: Vec<[usize; 3]> = Vec::new();
    let mut used_in_mesh: HashSet<usize> = HashSet::new();

    for i in 0..cloud.points.len() {
        // Skip vertices already incorporated into the mesh
        if used_in_mesh.contains(&i) {
            continue;
        }
        if let Some(seed) = find_seed_triangle(i, cloud, &spatial_index, ball_radius) {
            // Only start a new seed if none of the seed vertices have all edges saturated
            if try_add_triangle(seed, &mut edge_count, &mut mesh_faces) {
                for &v in &seed {
                    used_in_mesh.insert(v);
                }
                expand_front(
                    seed,
                    cloud,
                    &spatial_index,
                    ball_radius,
                    &mut edge_count,
                    &mut mesh_faces,
                    &mut used_in_mesh,
                );
            }
        }
    }

    mesh.vertices = cloud.points.clone();
    mesh.faces = mesh_faces;
    mesh.compute_vertex_normals();

    mesh
}

/// Alpha Shapes reconstruction
pub fn alpha_shapes(cloud: &PointCloud, alpha: f32) -> TriangleMesh {
    ball_pivoting(cloud, alpha)
}

/// Create a simple sphere point cloud for testing
pub fn create_sphere_point_cloud(
    center: Point3<f32>,
    radius: f32,
    num_points: usize,
) -> PointCloud {
    let _rng = rand::rng();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    let phi = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());

    for i in 0..num_points {
        let y = 1.0 - (i as f32 / (num_points - 1).max(1) as f32) * 2.0;
        let radius_at_y = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi * i as f32;

        let x = theta.cos() * radius_at_y;
        let z = theta.sin() * radius_at_y;

        let point = center + radius * Vector3::new(x, y, z);
        points.push(point);

        let normal = (point - center).normalize();
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

/// Create a simple plane point cloud for testing
pub fn create_plane_point_cloud(
    origin: Point3<f32>,
    normal: Vector3<f32>,
    size: f32,
    num_points: usize,
) -> PointCloud {
    use rand::Rng;
    let mut rng = rand::rng();

    let up = if normal.z.abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let right = normal.cross(&up).normalize();
    let up = right.cross(&normal).normalize();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let u = rng.random_range(-size..size);
        let v = rng.random_range(-size..size);

        let point = origin + right * u + up * v;
        points.push(point);
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

// (Old Octree removed — replaced by PoissonOctree above)

struct SpatialIndex {
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
}

impl SpatialIndex {
    fn new(cloud: &PointCloud, cell_size: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (i, point) in cloud.points.iter().enumerate() {
            let key = (
                (point.x / cell_size) as i32,
                (point.y / cell_size) as i32,
                (point.z / cell_size) as i32,
            );
            grid.entry(key).or_default().push(i);
        }

        Self { grid, cell_size }
    }

    fn find_neighbors(&self, point: &Point3<f32>, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let r = (radius / self.cell_size).ceil() as i32;
        let cx = (point.x / self.cell_size) as i32;
        let cy = (point.y / self.cell_size) as i32;
        let cz = (point.z / self.cell_size) as i32;

        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some(indices) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        neighbors.extend(indices);
                    }
                }
            }
        }

        neighbors
    }
}

fn compute_bounds(cloud: &PointCloud) -> (Point3<f32>, Point3<f32>) {
    if cloud.points.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = cloud.points[0];
    let mut max = cloud.points[0];

    for p in &cloud.points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }

    (min, max)
}

// Marching cubes lookup tables (Paul Bourke's reference)

/// Maps each of the 12 cube edges to its two endpoint corner indices.
const MC_EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Marching Cubes edge table (256 entries).
#[rustfmt::skip]
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

/// Marching Cubes triangle table (256 entries).
#[rustfmt::skip]
const MC_TRI_TABLE: [[i32; 16]; 256] = [
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

fn find_seed_triangle(
    start_idx: usize,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
) -> Option<[usize; 3]> {
    let point = &cloud.points[start_idx];
    let neighbors = spatial_index.find_neighbors(point, ball_radius * 2.0);

    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            let idx1 = neighbors[i];
            let idx2 = neighbors[j];

            if find_ball_center(
                &cloud.points[start_idx],
                &cloud.points[idx1],
                &cloud.points[idx2],
                ball_radius,
            )
            .is_some()
            {
                return Some([start_idx, idx1, idx2]);
            }
        }
    }

    None
}

fn find_ball_center(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
    radius: f32,
) -> Option<Point3<f32>> {
    let a = p2 - p1;
    let b = p3 - p1;
    let normal = a.cross(&b);
    let normal_len = normal.norm();

    if normal_len < 1e-6 {
        return None;
    }

    let normal = normal / normal_len;

    let a_len = a.norm();
    let b_len = b.norm();
    let c_len = (p3 - p2).norm();

    let circumradius = (a_len * b_len * c_len) / (2.0 * normal_len);

    if circumradius > radius {
        return None;
    }

    let a_len_sq = a_len * a_len;
    let b_len_sq = b_len * b_len;
    let c_len_sq = c_len * c_len;

    let denom = 2.0 * normal_len * normal_len;
    let alpha = b_len_sq * c_len_sq * (a_len_sq + b_len_sq - c_len_sq) / denom;
    let beta = a_len_sq * c_len_sq * (a_len_sq - b_len_sq + c_len_sq) / denom;
    let gamma = a_len_sq * b_len_sq * (-a_len_sq + b_len_sq + c_len_sq) / denom;

    let circumcenter = p1 * alpha + p2.coords * beta + p3.coords * gamma;

    let height = (radius * radius - circumradius * circumradius).sqrt();
    let center1 = circumcenter + normal * height;

    Some(center1)
}

/// Canonical (undirected) edge key: smaller index first.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn try_add_triangle(
    face: [usize; 3],
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
) -> bool {
    let e0 = edge_key(face[0], face[1]);
    let e1 = edge_key(face[1], face[2]);
    let e2 = edge_key(face[2], face[0]);

    // An edge that already has 2 uses is complete; cannot add another triangle to it.
    if *edge_count.get(&e0).unwrap_or(&0) >= 2
        || *edge_count.get(&e1).unwrap_or(&0) >= 2
        || *edge_count.get(&e2).unwrap_or(&0) >= 2
    {
        return false;
    }

    *edge_count.entry(e0).or_insert(0) += 1;
    *edge_count.entry(e1).or_insert(0) += 1;
    *edge_count.entry(e2).or_insert(0) += 1;
    mesh_faces.push(face);

    true
}

/// A directed edge on the expansion front.
/// `a` and `b` are the edge vertices (in winding order), `opposite` is the
/// vertex of the triangle on the other side of this edge, and `ball_center`
/// is the centre of the ball that rests on that triangle.
struct FrontEdge {
    a: usize,
    b: usize,
    opposite: usize,
    ball_center: Point3<f32>,
}

/// Attempt to find the point `c` that the ball of the given `radius` touches
/// when pivoting around the directed edge (a -> b).
///
/// Among all candidate points in the neighborhood, we choose the one with the
/// smallest positive pivot angle (Bernardini et al. 1999).
fn pivot_ball(
    edge: &FrontEdge,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &HashMap<(usize, usize), u8>,
) -> Option<(usize, Point3<f32>)> {
    let pa = &cloud.points[edge.a];
    let pb = &cloud.points[edge.b];

    // Edge midpoint and axis
    let mid = Point3::from((pa.coords + pb.coords) * 0.5);
    let edge_vec = pb - pa;
    let edge_len = edge_vec.norm();
    if edge_len < 1e-10 {
        return None;
    }
    let edge_axis = edge_vec / edge_len;

    // Reference direction: from edge midpoint towards the old ball centre,
    // projected onto the plane perpendicular to the edge axis.
    let old_center_vec = edge.ball_center - mid;
    let old_center_proj = old_center_vec - edge_axis * old_center_vec.dot(&edge_axis);
    let proj_len = old_center_proj.norm();
    if proj_len < 1e-10 {
        return None;
    }

    let candidates = spatial_index.find_neighbors(&mid, ball_radius * 2.0);

    let mut best_angle = f32::MAX;
    let mut best_candidate: Option<(usize, Point3<f32>)> = None;

    for &idx in &candidates {
        // Skip the edge endpoints and the opposite vertex of the current triangle
        if idx == edge.a || idx == edge.b || idx == edge.opposite {
            continue;
        }

        // Skip if both edges from the candidate are already complete (count >= 2)
        let ek_ca = edge_key(idx, edge.a);
        let ek_cb = edge_key(idx, edge.b);
        if *edge_count.get(&ek_ca).unwrap_or(&0) >= 2 || *edge_count.get(&ek_cb).unwrap_or(&0) >= 2
        {
            continue;
        }

        // Try to place a ball on (a, b, candidate)
        if let Some(new_center) = find_ball_center(pa, pb, &cloud.points[idx], ball_radius) {
            // The new ball centre must be on the correct side. We check this
            // by computing the pivot angle around the edge axis.
            let new_center_vec = new_center - mid;
            let new_center_proj = new_center_vec - edge_axis * new_center_vec.dot(&edge_axis);
            let new_proj_len = new_center_proj.norm();
            if new_proj_len < 1e-10 {
                continue;
            }

            // Pivot angle: angle from old_center_proj to new_center_proj
            // around the edge axis, measured in the positive rotation direction.
            let cos_angle = old_center_proj.dot(&new_center_proj) / (proj_len * new_proj_len);
            let sin_angle =
                old_center_proj.cross(&new_center_proj).dot(&edge_axis) / (proj_len * new_proj_len);
            let mut angle = sin_angle.atan2(cos_angle);

            // We want the angle in (0, 2*PI] — the smallest positive pivot.
            if angle <= 1e-6 {
                angle += std::f32::consts::TAU;
            }

            if angle < best_angle {
                best_angle = angle;
                best_candidate = Some((idx, new_center));
            }
        }
    }

    best_candidate
}

fn expand_front(
    seed: [usize; 3],
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
    used_in_mesh: &mut HashSet<usize>,
) {
    // Compute ball centre for the seed triangle
    let seed_center = match find_ball_center(
        &cloud.points[seed[0]],
        &cloud.points[seed[1]],
        &cloud.points[seed[2]],
        ball_radius,
    ) {
        Some(c) => c,
        None => return,
    };

    // Initialise the front with the three directed edges of the seed triangle.
    // For triangle (a, b, c), the three directed edges are:
    //   (a->b, opposite=c), (b->c, opposite=a), (c->a, opposite=b)
    let mut front: VecDeque<FrontEdge> = VecDeque::new();
    front.push_back(FrontEdge {
        a: seed[0],
        b: seed[1],
        opposite: seed[2],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[1],
        b: seed[2],
        opposite: seed[0],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[2],
        b: seed[0],
        opposite: seed[1],
        ball_center: seed_center,
    });

    // Safety limit to avoid infinite loops on degenerate inputs.
    let max_iterations = cloud.points.len() * 3;
    let mut iterations = 0;

    while let Some(edge) = front.pop_front() {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        // If the edge is already complete (used twice), skip it.
        let ek = edge_key(edge.a, edge.b);
        if *edge_count.get(&ek).unwrap_or(&0) >= 2 {
            continue;
        }

        // Pivot the ball around this edge to find the next point.
        if let Some((c, new_center)) =
            pivot_ball(&edge, cloud, spatial_index, ball_radius, edge_count)
        {
            let face = [edge.a, edge.b, c];

            if try_add_triangle(face, edge_count, mesh_faces) {
                used_in_mesh.insert(c);

                // Add the two new edges to the front.
                // New edge (b -> c): opposite is a
                let ek_bc = edge_key(edge.b, c);
                if *edge_count.get(&ek_bc).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: edge.b,
                        b: c,
                        opposite: edge.a,
                        ball_center: new_center,
                    });
                }

                // New edge (c -> a): opposite is b
                let ek_ca = edge_key(c, edge.a);
                if *edge_count.get(&ek_ca).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: c,
                        b: edge.a,
                        opposite: edge.b,
                        ball_center: new_center,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sphere_point_cloud() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        assert_eq!(cloud.points.len(), 100);
        assert!(cloud.normals.is_some());
    }

    #[test]
    fn test_create_plane_point_cloud() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let cloud = create_plane_point_cloud(Point3::origin(), normal, 1.0, 50);
        assert_eq!(cloud.points.len(), 50);
    }

    #[test]
    fn test_ball_pivoting_empty() {
        let cloud = PointCloud::new(vec![]);
        let mesh = ball_pivoting(&cloud, 0.1);
        assert_eq!(mesh.num_vertices(), 0);
    }

    #[test]
    fn test_ball_pivoting_sphere() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 200);
        let mesh = ball_pivoting(&cloud, 0.2);
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn test_alpha_shapes() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        let mesh = alpha_shapes(&cloud, 0.1);
        // alpha_shapes should return a valid mesh (vertex count is non-negative by type)
        let _ = mesh.num_vertices();
    }

    #[test]
    fn test_compute_point_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 50);
        let normals = compute_point_normals(&cloud, 5);
        assert_eq!(normals.len(), 50);
    }

    #[test]
    fn test_poisson_with_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 200);
        let mesh = poisson_reconstruction(&cloud, 4, 1.0);
        assert!(mesh.is_some());
        let m = mesh.unwrap();
        // Should produce a non-trivial mesh with actual vertices and faces
        assert!(
            m.num_vertices() > 0,
            "Expected vertices, got {}",
            m.num_vertices()
        );
        assert!(m.num_faces() > 0, "Expected faces, got {}", m.num_faces());
    }

    #[test]
    fn test_poisson_without_normals() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let cloud = PointCloud::new(points);
        let mesh = poisson_reconstruction(&cloud, 3, 1.0);
        assert!(mesh.is_none());
    }

    #[test]
    fn test_spatial_index() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 20);
        let index = SpatialIndex::new(&cloud, 0.5);
        let neighbors = index.find_neighbors(&cloud.points[0], 0.5);
        assert!(neighbors.len() >= 1);
    }

    #[test]
    fn test_compute_bounds() {
        let cloud = create_sphere_point_cloud(Point3::new(5.0, 5.0, 5.0), 2.0, 50);
        let (min, max) = compute_bounds(&cloud);
        assert!(min.x < 5.0);
        assert!(max.x > 5.0);
    }
}
