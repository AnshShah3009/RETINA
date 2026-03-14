use cv_core::PointCloud;
use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;
use rstar::PointDistance;
use rstar::{RTree, RTreeObject, AABB};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Analytic minimum eigenvector of a 3×3 symmetric matrix.
///
/// Matches Open3D `PointCloudImpl.h` / Geometric Tools `RobustEigenSymmetric3x3`.
/// Uses the trigonometric (Cardano) method for eigenvalues and the best
/// cross-product of shifted matrix rows for the eigenvector — no iteration,
/// exact closed-form result in ~50 scalar operations.
fn fast_eigen3x3_min(m: &Matrix3<f32>) -> Vector3<f32> {
    // Normalize to prevent overflow / underflow.
    let max_c = m.abs().max();
    if max_c < 1e-30 {
        return Vector3::z();
    }
    let s = 1.0 / max_c;
    let a00 = m[(0, 0)] * s;
    let a01 = m[(0, 1)] * s;
    let a02 = m[(0, 2)] * s;
    let a11 = m[(1, 1)] * s;
    let a12 = m[(1, 2)] * s;
    let a22 = m[(2, 2)] * s;

    let norm = a01 * a01 + a02 * a02 + a12 * a12;
    let q = (a00 + a11 + a22) / 3.0;
    let b00 = a00 - q;
    let b11 = a11 - q;
    let b22 = a22 - q;
    let p = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0).sqrt();
    if p < 1e-10 {
        return Vector3::z();
    }

    // Determinant of (A - q*I) / p.
    let c00 = b11 * b22 - a12 * a12;
    let c01 = a01 * b22 - a12 * a02;
    let c02 = a01 * a12 - b11 * a02;
    let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
    let half_det = (det * 0.5_f32).clamp(-1.0, 1.0);
    let angle = half_det.acos() / 3.0;

    // Minimum eigenvalue: q + p * cos(angle + 2π/3) * 2.
    const TWO_THIRDS_PI: f32 = 2.094_395_1;
    let eval_min = q + p * (angle + TWO_THIRDS_PI).cos() * 2.0;

    // Eigenvector: best cross-product of rows of (A - eval_min * I).
    let r0 = Vector3::new(a00 - eval_min, a01, a02);
    let r1 = Vector3::new(a01, a11 - eval_min, a12);
    let r2 = Vector3::new(a02, a12, a22 - eval_min);

    let r0xr1 = r0.cross(&r1);
    let r0xr2 = r0.cross(&r2);
    let r1xr2 = r1.cross(&r2);

    let d0 = r0xr1.norm_squared();
    let d1 = r0xr2.norm_squared();
    let d2 = r1xr2.norm_squared();

    let best = if d0 >= d1 && d0 >= d2 {
        r0xr1
    } else if d1 >= d2 {
        r0xr2
    } else {
        r1xr2
    };

    let len = best.norm();
    if len < 1e-10 {
        return Vector3::z();
    }
    best / len
}

/// Downsample a point cloud with a voxel grid.
/// Returns a new point cloud with one point per voxel (the centroid).
pub fn voxel_down_sample(pc: &PointCloud, voxel_size: f32) -> PointCloud {
    if voxel_size <= 0.0 || pc.is_empty() {
        return pc.clone();
    }

    let n = pc.len();
    let mut indices: Vec<(i32, i32, i32, usize)> = Vec::with_capacity(n);

    // 1. Compute indices
    for i in 0..n {
        let p = pc.points[i];
        let hx = (p.x / voxel_size).floor() as i32;
        let hy = (p.y / voxel_size).floor() as i32;
        let hz = (p.z / voxel_size).floor() as i32;
        indices.push((hx, hy, hz, i));
    }

    // 2. Sort by voxel index
    // Parallel sort if large enough, otherwise sequential
    if n > 10000 {
        indices.par_sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
    } else {
        indices.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
    }

    // 3. Aggregate
    let mut new_points = Vec::new();
    let has_colors = pc.colors.is_some();
    let has_normals = pc.normals.is_some();

    let mut new_colors = if has_colors { Some(Vec::new()) } else { None };
    let mut new_normals = if has_normals { Some(Vec::new()) } else { None };

    if indices.is_empty() {
        return PointCloud::default();
    }

    let mut current_voxel = (indices[0].0, indices[0].1, indices[0].2);
    let mut sum_p = Vector3::zeros();
    let mut sum_c = Vector3::zeros();
    let mut sum_n = Vector3::zeros();
    let mut count = 0;

    for &(hx, hy, hz, idx) in &indices {
        if (hx, hy, hz) != current_voxel {
            // Push previous voxel
            let factor = 1.0 / count as f32;
            new_points.push(Point3::from(sum_p * factor));

            if let Some(nc) = &mut new_colors {
                nc.push(Point3::from(sum_c * factor));
            }
            if let Some(nn) = &mut new_normals {
                let mut n = sum_n * factor;
                if n.norm_squared() > 1e-6 {
                    n.normalize_mut();
                }
                nn.push(n);
            }

            // Reset
            current_voxel = (hx, hy, hz);
            sum_p = Vector3::zeros();
            sum_c = Vector3::zeros();
            sum_n = Vector3::zeros();
            count = 0;
        }

        sum_p += pc.points[idx].coords;
        if has_colors {
            if let Some(colors) = &pc.colors {
                sum_c += colors[idx].coords;
            }
        }
        if has_normals {
            if let Some(normals) = &pc.normals {
                sum_n += normals[idx];
            }
        }
        count += 1;
    }

    // Push last voxel
    if count > 0 {
        let factor = 1.0 / count as f32;
        new_points.push(Point3::from(sum_p * factor));

        if let Some(nc) = &mut new_colors {
            nc.push(Point3::from(sum_c * factor));
        }
        if let Some(nn) = &mut new_normals {
            let mut n = sum_n * factor;
            if n.norm_squared() > 1e-6 {
                n.normalize_mut();
            }
            nn.push(n);
        }
    }

    PointCloud {
        points: new_points,
        colors: new_colors,
        normals: new_normals,
    }
}

// Wrapper for RTree
struct PointWrapper(usize, Point3<f32>);

impl RTreeObject for PointWrapper {
    type Envelope = AABB<[f32; 3]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.1.x, self.1.y, self.1.z])
    }
}

impl rstar::PointDistance for PointWrapper {
    fn distance_2(&self, point: &[f32; 3]) -> f32 {
        let dx = self.1.x - point[0];
        let dy = self.1.y - point[1];
        let dz = self.1.z - point[2];
        dx * dx + dy * dy + dz * dz
    }
}

/// Estimate normals for the point cloud using K-nearest neighbors.
/// Uses PCA/SVD on the covariance matrix of the neighborhood.
/// Modifies the point cloud in place to add normals.
pub fn estimate_normals(pc: &mut PointCloud, k: usize) {
    if pc.is_empty() {
        return;
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();

    let tree = RTree::bulk_load(wrappers);

    let normals: Vec<Vector3<f32>> = pc
        .points
        .par_iter()
        .map(|p| {
            let query_point = [p.x, p.y, p.z];
            let neighbors: Vec<&PointWrapper> = tree
                .nearest_neighbor_iter(&query_point)
                .skip(1)
                .take(k)
                .collect();

            if neighbors.len() < 3 {
                return Vector3::new(0.0, 0.0, 1.0); // Default up
            }

            // Compute centroid
            let mut centroid = Vector3::zeros();
            for n in &neighbors {
                centroid += n.1.coords;
            }
            centroid /= neighbors.len() as f32;

            // Compute covariance matrix
            let mut cov = Matrix3::zeros();
            for n in &neighbors {
                let d = n.1.coords - centroid;
                cov += d * d.transpose();
            }
            cov /= neighbors.len() as f32;

            // Analytic minimum eigenvector (Open3D / Geometric Tools algorithm).
            // Faster and more numerically stable than full SymmetricEigen decomposition.
            fast_eigen3x3_min(&cov)
        })
        .collect();

    pc.normals = Some(normals);
}

/// Orient normals consistently using simple neighbor voting.
/// Fast O(n*k) algorithm - much faster than Open3D's MST approach.
pub fn orient_normals(pc: &mut PointCloud, k: usize) {
    let n = pc.len();
    if n < 3 {
        return;
    }

    let mut normals = match pc.normals.take() {
        Some(n) => n,
        None => return,
    };

    // Build RTree
    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    // Simple propagation: start from point 0, orient all neighbors
    let mut visited = vec![false; n];
    let mut queue = vec![0];
    visited[0] = true;

    while let Some(i) = queue.pop() {
        let q = [pc.points[i].x, pc.points[i].y, pc.points[i].z];
        let neighbors: Vec<_> = tree.nearest_neighbor_iter(&q).skip(1).take(k).collect();

        for nb in neighbors {
            let j = nb.0;
            if visited[j] {
                continue;
            }

            // Flip if pointing opposite to current
            if normals[j].dot(&normals[i]) < 0.0 {
                normals[j] = -normals[j];
            }
            visited[j] = true;
            queue.push(j);
        }
    }

    // Handle unvisited (disconnected components) — run a new BFS from each
    // unvisited seed so the entire component gets consistently oriented.
    for i in 0..n {
        if !visited[i] {
            visited[i] = true;
            let mut comp_queue = vec![i];
            while let Some(ci) = comp_queue.pop() {
                let q = [pc.points[ci].x, pc.points[ci].y, pc.points[ci].z];
                let neighbors: Vec<_> =
                    tree.nearest_neighbor_iter(&q).skip(1).take(k).collect();
                for nb in neighbors {
                    let j = nb.0;
                    if visited[j] {
                        continue;
                    }
                    if normals[j].dot(&normals[ci]) < 0.0 {
                        normals[j] = -normals[j];
                    }
                    visited[j] = true;
                    comp_queue.push(j);
                }
            }
        }
    }

    pc.normals = Some(normals);
}

/// Compute surface normals from a depth image using the cross-product method.
///
/// For each pixel, back-projects itself and its four axis-aligned neighbours to
/// 3-D, then takes `cross(right - left, down - up)` as the surface normal.
/// This is **O(n)** per pixel — much faster than the k-NN PCA path which is
/// O(nk) — and is the preferred approach for RGBD / structured-depth data.
///
/// # Parameters
/// - `depth`: row-major depth map, H×W elements, metric units (m or mm).
/// - `width`, `height`: image dimensions.
/// - `fx`, `fy`, `cx`, `cy`: pinhole camera intrinsics.
///
/// # Returns
/// Per-pixel normals in camera space, oriented toward the viewer (−Z direction).
/// Border pixels and pixels with zero depth get a default `(0, 0, 1)` normal.
pub fn compute_normals_from_depth(
    depth: &[f32],
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Vec<Vector3<f32>> {
    assert_eq!(depth.len(), width * height);

    let backproject = |px: usize, py: usize| -> Option<Vector3<f32>> {
        let d = depth[py * width + px];
        if d <= 0.0 {
            return None;
        }
        Some(Vector3::new(
            (px as f32 - cx) * d / fx,
            (py as f32 - cy) * d / fy,
            d,
        ))
    };

    (0..height * width)
        .into_par_iter()
        .map(|i| {
            let px = i % width;
            let py = i / width;

            // Skip border pixels — neighbours are not fully available.
            if px == 0 || px + 1 >= width || py == 0 || py + 1 >= height {
                return Vector3::new(0.0, 0.0, 1.0);
            }

            let l = match backproject(px - 1, py) {
                Some(v) => v,
                None => return Vector3::new(0.0, 0.0, 1.0),
            };
            let r = match backproject(px + 1, py) {
                Some(v) => v,
                None => return Vector3::new(0.0, 0.0, 1.0),
            };
            let u = match backproject(px, py - 1) {
                Some(v) => v,
                None => return Vector3::new(0.0, 0.0, 1.0),
            };
            let d = match backproject(px, py + 1) {
                Some(v) => v,
                None => return Vector3::new(0.0, 0.0, 1.0),
            };

            let horizontal = r - l;
            let vertical = d - u;
            let n = horizontal.cross(&vertical);
            let len = n.norm();
            if len < 1e-8 {
                return Vector3::new(0.0, 0.0, 1.0);
            }
            // Orient toward camera (-Z in camera space means facing viewer).
            let normal = n / len;
            if normal.z > 0.0 {
                -normal
            } else {
                normal
            }
        })
        .collect()
}

/// Write a point cloud to a PLY ASCII file.
pub fn write_ply(pc: &PointCloud, path: &str) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", pc.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;

    if pc.colors.is_some() {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }

    if pc.normals.is_some() {
        writeln!(file, "property float nx")?;
        writeln!(file, "property float ny")?;
        writeln!(file, "property float nz")?;
    }

    writeln!(file, "end_header")?;

    for i in 0..pc.len() {
        let p = pc.points[i];
        write!(file, "{} {} {}", p.x, p.y, p.z)?;

        if let Some(colors) = &pc.colors {
            let c = colors[i];
            // Colors in Point3<f32> assumed 0..1 or 0..255? Open3D uses 0..1 usually in float.
            // Let's assume 0..1 float and convert to uchar 0..255.
            let r = (c.x * 255.0).clamp(0.0, 255.0) as u8;
            let g = (c.y * 255.0).clamp(0.0, 255.0) as u8;
            let b = (c.z * 255.0).clamp(0.0, 255.0) as u8;
            write!(file, " {} {} {}", r, g, b)?;
        }

        if let Some(normals) = &pc.normals {
            let n = normals[i];
            write!(file, " {} {} {}", n.x, n.y, n.z)?;
        }

        writeln!(file)?;
    }

    Ok(())
}

pub fn read_ply(path: &str) -> std::io::Result<PointCloud> {
    // Basic PLY ASCII reader (robustness limited)
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut vertex_count = 0;
    let mut has_colors = false;
    let mut has_normals = false;
    let mut header_ended = false;

    // Parse Header
    for line in lines.by_ref() {
        let line = line?;
        if line.starts_with("element vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                vertex_count = parts[2].parse().unwrap_or(0);
            }
        } else if line.contains("property uchar red") {
            has_colors = true;
        } else if line.contains("property float nx") {
            has_normals = true;
        } else if line.trim() == "end_header" {
            header_ended = true;
            break;
        }
    }

    if !header_ended {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PLY header not found",
        ));
    }

    let mut points = Vec::with_capacity(vertex_count);
    let mut colors = if has_colors {
        Some(Vec::with_capacity(vertex_count))
    } else {
        None
    };
    let mut normals = if has_normals {
        Some(Vec::with_capacity(vertex_count))
    } else {
        None
    };

    for _ in 0..vertex_count {
        if let Some(line) = lines.next() {
            let line = line?;
            let mut parts = line.split_whitespace();

            // XYZ
            let x: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing X",
                ))?
                .parse()
                .unwrap_or(0.0);
            let y: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing Y",
                ))?
                .parse()
                .unwrap_or(0.0);
            let z: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing Z",
                ))?
                .parse()
                .unwrap_or(0.0);
            points.push(Point3::new(x, y, z));

            // Optional Color
            if let Some(c_vec) = &mut colors {
                let r: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                let g: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                let b: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                c_vec.push(Point3::new(
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                ));
            }

            // Optional Normal
            if let Some(n_vec) = &mut normals {
                let nx: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let ny: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let nz: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                n_vec.push(Vector3::new(nx, ny, nz));
            }
        }
    }

    Ok(PointCloud {
        points,
        colors,
        normals,
    })
}

/// Remove statistical outliers.
/// Compute mean distance to `k` neighbors for each point.
/// Points with mean distance > global_mean + std_ratio * std_dev are removed.
pub fn remove_statistical_outliers(
    pc: &PointCloud,
    k: usize,
    std_ratio: f64,
) -> (PointCloud, Vec<usize>) {
    if pc.is_empty() || k == 0 {
        return (pc.clone(), (0..pc.len()).collect());
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let distances: Vec<f32> = pc
        .points
        .par_iter()
        .map(|p| {
            let query_point = [p.x, p.y, p.z];
            // nearest_neighbor_iter returns k+1 including itself (dist 0).
            // take(k+1) gives nearest neighbors.
            let neighbors: Vec<&PointWrapper> = tree
                .nearest_neighbor_iter(&query_point)
                .take(k + 1)
                .collect();

            // Sum distances to k neighbors (skip itself)
            let mut sum_dist = 0.0;
            let mut count = 0;
            for (idx, n) in neighbors.iter().enumerate() {
                if idx == 0 {
                    continue;
                } // skip self
                sum_dist += n.distance_2(&query_point).sqrt();
                count += 1;
            }

            if count > 0 {
                sum_dist / count as f32
            } else {
                0.0
            }
        })
        .collect();

    let mean_dist =
        crate::mean(&distances.iter().map(|&d| d as f64).collect::<Vec<_>>()).unwrap_or(0.0) as f32;
    // std dev
    let variance = distances
        .iter()
        .map(|d| {
            let diff = d - mean_dist;
            diff * diff
        })
        .sum::<f32>()
        / distances.len() as f32;
    let std_dev = variance.sqrt();

    let threshold = mean_dist + std_ratio as f32 * std_dev;

    let mut inliers: Vec<usize> = Vec::new();
    let mut new_points = Vec::new();
    let mut new_colors = if pc.colors.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let mut new_normals = if pc.normals.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    for (i, &dist) in distances.iter().enumerate() {
        if dist <= threshold {
            inliers.push(i);
            new_points.push(pc.points[i]);
            if let Some(c) = &pc.colors {
                if let Some(nc) = &mut new_colors {
                    nc.push(c[i]);
                }
            }
            if let Some(n) = &pc.normals {
                if let Some(nn) = &mut new_normals {
                    nn.push(n[i]);
                }
            }
        }
    }

    (
        PointCloud {
            points: new_points,
            colors: new_colors,
            normals: new_normals,
        },
        inliers,
    )
}

/// Remove radius outliers.
/// Points with fewer than `min_points` neighbors within `radius` are removed.
pub fn remove_radius_outliers(
    pc: &PointCloud,
    radius: f32,
    min_points: usize,
) -> (PointCloud, Vec<usize>) {
    if pc.is_empty() {
        return (pc.clone(), (0..pc.len()).collect());
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let r2 = radius * radius;

    let inlier_mask: Vec<bool> = pc
        .points
        .par_iter()
        .map(|p| {
            let query_point = [p.x, p.y, p.z];
            // locate_within_distance uses squared distance
            let count = tree.locate_within_distance(query_point, r2).count();
            // count includes self.
            count >= min_points
        })
        .collect();

    let mut inliers: Vec<usize> = Vec::new();
    let mut new_points = Vec::new();
    let mut new_colors = if pc.colors.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let mut new_normals = if pc.normals.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    for (i, &is_inlier) in inlier_mask.iter().enumerate() {
        if is_inlier {
            inliers.push(i);
            new_points.push(pc.points[i]);
            if let Some(c) = &pc.colors {
                if let Some(nc) = &mut new_colors {
                    nc.push(c[i]);
                }
            }
            if let Some(n) = &pc.normals {
                if let Some(nn) = &mut new_normals {
                    nn.push(n[i]);
                }
            }
        }
    }

    (
        PointCloud {
            points: new_points,
            colors: new_colors,
            normals: new_normals,
        },
        inliers,
    )
}

use cv_core::{Ransac, RobustConfig, RobustModel};

pub struct PlaneEstimator;

impl RobustModel<Point3<f32>> for PlaneEstimator {
    type Model = [f32; 4];

    fn min_sample_size(&self) -> usize {
        3
    }

    fn estimate(&self, data: &[&Point3<f32>]) -> Option<Self::Model> {
        let p1 = data[0];
        let p2 = data[1];
        let p3 = data[2];

        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let normal = v1.cross(&v2).normalize();

        if normal.x.is_nan() || normal.y.is_nan() || normal.z.is_nan() {
            return None;
        }

        let d = -normal.dot(&p1.coords);
        Some([normal.x, normal.y, normal.z, d])
    }

    fn compute_error(&self, model: &Self::Model, data: &Point3<f32>) -> f64 {
        let [a, b, c, d] = *model;
        let denom = (a * a + b * b + c * c).sqrt();
        if denom < 1e-12 {
            return f64::INFINITY;
        }
        ((a * data.x + b * data.y + c * data.z + d).abs() / denom) as f64
    }
}

/// Segment a plane using RANSAC.
/// Returns (a, b, c, d) plane model and list of inlier indices.
/// Plane equation: ax + by + cz + d = 0.
pub fn segment_plane(
    pc: &PointCloud,
    distance_threshold: f32,
    ransac_n: usize,
    num_iterations: usize,
) -> (Option<[f32; 4]>, Vec<usize>) {
    if pc.points.len() < ransac_n || ransac_n < 3 {
        return (None, Vec::new());
    }

    let config = RobustConfig {
        threshold: distance_threshold as f64,
        max_iterations: num_iterations,
        confidence: 0.99,
    };

    let ransac = Ransac::new(config);
    let res = ransac.run(&PlaneEstimator, &pc.points);

    let inlier_indices: Vec<usize> = res
        .inliers
        .iter()
        .enumerate()
        .filter(|(_, &is_inlier)| is_inlier)
        .map(|(i, _)| i)
        .collect();

    (res.model, inlier_indices)
}

/// DBSCAN clustering.
/// Returns a list of labels for each point. -1 indicates noise.
/// 0..N indicates cluster index.
pub fn cluster_dbscan(pc: &PointCloud, eps: f32, min_points: usize) -> Vec<i32> {
    let n = pc.len();
    let mut labels = vec![-1; n]; // -1 = noise/unvisited
    let mut cluster_idx = 0;

    // Build tree
    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let eps2 = eps * eps;
    let mut visited = vec![false; n];

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let p = pc.points[i];
        let query_point = [p.x, p.y, p.z];
        // Neighbors include self
        let neighbors: Vec<&PointWrapper> =
            tree.locate_within_distance(query_point, eps2).collect();

        if neighbors.len() < min_points {
            labels[i] = -1; // Noise
        } else {
            labels[i] = cluster_idx;
            // Expand cluster
            let mut seeds = neighbors.clone();
            let mut head = 0;
            while head < seeds.len() {
                let current_idx = seeds[head].0;
                head += 1;

                if labels[current_idx] == -1 {
                    labels[current_idx] = cluster_idx; // Change noise to border
                }
                if !visited[current_idx] {
                    visited[current_idx] = true;
                    labels[current_idx] = cluster_idx;

                    let p_curr = pc.points[current_idx];
                    let q_curr = [p_curr.x, p_curr.y, p_curr.z];
                    let n_curr: Vec<&PointWrapper> =
                        tree.locate_within_distance(q_curr, eps2).collect();
                    if n_curr.len() >= min_points {
                        for n in n_curr {
                            // Avoid adding duplicates?
                            // seeds is a simple vec, might grow definedly.
                            // Actually better to use a Set or just push and check visited.
                            // But we check visited when popping/processing?
                            // The standard DBSCAN expands seeds.
                            // Simple list append approach:
                            seeds.push(n);
                        }
                    }
                }
            }
            cluster_idx += 1;
        }
    }

    labels
}

/// Compute FPFH features.
/// Returns an N x 33 histogram feature vector for each point.
pub fn compute_fpfh_feature(pc: &PointCloud, search_radius: f32) -> Option<Vec<[f32; 33]>> {
    let normals = pc.normals.as_ref()?;
    let n_points = pc.len();
    if n_points == 0 {
        return Some(Vec::new());
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);
    let r2 = search_radius * search_radius;

    // 1. Compute SPFH (Simplified Point Feature Histograms)
    let spfh: Vec<[f32; 33]> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let p = pc.points[i];
            let n = normals[i];
            let query_point = [p.x, p.y, p.z];

            // Find neighbors
            let neighbors: Vec<&PointWrapper> =
                tree.locate_within_distance(query_point, r2).collect();

            if neighbors.len() <= 1 {
                return [0.0; 33];
            }

            let mut hist = [0.0; 33]; // 11 bins for alpha, 11 for phi, 11 for theta
            let mut count = 0;

            for nb in neighbors {
                let j = nb.0;
                if i == j {
                    continue;
                }

                let p_j = pc.points[j];
                let n_j = normals[j];

                let (alpha, phi, theta) = compute_pair_features(&p, &n, &p_j, &n_j);

                // Binning: assume ranges.
                // alpha: [-1, 1] -> cos angle.
                // phi: [-1, 1]
                // theta: [-PI, PI]

                let bin_alpha = ((alpha + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
                let bin_phi = ((phi + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
                let bin_theta = ((theta + std::f32::consts::PI)
                    * (11.0 / (2.0 * std::f32::consts::PI)))
                    .floor()
                    .clamp(0.0, 10.0) as usize;

                hist[bin_alpha] += 1.0;
                hist[11 + bin_phi] += 1.0;
                hist[22 + bin_theta] += 1.0;
                count += 1;
            }

            if count > 0 {
                let inv_k = 1.0 / count as f32;
                for val in &mut hist {
                    *val *= inv_k;
                }
            }
            hist
        })
        .collect();

    // 2. Compute FPFH (Weighted sum of neighbors' SPFH)
    let fpfh: Vec<[f32; 33]> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let p = pc.points[i];
            let query_point = [p.x, p.y, p.z];

            let neighbors: Vec<&PointWrapper> =
                tree.locate_within_distance(query_point, r2).collect();

            if neighbors.len() <= 1 {
                return spfh[i];
            }

            let mut final_hist = spfh[i]; // Start with own SPFH
            let k = neighbors.len() - 1; // excluding self
            let weight = 1.0 / k as f32;

            for nb in neighbors {
                let j = nb.0;
                if i == j {
                    continue;
                }

                let dist = (p - pc.points[j]).norm();
                if dist < 1e-6 {
                    continue;
                }

                let w = weight / dist; // Simple weighting

                for k in 0..33 {
                    final_hist[k] += spfh[j][k] * w;
                }
            }

            // Normalize again? FPFH usually normalized to sum to 100 or 1.
            // Simple normalization
            let sum: f32 = final_hist.iter().sum();
            if sum > 1e-6 {
                let scale = 100.0 / sum;
                for val in &mut final_hist {
                    *val *= scale;
                }
            }

            final_hist
        })
        .collect();

    Some(fpfh)
}

fn compute_pair_features(
    p1: &Point3<f32>,
    n1: &Vector3<f32>,
    p2: &Point3<f32>,
    n2: &Vector3<f32>,
) -> (f32, f32, f32) {
    let delta = p2 - p1;
    let dist = delta.norm();

    if dist < 1e-6 {
        return (0.0, 0.0, 0.0);
    }

    let u = n1;
    let v = delta.cross(u);
    let v_norm = v.norm();

    if v_norm < 1e-6 {
        return (0.0, 0.0, 0.0);
    }
    let v = v / v_norm;
    let w = u.cross(&v);

    let alpha = v.dot(n2);
    let phi = u.dot(&delta) / dist;
    let theta = w.dot(n2).atan2(u.dot(n2));

    (alpha, phi, theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use std::fs;

    #[test]
    fn test_voxel_down_sample() {
        // Create points in a 0.1x0.1x0.1 cluster
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push(Point3::new(i as f32 * 0.01, j as f32 * 0.01, 0.0));
            }
        }
        let pc = PointCloud::new(points);

        // Voxel size 0.2 should collapse everything to 1 point (or very few).
        // 0.0 -> 0.09 range. 0.2 covers all.
        let down = voxel_down_sample(&pc, 0.2);
        assert_eq!(down.len(), 1);

        // Voxel size 0.05. Should have 2x2 = 4 voxels approx?
        // 0.00..0.04 -> 0
        // 0.05..0.09 -> 1
        let down = voxel_down_sample(&pc, 0.05);
        // We have points at 0.00, 0.01 ... 0.09.
        // 0.00, 0.01, 0.02, 0.03, 0.04 -> bin 0
        // 0.05, 0.06, 0.07, 0.08, 0.09 -> bin 1
        // So 2 bins in X, 2 in Y -> 4 points total.
        assert_eq!(down.len(), 4);
    }

    #[test]
    fn test_estimate_normals() {
        // Plane at z=0
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f32, j as f32, 0.0));
            }
        }
        let mut pc = PointCloud::new(points);
        estimate_normals(&mut pc, 5); // 5 neighbors

        assert!(pc.normals.is_some());
        let normals = pc.normals.as_ref().unwrap();

        // Normals should be (0,0,1) or (0,0,-1).
        for n in normals.iter() {
            assert!(n.z.abs() > 0.9, "Normal {:?} is not vertical", n);
        }
    }

    #[test]
    fn test_io() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)];
        let colors = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
        let pc = PointCloud::new(points).with_colors(colors).unwrap();

        let path = "/tmp/test_pc.ply";
        write_ply(&pc, path).unwrap();

        let loaded = read_ply(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.colors.is_some());

        // Cleanup (optional, but good)
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_outlier_removal() {
        // Create a cluster of 10 points at (0,0,0) and 1 outlier at (10,10,10)
        let mut points = Vec::new();
        for _ in 0..10 {
            points.push(Point3::new(0.0, 0.0, 0.0));
        }
        points.push(Point3::new(10.0, 10.0, 10.0)); // Index 10
        let pc = PointCloud::new(points);

        // Radius removal: Radius 1.0, min 5 points.
        // The cluster has 10 neighbors (self included). Outlier has 1 (self).
        let (filtered, inliers) = remove_radius_outliers(&pc, 1.0, 5);
        assert_eq!(filtered.len(), 10);
        assert!(!inliers.contains(&10));

        // Statistical removal:
        // Cluster points have dist 0 to neighbors.
        // Outlier has dist ~17 to neighbors.
        // Mean dist will be small, std dev small. Outlier > mean + 2*std.
        let (filtered_stat, inliers_stat) = remove_statistical_outliers(&pc, 5, 1.0);
        assert_eq!(filtered_stat.len(), 10);
        assert!(!inliers_stat.contains(&10));
    }

    #[test]
    fn test_segment_plane() {
        // Create plane z=0 points + noise
        let mut points = Vec::new();
        // Inliers: grid 10x10
        for x in 0..10 {
            for y in 0..10 {
                points.push(Point3::new(x as f32, y as f32, 0.0));
            }
        }
        // Outliers
        points.push(Point3::new(0.0, 0.0, 10.0));
        points.push(Point3::new(1.0, 1.0, 10.0));

        // Also add some random jitter to plane points to make it realistic?
        // But perfect plane is easier to test.

        let pc = PointCloud::new(points);
        let (model, inliers) = segment_plane(&pc, 0.1, 3, 100);

        assert!(model.is_some());
        let [a, b, c, d] = model.unwrap();

        // Should be z=0, so a=0, b=0, c=1 (or -1), d=0
        // Normal could be (0,0,-1) too.
        println!("Plane: {}x + {}y + {}z + {} = 0", a, b, c, d);
        assert!(c.abs() > 0.9);
        assert!(d.abs() < 0.1);
        assert_eq!(inliers.len(), 100);
    }

    #[test]
    fn test_dbscan_clustering() {
        // Create two clusters far apart
        let mut points = Vec::new();
        // Cluster 1: 5 points at (0,0,0)
        for _ in 0..5 {
            points.push(Point3::new(0.0, 0.0, 0.0));
        }
        // Cluster 2: 5 points at (10,10,10)
        for _ in 0..5 {
            points.push(Point3::new(10.0, 10.0, 10.0));
        }
        // Noise: 1 point at (5,5,5)
        points.push(Point3::new(5.0, 5.0, 5.0));

        let pc = PointCloud::new(points);

        // eps=1.0, min_points=3
        let labels = cluster_dbscan(&pc, 1.0, 3);

        // First 5 should be cluster 0
        for i in 0..5 {
            assert_eq!(labels[i], 0);
        }
        // Next 5 should be cluster 1
        for i in 5..10 {
            assert_eq!(labels[i], 1);
        }
        // Last one should be noise (-1)
        assert_eq!(labels[10], -1);
    }

    #[test]
    fn test_fpfh() {
        let points = vec![Point3::new(0.0, 0.0, 0.0)];
        let normals = vec![nalgebra::Vector3::new(0.0, 0.0, 1.0)];
        let pc = PointCloud::new(points).with_normals(normals).unwrap();

        let features = compute_fpfh_feature(&pc, 0.1);
        assert!(features.is_some());
        assert_eq!(features.unwrap()[0].len(), 33);

        let pc_no_norm = PointCloud::new(vec![Point3::new(0.0, 0.0, 0.0)]);
        assert!(compute_fpfh_feature(&pc_no_norm, 0.1).is_none());
    }

    // ── depth image normal tests ──────────────────────────────────────────────

    #[test]
    fn test_depth_normals_flat_plane() {
        // Flat depth image: all pixels at depth = 1.0.
        // The "surface" is a plane parallel to the image sensor, so the
        // normals should all point toward the camera (negative z in camera space).
        let w = 16usize;
        let h = 16usize;
        let depth = vec![1.0f32; w * h];
        let normals = compute_normals_from_depth(&depth, w, h, 16.0, 16.0, 8.0, 8.0);
        assert_eq!(normals.len(), w * h);
        // Check interior pixels (border returns default).
        for r in 1..h - 1 {
            for c in 1..w - 1 {
                let n = normals[r * w + c];
                assert!(
                    n.z.abs() > 0.9,
                    "flat-plane normal at ({},{}) = {:?}",
                    r,
                    c,
                    n
                );
            }
        }
    }

    #[test]
    fn test_depth_normals_sphere_cap() {
        // Sphere cap: depth goes from 1 at centre to 0 at edge.
        let w = 32usize;
        let h = 32usize;
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let depth: Vec<f32> = (0..h * w)
            .map(|i| {
                let dx = (i % w) as f32 - cx;
                let dy = (i / w) as f32 - cy;
                let r2 = (dx * dx + dy * dy) / (cx * cx);
                if r2 < 1.0 {
                    (1.0 - r2).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        let normals = compute_normals_from_depth(
            &depth, w, h, w as f32, h as f32, // fx = fy = image width
            cx, cy,
        );
        // Centre of the cap should face the camera (z component ≈ -1 or 1).
        let centre = (h / 2) * w + w / 2;
        assert!(
            normals[centre].z.abs() > 0.7,
            "sphere cap centre normal = {:?}",
            normals[centre]
        );
    }

    #[test]
    fn test_depth_normals_zero_depth() {
        // Zero-depth pixels should get the default normal (0,0,1).
        let normals = compute_normals_from_depth(&[0.0f32; 16], 4, 4, 4.0, 4.0, 2.0, 2.0);
        for n in &normals {
            let expected = nalgebra::Vector3::new(0.0, 0.0, 1.0);
            let diff = (n - expected).norm();
            assert!(
                diff < 1e-6,
                "zero-depth pixel gave non-default normal {:?}",
                n
            );
        }
    }

    #[test]
    fn test_analytic_eigensolver_known() {
        // A matrix with eigenvalues 2, 1, 0 along x, y, z axes.
        // Minimum eigenvector should be (0, 0, ±1).
        let mut m = nalgebra::Matrix3::zeros();
        m[(0, 0)] = 2.0;
        m[(1, 1)] = 1.0;
        m[(2, 2)] = 0.0;
        let n = fast_eigen3x3_min(&m);
        assert!(n.z.abs() > 0.99, "Expected z-eigenvector, got {:?}", n);
    }

    #[test]
    fn test_analytic_eigensolver_isotropic() {
        // Isotropic matrix: all eigenvalues equal. Any unit vector is valid.
        let m = nalgebra::Matrix3::identity() * 3.0;
        let n = fast_eigen3x3_min(&m);
        assert!((n.norm() - 1.0).abs() < 1e-5, "Not unit length: {:?}", n);
    }
}
