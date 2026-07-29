use crate::filters::VoxelDownsampleResult;
use hashbrown::HashMap;
use nalgebra::{Point3, Vector3};

/// Voxel downsampling (Open3D equivalent).
///
/// Groups points into cubic voxels of side `voxel_size` and replaces each
/// voxel's contents with the centroid.  Optionally averages normals (and
/// re-normalises) and colors.
pub fn voxel_downsample(
    points: &[Point3<f64>],
    normals: Option<&[Vector3<f64>]>,
    colors: Option<&[Vector3<f64>]>,
    voxel_size: f64,
) -> VoxelDownsampleResult {
    if points.is_empty() || voxel_size <= 0.0 {
        return VoxelDownsampleResult {
            points: Vec::new(),
            normals: normals.map(|_| Vec::new()),
            colors: colors.map(|_| Vec::new()),
        };
    }

    let inv = 1.0 / voxel_size;

    // Group points by voxel key.
    let mut voxels: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        let key = (
            (p.x * inv).floor() as i64,
            (p.y * inv).floor() as i64,
            (p.z * inv).floor() as i64,
        );
        voxels.entry(key).or_default().push(i);
    }

    let has_normals = normals.is_some();
    let has_colors = colors.is_some();

    let mut out_points = Vec::with_capacity(voxels.len());
    let mut out_normals = if has_normals {
        Some(Vec::with_capacity(voxels.len()))
    } else {
        None
    };
    let mut out_colors = if has_colors {
        Some(Vec::with_capacity(voxels.len()))
    } else {
        None
    };

    for indices in voxels.values() {
        let n = indices.len() as f64;

        // Average position.
        let mut sum = Vector3::zeros();
        for &i in indices {
            sum += points[i].coords;
        }
        out_points.push(Point3::from(sum / n));

        // Average normals (then re-normalise).
        if let (Some(norms), Some(ref mut out_n)) = (&normals, &mut out_normals) {
            let mut nsum = Vector3::zeros();
            for &i in indices {
                nsum += norms[i];
            }
            let len = nsum.norm();
            if len > 1e-15 {
                out_n.push(nsum / len);
            } else {
                out_n.push(Vector3::new(0.0, 0.0, 1.0));
            }
        }

        // Average colors.
        if let (Some(cols), Some(ref mut out_c)) = (&colors, &mut out_colors) {
            let mut csum = Vector3::zeros();
            for &i in indices {
                csum += cols[i];
            }
            out_c.push(csum / n);
        }
    }

    VoxelDownsampleResult {
        points: out_points,
        normals: out_normals,
        colors: out_colors,
    }
}
