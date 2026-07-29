use nalgebra::Matrix4;

/// KinectFusion-style TSDF depth integration.
///
/// Integrates a single depth frame into a pre-allocated cubic TSDF volume
/// using a running weighted average. The volume is assumed to be a flat
/// cube with side length `n = cbrt(vol.len())`.
///
/// Parameters:
/// - `d`: Depth map (row-major, H x W)
/// - `w`, `h`: Image dimensions
/// - `p`: Camera extrinsics (world-to-camera transform)
/// - `i`: Intrinsics `[fx, fy, cx, cy]`
/// - `vol`: TSDF volume (pre-allocated, size = n^3)
/// - `weights`: Weight volume (same size as `vol`)
/// - `vs`: Voxel size in world units
/// - `tr`: Truncation distance
#[allow(clippy::too_many_arguments)]
pub fn integrate_depth(
    d: &[f32],
    w: u32,
    h: u32,
    p: &Matrix4<f32>,
    i: &[f32; 4],
    vol: &mut [f32],
    weights: &mut [f32],
    vs: f32,
    tr: f32,
) -> Result<(), String> {
    let total = vol.len();
    if total == 0 {
        return Err("Empty volume".to_string());
    }
    if weights.len() != total {
        return Err("vol and weights must have the same length".to_string());
    }

    // Determine cube side length: n = cbrt(total), must be exact.
    let n = (total as f64).cbrt().round() as usize;
    if n * n * n != total {
        return Err(format!(
            "Volume length {} is not a perfect cube (nearest n={})",
            total, n
        ));
    }

    let [fx, fy, cx, cy] = *i;
    let w_img = w as usize;
    let h_img = h as usize;
    let max_weight: f32 = 100.0;

    // Volume origin: centred so that voxel (0,0,0) maps to origin in world space.
    // This matches the common convention where the volume starts at the world origin.
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                // World-space centre of this voxel
                let wx = ix as f32 * vs;
                let wy = iy as f32 * vs;
                let wz = iz as f32 * vs;

                // Transform to camera coordinates: cam = p * [wx, wy, wz, 1]
                let cam_x = p[(0, 0)] * wx + p[(0, 1)] * wy + p[(0, 2)] * wz + p[(0, 3)];
                let cam_y = p[(1, 0)] * wx + p[(1, 1)] * wy + p[(1, 2)] * wz + p[(1, 3)];
                let cam_z = p[(2, 0)] * wx + p[(2, 1)] * wy + p[(2, 2)] * wz + p[(2, 3)];

                // Skip voxels behind the camera
                if cam_z <= 0.0 {
                    continue;
                }

                // Project to pixel coordinates
                let px = fx * cam_x / cam_z + cx;
                let py = fy * cam_y / cam_z + cy;

                let px_i = px as i32;
                let py_i = py as i32;

                if px_i < 0 || py_i < 0 || px_i >= w_img as i32 || py_i >= h_img as i32 {
                    continue;
                }

                let depth = d[py_i as usize * w_img + px_i as usize];
                if depth <= 0.0 {
                    continue;
                }

                // Signed distance: positive means the voxel is in front of the surface
                let sdf = depth - cam_z;

                if sdf < -tr {
                    continue;
                }

                let tsdf_val = (sdf / tr).clamp(-1.0, 1.0);

                // Running weighted average update
                let idx = ix + iy * n + iz * n * n;
                let old_tsdf = vol[idx];
                let old_w = weights[idx];
                let new_w = old_w + 1.0;
                vol[idx] = (old_tsdf * old_w + tsdf_val) / new_w;
                weights[idx] = new_w.min(max_weight);
            }
        }
    }

    Ok(())
}
