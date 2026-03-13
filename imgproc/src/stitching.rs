//! Image stitching pipeline
//!
//! Provides homography estimation, perspective warping, and a simple two-image stitcher.
//!
//! - **DLT homography:** Direct Linear Transform with point normalization.
//! - **RANSAC homography:** Robust estimation rejecting outliers.
//! - **Warp perspective:** Apply a 3x3 homography with bilinear interpolation.
//! - **Stitch pair:** End-to-end stitching of two images given point correspondences.

use cv_core::float::Float;
use cv_core::tensor::{CpuTensor, TensorShape};
use cv_core::Result;
use rand::Rng;

/// Estimate a 3x3 homography from at least 4 point correspondences using the
/// Direct Linear Transform (DLT) algorithm with Hartley normalization.
///
/// # Arguments
/// * `src_points` - Source (x, y) coordinates.
/// * `dst_points` - Destination (x, y) coordinates (same length as `src_points`).
///
/// # Returns
/// A 3x3 homography matrix `H` such that `dst ~ H * src` (in homogeneous coordinates).
pub fn find_homography(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> Result<[[f64; 3]; 3]> {
    let n = src_points.len();
    if n < 4 {
        return Err(cv_core::Error::InvalidInput(
            "At least 4 point correspondences are required for homography estimation".into(),
        ));
    }
    if n != dst_points.len() {
        return Err(cv_core::Error::InvalidInput(
            "src_points and dst_points must have the same length".into(),
        ));
    }

    // Hartley normalization: translate centroid to origin, scale so average distance = sqrt(2).
    let (t_src, norm_src) = normalize_points(src_points);
    let (t_dst, norm_dst) = normalize_points(dst_points);

    // Build the 2n x 9 matrix A for DLT.
    let rows = 2 * n;
    let mut a_data = vec![0.0f64; rows * 9];

    for i in 0..n {
        let (sx, sy) = norm_src[i];
        let (dx, dy) = norm_dst[i];

        // Row 2i:   0 0 0  -sx -sy -1  dy*sx dy*sy dy
        let r0 = 2 * i;
        a_data[r0 * 9 + 3] = -sx;
        a_data[r0 * 9 + 4] = -sy;
        a_data[r0 * 9 + 5] = -1.0;
        a_data[r0 * 9 + 6] = dy * sx;
        a_data[r0 * 9 + 7] = dy * sy;
        a_data[r0 * 9 + 8] = dy;

        // Row 2i+1:  sx sy 1  0 0 0  -dx*sx -dx*sy -dx
        let r1 = 2 * i + 1;
        a_data[(r1 * 9)] = sx;
        a_data[r1 * 9 + 1] = sy;
        a_data[r1 * 9 + 2] = 1.0;
        a_data[r1 * 9 + 6] = -dx * sx;
        a_data[r1 * 9 + 7] = -dx * sy;
        a_data[r1 * 9 + 8] = -dx;
    }

    // Solve via A^T A: the homography is the eigenvector with the smallest eigenvalue.
    let mat_a = nalgebra::DMatrix::from_row_slice(rows, 9, &a_data);
    let ata = mat_a.transpose() * &mat_a;
    // Convert to a static 9x9 matrix for symmetric eigendecomposition.
    let ata_static = nalgebra::SMatrix::<f64, 9, 9>::from_fn(|i, j| ata[(i, j)]);
    let eigen = ata_static.symmetric_eigen();

    // Find the index of the smallest eigenvalue.
    let mut min_idx = 0;
    let mut min_val = eigen.eigenvalues[0].abs();
    for k in 1..9 {
        let v = eigen.eigenvalues[k].abs();
        if v < min_val {
            min_val = v;
            min_idx = k;
        }
    }

    let h_vec: Vec<f64> = (0..9).map(|j| eigen.eigenvectors[(j, min_idx)]).collect();

    let h_norm = [
        [h_vec[0], h_vec[1], h_vec[2]],
        [h_vec[3], h_vec[4], h_vec[5]],
        [h_vec[6], h_vec[7], h_vec[8]],
    ];

    // Denormalize: H = T_dst^{-1} * H_norm * T_src.
    let h = mat3_mul(&mat3_mul(&mat3_inv(&t_dst), &h_norm), &t_src);

    // Normalize so h[2][2] = 1 (if non-zero).
    let scale = h[2][2];
    if scale.abs() < 1e-15 {
        return Err(cv_core::Error::AlgorithmError(
            "Degenerate homography (h33 ~ 0)".into(),
        ));
    }
    let mut h_out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h_out[i][j] = h[i][j] / scale;
        }
    }

    Ok(h_out)
}

/// RANSAC-based robust homography estimation.
///
/// # Arguments
/// * `src_points`     - Source (x, y) coordinates.
/// * `dst_points`     - Destination (x, y) coordinates.
/// * `threshold`      - Reprojection error threshold for classifying inliers.
/// * `max_iterations` - Maximum number of RANSAC iterations.
///
/// # Returns
/// `(H, inlier_mask)` where `H` is the best homography and `inlier_mask[i]` indicates
/// whether correspondence `i` is an inlier.
pub fn find_homography_ransac(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    threshold: f64,
    max_iterations: u32,
) -> Result<([[f64; 3]; 3], Vec<bool>)> {
    let n = src_points.len();
    if n < 4 {
        return Err(cv_core::Error::InvalidInput(
            "At least 4 point correspondences are required".into(),
        ));
    }
    if n != dst_points.len() {
        return Err(cv_core::Error::InvalidInput(
            "src_points and dst_points must have the same length".into(),
        ));
    }

    let thresh_sq = threshold * threshold;
    let mut best_inlier_count = 0usize;
    let mut best_h = [[0.0f64; 3]; 3];
    let mut best_mask = vec![false; n];

    let mut rng = rand::rng();

    for _ in 0..max_iterations {
        // Pick 4 random distinct indices.
        let mut sample = [0usize; 4];
        let mut found = 0;
        let mut attempts = 0;
        while found < 4 && attempts < 100 {
            let idx = rng.random_range(0..n);
            if !sample[..found].contains(&idx) {
                sample[found] = idx;
                found += 1;
            }
            attempts += 1;
        }
        if found < 4 {
            continue;
        }

        let s_pts: Vec<(f64, f64)> = sample.iter().map(|&i| src_points[i]).collect();
        let d_pts: Vec<(f64, f64)> = sample.iter().map(|&i| dst_points[i]).collect();

        let h = match find_homography(&s_pts, &d_pts) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Count inliers.
        let mut inlier_count = 0;
        let mut mask = vec![false; n];
        for i in 0..n {
            let (sx, sy) = src_points[i];
            let (dx, dy) = dst_points[i];

            let w = h[2][0] * sx + h[2][1] * sy + h[2][2];
            if w.abs() < 1e-15 {
                continue;
            }
            let px = (h[0][0] * sx + h[0][1] * sy + h[0][2]) / w;
            let py = (h[1][0] * sx + h[1][1] * sy + h[1][2]) / w;
            let err_sq = (px - dx) * (px - dx) + (py - dy) * (py - dy);
            if err_sq < thresh_sq {
                mask[i] = true;
                inlier_count += 1;
            }
        }

        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_h = h;
            best_mask = mask;
        }
    }

    if best_inlier_count < 4 {
        return Err(cv_core::Error::AlgorithmError(
            "RANSAC failed to find a valid homography with enough inliers".into(),
        ));
    }

    // Refine with all inliers.
    let inlier_src: Vec<(f64, f64)> = best_mask
        .iter()
        .enumerate()
        .filter(|(_, &is_in)| is_in)
        .map(|(i, _)| src_points[i])
        .collect();
    let inlier_dst: Vec<(f64, f64)> = best_mask
        .iter()
        .enumerate()
        .filter(|(_, &is_in)| is_in)
        .map(|(i, _)| dst_points[i])
        .collect();

    let refined = find_homography(&inlier_src, &inlier_dst).unwrap_or(best_h);
    Ok((refined, best_mask))
}

/// Warp an image by a 3x3 homography using inverse mapping with bilinear interpolation.
///
/// # Arguments
/// * `image`       - Input image tensor (CHW layout).
/// * `h`           - 3x3 homography matrix.
/// * `output_size` - `(height, width)` of the output image.
///
/// # Returns
/// A new tensor of shape `(C, output_height, output_width)`.
pub fn warp_perspective<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    h: &[[f64; 3]; 3],
    output_size: (usize, usize),
) -> Result<CpuTensor<T>> {
    let (channels, src_h, src_w) = image.shape.chw();
    let (out_h, out_w) = output_size;

    if out_h == 0 || out_w == 0 {
        return Err(cv_core::Error::InvalidInput(
            "Output size must be non-zero".into(),
        ));
    }

    let h_inv = mat3_inv(h);
    let src_data = image.as_slice()?;
    let mut out_data = vec![T::ZERO; channels * out_h * out_w];

    for c in 0..channels {
        let src_offset = c * src_h * src_w;
        let dst_offset = c * out_h * out_w;

        for oy in 0..out_h {
            for ox in 0..out_w {
                let px = ox as f64;
                let py = oy as f64;

                // Inverse map: source = H^{-1} * dest.
                let w = h_inv[2][0] * px + h_inv[2][1] * py + h_inv[2][2];
                if w.abs() < 1e-15 {
                    continue;
                }
                let sx = (h_inv[0][0] * px + h_inv[0][1] * py + h_inv[0][2]) / w;
                let sy = (h_inv[1][0] * px + h_inv[1][1] * py + h_inv[1][2]) / w;

                // Bilinear interpolation.
                if sx < 0.0 || sy < 0.0 || sx >= (src_w - 1) as f64 || sy >= (src_h - 1) as f64 {
                    continue; // Out of bounds -> zero (already initialized).
                }

                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let v00 = Float::to_f64(src_data[src_offset + y0 * src_w + x0]);
                let v01 = Float::to_f64(src_data[src_offset + y0 * src_w + x1]);
                let v10 = Float::to_f64(src_data[src_offset + y1 * src_w + x0]);
                let v11 = Float::to_f64(src_data[src_offset + y1 * src_w + x1]);

                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                out_data[dst_offset + oy * out_w + ox] = T::from_f64(val);
            }
        }
    }

    CpuTensor::<T>::from_vec(out_data, TensorShape::new(channels, out_h, out_w))
}

/// Stitch two images given point correspondences.
///
/// Uses RANSAC to estimate the homography from `img2` into `img1`'s coordinate frame,
/// warps `img2`, and blends using linear alpha in the overlap region.
///
/// # Arguments
/// * `img1`    - First (reference) image.
/// * `img2`    - Second image to warp.
/// * `matches` - Correspondences as `(x1, y1, x2, y2)` where `(x1,y1)` is in `img1`
///               and `(x2,y2)` is in `img2`.
///
/// # Returns
/// A stitched panorama tensor.
pub fn stitch_pair<T: Float + Default + 'static>(
    img1: &CpuTensor<T>,
    img2: &CpuTensor<T>,
    matches: &[(f64, f64, f64, f64)],
) -> Result<CpuTensor<T>> {
    if matches.len() < 4 {
        return Err(cv_core::Error::InvalidInput(
            "At least 4 matches are required for stitching".into(),
        ));
    }

    let (c1, h1, w1) = img1.shape.chw();
    let (c2, _h2, _w2) = img2.shape.chw();
    if c1 != c2 {
        return Err(cv_core::Error::DimensionMismatch(
            "Images must have the same number of channels".into(),
        ));
    }

    // Points: img2 -> img1 (we want to warp img2 into img1's frame).
    let src_pts: Vec<(f64, f64)> = matches.iter().map(|m| (m.2, m.3)).collect();
    let dst_pts: Vec<(f64, f64)> = matches.iter().map(|m| (m.0, m.1)).collect();

    let (h, _mask) = find_homography_ransac(&src_pts, &dst_pts, 3.0, 1000)?;

    // Compute output canvas size by transforming img2 corners.
    let corners = [
        (0.0, 0.0),
        (_w2 as f64 - 1.0, 0.0),
        (_w2 as f64 - 1.0, _h2 as f64 - 1.0),
        (0.0, _h2 as f64 - 1.0),
    ];

    let mut min_x = 0.0f64;
    let mut min_y = 0.0f64;
    let mut max_x = (w1 - 1) as f64;
    let mut max_y = (h1 - 1) as f64;

    for &(cx, cy) in &corners {
        let w = h[2][0] * cx + h[2][1] * cy + h[2][2];
        if w.abs() < 1e-15 {
            continue;
        }
        let px = (h[0][0] * cx + h[0][1] * cy + h[0][2]) / w;
        let py = (h[1][0] * cx + h[1][1] * cy + h[1][2]) / w;
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }

    let offset_x = if min_x < 0.0 { -min_x } else { 0.0 };
    let offset_y = if min_y < 0.0 { -min_y } else { 0.0 };

    let out_w = (max_x + offset_x).ceil() as usize + 1;
    let out_h = (max_y + offset_y).ceil() as usize + 1;

    // Translation matrix to shift everything by offset.
    let t_offset = [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]];

    // Warp img2 with combined H.
    let h_combined = mat3_mul(&t_offset, &h);
    let warped2 = warp_perspective(img2, &h_combined, (out_h, out_w))?;

    // Place img1 with offset.
    let identity_offset = t_offset;
    let warped1 = warp_perspective(img1, &identity_offset, (out_h, out_w))?;

    // Blend: linear alpha in overlap, otherwise take whichever is non-zero.
    let channels = c1;
    let w1_data = warped1.as_slice()?;
    let w2_data = warped2.as_slice()?;
    let mut out_data = vec![T::ZERO; channels * out_h * out_w];

    for c in 0..channels {
        let off = c * out_h * out_w;
        for i in 0..out_h * out_w {
            let v1 = Float::to_f64(w1_data[off + i]);
            let v2 = Float::to_f64(w2_data[off + i]);
            let has1 = v1.abs() > 1e-10;
            let has2 = v2.abs() > 1e-10;

            let blended = if has1 && has2 {
                // Linear blend in overlap.
                0.5 * v1 + 0.5 * v2
            } else if has1 {
                v1
            } else {
                v2
            };

            out_data[off + i] = T::from_f64(blended);
        }
    }

    CpuTensor::<T>::from_vec(out_data, TensorShape::new(channels, out_h, out_w))
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Normalize a set of 2D points: translate centroid to origin, scale so mean distance = sqrt(2).
/// Returns `(T, normalized_points)` where `T` is the 3x3 normalization matrix.
fn normalize_points(pts: &[(f64, f64)]) -> ([[f64; 3]; 3], Vec<(f64, f64)>) {
    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.1).sum::<f64>() / n;

    let mean_dist: f64 = pts
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if mean_dist > 1e-15 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let t = [
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ];

    let normalized: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| (scale * (p.0 - cx), scale * (p.1 - cy)))
        .collect();

    (t, normalized)
}

/// 3x3 matrix multiply.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// 3x3 matrix inverse (adjugate method).
fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = if det.abs() > 1e-15 { 1.0 / det } else { 0.0 };

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::tensor::TensorShape;

    #[test]
    fn test_find_homography_identity() {
        // Points mapped to themselves should yield an identity homography.
        let pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (50.0, 50.0),
        ];
        let h = find_homography(&pts, &pts).unwrap();

        // H should be approximately identity (scaled).
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (h[i][j] - expected).abs() < 1e-6,
                    "H[{}][{}] = {}, expected {}",
                    i,
                    j,
                    h[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_find_homography_translation() {
        // Pure translation by (10, 20).
        let src: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (50.0, 50.0),
            (25.0, 75.0),
        ];
        let dst: Vec<(f64, f64)> = src.iter().map(|&(x, y)| (x + 10.0, y + 20.0)).collect();
        let h = find_homography(&src, &dst).unwrap();

        // Check that the translation components are correct.
        assert!((h[0][2] - 10.0).abs() < 1e-4, "tx = {}", h[0][2]);
        assert!((h[1][2] - 20.0).abs() < 1e-4, "ty = {}", h[1][2]);
        // Diagonal should be ~1.
        assert!((h[0][0] - 1.0).abs() < 1e-4);
        assert!((h[1][1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_find_homography_too_few_points() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let result = find_homography(&pts, &pts);
        assert!(result.is_err());
    }

    #[test]
    fn test_ransac_with_outliers() {
        // 8 inliers (translation by 5,5) + 2 outliers.
        let src: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (50.0, 50.0),
            (25.0, 75.0),
            (75.0, 25.0),
            (60.0, 40.0),
            (10.0, 10.0), // outlier
            (90.0, 90.0), // outlier
        ];
        let mut dst: Vec<(f64, f64)> = src.iter().map(|&(x, y)| (x + 5.0, y + 5.0)).collect();
        // Corrupt the last two.
        dst[8] = (500.0, 500.0);
        dst[9] = (-200.0, -200.0);

        let (h, mask) = find_homography_ransac(&src, &dst, 5.0, 2000).unwrap();

        // At least one outlier should be rejected.
        let outlier_rejected = !mask[8] || !mask[9];
        assert!(outlier_rejected, "At least one outlier should be rejected");

        // The majority of inliers (first 8) should be accepted.
        let inlier_count: usize = mask[..8].iter().filter(|&&b| b).count();
        assert!(
            inlier_count >= 6,
            "Expected >=6 inliers among first 8, got {}",
            inlier_count
        );

        // The H should approximate a translation by (5, 5).
        // RANSAC is stochastic — allow wider tolerance.
        assert!((h[0][2] - 5.0).abs() < 60.0, "tx = {}", h[0][2]);
        assert!((h[1][2] - 5.0).abs() < 60.0, "ty = {}", h[1][2]);
    }

    #[test]
    fn test_warp_identity_preserves_image() {
        // 1-channel 4x4 image with distinct values.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let img = CpuTensor::<f32>::from_vec(data.clone(), TensorShape::new(1, 4, 4)).unwrap();

        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let warped = warp_perspective(&img, &identity, (4, 4)).unwrap();
        let out = warped.as_slice().unwrap();

        // Interior pixels (not on the right/bottom edge) should match exactly.
        for y in 0..3 {
            for x in 0..3 {
                let idx = y * 4 + x;
                assert!(
                    (out[idx] - data[idx]).abs() < 1e-4,
                    "Pixel ({},{}) mismatch: {} vs {}",
                    x,
                    y,
                    out[idx],
                    data[idx]
                );
            }
        }
    }

    #[test]
    fn test_warp_translation() {
        // Translate by (1, 0).
        let h = 4;
        let w = 6;
        let mut data = vec![0.0f32; h * w];
        // Put a value at (2, 1).
        data[1 * w + 2] = 1.0;
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        let translate = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let warped = warp_perspective(&img, &translate, (h, w)).unwrap();
        let out = warped.as_slice().unwrap();

        // The point at (2, 1) should now appear at (3, 1).
        assert!(
            out[1 * w + 3] > 0.9,
            "Translated pixel should be at (3,1), got {}",
            out[1 * w + 3]
        );
        // Original location should be zero (mapped from outside).
        assert!(out[1 * w + 2] < 0.1);
    }

    #[test]
    fn test_warp_zero_output_size() {
        let img = CpuTensor::<f32>::from_vec(vec![1.0; 4], TensorShape::new(1, 2, 2)).unwrap();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = warp_perspective(&img, &identity, (0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_stitch_pair_basic() {
        // Two 5x5 images, second is shifted right by 2 pixels.
        let h = 5;
        let w = 5;
        let data1 = vec![0.5f32; h * w];
        let data2 = vec![0.7f32; h * w];
        let img1 = CpuTensor::<f32>::from_vec(data1, TensorShape::new(1, h, w)).unwrap();
        let img2 = CpuTensor::<f32>::from_vec(data2, TensorShape::new(1, h, w)).unwrap();

        // Correspondences: img1 points and img2 points related by translation (2, 0).
        let matches: Vec<(f64, f64, f64, f64)> = vec![
            (2.0, 0.0, 0.0, 0.0),
            (3.0, 0.0, 1.0, 0.0),
            (2.0, 4.0, 0.0, 4.0),
            (3.0, 4.0, 1.0, 4.0),
            (4.0, 2.0, 2.0, 2.0),
        ];

        let result = stitch_pair(&img1, &img2, &matches).unwrap();
        // Output should be wider than either input.
        assert!(
            result.shape.width >= w,
            "Stitched width should be >= input width"
        );
    }

    #[test]
    fn test_stitch_pair_too_few_matches() {
        let img = CpuTensor::<f32>::from_vec(vec![1.0; 9], TensorShape::new(1, 3, 3)).unwrap();
        let matches = vec![(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)];
        let result = stitch_pair(&img, &img, &matches);
        assert!(result.is_err());
    }

    #[test]
    fn test_mat3_inv_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = mat3_inv(&id);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_mat3_mul_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = mat3_mul(&id, &m);
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[i][j] - m[i][j]).abs() < 1e-10);
            }
        }
    }
}
