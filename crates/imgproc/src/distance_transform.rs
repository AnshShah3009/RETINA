//! Distance transform: computes the distance from each pixel to the nearest zero (background) pixel.
//!
//! Supports L1 (Manhattan), L2 (Euclidean via Felzenszwalb-Huttenlocher), and Chessboard
//! (Chebyshev) distance metrics. Operates on `CpuTensor<T>` where zero values are background.

use cv_core::{CpuTensor, Float, TensorShape};
use rayon::prelude::*;

/// Distance metric for the distance transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    /// Manhattan distance (|dx| + |dy|).
    L1,
    /// Euclidean distance (approximate, Felzenszwalb-Huttenlocher).
    L2,
    /// Chebyshev distance (max(|dx|, |dy|)).
    Chessboard,
}

/// Compute the distance transform of a binary image.
///
/// Each pixel in the output holds the distance to the nearest zero-valued pixel
/// in the input. Non-zero pixels are treated as foreground.
///
/// # Arguments
/// * `binary` - Input tensor with shape (1, H, W). Zero = background.
/// * `dist_type` - Distance metric to use.
///
/// # Returns
/// A `CpuTensor<f32>` of the same spatial dimensions containing distances.
pub fn distance_transform<T: Float>(
    binary: &CpuTensor<T>,
    dist_type: DistanceType,
) -> crate::Result<CpuTensor<f32>> {
    let shape = binary.shape;
    let height = shape.height;
    let width = shape.width;

    if height == 0 || width == 0 {
        return Err(cv_core::Error::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let src = binary.as_slice()?;

    match dist_type {
        DistanceType::L1 => distance_transform_l1(src, width, height),
        DistanceType::Chessboard => distance_transform_chessboard(src, width, height),
        DistanceType::L2 => distance_transform_l2(src, width, height),
    }
}

/// Distance transform that also returns a label map indicating the index of the
/// nearest background pixel for each foreground pixel.
///
/// # Returns
/// `(distance_map, label_map)` where `label_map[i]` is the flat index of the
/// nearest zero pixel to pixel `i`. Background pixels map to themselves.
pub fn distance_transform_with_labels<T: Float>(
    binary: &CpuTensor<T>,
    dist_type: DistanceType,
) -> crate::Result<(CpuTensor<f32>, CpuTensor<i32>)> {
    let shape = binary.shape;
    let height = shape.height;
    let width = shape.width;

    if height == 0 || width == 0 {
        return Err(cv_core::Error::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let src = binary.as_slice()?;
    let n = height * width;

    // Build list of background pixel coordinates.
    let mut bg_coords: Vec<(usize, usize)> = Vec::new();
    let mut bg_indices: Vec<i32> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if src[y * width + x].to_f32() == 0.0 {
                bg_coords.push((x, y));
                bg_indices.push((y * width + x) as i32);
            }
        }
    }

    // Compute distance map.
    let dist = distance_transform(binary, dist_type)?;
    // For each pixel, find the nearest background pixel by brute-force scan of bg list.
    // This is O(n * bg_count). For large images a Voronoi approach would be better,
    // but this is correct and sufficient for moderate sizes.
    let out_shape = TensorShape::new(1, height, width);

    if bg_coords.is_empty() {
        // No background pixels: labels are all -1.
        let labels = vec![-1i32; n];
        let label_tensor = CpuTensor::from_vec(labels, out_shape)?;
        return Ok((dist, label_tensor));
    }

    let labels: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let px = idx % width;
            let py = idx / width;
            if src[idx].to_f32() == 0.0 {
                return idx as i32;
            }
            let mut best_dist_sq = f64::MAX;
            let mut best_label = -1i32;
            for (i, &(bx, by)) in bg_coords.iter().enumerate() {
                let dx = px as f64 - bx as f64;
                let dy = py as f64 - by as f64;
                let d = dx * dx + dy * dy;
                if d < best_dist_sq {
                    best_dist_sq = d;
                    best_label = bg_indices[i];
                }
            }
            best_label
        })
        .collect();

    let label_tensor = CpuTensor::from_vec(labels, out_shape)?;
    Ok((dist, label_tensor))
}

// ---------------------------------------------------------------------------
// L1 distance (two-pass Rosenfeld-Pfaltz)
// ---------------------------------------------------------------------------
fn distance_transform_l1<T: Float>(
    src: &[T],
    width: usize,
    height: usize,
) -> crate::Result<CpuTensor<f32>> {
    let n = width * height;
    let inf = (width + height) as f32;
    let mut dist = vec![0.0f32; n];

    // Initialize: background=0, foreground=INF
    for i in 0..n {
        dist[i] = if src[i].to_f32() == 0.0 { 0.0 } else { inf };
    }

    // Forward pass (top-left to bottom-right)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if y > 0 {
                dist[idx] = dist[idx].min(dist[(y - 1) * width + x] + 1.0);
            }
            if x > 0 {
                dist[idx] = dist[idx].min(dist[y * width + (x - 1)] + 1.0);
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            if y + 1 < height {
                dist[idx] = dist[idx].min(dist[(y + 1) * width + x] + 1.0);
            }
            if x + 1 < width {
                dist[idx] = dist[idx].min(dist[y * width + (x + 1)] + 1.0);
            }
        }
    }

    let shape = TensorShape::new(1, height, width);
    CpuTensor::from_vec(dist, shape)
}

// ---------------------------------------------------------------------------
// Chessboard distance (two-pass)
// ---------------------------------------------------------------------------
fn distance_transform_chessboard<T: Float>(
    src: &[T],
    width: usize,
    height: usize,
) -> crate::Result<CpuTensor<f32>> {
    let n = width * height;
    let inf = (width + height) as f32;
    let mut dist = vec![0.0f32; n];

    for i in 0..n {
        dist[i] = if src[i].to_f32() == 0.0 { 0.0 } else { inf };
    }

    // Forward pass
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if y > 0 {
                dist[idx] = dist[idx].min(dist[(y - 1) * width + x] + 1.0);
                if x > 0 {
                    dist[idx] = dist[idx].min(dist[(y - 1) * width + (x - 1)] + 1.0);
                }
                if x + 1 < width {
                    dist[idx] = dist[idx].min(dist[(y - 1) * width + (x + 1)] + 1.0);
                }
            }
            if x > 0 {
                dist[idx] = dist[idx].min(dist[y * width + (x - 1)] + 1.0);
            }
        }
    }

    // Backward pass
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            if y + 1 < height {
                dist[idx] = dist[idx].min(dist[(y + 1) * width + x] + 1.0);
                if x > 0 {
                    dist[idx] = dist[idx].min(dist[(y + 1) * width + (x - 1)] + 1.0);
                }
                if x + 1 < width {
                    dist[idx] = dist[idx].min(dist[(y + 1) * width + (x + 1)] + 1.0);
                }
            }
            if x + 1 < width {
                dist[idx] = dist[idx].min(dist[y * width + (x + 1)] + 1.0);
            }
        }
    }

    let shape = TensorShape::new(1, height, width);
    CpuTensor::from_vec(dist, shape)
}

// ---------------------------------------------------------------------------
// L2 distance (Felzenszwalb-Huttenlocher 1D transform, applied separably)
// ---------------------------------------------------------------------------
fn distance_transform_l2<T: Float>(
    src: &[T],
    width: usize,
    height: usize,
) -> crate::Result<CpuTensor<f32>> {
    let n = width * height;
    let inf = ((width * width + height * height) as f32) + 1.0;

    // Work buffer: squared distances.
    let mut d = vec![0.0f32; n];
    for i in 0..n {
        d[i] = if src[i].to_f32() == 0.0 { 0.0 } else { inf };
    }

    // 1D squared-distance transform along columns (vertical).
    let max_dim = width.max(height);
    // Process each column.
    for x in 0..width {
        let mut col = vec![0.0f32; height];
        for y in 0..height {
            col[y] = d[y * width + x];
        }
        let result = dt_1d(&col, max_dim);
        for y in 0..height {
            d[y * width + x] = result[y];
        }
    }

    // 1D squared-distance transform along rows (horizontal).
    for y in 0..height {
        let mut row = vec![0.0f32; width];
        for x in 0..width {
            row[x] = d[y * width + x];
        }
        let result = dt_1d(&row, max_dim);
        for x in 0..width {
            d[y * width + x] = result[x];
        }
    }

    // Square root to get Euclidean distance.
    d.par_iter_mut().for_each(|v| *v = v.sqrt());

    let shape = TensorShape::new(1, height, width);
    CpuTensor::from_vec(d, shape)
}

/// 1D squared Euclidean distance transform (Felzenszwalb-Huttenlocher).
/// `f[q]` is the initial squared distance for position q.
/// Returns the lower envelope of parabolas: `min_q (f[q] + (p-q)^2)` for each p.
#[allow(clippy::needless_range_loop)]
fn dt_1d(f: &[f32], _max_dim: usize) -> Vec<f32> {
    let n = f.len();
    if n == 0 {
        return Vec::new();
    }
    let mut d = vec![0.0f32; n];
    let mut v = vec![0usize; n]; // locations of parabolas
    let mut z = vec![0.0f32; n + 1]; // boundaries between parabolas
    let mut k = 0usize; // number of parabolas in lower envelope

    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        loop {
            let s = intersection(f, v[k], q);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    let mut ki = 0;
    for q in 0..n {
        while z[ki + 1] < q as f32 {
            ki += 1;
        }
        let dq = (q as f32) - (v[ki] as f32);
        d[q] = dq * dq + f[v[ki]];
    }

    d
}

#[inline]
fn intersection(f: &[f32], p: usize, q: usize) -> f32 {
    let fp = f[p];
    let fq = f[q];
    let p = p as f32;
    let q = q as f32;
    ((fq + q * q) - (fp + p * p)) / (2.0 * (q - p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binary(data: Vec<f32>, width: usize, height: usize) -> CpuTensor<f32> {
        CpuTensor::from_vec(data, TensorShape::new(1, height, width)).unwrap()
    }

    #[test]
    fn l1_simple_shape() {
        // 5x5 image with a single zero pixel at (2,2)
        let mut data = vec![1.0f32; 25];
        data[2 * 5 + 2] = 0.0; // center pixel is background
        let binary = make_binary(data, 5, 5);
        let dist = distance_transform(&binary, DistanceType::L1).unwrap();
        let d = dist.as_slice().unwrap();

        // Center pixel should be 0
        assert_eq!(d[2 * 5 + 2], 0.0);
        // Neighbors should be 1
        assert_eq!(d[2 * 5 + 1], 1.0); // left
        assert_eq!(d[2 * 5 + 3], 1.0); // right
        assert_eq!(d[1 * 5 + 2], 1.0); // above
        assert_eq!(d[3 * 5 + 2], 1.0); // below
                                       // Diagonal should be 2
        assert_eq!(d[1 * 5 + 1], 2.0);
        assert_eq!(d[3 * 5 + 3], 2.0);
        // Corner (0,0) should be 4
        assert_eq!(d[0], 4.0);
    }

    #[test]
    fn l2_euclidean_distances() {
        // 5x5 with zero at center
        let mut data = vec![1.0f32; 25];
        data[2 * 5 + 2] = 0.0;
        let binary = make_binary(data, 5, 5);
        let dist = distance_transform(&binary, DistanceType::L2).unwrap();
        let d = dist.as_slice().unwrap();

        assert_eq!(d[2 * 5 + 2], 0.0);
        // Cardinal neighbors = 1.0
        assert!((d[2 * 5 + 1] - 1.0).abs() < 0.01);
        assert!((d[1 * 5 + 2] - 1.0).abs() < 0.01);
        // Diagonal neighbor = sqrt(2) ~ 1.414
        assert!((d[1 * 5 + 1] - std::f32::consts::SQRT_2).abs() < 0.01);
        // Corner (0,0) = sqrt(8) ~ 2.828
        assert!((d[0] - (8.0f32).sqrt()).abs() < 0.01);
    }

    #[test]
    fn chessboard_distances() {
        // 5x5 with zero at center
        let mut data = vec![1.0f32; 25];
        data[2 * 5 + 2] = 0.0;
        let binary = make_binary(data, 5, 5);
        let dist = distance_transform(&binary, DistanceType::Chessboard).unwrap();
        let d = dist.as_slice().unwrap();

        assert_eq!(d[2 * 5 + 2], 0.0);
        // Cardinal neighbors = 1
        assert_eq!(d[2 * 5 + 1], 1.0);
        // Diagonal neighbor = 1 (chessboard)
        assert_eq!(d[1 * 5 + 1], 1.0);
        // Corner (0,0) = 2
        assert_eq!(d[0], 2.0);
    }

    #[test]
    fn all_background_returns_zeros() {
        let data = vec![0.0f32; 9];
        let binary = make_binary(data, 3, 3);
        let dist = distance_transform(&binary, DistanceType::L1).unwrap();
        let d = dist.as_slice().unwrap();
        assert!(d.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn all_foreground_l1() {
        // No background pixels: distances should be max (based on two-pass propagation limit)
        let data = vec![1.0f32; 9];
        let binary = make_binary(data, 3, 3);
        let dist = distance_transform(&binary, DistanceType::L1).unwrap();
        let d = dist.as_slice().unwrap();
        // All pixels should have distances >= 1 since there is no zero pixel
        assert!(d.iter().all(|&v| v >= 1.0));
    }

    #[test]
    fn distance_with_labels_basic() {
        // 3x3 with zero at (0,0)
        let mut data = vec![1.0f32; 9];
        data[0] = 0.0;
        let binary = make_binary(data, 3, 3);
        let (dist, labels) = distance_transform_with_labels(&binary, DistanceType::L1).unwrap();
        let d = dist.as_slice().unwrap();
        let l = labels.as_slice().unwrap();

        // (0,0) is background -> distance=0, label=0
        assert_eq!(d[0], 0.0);
        assert_eq!(l[0], 0);
        // All other pixels should point back to index 0 (only bg pixel)
        for i in 1..9 {
            assert_eq!(l[i], 0);
        }
    }

    #[test]
    fn l1_row_image() {
        // Single row: [0, 1, 1, 1, 0]
        let data = vec![0.0f32, 1.0, 1.0, 1.0, 0.0];
        let binary = make_binary(data, 5, 1);
        let dist = distance_transform(&binary, DistanceType::L1).unwrap();
        let d = dist.as_slice().unwrap();
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 2.0);
        assert_eq!(d[3], 1.0);
        assert_eq!(d[4], 0.0);
    }

    #[test]
    fn l2_row_image() {
        let data = vec![0.0f32, 1.0, 1.0, 1.0, 0.0];
        let binary = make_binary(data, 5, 1);
        let dist = distance_transform(&binary, DistanceType::L2).unwrap();
        let d = dist.as_slice().unwrap();
        assert!((d[0] - 0.0).abs() < 0.01);
        assert!((d[1] - 1.0).abs() < 0.01);
        assert!((d[2] - 2.0).abs() < 0.01);
        assert!((d[3] - 1.0).abs() < 0.01);
        assert!((d[4] - 0.0).abs() < 0.01);
    }

    #[test]
    fn empty_image_error() {
        let binary = CpuTensor::from_vec(Vec::<f32>::new(), TensorShape::new(1, 0, 0)).unwrap();
        assert!(distance_transform(&binary, DistanceType::L1).is_err());
    }
}
