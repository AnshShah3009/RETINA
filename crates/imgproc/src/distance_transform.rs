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
///
/// # Errors
/// Returns an error if the input is not single-channel or has a zero spatial
/// dimension. Multi-channel input used to be accepted and silently reduced to
/// its first plane (the whole buffer was indexed as `y * width + x`), so every
/// channel past the first was ignored while the output still claimed
/// `channels: 1`.
pub fn distance_transform<T: Float>(
    binary: &CpuTensor<T>,
    dist_type: DistanceType,
) -> crate::Result<CpuTensor<f32>> {
    let shape = binary.shape;
    let height = shape.height;
    let width = shape.width;

    if shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(format!(
            "Distance transform requires a single-channel image, got {} channels. \
             Only channel 0 would have been processed before.",
            shape.channels
        )));
    }
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
///
/// When several background pixels are equally near - which is common, e.g. two
/// background pixels straddling a foreground pixel - the label is the one with
/// the **smallest flat index**, i.e. the first in row-major order. That is the
/// tie-break the former brute-force scan produced (it walked the background
/// pixels in row-major order and only replaced the incumbent on a strict `<`),
/// and it is preserved so that labels do not change.
///
/// # Complexity
/// `O(height * width)` for every metric, instead of the former
/// `O(height * width * background_count)` scan. See [`chamfer_labels`] (L1,
/// Chessboard) and [`euclidean_labels`] (L2) for how each pixel's label is
/// derived.
///
/// # Errors
/// Same as [`distance_transform`]: the input must be single-channel with
/// non-zero spatial dimensions.
pub fn distance_transform_with_labels<T: Float>(
    binary: &CpuTensor<T>,
    dist_type: DistanceType,
) -> crate::Result<(CpuTensor<f32>, CpuTensor<i32>)> {
    let shape = binary.shape;
    let height = shape.height;
    let width = shape.width;

    if shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(format!(
            "Distance transform requires a single-channel image, got {} channels. \
             Only channel 0 would have been processed before.",
            shape.channels
        )));
    }
    if height == 0 || width == 0 {
        return Err(cv_core::Error::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let src = binary.as_slice()?;
    let n = height * width;

    // Compute distance map.
    let dist = distance_transform(binary, dist_type)?;
    let out_shape = TensorShape::new(1, height, width);

    if !src.iter().any(|v| v.to_f32() == Some(0.0)) {
        // No background pixels: labels are all -1.
        let labels = vec![-1i32; n];
        let label_tensor = CpuTensor::from_vec(labels, out_shape)?;
        return Ok((dist, label_tensor));
    }

    // The label of each pixel is a single-source nearest-neighbour query, which
    // is separable (or, for L1/Chessboard, chamfer-propagatable) in `O(n)`.
    let labels = labels_nearest_background(src, width, height, dist_type);

    let label_tensor = CpuTensor::from_vec(labels, out_shape)?;
    Ok((dist, label_tensor))
}

// ---------------------------------------------------------------------------
// Nearest-background label map
// ---------------------------------------------------------------------------
//
// All three label passes below minimise, over the background pixels, the pair
// `(distance, flat_index)` compared lexicographically. That is exactly the
// brute-force scan this replaced - it kept the first pixel it saw on a strict
// improvement while walking the background pixels in row-major order - so the
// label on a tie is the *smallest flat index*, i.e. the first in row-major
// order. Because a candidate pair is encoded as the single integer
// `distance * n + flat_index` (with `flat_index < n`), adding `n` to a key adds
// one unit of distance and leaves the index alone; comparing keys is then the
// same as comparing the pairs lexicographically.

/// Index of the nearest background pixel for every pixel, ties broken towards
/// the smallest flat index. `O(width * height)` for all metrics.
fn labels_nearest_background<T: Float>(
    src: &[T],
    width: usize,
    height: usize,
    dist_type: DistanceType,
) -> Vec<i32> {
    match dist_type {
        DistanceType::L1 => chamfer_labels(src, width, height, false),
        DistanceType::Chessboard => chamfer_labels(src, width, height, true),
        DistanceType::L2 => euclidean_labels(src, width, height),
    }
}

/// L1 / Chessboard labels by two raster scans of a chamfer propagation.
///
/// Each pixel holds the key of the best `(distance, label)` pair found so far
/// and relaxes its neighbours by one unit, so a pixel ends up with the
/// lexicographically smallest pair over all raster-order-compatible shortest
/// paths. Any shortest path in these metrics can be reordered into the
/// forward-mask steps followed by the backward-mask steps (the steps of a
/// monotone path all point into one or two neighbouring quadrants), so the two
/// scans reach the true minimum, and by the argument above the label is the
/// brute-force one.
fn chamfer_labels<T: Float>(src: &[T], width: usize, height: usize, diagonal: bool) -> Vec<i32> {
    let n = width * height;
    let step = n as i64; // one unit of distance
    const NONE: i64 = i64::MAX;

    let mut key = vec![NONE; n];
    for (i, k) in key.iter_mut().enumerate() {
        if src[i].to_f32() == 0.0 {
            *k = i as i64;
        }
    }

    let relax = |best: i64, from: i64| -> i64 {
        if from == NONE {
            best
        } else {
            best.min(from + step)
        }
    };

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut best = key[idx];
            if y > 0 {
                best = relax(best, key[idx - width]);
                if diagonal {
                    if x > 0 {
                        best = relax(best, key[idx - width - 1]);
                    }
                    if x + 1 < width {
                        best = relax(best, key[idx - width + 1]);
                    }
                }
            }
            if x > 0 {
                best = relax(best, key[idx - 1]);
            }
            key[idx] = best;
        }
    }

    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            let mut best = key[idx];
            if y + 1 < height {
                best = relax(best, key[idx + width]);
                if diagonal {
                    if x > 0 {
                        best = relax(best, key[idx + width - 1]);
                    }
                    if x + 1 < width {
                        best = relax(best, key[idx + width + 1]);
                    }
                }
            }
            if x + 1 < width {
                best = relax(best, key[idx + 1]);
            }
            key[idx] = best;
        }
    }

    key.into_iter()
        .map(|k| if k == NONE { -1 } else { (k % step) as i32 })
        .collect()
}

/// A real boundary between two parabolas, or one of the two infinities.
/// `Rat` keeps an exact numerator/positive denominator so neighbouring
/// intersections can be compared without rounding.
#[derive(Clone, Copy)]
enum Bound {
    NegInf,
    PosInf,
    Rat(i128, i128),
}

impl Bound {
    /// `self > other`
    fn gt(self, other: Bound) -> bool {
        match (self, other) {
            (Bound::PosInf, _) => !matches!(other, Bound::PosInf),
            (Bound::NegInf, _) => false,
            (_, Bound::PosInf) => false,
            (_, Bound::NegInf) => true,
            (Bound::Rat(an, ad), Bound::Rat(bn, bd)) => an * bd > bn * ad,
        }
    }

    /// `self < p` for an integer `p`.
    fn lt_int(self, p: i128) -> bool {
        match self {
            Bound::NegInf => true,
            Bound::PosInf => false,
            Bound::Rat(nm, den) => nm < p * den,
        }
    }
}

/// L2 labels: the exact separable Euclidean transform, with the argmin carried
/// along instead of the distance.
///
/// The first pass keeps, per column, the squared vertical distance to the
/// nearest background pixel in that column and that pixel's flat index (ties
/// in a column go to the smaller row, hence the smaller index). The column
/// values are encoded as keys `sq * n + label` and combined along each row as
/// the lower envelope of the parabolas `key[x] + n * (p - x)^2` - the
/// Felzenszwalb-Huttenlocher transform run on keys rather than on distances.
/// Minimising that key is minimising `(squared distance, label)`, and because a
/// column contributes a single, distinct label, the parabolas never tie at an
/// integer position at all: the returned argmin is the brute-force nearest
/// background pixel with its smallest-index tie-break.
fn euclidean_labels<T: Float>(src: &[T], width: usize, height: usize) -> Vec<i32> {
    let n = width * height;
    let big = n as i64; // scale that makes a key lexicographic in (dist, label)

    // --- Pass 1: vertical squared distance and label, per column. ---
    // Column-major so that each column is a contiguous, independently writable
    // slice and the columns can run in parallel (as the old pixel scan did).
    let mut vert_sq = vec![0i64; n];
    let mut vert_lab = vec![-1i32; n];
    vert_sq
        .par_chunks_mut(height)
        .zip(vert_lab.par_chunks_mut(height))
        .enumerate()
        .for_each(|(x, (sq_col, lab_col))| {
            let mut above: Option<usize> = None;
            for y in 0..height {
                let i = y * width + x;
                if src[i].to_f32() == 0.0 {
                    above = Some(y);
                    sq_col[y] = 0;
                    lab_col[y] = i as i32;
                } else if let Some(y0) = above {
                    let dy = (y - y0) as i64;
                    sq_col[y] = dy * dy;
                    lab_col[y] = (y0 * width + x) as i32;
                }
            }
            let mut below: Option<usize> = None;
            for y in (0..height).rev() {
                let i = y * width + x;
                if src[i].to_f32() == 0.0 {
                    below = Some(y);
                    continue;
                }
                if let Some(y0) = below {
                    let dy = y as i64 - y0 as i64;
                    let d = dy * dy;
                    // Strict: on a tie keep the pixel above, i.e. the
                    // smaller flat index.
                    if lab_col[y] < 0 || d < sq_col[y] {
                        sq_col[y] = d;
                        lab_col[y] = (y0 * width + x) as i32;
                    }
                }
            }
        });

    // --- Pass 2: lower envelope of the key parabolas, per row. ---
    // `i128` because a key is up to `n^3`, which would overflow `i64` past
    // roughly 2000x2000. Rows are independent, so this runs in parallel too.
    let mut labels = vec![-1i32; n];
    labels
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_labels)| {
            // A column contributes one parabola per row, at key
            // `sq*n + label`, so its vertex cost is `key + n*(p - u)^2`.
            let mut cand: Vec<usize> = Vec::with_capacity(width);
            let mut fkey: Vec<i128> = Vec::with_capacity(width);
            for x in 0..width {
                let v = x * height + y;
                if vert_lab[v] >= 0 {
                    cand.push(x);
                    fkey.push(vert_sq[v] as i128 * big as i128 + vert_lab[v] as i128);
                }
            }
            if cand.is_empty() {
                // Cannot happen: the caller checked for a background pixel.
                return;
            }
            let m = cand.len();
            let mut vpos = vec![0usize; m];
            let mut zs = vec![Bound::NegInf; m + 1];
            zs[1] = Bound::PosInf;
            let mut k = 0usize;
            for q in 1..m {
                loop {
                    // Boundary between the parabola at vpos[k] and the one at q:
                    //   s = (f_b - f_a + big*(u_b - u_a)*(u_a + u_b))
                    //       / (2*big*(u_b - u_a))
                    let (ua, ub) = (cand[vpos[k]] as i128, cand[q] as i128);
                    let (fa, fb) = (fkey[vpos[k]], fkey[q]);
                    let den = 2 * (big as i128) * (ub - ua); // > 0, ua < ub
                    let num = 2 * (fb - fa) + den * (ua + ub);
                    let s = Bound::Rat(num, 2 * den);
                    if s.gt(zs[k]) {
                        k += 1;
                        vpos[k] = q;
                        zs[k] = s;
                        zs[k + 1] = Bound::PosInf;
                        break;
                    }
                    if k == 0 {
                        vpos[0] = q;
                        zs[0] = Bound::NegInf;
                        zs[1] = Bound::PosInf;
                        break;
                    }
                    k -= 1;
                }
            }

            let mut ki = 0usize;
            for (x, slot) in row_labels.iter_mut().enumerate() {
                while zs[ki + 1].lt_int(x as i128) {
                    ki += 1;
                }
                let u = cand[vpos[ki]];
                *slot = vert_lab[u * height + y];
            }
        });

    labels
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
        assert_eq!(d[5 + 2], 1.0); // above
        assert_eq!(d[3 * 5 + 2], 1.0); // below
        // Diagonal should be 2
        assert_eq!(d[5 + 1], 2.0);
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
        assert!((d[5 + 2] - 1.0).abs() < 0.01);
        // Diagonal neighbor = sqrt(2) ~ 1.414
        assert!((d[5 + 1] - std::f32::consts::SQRT_2).abs() < 0.01);
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
        assert_eq!(d[5 + 1], 1.0);
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
