//! Image segmentation algorithms: watershed, GrabCut, and flood fill.

use cv_core::{CpuTensor, Float};
use std::collections::BinaryHeap;
use std::collections::VecDeque;

type Result<T> = cv_core::Result<T>;

// ---------------------------------------------------------------------------
// Watershed
// ---------------------------------------------------------------------------

/// A pixel entry for the watershed priority queue.
/// Lower gradient = higher priority, so we reverse the ordering.
#[derive(Debug, Clone)]
struct WatershedPixel {
    gradient: f64,
    row: usize,
    col: usize,
}

impl PartialEq for WatershedPixel {
    fn eq(&self, other: &Self) -> bool {
        self.gradient == other.gradient
    }
}

impl Eq for WatershedPixel {}

impl PartialOrd for WatershedPixel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WatershedPixel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order: smallest gradient has highest priority
        other
            .gradient
            .partial_cmp(&self.gradient)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Watershed segmentation via priority-queue region growing.
///
/// # Arguments
/// * `image` - 3-channel color image in CHW layout (channels=3).
/// * `markers` - Integer marker tensor (channels=1, same H x W as image).
///   - Positive values are seed labels.
///   - `0` means unknown / unlabeled.
///   - After the call, boundary pixels are set to `-1`.
///
/// # Errors
/// Returns an error if the image is not 3-channel, the markers are not 1-channel,
/// or if the spatial dimensions do not match.
pub fn watershed<T: Float>(image: &CpuTensor<T>, markers: &mut CpuTensor<i32>) -> Result<()> {
    let (c, h, w) = image.shape.chw();
    if c != 3 {
        return Err(cv_core::Error::InvalidInput(
            "Watershed requires a 3-channel image".into(),
        ));
    }
    if markers.shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Markers must be a 1-channel tensor".into(),
        ));
    }
    if markers.shape.height != h || markers.shape.width != w {
        return Err(cv_core::Error::DimensionMismatch(format!(
            "Image size {}x{} does not match markers size {}x{}",
            w, h, markers.shape.width, markers.shape.height
        )));
    }

    // Pre-compute gradient magnitude for every pixel (sum of squared differences
    // between neighbours across all 3 channels, approximated by Sobel-like
    // forward differences).
    let img_data = image.as_slice()?;
    let hw = h * w;

    let gradient = compute_gradient_magnitude::<T>(img_data, h, w, hw);

    // 4-connected neighbours
    let offsets: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    let marker_data = markers.as_mut_slice()?;

    // Seed the priority queue with every unlabeled pixel that is adjacent to a
    // labeled seed pixel.
    let mut heap = BinaryHeap::new();
    let mut in_queue = vec![false; hw];

    for r in 0..h {
        for col in 0..w {
            let idx = r * w + col;
            if marker_data[idx] > 0 {
                // This pixel is a seed. Push unlabeled neighbours.
                for &(dr, dc) in &offsets {
                    let nr = r as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let nidx = nr as usize * w + nc as usize;
                        if marker_data[nidx] == 0 && !in_queue[nidx] {
                            in_queue[nidx] = true;
                            heap.push(WatershedPixel {
                                gradient: gradient[nidx],
                                row: nr as usize,
                                col: nc as usize,
                            });
                        }
                    }
                }
            }
        }
    }

    // Region growing
    while let Some(px) = heap.pop() {
        let idx = px.row * w + px.col;
        // Already assigned (could happen if pushed from multiple sides)
        if marker_data[idx] != 0 {
            continue;
        }

        // Find the unique label among labelled neighbours
        let mut label: i32 = 0;
        let mut is_boundary = false;

        for &(dr, dc) in &offsets {
            let nr = px.row as i32 + dr;
            let nc = px.col as i32 + dc;
            if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                let nidx = nr as usize * w + nc as usize;
                let nl = marker_data[nidx];
                if nl > 0 {
                    if label == 0 {
                        label = nl;
                    } else if nl != label {
                        is_boundary = true;
                        break;
                    }
                }
            }
        }

        if is_boundary {
            marker_data[idx] = -1;
        } else if label > 0 {
            marker_data[idx] = label;
        } else {
            // No labelled neighbour yet — should not happen, but be safe
            continue;
        }

        // Push unlabeled neighbours
        for &(dr, dc) in &offsets {
            let nr = px.row as i32 + dr;
            let nc = px.col as i32 + dc;
            if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                let nidx = nr as usize * w + nc as usize;
                if marker_data[nidx] == 0 && !in_queue[nidx] {
                    in_queue[nidx] = true;
                    heap.push(WatershedPixel {
                        gradient: gradient[nidx],
                        row: nr as usize,
                        col: nc as usize,
                    });
                }
            }
        }
    }

    Ok(())
}

/// Compute per-pixel gradient magnitude from a 3-channel CHW image.
fn compute_gradient_magnitude<T: Float>(img_data: &[T], h: usize, w: usize, hw: usize) -> Vec<f64> {
    let mut gradient = vec![0.0f64; hw];
    for r in 0..h {
        for col in 0..w {
            let mut g = 0.0f64;
            for ch in 0..3usize {
                let base = ch * hw + r * w + col;
                let v = img_data[base].to_f64();

                // Horizontal difference
                if col + 1 < w {
                    let d = img_data[base + 1].to_f64() - v;
                    g += d * d;
                }
                // Vertical difference
                if r + 1 < h {
                    let d = img_data[base + w].to_f64() - v;
                    g += d * d;
                }
            }
            gradient[r * w + col] = g.sqrt();
        }
    }
    gradient
}

// ---------------------------------------------------------------------------
// GrabCut (simplified)
// ---------------------------------------------------------------------------

/// Mode for `grab_cut`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrabCutMode {
    /// Initialize the mask from a bounding rectangle.
    InitWithRect,
    /// Initialise from the provided mask (caller sets PR_FGD / PR_BGD etc.).
    InitWithMask,
    /// Run additional iterations on an already-initialised mask.
    Eval,
}

/// GrabCut mask values.
pub const GC_BGD: u8 = 0;
pub const GC_FGD: u8 = 1;
pub const GC_PR_BGD: u8 = 2;
pub const GC_PR_FGD: u8 = 3;

/// Simplified GrabCut segmentation using K-means colour models (K=5).
///
/// This is an iterative foreground / background segmentation. A full GrabCut
/// would use Gaussian Mixture Models and graph-cut (max-flow / min-cut); this
/// simplified version replaces those with K-means colour prototypes and a
/// 4-connected smoothness pass.
///
/// # Arguments
/// * `image` - 3-channel colour image (CHW).
/// * `mask`  - 1-channel `u8` tensor (same H x W). Values: 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD.
/// * `rect`  - Bounding rectangle `(x, y, w, h)`, required when `mode == InitWithRect`.
/// * `iter_count` - Number of EM-like iterations.
/// * `mode`  - Initialisation mode.
///
/// # Errors
/// Returns an error on dimension / argument problems.
pub fn grab_cut<T: Float>(
    image: &CpuTensor<T>,
    mask: &mut CpuTensor<u8>,
    rect: Option<(u32, u32, u32, u32)>,
    iter_count: u32,
    mode: GrabCutMode,
) -> Result<()> {
    let (c, h, w) = image.shape.chw();
    if c != 3 {
        return Err(cv_core::Error::InvalidInput(
            "GrabCut requires a 3-channel image".into(),
        ));
    }
    if mask.shape.channels != 1 || mask.shape.height != h || mask.shape.width != w {
        return Err(cv_core::Error::DimensionMismatch(
            "Mask dimensions must match image spatial dimensions (1, H, W)".into(),
        ));
    }

    let hw = h * w;
    let img_data = image.as_slice()?;

    // --- Initialisation ---
    if mode == GrabCutMode::InitWithRect {
        let (rx, ry, rw, rh) = rect.ok_or_else(|| {
            cv_core::Error::InvalidInput("rect is required for InitWithRect mode".into())
        })?;
        let mask_data = mask.as_mut_slice()?;
        for r in 0..h {
            for col in 0..w {
                let inside = (col as u32) >= rx
                    && (col as u32) < rx + rw
                    && (r as u32) >= ry
                    && (r as u32) < ry + rh;
                mask_data[r * w + col] = if inside { GC_PR_FGD } else { GC_BGD };
            }
        }
    }

    // Collect pixel colours (as f64 triples) for convenience.
    let pixels: Vec<[f64; 3]> = (0..hw)
        .map(|i| {
            [
                img_data[i].to_f64(),
                img_data[hw + i].to_f64(),
                img_data[2 * hw + i].to_f64(),
            ]
        })
        .collect();

    // --- Iterative refinement ---
    const K: usize = 5;

    for _iter in 0..iter_count {
        let mask_data = mask.as_slice()?;

        // 1. Build colour models via K-means
        let fg_model = build_kmeans_model(&pixels, mask_data, true, K);
        let bg_model = build_kmeans_model(&pixels, mask_data, false, K);

        // 2. Assign probable pixels
        let mask_data = mask.as_mut_slice()?;
        for i in 0..hw {
            let m = mask_data[i];
            if m == GC_PR_FGD || m == GC_PR_BGD {
                let fg_dist = nearest_cluster_dist(&pixels[i], &fg_model);
                let bg_dist = nearest_cluster_dist(&pixels[i], &bg_model);
                mask_data[i] = if fg_dist <= bg_dist {
                    GC_PR_FGD
                } else {
                    GC_PR_BGD
                };
            }
        }

        // 3. Smoothness: 4-connected consistency (simple majority vote among neighbours
        //    for probable pixels only).
        let snap: Vec<u8> = mask_data.to_vec();
        let offsets: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for r in 0..h {
            for col in 0..w {
                let idx = r * w + col;
                let m = snap[idx];
                if m != GC_PR_FGD && m != GC_PR_BGD {
                    continue;
                }
                let mut fg_count = 0i32;
                let mut bg_count = 0i32;
                for &(dr, dc) in &offsets {
                    let nr = r as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let nm = snap[nr as usize * w + nc as usize];
                        if nm == GC_FGD || nm == GC_PR_FGD {
                            fg_count += 1;
                        } else {
                            bg_count += 1;
                        }
                    }
                }
                // Only flip if neighbours strongly disagree (>= 3 of 4)
                if m == GC_PR_FGD && bg_count >= 3 {
                    mask_data[idx] = GC_PR_BGD;
                } else if m == GC_PR_BGD && fg_count >= 3 {
                    mask_data[idx] = GC_PR_FGD;
                }
            }
        }
    }

    Ok(())
}

/// Build a K-means colour model from all pixels whose mask indicates foreground
/// (if `foreground == true`) or background.
fn build_kmeans_model(
    pixels: &[[f64; 3]],
    mask: &[u8],
    foreground: bool,
    k: usize,
) -> Vec<[f64; 3]> {
    // Collect relevant pixel colours
    let relevant: Vec<[f64; 3]> = pixels
        .iter()
        .zip(mask.iter())
        .filter_map(|(&px, &m)| {
            let is_fg = m == GC_FGD || m == GC_PR_FGD;
            if foreground == is_fg {
                Some(px)
            } else {
                None
            }
        })
        .collect();

    if relevant.is_empty() {
        return vec![[0.0; 3]; k];
    }

    // Initialise centroids by uniform sampling
    let mut centroids: Vec<[f64; 3]> = (0..k)
        .map(|i| {
            let idx = i * relevant.len() / k.max(1);
            relevant[idx.min(relevant.len() - 1)]
        })
        .collect();

    // Run a few K-means iterations
    let max_iters = 10;
    for _ in 0..max_iters {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];

        for px in &relevant {
            let best = nearest_cluster_idx(px, &centroids);
            for d in 0..3 {
                sums[best][d] += px[d];
            }
            counts[best] += 1;
        }

        let mut changed = false;
        for j in 0..k {
            if counts[j] > 0 {
                let new_c = [
                    sums[j][0] / counts[j] as f64,
                    sums[j][1] / counts[j] as f64,
                    sums[j][2] / counts[j] as f64,
                ];
                if (new_c[0] - centroids[j][0]).abs() > 1e-6
                    || (new_c[1] - centroids[j][1]).abs() > 1e-6
                    || (new_c[2] - centroids[j][2]).abs() > 1e-6
                {
                    changed = true;
                }
                centroids[j] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    centroids
}

fn nearest_cluster_idx(px: &[f64; 3], centroids: &[[f64; 3]]) -> usize {
    let mut best = 0;
    let mut best_d = f64::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist3(px, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn nearest_cluster_dist(px: &[f64; 3], centroids: &[[f64; 3]]) -> f64 {
    centroids
        .iter()
        .map(|c| sq_dist3(px, c))
        .fold(f64::MAX, f64::min)
}

#[inline]
fn sq_dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

// ---------------------------------------------------------------------------
// Flood fill
// ---------------------------------------------------------------------------

/// BFS-based flood fill from a seed point.
///
/// Fills all connected pixels whose value is within
/// `[seed_value - lo_diff, seed_value + up_diff]` with `new_value`.
///
/// # Arguments
/// * `image` - 1-channel (grayscale) tensor, modified in-place.
/// * `seed`  - `(row, col)` starting point.
/// * `new_value` - The value to paint.
/// * `lo_diff` - Lower tolerance relative to the seed pixel value.
/// * `up_diff` - Upper tolerance relative to the seed pixel value.
/// * `connectivity` - `4` or `8`.
///
/// # Returns
/// The number of pixels filled.
pub fn flood_fill<T: Float>(
    image: &mut CpuTensor<T>,
    seed: (u32, u32),
    new_value: T,
    lo_diff: T,
    up_diff: T,
    connectivity: u8,
) -> Result<u32> {
    if image.shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Flood fill requires a 1-channel image".into(),
        ));
    }
    if connectivity != 4 && connectivity != 8 {
        return Err(cv_core::Error::InvalidInput(
            "Connectivity must be 4 or 8".into(),
        ));
    }

    let h = image.shape.height;
    let w = image.shape.width;
    let (sr, sc) = (seed.0 as usize, seed.1 as usize);

    if sr >= h || sc >= w {
        return Err(cv_core::Error::InvalidInput(format!(
            "Seed ({}, {}) out of bounds for {}x{} image",
            sr, sc, h, w
        )));
    }

    let data = image.as_mut_slice()?;
    let seed_val = data[sr * w + sc];
    let lo = seed_val - lo_diff;
    let hi = seed_val + up_diff;

    let offsets_4: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let offsets_8: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    let mut visited = vec![false; h * w];
    let mut queue = VecDeque::new();
    queue.push_back((sr, sc));
    visited[sr * w + sc] = true;
    let mut count = 0u32;

    while let Some((r, c)) = queue.pop_front() {
        let idx = r * w + c;
        let v = data[idx];
        if v >= lo && v <= hi {
            data[idx] = new_value;
            count += 1;

            let neighbors: &[(i32, i32)] = if connectivity == 8 {
                &offsets_8
            } else {
                &offsets_4
            };
            for &(dr, dc) in neighbors {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                    let nidx = nr as usize * w + nc as usize;
                    if !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((nr as usize, nc as usize));
                    }
                }
            }
        }
    }

    Ok(count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{CpuTensor, TensorShape};

    /// Helper: create a 3-channel CHW image from per-channel flat arrays.
    fn make_color_image(
        h: usize,
        w: usize,
        ch0: &[f32],
        ch1: &[f32],
        ch2: &[f32],
    ) -> CpuTensor<f32> {
        let mut data = Vec::with_capacity(3 * h * w);
        data.extend_from_slice(ch0);
        data.extend_from_slice(ch1);
        data.extend_from_slice(ch2);
        CpuTensor::<f32>::from_vec(data, TensorShape::new(3, h, w)).unwrap()
    }

    // -----------------------------------------------------------------------
    // Watershed tests
    // -----------------------------------------------------------------------

    #[test]
    fn watershed_two_blobs() {
        // 10x10 image with two distinct blobs:
        //   left half  (cols 0..4) = dark  (value 0.0)
        //   right half (cols 6..9) = bright (value 1.0)
        //   middle column 5       = gradient (0.5)
        let h = 10usize;
        let w = 10usize;
        let hw = h * w;

        let mut ch = vec![0.0f32; hw];
        for r in 0..h {
            for c in 0..w {
                ch[r * w + c] = if c <= 4 {
                    0.0
                } else if c >= 6 {
                    1.0
                } else {
                    0.5
                };
            }
        }
        let image = make_color_image(h, w, &ch, &ch, &ch);

        // Markers: label 1 at (5,2), label 2 at (5,8), rest 0
        let mut marker_data = vec![0i32; hw];
        marker_data[5 * w + 2] = 1;
        marker_data[5 * w + 8] = 2;
        let mut markers =
            CpuTensor::<i32>::from_vec(marker_data, TensorShape::new(1, h, w)).unwrap();

        watershed(&image, &mut markers).unwrap();

        let m = markers.as_slice().unwrap();

        // All pixels should be assigned (no zeros left)
        let zeros: usize = m.iter().filter(|&&v| v == 0).count();
        assert_eq!(zeros, 0, "all pixels should be labelled or boundary");

        // The left seed region should dominate left columns, right seed the right columns
        assert_eq!(m[5 * w + 0], 1);
        assert_eq!(m[5 * w + 9], 2);

        // There should be at least one boundary pixel (-1) somewhere in the middle
        let boundaries: usize = m.iter().filter(|&&v| v == -1).count();
        assert!(boundaries > 0, "expected boundary pixels between blobs");
    }

    #[test]
    fn watershed_dimension_mismatch() {
        let image = make_color_image(4, 4, &[0.0; 16], &[0.0; 16], &[0.0; 16]);
        let mut markers =
            CpuTensor::<i32>::from_vec(vec![0i32; 9], TensorShape::new(1, 3, 3)).unwrap();
        assert!(watershed(&image, &mut markers).is_err());
    }

    // -----------------------------------------------------------------------
    // GrabCut tests
    // -----------------------------------------------------------------------

    #[test]
    fn grabcut_init_with_rect_two_colors() {
        // 10x10 image: left half red (1,0,0), right half blue (0,0,1)
        let h = 10usize;
        let w = 10usize;
        let hw = h * w;

        let mut ch_r = vec![0.0f32; hw];
        let ch_g = vec![0.0f32; hw];
        let mut ch_b = vec![0.0f32; hw];
        for r in 0..h {
            for c in 0..w {
                if c < 5 {
                    ch_r[r * w + c] = 1.0;
                } else {
                    ch_b[r * w + c] = 1.0;
                }
            }
        }
        let image = make_color_image(h, w, &ch_r, &ch_g, &ch_b);

        let mut mask = CpuTensor::<u8>::from_vec(vec![0u8; hw], TensorShape::new(1, h, w)).unwrap();

        // Rectangle covers only the left half -> foreground should be left half
        grab_cut(
            &image,
            &mut mask,
            Some((0, 0, 5, 10)),
            5,
            GrabCutMode::InitWithRect,
        )
        .unwrap();

        let m = mask.as_slice().unwrap();
        // Pixels inside rect should mostly be PR_FGD (3) or FGD (1)
        let inside_fg: usize = (0..h)
            .flat_map(|r| (0..5).map(move |c| r * w + c))
            .filter(|&i| m[i] == GC_FGD || m[i] == GC_PR_FGD)
            .count();
        assert!(
            inside_fg > hw / 4,
            "most inside-rect pixels should be foreground, got {}",
            inside_fg
        );

        // Pixels outside rect should remain BGD (0)
        let outside_bg: usize = (0..h)
            .flat_map(|r| (5..w).map(move |c| r * w + c))
            .filter(|&i| m[i] == GC_BGD)
            .count();
        assert!(
            outside_bg > hw / 4,
            "outside-rect pixels should stay background, got {}",
            outside_bg
        );
    }

    #[test]
    fn grabcut_requires_rect_for_init_with_rect() {
        let image = make_color_image(4, 4, &[0.0; 16], &[0.0; 16], &[0.0; 16]);
        let mut mask = CpuTensor::<u8>::from_vec(vec![0u8; 16], TensorShape::new(1, 4, 4)).unwrap();
        let res = grab_cut(&image, &mut mask, None, 1, GrabCutMode::InitWithRect);
        assert!(res.is_err());
    }

    // -----------------------------------------------------------------------
    // Flood fill tests
    // -----------------------------------------------------------------------

    #[test]
    fn flood_fill_uniform_region() {
        // 5x5 image, all zeros except a border of 1.0
        let h = 5usize;
        let w = 5usize;
        let mut data = vec![1.0f32; h * w];
        // Inner 3x3 = 0.0
        for r in 1..4 {
            for c in 1..4 {
                data[r * w + c] = 0.0;
            }
        }
        let mut image = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        let count = flood_fill(&mut image, (2, 2), 0.5, 0.0, 0.0, 4).unwrap();

        // Only the 3x3 inner region (value == 0.0) should be filled
        assert_eq!(count, 9);
        let d = image.as_slice().unwrap();
        // Centre should now be 0.5
        assert!((d[2 * w + 2] - 0.5).abs() < 1e-6);
        // Border should be unchanged
        assert!((d[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flood_fill_with_tolerance() {
        // 4x4 image with a slight gradient
        let h = 4usize;
        let w = 4usize;
        let data: Vec<f32> = (0..h * w)
            .map(|i| {
                let r = i / w;
                let c = i % w;
                (r + c) as f32 * 0.1 // values from 0.0 to 0.6
            })
            .collect();
        let mut image = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        // Seed at (0,0) value=0.0, tolerance [0.0 - 0.15, 0.0 + 0.15] = [-0.15, 0.15]
        let count = flood_fill(&mut image, (0, 0), 9.0, 0.15, 0.15, 4).unwrap();

        // (0,0)=0.0, (0,1)=0.1, (1,0)=0.1 are within tolerance; (1,1)=0.2 is not reachable via 4-conn from seed through in-range pixels
        // Actually (1,1) = 0.2 > 0.15 so it's blocked. (0,1) and (1,0) are reachable.
        assert!(
            count >= 3,
            "at least seed + 2 neighbours should be filled, got {}",
            count
        );
    }

    #[test]
    fn flood_fill_8_connectivity() {
        // 3x3 image, centre and all 8-connected = 0, but we put a 4-connected
        // barrier; 8-conn should still reach diagonals.
        let h = 3usize;
        let w = 3usize;
        // 0 1 0
        // 1 0 1
        // 0 1 0
        let data = vec![0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut image = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        // 4-conn from centre: only (1,1) itself
        let mut img4 = image.clone();
        let count4 = flood_fill(&mut img4, (1, 1), 5.0, 0.0, 0.0, 4).unwrap();
        assert_eq!(count4, 1, "4-conn: only centre should be filled");

        // 8-conn from centre: centre + 4 corners
        let count8 = flood_fill(&mut image, (1, 1), 5.0, 0.0, 0.0, 8).unwrap();
        assert_eq!(count8, 5, "8-conn: centre + 4 corners");
    }

    #[test]
    fn flood_fill_out_of_bounds_seed() {
        let mut image =
            CpuTensor::<f32>::from_vec(vec![0.0; 4], TensorShape::new(1, 2, 2)).unwrap();
        assert!(flood_fill(&mut image, (5, 5), 1.0, 0.0, 0.0, 4).is_err());
    }
}
