//! Connected component labeling using two-pass union-find algorithm.
//!
//! Operates on `CpuTensor<T>` where non-zero values are foreground.
//! Supports 4-connected and 8-connected neighborhoods.

use cv_core::{CpuTensor, TensorShape};

/// Per-component statistics from connected component labeling.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentStat {
    /// Number of pixels in this component.
    pub area: u32,
    /// Left edge of the bounding box (min x).
    pub left: u32,
    /// Top edge of the bounding box (min y).
    pub top: u32,
    /// Width of the bounding box.
    pub width: u32,
    /// Height of the bounding box.
    pub height: u32,
    /// X-coordinate of the centroid.
    pub centroid_x: f64,
    /// Y-coordinate of the centroid.
    pub centroid_y: f64,
}

/// Result of connected component labeling with statistics.
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    /// Label image where each pixel holds its component label (0 = background).
    pub labels: CpuTensor<u32>,
    /// Total number of labels (including background label 0).
    pub num_labels: u32,
    /// Per-component statistics, indexed by label-1 (only foreground components).
    pub stats: Vec<ComponentStat>,
}

// --- Union-Find (disjoint set) with path compression and union by rank ---

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        let mut parent = Vec::with_capacity(size);
        for i in 0..size {
            parent.push(i as u32);
        }
        Self {
            parent,
            rank: vec![0; size],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path compression (path halving)
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        // Union by rank
        if self.rank[ra as usize] < self.rank[rb as usize] {
            self.parent[ra as usize] = rb;
            rb
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            self.parent[rb as usize] = ra;
            ra
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
            ra
        }
    }
}

/// Label connected components in a binary image using a two-pass union-find algorithm.
///
/// Input must be a single-channel `CpuTensor` (channels=1). Non-zero values are foreground.
/// `connectivity` must be 4 or 8.
///
/// Returns `(label_image, num_labels)` where `num_labels` includes background (label 0).
#[allow(clippy::needless_range_loop)]
pub fn connected_components<T>(
    binary: &CpuTensor<T>,
    connectivity: u8,
) -> crate::Result<(CpuTensor<u32>, u32)>
where
    T: Clone + Copy + Default + PartialEq + std::fmt::Debug + Into<f64> + 'static,
{
    if connectivity != 4 && connectivity != 8 {
        return Err(cv_core::Error::InvalidInput(
            "Connectivity must be 4 or 8".into(),
        ));
    }
    if binary.shape.channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Input must be a single-channel image".into(),
        ));
    }

    let h = binary.shape.height;
    let w = binary.shape.width;
    let data = binary.as_slice()?;
    let total = h * w;

    // We'll allocate labels with max possible label count = total + 1
    let mut labels = vec![0u32; total];
    let mut uf = UnionFind::new(total + 1); // label 0 unused for foreground
    let mut next_label = 1u32;

    let is_fg = |idx: usize| -> bool {
        let v: f64 = data[idx].into();
        v != 0.0
    };

    // --- Pass 1: assign provisional labels ---
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !is_fg(idx) {
                continue;
            }

            // Collect neighbor labels
            let mut neighbors = [0u32; 4];
            let mut n_count = 0usize;

            // West neighbor
            if x > 0 && labels[idx - 1] != 0 {
                neighbors[n_count] = labels[idx - 1];
                n_count += 1;
            }
            // North neighbor
            if y > 0 && labels[idx - w] != 0 {
                neighbors[n_count] = labels[idx - w];
                n_count += 1;
            }

            if connectivity == 8 {
                // North-West
                if x > 0 && y > 0 && labels[idx - w - 1] != 0 {
                    neighbors[n_count] = labels[idx - w - 1];
                    n_count += 1;
                }
                // North-East
                if x + 1 < w && y > 0 && labels[idx - w + 1] != 0 {
                    neighbors[n_count] = labels[idx - w + 1];
                    n_count += 1;
                }
            }

            if n_count == 0 {
                // New label
                labels[idx] = next_label;
                next_label += 1;
            } else {
                // Find minimum root label among neighbors
                let mut min_label = uf.find(neighbors[0]);
                for i in 1..n_count {
                    let root = uf.find(neighbors[i]);
                    if root < min_label {
                        min_label = root;
                    }
                }
                labels[idx] = min_label;
                // Union all neighbor labels
                for i in 0..n_count {
                    uf.union(min_label, neighbors[i]);
                }
            }
        }
    }

    // --- Pass 2: flatten labels to consecutive integers ---
    let mut label_map = vec![0u32; next_label as usize];
    let mut final_label = 0u32;

    for i in 0..total {
        if labels[i] != 0 {
            let root = uf.find(labels[i]);
            if label_map[root as usize] == 0 {
                final_label += 1;
                label_map[root as usize] = final_label;
            }
            labels[i] = label_map[root as usize];
        }
    }

    let num_labels = final_label + 1; // +1 for background
    let shape = TensorShape::new(1, h, w);
    let label_tensor = CpuTensor::<u32>::from_vec(labels, shape)?;

    Ok((label_tensor, num_labels))
}

/// Label connected components and compute per-component statistics.
///
/// Returns a `ConnectedComponentsResult` containing the label image,
/// number of labels, and per-component statistics (area, bounding box, centroid).
pub fn connected_components_with_stats_tensor<T>(
    binary: &CpuTensor<T>,
    connectivity: u8,
) -> crate::Result<ConnectedComponentsResult>
where
    T: Clone + Copy + Default + PartialEq + std::fmt::Debug + Into<f64> + 'static,
{
    let (label_tensor, num_labels) = connected_components(binary, connectivity)?;
    let label_data = label_tensor.as_slice()?;
    let h = label_tensor.shape.height;
    let w = label_tensor.shape.width;

    // Compute stats for each foreground label (1..num_labels)
    let fg_count = (num_labels - 1) as usize;
    let mut areas = vec![0u32; fg_count];
    let mut min_x = vec![u32::MAX; fg_count];
    let mut min_y = vec![u32::MAX; fg_count];
    let mut max_x = vec![0u32; fg_count];
    let mut max_y = vec![0u32; fg_count];
    let mut sum_x = vec![0.0f64; fg_count];
    let mut sum_y = vec![0.0f64; fg_count];

    for y in 0..h {
        for x in 0..w {
            let lbl = label_data[y * w + x];
            if lbl == 0 {
                continue;
            }
            let li = (lbl - 1) as usize;
            areas[li] += 1;
            let xu = x as u32;
            let yu = y as u32;
            if xu < min_x[li] {
                min_x[li] = xu;
            }
            if xu > max_x[li] {
                max_x[li] = xu;
            }
            if yu < min_y[li] {
                min_y[li] = yu;
            }
            if yu > max_y[li] {
                max_y[li] = yu;
            }
            sum_x[li] += x as f64;
            sum_y[li] += y as f64;
        }
    }

    let mut stats = Vec::with_capacity(fg_count);
    for i in 0..fg_count {
        stats.push(ComponentStat {
            area: areas[i],
            left: min_x[i],
            top: min_y[i],
            width: max_x[i] - min_x[i] + 1,
            height: max_y[i] - min_y[i] + 1,
            centroid_x: sum_x[i] / areas[i] as f64,
            centroid_y: sum_y[i] / areas[i] as f64,
        });
    }

    Ok(ConnectedComponentsResult {
        labels: label_tensor,
        num_labels,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::TensorShape;

    fn make_binary_image(w: usize, h: usize, pixels: &[(usize, usize)]) -> CpuTensor<f32> {
        let mut data = vec![0.0f32; h * w];
        for &(x, y) in pixels {
            data[y * w + x] = 1.0;
        }
        CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap()
    }

    #[test]
    fn test_single_rectangle() {
        // 10x10 image with a 4x3 filled rectangle at (2,3)-(5,5)
        let mut pixels = Vec::new();
        for y in 3..6 {
            for x in 2..6 {
                pixels.push((x, y));
            }
        }
        let img = make_binary_image(10, 10, &pixels);
        let (labels, num_labels) = connected_components(&img, 8).unwrap();
        assert_eq!(num_labels, 2); // background + 1 component
                                   // Check a foreground pixel
        assert_eq!(labels.index(0, 3, 2).unwrap(), 1);
        // Check a background pixel
        assert_eq!(labels.index(0, 0, 0).unwrap(), 0);
    }

    #[test]
    fn test_two_blobs_8connected() {
        // Two separate 2x2 blocks
        let pixels: Vec<(usize, usize)> = vec![
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (6, 6),
            (7, 6),
            (6, 7),
            (7, 7),
        ];
        let img = make_binary_image(10, 10, &pixels);
        let (_, num_labels) = connected_components(&img, 8).unwrap();
        assert_eq!(num_labels, 3); // background + 2 components
    }

    #[test]
    fn test_l_shape_4connected() {
        // L-shape: a vertical bar + horizontal bar meeting at corner
        // Should be 1 component even with 4-connectivity
        let mut pixels = Vec::new();
        // Vertical bar at x=1, y=0..4
        for y in 0..4 {
            pixels.push((1, y));
        }
        // Horizontal bar at y=3, x=1..4
        for x in 1..4 {
            pixels.push((x, 3));
        }
        let img = make_binary_image(6, 6, &pixels);
        let (_, num_labels) = connected_components(&img, 4).unwrap();
        assert_eq!(num_labels, 2); // background + 1 L-shaped component
    }

    #[test]
    fn test_diagonal_4_vs_8() {
        // Two pixels diagonally adjacent: (1,1) and (2,2)
        let pixels = vec![(1usize, 1usize), (2, 2)];
        let img = make_binary_image(5, 5, &pixels);

        // With 4-connectivity they should be separate
        let (_, num4) = connected_components(&img, 4).unwrap();
        assert_eq!(num4, 3); // bg + 2

        // With 8-connectivity they should be connected
        let (_, num8) = connected_components(&img, 8).unwrap();
        assert_eq!(num8, 2); // bg + 1
    }

    #[test]
    fn test_stats_rectangle() {
        let mut pixels = Vec::new();
        for y in 2..5 {
            for x in 3..7 {
                pixels.push((x, y));
            }
        }
        let img = make_binary_image(10, 10, &pixels);
        let result = connected_components_with_stats_tensor(&img, 8).unwrap();
        assert_eq!(result.num_labels, 2);
        assert_eq!(result.stats.len(), 1);

        let s = &result.stats[0];
        assert_eq!(s.area, 12); // 4*3
        assert_eq!(s.left, 3);
        assert_eq!(s.top, 2);
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 3);
        // Centroid should be at (4.5, 3.0)
        assert!((s.centroid_x - 4.5).abs() < 1e-9);
        assert!((s.centroid_y - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_multiple_blobs() {
        let mut pixels = Vec::new();
        // Blob 1: 3x3 at (1,1)
        for y in 1..4 {
            for x in 1..4 {
                pixels.push((x, y));
            }
        }
        // Blob 2: 2x4 at (7,5)
        for y in 5..9 {
            for x in 7..9 {
                pixels.push((x, y));
            }
        }
        let img = make_binary_image(12, 12, &pixels);
        let result = connected_components_with_stats_tensor(&img, 8).unwrap();
        assert_eq!(result.num_labels, 3);
        assert_eq!(result.stats.len(), 2);

        let total_area: u32 = result.stats.iter().map(|s| s.area).sum();
        assert_eq!(total_area, 9 + 8);
    }

    #[test]
    fn test_empty_image() {
        let img = make_binary_image(5, 5, &[]);
        let (_, num_labels) = connected_components(&img, 8).unwrap();
        assert_eq!(num_labels, 1); // only background
    }

    #[test]
    fn test_invalid_connectivity() {
        let img = make_binary_image(5, 5, &[]);
        assert!(connected_components(&img, 6).is_err());
    }

    #[test]
    fn test_u8_input() {
        // Verify it works with u8 tensors too
        let mut data = vec![0u8; 25];
        data[6] = 255;
        data[7] = 255;
        data[11] = 255;
        data[12] = 255;
        let img = CpuTensor::<u8>::from_vec(data, TensorShape::new(1, 5, 5)).unwrap();
        let (_, num_labels) = connected_components(&img, 8).unwrap();
        assert_eq!(num_labels, 2);
    }
}
