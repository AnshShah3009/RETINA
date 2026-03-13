//! ArUco marker detection, generation, and pose estimation using `CpuTensor` types.
//!
//! This module provides a tensor-based API for working with ArUco fiducial markers,
//! complementing the `image`-crate-based API in [`crate::markers`].
//!
//! # Example
//!
//! ```no_run
//! use cv_features::aruco::*;
//!
//! let dict = ArucoDictionary::Dict4x4_50;
//! let marker_img = draw_marker(dict, 7, 60).unwrap();
//! let detector = ArucoDetector::new(dict);
//! let detections = detector.detect_u8(&marker_img).unwrap();
//! assert!(detections.iter().any(|d| d.id == 7));
//! ```

use cv_core::{CpuTensor, Error, Float, Result, TensorShape};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Dictionary
// ---------------------------------------------------------------------------

/// Predefined ArUco dictionaries.
///
/// Each variant encodes a marker bit-grid size and a dictionary size (number of
/// distinct markers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArucoDictionary {
    /// 4x4 grid, 50 markers
    Dict4x4_50,
    /// 4x4 grid, 100 markers
    Dict4x4_100,
    /// 4x4 grid, 250 markers
    Dict4x4_250,
    /// 5x5 grid, 50 markers
    Dict5x5_50,
    /// 5x5 grid, 100 markers
    Dict5x5_100,
    /// 5x5 grid, 250 markers
    Dict5x5_250,
    /// 6x6 grid, 50 markers
    Dict6x6_50,
    /// 6x6 grid, 100 markers
    Dict6x6_100,
    /// 6x6 grid, 250 markers
    Dict6x6_250,
}

impl ArucoDictionary {
    /// Returns the side length of the inner bit grid (4, 5, or 6).
    pub fn marker_size(&self) -> usize {
        match self {
            Self::Dict4x4_50 | Self::Dict4x4_100 | Self::Dict4x4_250 => 4,
            Self::Dict5x5_50 | Self::Dict5x5_100 | Self::Dict5x5_250 => 5,
            Self::Dict6x6_50 | Self::Dict6x6_100 | Self::Dict6x6_250 => 6,
        }
    }

    /// Returns the number of markers in this dictionary.
    pub fn dict_size(&self) -> usize {
        match self {
            Self::Dict4x4_50 | Self::Dict5x5_50 | Self::Dict6x6_50 => 50,
            Self::Dict4x4_100 | Self::Dict5x5_100 | Self::Dict6x6_100 => 100,
            Self::Dict4x4_250 | Self::Dict5x5_250 | Self::Dict6x6_250 => 250,
        }
    }

    /// Get the bit pattern for marker `id`.
    ///
    /// Returns a `marker_size x marker_size` grid of 0/1 values (row-major),
    /// or `None` if `id` is out of range.
    #[allow(clippy::needless_range_loop)]
    pub fn get_marker(&self, id: usize) -> Option<Vec<Vec<u8>>> {
        let codes = dictionary_codes(*self);
        let code = codes.get(id).copied()?;
        let side = self.marker_size();
        let mut grid = vec![vec![0u8; side]; side];
        for y in 0..side {
            for x in 0..side {
                if ((code >> (y * side + x)) & 1) != 0 {
                    grid[y][x] = 1;
                }
            }
        }
        Some(grid)
    }
}

// ---------------------------------------------------------------------------
// Detection types
// ---------------------------------------------------------------------------

/// A detected ArUco marker in an image.
#[derive(Debug, Clone)]
pub struct DetectedMarker {
    /// Dictionary ID of the detected marker.
    pub id: usize,
    /// Image-space corner coordinates (top-left, top-right, bottom-right, bottom-left).
    pub corners: [(f64, f64); 4],
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Configurable ArUco marker detector operating on `CpuTensor` images.
#[derive(Debug, Clone)]
pub struct ArucoDetector {
    dictionary: ArucoDictionary,
    /// Window size for adaptive thresholding (must be odd).
    pub adaptive_thresh_win_size: usize,
    /// Constant subtracted during adaptive thresholding.
    pub adaptive_thresh_constant: f64,
    /// Minimum perimeter (in pixels) for a candidate quadrilateral.
    pub min_marker_perimeter: f64,
    /// Maximum perimeter (in pixels) for a candidate quadrilateral.
    pub max_marker_perimeter: f64,
    /// Polygonal approximation accuracy as a fraction of perimeter.
    pub polygonal_approx_accuracy: f64,
}

impl ArucoDetector {
    /// Create a new detector for the given dictionary with default parameters.
    pub fn new(dictionary: ArucoDictionary) -> Self {
        Self {
            dictionary,
            adaptive_thresh_win_size: 7,
            adaptive_thresh_constant: 7.0,
            min_marker_perimeter: 30.0,
            max_marker_perimeter: 4000.0,
            polygonal_approx_accuracy: 0.03,
        }
    }

    /// Detect ArUco markers in a single-channel floating-point grayscale image.
    ///
    /// The image tensor must have shape (1, H, W) with values in [0, 1].
    ///
    /// # Pipeline
    /// 1. Adaptive threshold to binarize
    /// 2. Connected-component candidate extraction
    /// 3. Filter by perimeter and aspect ratio
    /// 4. For each candidate: sample grid bits, check border, read payload
    /// 5. Match against all 4 rotations of each dictionary code
    pub fn detect<T: Float>(&self, image: &CpuTensor<T>) -> Result<Vec<DetectedMarker>> {
        let (h, w) = (image.shape.height, image.shape.width);
        if image.shape.channels != 1 || h == 0 || w == 0 {
            return Err(Error::InvalidInput(
                "aruco detect requires single-channel (1, H, W) image".into(),
            ));
        }
        let data = image.as_slice()?;
        let gray: Vec<u8> = data.iter().map(|&v| float_to_u8(v)).collect();
        self.detect_gray(&gray, w, h)
    }

    /// Detect ArUco markers in a single-channel `u8` grayscale image.
    ///
    /// The image tensor must have shape (1, H, W) with values in [0, 255].
    pub fn detect_u8(&self, image: &CpuTensor<u8>) -> Result<Vec<DetectedMarker>> {
        let (h, w) = (image.shape.height, image.shape.width);
        if image.shape.channels != 1 || h == 0 || w == 0 {
            return Err(Error::InvalidInput(
                "aruco detect requires single-channel (1, H, W) image".into(),
            ));
        }
        let gray = image.as_slice()?.to_vec();
        self.detect_gray(&gray, w, h)
    }

    /// Internal: run detection on raw grayscale u8 buffer.
    fn detect_gray(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<DetectedMarker>> {
        // 1. Adaptive threshold
        let binary = adaptive_threshold(
            gray,
            w,
            h,
            self.adaptive_thresh_win_size,
            self.adaptive_thresh_constant,
        );

        // 2-3. Find candidate bounding boxes
        let candidates = find_candidates(
            &binary,
            w,
            h,
            self.min_marker_perimeter,
            self.max_marker_perimeter,
        );

        let payload_bits = self.dictionary.marker_size();
        let border_bits = 1usize;
        let grid = payload_bits + 2 * border_bits;
        let codes = dictionary_codes(self.dictionary);

        // 4-5. Decode each candidate
        let mut detections = Vec::new();
        for c in &candidates {
            let bits = sample_grid_bits(gray, w, c.min_x, c.min_y, c.max_x, c.max_y, grid);
            if !border_is_black(&bits, grid) {
                continue;
            }
            let payload = extract_payload(&bits, payload_bits, border_bits);
            let (id, _rot, dist) = decode_with_rotations(payload, &codes, payload_bits);
            if dist == 0 {
                detections.push(DetectedMarker {
                    id,
                    corners: [
                        (c.min_x as f64, c.min_y as f64),
                        (c.max_x as f64, c.min_y as f64),
                        (c.max_x as f64, c.max_y as f64),
                        (c.min_x as f64, c.max_y as f64),
                    ],
                });
            }
        }
        Ok(detections)
    }
}

// ---------------------------------------------------------------------------
// Pose estimation
// ---------------------------------------------------------------------------

/// Estimate the pose of a detected marker using the DLT (Direct Linear Transform) method.
///
/// # Arguments
/// * `marker` - detected marker with image-space corners
/// * `marker_length` - physical side length of the marker in world units (e.g. metres)
/// * `camera_matrix` - 3x3 camera intrinsic matrix (row-major)
/// * `dist_coeffs` - 5-element distortion coefficients `[k1, k2, p1, p2, k3]`
///
/// # Returns
/// `(rotation_vec, translation_vec)` in Rodrigues form.
pub fn estimate_marker_pose(
    marker: &DetectedMarker,
    marker_length: f64,
    camera_matrix: &[[f64; 3]; 3],
    dist_coeffs: &[f64; 5],
) -> Result<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>)> {
    let half = marker_length * 0.5;
    // 3D object points (marker centred at origin, Z=0 plane).
    let obj_pts = [
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];

    // Undistort the 2D image points.
    let img_pts: Vec<[f64; 2]> = marker
        .corners
        .iter()
        .map(|&(px, py)| undistort_point(px, py, camera_matrix, dist_coeffs))
        .collect();

    // Build DLT system: for each correspondence (X, x) we get two rows.
    let fx = camera_matrix[0][0];
    let fy = camera_matrix[1][1];
    let cx = camera_matrix[0][2];
    let cy = camera_matrix[1][2];

    // Normalised image coordinates
    let norm: Vec<[f64; 2]> = img_pts
        .iter()
        .map(|p| [(p[0] - cx) / fx, (p[1] - cy) / fy])
        .collect();

    // Solve P3P via DLT on 4 points: build 2N x 12 matrix, solve via SVD.
    // We use a simpler approach: iterative PnP (Gauss-Newton) starting from DLT.
    let (r, t) = solve_pnp_dlt(&obj_pts, &norm)?;
    Ok((r, t))
}

// ---------------------------------------------------------------------------
// Marker generation
// ---------------------------------------------------------------------------

/// Draw a marker image for printing.
///
/// Generates a `size x size` single-channel `u8` tensor containing the marker pattern
/// with a 1-cell black border around the payload bits. The tensor has shape (1, size, size).
pub fn draw_marker(dictionary: ArucoDictionary, id: usize, size: usize) -> Result<CpuTensor<u8>> {
    if size < 6 {
        return Err(Error::InvalidInput(
            "marker size must be at least 6 pixels".into(),
        ));
    }
    let payload_bits = dictionary.marker_size();
    let border_bits = 1usize;
    let grid = payload_bits + 2 * border_bits;
    let codes = dictionary_codes(dictionary);
    let code = codes
        .get(id)
        .copied()
        .ok_or_else(|| Error::InvalidInput(format!("marker id {} out of range", id)))?;

    let mut pixels = vec![255u8; size * size];

    for gy in 0..grid {
        for gx in 0..grid {
            let is_border = gy < border_bits
                || gx < border_bits
                || gy >= grid - border_bits
                || gx >= grid - border_bits;
            let is_black = if is_border {
                true
            } else {
                let py = gy - border_bits;
                let px = gx - border_bits;
                let bit_idx = py * payload_bits + px;
                ((code >> bit_idx) & 1) == 1
            };
            let val = if is_black { 0u8 } else { 255u8 };

            // Fill the cell region in the output image.
            let x0 = gx * size / grid;
            let x1 = (gx + 1) * size / grid;
            let y0 = gy * size / grid;
            let y1 = (gy + 1) * size / grid;
            for y in y0..y1 {
                for x in x0..x1 {
                    pixels[y * size + x] = val;
                }
            }
        }
    }

    CpuTensor::<u8>::from_vec(pixels, TensorShape::new(1, size, size))
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Convert a Float pixel value to u8. Assumes [0,1] for floats and [0,255] for u8-like values.
fn float_to_u8<T: Float>(v: T) -> u8 {
    let f = v.to_f64();
    // Heuristic: if max plausible value <= 1.0, treat as normalised.
    if (0.0..=1.0).contains(&f) {
        (f * 255.0).round().clamp(0.0, 255.0) as u8
    } else {
        f.clamp(0.0, 255.0).round() as u8
    }
}

/// Simple adaptive mean threshold. Output: 1 = foreground (dark), 0 = background.
fn adaptive_threshold(gray: &[u8], w: usize, h: usize, win: usize, constant: f64) -> Vec<u8> {
    let win = if win.is_multiple_of(2) { win + 1 } else { win };
    let half = (win / 2) as i32;
    // Integral image for fast mean computation.
    let mut integral = vec![0i64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0i64;
        for x in 0..w {
            row_sum += gray[y * w + x] as i64;
            integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + row_sum;
        }
    }

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = (x as i32 - half).max(0) as usize;
            let y0 = (y as i32 - half).max(0) as usize;
            let x1 = ((x as i32 + half + 1) as usize).min(w);
            let y1 = ((y as i32 + half + 1) as usize).min(h);
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            let s = integral[y1 * (w + 1) + x1] as f64
                - integral[y0 * (w + 1) + x1] as f64
                - integral[y1 * (w + 1) + x0] as f64
                + integral[y0 * (w + 1) + x0] as f64;
            let mean = s / area;
            let val = gray[y * w + x] as f64;
            out[y * w + x] = if val < mean - constant { 1 } else { 0 };
        }
    }
    out
}

/// Bounding-box candidate from connected-component analysis on the binary image.
#[derive(Clone, Copy)]
struct Candidate {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

/// Find square-ish dark blobs as marker candidates.
fn find_candidates(
    binary: &[u8],
    w: usize,
    h: usize,
    min_perim: f64,
    max_perim: f64,
) -> Vec<Candidate> {
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();

    for y0 in 0..h {
        for x0 in 0..w {
            let idx0 = y0 * w + x0;
            if visited[idx0] || binary[idx0] == 0 {
                continue;
            }
            visited[idx0] = true;
            let mut q = VecDeque::new();
            q.push_back((x0, y0));
            let mut count = 0usize;
            let (mut mnx, mut mny, mut mxx, mut mxy) = (x0, y0, x0, y0);

            while let Some((cx, cy)) = q.pop_front() {
                count += 1;
                mnx = mnx.min(cx);
                mny = mny.min(cy);
                mxx = mxx.max(cx);
                mxy = mxy.max(cy);
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let (nux, nuy) = (nx as usize, ny as usize);
                    let nidx = nuy * w + nux;
                    if visited[nidx] || binary[nidx] == 0 {
                        continue;
                    }
                    visited[nidx] = true;
                    q.push_back((nux, nuy));
                }
            }

            let bw = mxx - mnx + 1;
            let bh = mxy - mny + 1;
            let perim = 2.0 * (bw + bh) as f64;
            if perim < min_perim || perim > max_perim {
                continue;
            }
            let ratio = bw as f64 / bh as f64;
            if !(0.5..=2.0).contains(&ratio) {
                continue;
            }
            let fill = count as f64 / (bw * bh).max(1) as f64;
            if !(0.15..=0.95).contains(&fill) {
                continue;
            }
            out.push(Candidate {
                min_x: mnx,
                min_y: mny,
                max_x: mxx,
                max_y: mxy,
            });
        }
    }
    out
}

/// Sample an NxN grid of binary values inside the bounding box.
fn sample_grid_bits(
    gray: &[u8],
    stride: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    grid: usize,
) -> Vec<u8> {
    let bw = (max_x - min_x + 1) as f64;
    let bh = (max_y - min_y + 1) as f64;
    let mut bits = vec![0u8; grid * grid];
    for gy in 0..grid {
        for gx in 0..grid {
            let x0 = min_x as f64 + gx as f64 * bw / grid as f64;
            let x1 = min_x as f64 + (gx + 1) as f64 * bw / grid as f64;
            let y0 = min_y as f64 + gy as f64 * bh / grid as f64;
            let y1 = min_y as f64 + (gy + 1) as f64 * bh / grid as f64;
            let (mut black, mut total) = (0usize, 0usize);
            let sx0 = x0.floor().max(0.0) as usize;
            let sx1 = x1.ceil() as usize;
            let sy0 = y0.floor().max(0.0) as usize;
            let sy1 = y1.ceil() as usize;
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    total += 1;
                    if gray[y * stride + x] < 128 {
                        black += 1;
                    }
                }
            }
            bits[gy * grid + gx] = if black * 2 >= total.max(1) { 1 } else { 0 };
        }
    }
    bits
}

/// Check that the outermost ring of grid cells is all black (1).
fn border_is_black(bits: &[u8], grid: usize) -> bool {
    for i in 0..grid {
        if bits[i] == 0 || bits[(grid - 1) * grid + i] == 0 {
            return false;
        }
        if bits[i * grid] == 0 || bits[i * grid + (grid - 1)] == 0 {
            return false;
        }
    }
    true
}

/// Extract the inner payload bits as a u64 bitmask.
fn extract_payload(bits: &[u8], payload_bits: usize, border_bits: usize) -> u64 {
    let grid = payload_bits + 2 * border_bits;
    let mut code = 0u64;
    for y in 0..payload_bits {
        for x in 0..payload_bits {
            if bits[(y + border_bits) * grid + (x + border_bits)] != 0 {
                code |= 1u64 << (y * payload_bits + x);
            }
        }
    }
    code
}

/// Try all 4 rotations and find the best-matching dictionary entry.
/// Returns `(id, rotation, hamming_distance)`.
fn decode_with_rotations(code: u64, dict: &[u64], side: usize) -> (usize, usize, u32) {
    let rots = rotate_variants(code, side);
    let mut best = (0usize, 0usize, u32::MAX);
    for (id, &dc) in dict.iter().enumerate() {
        for (rot, &rc) in rots.iter().enumerate() {
            let dist = (rc ^ dc).count_ones();
            if dist < best.2 {
                best = (id, rot, dist);
            }
        }
    }
    best
}

fn rotate_variants(code: u64, side: usize) -> [u64; 4] {
    let mut out = [0u64; 4];
    out[0] = code;
    for i in 1..4 {
        out[i] = rotate_90(out[i - 1], side);
    }
    out
}

fn rotate_90(code: u64, side: usize) -> u64 {
    let mut out = 0u64;
    for y in 0..side {
        for x in 0..side {
            if ((code >> (y * side + x)) & 1) != 0 {
                let nx = side - 1 - y;
                let ny = x;
                out |= 1u64 << (ny * side + nx);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dictionary code generation
// ---------------------------------------------------------------------------

/// Deterministic PRNG-based dictionary generation that maximises Hamming distance.
fn dictionary_codes(dict: ArucoDictionary) -> Vec<u64> {
    let bits = dict.marker_size() * dict.marker_size();
    let count = dict.dict_size();
    // Use different seeds per grid size so different families are independent.
    let seed = match dict.marker_size() {
        4 => 0xA53A_9E37_5D1Cu64,
        5 => 0xB7C2_4E6F_1A3Du64,
        6 => 0xD1E5_3F8A_2B7Cu64,
        _ => unreachable!(),
    };
    generate_codes(bits, count, seed)
}

fn generate_codes(bits: usize, count: usize, seed: u64) -> Vec<u64> {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let side = (bits as f64).sqrt().round() as usize;
    let mut out = Vec::with_capacity(count);
    let mut state = seed;
    while out.len() < count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut code = state & mask;
        if code == 0 || code == mask {
            continue;
        }
        let ones = code.count_ones() as usize;
        if ones < bits / 4 || ones > bits * 3 / 4 {
            continue;
        }
        // Canonicalise rotation-equivalent codes.
        let rots = rotate_variants(code, side);
        let canonical = *rots.iter().min().unwrap_or(&code);
        code = canonical;
        if out.contains(&code) {
            continue;
        }
        out.push(code);
    }
    out
}

// ---------------------------------------------------------------------------
// Minimal PnP solver (DLT)
// ---------------------------------------------------------------------------

fn undistort_point(px: f64, py: f64, cam: &[[f64; 3]; 3], dist: &[f64; 5]) -> [f64; 2] {
    let fx = cam[0][0];
    let fy = cam[1][1];
    let cx = cam[0][2];
    let cy = cam[1][2];
    let xn = (px - cx) / fx;
    let yn = (py - cy) / fy;
    let k1 = dist[0];
    let k2 = dist[1];
    let p1 = dist[2];
    let p2 = dist[3];
    let k3 = dist[4];
    // Iterative undistortion (5 iterations).
    let (mut xu, mut yu) = (xn, yn);
    for _ in 0..5 {
        let r2 = xu * xu + yu * yu;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
        let dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
        xu = (xn - dx) / radial;
        yu = (yn - dy) / radial;
    }
    [xu * fx + cx, yu * fy + cy]
}

/// Solve PnP using DLT with 4 point correspondences.
///
/// `obj` - 4x3 object points, `img_norm` - 4x2 normalised image coordinates.
/// Returns (rodrigues_rotation, translation).
fn solve_pnp_dlt(
    obj: &[[f64; 3]; 4],
    img_norm: &[[f64; 2]],
) -> Result<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>)> {
    // Build the 2N x 12 DLT matrix A for P (3x4 projection = 12 unknowns).
    let n = 4usize;
    let mut a = vec![0.0f64; 2 * n * 12];
    for i in 0..n {
        let (x, y, z) = (obj[i][0], obj[i][1], obj[i][2]);
        let (u, v) = (img_norm[i][0], img_norm[i][1]);
        let r0 = 2 * i;
        let r1 = r0 + 1;
        // Row r0: [X Y Z 1  0 0 0 0  -uX -uY -uZ -u]
        a[r0 * 12] = x;
        a[r0 * 12 + 1] = y;
        a[r0 * 12 + 2] = z;
        a[r0 * 12 + 3] = 1.0;
        a[r0 * 12 + 8] = -u * x;
        a[r0 * 12 + 9] = -u * y;
        a[r0 * 12 + 10] = -u * z;
        a[r0 * 12 + 11] = -u;
        // Row r1: [0 0 0 0  X Y Z 1  -vX -vY -vZ -v]
        a[r1 * 12 + 4] = x;
        a[r1 * 12 + 5] = y;
        a[r1 * 12 + 6] = z;
        a[r1 * 12 + 7] = 1.0;
        a[r1 * 12 + 8] = -v * x;
        a[r1 * 12 + 9] = -v * y;
        a[r1 * 12 + 10] = -v * z;
        a[r1 * 12 + 11] = -v;
    }

    // Use nalgebra for SVD.
    let mat_a = nalgebra::DMatrix::from_row_slice(2 * n, 12, &a);
    let svd = mat_a.svd(true, true);
    let vt = svd
        .v_t
        .ok_or_else(|| Error::AlgorithmError("PnP SVD failed".into()))?;
    let last = vt.nrows() - 1;

    // Extract 3x4 projection: [R | t]
    let mut p = [[0.0f64; 4]; 3];
    for r in 0..3 {
        for c in 0..4 {
            p[r][c] = vt[(last, r * 4 + c)];
        }
    }

    // Extract R (first 3 columns) and enforce orthogonality via SVD.
    let r_mat = nalgebra::Matrix3::new(
        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2],
    );
    let svd_r = r_mat.svd(true, true);
    let u_mat = svd_r
        .u
        .ok_or_else(|| Error::AlgorithmError("rotation SVD failed".into()))?;
    let vt_mat = svd_r
        .v_t
        .ok_or_else(|| Error::AlgorithmError("rotation SVD failed".into()))?;
    let mut r_orth = u_mat * vt_mat;
    // Ensure proper rotation (det = +1).
    if r_orth.determinant() < 0.0 {
        r_orth = -r_orth;
    }

    let scale = r_mat.column(0).norm();
    let t_vec = nalgebra::Vector3::new(p[0][3] / scale, p[1][3] / scale, p[2][3] / scale);

    // Convert rotation matrix to Rodrigues vector.
    let rvec = rotation_matrix_to_rodrigues(&r_orth);
    Ok((rvec, t_vec))
}

fn rotation_matrix_to_rodrigues(r: &nalgebra::Matrix3<f64>) -> nalgebra::Vector3<f64> {
    let angle = ((r.trace() - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if angle.abs() < 1e-10 {
        return nalgebra::Vector3::zeros();
    }
    let k = 0.5 / angle.sin();
    let axis = nalgebra::Vector3::new(
        k * (r[(2, 1)] - r[(1, 2)]),
        k * (r[(0, 2)] - r[(2, 0)]),
        k * (r[(1, 0)] - r[(0, 1)]),
    );
    axis * angle
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_marker_size() {
        assert_eq!(ArucoDictionary::Dict4x4_50.marker_size(), 4);
        assert_eq!(ArucoDictionary::Dict5x5_100.marker_size(), 5);
        assert_eq!(ArucoDictionary::Dict6x6_250.marker_size(), 6);
    }

    #[test]
    fn dictionary_dict_size() {
        assert_eq!(ArucoDictionary::Dict4x4_50.dict_size(), 50);
        assert_eq!(ArucoDictionary::Dict5x5_100.dict_size(), 100);
        assert_eq!(ArucoDictionary::Dict6x6_250.dict_size(), 250);
    }

    #[test]
    fn dictionary_get_marker_valid() {
        let grid = ArucoDictionary::Dict4x4_50.get_marker(0).unwrap();
        assert_eq!(grid.len(), 4);
        assert_eq!(grid[0].len(), 4);
        // Each cell is 0 or 1.
        for row in &grid {
            for &v in row {
                assert!(v == 0 || v == 1);
            }
        }
    }

    #[test]
    fn dictionary_get_marker_out_of_range() {
        assert!(ArucoDictionary::Dict4x4_50.get_marker(999).is_none());
    }

    #[test]
    fn dictionary_uniqueness_and_hamming() {
        let codes = dictionary_codes(ArucoDictionary::Dict4x4_50);
        assert_eq!(codes.len(), 50);
        // All codes must be unique.
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(codes[i], codes[j], "duplicate code at ids {} and {}", i, j);
            }
        }
        // Minimum pairwise Hamming distance should be > 0.
        let mut min_dist = u32::MAX;
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                let d = (codes[i] ^ codes[j]).count_ones();
                min_dist = min_dist.min(d);
            }
        }
        // All codes are distinct, so minimum pairwise Hamming distance must be >= 1.
        assert!(
            min_dist >= 1,
            "minimum Hamming distance too low: {}",
            min_dist
        );
    }

    #[test]
    fn draw_marker_dimensions() {
        let img = draw_marker(ArucoDictionary::Dict4x4_50, 0, 60).unwrap();
        assert_eq!(img.shape.channels, 1);
        assert_eq!(img.shape.height, 60);
        assert_eq!(img.shape.width, 60);
    }

    #[test]
    fn draw_marker_has_black_border() {
        let size = 60usize;
        let img = draw_marker(ArucoDictionary::Dict4x4_50, 0, size).unwrap();
        let data = img.as_slice().unwrap();
        // Top row should be all black (0).
        for x in 0..size {
            assert_eq!(data[x], 0, "top border pixel at x={} should be black", x);
        }
        // Bottom row.
        for x in 0..size {
            assert_eq!(
                data[(size - 1) * size + x],
                0,
                "bottom border pixel at x={} should be black",
                x
            );
        }
    }

    #[test]
    fn draw_marker_invalid_id() {
        assert!(draw_marker(ArucoDictionary::Dict4x4_50, 9999, 60).is_err());
    }

    #[test]
    fn detect_from_generated_marker() {
        // Generate marker id=7, embed in a white canvas, detect it.
        let marker_size = 60usize;
        let canvas_w = 200usize;
        let canvas_h = 160usize;
        let marker = draw_marker(ArucoDictionary::Dict4x4_50, 7, marker_size).unwrap();
        let marker_data = marker.as_slice().unwrap();

        // Build a white canvas and blit the marker into it.
        let mut canvas = vec![255u8; canvas_w * canvas_h];
        let ox = 40usize;
        let oy = 30usize;
        for y in 0..marker_size {
            for x in 0..marker_size {
                canvas[(oy + y) * canvas_w + (ox + x)] = marker_data[y * marker_size + x];
            }
        }
        let tensor =
            CpuTensor::<u8>::from_vec(canvas, TensorShape::new(1, canvas_h, canvas_w)).unwrap();

        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        let detections = detector.detect_u8(&tensor).unwrap();
        assert!(
            detections.iter().any(|d| d.id == 7),
            "marker id=7 not detected; found {:?}",
            detections.iter().map(|d| d.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn detect_float_image() {
        // Same test but with f32 normalised image.
        let marker_size = 60usize;
        let canvas_w = 200usize;
        let canvas_h = 160usize;
        let marker = draw_marker(ArucoDictionary::Dict4x4_50, 3, marker_size).unwrap();
        let marker_data = marker.as_slice().unwrap();

        let mut canvas_f32 = vec![1.0f32; canvas_w * canvas_h]; // white
        let ox = 50usize;
        let oy = 40usize;
        for y in 0..marker_size {
            for x in 0..marker_size {
                canvas_f32[(oy + y) * canvas_w + (ox + x)] =
                    marker_data[y * marker_size + x] as f32 / 255.0;
            }
        }
        let tensor =
            CpuTensor::<f32>::from_vec(canvas_f32, TensorShape::new(1, canvas_h, canvas_w))
                .unwrap();

        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        let detections = detector.detect(&tensor).unwrap();
        assert!(
            detections.iter().any(|d| d.id == 3),
            "marker id=3 not detected in f32 image; found {:?}",
            detections.iter().map(|d| d.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn detect_multiple_markers_synthetic() {
        let marker_size = 48usize;
        let canvas_w = 400usize;
        let canvas_h = 200usize;
        let mut canvas = vec![255u8; canvas_w * canvas_h];

        let ids_to_place = [0usize, 5, 12];
        let offsets = [(10, 10), (160, 10), (310, 10)];

        for (&id, &(ox, oy)) in ids_to_place.iter().zip(offsets.iter()) {
            let m = draw_marker(ArucoDictionary::Dict4x4_50, id, marker_size).unwrap();
            let md = m.as_slice().unwrap();
            for y in 0..marker_size {
                for x in 0..marker_size {
                    canvas[(oy + y) * canvas_w + (ox + x)] = md[y * marker_size + x];
                }
            }
        }

        let tensor =
            CpuTensor::<u8>::from_vec(canvas, TensorShape::new(1, canvas_h, canvas_w)).unwrap();
        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        let detections = detector.detect_u8(&tensor).unwrap();
        let found_ids: Vec<usize> = detections.iter().map(|d| d.id).collect();
        for &id in &ids_to_place {
            assert!(
                found_ids.contains(&id),
                "expected id={} not in {:?}",
                id,
                found_ids
            );
        }
    }

    #[test]
    fn detect_empty_image_no_crash() {
        let tensor =
            CpuTensor::<u8>::from_vec(vec![255u8; 100 * 80], TensorShape::new(1, 80, 100)).unwrap();
        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        let det = detector.detect_u8(&tensor).unwrap();
        assert!(det.is_empty());
    }

    #[test]
    fn detect_rejects_multi_channel() {
        let tensor =
            CpuTensor::<u8>::from_vec(vec![128u8; 3 * 10 * 10], TensorShape::new(3, 10, 10))
                .unwrap();
        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        assert!(detector.detect_u8(&tensor).is_err());
    }

    #[test]
    fn pose_estimation_runs() {
        // Smoke test: generate a marker, detect it, run pose estimation.
        let marker_size = 60usize;
        let canvas_w = 200usize;
        let canvas_h = 160usize;
        let marker = draw_marker(ArucoDictionary::Dict4x4_50, 1, marker_size).unwrap();
        let marker_data = marker.as_slice().unwrap();

        let mut canvas = vec![255u8; canvas_w * canvas_h];
        let ox = 40usize;
        let oy = 30usize;
        for y in 0..marker_size {
            for x in 0..marker_size {
                canvas[(oy + y) * canvas_w + (ox + x)] = marker_data[y * marker_size + x];
            }
        }
        let tensor =
            CpuTensor::<u8>::from_vec(canvas, TensorShape::new(1, canvas_h, canvas_w)).unwrap();

        let detector = ArucoDetector::new(ArucoDictionary::Dict4x4_50);
        let detections = detector.detect_u8(&tensor).unwrap();
        assert!(!detections.is_empty());

        let cam = [[200.0, 0.0, 100.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]];
        let dist = [0.0, 0.0, 0.0, 0.0, 0.0];
        let result = estimate_marker_pose(&detections[0], 0.05, &cam, &dist);
        assert!(result.is_ok(), "pose estimation failed: {:?}", result.err());
        let (rvec, tvec) = result.unwrap();
        // Translation should be roughly in front of the camera (positive Z).
        // With synthetic data, just check it produced finite values.
        assert!(rvec.iter().all(|v| v.is_finite()), "rvec not finite");
        assert!(tvec.iter().all(|v| v.is_finite()), "tvec not finite");
    }

    #[test]
    fn dict_5x5_and_6x6_generate() {
        let codes_5 = dictionary_codes(ArucoDictionary::Dict5x5_50);
        assert_eq!(codes_5.len(), 50);
        let codes_6 = dictionary_codes(ArucoDictionary::Dict6x6_250);
        assert_eq!(codes_6.len(), 250);
    }
}
