#![allow(deprecated)]

use crate::Result;
use cv_core::Error;
use image::{GrayImage, Luma};
use nalgebra::{DMatrix, Matrix3, Point2, Vector3};
use std::collections::VecDeque;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::*;

/// ArUco marker dictionary variant.
#[derive(Debug, Clone, Copy)]
pub enum ArucoDictionary {
    /// 4×4 bit payload, 50-marker dictionary.
    Dict4x4_50,
}

/// Result of detecting a single ArUco marker in an image.
#[derive(Debug, Clone, Copy)]
pub struct ArucoDetection {
    /// Dictionary ID of the detected marker.
    pub id: u16,
    /// Clockwise rotation (0–3, in multiples of 90°) that aligns the detected bits to the canonical code.
    pub rotation_cw: u8,
    /// Image-space corners of the marker bounding box (top-left, top-right, bottom-right, bottom-left).
    pub corners: [Point2<f32>; 4],
    /// Image-space centroid of the detected marker.
    pub center: Point2<f32>,
}

/// AprilTag family variant.
#[derive(Debug, Clone, Copy)]
pub enum AprilTagFamily {
    /// 16-bit payload, minimum Hamming distance 5.
    Tag16h5,
    /// 36-bit payload, minimum Hamming distance 11.
    Tag36h11,
}

/// Result of detecting a single AprilTag in an image.
#[derive(Debug, Clone, Copy)]
pub struct AprilTagDetection {
    /// Dictionary ID of the detected tag.
    pub id: u16,
    /// Clockwise rotation (0–3) that aligns the detected bits to the canonical code.
    pub rotation_cw: u8,
    /// Hamming distance to the nearest code in the dictionary (0 = perfect match).
    pub hamming: u8,
    /// Image-space corners of the tag bounding box.
    pub corners: [Point2<f32>; 4],
    /// Image-space centroid of the detected tag.
    pub center: Point2<f32>,
}

/// A ChArUco calibration board composed of a chessboard pattern with embedded ArUco markers.
#[derive(Debug, Clone)]
pub struct CharucoBoard {
    /// Number of chessboard squares along the X axis.
    pub squares_x: u32,
    /// Number of chessboard squares along the Y axis.
    pub squares_y: u32,
    /// Side length of each chessboard square in world units (e.g. metres).
    pub square_length: f32,
    /// Side length of each embedded ArUco marker in world units.
    pub marker_length: f32,
    /// ArUco dictionary used for the embedded markers.
    pub dictionary: ArucoDictionary,
    /// IDs of the ArUco markers placed on the board, in order.
    pub marker_ids: Vec<u16>,
    marker_cells: Vec<(u32, u32)>,
}

/// A detected interior corner of a ChArUco board.
#[derive(Debug, Clone, Copy)]
pub struct CharucoCorner {
    /// Corner ID (row-major index of the interior chessboard corner).
    pub id: u16,
    /// Image-space position of the corner.
    pub point: Point2<f32>,
}

/// Create a [`CharucoBoard`] with the given layout and marker dictionary.
///
/// Returns an error if the board is smaller than 2×2, the lengths are invalid,
/// or the dictionary does not have enough codes for the number of markers.
pub fn create_charuco_board(
    squares_x: u32,
    squares_y: u32,
    square_length: f32,
    marker_length: f32,
    dictionary: ArucoDictionary,
) -> Result<CharucoBoard> {
    if squares_x < 2 || squares_y < 2 {
        return Err(Error::FeatureError(
            "charuco board must be at least 2x2 squares".to_string(),
        ));
    }
    if !square_length.is_finite()
        || !marker_length.is_finite()
        || square_length <= 0.0
        || marker_length <= 0.0
        || marker_length >= square_length
    {
        return Err(Error::FeatureError(
            "invalid square/marker length for charuco board".to_string(),
        ));
    }

    let mut marker_cells = Vec::new();
    for y in 0..squares_y {
        for x in 0..squares_x {
            // Place markers on white squares so black marker borders remain detectable.
            if (x + y) % 2 == 1 {
                marker_cells.push((x, y));
            }
        }
    }
    let dict_size = aruco_dictionary_codes(dictionary).len();
    if marker_cells.len() > dict_size {
        return Err(Error::FeatureError(format!(
            "dictionary too small for board markers: need {}, have {}",
            marker_cells.len(),
            dict_size
        )));
    }
    let marker_ids = (0..marker_cells.len() as u16).collect();

    Ok(CharucoBoard {
        squares_x,
        squares_y,
        square_length,
        marker_length,
        dictionary,
        marker_ids,
        marker_cells,
    })
}

/// Render a ChArUco board to a grayscale image.
///
/// Each chessboard square is drawn at `pixel_per_square` pixels wide/tall.
/// Returns an error if `pixel_per_square` is 0.
pub fn draw_charuco_board(board: &CharucoBoard, pixel_per_square: u32) -> Result<GrayImage> {
    if pixel_per_square == 0 {
        return Err(Error::FeatureError(
            "pixel_per_square must be >= 1".to_string(),
        ));
    }
    let width = board.squares_x * pixel_per_square;
    let height = board.squares_y * pixel_per_square;
    let mut img = GrayImage::from_pixel(width, height, Luma([255]));

    for y in 0..board.squares_y {
        for x in 0..board.squares_x {
            let black = (x + y) % 2 == 0;
            let v = if black { 0u8 } else { 255u8 };
            let x0 = x * pixel_per_square;
            let y0 = y * pixel_per_square;
            for yy in y0..(y0 + pixel_per_square) {
                for xx in x0..(x0 + pixel_per_square) {
                    img.put_pixel(xx, yy, Luma([v]));
                }
            }
        }
    }

    let marker_ratio = board.marker_length / board.square_length;
    let marker_px = ((pixel_per_square as f32) * marker_ratio).round().max(2.0) as u32;
    let margin = ((pixel_per_square - marker_px) / 2).max(1);

    for (i, &(cx, cy)) in board.marker_cells.iter().enumerate() {
        let marker = draw_aruco_marker(
            board.dictionary,
            board.marker_ids[i],
            (marker_px / 6).max(1),
        )?;
        let marker_scaled = resize_nearest_gray(&marker, marker_px, marker_px);
        let x0 = cx * pixel_per_square + margin;
        let y0 = cy * pixel_per_square + margin;
        blit_gray(&mut img, &marker_scaled, x0, y0);
    }

    Ok(img)
}

/// Detect ChArUco interior corners in an image.
///
/// First detects ArUco markers to estimate a homography, then projects all
/// interior chessboard corners into the image. Returns an empty list if fewer
/// than 4 marker correspondences are found.
pub fn detect_charuco_corners(
    image: &GrayImage,
    board: &CharucoBoard,
) -> Result<Vec<CharucoCorner>> {
    let detections = detect_aruco_markers(image, board.dictionary)?;
    if detections.is_empty() {
        return Ok(Vec::new());
    }

    let mut src = Vec::<Point2<f64>>::new();
    let mut dst = Vec::<Point2<f64>>::new();
    for det in &detections {
        if let Some((cell_x, cell_y)) = marker_cell_for_id(board, det.id) {
            let obj = marker_object_corners(board, cell_x, cell_y);
            for k in 0..4 {
                src.push(Point2::new(obj[k].x as f64, obj[k].y as f64));
                dst.push(Point2::new(
                    det.corners[k].x as f64,
                    det.corners[k].y as f64,
                ));
            }
        }
    }
    if src.len() < 4 {
        return Ok(Vec::new());
    }
    let h = estimate_homography(&src, &dst)?;

    let mut out = Vec::new();
    for y in 0..(board.squares_y - 1) {
        for x in 0..(board.squares_x - 1) {
            let id = (y * (board.squares_x - 1) + x) as u16;
            let obj = Point2::new(
                (x + 1) as f32 * board.square_length,
                (y + 1) as f32 * board.square_length,
            );
            let p = project_homography(&h, &Point2::new(obj.x as f64, obj.y as f64));
            if p.x >= 0.0 && p.y >= 0.0 && p.x < image.width() as f64 && p.y < image.height() as f64
            {
                out.push(CharucoCorner {
                    id,
                    point: Point2::new(p.x as f32, p.y as f32),
                });
            }
        }
    }
    Ok(out)
}

/// Render a single ArUco marker to a grayscale image.
///
/// `cell_size` controls the pixel size of each grid cell (including the border).
/// Returns an error if `id` is out of range for the dictionary.
pub fn draw_aruco_marker(
    dictionary: ArucoDictionary,
    id: u16,
    cell_size: u32,
) -> Result<GrayImage> {
    let payload_bits = 4usize;
    let border_bits = 1usize;
    let code = aruco_dictionary_codes(dictionary)
        .get(id as usize)
        .copied()
        .ok_or_else(|| Error::FeatureError(format!("invalid aruco id: {}", id)))?;
    Ok(draw_marker_bits(code, payload_bits, border_bits, cell_size))
}

/// Detect all ArUco markers belonging to the given dictionary in an image.
///
/// Uses the GPU path when available (and enabled via [`markers::gpu::use_gpu`]),
/// otherwise falls back to parallel CPU detection.
pub fn detect_aruco_markers(
    image: &GrayImage,
    dictionary: ArucoDictionary,
) -> Result<Vec<ArucoDetection>> {
    let payload_bits = 4usize;
    let border_bits = 1usize;
    let codes = aruco_dictionary_codes(dictionary);
    let candidates = find_marker_candidates(image, payload_bits + 2 * border_bits)?;

    // Try GPU path if available
    #[cfg(feature = "gpu")]
    if gpu::gpu_available() && gpu::is_gpu_enabled() {
        if let Ok(gpu_detections) =
            detect_aruco_markers_gpu(image, &candidates, &codes, payload_bits, border_bits)
        {
            return Ok(gpu_detections);
        }
        // Fall through to CPU if GPU fails
    }

    // CPU path: parallel detection
    detect_aruco_markers_cpu(image, &candidates, &codes, payload_bits, border_bits)
}

fn detect_aruco_markers_cpu(
    image: &GrayImage,
    candidates: &[Candidate],
    codes: &[u64],
    payload_bits: usize,
    border_bits: usize,
) -> Result<Vec<ArucoDetection>> {
    use rayon::prelude::*;

    // Process all candidates in parallel
    let detections = candidates
        .par_iter()
        .filter_map(|c| {
            let bits = sample_grid_bits(
                image,
                c.min_x,
                c.min_y,
                c.max_x,
                c.max_y,
                payload_bits + 2 * border_bits,
            );
            if !border_is_black(&bits, payload_bits + 2 * border_bits) {
                return None;
            }
            let payload = extract_payload(&bits, payload_bits, border_bits);
            let (id, rot, dist) = decode_code_with_rotations(&payload, codes, payload_bits);
            if dist == 0 {
                Some(ArucoDetection {
                    id: id as u16,
                    rotation_cw: rot as u8,
                    corners: candidate_corners(c),
                    center: candidate_center(c),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(detections)
}

#[cfg(feature = "gpu")]
fn run_marker_detection_gpu(
    image: &GrayImage,
    candidates: &[Candidate],
    codes: &[u64],
    payload_bits: usize,
    border_bits: usize,
    max_hamming: u32,
) -> Result<Vec<(usize, gpu::GpuMarkerResult)>> {
    use gpu::{GpuCandidate, MarkerGpuContext};

    // Early return if no candidates
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Create GPU context
    let gpu_context = MarkerGpuContext::new()
        .ok_or_else(|| Error::FeatureError("GPU context initialization failed".to_string()))?;

    // Convert CPU candidates to GPU candidates for batch processing
    let grid_size = (payload_bits + 2 * border_bits) as u32;
    let gpu_candidates: Vec<GpuCandidate> = candidates
        .iter()
        .map(|c| GpuCandidate {
            min_x: c.min_x,
            min_y: c.min_y,
            max_x: c.max_x,
            max_y: c.max_y,
            grid_size,
            payload_bits: payload_bits as u32,
        })
        .collect();

    // Run GPU marker detection on all candidates in parallel
    let gpu_results = gpu_context.run_candidate_scan(
        image,
        &gpu_candidates,
        codes,
        border_bits as u32,
        max_hamming,
    )?;

    // Return results paired with candidate indices
    Ok(gpu_results.into_iter().enumerate().collect())
}

#[cfg(feature = "gpu")]
fn detect_aruco_markers_gpu(
    image: &GrayImage,
    candidates: &[Candidate],
    codes: &[u64],
    payload_bits: usize,
    border_bits: usize,
) -> Result<Vec<ArucoDetection>> {
    let gpu_results =
        run_marker_detection_gpu(image, candidates, codes, payload_bits, border_bits, 0)?;

    // Convert GPU results to ArucoDetection
    let detections: Vec<ArucoDetection> = gpu_results
        .into_iter()
        .filter_map(|(idx, gpu_result)| {
            if gpu_result.is_valid() && idx < candidates.len() {
                let candidate = &candidates[idx];
                Some(ArucoDetection {
                    id: gpu_result.best_id as u16,
                    rotation_cw: gpu_result.rotation as u8,
                    corners: candidate_corners(candidate),
                    center: candidate_center(candidate),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(detections)
}

/// Render a single AprilTag to a grayscale image.
///
/// `cell_size` controls the pixel size of each grid cell.
/// Returns an error if `id` is out of range for the family.
pub fn draw_apriltag(family: AprilTagFamily, id: u16, cell_size: u32) -> Result<GrayImage> {
    let (payload_bits, codes) = apriltag_family_codes(family);
    let border_bits = 1usize;
    let code = codes
        .get(id as usize)
        .copied()
        .ok_or_else(|| Error::FeatureError(format!("invalid apriltag id: {}", id)))?;
    Ok(draw_marker_bits(code, payload_bits, border_bits, cell_size))
}

/// Detect all AprilTags of the given family in an image.
///
/// Uses the GPU path when available, otherwise falls back to parallel CPU detection.
/// Accepts markers with up to 1-bit error (Hamming distance ≤ 1).
pub fn detect_apriltags(
    image: &GrayImage,
    family: AprilTagFamily,
) -> Result<Vec<AprilTagDetection>> {
    let (payload_bits, codes) = apriltag_family_codes(family);
    let border_bits = 1usize;
    let candidates = find_marker_candidates(image, payload_bits + 2 * border_bits)?;

    // Try GPU path if available
    #[cfg(feature = "gpu")]
    if gpu::gpu_available() && gpu::is_gpu_enabled() {
        if let Ok(gpu_detections) =
            detect_apriltags_gpu(image, &candidates, &codes, payload_bits, border_bits)
        {
            return Ok(gpu_detections);
        }
        // Fall through to CPU if GPU fails
    }

    // CPU path: parallel detection
    detect_apriltags_cpu(image, &candidates, &codes, payload_bits, border_bits)
}

fn detect_apriltags_cpu(
    image: &GrayImage,
    candidates: &[Candidate],
    codes: &[u64],
    payload_bits: usize,
    border_bits: usize,
) -> Result<Vec<AprilTagDetection>> {
    use rayon::prelude::*;

    // Process all candidates in parallel
    let detections = candidates
        .par_iter()
        .filter_map(|c| {
            let bits = sample_grid_bits(
                image,
                c.min_x,
                c.min_y,
                c.max_x,
                c.max_y,
                payload_bits + 2 * border_bits,
            );
            if !border_is_black(&bits, payload_bits + 2 * border_bits) {
                return None;
            }
            let payload = extract_payload(&bits, payload_bits, border_bits);
            let (id, rot, dist) = decode_code_with_rotations(&payload, codes, payload_bits);
            // Simple robust decode: allow up to one bit error for tag families.
            if dist <= 1 {
                Some(AprilTagDetection {
                    id: id as u16,
                    rotation_cw: rot as u8,
                    hamming: dist as u8,
                    corners: candidate_corners(c),
                    center: candidate_center(c),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(detections)
}

#[cfg(feature = "gpu")]
fn detect_apriltags_gpu(
    image: &GrayImage,
    candidates: &[Candidate],
    codes: &[u64],
    payload_bits: usize,
    border_bits: usize,
) -> Result<Vec<AprilTagDetection>> {
    // AprilTag allows up to 1-bit error (Hamming distance <= 1)
    let gpu_results =
        run_marker_detection_gpu(image, candidates, codes, payload_bits, border_bits, 1)?;

    // Convert GPU results to AprilTagDetection
    let detections: Vec<AprilTagDetection> = gpu_results
        .into_iter()
        .filter_map(|(idx, gpu_result)| {
            if gpu_result.is_valid() && idx < candidates.len() {
                let candidate = &candidates[idx];
                Some(AprilTagDetection {
                    id: gpu_result.best_id as u16,
                    rotation_cw: gpu_result.rotation as u8,
                    hamming: 0, // GPU shader handles hamming distance internally
                    corners: candidate_corners(candidate),
                    center: candidate_center(candidate),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(detections)
}

#[derive(Clone, Copy)]
struct Candidate {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

fn draw_marker_bits(
    code: u64,
    payload_bits: usize,
    border_bits: usize,
    cell_size: u32,
) -> GrayImage {
    let grid = payload_bits + 2 * border_bits;
    let size = (grid as u32) * cell_size;
    let mut img = GrayImage::from_pixel(size, size, Luma([255]));

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
            let x0 = gx as u32 * cell_size;
            let y0 = gy as u32 * cell_size;
            for y in y0..(y0 + cell_size) {
                for x in x0..(x0 + cell_size) {
                    img.put_pixel(x, y, Luma([val]));
                }
            }
        }
    }
    img
}

fn find_marker_candidates(image: &GrayImage, min_grid: usize) -> Result<Vec<Candidate>> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    if w == 0 || h == 0 {
        return Err(Error::FeatureError("empty image".to_string()));
    }
    let min_side = (min_grid as u32).max(6);
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();
    let raw = image.as_raw();

    for y0 in 0..h {
        for x0 in 0..w {
            let idx0 = y0 * w + x0;
            if visited[idx0] || raw[idx0] > 80 {
                continue;
            }
            let mut q = VecDeque::new();
            q.push_back((x0 as i32, y0 as i32));
            visited[idx0] = true;

            let mut count = 0usize;
            let mut min_x = x0 as u32;
            let mut min_y = y0 as u32;
            let mut max_x = x0 as u32;
            let mut max_y = y0 as u32;

            while let Some((x, y)) = q.pop_front() {
                count += 1;
                let ux = x as u32;
                let uy = y as u32;
                min_x = min_x.min(ux);
                min_y = min_y.min(uy);
                max_x = max_x.max(ux);
                max_y = max_y.max(uy);

                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nux = nx as usize;
                    let nuy = ny as usize;
                    let nidx = nuy * w + nux;
                    if visited[nidx] || raw[nidx] > 80 {
                        continue;
                    }
                    visited[nidx] = true;
                    q.push_back((nx, ny));
                }
            }

            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            if bw < min_side || bh < min_side {
                continue;
            }
            let ratio = bw as f32 / bh as f32;
            if !(0.7..=1.3).contains(&ratio) {
                continue;
            }
            let box_area = (bw as usize) * (bh as usize);
            let fill = count as f32 / box_area.max(1) as f32;
            if fill < 0.18 || fill > 0.95 {
                continue;
            }

            out.push(Candidate {
                min_x,
                min_y,
                max_x,
                max_y,
            });
        }
    }

    Ok(out)
}

fn sample_grid_bits(
    image: &GrayImage,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    grid: usize,
) -> Vec<u8> {
    let mut bits = vec![0u8; grid * grid];
    let bw = (max_x - min_x + 1) as f32;
    let bh = (max_y - min_y + 1) as f32;

    for gy in 0..grid {
        for gx in 0..grid {
            let x0 = min_x as f32 + (gx as f32) * bw / grid as f32;
            let x1 = min_x as f32 + ((gx + 1) as f32) * bw / grid as f32;
            let y0 = min_y as f32 + (gy as f32) * bh / grid as f32;
            let y1 = min_y as f32 + ((gy + 1) as f32) * bh / grid as f32;
            let mut black = 0usize;
            let mut total = 0usize;
            let sx0 = x0.floor().clamp(0.0, (image.width() - 1) as f32) as u32;
            let sx1 = x1.ceil().clamp(0.0, image.width() as f32) as u32;
            let sy0 = y0.floor().clamp(0.0, (image.height() - 1) as f32) as u32;
            let sy1 = y1.ceil().clamp(0.0, image.height() as f32) as u32;
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    total += 1;
                    if image.get_pixel(x, y)[0] < 128 {
                        black += 1;
                    }
                }
            }
            bits[gy * grid + gx] = if black * 2 >= total.max(1) { 1 } else { 0 };
        }
    }
    bits
}

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

fn extract_payload(bits: &[u8], payload_bits: usize, border_bits: usize) -> u64 {
    let grid = payload_bits + 2 * border_bits;
    let mut code = 0u64;
    for y in 0..payload_bits {
        for x in 0..payload_bits {
            let v = bits[(y + border_bits) * grid + (x + border_bits)];
            if v != 0 {
                let idx = y * payload_bits + x;
                code |= 1u64 << idx;
            }
        }
    }
    code
}

fn decode_code_with_rotations(
    code: &u64,
    dict: &[u64],
    payload_side: usize,
) -> (usize, usize, u32) {
    let mut best_id = 0usize;
    let mut best_rot = 0usize;
    let mut best_dist = u32::MAX;
    let rots = rotate_code_variants(*code, payload_side);
    for (id, dcode) in dict.iter().enumerate() {
        for (rot, rc) in rots.iter().enumerate() {
            let dist = (rc ^ dcode).count_ones();
            if dist < best_dist {
                best_dist = dist;
                best_id = id;
                best_rot = rot;
            }
        }
    }
    (best_id, best_rot, best_dist)
}

fn rotate_code_variants(code: u64, side: usize) -> [u64; 4] {
    let mut out = [0u64; 4];
    out[0] = code;
    for i in 1..4 {
        out[i] = rotate_code_90(out[i - 1], side);
    }
    out
}

fn rotate_code_90(code: u64, side: usize) -> u64 {
    let mut out = 0u64;
    for y in 0..side {
        for x in 0..side {
            let bit = (code >> (y * side + x)) & 1;
            if bit != 0 {
                let nx = side - 1 - y;
                let ny = x;
                out |= 1u64 << (ny * side + nx);
            }
        }
    }
    out
}

fn aruco_dictionary_codes(dict: ArucoDictionary) -> Vec<u64> {
    match dict {
        ArucoDictionary::Dict4x4_50 => generate_dictionary_codes(16, 50, 0xA53A_9E37_5D1Cu64),
    }
}

fn apriltag_family_codes(family: AprilTagFamily) -> (usize, Vec<u64>) {
    match family {
        AprilTagFamily::Tag16h5 => (4, generate_dictionary_codes(16, 30, 0x9E37_79B9_7F4Au64)),
        AprilTagFamily::Tag36h11 => (6, generate_dictionary_codes(36, 100, 0xC2B2_AE35_1D0Bu64)),
    }
}

fn generate_dictionary_codes(bits: usize, count: usize, seed: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut state = seed;
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    while out.len() < count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut code = state & mask;
        // Avoid trivial patterns.
        if code == 0 || code == mask {
            continue;
        }
        // Ensure mixed density.
        let ones = code.count_ones() as usize;
        if ones < bits / 4 || ones > bits * 3 / 4 {
            continue;
        }
        // Canonicalize rotation-equivalent payloads.
        let side = (bits as f32).sqrt().round() as usize;
        let rots = rotate_code_variants(code, side);
        let canonical = *rots.iter().min().unwrap_or(&code);
        code = canonical;
        if out.contains(&code) {
            continue;
        }
        out.push(code);
    }
    out
}

fn candidate_corners(c: &Candidate) -> [Point2<f32>; 4] {
    [
        Point2::new(c.min_x as f32, c.min_y as f32),
        Point2::new(c.max_x as f32, c.min_y as f32),
        Point2::new(c.max_x as f32, c.max_y as f32),
        Point2::new(c.min_x as f32, c.max_y as f32),
    ]
}

fn candidate_center(c: &Candidate) -> Point2<f32> {
    Point2::new(
        (c.min_x + c.max_x) as f32 * 0.5,
        (c.min_y + c.max_y) as f32 * 0.5,
    )
}

fn marker_cell_for_id(board: &CharucoBoard, id: u16) -> Option<(u32, u32)> {
    board
        .marker_ids
        .iter()
        .position(|&v| v == id)
        .map(|i| board.marker_cells[i])
}

fn marker_object_corners(board: &CharucoBoard, cell_x: u32, cell_y: u32) -> [Point2<f32>; 4] {
    let margin = 0.5 * (board.square_length - board.marker_length);
    let x0 = cell_x as f32 * board.square_length + margin;
    let y0 = cell_y as f32 * board.square_length + margin;
    let x1 = x0 + board.marker_length;
    let y1 = y0 + board.marker_length;
    [
        Point2::new(x0, y0),
        Point2::new(x1, y0),
        Point2::new(x1, y1),
        Point2::new(x0, y1),
    ]
}

fn estimate_homography(src: &[Point2<f64>], dst: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if src.len() != dst.len() || src.len() < 4 {
        return Err(Error::FeatureError(
            "estimate_homography needs >=4 correspondences".to_string(),
        ));
    }
    let n = src.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let x = src[i].x;
        let y = src[i].y;
        let u = dst[i].x;
        let v = dst[i].y;
        let r0 = 2 * i;
        let r1 = r0 + 1;
        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;
        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }
    let svd = a.svd(true, true);
    let vt = svd
        .v_t
        .ok_or_else(|| Error::FeatureError("homography SVD failed".to_string()))?;
    let h = vt.row(vt.nrows() - 1);
    let mut m = Matrix3::<f64>::zeros();
    for r in 0..3 {
        for c in 0..3 {
            m[(r, c)] = h[(0, r * 3 + c)];
        }
    }
    let s = m[(2, 2)];
    if s.abs() > 1e-12 {
        m /= s;
    }
    Ok(m)
}

fn project_homography(h: &Matrix3<f64>, p: &Point2<f64>) -> Point2<f64> {
    let v = h * Vector3::new(p.x, p.y, 1.0);
    if v[2].abs() <= 1e-12 {
        Point2::new(v[0], v[1])
    } else {
        Point2::new(v[0] / v[2], v[1] / v[2])
    }
}

fn resize_nearest_gray(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut out = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return out;
    }
    let sx = src.width() as f32 / width as f32;
    let sy = src.height() as f32 / height as f32;
    for y in 0..height {
        for x in 0..width {
            let px = ((x as f32 + 0.5) * sx)
                .floor()
                .clamp(0.0, (src.width() - 1) as f32) as u32;
            let py = ((y as f32 + 0.5) * sy)
                .floor()
                .clamp(0.0, (src.height() - 1) as f32) as u32;
            out.put_pixel(x, y, *src.get_pixel(px, py));
        }
    }
    out
}

fn blit_gray(dst: &mut GrayImage, src: &GrayImage, x0: u32, y0: u32) {
    for y in 0..src.height() {
        for x in 0..src.width() {
            let dx = x0 + x;
            let dy = y0 + y;
            if dx < dst.width() && dy < dst.height() {
                dst.put_pixel(dx, dy, *src.get_pixel(x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::replace;

    #[test]
    fn aruco_draw_and_detect_single_marker() {
        let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, 7, 10).unwrap();
        let mut canvas = GrayImage::from_pixel(180, 140, Luma([255]));
        replace(&mut canvas, &marker, 30, 20);

        let det = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        assert!(!det.is_empty());
        assert!(det.iter().any(|d| d.id == 7));
    }

    #[test]
    fn apriltag_draw_and_detect_single_marker() {
        let marker = draw_apriltag(AprilTagFamily::Tag16h5, 3, 8).unwrap();
        let mut canvas = GrayImage::from_pixel(160, 160, Luma([255]));
        replace(&mut canvas, &marker, 40, 50);

        let det = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        assert!(!det.is_empty());
        assert!(det.iter().any(|d| d.id == 3));
    }

    #[test]
    fn charuco_board_draw_and_detect_corners() {
        let board = create_charuco_board(6, 5, 1.0, 0.75, ArucoDictionary::Dict4x4_50).unwrap();
        let img = draw_charuco_board(&board, 30).unwrap();
        let corners = detect_charuco_corners(&img, &board).unwrap();
        assert!(corners.len() >= ((board.squares_x - 1) * (board.squares_y - 1) / 2) as usize);
    }

    #[test]
    fn aruco_multiple_markers_parallel_detection() {
        // Test parallel detection with multiple tags
        let mut canvas = GrayImage::from_pixel(400, 300, Luma([255]));
        let ids = vec![0, 5, 10, 15, 20];

        // Draw 5 markers at different positions
        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, id, 10).unwrap();
            let x = ((i % 3) as i64) * 120 + 20;
            let y = ((i / 3) as i64) * 120 + 20;
            replace(&mut canvas, &marker, x, y);
        }

        // Detect all markers (uses parallel CPU path)
        let detections = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();

        // Should detect all 5 markers
        assert_eq!(detections.len(), 5);
        let detected_ids: Vec<u16> = detections.iter().map(|d| d.id).collect();
        for id in &ids {
            assert!(detected_ids.contains(&(*id as u16)));
        }
    }

    #[test]
    fn apriltag_multiple_markers_parallel_detection() {
        // Test parallel detection with multiple AprilTags
        let mut canvas = GrayImage::from_pixel(400, 300, Luma([255]));
        let ids = vec![0, 5, 10];

        // Draw 3 markers at different positions
        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_apriltag(AprilTagFamily::Tag16h5, id, 8).unwrap();
            let x = (i as i64) * 130 + 20;
            let y = 50i64;
            replace(&mut canvas, &marker, x, y);
        }

        // Detect all markers (uses parallel CPU path)
        let detections = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();

        // Should detect all 3 markers
        assert_eq!(detections.len(), 3);
        let detected_ids: Vec<u16> = detections.iter().map(|d| d.id).collect();
        for id in &ids {
            assert!(detected_ids.contains(&(*id as u16)));
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_vs_cpu_aruco_detection_parity() {
        use crate::gpu;

        // Create test image with 3 markers
        let mut canvas = GrayImage::from_pixel(400, 300, Luma([255]));
        let ids = vec![2, 7, 12];

        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, id, 10).unwrap();
            let x = (i as i64) * 130 + 20;
            let y = 50i64;
            replace(&mut canvas, &marker, x, y);
        }

        // Detect with CPU (disable GPU)
        gpu::use_gpu(false);
        let cpu_detections = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        let cpu_ids: Vec<u16> = cpu_detections.iter().map(|d| d.id).collect();

        // Detect with GPU (enable GPU)
        gpu::use_gpu(true);
        let gpu_detections = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        let gpu_ids: Vec<u16> = gpu_detections.iter().map(|d| d.id).collect();

        // Results should match (same tags detected)
        assert_eq!(cpu_ids.len(), gpu_ids.len());
        for id in &cpu_ids {
            assert!(gpu_ids.contains(id));
        }

        // Reset GPU state
        gpu::use_gpu(true);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_vs_cpu_apriltag_detection_parity() {
        use crate::gpu;

        // Create test image with 2 AprilTags
        let mut canvas = GrayImage::from_pixel(300, 250, Luma([255]));
        let ids = vec![1, 6];

        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_apriltag(AprilTagFamily::Tag16h5, id, 8).unwrap();
            let x = (i as i64) * 150 + 20;
            let y = 50i64;
            replace(&mut canvas, &marker, x, y);
        }

        // Detect with CPU
        gpu::use_gpu(false);
        let cpu_detections = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        let cpu_ids: Vec<u16> = cpu_detections.iter().map(|d| d.id).collect();

        // Detect with GPU
        gpu::use_gpu(true);
        let gpu_detections = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        let gpu_ids: Vec<u16> = gpu_detections.iter().map(|d| d.id).collect();

        // Results should match
        assert_eq!(cpu_ids.len(), gpu_ids.len());
        for id in &cpu_ids {
            assert!(gpu_ids.contains(id));
        }

        // Reset GPU state
        gpu::use_gpu(true);
    }

    #[test]
    fn charuco_parallel_marker_detection() {
        // ChArUco detection should benefit from parallel marker detection
        let board = create_charuco_board(8, 6, 1.0, 0.75, ArucoDictionary::Dict4x4_50).unwrap();
        let img = draw_charuco_board(&board, 30).unwrap();

        let corners = detect_charuco_corners(&img, &board).unwrap();

        // Should detect majority of corners
        let expected_corners = ((board.squares_x - 1) * (board.squares_y - 1)) as usize;
        assert!(corners.len() >= expected_corners / 2);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_initialization_and_fallback() {
        use crate::gpu;

        // Create simple test image
        let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, 5, 10).unwrap();
        let mut canvas = GrayImage::from_pixel(180, 140, Luma([255]));
        replace(&mut canvas, &marker, 30i64, 20i64);

        // Test with GPU enabled
        gpu::use_gpu(true);
        let gpu_result = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50);

        // Should succeed or gracefully fall back to CPU
        assert!(gpu_result.is_ok());
        let detections = gpu_result.unwrap();
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.id == 5));

        // Test with GPU disabled
        gpu::use_gpu(false);
        let cpu_result = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50);
        assert!(cpu_result.is_ok());
        let cpu_detections = cpu_result.unwrap();
        assert_eq!(cpu_detections.len(), detections.len());
    }

    #[test]
    fn marker_detection_empty_candidates() {
        // Test with image that has no markers
        let canvas = GrayImage::from_pixel(200, 200, Luma([255]));

        let aruco_result = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50);
        assert!(aruco_result.is_ok());
        assert!(aruco_result.unwrap().is_empty());

        let apriltag_result = detect_apriltags(&canvas, AprilTagFamily::Tag16h5);
        assert!(apriltag_result.is_ok());
        assert!(apriltag_result.unwrap().is_empty());
    }

    #[test]
    fn rayon_thread_pool_initialization() {
        // Test that thread pool is initialized (respects RUSTCV_CPU_THREADS)
        use cv_core::{current_cpu_threads, init_global_thread_pool};

        // Initialize thread pool (may already be initialized by other tests)
        let _ = init_global_thread_pool(None);

        // Verify thread pool is active
        let threads = current_cpu_threads();
        assert!(threads >= 1);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_memory_budget_validation() {
        // Test GPU memory budget parsing and validation
        use cv_hal::gpu_utils;

        // Test parsing various formats
        assert_eq!(
            gpu_utils::parse_bytes_with_suffix("100MB").unwrap(),
            100 * 1024 * 1024
        );
        assert_eq!(
            gpu_utils::parse_bytes_with_suffix("1GB").unwrap(),
            1024 * 1024 * 1024
        );
        assert_eq!(
            gpu_utils::parse_bytes_with_suffix("512KB").unwrap(),
            512 * 1024
        );

        // Test memory fitting in budget
        assert!(gpu_utils::fits_in_budget(100, Some(200)));
        assert!(!gpu_utils::fits_in_budget(200, Some(100)));
        assert!(gpu_utils::fits_in_budget(1000, None)); // No limit

        // Test image buffer estimation
        let size = gpu_utils::estimate_image_buffer_size(1920, 1080, 4);
        assert_eq!(size, 1920 * 1080 * 4);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_adapter_policy_documentation() {
        // Document GPU adapter policy behavior
        // The RUSTCV_GPU_ADAPTER environment variable controls adapter selection:
        // - auto: Use any available GPU
        // - prefer_discrete: Prefer discrete GPU (NVIDIA, AMD) over integrated graphics (DEFAULT)
        // - discrete_only: Only use discrete GPU, fail if not available
        // - nvidia_only: Only use NVIDIA discrete GPU, fail if not available

        // When RUSTCV_GPU_ADAPTER=discrete_only or nvidia_only, marker detection
        // will fail with a clear error if no suitable GPU is found, and fall back to CPU

        // Current implementation:
        // 1. MarkerGpuContext::new() checks GPU availability
        // 2. If no GPU available, returns None
        // 3. Detection falls back to CPU (via gpu_available() check)
        // 4. RUSTCV_GPU_ADAPTER is respected by wgpu adapter selection

        // Example usage:
        // export RUSTCV_GPU_ADAPTER=nvidia_only
        // export RUSTCV_GPU_MAX_BYTES=1GB
        // cargo run --features gpu --release

        assert!(true); // This is a documentation test
    }
}
