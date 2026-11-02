//! ArUco, AprilTag and ChArUco helpers on [`image::GrayImage`].
//!
//! This is the image-API front end of [`crate::aruco`]: the dictionaries are
//! [`crate::aruco::ArucoDictionary`] (OpenCV's predefined `DICT_*`
//! dictionaries, AprilTag families included), markers are drawn by
//! [`crate::aruco::draw_marker`] and detected by
//! [`crate::aruco::ArucoDetector`] with OpenCV's default parameters, so
//! markers drawn here are OpenCV's and markers drawn by OpenCV are read here.
//! ChArUco boards are rendered as OpenCV's `CharucoBoard::generateImage`.

use crate::Result;
use crate::aruco::{ArucoDetector, DetectedMarker, draw_marker};
use cv_core::{CpuTensor, Error, TensorShape};
use image::{GrayImage, Luma};
use nalgebra::{Matrix3, Point2, Vector3};

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::*;

pub use crate::aruco::ArucoDictionary;

/// Result of detecting a single ArUco marker in an image.
#[derive(Debug, Clone, Copy)]
pub struct ArucoDetection {
    /// Dictionary ID of the detected marker.
    pub id: u16,
    /// Quarter turns clockwise that bring the marker, as it appears in the
    /// image, upright (0 when it is upright): the marker appears turned
    /// counter-clockwise by this many quarter turns. Turning the bits read in
    /// image orientation clockwise this many times gives the dictionary code
    /// (OpenCV's identification rotation).
    pub rotation_cw: u8,
    /// Image-space corners of the marker, starting at the marker's own
    /// top-left corner and going clockwise (top-left, top-right,
    /// bottom-right, bottom-left of the marker as drawn), whatever its
    /// rotation in the image: OpenCV's `detectMarkers` corners. Pixel centres
    /// are at integer coordinates.
    pub corners: [Point2<f32>; 4],
    /// Image-space centroid of the four corners.
    pub center: Point2<f32>,
}

/// AprilTag family variant: OpenCV's `DICT_APRILTAG_*` dictionaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprilTagFamily {
    /// 16-bit payload, minimum Hamming distance 5 (`DICT_APRILTAG_16h5`).
    Tag16h5,
    /// 25-bit payload, minimum Hamming distance 9 (`DICT_APRILTAG_25h9`).
    Tag25h9,
    /// 36-bit payload, minimum Hamming distance 10 (`DICT_APRILTAG_36h10`).
    Tag36h10,
    /// 36-bit payload, minimum Hamming distance 11 (`DICT_APRILTAG_36h11`).
    Tag36h11,
}

impl AprilTagFamily {
    /// The dictionary holding this family's codes.
    pub fn dictionary(&self) -> ArucoDictionary {
        match self {
            Self::Tag16h5 => ArucoDictionary::AprilTag16h5,
            Self::Tag25h9 => ArucoDictionary::AprilTag25h9,
            Self::Tag36h10 => ArucoDictionary::AprilTag36h10,
            Self::Tag36h11 => ArucoDictionary::AprilTag36h11,
        }
    }
}

/// Result of detecting a single AprilTag in an image.
#[derive(Debug, Clone, Copy)]
pub struct AprilTagDetection {
    /// Dictionary ID of the detected tag.
    pub id: u16,
    /// Quarter turns clockwise that bring the tag upright, as
    /// [`ArucoDetection::rotation_cw`].
    pub rotation_cw: u8,
    /// Hamming distance to the tag's code. Always 0: OpenCV's AprilTag
    /// dictionaries allow no bit correction, so only exact codes are read.
    pub hamming: u8,
    /// Image-space corners of the tag, in the tag's own orientation, as
    /// [`ArucoDetection::corners`].
    pub corners: [Point2<f32>; 4],
    /// Image-space centroid of the four corners.
    pub center: Point2<f32>,
}

/// A ChArUco calibration board composed of a chessboard pattern with embedded ArUco markers.
///
/// The layout is OpenCV's `cv::aruco::CharucoBoard` (non-legacy pattern):
/// square `(x, y)` is black when `x + y` is even, and the markers sit on the
/// white squares in row-major order. With the default `marker_ids`
/// (`0, 1, 2, ...`) it is the board of [`crate::charuco::CharucoBoard`] with
/// the same dimensions.
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
        return Err(Error::AlgorithmError(
            "charuco board must be at least 2x2 squares".to_string(),
        ));
    }
    if !square_length.is_finite()
        || !marker_length.is_finite()
        || square_length <= 0.0
        || marker_length <= 0.0
        || marker_length >= square_length
    {
        return Err(Error::AlgorithmError(
            "invalid square/marker length for charuco board".to_string(),
        ));
    }

    let mut marker_cells = Vec::new();
    for y in 0..squares_y {
        for x in 0..squares_x {
            // OpenCV: black squares where x + y is even, markers on the others.
            if (x + y) % 2 == 1 {
                marker_cells.push((x, y));
            }
        }
    }
    let dict_size = dictionary.dict_size();
    if marker_cells.len() > dict_size {
        return Err(Error::AlgorithmError(format!(
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
/// Each chessboard square is drawn at `pixel_per_square` pixels wide/tall,
/// with no margin: OpenCV's
/// `board.generateImage((squares_x * pps, squares_y * pps), marginSize=0, borderBits=1)`.
/// Returns an error if `pixel_per_square` is 0 or the markers would be
/// smaller than one pixel per cell.
pub fn draw_charuco_board(board: &CharucoBoard, pixel_per_square: u32) -> Result<GrayImage> {
    if pixel_per_square == 0 {
        return Err(Error::AlgorithmError(
            "pixel_per_square must be >= 1".to_string(),
        ));
    }
    generate_charuco_board_image(
        board,
        board.squares_x * pixel_per_square,
        board.squares_y * pixel_per_square,
        0,
    )
}

/// Render a ChArUco board into a `width` x `height` image with a white
/// margin of at least `margin_size` pixels, pixel for pixel as OpenCV 4.13's
/// `CharucoBoard::generateImage((width, height), img, margin_size, 1)`.
///
/// The board keeps its aspect ratio and is centred in the area inside the
/// margin. Returns an error if that area is empty or the markers would be
/// smaller than one pixel per cell.
pub fn generate_charuco_board_image(
    board: &CharucoBoard,
    width: u32,
    height: u32,
    margin_size: u32,
) -> Result<GrayImage> {
    let (w, h, margin) = (width as i64, height as i64, margin_size as i64);
    let (cols, rows) = (w - 2 * margin, h - 2 * margin);
    if cols <= 0 || rows <= 0 {
        return Err(Error::AlgorithmError(
            "charuco board image is empty inside its margin".to_string(),
        ));
    }
    let mut img = GrayImage::from_pixel(width, height, Luma([255]));
    let (sx, sy) = (board.squares_x as i64, board.squares_y as i64);
    // OpenCV's `cvRound` on a `float`: round half to even.
    let round = |v: f32| v.round_ties_even() as i64;

    // The chessboard zone: the board scaled to fit inside the margin,
    // centred along the axis with room to spare. `f32`, as OpenCV.
    let px_x = cols as f32 / sx as f32;
    let px_y = rows as f32 / sy as f32;
    let (square_px, zone) = if px_x <= px_y {
        let zone_h = round(px_x * sy as f32);
        let pad = (rows - zone_h) / 2;
        (px_x, Rect::new(margin, margin + pad, cols, zone_h))
    } else {
        let zone_w = round(px_y * sx as f32);
        let pad = (cols - zone_w) / 2;
        (px_y, Rect::new(margin + pad, margin, zone_w, rows))
    };

    // The markers' zone: from the first marker's top-left corner to the last
    // marker's bottom-right one.
    let marker_px = board.marker_length / board.square_length * square_px;
    let inset_px = 0.5 * (square_px - marker_px);
    let start = round(inset_px);
    let end_x = round(square_px * (sx - 1) as f32 + inset_px + marker_px);
    let end_y = round(square_px * (sy - 1) as f32 + inset_px + marker_px);
    let markers = Rect::new(zone.x + start, zone.y + start, end_x - start, end_y - start);
    draw_board_markers(board, &mut img, markers)?;

    // Black squares where x + y is even.
    for y in 0..sy {
        for x in 0..sx {
            if (x + y) % 2 != 0 {
                continue;
            }
            let (x0, y0) = (square_px * x as f32, square_px * y as f32);
            let (c0, c1) = (round(x0), round(x0 + square_px));
            let (r0, r1) = (round(y0), round(y0 + square_px));
            for r in r0.max(0)..r1.min(zone.h) {
                for c in c0.max(0)..c1.min(zone.w) {
                    img.put_pixel((zone.x + c) as u32, (zone.y + r) as u32, Luma([0]));
                }
            }
        }
    }
    Ok(img)
}

/// Integer pixel rectangle.
#[derive(Debug, Clone, Copy)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

impl Rect {
    fn new(x: i64, y: i64, w: i64, h: i64) -> Self {
        Self { x, y, w, h }
    }
}

/// OpenCV's `Board::Impl::generateImage` (no margin) for the axis-aligned
/// marker layout of a ChArUco board, painting into `area` of `img`: the
/// markers' bounding box is stretched over `area`, and every marker is drawn
/// by `generateImageMarker` at its mapped top-left corner with the mapped
/// width as its side. The arithmetic is OpenCV's `f32` arithmetic on the
/// marker object points, with `cvRound` where OpenCV converts to pixels.
fn draw_board_markers(board: &CharucoBoard, img: &mut GrayImage, area: Rect) -> Result<()> {
    let too_small =
        || Error::AlgorithmError("charuco board image too small for its markers".to_string());
    if area.w <= 0 || area.h <= 0 {
        return Err(too_small());
    }
    // Marker object points (top-left, top-right), as
    // `CharucoBoardImpl::createCharucoBoard` computes them in `f32`.
    let (l, m) = (board.square_length, board.marker_length);
    let d = (l - m) / 2.0;
    let obj: Vec<[(f32, f32); 2]> = board
        .marker_cells
        .iter()
        .map(|&(cx, cy)| {
            let c0 = (cx as f32 * l + d, cy as f32 * l + d);
            [c0, (c0.0 + m, c0.1)]
        })
        .collect();
    // The bounding box of all four corners of every marker.
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for &[(x, y), _] in &obj {
        min_x = min_x.min(x);
        max_x = max_x.max(x + m);
        min_y = min_y.min(y);
        max_y = max_y.max(y + m);
    }
    let (size_x, size_y) = (max_x - min_x, max_y - min_y);

    for (corners, &id) in obj.iter().zip(&board.marker_ids) {
        let [p0, p1] = corners.map(|(x, y)| {
            (
                (x - min_x) / size_x * area.w as f32,
                (y - min_y) / size_y * area.h as f32,
            )
        });
        // `dst_sz`: the rounded top edge length (the diagonal, OpenCV's
        // "height", is always the longer one).
        let width = f64::from(p1.0 - p0.0).hypot(f64::from(p1.1 - p0.1)) as f32;
        let side = width.round_ties_even() as i64;
        let (ox, oy) = (p0.0.round_ties_even() as i64, p0.1.round_ties_even() as i64);
        if side < dictionary_cells(board.dictionary) as i64
            || ox < 0
            || oy < 0
            || ox + side > area.w
            || oy + side > area.h
        {
            return Err(too_small());
        }
        let marker = draw_marker(board.dictionary, id as usize, side as usize)?;
        let marker = marker.as_slice()?;
        for (r, row) in marker.chunks(side as usize).enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let (x, y) = (area.x + ox + c as i64, area.y + oy + r as i64);
                img.put_pixel(x as u32, y as u32, Luma([v]));
            }
        }
    }
    Ok(())
}

/// Cells per marker side, border included.
fn dictionary_cells(dictionary: ArucoDictionary) -> usize {
    dictionary.marker_size() + 2
}

/// Detect ChArUco interior corners in an image.
///
/// First detects ArUco markers to estimate a board-to-image homography, then
/// projects all interior chessboard corners into the image. Returns an empty
/// list if no marker of the board is found. The corners are not refined; use
/// [`crate::charuco::CharucoDetector`] for OpenCV's `detectBoard` with
/// sub-pixel refinement.
#[allow(clippy::needless_range_loop)]
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
            // Detected corners are in the marker's own orientation, matching
            // the object corners.
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
/// `cell_size` is the pixel size of each grid cell, the 1-cell black border
/// included; the image is OpenCV's
/// `generateImageMarker(dictionary, id, (marker_size + 2) * cell_size)`.
/// Returns an error if `id` is out of range for the dictionary or
/// `cell_size` is 0.
pub fn draw_aruco_marker(
    dictionary: ArucoDictionary,
    id: u16,
    cell_size: u32,
) -> Result<GrayImage> {
    draw_marker_image(dictionary, id, cell_size)
}

/// Detect all ArUco markers belonging to the given dictionary in an image.
///
/// Runs [`ArucoDetector`] (OpenCV's `detectMarkers` pipeline, default
/// parameters, no corner refinement). Markers need a white quiet zone around
/// their black border.
pub fn detect_aruco_markers(
    image: &GrayImage,
    dictionary: ArucoDictionary,
) -> Result<Vec<ArucoDetection>> {
    Ok(detect(image, dictionary)?
        .into_iter()
        .map(|m| {
            let (id, rotation_cw, corners, center) = detection_fields(&m);
            ArucoDetection {
                id,
                rotation_cw,
                corners,
                center,
            }
        })
        .collect())
}

/// Render a single AprilTag to a grayscale image.
///
/// `cell_size` is the pixel size of each grid cell, the 1-cell black border
/// included; the image is OpenCV's `generateImageMarker` for the family's
/// `DICT_APRILTAG_*` dictionary (no white outer ring: leave a white quiet
/// zone around it). Returns an error if `id` is out of range for the family
/// or `cell_size` is 0.
pub fn draw_apriltag(family: AprilTagFamily, id: u16, cell_size: u32) -> Result<GrayImage> {
    draw_marker_image(family.dictionary(), id, cell_size)
}

/// Detect all AprilTags of the given family in an image.
///
/// Runs [`ArucoDetector`] with the family's `DICT_APRILTAG_*` dictionary, as
/// OpenCV's `detectMarkers` does: only exact codes are accepted.
pub fn detect_apriltags(
    image: &GrayImage,
    family: AprilTagFamily,
) -> Result<Vec<AprilTagDetection>> {
    Ok(detect(image, family.dictionary())?
        .into_iter()
        .map(|m| {
            let (id, rotation_cw, corners, center) = detection_fields(&m);
            AprilTagDetection {
                id,
                rotation_cw,
                hamming: 0,
                corners,
                center,
            }
        })
        .collect())
}

fn draw_marker_image(dictionary: ArucoDictionary, id: u16, cell_size: u32) -> Result<GrayImage> {
    if cell_size == 0 {
        return Err(Error::AlgorithmError("cell_size must be >= 1".to_string()));
    }
    let size = dictionary_cells(dictionary) * cell_size as usize;
    let marker = draw_marker(dictionary, id as usize, size)
        .map_err(|_| Error::AlgorithmError(format!("invalid marker id: {}", id)))?;
    GrayImage::from_raw(size as u32, size as u32, marker.as_slice()?.to_vec())
        .ok_or_else(|| Error::AlgorithmError("marker image size".to_string()))
}

fn detect(image: &GrayImage, dictionary: ArucoDictionary) -> Result<Vec<DetectedMarker>> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w == 0 || h == 0 {
        return Err(Error::AlgorithmError("empty image".to_string()));
    }
    let tensor = CpuTensor::from_vec(image.as_raw().clone(), TensorShape::new(1, h, w))?;
    ArucoDetector::new(dictionary).detect_u8(&tensor)
}

/// `(id, rotation_cw, corners, center)` of a detected marker.
fn detection_fields(m: &DetectedMarker) -> (u16, u8, [Point2<f32>; 4], Point2<f32>) {
    let corners = m.corners.map(|(x, y)| Point2::new(x as f32, y as f32));
    let center = Point2::new(
        corners.iter().map(|p| p.x).sum::<f32>() / 4.0,
        corners.iter().map(|p| p.y).sum::<f32>() / 4.0,
    );
    // The marker's top edge (corner 0 to 1) points along +x when upright and
    // turns with the marker; `y` grows downwards, so an angle of +90 degrees
    // is a clockwise quarter turn.
    let (dx, dy) = (corners[1].x - corners[0].x, corners[1].y - corners[0].y);
    let turns_cw = (dy.atan2(dx).to_degrees() / 90.0).round() as i32;
    let rotation_cw = (4 - turns_cw.rem_euclid(4)) % 4;
    (m.id as u16, rotation_cw as u8, corners, center)
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
        return Err(Error::AlgorithmError(
            "estimate_homography needs >=4 correspondences".to_string(),
        ));
    }
    // Normalised DLT, shared with `cv-calib3d`. This used the raw (unnormalised)
    // system, which is badly conditioned here because the board coordinates are
    // O(1) while the image coordinates are O(1000) pixels.
    let s: Vec<[f64; 2]> = src.iter().map(|p| [p.x, p.y]).collect();
    let d: Vec<[f64; 2]> = dst.iter().map(|p| [p.x, p.y]).collect();
    cv_calib3d::dlt::solve_dlt_homography(&s, &d).ok_or_else(|| {
        Error::AlgorithmError("homography DLT failed (degenerate configuration)".to_string())
    })
}

fn project_homography(h: &Matrix3<f64>, p: &Point2<f64>) -> Point2<f64> {
    let v = h * Vector3::new(p.x, p.y, 1.0);
    if v[2].abs() <= 1e-12 {
        Point2::new(v[0], v[1])
    } else {
        Point2::new(v[0] / v[2], v[1] / v[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::{replace, rotate90};

    #[test]
    fn aruco_draw_and_detect_single_marker() {
        let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, 7, 10).unwrap();
        let mut canvas = GrayImage::from_pixel(180, 140, Luma([255]));
        replace(&mut canvas, &marker, 30, 20);

        let det = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        assert_eq!(det.len(), 1);
        assert_eq!(det[0].id, 7);
        assert_eq!(det[0].rotation_cw, 0);
        let want = [(30.0, 20.0), (89.0, 20.0), (89.0, 79.0), (30.0, 79.0)];
        for (p, (x, y)) in det[0].corners.iter().zip(want) {
            assert_eq!((p.x, p.y), (x, y));
        }
        assert_eq!((det[0].center.x, det[0].center.y), (59.5, 49.5));
    }

    /// Rotation and corner order follow the marker, not the image.
    #[test]
    fn aruco_rotation_and_corner_order() {
        let marker = draw_aruco_marker(ArucoDictionary::Dict5x5_100, 42, 10).unwrap();
        let mut canvas = GrayImage::from_pixel(120, 120, Luma([255]));
        replace(&mut canvas, &marker, 25, 25);
        // `rotate90` turns clockwise; the marker's top-left moves to the
        // image's top-right, and turning it back takes three clockwise turns.
        let mut img = canvas;
        for (turns_cw, top_left) in [(0u8, (25.0, 25.0)), (1, (94.0, 25.0)), (2, (94.0, 94.0))] {
            let det = detect_aruco_markers(&img, ArucoDictionary::Dict5x5_100).unwrap();
            assert_eq!(det.len(), 1);
            assert_eq!(det[0].id, 42);
            assert_eq!(det[0].rotation_cw, (4 - turns_cw) % 4);
            assert_eq!((det[0].corners[0].x, det[0].corners[0].y), top_left);
            img = rotate90(&img);
        }
    }

    #[test]
    fn apriltag_draw_and_detect_single_marker() {
        let marker = draw_apriltag(AprilTagFamily::Tag16h5, 3, 8).unwrap();
        let mut canvas = GrayImage::from_pixel(160, 160, Luma([255]));
        replace(&mut canvas, &marker, 40, 50);

        let det = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        assert_eq!(det.len(), 1);
        assert_eq!(det[0].id, 3);
        assert_eq!(det[0].hamming, 0);
    }

    #[test]
    fn apriltag_all_families_round_trip() {
        for (family, id) in [
            (AprilTagFamily::Tag16h5, 29),
            (AprilTagFamily::Tag25h9, 34),
            (AprilTagFamily::Tag36h10, 2319),
            (AprilTagFamily::Tag36h11, 586),
        ] {
            let marker = draw_apriltag(family, id, 8).unwrap();
            let mut canvas = GrayImage::from_pixel(120, 120, Luma([255]));
            replace(&mut canvas, &marker, 30, 30);
            let det = detect_apriltags(&canvas, family).unwrap();
            assert_eq!(det.len(), 1, "{family:?}");
            assert_eq!(det[0].id, id, "{family:?}");
        }
        let n = AprilTagFamily::Tag36h11.dictionary().dict_size() as u16;
        assert!(draw_apriltag(AprilTagFamily::Tag36h11, n, 8).is_err());
    }

    #[test]
    fn charuco_board_draw_and_detect_corners() {
        let board = create_charuco_board(6, 5, 1.0, 0.75, ArucoDictionary::Dict4x4_50).unwrap();
        let img = draw_charuco_board(&board, 30).unwrap();
        let corners = detect_charuco_corners(&img, &board).unwrap();
        assert_eq!(corners.len(), 20);
        // Interior corner (x + 1, y + 1) lies on the pixel boundary at
        // 30 * (x + 1), i.e. at 30 * (x + 1) - 0.5 in pixel-centre coordinates.
        for c in &corners {
            let (gx, gy) = ((c.id % 5 + 1) as f32, (c.id / 5 + 1) as f32);
            assert!((c.point.x - (30.0 * gx - 0.5)).abs() < 1.0, "{c:?}");
            assert!((c.point.y - (30.0 * gy - 0.5)).abs() < 1.0, "{c:?}");
        }
    }

    #[test]
    fn charuco_board_layout() {
        let board = create_charuco_board(4, 3, 1.0, 0.5, ArucoDictionary::Dict4x4_50).unwrap();
        assert_eq!(board.marker_ids, (0..6).collect::<Vec<u16>>());
        let img = draw_charuco_board(&board, 40).unwrap();
        assert_eq!(img.dimensions(), (160, 120));
        for y in 0..3u32 {
            for x in 0..4u32 {
                // Square corners: black squares are black up to the edge,
                // white squares keep a white margin around the marker.
                let v = img.get_pixel(x * 40 + 2, y * 40 + 2)[0];
                assert_eq!(v, if (x + y) % 2 == 0 { 0 } else { 255 }, "({x}, {y})");
            }
        }
        let det = detect_aruco_markers(&img, ArucoDictionary::Dict4x4_50).unwrap();
        let mut found: Vec<(u16, u32, u32)> = det
            .iter()
            .map(|d| (d.id, d.center.x as u32 / 40, d.center.y as u32 / 40))
            .collect();
        found.sort();
        assert_eq!(
            found,
            [
                (0, 1, 0),
                (1, 3, 0),
                (2, 0, 1),
                (3, 2, 1),
                (4, 1, 2),
                (5, 3, 2)
            ]
        );
    }

    #[test]
    fn aruco_multiple_markers_detection() {
        let mut canvas = GrayImage::from_pixel(400, 300, Luma([255]));
        let ids = [0u16, 5, 10, 15, 20];
        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, id, 10).unwrap();
            let x = ((i % 3) as i64) * 120 + 20;
            let y = ((i / 3) as i64) * 120 + 20;
            replace(&mut canvas, &marker, x, y);
        }
        let detections = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        let mut detected_ids: Vec<u16> = detections.iter().map(|d| d.id).collect();
        detected_ids.sort();
        assert_eq!(detected_ids, ids);
    }

    #[test]
    fn apriltag_multiple_markers_detection() {
        let mut canvas = GrayImage::from_pixel(400, 300, Luma([255]));
        let ids = [0u16, 5, 10];
        for (i, &id) in ids.iter().enumerate() {
            let marker = draw_apriltag(AprilTagFamily::Tag16h5, id, 8).unwrap();
            replace(&mut canvas, &marker, (i as i64) * 130 + 20, 50);
        }
        let detections = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        let mut detected_ids: Vec<u16> = detections.iter().map(|d| d.id).collect();
        detected_ids.sort();
        assert_eq!(detected_ids, ids);
    }

    #[test]
    fn invalid_inputs_are_errors() {
        assert!(draw_aruco_marker(ArucoDictionary::Dict4x4_50, 50, 10).is_err());
        assert!(draw_aruco_marker(ArucoDictionary::Dict4x4_50, 0, 0).is_err());
        assert!(draw_apriltag(AprilTagFamily::Tag16h5, 30, 8).is_err());
        let board = create_charuco_board(5, 5, 1.0, 0.7, ArucoDictionary::Dict4x4_50).unwrap();
        assert!(draw_charuco_board(&board, 0).is_err());
        assert!(generate_charuco_board_image(&board, 100, 100, 50).is_err());
        // Markers of fewer than six pixels cannot be drawn.
        assert!(draw_charuco_board(&board, 4).is_err());
        assert!(create_charuco_board(1, 5, 1.0, 0.7, ArucoDictionary::Dict4x4_50).is_err());
        assert!(create_charuco_board(5, 5, 1.0, 1.0, ArucoDictionary::Dict4x4_50).is_err());
        // 20 x 10 squares need 100 markers.
        assert!(create_charuco_board(20, 10, 1.0, 0.7, ArucoDictionary::Dict4x4_50).is_err());
        assert!(create_charuco_board(20, 10, 1.0, 0.7, ArucoDictionary::Dict4x4_100).is_ok());
    }

    #[test]
    fn marker_detection_empty_candidates() {
        let canvas = GrayImage::from_pixel(200, 200, Luma([255]));
        let aruco_result = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50);
        assert!(aruco_result.unwrap().is_empty());
        let apriltag_result = detect_apriltags(&canvas, AprilTagFamily::Tag16h5);
        assert!(apriltag_result.unwrap().is_empty());
        assert!(detect_aruco_markers(&GrayImage::new(0, 0), ArucoDictionary::Dict4x4_50).is_err());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_memory_budget_validation() {
        // Test GPU memory budget parsing and validation
        use cv_hal::gpu_utils;

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

        assert!(gpu_utils::fits_in_budget(100, Some(200)));
        assert!(!gpu_utils::fits_in_budget(200, Some(100)));
        assert!(gpu_utils::fits_in_budget(1000, None)); // No limit

        let size = gpu_utils::estimate_image_buffer_size(1920, 1080, 4);
        assert_eq!(size, 1920 * 1080 * 4);
    }
}
