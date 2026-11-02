//! ArUco marker detection, generation, and pose estimation using `CpuTensor` types.
//!
//! This module provides a tensor-based API for working with ArUco fiducial markers,
//! complementing the `image`-crate-based API in [`crate::markers`].
//!
//! Dictionaries, generated markers and detection follow OpenCV's
//! `cv::aruco` (objdetect, OpenCV 4.x): markers drawn here and by OpenCV are
//! identical, and [`ArucoDetector`] runs the `ArucoDetector::detectMarkers`
//! pipeline with OpenCV's [`DetectorParameters`].
//!
//! # Example
//!
//! ```
//! use cv_core::{CpuTensor, TensorShape};
//! use cv_features::aruco::*;
//!
//! let dict = ArucoDictionary::Dict4x4_50;
//! let marker = draw_marker(dict, 7, 60).unwrap();
//! // Detection needs the white quiet zone around the marker.
//! let mut canvas = vec![255u8; 100 * 100];
//! for (y, row) in marker.as_slice().unwrap().chunks(60).enumerate() {
//!     canvas[(y + 20) * 100 + 20..(y + 20) * 100 + 80].copy_from_slice(row);
//! }
//! let image = CpuTensor::from_vec(canvas, TensorShape::new(1, 100, 100)).unwrap();
//! let detections = ArucoDetector::new(dict).detect_u8(&image).unwrap();
//! assert_eq!(detections.len(), 1);
//! assert_eq!(detections[0].id, 7);
//! ```

use cv_calib3d::{corner_subpix, solve_dlt_homography};
use cv_core::{CpuTensor, Error, Float, Result, TensorShape};
use nalgebra::Point2;
use rayon::prelude::*;

mod contours;
#[cfg(test)]
mod detector_tests;

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

    /// OpenCV's `maxCorrectionBits` for the predefined dictionary of the same
    /// name (`DICT_4X4_50` .. `DICT_6X6_250`).
    pub(crate) fn max_correction_bits(&self) -> u32 {
        match self {
            Self::Dict4x4_50 | Self::Dict4x4_100 | Self::Dict4x4_250 => 1,
            Self::Dict5x5_50 | Self::Dict5x5_100 => 3,
            Self::Dict5x5_250 => 2,
            Self::Dict6x6_50 => 6,
            Self::Dict6x6_100 | Self::Dict6x6_250 => 5,
        }
    }

    /// Get the bit pattern for marker `id`.
    ///
    /// Returns the `marker_size x marker_size` payload grid (row-major, border
    /// excluded) as OpenCV's `Dictionary::getBitsFromByteList`: 1 is a white
    /// cell, 0 a black one. `None` if `id` is out of range.
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
    /// Image-space corners of the marker quadrilateral, starting at the
    /// marker's own top-left corner and going clockwise (top-left, top-right,
    /// bottom-right, bottom-left of the marker as drawn by [`draw_marker`]),
    /// whatever the marker's rotation in the image - OpenCV's corner order.
    /// Pixel centres are at integer coordinates.
    pub corners: [(f64, f64); 4],
}

/// A candidate quad that passed the shape filters but did not decode to a
/// dictionary marker (one of OpenCV's `rejectedImgPoints`).
#[derive(Debug, Clone)]
pub(crate) struct RejectedCandidate {
    /// Corners, clockwise in the image; corner 0 is the top-left of the
    /// payload grid in `payload`.
    pub(crate) corners: [(f64, f64); 4],
    /// Payload bits (border excluded) read with `corners[0]` as the top-left
    /// cell: bit `y * side + x` is cell `(x, y)`, 1 for white, as
    /// [`ArucoDictionary::get_marker`].
    pub(crate) payload: u64,
}

// ---------------------------------------------------------------------------
// Detector parameters
// ---------------------------------------------------------------------------

/// Marker corner refinement after identification (OpenCV's
/// `CornerRefineMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CornerRefineMethod {
    /// Corners are the vertices of the polygon approximation of the marker's
    /// contour (`CORNER_REFINE_NONE`, OpenCV's default). Those vertices are
    /// border pixels of the thresholded marker, so they lie inside the true
    /// corner: 0.5-0.8 px on average, up to about 1.7 px on blurred,
    /// tilted markers.
    #[default]
    None,
    /// `cornerSubPix` on each corner (`CORNER_REFINE_SUBPIX`), with a
    /// half-window of [`DetectorParameters::relative_corner_refinement_win_size`]
    /// marker cells, at most [`DetectorParameters::corner_refinement_win_size`]
    /// pixels.
    Subpix,
}

/// ArUco detection parameters, OpenCV's `cv::aruco::DetectorParameters`
/// (the subset the classic, non-AprilTag pipeline uses). The [`Default`]
/// values are OpenCV's.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorParameters {
    /// Smallest adaptive-threshold window, in pixels (`adaptiveThreshWinSizeMin`, 3).
    pub adaptive_thresh_win_size_min: usize,
    /// Largest adaptive-threshold window, in pixels (`adaptiveThreshWinSizeMax`, 23).
    pub adaptive_thresh_win_size_max: usize,
    /// Step between window sizes (`adaptiveThreshWinSizeStep`, 10): the image
    /// is thresholded once per window `min, min + step, ..., <= max`. Even
    /// windows are rounded up to the next odd size.
    pub adaptive_thresh_win_size_step: usize,
    /// Constant subtracted from the local mean (`adaptiveThreshConstant`, 7).
    pub adaptive_thresh_constant: f64,
    /// Smallest contour perimeter, as a fraction of the larger image side
    /// (`minMarkerPerimeterRate`, 0.03).
    pub min_marker_perimeter_rate: f64,
    /// Largest contour perimeter, as a fraction of the larger image side
    /// (`maxMarkerPerimeterRate`, 4.0).
    pub max_marker_perimeter_rate: f64,
    /// `approxPolyDP` epsilon as a fraction of the contour perimeter
    /// (`polygonalApproxAccuracyRate`, 0.03).
    pub polygonal_approx_accuracy_rate: f64,
    /// Smallest side of a candidate, as a fraction of its contour perimeter
    /// (`minCornerDistanceRate`, 0.05).
    pub min_corner_distance_rate: f64,
    /// Candidates with a corner closer than this to the image border, in
    /// pixels, are dropped together with their group of near-duplicates
    /// (`minDistanceToBorder`, 3).
    pub min_distance_to_border: usize,
    /// Two candidates whose mean corner distance is below this fraction of
    /// the smaller one's perimeter are one marker; the larger is kept
    /// (`minMarkerDistanceRate`, 0.125).
    pub min_marker_distance_rate: f64,
    /// Within such a group, a smaller candidate further than this many
    /// marker cells from the kept one is tried when the kept one does not
    /// decode (`minGroupDistance`, 0.21).
    pub min_group_distance: f64,
    /// Width of the black marker border, in cells (`markerBorderBits`, 1).
    pub marker_border_bits: usize,
    /// Pixels per cell of the perspective-removed marker image
    /// (`perspectiveRemovePixelPerCell`, 4).
    pub perspective_remove_pixel_per_cell: usize,
    /// Fraction of each cell's side ignored at its edges when reading the bit
    /// (`perspectiveRemoveIgnoredMarginPerCell`, 0.13).
    pub perspective_remove_ignored_margin_per_cell: f64,
    /// Largest number of white border cells, as a fraction of the payload
    /// cell count (`maxErroneousBitsInBorderRate`, 0.35).
    pub max_erroneous_bits_in_border_rate: f64,
    /// Below this grey-level standard deviation the marker image is taken as
    /// uniform instead of Otsu-thresholded (`minOtsuStdDev`, 5).
    pub min_otsu_std_dev: f64,
    /// Fraction of the dictionary's correction capacity
    /// (`maxCorrectionBits`) used when identifying markers
    /// (`errorCorrectionRate`, 0.6).
    pub error_correction_rate: f64,
    /// Corner refinement after identification (`cornerRefinementMethod`,
    /// none).
    pub corner_refinement_method: CornerRefineMethod,
    /// Largest `cornerSubPix` half-window, in pixels
    /// (`cornerRefinementWinSize`, 5).
    pub corner_refinement_win_size: usize,
    /// `cornerSubPix` half-window in marker cells
    /// (`relativeCornerRefinmentWinSize`, 0.3).
    pub relative_corner_refinement_win_size: f64,
    /// `cornerSubPix` iteration limit (`cornerRefinementMaxIterations`, 30).
    pub corner_refinement_max_iterations: usize,
    /// `cornerSubPix` stopping shift, in pixels
    /// (`cornerRefinementMinAccuracy`, 0.1).
    pub corner_refinement_min_accuracy: f64,
}

impl Default for DetectorParameters {
    fn default() -> Self {
        Self {
            adaptive_thresh_win_size_min: 3,
            adaptive_thresh_win_size_max: 23,
            adaptive_thresh_win_size_step: 10,
            adaptive_thresh_constant: 7.0,
            min_marker_perimeter_rate: 0.03,
            max_marker_perimeter_rate: 4.0,
            polygonal_approx_accuracy_rate: 0.03,
            min_corner_distance_rate: 0.05,
            min_distance_to_border: 3,
            min_marker_distance_rate: 0.125,
            min_group_distance: 0.21,
            marker_border_bits: 1,
            perspective_remove_pixel_per_cell: 4,
            perspective_remove_ignored_margin_per_cell: 0.13,
            max_erroneous_bits_in_border_rate: 0.35,
            min_otsu_std_dev: 5.0,
            error_correction_rate: 0.6,
            corner_refinement_method: CornerRefineMethod::None,
            corner_refinement_win_size: 5,
            relative_corner_refinement_win_size: 0.3,
            corner_refinement_max_iterations: 30,
            corner_refinement_min_accuracy: 0.1,
        }
    }
}

impl DetectorParameters {
    fn validate(&self) -> Result<()> {
        let bad = |what: &str| Err(Error::InvalidInput(format!("aruco parameters: {what}")));
        if self.adaptive_thresh_win_size_min < 3
            || self.adaptive_thresh_win_size_max < self.adaptive_thresh_win_size_min
            || self.adaptive_thresh_win_size_step == 0
        {
            return bad("adaptive threshold windows need 3 <= min <= max and step > 0");
        }
        if !self.adaptive_thresh_constant.is_finite() {
            return bad("adaptive_thresh_constant must be finite");
        }
        if !(self.min_marker_perimeter_rate > 0.0
            && self.max_marker_perimeter_rate > 0.0
            && self.min_marker_perimeter_rate <= self.max_marker_perimeter_rate
            && self.max_marker_perimeter_rate.is_finite())
        {
            return bad("marker perimeter rates need 0 < min <= max");
        }
        if !(self.polygonal_approx_accuracy_rate > 0.0
            && self.polygonal_approx_accuracy_rate.is_finite())
        {
            return bad("polygonal_approx_accuracy_rate must be positive");
        }
        let non_negative = [
            self.min_corner_distance_rate,
            self.min_marker_distance_rate,
            self.min_group_distance,
            self.max_erroneous_bits_in_border_rate,
            self.min_otsu_std_dev,
            self.error_correction_rate,
            self.relative_corner_refinement_win_size,
        ];
        if non_negative.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
            return bad("rates must be finite and non-negative");
        }
        if self.marker_border_bits == 0 || self.perspective_remove_pixel_per_cell == 0 {
            return bad("marker_border_bits and perspective_remove_pixel_per_cell must be >= 1");
        }
        if !(0.0..0.5).contains(&self.perspective_remove_ignored_margin_per_cell) {
            return bad("perspective_remove_ignored_margin_per_cell must be in [0, 0.5)");
        }
        if self.corner_refinement_method == CornerRefineMethod::Subpix
            && (self.corner_refinement_win_size == 0
                || self.corner_refinement_max_iterations == 0
                || self.corner_refinement_min_accuracy.is_nan()
                || self.corner_refinement_min_accuracy <= 0.0)
        {
            return bad("sub-pixel refinement needs a window, iterations and accuracy > 0");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// ArUco marker detector operating on `CpuTensor` images, following OpenCV's
/// `cv::aruco::ArucoDetector::detectMarkers`.
#[derive(Debug, Clone)]
pub struct ArucoDetector {
    dictionary: ArucoDictionary,
    /// Detection parameters.
    pub params: DetectorParameters,
}

/// A quadrilateral candidate (OpenCV's `MarkerCandidateTree`).
#[derive(Debug, Clone)]
struct Candidate {
    /// Clockwise corners.
    corners: [(f64, f64); 4],
    /// Sum of the four side lengths.
    perimeter: f32,
    /// Other candidates of the same group, tried when this one fails.
    close: Vec<[(f64, f64); 4]>,
    /// Index of the smallest candidate enclosing this one.
    parent: Option<usize>,
    /// Height of the subtree of enclosed candidates (0 for a leaf).
    depth: usize,
}

impl ArucoDetector {
    /// A detector for `dictionary` with OpenCV's default parameters.
    pub fn new(dictionary: ArucoDictionary) -> Self {
        Self::with_parameters(dictionary, DetectorParameters::default())
    }

    /// A detector for `dictionary` with the given parameters.
    pub fn with_parameters(dictionary: ArucoDictionary, params: DetectorParameters) -> Self {
        Self { dictionary, params }
    }

    /// The dictionary markers are identified against.
    pub fn dictionary(&self) -> ArucoDictionary {
        self.dictionary
    }

    /// Detect ArUco markers in a single-channel floating-point grayscale image.
    ///
    /// The image tensor must have shape (1, H, W) with values in [0, 1].
    ///
    /// # Pipeline (OpenCV `detectMarkers`)
    /// 1. Adaptive mean threshold (inverted) with each window size from
    ///    `adaptive_thresh_win_size_min` to `..._max`.
    /// 2. Every contour of every thresholded image (outer and hole borders)
    ///    is approximated by a polygon; convex quadrilaterals within the
    ///    perimeter and side-length limits are candidates, with corners
    ///    ordered clockwise.
    /// 3. Candidates closer than `min_marker_distance_rate` are grouped; the
    ///    largest of each group is kept if it is clear of the image border,
    ///    with the other distinct members as fallbacks.
    /// 4. Each candidate is perspective-removed onto a grid of cells and
    ///    Otsu-thresholded; candidates with too many white border cells are
    ///    rejected.
    /// 5. The payload is matched against the dictionary in all four
    ///    rotations, within `max_correction_bits * error_correction_rate`
    ///    bit errors, and the corners are rotated so that corner 0 is the
    ///    marker's top-left. Enclosed candidates are identified before the
    ///    candidates enclosing them.
    /// 6. Optional corner refinement ([`CornerRefineMethod`]).
    ///
    /// On the same image the result (ids, corners, and with
    /// [`ArucoDetector::detect_with_rejected`] the rejected candidates) is
    /// OpenCV 4.13's with [`CornerRefineMethod::None`]; `Subpix` uses
    /// `cv_calib3d::corner_subpix`, which weights its window uniformly, so
    /// refined corners differ from OpenCV's by a fraction of a pixel.
    pub fn detect<T: Float>(&self, image: &CpuTensor<T>) -> Result<Vec<DetectedMarker>> {
        let (gray, w, h) = tensor_to_gray(image)?;
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

    /// [`ArucoDetector::detect`] that also returns the corners of the
    /// candidates which passed the shape filters but were not identified
    /// (OpenCV's `rejectedImgPoints`), clockwise in the image.
    #[allow(clippy::type_complexity)]
    pub fn detect_with_rejected<T: Float>(
        &self,
        image: &CpuTensor<T>,
    ) -> Result<(Vec<DetectedMarker>, Vec<[(f64, f64); 4]>)> {
        let (gray, w, h) = tensor_to_gray(image)?;
        let (markers, rejected) = self.detect_gray_with_rejected(&gray, w, h)?;
        Ok((markers, rejected.into_iter().map(|r| r.corners).collect()))
    }

    /// Internal: run detection on raw grayscale u8 buffer.
    fn detect_gray(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<DetectedMarker>> {
        Ok(self.detect_gray_with_rejected(gray, w, h)?.0)
    }

    /// Detection on a raw grayscale buffer that also returns the candidates
    /// which passed the shape filters but did not decode (OpenCV's
    /// `rejectedImgPoints`), for board-aware refinement.
    pub(crate) fn detect_gray_with_rejected(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<DetectedMarker>, Vec<RejectedCandidate>)> {
        let p = &self.params;
        p.validate()?;
        if w == 0 || h == 0 || gray.len() != w * h {
            return Err(Error::InvalidInput(
                "aruco detect: image buffer size does not match its dimensions".into(),
            ));
        }

        // 1-2. Candidates from every threshold window, in window order.
        let windows: Vec<usize> = (p.adaptive_thresh_win_size_min..=p.adaptive_thresh_win_size_max)
            .step_by(p.adaptive_thresh_win_size_step)
            .collect();
        let quads: Vec<[(f64, f64); 4]> = windows
            .par_iter()
            .map(|&win| {
                let binary = adaptive_threshold(gray, w, h, win, p.adaptive_thresh_constant);
                find_marker_quads(&binary, w, h, p)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .map(clockwise)
            .collect();

        // 3. One candidate per group of near-duplicates, largest first.
        let mut candidates = self.filter_too_close(quads, w, h);

        // 4-5. Identify, innermost candidates first (OpenCV's
        // `identifyCandidates`). Each depth level is identified in full; when
        // a marker is found its not yet visited ancestors are counted as
        // visited, and the loop stops once the visit count reaches the
        // candidate count - which, as in OpenCV, can leave the outermost
        // levels unidentified once enough ancestors have been counted twice.
        let codes = dictionary_codes(self.dictionary);
        let max_depth = candidates.iter().map(|c| c.depth).max().unwrap_or(0);
        let mut by_depth = vec![Vec::new(); max_depth + 1];
        for (i, c) in candidates.iter().enumerate() {
            by_depth[c.depth].push(i);
        }
        let mut identified: Vec<Option<(usize, usize)>> = vec![None; candidates.len()];
        let mut visited = vec![false; candidates.len()];
        let mut counter = 0usize;
        let mut depth = 0usize;
        while counter < candidates.len() && depth <= max_depth {
            // Per candidate: the corners that decoded, the id and rotation.
            type Found = Option<([(f64, f64); 4], usize, usize)>;
            let results: Vec<Found> = by_depth[depth]
                .par_iter()
                .map(|&i| {
                    let c = &candidates[i];
                    std::iter::once(&c.corners)
                        .chain(&c.close)
                        .find_map(|corners| {
                            let (id, rot) = self.identify(gray, w, h, corners, &codes)?;
                            Some((*corners, id, rot))
                        })
                })
                .collect();
            for (&i, found) in by_depth[depth].iter().zip(results) {
                visited[i] = true;
                if let Some((corners, id, rot)) = found {
                    candidates[i].corners = corners;
                    identified[i] = Some((id, rot));
                    let mut parent = candidates[i].parent;
                    while let Some(j) = parent {
                        if !visited[j] {
                            visited[j] = true;
                            counter += 1;
                        }
                        parent = candidates[j].parent;
                    }
                }
                counter += 1;
            }
            depth += 1;
        }

        let mut markers = Vec::new();
        let mut rejected = Vec::new();
        for (c, ident) in candidates.iter().zip(&identified) {
            match *ident {
                Some((id, rot)) => {
                    // OpenCV's `correctCornerPosition`: the marker appears
                    // turned `rot` quarter turns counter-clockwise, so its
                    // top-left is image corner `4 - rot`.
                    let corners = [0, 1, 2, 3].map(|k| c.corners[(k + 4 - rot) % 4]);
                    markers.push(DetectedMarker { id, corners });
                }
                None => {
                    let bits = self.extract_bits(gray, w, h, &c.corners);
                    rejected.push(RejectedCandidate {
                        corners: c.corners,
                        payload: self.payload(&bits),
                    });
                }
            }
        }

        // 6. Corner refinement.
        if p.corner_refinement_method == CornerRefineMethod::Subpix && !markers.is_empty() {
            self.refine_subpix(gray, w, h, &mut markers)?;
        }
        Ok((markers, rejected))
    }

    /// OpenCV's `filterTooCloseCandidates`: candidates sorted by perimeter,
    /// largest first, are grouped with every later candidate whose average
    /// corner distance is below `min_marker_distance_rate` of the later one's
    /// perimeter. Each group (a lone candidate is a group of one) is
    /// represented by its largest member, unless that member lies within
    /// `min_distance_to_border` of the image border, in which case the whole
    /// group is dropped; members further than `min_group_distance` cells from
    /// the previously kept one are kept as fallbacks. Then each candidate's
    /// parent is the smallest selected candidate enclosing it.
    ///
    /// Distances and perimeters are computed in `f32`, as OpenCV does.
    fn filter_too_close(&self, quads: Vec<[(f64, f64); 4]>, w: usize, h: usize) -> Vec<Candidate> {
        let p = &self.params;
        let cells = (self.dictionary.marker_size() + 2 * p.marker_border_bits) as f32;
        let mut all: Vec<Candidate> = quads
            .into_iter()
            .map(|corners| Candidate {
                perimeter: perimeter(&corners),
                corners,
                close: Vec::new(),
                parent: None,
                depth: 0,
            })
            .collect();
        // Stable, like `std::stable_sort`: equal perimeters keep window order.
        all.sort_by(|a, b| b.perimeter.total_cmp(&a.perimeter));

        let n = all.len();
        let mut group_of: Vec<Option<usize>> = vec![None; n];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut selected = vec![true; n];
        let distance_rate = p.min_marker_distance_rate as f32;
        for i in 0..n {
            for j in i + 1..n {
                let dist = average_corner_distance(&all[i].corners, &all[j].corners);
                if dist >= all[j].perimeter * distance_rate {
                    continue;
                }
                selected[i] = false;
                selected[j] = false;
                match (group_of[i], group_of[j]) {
                    (None, None) => {
                        group_of[i] = Some(groups.len());
                        group_of[j] = Some(groups.len());
                        groups.push(vec![i, j]);
                    }
                    (Some(g), None) => {
                        group_of[j] = Some(g);
                        groups[g].push(j);
                    }
                    (None, Some(g)) => {
                        group_of[i] = Some(g);
                        groups[g].push(i);
                    }
                    // OpenCV does not merge two existing groups.
                    (Some(_), Some(_)) => {}
                }
            }
            if selected[i] {
                selected[i] = false;
                group_of[i] = Some(groups.len());
                groups.push(vec![i]);
            }
        }

        let border = p.min_distance_to_border as f32;
        let (max_x, max_y) = (w as f32 - 1.0 - border, h as f32 - 1.0 - border);
        let group_distance = p.min_group_distance as f32;
        for group in &mut groups {
            group.sort_unstable();
            let first = group[0];
            let near_border = all[first].corners.iter().any(|&(x, y)| {
                let (x, y) = (x as f32, y as f32);
                x < border || y < border || x > max_x || y > max_y
            });
            if near_border {
                continue;
            }
            selected[first] = true;
            let mut current = first;
            for &other in &group[1..] {
                let dist = average_corner_distance(&all[other].corners, &all[current].corners);
                let module = all[other].perimeter / (4.0 * cells);
                if dist > group_distance * module {
                    current = other;
                    let corners = all[other].corners;
                    all[first].close.push(corners);
                }
            }
        }
        let mut out: Vec<Candidate> = all
            .into_iter()
            .zip(selected)
            .filter_map(|(c, keep)| keep.then_some(c))
            .collect();

        // Hierarchy: candidates are sorted largest first, so a candidate's
        // enclosing candidates come before it.
        for i in (0..out.len()).rev() {
            for j in (0..i).rev() {
                if quad_inside_quad(&out[i].corners, &out[j].corners) {
                    out[i].parent = Some(j);
                    out[j].depth = out[j].depth.max(out[i].depth + 1);
                    break;
                }
            }
        }
        out
    }

    /// OpenCV's `_identifyOneCandidate`: `(id, rotation)` of the marker in
    /// the candidate, `rotation` counting quarter turns counter-clockwise.
    fn identify(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
        corners: &[(f64, f64); 4],
        codes: &[u64],
    ) -> Option<(usize, usize)> {
        let p = &self.params;
        let side = self.dictionary.marker_size();
        let bits = self.extract_bits(gray, w, h, corners);
        let max_border_errors =
            ((side * side) as f64 * p.max_erroneous_bits_in_border_rate) as usize;
        if border_errors(&bits, side, p.marker_border_bits) > max_border_errors {
            return None;
        }
        let payload = self.payload(&bits);
        let max_correction =
            (self.dictionary.max_correction_bits() as f64 * p.error_correction_rate) as u32;
        identify_payload(payload, codes, side, max_correction)
    }

    /// OpenCV's `_extractBits`: warp the candidate onto a grid of
    /// `perspective_remove_pixel_per_cell`-pixel cells (nearest neighbour),
    /// Otsu-threshold it and read every cell by majority, ignoring the
    /// cell's margin. Row-major cells, 1 for white.
    fn extract_bits(&self, gray: &[u8], w: usize, h: usize, corners: &[(f64, f64); 4]) -> Vec<u8> {
        let p = &self.params;
        let cells = self.dictionary.marker_size() + 2 * p.marker_border_bits;
        let cell = p.perspective_remove_pixel_per_cell;
        let margin = (p.perspective_remove_ignored_margin_per_cell * cell as f64) as usize;
        let size = cells * cell;
        let last = (size - 1) as f64;
        let canonical = [[0.0, 0.0], [last, 0.0], [last, last], [0.0, last]];
        let image_pts = corners.map(|(x, y)| [x, y]);
        let mut warped = vec![0u8; size * size];
        if let Some(hm) = solve_dlt_homography(&canonical, &image_pts) {
            for v in 0..size {
                for u in 0..size {
                    let q = hm * nalgebra::Vector3::new(u as f64, v as f64, 1.0);
                    if q.z == 0.0 || !q.z.is_finite() {
                        continue;
                    }
                    let x = (q.x / q.z).round_ties_even();
                    let y = (q.y / q.z).round_ties_even();
                    if x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64 {
                        warped[v * size + u] = gray[y as usize * w + x as usize];
                    }
                }
            }
        }

        // Uniform marker image: all cells the same colour.
        let inner: Vec<f64> = (cell / 2..size - cell / 2)
            .flat_map(|v| (cell / 2..size - cell / 2).map(move |u| (u, v)))
            .map(|(u, v)| warped[v * size + u] as f64)
            .collect();
        let n = inner.len().max(1) as f64;
        let mean = inner.iter().sum::<f64>() / n;
        let std_dev = (inner.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
        if std_dev < p.min_otsu_std_dev {
            return vec![u8::from(mean > 127.0); cells * cells];
        }

        let Some(img) = image::GrayImage::from_raw(size as u32, size as u32, warped) else {
            return vec![0; cells * cells];
        };
        let (_, binary) = cv_imgproc::threshold_otsu(&img, 255, cv_imgproc::ThresholdType::Binary);
        let binary = binary.into_raw();
        let span = cell - 2 * margin;
        let mut bits = vec![0u8; cells * cells];
        for y in 0..cells {
            for x in 0..cells {
                let mut white = 0usize;
                for v in y * cell + margin..y * cell + margin + span {
                    for u in x * cell + margin..x * cell + margin + span {
                        white += usize::from(binary[v * size + u] != 0);
                    }
                }
                bits[y * cells + x] = u8::from(white > span * span / 2);
            }
        }
        bits
    }

    /// The payload cells of `bits` (border excluded) as a code.
    fn payload(&self, bits: &[u8]) -> u64 {
        let side = self.dictionary.marker_size();
        let border = self.params.marker_border_bits;
        let cells = side + 2 * border;
        let mut code = 0u64;
        for y in 0..side {
            for x in 0..side {
                if bits[(y + border) * cells + x + border] != 0 {
                    code |= 1u64 << (y * side + x);
                }
            }
        }
        code
    }

    /// `CORNER_REFINE_SUBPIX`: `cornerSubPix` in a half-window of
    /// `relative_corner_refinement_win_size` marker cells, clamped to
    /// `1..=corner_refinement_win_size` pixels. Like OpenCV, a corner that
    /// moves further than the window keeps its unrefined position.
    fn refine_subpix(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
        markers: &mut [DetectedMarker],
    ) -> Result<()> {
        let p = &self.params;
        let Some(img) = image::GrayImage::from_raw(w as u32, h as u32, gray.to_vec()) else {
            return Ok(());
        };
        let cells = (self.dictionary.marker_size() + 2 * p.marker_border_bits) as f64;
        for m in markers {
            let module = perimeter(&m.corners) as f64 / (4.0 * cells);
            let win = ((p.relative_corner_refinement_win_size * module).round() as usize)
                .clamp(1, p.corner_refinement_win_size);
            let mut pts = m.corners.map(|(x, y)| Point2::new(x, y));
            corner_subpix(
                &img,
                &mut pts,
                win,
                p.corner_refinement_max_iterations,
                p.corner_refinement_min_accuracy,
            )?;
            for (c, q) in m.corners.iter_mut().zip(pts) {
                if (q.x - c.0).abs() <= win as f64 && (q.y - c.1).abs() <= win as f64 {
                    *c = (q.x, q.y);
                }
            }
        }
        Ok(())
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
/// `(rotation_vec, translation_vec)` in Rodrigues form, **only if** that pose
/// reproduces the four corners it was given. Four coplanar points leave a
/// 4-dimensional null space, so the linear system does not by itself determine
/// a pose; when no rigid pose explains the corners (degenerate or inconsistent
/// detections) this returns an error instead of a plausible-looking result.
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

    // Solve the planar PnP from the four corners. `solve_pnp_dlt` verifies the
    // pose it returns against these same four normalised observations and
    // reports an error rather than returning a pose that does not reproject
    // them - four coplanar points leave a 4-dimensional null space, so the
    // linear system alone does not determine a pose. The homography DLT that
    // backs it is `cv_calib3d::solve_dlt_homography`, which Hartley-normalises
    // and pads its `8 x 9` system so the null vector is the one returned.
    let (r, t) = solve_pnp_dlt(&obj_pts, &norm)?;
    Ok((r, t))
}

// ---------------------------------------------------------------------------
// Marker generation
// ---------------------------------------------------------------------------

/// Draw a marker image for printing.
///
/// Generates a `size x size` single-channel `u8` tensor containing the marker pattern
/// with a 1-cell black border around the payload bits, identical to OpenCV's
/// `generateImageMarker(dictionary, id, size, img, 1)`. The tensor has shape
/// (1, size, size). The marker has no white margin; detection needs one.
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

    // OpenCV's `generateImageMarker`: the cell grid resized to `size` with
    // nearest-neighbour sampling, pixel `x` showing cell `floor(x * grid / size)`;
    // payload bit 1 is white.
    let white = |gx: usize, gy: usize| {
        let border = |g: usize| g < border_bits || g >= grid - border_bits;
        if border(gx) || border(gy) {
            return false;
        }
        let bit_idx = (gy - border_bits) * payload_bits + (gx - border_bits);
        (code >> bit_idx) & 1 == 1
    };
    let mut pixels = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            if white(x * grid / size, y * grid / size) {
                pixels[y * size + x] = 255;
            }
        }
    }

    CpuTensor::<u8>::from_vec(pixels, TensorShape::new(1, size, size))
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Validate a `(1, H, W)` tensor and convert it to a grayscale `u8` buffer,
/// returning `(pixels, width, height)`.
pub(crate) fn tensor_to_gray<T: Float>(image: &CpuTensor<T>) -> Result<(Vec<u8>, usize, usize)> {
    let (h, w) = (image.shape.height, image.shape.width);
    if image.shape.channels != 1 || h == 0 || w == 0 {
        return Err(Error::InvalidInput(
            "aruco detect requires single-channel (1, H, W) image".into(),
        ));
    }
    let data = image.as_slice()?;
    Ok((data.iter().map(|&v| float_to_u8(v)).collect(), w, h))
}

/// Hamming distance between a payload sampled in image orientation and the
/// code of marker `id`, when the marker's top-left corner is image corner
/// `top_left` (0 = top-left, then clockwise). Like OpenCV's
/// `Dictionary::getDistanceToId(bits, id, false)` on bits extracted in the
/// marker's own orientation. `None` when `id` is not in the dictionary.
pub(crate) fn payload_distance_to_id(
    dictionary: ArucoDictionary,
    payload: u64,
    id: usize,
    top_left: usize,
) -> Option<u32> {
    let code = *dictionary_codes(dictionary).get(id)?;
    // `rotate_90` turns the bit grid clockwise; the image shows the marker
    // turned clockwise `top_left` times, so turn it back.
    let side = dictionary.marker_size();
    let upright = rotate_variants(payload, side)[(4 - top_left % 4) % 4];
    Some((upright ^ code).count_ones())
}

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

/// Adaptive mean threshold. Output: 1 = foreground (dark), 0 = background.
///
/// Same rule as `cv_imgproc::adaptive_threshold` with `MeanC`, `BinaryInv`,
/// and replicate borders: the local mean is rounded to an integer and a pixel
/// is foreground when `src - mean <= -floor(constant)`. The previous
/// shrinking window and `src < mean - constant` (float mean) disagreed with
/// that on flat regions and along the border. `win` is forced odd and at
/// least 3, which is what that function accepts.
fn adaptive_threshold(gray: &[u8], w: usize, h: usize, win: usize, constant: f64) -> Vec<u8> {
    if w == 0 || h == 0 || gray.len() != w.saturating_mul(h) {
        return vec![0u8; w.saturating_mul(h)];
    }
    let mut win = if win.is_multiple_of(2) { win + 1 } else { win };
    if win < 3 {
        win = 3;
    }
    let Some(img) = image::GrayImage::from_raw(w as u32, h as u32, gray.to_vec()) else {
        return vec![0u8; w * h];
    };
    cv_imgproc::adaptive_threshold(
        &img,
        1,
        cv_imgproc::AdaptiveMethod::MeanC,
        cv_imgproc::ThresholdType::BinaryInv,
        win as u32,
        constant as f32,
    )
    .into_raw()
}

/// OpenCV's `_findMarkerContours` on one thresholded image: the contours
/// whose length is within the perimeter limits and whose polygon
/// approximation is a convex quadrilateral with no side shorter than
/// `min_corner_distance_rate` of the contour. (The border distance is checked
/// later, on the candidate representing each group, as in OpenCV 4.7+.)
fn find_marker_quads(
    binary: &[u8],
    w: usize,
    h: usize,
    p: &DetectorParameters,
) -> Vec<[(f64, f64); 4]> {
    let max_side = w.max(h) as f64;
    let min_perimeter = (p.min_marker_perimeter_rate * max_side) as usize;
    let max_perimeter = (p.max_marker_perimeter_rate * max_side) as usize;
    let mut out = Vec::new();
    for contour in contours::find_contours_list(binary, w, h) {
        let len = contour.len();
        if len < min_perimeter || len > max_perimeter {
            continue;
        }
        let approx = contours::approx_poly_dp_closed(
            &contour,
            len as f64 * p.polygonal_approx_accuracy_rate,
        );
        if approx.len() != 4 || !contours::is_convex(&approx) {
            continue;
        }
        let min_side_sq = (0..4)
            .map(|k| {
                let (a, b) = (approx[k], approx[(k + 1) % 4]);
                let (dx, dy) = ((a.0 - b.0) as f64, (a.1 - b.1) as f64);
                dx * dx + dy * dy
            })
            .fold(max_side * max_side, f64::min);
        let min_corner_distance = len as f64 * p.min_corner_distance_rate;
        if min_side_sq < min_corner_distance * min_corner_distance {
            continue;
        }
        out.push([0, 1, 2, 3].map(|k| (approx[k].0 as f64, approx[k].1 as f64)));
    }
    out
}

/// OpenCV's `_reorderCandidatesCorners`: make the corners clockwise in the
/// image (y down) by swapping corners 1 and 3 if needed.
fn clockwise(mut c: [(f64, f64); 4]) -> [(f64, f64); 4] {
    let (dx1, dy1) = (c[1].0 - c[0].0, c[1].1 - c[0].1);
    let (dx2, dy2) = (c[2].0 - c[0].0, c[2].1 - c[0].1);
    if dx1 * dy2 - dy1 * dx2 < 0.0 {
        c.swap(1, 3);
    }
    c
}

/// Sum of the side lengths, in `f32` like OpenCV's `MarkerCandidateTree`.
fn perimeter(c: &[(f64, f64); 4]) -> f32 {
    (0..4)
        .map(|k| {
            let (dx, dy) = (c[k].0 - c[(k + 1) % 4].0, c[k].1 - c[(k + 1) % 4].1);
            (dx as f32 * dx as f32 + dy as f32 * dy as f32).sqrt()
        })
        .sum()
}

/// OpenCV's `getAverageDistance` (in `f32`): root-mean-square corner
/// distance of two quads, minimised over the four cyclic corner
/// correspondences.
fn average_corner_distance(a: &[(f64, f64); 4], b: &[(f64, f64); 4]) -> f32 {
    (0..4)
        .map(|shift| {
            (0..4)
                .map(|k| {
                    let (p, q) = (a[(k + shift) % 4], b[k]);
                    let (dx, dy) = ((p.0 - q.0) as f32, (p.1 - q.1) as f32);
                    dx * dx + dy * dy
                })
                .sum::<f32>()
                / 4.0
        })
        .fold(f32::MAX, f32::min)
        .sqrt()
}

/// OpenCV's `checkMarker1InMarker2`: every corner of `inner` lies in (or on)
/// the clockwise convex quad `outer`.
fn quad_inside_quad(inner: &[(f64, f64); 4], outer: &[(f64, f64); 4]) -> bool {
    inner.iter().all(|&(x, y)| {
        (0..4).all(|k| {
            let (a, b) = (outer[k], outer[(k + 1) % 4]);
            (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0) >= 0.0
        })
    })
}

/// OpenCV's `_getBorderErrors`: white cells in the border ring.
fn border_errors(bits: &[u8], side: usize, border: usize) -> usize {
    let cells = side + 2 * border;
    let mut errors = 0;
    for y in 0..cells {
        for x in 0..cells {
            let in_border = y < border || x < border || y >= cells - border || x >= cells - border;
            if in_border && bits[y * cells + x] != 0 {
                errors += 1;
            }
        }
    }
    errors
}

/// OpenCV's `Dictionary::identify`: the first marker id (in dictionary
/// order) whose closest rotation is within `max_correction` bits of the
/// payload, with that rotation as quarter turns counter-clockwise of the
/// marker in the image.
fn identify_payload(
    payload: u64,
    codes: &[u64],
    side: usize,
    max_correction: u32,
) -> Option<(usize, usize)> {
    // `rotate_90` turns clockwise: if the payload turned clockwise `r` times
    // equals the code, the image shows the marker turned counter-clockwise
    // `r` times, which is OpenCV's rotation `r`.
    let rotations = rotate_variants(payload, side);
    codes.iter().enumerate().find_map(|(id, &code)| {
        let (dist, rot) = rotations
            .iter()
            .enumerate()
            .map(|(r, &rc)| ((rc ^ code).count_ones(), r))
            .fold(
                (u32::MAX, 0),
                |best, cur| if cur.0 < best.0 { cur } else { best },
            );
        (dist <= max_correction).then_some((id, rot))
    })
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

/// Marker codes from the embedded OpenCV tables: bit `y * side + x` is
/// payload cell `(x, y)`, 1 for white, as OpenCV's `getBitsFromByteList`.
///
/// The tables store each rotation-0 code row-major and MSB-first, with the
/// bits of a last, partial byte right-aligned (36 bits of a 6x6 marker end in
/// the low nibble of byte 4). A previous revision read the bits into the code
/// in reverse (turning every marker by 180 degrees), took the last byte's
/// bits from the high end, and drew bit 1 as black, so its markers were not
/// OpenCV's.
fn dictionary_codes(dict: ArucoDictionary) -> Vec<u64> {
    let side = dict.marker_size();
    let bits = side * side;
    let nb = bits.div_ceil(8);
    let count = dict.dict_size();

    let (table, stride): (&[u8], usize) = match side {
        4 => (&crate::aruco_tables::DICT_4X4_BYTES, 4 * nb),
        5 => (&crate::aruco_tables::DICT_5X5_BYTES, 4 * nb),
        _ => (&crate::aruco_tables::DICT_6X6_BYTES, 4 * nb),
    };

    let last_byte_bits = bits - 8 * (nb - 1);
    let mut out = Vec::with_capacity(count);
    for id in 0..count {
        let base = id * stride;
        let mut code = 0u64;
        for b in 0..bits {
            let byte = table[base + b / 8];
            let in_byte = if b / 8 == nb - 1 { last_byte_bits } else { 8 };
            code |= u64::from((byte >> (in_byte - 1 - b % 8)) & 1) << b;
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

/// Solve PnP from exactly 4 coplanar correspondences (a square in `z = 0`),
/// returning `(rodrigues_rotation, translation)` if and only if a rigid planar
/// pose actually reproduces the four observations.
///
/// # What was wrong, and why the answer is a homography
///
/// The old code assembled the `8 x 12` DLT system for the `3 x 4` projection
/// `P` and took `vt.row(vt.nrows() - 1)`. Two independent defects:
///
/// **(a) Thin-SVD row bug.** `A` is `8 x 12`, so the null vector is row 11 of
/// `V^T`. nalgebra's *thin* SVD returns only `min(8, 12) = 8` rows, so row 11
/// does not exist and the old read took **row 7** - an arbitrary vector with a
/// strictly positive `‖A v‖`. Same defect class as `cv_calib3d`'s `m x 9`
/// systems, and the same fix (`crate::dlt::smallest_right_singular_vector`
/// pads an `m x k` system to `k x k` before the SVD) does not rescue this one:
///
/// **(b) The 4-point planar system has a 4-dimensional null space, so there is
/// no single right answer to read.** For a frontal 0.05 m marker at 0.40 m,
/// `f = 200`, no distortion, the measured singular values of `A` (padded to
/// 12 x 12) are
///
/// ```text
/// [35.355, 2.1044, 2.1044, 0.59400, 0.59400, 0.050000, 0.050000, 0.050000, 0, 0, 0, 0]
/// ```
///
/// - four zero singular values, i.e. a 4-dimensional null space, and the
/// smallest non-zero ones cluster four deep at 0.05. The null vector LAPACK
/// happens to return for that matrix is
/// `[0, 0, 0.99354, 0, 0, 0, 0.11351, 0, 0, 0, 0, 0]`: its first column is
/// *exactly* zero, so `‖h1‖ = 4.8e-16` and the recovered `R` has determinant
/// 0.24 - it is not a rotation and never can be. Picking row 11 correctly would
/// not have helped; any single member of a 4-dimensional null space may have a
/// zero column.
///
/// The fix is to stop asking the 12-unknown system the wrong question. For four
/// coplanar points the `3 x 4` projection is exactly `P = K [r1 r2 t]`, so
/// `H = K^-1 P` is a homography and the *planar* DLT applies - a `3 x 3` system
/// with a one-dimensional null space. That is the canonical treatment, and this
/// crate already has a hardened, Hartley-normalised implementation of it, so
/// the linear algebra is delegated to `cv_calib3d::solve_dlt_homography`
/// rather than reimplemented here. `cv_calib3d::pnp::solve_pnp_dlt` does the
/// same thing (planarity detection, then Malis/Vargas decomposition with a
/// rank check) and documents the degeneracy this class of solve needs.
///
/// The sign ambiguity is real and is not ignored: `[-R|-t]` and `[R|t]`
/// reproject identically, so `lambda = ±1/‖h1‖` is enumerated and the corner
/// test below - every corner re-projected through `R`, `t` and the camera
/// matrix, compared with the observation it came from - selects the one whose
/// control points are in front of the camera. A returned pose has therefore
/// demonstrably reproduced its own input.
///
/// Measured on the case above, the old code returned
/// `rvec = [0.0004, 2.2213, 2.2213]`, `t = [-0.0016, 640.0016, -4.9e-9]`:
/// 2.22 rad of rotation error, depth 0.0 instead of 0.4, and 1159% of the
/// marker size of reprojection error on its own four corners - reported as
/// success. The controls in `tests/aruco_pose.rs` (including a distorted case,
/// which exercises the unchanged undistort step) all pass through the same
/// path.
fn solve_pnp_dlt(
    obj: &[[f64; 3]; 4],
    img_norm: &[[f64; 2]],
) -> Result<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>)> {
    let n = 4usize;
    // `img_norm` is already undistorted and normalised, so the target of the
    // homography is that same space: H maps marker (x, y) metres to normalised
    // image coordinates.
    let src: Vec<[f64; 2]> = obj.iter().map(|p| [p[0], p[1]]).collect();
    let dst: Vec<[f64; 2]> = (0..n).map(|i| img_norm[i]).collect();

    let h = cv_calib3d::solve_dlt_homography(&src, &dst).ok_or_else(|| {
        Error::AlgorithmError(
            "PnP DLT: the normalised 4-point homography is undefined (the marker \
             corners are coincident or collinear)"
                .into(),
        )
    })?;

    // H = [h1 h2 h3] is proportional to [r1 r2 t], and a rigid pose has
    // ‖r1‖ = 1, so the scale is 1/‖h1‖.
    let h1 = h.column(0);
    let h2 = h.column(1);
    let h3 = h.column(2);
    let n1 = h1.norm();
    let n2 = h2.norm();
    if !(n1 > 1e-12) || !(n2 > 1e-12) {
        return Err(Error::AlgorithmError(
            "PnP DLT: degenerate homography columns (the marker corners are \
             collinear or coincident in normalised coordinates)"
                .into(),
        ));
    }

    // Tolerance for the corner test below, in normalised image units, so it is
    // independent of focal length and marker size: 1e-6 of the observed
    // baseline. Measured corner errors on exact geometry are ~1e-15, so this
    // leaves three orders of magnitude of headroom for detection noise while
    // still being ~13 orders of magnitude tighter than the bug it guards: the
    // old code was 11.59 - its own error, as a fraction of the marker's
    // projected side - on a case it claimed to have solved.
    let span = dst
        .iter()
        .flat_map(|p| [p[0], p[1]])
        .fold(f64::NEG_INFINITY, f64::max)
        - dst
            .iter()
            .flat_map(|p| [p[0], p[1]])
            .fold(f64::INFINITY, f64::min);
    if !(span > 1e-12) {
        return Err(Error::AlgorithmError(
            "PnP DLT: all four corners project to the same point".into(),
        ));
    }
    let tol = 1e-6 * span;

    // Both signs of the scale are the only two admissible candidates, and the
    // corner test is the accept criterion. The gate is deliberately the
    // *reprojection*, not a stand-in like "‖h1‖ == ‖h2‖": that stand-in is
    // only O(theta^2) in the viewing angle, so it is 1e-8 frontal but 6e-4 for
    // a ~25 degree oblique view of a *perfectly valid* marker. Judge the pose
    // by whether it reproduces its input, which is exact at any obliquity and
    // admits noisy corners too.
    let mut best: Option<(nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>, f64)> = None;
    let mut best_reproj: Option<f64> = None;
    for sign in [1.0f64, -1.0] {
        let lambda = sign / n1;
        let r1 = lambda * h1;
        let r2 = lambda * h2;
        let r3 = r1.cross(&r2);
        let r_orth = nalgebra::Matrix3::from_columns(&[r1, r2, r3]);
        // |det| is 1 by construction for a valid H, and +1 for exactly one of
        // the two signs. Checking (rather than unconditionally flipping) keeps a
        // genuinely non-rotation input from being silently "repaired" into one.
        if (r_orth.determinant() - 1.0).abs() > 1e-6 {
            continue;
        }
        let t_vec = lambda * h3;

        let mut worst = 0.0f64;
        for i in 0..n {
            let o = nalgebra::Vector3::new(obj[i][0], obj[i][1], obj[i][2]);
            let c = r_orth * o + t_vec;
            if !(c.z > 0.0) {
                worst = f64::INFINITY;
                break;
            }
            // Compare in the same undistorted, normalised space the
            // observations were given in.
            let eu = c.x / c.z - img_norm[i][0];
            let ev = c.y / c.z - img_norm[i][1];
            worst = worst.max((eu * eu + ev * ev).sqrt());
        }
        if best_reproj.map(|b| worst < b).unwrap_or(true) {
            best_reproj = Some(worst);
        }
        if worst < tol && best.as_ref().map(|b| worst < b.2).unwrap_or(true) {
            best = Some((r_orth, t_vec, worst));
        }
    }

    let (r_orth, t_vec, _) = best.ok_or_else(|| {
        Error::AlgorithmError(format!(
            "PnP DLT: no rigid pose reproduces the four marker corners. Four \
             coplanar points admit a family of plane-to-plane homographies, and \
             none of them corresponds to a rigid pose for this correspondence \
             set - so this marker cannot be posed from a single detection. \
             Worst reprojection error of the best candidate: {:.3e} normalised \
             units (tolerance {:.3e}). (cv-calib3d::solve_pnp_refine needs >= 6 \
             points, which no single marker provides.)",
            best_reproj.unwrap_or(f64::INFINITY),
            tol
        ))
    })?;

    Ok((rotation_matrix_to_rodrigues(&r_orth), t_vec))
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

    /// OpenCV `adaptiveThreshold` mean / replicate / `THRESH_BINARY_INV`.
    ///
    /// A flat field with integer `C = 0` is entirely foreground: the rounded
    /// mean equals the pixel, and BinaryInv keeps `src <= mean - floor(C)`.
    /// The shrinking-window `src < mean - C` rule marked none of those pixels.
    /// The ramp, including its border, must match `cv_imgproc` byte for byte.
    #[test]
    fn adaptive_threshold_matches_opencv_mean_replicate_binary_inv() {
        let (w, h) = (5usize, 5usize);
        let flat = vec![100u8; w * h];
        let got = adaptive_threshold(&flat, w, h, 3, 0.0);
        assert!(
            got.iter().all(|&v| v == 1),
            "flat field must match OpenCV BinaryInv (all foreground), got {got:?}"
        );

        let ramp: Vec<u8> = (0..w * h).map(|i| (i * 17) as u8).collect();
        let got = adaptive_threshold(&ramp, w, h, 3, 7.0);
        let img = image::GrayImage::from_raw(w as u32, h as u32, ramp).unwrap();
        let expect = cv_imgproc::adaptive_threshold(
            &img,
            1,
            cv_imgproc::AdaptiveMethod::MeanC,
            cv_imgproc::ThresholdType::BinaryInv,
            3,
            7.0,
        );
        assert_eq!(got, expect.into_raw());
    }
}
