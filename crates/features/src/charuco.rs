//! ChArUco board detection — fiducial marker board with chessboard corners
//!
//! A ChArUco board combines ArUco markers with a chessboard pattern for
//! precise camera calibration and sub-pixel corner refinement.
//!
//! Detection follows OpenCV's `cv::aruco::CharucoDetector::detectBoard`
//! (objdetect, OpenCV 4.x):
//!
//! 1. detect the ArUco markers, keeping the rejected candidates;
//! 2. with [`CharucoParameters::try_refine_markers`], recover board markers
//!    from the rejected candidates (`refineDetectedMarkers`);
//! 3. interpolate the chessboard corners, from a camera pose when
//!    [`CharucoParameters::camera_matrix`] is set (`interpolateCornersCharucoApproxCalib`)
//!    and from one local homography per marker otherwise
//!    (`interpolateCornersCharucoLocalHom`);
//! 4. drop corners closer than 2 px to the image border and refine the rest
//!    with `cornerSubPix`, each in the largest window that stays clear of the
//!    neighbouring marker corners (`getMaximumSubPixWindowSizes`);
//! 5. drop corners seen by fewer than [`CharucoParameters::min_markers`]
//!    adjacent markers (`filterCornersWithoutMinMarkers`);
//! 6. with [`CharucoParameters::check_markers`], reject the whole result when
//!    the markers contradict the board layout (`checkBoard`).
//!
//! `check_markers` additionally removes individual markers that disagree with
//! the board homography fitted to the others before any of that runs; OpenCV
//! has no such step (see [`CharucoParameters::check_markers`]).

use crate::aruco::{
    ArucoDetector, ArucoDictionary, DetectedMarker, RejectedCandidate, payload_distance_to_id,
    tensor_to_gray,
};
use cv_calib3d::{
    corner_subpix, project_points_with_distortion, solve_dlt_homography, solve_pnp_refine,
    undistort_points,
};
use cv_core::{CameraIntrinsics, CpuTensor, Distortion, Error, Float, Pose, Result};
use image::GrayImage;
use nalgebra::{Matrix3, Point2, Point3, Vector3};

/// `DetectorParameters::cornerRefinementWinSize`: the `cornerSubPix`
/// half-window OpenCV falls back to when no adjacent marker bounds it.
const SUBPIX_DEFAULT_HALF_WIN: usize = 5;
/// `DetectorParameters::cornerRefinementMaxIterations`.
const SUBPIX_MAX_ITERS: usize = 30;
/// `DetectorParameters::cornerRefinementMinAccuracy`, in pixels.
const SUBPIX_EPS: f64 = 0.1;
/// `selectAndRefineChessboardCorners` keeps corners at least this far inside
/// the image.
const MIN_DIST_TO_BORDER: f64 = 2.0;
/// `RefineParameters::minRepDistance`: how close (largest corner distance, in
/// pixels) a rejected candidate must lie to a projected missing marker.
const REFINE_MIN_REP_DISTANCE: f64 = 10.0;
/// `RefineParameters::errorCorrectionRate`.
const REFINE_ERROR_CORRECTION_RATE: f64 = 3.0;
/// Smallest marker set the leave-one-out consistency check runs on. With
/// three markers the homography fitted to the other two is exactly
/// determined, so it absorbs an outlier and cannot tell which marker is wrong.
const CONSISTENCY_MIN_MARKERS: usize = 4;

/// ChArUco board combining ArUco markers with chessboard pattern
#[derive(Clone)]
pub struct CharucoBoard {
    /// Number of chessboard squares along X.
    pub squares_x: usize,
    /// Number of chessboard squares along Y.
    pub squares_y: usize,
    /// Side of one chessboard square, in world units.
    pub square_length: f64,
    /// Side of one embedded marker, in world units.
    pub marker_length: f64,
    /// Dictionary the embedded markers are drawn from.
    pub dictionary: ArucoDictionary,
}

impl CharucoBoard {
    /// A board of `squares_x` x `squares_y` squares. Markers sit on the cells
    /// where `row + col` is odd, with ids assigned row-major (OpenCV layout).
    pub fn new(
        squares_x: usize,
        squares_y: usize,
        square_length: f64,
        marker_length: f64,
        dictionary: ArucoDictionary,
    ) -> Self {
        Self {
            squares_x,
            squares_y,
            square_length,
            marker_length,
            dictionary,
        }
    }

    /// Chessboard corner world coordinates (z=0 plane).
    ///
    /// Corners lie at the interior grid INTERSECTIONS: (col*L, row*L) with
    /// col in 1..squares_x, row in 1..squares_y, ordered row-major so
    /// that index i corresponds to corner id i from `interpolate_corners`
    /// (which builds `(squares_x-1) * (squares_y-1)` ids). A previous revision
    /// iterated `1..squares_x-1` / `1..squares_y-1`, producing only
    /// `(squares_x-2) * (squares_y-2)` points so the id contract did not hold.
    /// Even earlier it used square CENTERS ((col+0.5)*L), offsetting every
    /// calibration observation by half a square.
    pub fn chessboard_corners(&self) -> Vec<Point3<f64>> {
        let mut pts = Vec::new();
        for row in 1..self.squares_y {
            for col in 1..self.squares_x {
                pts.push(Point3::new(
                    col as f64 * self.square_length,
                    row as f64 * self.square_length,
                    0.0,
                ));
            }
        }
        pts
    }

    /// World coordinates (z=0 plane) of the four corners of every marker,
    /// indexed by marker id, in the order top-left, top-right, bottom-right,
    /// bottom-left (OpenCV's `Board::getObjPoints`).
    ///
    /// Marker `id` sits centred in the `id`-th cell (row-major) with
    /// `row + col` odd, inset by `(square_length - marker_length) / 2`.
    pub fn marker_object_points(&self) -> Vec<[Point3<f64>; 4]> {
        let l = self.square_length;
        let m = self.marker_length;
        let inset = (l - m) / 2.0;
        marker_cells(self.squares_x, self.squares_y)
            .into_iter()
            .map(|(col, row)| {
                let x0 = col as f64 * l + inset;
                let y0 = row as f64 * l + inset;
                [
                    Point3::new(x0, y0, 0.0),
                    Point3::new(x0 + m, y0, 0.0),
                    Point3::new(x0 + m, y0 + m, 0.0),
                    Point3::new(x0, y0 + m, 0.0),
                ]
            })
            .collect()
    }
}

/// Cell `(col, row)` of each marker id: the cells with `row + col` odd, in
/// row-major order.
fn marker_cells(squares_x: usize, squares_y: usize) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for row in 0..squares_y {
        for col in 0..squares_x {
            if (row + col) % 2 == 1 {
                cells.push((col, row));
            }
        }
    }
    cells
}

/// For each chessboard corner id, the markers that touch it and the index of
/// the marker corner lying on it (OpenCV's `getNearestMarkerIdx` /
/// `getNearestMarkerCorners`). An interior corner meets four squares, two of
/// them white, so each list has exactly two entries.
fn nearest_markers(squares_x: usize, squares_y: usize) -> Vec<Vec<(usize, usize)>> {
    if squares_x < 2 || squares_y < 2 {
        return Vec::new();
    }
    let mut id_of_cell = vec![None; squares_x * squares_y];
    for (id, &(col, row)) in marker_cells(squares_x, squares_y).iter().enumerate() {
        id_of_cell[row * squares_x + col] = Some(id);
    }
    let mut out = Vec::with_capacity((squares_x - 1) * (squares_y - 1));
    for gy in 1..squares_y {
        for gx in 1..squares_x {
            // Grid point (gx, gy) is the BR corner of cell (gx-1, gy-1), the BL
            // corner of (gx, gy-1), the TR corner of (gx-1, gy) and the TL
            // corner of (gx, gy). Marker corners are TL, TR, BR, BL = 0..4.
            let mut adjacent = Vec::with_capacity(2);
            for (col, row, corner) in [
                (gx - 1, gy - 1, 2),
                (gx, gy - 1, 3),
                (gx - 1, gy, 1),
                (gx, gy, 0),
            ] {
                if let Some(id) = id_of_cell[row * squares_x + col] {
                    adjacent.push((id, corner));
                }
            }
            out.push(adjacent);
        }
    }
    out
}

/// ChArUco detection parameters (OpenCV's `cv::aruco::CharucoParameters`).
#[derive(Clone)]
pub struct CharucoParameters {
    /// Camera intrinsics `[fx 0 cx; 0 fy cy; 0 0 1]`.
    ///
    /// When set, corners are interpolated by approximate calibration, as
    /// OpenCV's `interpolateCornersCharucoApproxCalib`: the board pose is
    /// solved from every detected marker corner (homography decomposition
    /// refined by Levenberg-Marquardt on the distorted reprojection error,
    /// OpenCV's planar `solvePnP`), and all chessboard corners are projected
    /// with that pose and [`dist_coeffs`](Self::dist_coeffs).
    ///
    /// When `None`, each corner is projected through the homography of each
    /// adjacent detected marker (marker plane to image, from its own four
    /// corners) and the two projections are averaged, as OpenCV's
    /// `interpolateCornersCharucoLocalHom`. This is exact under any
    /// perspective when the marker corners are; it does not model lens
    /// distortion.
    ///
    /// A skewed matrix (non-zero `[0][1]`), a bottom row other than
    /// `[0 0 1]` or a non-positive focal length is rejected by `detect` with
    /// `Error::InvalidInput`, since the projection model cannot represent it.
    pub camera_matrix: Option<Matrix3<f64>>,
    /// Distortion coefficients in OpenCV order `[k1, k2, p1, p2, k3, ...]`;
    /// only used together with `camera_matrix`.
    ///
    /// Accepts 4 or 5 coefficients, or more if every coefficient past `k3`
    /// is zero; the rational, thin-prism and tilt terms OpenCV accepts in
    /// 8/12/14-element vectors are not modelled and are reported as
    /// `Error::InvalidInput` when non-zero.
    pub dist_coeffs: Option<Vec<f64>>,
    /// How many of a corner's two adjacent markers must be detected for the
    /// corner to be reported: 0, 1 or 2 (OpenCV's `minMarkers`, default 2).
    /// Only the pose mode can produce a corner with no adjacent marker
    /// detected, so 0 differs from 1 only when `camera_matrix` is set. Values
    /// above 2 are rejected by `detect` with `Error::InvalidInput`.
    pub min_markers: usize,
    /// Try to recover board markers the ArUco detector rejected, as OpenCV's
    /// `refineDetectedMarkers` with default `RefineParameters`.
    ///
    /// The missing markers are projected through the homography fitted to
    /// the detected ones; a rejected candidate is accepted for a missing
    /// marker when, in its best cyclic corner order, every corner lies within
    /// 10 px of the projection (`minRepDistance`) and its payload, read in
    /// that orientation, is fewer than `maxCorrectionBits * 3` bits
    /// (`errorCorrectionRate`) from the expected code. The rescued marker's
    /// corners are reordered to the board's orientation. Needs at least one
    /// detected marker. Only `detect` has rejected candidates to work with.
    pub try_refine_markers: bool,
    /// Check the markers against the board layout.
    ///
    /// Two checks run when this is set:
    ///
    /// - **OpenCV `checkBoard`**: after interpolation, each corner must be
    ///   closer to the corners of its own two markers than to the centre of
    ///   any other detected marker, and closer to its marker corner than to
    ///   the midpoints of that corner's two edges. If any corner fails, the
    ///   board is taken to be built incorrectly and *no* corners are
    ///   returned, as OpenCV does.
    /// - **Per-marker consistency (not in OpenCV)**: before interpolation,
    ///   the board homography is fitted to all markers but one and that
    ///   marker's corners are predicted from it; the marker with the largest
    ///   mean prediction error is dropped while that error exceeds half the
    ///   smallest distance between two marker cells (`√2/2 · square_length`,
    ///   scaled to pixels by the marker's own size), repeating on the rest.
    ///   A misidentified marker therefore loses only its own corners instead
    ///   of corrupting its neighbours. Every copy of a repeated id is checked
    ///   on its own, so the first *consistent* copy is the one kept (without
    ///   the check, the first copy is kept). It runs on four or more markers;
    ///   with `camera_matrix` set the corners are undistorted first.
    pub check_markers: bool,
}

impl Default for CharucoParameters {
    /// OpenCV's defaults: no camera, `min_markers = 2`, no marker
    /// refinement, `check_markers = true`.
    fn default() -> Self {
        Self {
            camera_matrix: None,
            dist_coeffs: None,
            min_markers: 2,
            try_refine_markers: false,
            check_markers: true,
        }
    }
}

/// A detected ChArUco chessboard corner
#[derive(Debug, Clone)]
pub struct CharucoCorners {
    /// Image position of each detected interior corner.
    pub corners: Vec<[f32; 2]>,
    /// Id of each corner, `row * (squares_x - 1) + col`, in ascending order.
    pub ids: Vec<i32>,
}

impl CharucoCorners {
    fn empty() -> Self {
        Self {
            corners: Vec::new(),
            ids: Vec::new(),
        }
    }
}

/// Camera model parsed from [`CharucoParameters`].
struct Camera {
    intrinsics: CameraIntrinsics,
    distortion: Distortion,
}

/// ChArUco detector
pub struct CharucoDetector {
    board: CharucoBoard,
    params: CharucoParameters,
    aruco_detector: ArucoDetector,
    /// Adjacent markers of every chessboard corner, see [`nearest_markers`].
    nearest: Vec<Vec<(usize, usize)>>,
    /// Marker corner world coordinates, indexed by marker id.
    marker_obj: Vec<[Point3<f64>; 4]>,
}

impl CharucoDetector {
    /// A detector for `board` that finds markers with `detector`.
    pub fn new(board: CharucoBoard, params: CharucoParameters, detector: ArucoDetector) -> Self {
        let nearest = nearest_markers(board.squares_x, board.squares_y);
        let marker_obj = board.marker_object_points();
        Self {
            board,
            params,
            aruco_detector: detector,
            nearest,
            marker_obj,
        }
    }

    /// Detect ChArUco board corners from image.
    ///
    /// Detects the ArUco markers, then runs the pipeline described in the
    /// module documentation: marker refinement, corner interpolation,
    /// border filtering, `cornerSubPix` refinement, the `min_markers`
    /// filter and the board check.
    ///
    /// Marker corners come from [`ArucoDetector`], which reports each
    /// marker's axis-aligned bounding box in image order (top-left first),
    /// so the board must appear upright, and under perspective the marker
    /// corners — and the interpolated corners — are approximate until the
    /// sub-pixel refinement.
    pub fn detect<T: Float>(&self, image: &CpuTensor<T>) -> Result<CharucoCorners> {
        let (gray, w, h) = tensor_to_gray(image)?;
        let (markers, rejected) = self.aruco_detector.detect_gray_with_rejected(&gray, w, h)?;
        let gray = GrayImage::from_raw(w as u32, h as u32, gray)
            .ok_or_else(|| Error::InvalidInput("charuco detect: image buffer size".into()))?;
        self.detect_board(&markers, &rejected, Some(&gray))
    }

    /// Corners interpolated from already detected markers, with no image: no
    /// marker refinement, border filter or sub-pixel refinement.
    #[cfg(test)]
    fn interpolate_corners(&self, markers: &[DetectedMarker]) -> Result<CharucoCorners> {
        self.detect_board(markers, &[], None)
    }

    fn detect_board(
        &self,
        markers: &[DetectedMarker],
        rejected: &[RejectedCandidate],
        image: Option<&GrayImage>,
    ) -> Result<CharucoCorners> {
        if self.params.min_markers > 2 {
            return Err(Error::InvalidInput(format!(
                "charuco min_markers must be 0, 1 or 2, got {}",
                self.params.min_markers
            )));
        }
        let camera = self.camera()?;
        if self.nearest.is_empty() {
            return Ok(CharucoCorners::empty());
        }

        // Board markers only. The consistency check sees every copy of a
        // repeated id, so a copy read in the wrong place is removed before
        // the first remaining copy of each id is kept.
        let mut markers: Vec<DetectedMarker> = markers
            .iter()
            .filter(|m| m.id < self.marker_obj.len())
            .cloned()
            .collect();
        if self.params.check_markers {
            markers = self.reject_inconsistent_markers(markers, camera.as_ref());
        }
        let mut seen = vec![false; self.marker_obj.len()];
        markers.retain(|m| !std::mem::replace(&mut seen[m.id], true));
        if self.params.try_refine_markers && image.is_some() {
            self.refine_markers(&mut markers, rejected);
        }

        let mut index_of_id = vec![None; self.marker_obj.len()];
        for (i, m) in markers.iter().enumerate() {
            index_of_id[m.id] = Some(i);
        }

        let mut positions = match &camera {
            Some(cam) => self.interpolate_from_pose(&markers, cam),
            None => self.interpolate_local_homography(&markers, &index_of_id),
        };

        for (id, slot) in positions.iter_mut().enumerate() {
            let Some(p) = *slot else { continue };
            let seen_by = self.nearest[id]
                .iter()
                .filter(|(mid, _)| index_of_id[*mid].is_some())
                .count();
            let inside = image.is_none_or(|img| {
                let (w, h) = (img.width() as f64, img.height() as f64);
                p[0] >= MIN_DIST_TO_BORDER
                    && p[1] >= MIN_DIST_TO_BORDER
                    && p[0] < w - MIN_DIST_TO_BORDER
                    && p[1] < h - MIN_DIST_TO_BORDER
            });
            if seen_by < self.params.min_markers
                || !inside
                || !p[0].is_finite()
                || !p[1].is_finite()
            {
                *slot = None;
            }
        }

        if let Some(img) = image {
            self.refine_subpix(img, &mut positions, &markers, &index_of_id)?;
        }

        let mut out = CharucoCorners::empty();
        for (id, p) in positions.iter().enumerate() {
            if let Some(p) = p {
                out.corners.push([p[0] as f32, p[1] as f32]);
                out.ids.push(id as i32);
            }
        }

        if self.params.check_markers && !self.check_board(&markers, &positions) {
            return Ok(CharucoCorners::empty());
        }
        Ok(out)
    }

    /// Parse `camera_matrix` / `dist_coeffs`.
    fn camera(&self) -> Result<Option<Camera>> {
        let Some(k) = self.params.camera_matrix else {
            return Ok(None);
        };
        let valid = k.iter().all(|v| v.is_finite())
            && k[(0, 0)] > 0.0
            && k[(1, 1)] > 0.0
            && k[(0, 1)] == 0.0
            && k[(1, 0)] == 0.0
            && k[(2, 0)] == 0.0
            && k[(2, 1)] == 0.0
            && k[(2, 2)] == 1.0;
        if !valid {
            return Err(Error::InvalidInput(format!(
                "charuco camera_matrix must be [fx 0 cx; 0 fy cy; 0 0 1] with fx, fy > 0, got {k}"
            )));
        }
        let distortion = match self.params.dist_coeffs.as_deref() {
            None | Some([]) => Distortion::none(),
            Some(c) if c.iter().any(|v| !v.is_finite()) => {
                return Err(Error::InvalidInput(
                    "charuco dist_coeffs must be finite".into(),
                ));
            }
            Some(&[k1, k2, p1, p2]) => Distortion::new(k1, k2, p1, p2, 0.0),
            Some(c) if c.len() >= 5 && c[5..].iter().all(|&v| v == 0.0) => {
                Distortion::new(c[0], c[1], c[2], c[3], c[4])
            }
            Some(c) => {
                return Err(Error::InvalidInput(format!(
                    "charuco dist_coeffs: expected [k1, k2, p1, p2(, k3)] (terms past k3 must be \
                     zero), got {} coefficients {c:?}",
                    c.len()
                )));
            }
        };
        Ok(Some(Camera {
            intrinsics: CameraIntrinsics::new(k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)], 0, 0),
            distortion,
        }))
    }

    /// `interpolateCornersCharucoLocalHom`: project each corner through the
    /// homography of every adjacent detected marker and average.
    fn interpolate_local_homography(
        &self,
        markers: &[DetectedMarker],
        index_of_id: &[Option<usize>],
    ) -> Vec<Option<[f64; 2]>> {
        let homographies: Vec<Option<Matrix3<f64>>> = markers
            .iter()
            .map(|m| {
                let src = self.marker_obj[m.id].map(|p| [p.x, p.y]);
                let dst = m.corners.map(|(x, y)| [x, y]);
                marker_homography(&src, &dst)
            })
            .collect();
        self.board
            .chessboard_corners()
            .iter()
            .zip(&self.nearest)
            .map(|(obj, adjacent)| {
                let projected: Vec<[f64; 2]> = adjacent
                    .iter()
                    .filter_map(|(mid, _)| homographies[index_of_id[*mid]?].as_ref())
                    .filter_map(|h| apply_homography(h, [obj.x, obj.y]))
                    .collect();
                match projected.as_slice() {
                    [] => None,
                    [p] => Some(*p),
                    [a, b, ..] => Some([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]),
                }
            })
            .collect()
    }

    /// `interpolateCornersCharucoApproxCalib`: solve the board pose from the
    /// marker corners and project every chessboard corner.
    fn interpolate_from_pose(
        &self,
        markers: &[DetectedMarker],
        cam: &Camera,
    ) -> Vec<Option<[f64; 2]>> {
        let n = self.nearest.len();
        let mut obj = Vec::with_capacity(markers.len() * 4);
        let mut img = Vec::with_capacity(markers.len() * 4);
        for m in markers {
            for (p, &(x, y)) in self.marker_obj[m.id].iter().zip(&m.corners) {
                obj.push(*p);
                img.push(Point2::new(x, y));
            }
        }
        if obj.len() < 4 {
            return vec![None; n];
        }
        let Some(pose) = estimate_planar_pose(&obj, &img, cam) else {
            return vec![None; n];
        };
        match project_points_with_distortion(
            &self.board.chessboard_corners(),
            &cam.intrinsics,
            &pose,
            &cam.distortion,
        ) {
            Ok(pts) => pts.into_iter().map(|p| Some([p.x, p.y])).collect(),
            Err(_) => vec![None; n],
        }
    }

    /// `selectAndRefineChessboardCorners` with `getMaximumSubPixWindowSizes`:
    /// `cornerSubPix` each corner in a half-window of `int(d - 2)` pixels,
    /// clamped to `1..=10`; corners without a detected adjacent marker use
    /// the default half-window of 5.
    ///
    /// OpenCV takes `d` as the Euclidean distance to the nearest corner of an
    /// adjacent detected marker. That marker corner lies diagonally from the
    /// chessboard corner, so a square window of half-size `d - 2` still
    /// reaches into the marker; OpenCV's `cornerSubPix` damps the window's
    /// rim with a Gaussian weight, but `cv_calib3d::corner_subpix` weights
    /// every pixel equally and the marker's edges pull the corner away
    /// (measured on the tilted test render: half-window 10 gives mean
    /// 0.86 px / max 2.16 px error, half-window 8 gives 0.13 / 0.26). So `d`
    /// here is the smaller of the horizontal and vertical offsets to that
    /// marker corner, which keeps the window clear of the marker.
    ///
    /// OpenCV shifts the point by -0.5 px before `cornerSubPix` and back
    /// after, because its marker corners use the pixel-edge convention; this
    /// workspace's marker corners and `corner_subpix` both put pixel centres
    /// on integers, so no shift is applied.
    fn refine_subpix(
        &self,
        image: &GrayImage,
        positions: &mut [Option<[f64; 2]>],
        markers: &[DetectedMarker],
        index_of_id: &[Option<usize>],
    ) -> Result<()> {
        for (id, slot) in positions.iter_mut().enumerate() {
            let Some(p) = *slot else { continue };
            let min_dist = self.nearest[id]
                .iter()
                .filter_map(|&(mid, corner)| {
                    let (x, y) = markers[index_of_id[mid]?].corners[corner];
                    Some((x - p[0]).abs().min((y - p[1]).abs()))
                })
                .reduce(f64::min);
            let half_win = match min_dist {
                Some(d) => ((d - 2.0) as i64).clamp(1, 10) as usize,
                None => SUBPIX_DEFAULT_HALF_WIN,
            };
            let mut pt = [Point2::new(p[0], p[1])];
            corner_subpix(image, &mut pt, half_win, SUBPIX_MAX_ITERS, SUBPIX_EPS)?;
            *slot = Some([pt[0].x, pt[0].y]);
        }
        Ok(())
    }

    /// OpenCV's `CharucoDetectorImpl::checkBoard`: `false` when an
    /// interpolated corner lies closer to another marker's centre than to the
    /// corners of the two markers it was built from, or closer to the
    /// midpoint of one of its marker corner's edges than to that corner.
    fn check_board(&self, markers: &[DetectedMarker], positions: &[Option<[f64; 2]>]) -> bool {
        let dist = |a: (f64, f64), b: [f64; 2]| (a.0 - b[0]).hypot(a.1 - b[1]);
        for (id, p) in positions.iter().enumerate() {
            let Some(p) = *p else { continue };
            let adjacent = &self.nearest[id];
            let mut to_own = 0.0f64;
            let mut to_other = f64::INFINITY;
            for m in markers {
                if let Some(&(_, k)) = adjacent.iter().find(|(mid, _)| *mid == m.id) {
                    let c = m.corners[k];
                    let d = dist(c, p);
                    to_own = to_own.max(d);
                    let mid = |o: (f64, f64)| ((o.0 + c.0) / 2.0, (o.1 + c.1) / 2.0);
                    let to_edge = dist(mid(m.corners[(k + 1) % 4]), p)
                        .min(dist(mid(m.corners[(k + 3) % 4]), p));
                    if to_edge < d {
                        return false;
                    }
                } else {
                    let cx = m.corners.iter().map(|c| c.0).sum::<f64>() / 4.0;
                    let cy = m.corners.iter().map(|c| c.1).sum::<f64>() / 4.0;
                    to_other = to_other.min(dist((cx, cy), p));
                }
            }
            if to_own > 0.0 && to_other.is_finite() && to_own > to_other {
                return false;
            }
        }
        true
    }

    /// Drop markers that disagree with the board homography fitted to the
    /// others (see [`CharucoParameters::check_markers`]).
    fn reject_inconsistent_markers(
        &self,
        markers: Vec<DetectedMarker>,
        camera: Option<&Camera>,
    ) -> Vec<DetectedMarker> {
        if markers.len() < CONSISTENCY_MIN_MARKERS {
            return markers;
        }
        // Image corners, undistorted (still in pixels) when a camera is known
        // so the board-to-image map is a homography.
        let image_pts: Vec<[[f64; 2]; 4]> = markers
            .iter()
            .map(|m| {
                let raw = m.corners.map(|(x, y)| [x, y]);
                let Some(cam) = camera else { return raw };
                let pts = m.corners.map(|(x, y)| Point2::new(x, y));
                match undistort_points(&pts, &cam.intrinsics, &cam.distortion) {
                    Ok(u) => [0, 1, 2, 3].map(|k| [u[k].x, u[k].y]),
                    Err(_) => raw,
                }
            })
            .collect();
        let obj_pts: Vec<[[f64; 2]; 4]> = markers
            .iter()
            .map(|m| self.marker_obj[m.id].map(|p| [p.x, p.y]))
            .collect();
        // Half the smallest centre distance between two marker cells, as a
        // fraction of the marker side.
        let tolerance =
            std::f64::consts::FRAC_1_SQRT_2 * self.board.square_length / self.board.marker_length;

        let mut alive: Vec<usize> = (0..markers.len()).collect();
        while alive.len() >= CONSISTENCY_MIN_MARKERS {
            let mut worst: Option<(f64, usize)> = None;
            for (slot, &i) in alive.iter().enumerate() {
                let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = alive
                    .iter()
                    .filter(|&&j| j != i)
                    .flat_map(|&j| obj_pts[j].into_iter().zip(image_pts[j]))
                    .unzip();
                let Some(h) = solve_dlt_homography(&src, &dst) else {
                    continue;
                };
                let side = (0..4)
                    .map(|k| {
                        let (a, b) = (image_pts[i][k], image_pts[i][(k + 1) % 4]);
                        (a[0] - b[0]).hypot(a[1] - b[1])
                    })
                    .sum::<f64>()
                    / 4.0;
                if !side.is_finite() || side <= 0.0 {
                    continue;
                }
                let err = (0..4)
                    .map(|k| match apply_homography(&h, obj_pts[i][k]) {
                        Some(p) => (p[0] - image_pts[i][k][0]).hypot(p[1] - image_pts[i][k][1]),
                        None => f64::INFINITY,
                    })
                    .sum::<f64>()
                    / 4.0
                    / side;
                if worst.is_none_or(|(e, _)| err > e) {
                    worst = Some((err, slot));
                }
            }
            match worst {
                Some((err, slot)) if err > tolerance => {
                    alive.remove(slot);
                }
                _ => break,
            }
        }
        let keep: Vec<bool> = {
            let mut k = vec![false; markers.len()];
            for &i in &alive {
                k[i] = true;
            }
            k
        };
        markers
            .into_iter()
            .zip(keep)
            .filter_map(|(m, k)| k.then_some(m))
            .collect()
    }

    /// OpenCV's `refineDetectedMarkers` (default `RefineParameters`, no
    /// camera): rescue missing board markers from the rejected candidates.
    fn refine_markers(&self, markers: &mut Vec<DetectedMarker>, rejected: &[RejectedCandidate]) {
        if markers.is_empty() || rejected.is_empty() {
            return;
        }
        let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = markers
            .iter()
            .flat_map(|m| {
                self.marker_obj[m.id]
                    .map(|p| [p.x, p.y])
                    .into_iter()
                    .zip(m.corners.map(|(x, y)| [x, y]))
            })
            .unzip();
        let Some(h) = solve_dlt_homography(&src, &dst) else {
            return;
        };
        let dictionary = self.board.dictionary;
        // `maxCorrectionRecalculed`; OpenCV accepts `codeDistance < it`.
        let max_code_distance =
            (dictionary.max_correction_bits() as f64 * REFINE_ERROR_CORRECTION_RATE) as u32;

        let mut detected = vec![false; self.marker_obj.len()];
        for m in markers.iter() {
            detected[m.id] = true;
        }
        let mut used = vec![false; rejected.len()];
        let marker_count = self.marker_obj.len().min(dictionary.dict_size());
        for (id, &is_detected) in detected.iter().enumerate().take(marker_count) {
            if is_detected {
                continue;
            }
            let Some(projected) = self.marker_obj[id]
                .iter()
                .map(|p| apply_homography(&h, [p.x, p.y]))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let mut best: Option<(f64, usize, usize)> = None;
            for (j, cand) in rejected.iter().enumerate() {
                if used[j] {
                    continue;
                }
                // Best cyclic order: candidate corner `rot` is the marker's TL.
                let (dist, rot) = (0..4)
                    .map(|rot| {
                        let d = (0..4)
                            .map(|k| {
                                let (x, y) = cand.corners[(rot + k) % 4];
                                (x - projected[k][0]).hypot(y - projected[k][1])
                            })
                            .fold(0.0f64, f64::max);
                        (d, rot)
                    })
                    .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
                let limit = best.map_or(REFINE_MIN_REP_DISTANCE, |b| b.0);
                if dist >= limit {
                    continue;
                }
                let Some(code_distance) = payload_distance_to_id(dictionary, cand.payload, id, rot)
                else {
                    continue;
                };
                if code_distance < max_code_distance {
                    best = Some((dist, j, rot));
                }
            }
            if let Some((_, j, rot)) = best {
                used[j] = true;
                let c = rejected[j].corners;
                markers.push(DetectedMarker {
                    id,
                    corners: [0, 1, 2, 3].map(|k| c[(rot + k) % 4]),
                });
            }
        }
    }
}

/// Homography mapping a marker's four board-plane corners to its image
/// corners (OpenCV `getPerspectiveTransform`), normalised to `h[2][2] = 1`;
/// `None` when singular, as OpenCV's `|det| > 1e-6` test.
fn marker_homography(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Matrix3<f64>> {
    let h = solve_dlt_homography(src, dst)?;
    let s = h[(2, 2)];
    if s.is_nan() || s.abs() <= f64::EPSILON {
        return None;
    }
    let h = h / s;
    (h.determinant().abs() > 1e-6).then_some(h)
}

fn apply_homography(h: &Matrix3<f64>, p: [f64; 2]) -> Option<[f64; 2]> {
    let v = h * Vector3::new(p[0], p[1], 1.0);
    if v.z.is_nan() || v.z.abs() <= f64::EPSILON {
        return None;
    }
    let out = [v.x / v.z, v.y / v.z];
    (out[0].is_finite() && out[1].is_finite()).then_some(out)
}

/// Pose of a planar target (`z = 0`) from >= 4 correspondences, as OpenCV's
/// `solvePnP(SOLVEPNP_ITERATIVE)` does for planar input: undistort the image
/// points, decompose the homography from the board plane to normalised
/// coordinates into `[r1 r2 t]`, then refine by Levenberg-Marquardt on the
/// distorted reprojection error.
fn estimate_planar_pose(obj: &[Point3<f64>], img: &[Point2<f64>], cam: &Camera) -> Option<Pose> {
    let k = &cam.intrinsics;
    let undistorted = undistort_points(img, k, &cam.distortion).ok()?;
    let normalized: Vec<[f64; 2]> = undistorted
        .iter()
        .map(|p| [(p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy])
        .collect();
    let plane: Vec<[f64; 2]> = obj.iter().map(|p| [p.x, p.y]).collect();
    let h = solve_dlt_homography(&plane, &normalized)?;
    let (h1, h2, h3) = (h.column(0), h.column(1), h.column(2));
    let (n1, n2) = (h1.norm(), h2.norm());
    if !(n1 > 0.0 && n2 > 0.0) {
        return None;
    }
    let mut scale = 2.0 / (n1 + n2);
    // The board is in front of the camera.
    if h3.z * scale < 0.0 {
        scale = -scale;
    }
    let r1: Vector3<f64> = h1 * scale;
    let r2: Vector3<f64> = h2 * scale;
    let t: Vector3<f64> = h3 * scale;
    let approx = Matrix3::from_columns(&[r1, r2, r1.cross(&r2)]);
    let svd = approx.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        r = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0)) * v_t;
    }
    let initial = Pose::new(r, t);

    let rms = |pose: &Pose| -> f64 {
        match project_points_with_distortion(obj, k, pose, &cam.distortion) {
            Ok(p) => p
                .iter()
                .zip(img)
                .map(|(a, b)| (a - b).norm_squared())
                .sum::<f64>(),
            Err(_) => f64::INFINITY,
        }
    };
    match solve_pnp_refine(&initial, obj, img, k, Some(&cam.distortion), 20) {
        Ok(refined) if rms(&refined) <= rms(&initial) => Some(refined),
        _ => rms(&initial).is_finite().then_some(initial),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::TensorShape;
    use nalgebra::Rotation3;

    /// A pinhole camera (with optional Brown-Conrady distortion) looking at a
    /// board in the `z = 0` plane: world point `p` maps to `R p + t`.
    struct Scene {
        k: Matrix3<f64>,
        dist: Distortion,
        r: Matrix3<f64>,
        t: Vector3<f64>,
        /// Inverse of `[r1 r2 t]`: normalised image point -> board plane.
        plane_from_ray: Matrix3<f64>,
        width: usize,
        height: usize,
    }

    impl Scene {
        /// Board centred in an 800x600 image, tilted `rx` degrees about the
        /// x axis and `ry` about the y axis, `distance` units away.
        fn tilted(board: &CharucoBoard, rx: f64, ry: f64, distance: f64, dist: Distortion) -> Self {
            let k = Matrix3::new(900.0, 0.0, 400.0, 0.0, 900.0, 300.0, 0.0, 0.0, 1.0);
            let r = (Rotation3::from_axis_angle(&Vector3::x_axis(), rx.to_radians())
                * Rotation3::from_axis_angle(&Vector3::y_axis(), ry.to_radians()))
            .into_inner();
            let centre = Vector3::new(
                board.squares_x as f64 * board.square_length / 2.0,
                board.squares_y as f64 * board.square_length / 2.0,
                0.0,
            );
            let t = Vector3::new(0.0, 0.0, distance) - r * centre;
            let plane_from_ray =
                Matrix3::from_columns(&[r.column(0).into_owned(), r.column(1).into_owned(), t])
                    .try_inverse()
                    .unwrap();
            Self {
                k,
                dist,
                r,
                t,
                plane_from_ray,
                width: 800,
                height: 600,
            }
        }

        fn project(&self, x: f64, y: f64) -> [f64; 2] {
            let pc = self.r * Vector3::new(x, y, 0.0) + self.t;
            let (xd, yd) = self.dist.apply(pc.x / pc.z, pc.y / pc.z);
            [
                self.k[(0, 0)] * xd + self.k[(0, 2)],
                self.k[(1, 1)] * yd + self.k[(1, 2)],
            ]
        }

        /// Board-plane point imaged at pixel `(px, py)`.
        fn unproject(&self, px: f64, py: f64) -> Option<[f64; 2]> {
            let xd = (px - self.k[(0, 2)]) / self.k[(0, 0)];
            let yd = (py - self.k[(1, 2)]) / self.k[(1, 1)];
            let (x, y) = self.dist.remove_checked(xd, yd)?;
            let b = self.plane_from_ray * Vector3::new(x, y, 1.0);
            Some([b.x / b.z, b.y / b.z])
        }

        /// Ground-truth image position of every chessboard corner, by id.
        fn corner_truth(&self, board: &CharucoBoard) -> Vec<[f64; 2]> {
            board
                .chessboard_corners()
                .iter()
                .map(|p| self.project(p.x, p.y))
                .collect()
        }

        /// Every board marker with its exactly projected corners.
        fn exact_markers(&self, board: &CharucoBoard) -> Vec<DetectedMarker> {
            board
                .marker_object_points()
                .iter()
                .enumerate()
                .map(|(id, obj)| DetectedMarker {
                    id,
                    corners: obj.map(|p| {
                        let q = self.project(p.x, p.y);
                        (q[0], q[1])
                    }),
                })
                .collect()
        }

        /// Render the board, 3x3 supersampled, white outside it. `flip`
        /// inverts payload bit `(x, y)` of one marker id.
        fn render(
            &self,
            board: &CharucoBoard,
            flip: Option<(usize, usize, usize)>,
        ) -> CpuTensor<f32> {
            const SS: usize = 3;
            let cells = marker_cells(board.squares_x, board.squares_y);
            let mut id_of_cell = vec![None; board.squares_x * board.squares_y];
            for (id, &(col, row)) in cells.iter().enumerate() {
                id_of_cell[row * board.squares_x + col] = Some(id);
            }
            let grids: Vec<Option<Vec<Vec<u8>>>> = (0..cells.len())
                .map(|id| board.dictionary.get_marker(id))
                .collect();
            let side = board.dictionary.marker_size();
            let grid = side + 2;
            let ratio = board.marker_length / board.square_length;
            let inset = (1.0 - ratio) / 2.0;
            let value = |b: [f64; 2]| -> f32 {
                let (u, v) = (b[0] / board.square_length, b[1] / board.square_length);
                if !(u >= 0.0
                    && v >= 0.0
                    && u < board.squares_x as f64
                    && v < board.squares_y as f64)
                {
                    return 1.0;
                }
                let (col, row) = (u.floor() as usize, v.floor() as usize);
                if (col + row) % 2 == 0 {
                    return 0.0;
                }
                let Some(id) = id_of_cell[row * board.squares_x + col] else {
                    return 1.0;
                };
                let Some(bits) = &grids[id] else { return 1.0 };
                let mu = (u - col as f64 - inset) / ratio;
                let mv = (v - row as f64 - inset) / ratio;
                if !((0.0..1.0).contains(&mu) && (0.0..1.0).contains(&mv)) {
                    return 1.0;
                }
                let gx = (mu * grid as f64) as usize;
                let gy = (mv * grid as f64) as usize;
                if gx == 0 || gy == 0 || gx == grid - 1 || gy == grid - 1 {
                    return 0.0;
                }
                let mut bit = bits[gy - 1][gx - 1];
                if flip == Some((id, gx - 1, gy - 1)) {
                    bit ^= 1;
                }
                if bit == 1 { 0.0 } else { 1.0 }
            };
            let mut pixels = vec![0.0f32; self.width * self.height];
            for y in 0..self.height {
                for x in 0..self.width {
                    let mut acc = 0.0;
                    for sy in 0..SS {
                        for sx in 0..SS {
                            let px = x as f64 + (sx as f64 + 0.5) / SS as f64 - 0.5;
                            let py = y as f64 + (sy as f64 + 0.5) / SS as f64 - 0.5;
                            acc += self.unproject(px, py).map_or(1.0, value);
                        }
                    }
                    pixels[y * self.width + x] = acc / (SS * SS) as f32;
                }
            }
            CpuTensor::from_vec(pixels, TensorShape::new(1, self.height, self.width)).unwrap()
        }
    }

    /// 7x5 squares, markers 3/4 of a square: 17 markers, 24 corners.
    fn test_board() -> CharucoBoard {
        CharucoBoard::new(7, 5, 1.0, 0.75, ArucoDictionary::Dict4x4_50)
    }

    fn detector(board: &CharucoBoard, params: CharucoParameters) -> CharucoDetector {
        CharucoDetector::new(
            board.clone(),
            params,
            ArucoDetector::new(ArucoDictionary::Dict4x4_50),
        )
    }

    fn with_camera(scene: &Scene, dist: Option<Vec<f64>>) -> CharucoParameters {
        CharucoParameters {
            camera_matrix: Some(scene.k),
            dist_coeffs: dist,
            ..Default::default()
        }
    }

    /// `(mean, max)` pixel distance of the found corners from the truth.
    fn corner_errors(found: &CharucoCorners, truth: &[[f64; 2]]) -> (f64, f64) {
        assert!(!found.ids.is_empty(), "no corners found");
        let errs: Vec<f64> = found
            .ids
            .iter()
            .zip(&found.corners)
            .map(|(&id, c)| {
                let t = truth[id as usize];
                (c[0] as f64 - t[0]).hypot(c[1] as f64 - t[1])
            })
            .collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        (mean, errs.iter().cloned().fold(0.0, f64::max))
    }

    /// The interpolation this module used before local homographies: the
    /// mean of the adjacent marker corners lying on each chessboard corner.
    fn diagonal_average(board: &CharucoBoard, markers: &[DetectedMarker]) -> CharucoCorners {
        let mut out = CharucoCorners::empty();
        for (id, adjacent) in nearest_markers(board.squares_x, board.squares_y)
            .iter()
            .enumerate()
        {
            let pts: Vec<(f64, f64)> = adjacent
                .iter()
                .filter_map(|&(mid, k)| markers.iter().find(|m| m.id == mid).map(|m| m.corners[k]))
                .collect();
            if pts.len() == 2 {
                out.corners.push([
                    ((pts[0].0 + pts[1].0) / 2.0) as f32,
                    ((pts[0].1 + pts[1].1) / 2.0) as f32,
                ]);
                out.ids.push(id as i32);
            }
        }
        out
    }

    /// With exact marker corners under strong perspective, each local
    /// homography reproduces the board plane exactly; the midpoint of two
    /// marker corners does not, since perspective does not preserve
    /// midpoints.
    #[test]
    fn local_homography_is_exact_under_perspective_where_averaging_is_not() {
        let board = test_board();
        let scene = Scene::tilted(&board, 25.0, -35.0, 10.0, Distortion::none());
        let truth = scene.corner_truth(&board);
        let markers = scene.exact_markers(&board);

        let found = detector(&board, CharucoParameters::default())
            .interpolate_corners(&markers)
            .unwrap();
        assert_eq!(found.ids, (0..24).collect::<Vec<i32>>());
        let (hom_mean, hom_max) = corner_errors(&found, &truth);
        let (avg_mean, avg_max) = corner_errors(&diagonal_average(&board, &markers), &truth);
        eprintln!(
            "exact marker corners, perspective: local homography mean {hom_mean:.2e} max \
             {hom_max:.2e} px; diagonal average mean {avg_mean:.3} max {avg_max:.3} px"
        );
        assert!(hom_max < 1e-3, "local homography max error {hom_max}");
        assert!(avg_max > 0.1, "averaging should be visibly off: {avg_max}");
    }

    /// The pose mode recovers the board pose through lens distortion, so it
    /// places every corner, even one whose markers were not detected when
    /// `min_markers = 0`. Local homographies cannot model the distortion.
    #[test]
    fn pose_mode_projects_through_lens_distortion() {
        let board = test_board();
        let coeffs = vec![-0.25, 0.08, 0.001, -0.002, 0.0];
        let dist = Distortion::new(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
        let scene = Scene::tilted(&board, 20.0, 30.0, 9.0, dist);
        let truth = scene.corner_truth(&board);
        let markers = scene.exact_markers(&board);

        let pose = detector(&board, with_camera(&scene, Some(coeffs.clone())))
            .interpolate_corners(&markers)
            .unwrap();
        let local = detector(&board, CharucoParameters::default())
            .interpolate_corners(&markers)
            .unwrap();
        assert_eq!(pose.ids.len(), 24);
        let (pose_mean, pose_max) = corner_errors(&pose, &truth);
        let (local_mean, local_max) = corner_errors(&local, &truth);
        eprintln!(
            "exact marker corners, distortion: pose mean {pose_mean:.2e} max {pose_max:.2e} px; \
             local homography mean {local_mean:.3} max {local_max:.3} px"
        );
        assert!(pose_max < 1e-3, "pose max error {pose_max}");
        assert!(local_max > pose_max * 10.0);

        // Keep only the markers of the top row: the pose still places every
        // corner, but only min_markers = 0 reports the unseen ones.
        let top: Vec<DetectedMarker> = markers.iter().filter(|m| m.id < 3).cloned().collect();
        let seen = detector(&board, with_camera(&scene, Some(coeffs.clone())))
            .interpolate_corners(&top)
            .unwrap();
        let all = detector(
            &board,
            CharucoParameters {
                min_markers: 0,
                ..with_camera(&scene, Some(coeffs))
            },
        )
        .interpolate_corners(&top)
        .unwrap();
        assert_eq!(all.ids.len(), 24);
        assert!(seen.ids.len() < 24, "min_markers 2 kept {:?}", seen.ids);
        let (_, all_max) = corner_errors(&all, &truth);
        assert!(
            all_max < 1e-2,
            "pose from three markers, max error {all_max}"
        );
    }

    /// End to end on rendered boards: every corner is found with its id, and
    /// `cornerSubPix` brings both interpolation modes onto the true corner
    /// even though the detector's marker corners are bounding boxes. The
    /// first view is turned 22 degrees about the vertical axis; in the second
    /// (8 degrees about the horizontal axis) the ArUco detector also reports
    /// marker 5 twice and a stray id 17, which the board does not have.
    #[test]
    fn detect_finds_every_corner_of_a_tilted_board() {
        let board = test_board();
        for (rx, ry, expected_markers) in [(0.0, 22.0, 17), (8.0, 0.0, 19)] {
            let scene = Scene::tilted(&board, rx, ry, 11.0, Distortion::none());
            let truth = scene.corner_truth(&board);
            let image = scene.render(&board, None);

            let markers = ArucoDetector::new(ArucoDictionary::Dict4x4_50)
                .detect(&image)
                .unwrap();
            let mut ids: Vec<usize> = markers.iter().map(|m| m.id).collect();
            ids.sort_unstable();
            ids.dedup();
            assert!(ids.starts_with(&(0..17).collect::<Vec<_>>()), "ids {ids:?}");
            assert_eq!(markers.len(), expected_markers, "ids {ids:?}");

            let local = detector(&board, CharucoParameters::default());
            let pose = detector(&board, with_camera(&scene, None));
            let (avg_mean, avg_max) = corner_errors(&diagonal_average(&board, &markers), &truth);
            let (hom_mean, hom_max) =
                corner_errors(&local.interpolate_corners(&markers).unwrap(), &truth);
            let (pose_mean, pose_max) =
                corner_errors(&pose.interpolate_corners(&markers).unwrap(), &truth);
            eprintln!(
                "rendered board rx {rx} ry {ry}, before cornerSubPix: diagonal average mean \
                 {avg_mean:.3} max {avg_max:.3}; local homography mean {hom_mean:.3} max \
                 {hom_max:.3}; pose mean {pose_mean:.3} max {pose_max:.3} px"
            );

            for (name, det) in [("local homography", &local), ("pose", &pose)] {
                let found = det.detect(&image).unwrap();
                assert_eq!(found.ids, (0..24).collect::<Vec<i32>>(), "{name}");
                let (mean, max) = corner_errors(&found, &truth);
                eprintln!(
                    "rendered board rx {rx} ry {ry}, detect ({name}): mean {mean:.3} max \
                     {max:.3} px"
                );
                assert!(mean < 0.2 && max < 0.4, "{name}: mean {mean}, max {max}");
            }
        }
    }

    /// A marker read with the wrong id sits where the board says another
    /// marker should be. Without the check it drags the corners next to its
    /// claimed cell by squares; with it, the marker is dropped and every
    /// reported corner is exact. Covers the real marker being missed and the
    /// real marker being detected after the misread copy (which plain
    /// first-come deduplication would discard).
    #[test]
    fn check_markers_drops_a_misidentified_marker() {
        let board = test_board();
        let scene = Scene::tilted(&board, 25.0, -35.0, 10.0, Distortion::none());
        let truth = scene.corner_truth(&board);
        for real_16_detected in [false, true] {
            let mut markers = scene.exact_markers(&board);
            // Marker 0 (top-left) is read as 16 (bottom-right).
            if !real_16_detected {
                markers.retain(|m| m.id != 16);
            }
            markers[0].id = 16;
            let params = |check_markers| CharucoParameters {
                check_markers,
                min_markers: 1,
                ..Default::default()
            };

            let unchecked = detector(&board, params(false))
                .interpolate_corners(&markers)
                .unwrap();
            let (_, unchecked_max) = corner_errors(&unchecked, &truth);
            let checked_detector = detector(&board, params(true));
            let checked = checked_detector.interpolate_corners(&markers).unwrap();
            let (_, checked_max) = corner_errors(&checked, &truth);
            eprintln!(
                "misidentified marker (real 16 detected: {real_16_detected}): max corner error \
                 {unchecked_max:.1} px unchecked, {checked_max:.2e} px checked ({} corners)",
                checked.ids.len()
            );
            assert!(unchecked_max > 50.0, "unchecked max error {unchecked_max}");
            assert!(checked_max < 1e-3, "checked max error {checked_max}");
            // Every corner touches two markers, and no two adjacent markers
            // are lost, so min_markers = 1 keeps all of them.
            assert_eq!(checked.ids.len(), 24, "ids {:?}", checked.ids);

            let kept = checked_detector.reject_inconsistent_markers(markers.clone(), None);
            assert_eq!(kept.len(), markers.len() - 1);
            let sixteens: Vec<_> = kept.iter().filter(|m| m.id == 16).collect();
            assert_eq!(sixteens.len(), usize::from(real_16_detected));
        }
    }

    /// OpenCV's `checkBoard`: with too few markers to fit a board homography,
    /// a marker lying on top of another marker's corner contradicts the
    /// layout, and the whole board is rejected.
    #[test]
    fn check_markers_rejects_a_board_that_contradicts_the_layout() {
        let board = test_board();
        let scene = Scene::tilted(&board, 0.0, 0.0, 10.0, Distortion::none());
        let exact = scene.exact_markers(&board);
        // Marker 4 (cell (2, 1)) plus a small "marker 16" centred on the
        // corner marker 4 forms with its top-left neighbour.
        let corner = scene.project(2.0, 1.0);
        let fake = DetectedMarker {
            id: 16,
            corners: [(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)]
                .map(|(dx, dy)| (corner[0] + dx, corner[1] + dy)),
        };
        let markers = vec![exact[4].clone(), fake];
        let params = |check_markers| CharucoParameters {
            check_markers,
            min_markers: 1,
            ..Default::default()
        };
        let unchecked = detector(&board, params(false))
            .interpolate_corners(&markers)
            .unwrap();
        let checked = detector(&board, params(true))
            .interpolate_corners(&markers)
            .unwrap();
        assert!(!unchecked.ids.is_empty());
        assert!(checked.ids.is_empty(), "checked ids {:?}", checked.ids);

        // The genuine marker alone passes.
        let alone = detector(&board, params(true))
            .interpolate_corners(&markers[..1])
            .unwrap();
        assert_eq!(alone.ids.len(), 4, "ids {:?}", alone.ids);
    }

    /// A marker with one damaged payload bit is rejected by the ArUco
    /// detector; `try_refine_markers` finds the rejected candidate where the
    /// board says marker 8 belongs and accepts its 1-bit code distance.
    #[test]
    fn try_refine_markers_recovers_a_damaged_marker() {
        let board = test_board();
        let scene = Scene::tilted(&board, 0.0, 10.0, 11.0, Distortion::none());
        let truth = scene.corner_truth(&board);
        let image = scene.render(&board, Some((8, 1, 2)));

        let markers = ArucoDetector::new(ArucoDictionary::Dict4x4_50)
            .detect(&image)
            .unwrap();
        assert!(markers.iter().all(|m| m.id != 8));
        assert_eq!(markers.len(), 16);

        let plain = detector(&board, CharucoParameters::default())
            .detect(&image)
            .unwrap();
        let refined = detector(
            &board,
            CharucoParameters {
                try_refine_markers: true,
                ..Default::default()
            },
        )
        .detect(&image)
        .unwrap();
        // Marker 8 sits in cell (3, 2) and touches corners 8, 9, 14, 15.
        for id in [8, 9, 14, 15] {
            assert!(!plain.ids.contains(&id), "plain ids {:?}", plain.ids);
        }
        assert_eq!(refined.ids, (0..24).collect::<Vec<i32>>());
        let (_, max) = corner_errors(&refined, &truth);
        assert!(max < 0.4, "refined max error {max}");
    }

    /// `payload_distance_to_id` reads the marker in board orientation: a
    /// marker seen turned clockwise once has its top-left corner at image
    /// corner 1.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn rotated_payload_is_read_in_board_orientation() {
        let dict = ArucoDictionary::Dict4x4_50;
        let grid = dict.get_marker(7).unwrap();
        let side = grid.len();
        // Turn the bit grid clockwise: image (x, y) shows marker (y, side-1-x).
        let mut payload = 0u64;
        for y in 0..side {
            for x in 0..side {
                if grid[side - 1 - x][y] == 1 {
                    payload |= 1 << (y * side + x);
                }
            }
        }
        assert_eq!(payload_distance_to_id(dict, payload, 7, 1), Some(0));
        assert_ne!(payload_distance_to_id(dict, payload, 7, 0), Some(0));
        assert_ne!(payload_distance_to_id(dict, payload, 7, 3), Some(0));
    }

    #[test]
    fn invalid_parameters_are_reported() {
        let board = test_board();
        let markers =
            Scene::tilted(&board, 0.0, 0.0, 10.0, Distortion::none()).exact_markers(&board);
        let run = |params| detector(&board, params).interpolate_corners(&markers);
        assert!(
            run(CharucoParameters {
                min_markers: 3,
                ..Default::default()
            })
            .is_err()
        );
        let mut skewed = Matrix3::new(900.0, 1.0, 400.0, 0.0, 900.0, 300.0, 0.0, 0.0, 1.0);
        assert!(
            run(CharucoParameters {
                camera_matrix: Some(skewed),
                ..Default::default()
            })
            .is_err()
        );
        skewed[(0, 1)] = 0.0;
        assert!(
            run(CharucoParameters {
                camera_matrix: Some(skewed),
                dist_coeffs: Some(vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            run(CharucoParameters {
                camera_matrix: Some(skewed),
                dist_coeffs: Some(vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn chessboard_corners_cover_full_id_grid() {
        // 3x4 squares yields (3-1)*(4-1) = 6 interior corners, row-major, so
        // the index i of each corner equals the id assigned by
        // `interpolate_corners`.
        let board = CharucoBoard::new(3, 4, 1.0, 0.5, ArucoDictionary::Dict4x4_50);
        let pts = board.chessboard_corners();
        assert_eq!(pts.len(), (3 - 1) * (4 - 1));

        let expected = [
            (1.0, 1.0),
            (2.0, 1.0),
            (1.0, 2.0),
            (2.0, 2.0),
            (1.0, 3.0),
            (2.0, 3.0),
        ];
        for (i, (x, y)) in expected.iter().enumerate() {
            assert_eq!(pts[i].x, *x, "corner {} has wrong x", i);
            assert_eq!(pts[i].y, *y, "corner {} has wrong y", i);
            assert_eq!(pts[i].z, 0.0);
        }
    }

    /// Markers sit inside the white squares, inset from the square's edges, so
    /// an interior chessboard corner is the midpoint of the two marker corners
    /// that face it diagonally. With every marker seen exactly, interpolation
    /// must put each corner on its grid point, with ids ascending.
    #[test]
    fn interpolated_corners_land_on_their_grid_points() {
        let (sx, sy) = (5usize, 4usize);
        let (scale, inset) = (100.0, 10.0);
        let board = CharucoBoard::new(sx, sy, 1.0, 0.8, ArucoDictionary::Dict4x4_50);
        let params = CharucoParameters {
            min_markers: 2,
            ..Default::default()
        };
        let detector = CharucoDetector::new(
            board.clone(),
            params,
            ArucoDetector::new(ArucoDictionary::Dict4x4_50),
        );

        let mut markers = Vec::new();
        let mut id = 0;
        for row in 0..sy {
            for col in 0..sx {
                if (row + col) % 2 == 1 {
                    let (x0, y0) = (col as f64 * scale, row as f64 * scale);
                    let (x1, y1) = (x0 + scale, y0 + scale);
                    markers.push(DetectedMarker {
                        id,
                        corners: [
                            (x0 + inset, y0 + inset),
                            (x1 - inset, y0 + inset),
                            (x1 - inset, y1 - inset),
                            (x0 + inset, y1 - inset),
                        ],
                    });
                    id += 1;
                }
            }
        }

        let found = detector.interpolate_corners(&markers).unwrap();
        let expected = board.chessboard_corners();
        assert_eq!(found.ids.len(), expected.len(), "ids: {:?}", found.ids);
        for (i, (&id, c)) in found.ids.iter().zip(&found.corners).enumerate() {
            assert_eq!(id as usize, i, "ids not ascending: {:?}", found.ids);
            let e = &expected[i];
            assert!(
                (c[0] as f64 - e.x * scale).abs() < 1e-3
                    && (c[1] as f64 - e.y * scale).abs() < 1e-3,
                "corner {id} at {c:?}, expected ({}, {})",
                e.x * scale,
                e.y * scale
            );
        }
    }
}
