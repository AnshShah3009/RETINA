//! ChArUco board detection — fiducial marker board with chessboard corners
//!
//! A ChArUco board combines ArUco markers with a chessboard pattern for
//! precise camera calibration and sub-pixel corner refinement.

use crate::aruco::{ArucoDetector, ArucoDictionary, DetectedMarker};
use cv_core::{CpuTensor, Float, Result};
use nalgebra::{Matrix3, Point3};
use std::collections::HashMap;

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
}

/// ChArUco detection parameters
#[derive(Clone)]
pub struct CharucoParameters {
    /// Camera intrinsics. Not used yet: corners are interpolated from marker
    /// corners in the image, without a homography or undistortion step.
    pub camera_matrix: Option<Matrix3<f64>>,
    /// Distortion coefficients. Not used yet (see `camera_matrix`).
    pub dist_coeffs: Option<Vec<f64>>,
    /// How many adjacent marker corners must observe an interior corner
    /// before it is reported (1 or 2; each interior corner touches at most
    /// two markers).
    pub min_markers: usize,
    /// Refine rejected markers against the board layout. Not used yet.
    pub try_refine_markers: bool,
    /// Check marker consistency against the board. Not used yet.
    pub check_markers: bool,
}

impl Default for CharucoParameters {
    fn default() -> Self {
        Self {
            camera_matrix: None,
            dist_coeffs: None,
            min_markers: 1,
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

/// ChArUco detector
pub struct CharucoDetector {
    board: CharucoBoard,
    params: CharucoParameters,
    aruco_detector: ArucoDetector,
}

impl CharucoDetector {
    /// A detector for `board` that finds markers with `detector`.
    pub fn new(board: CharucoBoard, params: CharucoParameters, detector: ArucoDetector) -> Self {
        Self {
            board,
            params,
            aruco_detector: detector,
        }
    }

    /// Detect ChArUco board corners from image.
    /// Internally detects ArUco markers first, then interpolates chessboard corner positions.
    pub fn detect<T: Float>(&self, image: &CpuTensor<T>) -> Result<CharucoCorners> {
        let markers = self.aruco_detector.detect(image)?;
        self.interpolate_corners(&markers)
    }

    fn interpolate_corners(&self, markers: &[DetectedMarker]) -> Result<CharucoCorners> {
        // Marker placement follows the OpenCV ChArUco convention: markers
        // sit on cells where (row+col) is odd, ids assigned row-major over
        // those cells. The previous id/(sx-1) mapping matched no board
        // layout, and corners were paired to intersections inconsistently.
        let sx = self.board.squares_x;
        let sy = self.board.squares_y;
        if sx < 2 || sy < 2 {
            return Ok(CharucoCorners {
                corners: vec![],
                ids: vec![],
            });
        }

        // marker id -> cell (col, row)
        let mut cell_of_id: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut next_id = 0usize;
        for row in 0..sy {
            for col in 0..sx {
                if (row + col) % 2 == 1 {
                    cell_of_id.insert(next_id, (col, row));
                    next_id += 1;
                }
            }
        }

        // Accumulate observations per interior corner (ic, ir). Interior
        // corner (ic, ir) is grid point (ic + 1, ir + 1); ic in 0..=sx-2.
        // Each marker corner sits, inset, next to the grid point of its own
        // square corner, and the two white squares meeting at an interior
        // corner face it diagonally, so averaging their two marker corners
        // cancels the inset. (This used grid point (col, row) as the corner
        // index itself, which put every corner one square off and dropped
        // the outer-ring markers that do touch interior corners.)
        let mut corner_data: HashMap<(usize, usize), Vec<[f64; 2]>> = HashMap::new();

        for marker in markers {
            let Some(&(col, row)) = cell_of_id.get(&{ marker.id }) else {
                continue;
            };

            // Marker image corners are TL, TR, BR, BL; grid points:
            //   TL -> (col,     row)
            //   TR -> (col + 1, row)
            //   BR -> (col + 1, row + 1)
            //   BL -> (col,     row + 1)
            for (gx, gy, pt) in [
                (col, row, marker.corners[0]),
                (col + 1, row, marker.corners[1]),
                (col + 1, row + 1, marker.corners[2]),
                (col, row + 1, marker.corners[3]),
            ] {
                if (1..sx).contains(&gx) && (1..sy).contains(&gy) {
                    corner_data
                        .entry((gx - 1, gy - 1))
                        .or_default()
                        .push([pt.0, pt.1]);
                }
            }
        }

        let mut corners = Vec::new();
        let mut ids = Vec::new();
        // Emit row-major over intersections: id = ir*(sx-1) + ic.
        let mut sorted_keys: Vec<(usize, usize)> = corner_data.keys().copied().collect();
        // Keys are (col, row); sort by (row, col) so ids come out ascending,
        // as OpenCV returns them. A plain sort() ordered them column-major.
        sorted_keys.sort_by_key(|&(ic, ir)| (ir, ic));
        for (ic, ir) in sorted_keys {
            let Some(pts) = corner_data.get(&(ic, ir)) else {
                continue;
            };
            if pts.len() < self.params.min_markers {
                continue;
            }
            let avg_x = pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64;
            let avg_y = pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64;
            corners.push([avg_x as f32, avg_y as f32]);
            ids.push((ir * (sx - 1) + ic) as i32);
        }

        Ok(CharucoCorners { corners, ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
