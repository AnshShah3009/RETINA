//! ChArUco board detection — fiducial marker board with chessboard corners
//!
//! A ChArUco board combines ArUco markers with a chessboard pattern for
//! precise camera calibration and sub-pixel corner refinement.

use crate::aruco::{ArucoDetector, ArucoDictionary, DetectedMarker};
use cv_core::{Float, CpuTensor, Result};
use nalgebra::{Matrix3, Point3};
use std::collections::HashMap;

/// ChArUco board combining ArUco markers with chessboard pattern
#[derive(Clone)]
pub struct CharucoBoard {
    pub squares_x: usize,
    pub squares_y: usize,
    pub square_length: f64,
    pub marker_length: f64,
    pub dictionary: ArucoDictionary,
}

impl CharucoBoard {
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
    /// col in 1..squares_x-1, row in 1..squares_y-1, ordered row-major so
    /// that index i corresponds to corner id i from `interpolate_corners`.
    /// A previous revision used square CENTERS ((col+0.5)*L), offsetting
    /// every calibration observation by half a square.
    pub fn chessboard_corners(&self) -> Vec<Point3<f64>> {
        let mut pts = Vec::new();
        for row in 1..self.squares_y.saturating_sub(1) {
            for col in 1..self.squares_x.saturating_sub(1) {
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
    pub camera_matrix: Option<Matrix3<f64>>,
    pub dist_coeffs: Option<Vec<f64>>,
    pub min_markers: usize,
    pub try_refine_markers: bool,
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
    pub corners: Vec<[f32; 2]>,
    pub ids: Vec<i32>,
}

/// ChArUco detector
pub struct CharucoDetector {
    board: CharucoBoard,
    params: CharucoParameters,
    aruco_detector: ArucoDetector,
}

impl CharucoDetector {
    pub fn new(
        board: CharucoBoard,
        params: CharucoParameters,
        detector: ArucoDetector,
    ) -> Self {
        Self {
            board,
            params,
            aruco_detector: detector,
        }
    }

    /// Detect ChArUco board corners from image.
    /// Internally detects ArUco markers first, then interpolates chessboard corner positions.
    pub fn detect<T: Float>(
        &self,
        image: &CpuTensor<T>,
    ) -> Result<CharucoCorners> {
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
            return Ok(CharucoCorners { corners: vec![], ids: vec![] });
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

        // Accumulate observations per INTERSECTION (ic, ir), where
        // ic in 0..=sx-2 indexes intersections by column.
        let mut corner_data: HashMap<(usize, usize), Vec<[f64; 2]>> = HashMap::new();

        for marker in markers {
            let Some(&(col, row)) = cell_of_id.get(&(marker.id as usize)) else {
                continue;
            };
            if col >= sx - 1 || row >= sy - 1 {
                continue; // marker on outer ring contributes no interior corners
            }

            // Marker image corners are TL, TR, BR, BL; each maps to the
            // intersection at its own grid corner:
            //   TL -> (col,     row)
            //   TR -> (col + 1, row)
            //   BR -> (col + 1, row + 1)
            //   BL -> (col,     row + 1)
            for (ic, ir, pt) in [
                (col, row, marker.corners[0]),
                (col + 1, row, marker.corners[1]),
                (col + 1, row + 1, marker.corners[2]),
                (col, row + 1, marker.corners[3]),
            ] {
                if ic <= sx - 2 && ir <= sy - 2 {
                    corner_data
                        .entry((ic, ir))
                        .or_default()
                        .push([pt.0, pt.1]);
                }
            }
        }

        let mut corners = Vec::new();
        let mut ids = Vec::new();
        // Emit row-major over intersections: id = ir*(sx-1) + ic.
        let mut sorted_keys: Vec<(usize, usize)> = corner_data.keys().copied().collect();
        sorted_keys.sort();
        for (ic, ir) in sorted_keys {
            let Some(pts) = corner_data.get(&(ic, ir)) else { continue };
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
