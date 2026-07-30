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

    /// Chessboard corner world coordinates (z=0 plane)
    pub fn chessboard_corners(&self) -> Vec<Point3<f64>> {
        let mut pts = Vec::new();
        for row in 0..self.squares_y.saturating_sub(1) {
            for col in 0..self.squares_x.saturating_sub(1) {
                pts.push(Point3::new(
                    (col as f64 + 0.5) * self.square_length,
                    (row as f64 + 0.5) * self.square_length,
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
            min_markers: 2,
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
        let mut corner_data: HashMap<(usize, usize), Vec<[f64; 2]>> = HashMap::new();

        for marker in markers {
            let sx = self.board.squares_x;
            if sx <= 1 { continue; }
            let grid_i = marker.id / (sx - 1);
            let grid_j = marker.id % (sx - 1);

            let corner_indices: [(usize, usize); 4] = [
                (grid_i, grid_j),
                (grid_i, grid_j + 1),
                (grid_i + 1, grid_j),
                (grid_i + 1, grid_j + 1),
            ];

            for (idx, &(ci, cj)) in corner_indices.iter().enumerate() {
                if ci < self.board.squares_y - 1 && cj < sx - 1 {
                    let pt = marker.corners[idx];
                    corner_data.entry((ci, cj)).or_default().push([pt.0, pt.1]);
                }
            }
        }

        let mut corners = Vec::new();
        let mut ids = Vec::new();

        for (&(gi, gj), pts) in &corner_data {
            if pts.len() < self.params.min_markers {
                continue;
            }
            let avg_x = pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64;
            let avg_y = pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64;
            let corner_id = (gi * (self.board.squares_x - 1) + gj) as i32;
            corners.push([avg_x as f32, avg_y as f32]);
            ids.push(corner_id);
        }

        Ok(CharucoCorners { corners, ids })
    }
}
