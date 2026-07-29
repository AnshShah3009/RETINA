pub type CalibError = cv_core::Error;
pub type Result<T> = cv_core::Result<T>;

// Module declarations
pub mod distortion;
pub use distortion::{init_undistort_rectify_map, undistort_image, undistort_points};

pub mod project;
pub use project::{
    project_points, project_points_with_distortion, project_points_with_jacobian,
    ProjectPointsOptions, ProjectPointsResult,
};

pub mod pattern;
pub use pattern::{corner_subpix, find_chessboard_corners};

pub mod pnp;
pub use pnp::{solve_pnp_dlt, solve_pnp_ransac, solve_pnp_refine};

pub mod essential_fundamental;
pub use essential_fundamental::{
    essential_from_extrinsics, find_essential_mat, find_essential_mat_ransac, find_fundamental_mat,
    find_fundamental_mat_ransac, fundamental_from_essential,
};

pub mod triangulation;
pub use triangulation::{recover_pose_from_essential, triangulate_points};

pub mod calibration;
pub use calibration::{
    calibrate_camera_from_chessboard_files, calibrate_camera_from_chessboard_files_with_options,
    calibrate_camera_from_chessboard_images, calibrate_camera_from_chessboard_images_with_options,
    calibrate_camera_planar, calibrate_camera_planar_with_options,
    generate_chessboard_object_points, refine_camera_calibration_iterative, CalibrationFileReport,
    CameraCalibrationOptions, CameraCalibrationResult,
};

pub mod essential;
pub mod fundamental;
pub mod homography;
pub mod multiview;
pub use multiview::{
    EssentialSolver, FundamentalSolver, HomographySolver, PnpSolver, Triangulator,
};

pub mod stereo;
pub use stereo::{
    stereo_calibrate_from_chessboard_files, stereo_calibrate_from_chessboard_files_with_options,
    stereo_calibrate_from_chessboard_images, stereo_calibrate_from_chessboard_images_with_options,
    stereo_calibrate_planar, stereo_calibrate_planar_with_options, stereo_rectify_matrices,
    StereoCalibrationFileReport, StereoCalibrationResult, StereoRectifyMatrices,
};

pub mod stereo_matching;
pub use stereo_matching::*;

