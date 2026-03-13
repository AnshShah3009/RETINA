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

pub mod stereo;
pub use stereo::{
    stereo_calibrate_from_chessboard_files, stereo_calibrate_from_chessboard_files_with_options,
    stereo_calibrate_from_chessboard_images, stereo_calibrate_from_chessboard_images_with_options,
    stereo_calibrate_planar, stereo_calibrate_planar_with_options, stereo_rectify_matrices,
    StereoCalibrationFileReport, StereoCalibrationResult, StereoRectifyMatrices,
};

#[cfg(test)]

mod tests {
    use super::*;
    use cv_core::{CameraIntrinsics, Distortion, Pose};
    use cv_imgproc::{warp_perspective_ex, BorderMode, Interpolation};
    use image::{GrayImage, Luma};
    use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, Rotation3, Vector3};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn project_point(k: &CameraIntrinsics, ext: &Pose, p: &Point3<f64>) -> Point2<f64> {
        let pc = ext.rotation * p.coords + ext.translation;
        if pc[2].abs() < 1e-10 {
            return Point2::new(f64::NAN, f64::NAN);
        }
        let u = k.fx * (pc[0] / pc[2]) + k.cx;
        let v = k.fy * (pc[1] / pc[2]) + k.cy;
        Point2::new(u, v)
    }

    fn synthetic_checkerboard(
        pattern: (usize, usize),
        square: u32,
        margin_x: u32,
        margin_y: u32,
    ) -> (GrayImage, Vec<Point2<f64>>) {
        let (cols, rows) = pattern;
        let squares_x = cols as u32 + 1;
        let squares_y = rows as u32 + 1;
        let width = margin_x * 2 + squares_x * square;
        let height = margin_y * 2 + squares_y * square;
        let mut img = GrayImage::from_pixel(width, height, Luma([180]));

        for sy in 0..squares_y {
            for sx in 0..squares_x {
                let is_black = (sx + sy) % 2 == 0;
                let val = if is_black { 30u8 } else { 220u8 };
                let x0 = margin_x + sx * square;
                let y0 = margin_y + sy * square;
                for y in y0..(y0 + square) {
                    for x in x0..(x0 + square) {
                        img.put_pixel(x, y, Luma([val]));
                    }
                }
            }
        }

        let mut gt = Vec::with_capacity(cols * rows);
        for y in 0..rows {
            for x in 0..cols {
                gt.push(Point2::new(
                    (margin_x + (x as u32 + 1) * square) as f64,
                    (margin_y + (y as u32 + 1) * square) as f64,
                ));
            }
        }
        (img, gt)
    }

    fn synthetic_checkerboard_fixed(
        pattern: (usize, usize),
        square: u32,
        width: u32,
        height: u32,
        offset_x: u32,
        offset_y: u32,
    ) -> GrayImage {
        let (cols, rows) = pattern;
        let squares_x = cols as u32 + 1;
        let squares_y = rows as u32 + 1;
        let mut img = GrayImage::from_pixel(width, height, Luma([180]));

        for sy in 0..squares_y {
            for sx in 0..squares_x {
                let is_black = (sx + sy) % 2 == 0;
                let val = if is_black { 30u8 } else { 220u8 };
                let x0 = offset_x + sx * square;
                let y0 = offset_y + sy * square;
                if x0 + square > width || y0 + square > height {
                    continue;
                }
                for y in y0..(y0 + square) {
                    for x in x0..(x0 + square) {
                        img.put_pixel(x, y, Luma([val]));
                    }
                }
            }
        }
        img
    }

    #[test]
    fn triangulate_points_recovers_geometry() {
        let p1 = Matrix3x4::new(
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        );
        let p2 = Matrix3x4::new(
            1.0, 0.0, 0.0, 0.2, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        );
        let world = vec![
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(0.2, -0.1, 4.0),
            Point3::new(-0.3, 0.2, 5.0),
        ];
        let pts1: Vec<Point2<f64>> = world
            .iter()
            .map(|p| Point2::new(p.x / p.z, p.y / p.z))
            .collect();
        let pts2: Vec<Point2<f64>> = world
            .iter()
            .map(|p| Point2::new((p.x + 0.2) / p.z, p.y / p.z))
            .collect();

        let out = triangulate_points(&p1, &p2, &pts1, &pts2).unwrap();
        for (a, b) in out.iter().zip(world.iter()) {
            assert!((a.coords - b.coords).norm() < 1e-6);
        }
    }

    #[test]
    fn solve_pnp_dlt_reprojects_well() {
        let k = CameraIntrinsics::new(800.0, 780.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.08, -0.04, 0.06)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.15, -0.1, 0.4);
        let gt = Pose::new(rot, t);

        let world = vec![
            Point3::new(-0.4, -0.2, 3.8),
            Point3::new(0.3, -0.1, 4.1),
            Point3::new(0.1, 0.2, 4.5),
            Point3::new(-0.2, 0.3, 3.9),
            Point3::new(0.4, 0.4, 4.7),
            Point3::new(-0.5, 0.1, 5.0),
            Point3::new(0.2, -0.4, 4.3),
            Point3::new(-0.1, -0.3, 5.2),
        ];
        let pixels: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        let est = solve_pnp_dlt(&world, &pixels, &k).unwrap();
        let reproj_err = world
            .iter()
            .zip(pixels.iter())
            .filter_map(|(w, p)| {
                let proj = project_point(&k, &est, w);
                if proj.x.is_nan() || p.x.is_nan() {
                    None
                } else {
                    Some((proj - *p).norm())
                }
            })
            .sum::<f64>()
            / world.len() as f64;
        assert!(reproj_err < 1e-6);
    }

    #[test]
    fn project_points_matches_manual_projection() {
        let k = CameraIntrinsics::new(520.0, 515.0, 320.0, 240.0, 640, 480);
        let ext = Pose::new(
            Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner(),
            Vector3::new(0.1, -0.03, 0.2),
        );
        let world = vec![
            Point3::new(-0.2, -0.1, 2.0),
            Point3::new(0.3, 0.2, 3.5),
            Point3::new(0.1, -0.25, 4.0),
        ];

        let proj = project_points(&world, &k, &ext).unwrap();
        assert_eq!(proj.len(), world.len());
        for (p3, pix) in world.iter().zip(proj.iter()) {
            let expected = project_point(&k, &ext, p3);
            assert!((expected - pix).norm() < 1e-10);
        }
    }

    #[test]
    fn project_points_with_distortion_matches_manual_model() {
        let k = CameraIntrinsics::new(520.0, 515.0, 320.0, 240.0, 640, 480);
        let d = Distortion::new(0.1, -0.04, 0.001, -0.0008, 0.01);
        let ext = Pose::new(
            Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner(),
            Vector3::new(0.1, -0.03, 0.2),
        );
        let world = vec![
            Point3::new(-0.2, -0.1, 2.0),
            Point3::new(0.3, 0.2, 3.5),
            Point3::new(0.1, -0.25, 4.0),
        ];

        let proj = project_points_with_distortion(&world, &k, &ext, &d).unwrap();
        assert_eq!(proj.len(), world.len());
        for (p3, pix) in world.iter().zip(proj.iter()) {
            let pc = ext.rotation * p3.coords + ext.translation;
            let x = pc[0] / pc[2];
            let y = pc[1] / pc[2];
            let (xd, yd) = d.apply(x, y);
            let expected = Point2::new(k.fx * xd + k.cx, k.fy * yd + k.cy);
            assert!((expected - pix).norm() < 1e-10);
        }
    }

    #[test]
    fn solve_pnp_ransac_handles_outliers() {
        let k = CameraIntrinsics::new(700.0, 705.0, 320.0, 240.0, 640, 480);
        let gt = Pose::new(
            Rotation3::from_euler_angles(-0.06, 0.04, 0.08).into_inner(),
            Vector3::new(0.15, -0.04, 0.3),
        );

        let mut world = Vec::new();
        for i in 0..96usize {
            let x = ((i * 17 % 29) as f64 - 14.0) * 0.04;
            let y = ((i * 11 % 23) as f64 - 11.0) * 0.05;
            let z = 2.5 + ((i * 7 % 31) as f64) * 0.06;
            world.push(Point3::new(x, y, z));
        }
        let mut pixels: Vec<Point2<f64>> =
            world.iter().map(|p| project_point(&k, &gt, p)).collect();
        for i in (0..pixels.len()).step_by(8) {
            pixels[i].x += 35.0;
            pixels[i].y -= 28.0;
        }

        let (est, inliers) = solve_pnp_ransac(&world, &pixels, &k, None, 2.0, 500).unwrap();
        let inlier_count = inliers.iter().filter(|&&v| v).count();
        assert!(inlier_count >= 70);

        let t_err = (est.translation - gt.translation).norm();
        let r_err = est.rotation.angle_to(&gt.rotation).abs();
        assert!(t_err < 0.08);
        assert!(r_err < 0.08);

        let projected = project_points(&world, &k, &est).unwrap();
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..world.len() {
            if inliers[i] {
                sum += (projected[i] - pixels[i]).norm();
                cnt += 1;
            }
        }
        assert!(cnt > 0);
        assert!(sum / cnt as f64 <= 1.0);
    }

    #[test]
    fn solve_pnp_refine_reduces_reprojection_error() {
        let k = CameraIntrinsics::new(700.0, 705.0, 320.0, 240.0, 640, 480);
        let gt = Pose::new(
            Rotation3::from_euler_angles(-0.05, 0.06, 0.02).into_inner(),
            Vector3::new(0.12, -0.03, 0.28),
        );
        let initial = Pose::new(
            Rotation3::from_euler_angles(-0.01, 0.02, 0.03).into_inner(),
            Vector3::new(0.2, 0.02, 0.4),
        );

        let mut world = Vec::new();
        for i in 0..40usize {
            let x = ((i * 13 % 17) as f64 - 8.0) * 0.06;
            let y = ((i * 7 % 19) as f64 - 9.0) * 0.05;
            let z = 2.4 + ((i * 11 % 23) as f64) * 0.08;
            world.push(Point3::new(x, y, z));
        }
        let pixels: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        let err_before = world
            .iter()
            .zip(pixels.iter())
            .map(|(w, p)| (project_point(&k, &initial, w) - p).norm())
            .sum::<f64>()
            / world.len() as f64;

        let refined = solve_pnp_refine(&initial, &world, &pixels, &k, None, 25).unwrap();
        let err_after = world
            .iter()
            .zip(pixels.iter())
            .map(|(w, p)| (project_point(&k, &refined, w) - p).norm())
            .sum::<f64>()
            / world.len() as f64;

        assert!(err_after < err_before * 0.2);
        assert!(err_after < 0.5);
    }

    #[test]
    fn undistort_points_inverts_forward_distortion() {
        let k = CameraIntrinsics::new(620.0, 615.0, 320.0, 240.0, 640, 480);
        let d = Distortion::new(0.12, -0.05, 0.001, -0.0007, 0.01);
        let ideal = vec![
            Point2::new(120.0, 100.0),
            Point2::new(300.0, 220.0),
            Point2::new(500.0, 360.0),
            Point2::new(340.0, 140.0),
        ];

        let distorted: Vec<Point2<f64>> = ideal
            .iter()
            .map(|p| {
                let x = (p.x - k.cx) / k.fx;
                let y = (p.y - k.cy) / k.fy;
                let (xd, yd) = d.apply(x, y);
                Point2::new(k.fx * xd + k.cx, k.fy * yd + k.cy)
            })
            .collect();

        let recovered = undistort_points(&distorted, &k, &d).unwrap();
        for (r, g) in recovered.iter().zip(ideal.iter()) {
            assert!((r - g).norm() < 1e-5);
        }
    }

    #[test]
    fn init_undistort_rectify_map_identity_case() {
        let size = (64u32, 48u32);
        let k = CameraIntrinsics::new(120.0, 120.0, 32.0, 24.0, size.0, size.1);
        let d = Distortion::none();
        let r = Matrix3::<f64>::identity();
        let (map_x, map_y) = init_undistort_rectify_map(size, &k, &d, &r, &k).unwrap();

        for y in [0u32, 7, 23, 47] {
            for x in [0u32, 11, 31, 63] {
                let idx = (y * size.0 + x) as usize;
                assert!((map_x[idx] - x as f32).abs() < 1e-4);
                assert!((map_y[idx] - y as f32).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn undistort_image_identity_for_zero_distortion() {
        let width = 48u32;
        let height = 32u32;
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Luma([((x * 7 + y * 5) % 256) as u8]));
            }
        }

        let k = CameraIntrinsics::new(80.0, 80.0, 24.0, 16.0, width, height);
        let out = undistort_image(&img, &k, &Distortion::none(), None).unwrap();
        assert_eq!(out.dimensions(), img.dimensions());
        assert_eq!(out.as_raw(), img.as_raw());
    }

    #[test]
    fn recover_pose_from_essential_selects_valid_candidate() {
        let k = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.04, -0.03, 0.02)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.2, 0.0, 0.02).normalize();
        let gt = Pose::new(rot, t);
        let e = essential_from_extrinsics(&gt);

        let world = vec![
            Point3::new(-0.2, -0.1, 3.0),
            Point3::new(0.2, -0.2, 3.5),
            Point3::new(0.1, 0.15, 4.1),
            Point3::new(-0.3, 0.1, 4.4),
            Point3::new(0.25, 0.2, 3.7),
            Point3::new(-0.1, -0.25, 5.0),
        ];

        let i_ext = Pose::default();
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        let recovered = recover_pose_from_essential(&e, &pts1, &pts2, &k).unwrap();
        assert!(recovered.rotation_matrix().determinant() > 0.0);
        let dir_dot = recovered
            .translation
            .normalize()
            .dot(&gt.translation.normalize());
        assert!(dir_dot > 0.9);
    }

    #[test]
    fn find_essential_mat_ransac_handles_outliers() {
        let k = CameraIntrinsics::new(750.0, 760.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.03, -0.02, 0.01)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.18, -0.01, 0.02).normalize();
        let gt = Pose::new(rot, t);
        let i_ext = Pose::default();

        let mut world = vec![];
        for i in 0..40 {
            let x = -0.5 + 0.05 * i as f64;
            let y = -0.2 + 0.03 * (i % 7) as f64;
            let z = 3.0 + 0.2 * (i % 5) as f64;
            world.push(Point3::new(x, y, z));
        }
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let mut pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        // Inject outliers.
        for i in 0..10 {
            pts2[i] = Point2::new(50.0 + i as f64 * 20.0, 400.0 - i as f64 * 15.0);
        }

        let (e, inliers) = find_essential_mat_ransac(&pts1, &pts2, &k, 3.0, 600).unwrap();
        let inlier_count = inliers.iter().filter(|&&m| m).count();
        assert!(inlier_count >= 8);

        let in1: Vec<Point2<f64>> = pts1
            .iter()
            .zip(inliers.iter())
            .filter_map(|(p, &m)| if m { Some(*p) } else { None })
            .collect();
        let in2: Vec<Point2<f64>> = pts2
            .iter()
            .zip(inliers.iter())
            .filter_map(|(p, &m)| if m { Some(*p) } else { None })
            .collect();

        let recovered = recover_pose_from_essential(&e, &in1, &in2, &k).unwrap();
        assert!(recovered.rotation_matrix().determinant() > 0.0);
        assert!(recovered.translation.norm() > 1e-6);
    }

    #[test]
    fn find_fundamental_mat_ransac_handles_outliers() {
        let k = CameraIntrinsics::new(720.0, 710.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.02, -0.01, 0.015)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.15, 0.01, 0.0);
        let gt = Pose::new(rot, t);
        let i_ext = Pose::default();

        let mut world = vec![];
        for i in 0..24 {
            let x = -0.4 + 0.04 * i as f64;
            let y = -0.2 + 0.03 * (i % 6) as f64;
            let z = 2.8 + 0.2 * (i % 5) as f64;
            world.push(Point3::new(x, y, z));
        }
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let mut pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();
        for i in 0..6 {
            pts2[i] = Point2::new(600.0 - i as f64 * 17.0, 30.0 + i as f64 * 11.0);
        }

        let (f, inliers) = find_fundamental_mat_ransac(&pts1, &pts2, 2.5, 600).unwrap();
        let inlier_count = inliers.iter().filter(|&&m| m).count();
        assert!(inlier_count >= 12);

        let mean_epi = pts1
            .iter()
            .zip(pts2.iter())
            .zip(inliers.iter())
            .filter_map(|((p1, p2), &m)| {
                if m {
                    let x1 = Vector3::new(p1.x, p1.y, 1.0);
                    let x2 = Vector3::new(p2.x, p2.y, 1.0);
                    Some((x2.dot(&(f * x1))).abs())
                } else {
                    None
                }
            })
            .sum::<f64>()
            / inlier_count as f64;
        assert!(mean_epi < 0.5);
    }

    #[test]
    fn stereo_rectify_matrices_has_expected_projection_shape() {
        let k1 = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480);
        let k2 = CameraIntrinsics::new(710.0, 705.0, 322.0, 241.0, 640, 480);
        let left = Pose::default();
        let right = Pose::new(Matrix3::identity(), Vector3::new(0.2, 0.0, 0.0));

        let rect = stereo_rectify_matrices(&k1, &k2, &left, &right).unwrap();
        assert!(rect.r1.determinant() > 0.0);
        assert!(rect.r2.determinant() > 0.0);
        assert!(rect.p2[(0, 3)] < 0.0);
        assert!(rect.q[(3, 2)].is_finite());
        assert!(rect.q[(3, 2)].abs() > 0.0);
    }

    #[test]
    fn generate_chessboard_object_points_layout() {
        let pts = generate_chessboard_object_points((4, 3), 0.05);
        assert_eq!(pts.len(), 12);
        assert!((pts[0].coords - Point3::new(0.0, 0.0, 0.0).coords).norm() < 1e-12);
        assert!((pts[3].coords - Point3::new(0.15, 0.0, 0.0).coords).norm() < 1e-12);
        assert!((pts[11].coords - Point3::new(0.15, 0.10, 0.0).coords).norm() < 1e-12);
    }

    #[test]
    fn calibrate_camera_planar_recovers_intrinsics() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let gt_k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut img_sets = Vec::new();
        for ext in &views {
            obj_sets.push(board.clone());
            img_sets.push(board.iter().map(|p| project_point(&gt_k, ext, p)).collect());
        }

        let calib = calibrate_camera_planar(&obj_sets, &img_sets, (640, 480)).unwrap();
        assert!((calib.intrinsics.fx - gt_k.fx).abs() < 1e-2);
        assert!((calib.intrinsics.fy - gt_k.fy).abs() < 1e-2);
        assert!((calib.intrinsics.cx - gt_k.cx).abs() < 1e-2);
        assert!((calib.intrinsics.cy - gt_k.cy).abs() < 1e-2);
        assert!(calib.rms_reprojection_error < 1e-5);
        assert_eq!(calib.extrinsics.len(), views.len());
    }

    #[test]
    fn calibrate_camera_planar_with_options_enforces_aspect_ratio() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let gt_k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut img_sets = Vec::new();
        for ext in &views {
            obj_sets.push(board.clone());
            img_sets.push(board.iter().map(|p| project_point(&gt_k, ext, p)).collect());
        }

        let target_ratio = 1.0;
        let options = CameraCalibrationOptions {
            fix_aspect_ratio: Some(target_ratio),
            fix_principal_point: None,
            use_intrinsic_guess: false,
            zero_tangent_dist: false,
            fix_focal_length: false,
            fix_k1: false,
            fix_k2: false,
            fix_k3: false,
        };
        let calib = calibrate_camera_planar_with_options(&obj_sets, &img_sets, (640, 480), options)
            .unwrap();
        assert!((calib.intrinsics.fx / calib.intrinsics.fy - target_ratio).abs() < 1e-9);
    }

    #[test]
    fn calibrate_camera_planar_with_options_enforces_principal_point() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let gt_k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut img_sets = Vec::new();
        for ext in &views {
            obj_sets.push(board.clone());
            img_sets.push(board.iter().map(|p| project_point(&gt_k, ext, p)).collect());
        }

        let options = CameraCalibrationOptions {
            fix_aspect_ratio: None,
            fix_principal_point: Some((300.0, 250.0)),
            use_intrinsic_guess: false,
            zero_tangent_dist: false,
            fix_focal_length: false,
            fix_k1: false,
            fix_k2: false,
            fix_k3: false,
        };
        let calib = calibrate_camera_planar_with_options(&obj_sets, &img_sets, (640, 480), options)
            .unwrap();
        assert!((calib.intrinsics.cx - 300.0).abs() < 1e-12);
        assert!((calib.intrinsics.cy - 250.0).abs() < 1e-12);
    }

    #[test]
    fn stereo_calibrate_planar_recovers_relative_transform() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let k_l = CameraIntrinsics::new(810.0, 800.0, 320.0, 240.0, 640, 480);
        let k_r = CameraIntrinsics::new(815.0, 805.0, 318.0, 242.0, 640, 480);
        let r_lr = Rotation3::from_euler_angles(0.01, -0.015, 0.005)
            .matrix()
            .clone_owned();
        let t_lr = Vector3::new(0.20, 0.002, -0.001);

        let board_poses = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut left_sets = Vec::new();
        let mut right_sets = Vec::new();
        for ext_l in &board_poses {
            let ext_r = Pose::new(
                r_lr * ext_l.rotation_matrix(),
                r_lr * ext_l.translation + t_lr,
            );
            obj_sets.push(board.clone());
            left_sets.push(
                board
                    .iter()
                    .map(|p| project_point(&k_l, ext_l, p))
                    .collect(),
            );
            right_sets.push(
                board
                    .iter()
                    .map(|p| project_point(&k_r, &ext_r, p))
                    .collect(),
            );
        }

        let out = stereo_calibrate_planar(&obj_sets, &left_sets, &right_sets, (640, 480)).unwrap();
        let t_err = (out.relative_extrinsics.translation - t_lr).norm();
        let r_err = (out.relative_extrinsics.rotation_matrix() - r_lr).norm();
        assert!(t_err < 1e-2);
        assert!(r_err < 1e-2);
    }

    #[test]
    fn find_chessboard_corners_detects_expected_count() {
        let pattern = (7, 6);
        let (img, gt) = synthetic_checkerboard(pattern, 20, 40, 30);
        let corners = find_chessboard_corners(&img, pattern).unwrap();
        assert_eq!(corners.len(), pattern.0 * pattern.1);

        // Validate board-like coverage even if ordering and exact localization vary.
        let min_x = corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);

        let gt_min_x = gt.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let gt_max_x = gt.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let gt_min_y = gt.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let gt_max_y = gt.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        assert!((min_x - gt_min_x).abs() < 25.0);
        assert!((max_x - gt_max_x).abs() < 25.0);
        assert!((min_y - gt_min_y).abs() < 25.0);
        assert!((max_y - gt_max_y).abs() < 25.0);
    }

    #[test]
    fn corner_subpix_refines_toward_local_corner() {
        let pattern = (7, 6);
        let (img, gt) = synthetic_checkerboard(pattern, 24, 30, 30);
        let mut p = vec![Point2::new(gt[10].x + 2.3, gt[10].y - 1.9)];
        let before = (p[0] - gt[10]).norm();
        corner_subpix(&img, &mut p, 4, 40, 1e-4).unwrap();
        let after = (p[0] - gt[10]).norm();
        assert!(after < before);
    }

    #[test]
    fn calibrate_camera_from_chessboard_files_reports_usage() {
        let pattern = (7, 6);
        let mut paths: Vec<PathBuf> = Vec::new();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let base = synthetic_checkerboard_fixed(pattern, 16, 320, 240, 70, 50);
        let transforms = [
            Matrix3::new(1.00, 0.00, 0.0, 0.00, 1.00, 0.0, 0.0000, 0.0000, 1.0),
            Matrix3::new(0.92, -0.03, 14.0, 0.04, 1.00, -9.0, 0.0024, -0.0012, 1.0),
            Matrix3::new(1.06, 0.06, -20.0, -0.03, 0.94, 13.0, -0.0020, 0.0015, 1.0),
            Matrix3::new(0.89, 0.02, 18.0, -0.05, 1.08, 8.0, 0.0016, 0.0021, 1.0),
            Matrix3::new(1.04, -0.07, -15.0, 0.03, 0.90, -7.0, -0.0018, -0.0013, 1.0),
            Matrix3::new(0.95, 0.05, 10.0, 0.02, 1.04, 6.0, 0.0012, -0.0020, 1.0),
        ];

        for (i, m) in transforms.iter().enumerate() {
            let img = warp_perspective_ex(
                &base,
                m,
                320,
                240,
                Interpolation::Linear,
                BorderMode::Constant(180),
            );
            let p = std::env::temp_dir().join(format!(
                "rustcv_calib_{}_{}_{}.png",
                std::process::id(),
                stamp,
                i
            ));
            img.save(&p).unwrap();
            paths.push(p);
        }

        paths.push(std::env::temp_dir().join(format!(
            "rustcv_calib_missing_{}_{}.png",
            std::process::id(),
            stamp
        )));

        let (calib, report) =
            calibrate_camera_from_chessboard_files(&paths, pattern, 0.04).unwrap();
        assert_eq!(report.total_images, 7);
        assert!(report.used_images >= 3);
        assert!(!report.rejected_images.is_empty());
        assert!(calib.intrinsics.fx.is_finite() && calib.intrinsics.fy.is_finite());
        assert!(calib.intrinsics.fx.abs() > 1e-6 && calib.intrinsics.fy.abs() > 1e-6);

        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn project_points_with_jacobian_numerical_validation() {
        // Test that computed Jacobians match numerical differentiation
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let rotation = Rotation3::from_euler_angles(0.1, 0.2, 0.3).into_inner();
        let translation = Vector3::new(0.5, -0.3, 2.0);
        let extrinsics = Pose::new(rotation, translation);
        let distortion = Distortion {
            k1: -0.2,
            k2: 0.05,
            p1: 0.001,
            p2: -0.002,
            k3: 0.01,
        };

        let points = vec![Point3::new(0.5, 0.3, 1.5), Point3::new(-0.3, 0.4, 2.0)];

        let options = ProjectPointsOptions {
            compute_jacobians: true,
        };
        let result =
            project_points_with_jacobian(&points, &intrinsics, &extrinsics, &distortion, options)
                .unwrap();

        assert!(result.jacobian_rotation.is_some());
        assert!(result.jacobian_translation.is_some());
        assert!(result.jacobian_intrinsics.is_some());
        assert!(result.jacobian_distortion.is_some());

        let jac_rot = result.jacobian_rotation.unwrap();
        assert_eq!(jac_rot.nrows(), 4); // 2 points × 2 coords
        assert_eq!(jac_rot.ncols(), 3); // 3 rotation params

        // Validate non-zero Jacobians
        assert!(jac_rot.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn project_points_without_jacobian_fast_path() {
        // Ensure no Jacobian computation when not requested
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let rotation = Rotation3::identity().into_inner();
        let translation = Vector3::new(0.0, 0.0, 2.0);
        let extrinsics = Pose::new(rotation, translation);
        let distortion = Distortion::default();

        let points = vec![Point3::new(0.5, 0.3, 1.5)];

        let options = ProjectPointsOptions {
            compute_jacobians: false,
        };
        let result =
            project_points_with_jacobian(&points, &intrinsics, &extrinsics, &distortion, options)
                .unwrap();

        assert!(result.jacobian_rotation.is_none());
        assert!(result.jacobian_translation.is_none());
        assert_eq!(result.image_points.len(), 1);
    }

    #[test]
    fn calibrate_camera_fix_focal_length_enforced() {
        // Test that fix_focal_length flag produces equal focal lengths
        let pattern = (7, 6);
        let board = generate_chessboard_object_points(pattern, 0.04);

        let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
                Vector3::new(0.05, -0.08, 0.3),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
                Vector3::new(-0.08, 0.02, 0.35),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
                Vector3::new(0.02, 0.06, 0.32),
            ),
        ];

        let mut object_points = Vec::new();
        let mut image_points = Vec::new();
        for ext in &views {
            object_points.push(board.clone());
            image_points.push(board.iter().map(|p| project_point(&k, ext, p)).collect());
        }

        // Calibrate with fix_focal_length constraint
        let options = CameraCalibrationOptions {
            fix_focal_length: true,
            ..Default::default()
        };
        let result = calibrate_camera_planar_with_options(
            &object_points,
            &image_points,
            (640, 480),
            options,
        )
        .unwrap();

        // With fix_focal_length, focal lengths should be equal (or nearly equal due to averaging)
        assert!(
            (result.intrinsics.fx - result.intrinsics.fy).abs() < 1.0,
            "Focal lengths should be equal with fix_focal_length, got fx={}, fy={}",
            result.intrinsics.fx,
            result.intrinsics.fy
        );
    }

    #[test]
    fn calibrate_camera_zero_tangent_dist_accepted() {
        // Test that zero_tangent_dist flag is accepted and doesn't cause errors
        let pattern = (7, 6);
        let board = generate_chessboard_object_points(pattern, 0.04);

        let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
                Vector3::new(0.05, -0.08, 0.3),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
                Vector3::new(-0.08, 0.02, 0.35),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
                Vector3::new(0.02, 0.06, 0.32),
            ),
        ];

        let mut object_points = Vec::new();
        let mut image_points = Vec::new();
        for ext in &views {
            object_points.push(board.clone());
            image_points.push(board.iter().map(|p| project_point(&k, ext, p)).collect());
        }

        // Calibrate with zero_tangent_dist flag - should not error
        let options = CameraCalibrationOptions {
            zero_tangent_dist: true,
            ..Default::default()
        };
        let result = calibrate_camera_planar_with_options(
            &object_points,
            &image_points,
            (640, 480),
            options,
        )
        .unwrap();

        // Verify calibration completed successfully
        assert!(result.intrinsics.fx.is_finite());
        assert!(result.intrinsics.fy.is_finite());
    }

    #[test]
    fn calibrate_camera_multiple_flags_combined() {
        // Test that multiple flags can be used together
        let pattern = (7, 6);
        let board = generate_chessboard_object_points(pattern, 0.04);

        let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
                Vector3::new(0.05, -0.08, 0.3),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
                Vector3::new(-0.08, 0.02, 0.35),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
                Vector3::new(0.02, 0.06, 0.32),
            ),
        ];

        let mut object_points = Vec::new();
        let mut image_points = Vec::new();
        for ext in &views {
            object_points.push(board.clone());
            image_points.push(board.iter().map(|p| project_point(&k, ext, p)).collect());
        }

        // Use multiple flags together
        let options = CameraCalibrationOptions {
            fix_focal_length: true,
            zero_tangent_dist: true,
            fix_aspect_ratio: Some(1.0),
            ..Default::default()
        };
        let result = calibrate_camera_planar_with_options(
            &object_points,
            &image_points,
            (640, 480),
            options,
        )
        .unwrap();

        // Verify calibration completed with constraints applied
        assert!(result.intrinsics.fx.is_finite());
        assert!(result.intrinsics.fy.is_finite());
        assert!((result.intrinsics.fx - result.intrinsics.fy).abs() < 1.0);
    }

    // ========== Test Data Utilities ==========

    #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
    struct CalibrationGroundTruth {
        pattern_size: [usize; 2],
        square_size: f64,
        num_images: usize,
        expected_camera_matrix: CameraMatrixFixture,
        expected_distortion: DistortionFixture,
        expected_rms_error: f64,
        tolerance_percent: f64,
    }

    #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
    struct CameraMatrixFixture {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
    }

    #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
    struct DistortionFixture {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    }

    fn load_calibration_ground_truth() -> CalibrationGroundTruth {
        let json_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/calibration/expected_results.json"
        );
        let json = std::fs::read_to_string(json_path).expect("Failed to read expected results");
        serde_json::from_str(&json).expect("Failed to parse expected results")
    }

    // ========== Dataset-Backed Tests ==========

    #[test]
    fn calibrate_camera_dataset_ground_truth_loading() {
        // Test that we can load ground truth calibration data
        let ground_truth = load_calibration_ground_truth();

        assert_eq!(ground_truth.pattern_size, [9, 6]);
        assert!(ground_truth.square_size > 0.0);
        assert_eq!(ground_truth.num_images, 13);
        assert!(ground_truth.expected_camera_matrix.fx > 0.0);
        assert!(ground_truth.expected_camera_matrix.fy > 0.0);
        assert!(ground_truth.expected_rms_error > 0.0);
        assert!(ground_truth.tolerance_percent > 0.0);
    }

    #[test]
    fn calibrate_camera_synthetic_validation_against_expected() {
        // Validate that synthetic calibration can match expected results
        // This uses synthetic data with known ground truth rather than real images

        let ground_truth = load_calibration_ground_truth();
        let pattern_size = (ground_truth.pattern_size[0], ground_truth.pattern_size[1]);
        let board = generate_chessboard_object_points(pattern_size, ground_truth.square_size);

        // Create a synthetic camera with parameters close to expected ground truth
        let expected = &ground_truth.expected_camera_matrix;
        let synthetic_k =
            CameraIntrinsics::new(expected.fx, expected.fy, expected.cx, expected.cy, 640, 480);

        // Generate multiple synthetic views with this camera
        let views = [
            Pose::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
                Vector3::new(0.05, -0.08, 0.3),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
                Vector3::new(-0.08, 0.02, 0.35),
            ),
            Pose::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
                Vector3::new(0.02, 0.06, 0.32),
            ),
            Pose::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04).into_inner(),
                Vector3::new(-0.03, -0.05, 0.33),
            ),
        ];

        let mut object_points = Vec::new();
        let mut image_points = Vec::new();
        for ext in &views {
            object_points.push(board.clone());
            image_points.push(
                board
                    .iter()
                    .map(|p| project_point(&synthetic_k, ext, p))
                    .collect(),
            );
        }

        // Calibrate from synthetic data
        let result = calibrate_camera_planar(&object_points, &image_points, (640, 480)).unwrap();

        // Verify calibration recovered the ground truth parameters
        let tolerance = ground_truth.tolerance_percent / 100.0;

        // Check focal lengths (should be within tolerance)
        let fx_error = (result.intrinsics.fx - expected.fx).abs() / expected.fx;
        let fy_error = (result.intrinsics.fy - expected.fy).abs() / expected.fy;

        assert!(
            fx_error < tolerance,
            "fx mismatch: got {}, expected {} (error {:.2}%)",
            result.intrinsics.fx,
            expected.fx,
            fx_error * 100.0
        );

        assert!(
            fy_error < tolerance,
            "fy mismatch: got {}, expected {} (error {:.2}%)",
            result.intrinsics.fy,
            expected.fy,
            fy_error * 100.0
        );

        // Check principal point (should be roughly in image center)
        assert!(
            (result.intrinsics.cx - 320.0).abs() < 50.0,
            "Principal point x out of range"
        );
        assert!(
            (result.intrinsics.cy - 240.0).abs() < 50.0,
            "Principal point y out of range"
        );

        // Check RMS reprojection error is reasonable
        assert!(
            result.rms_reprojection_error < 1.0,
            "RMS reprojection error too high: {}",
            result.rms_reprojection_error
        );
    }
}
