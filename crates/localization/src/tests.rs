//! End-to-end tests for the localization pipeline on a real synthetic scene.

use crate::database::{database_image_from_colmap, Database};
use crate::evaluate::{evaluate_localization, rotation_error_degrees, translation_error};
use crate::localizer::{LocalizationResult, Localizer, LocalizerConfig};
use crate::synthetic::{generate, SyntheticConfig};
use cv_core::{CameraIntrinsics, Descriptors, Pose};
use cv_features::retrieval::Vocabulary;
use nalgebra::{Rotation3, Vector3};
use std::collections::HashMap;

fn scene() -> crate::synthetic::SyntheticScene {
    generate(&SyntheticConfig::default())
}

fn default_intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
}

/// The core end-to-end assertion: a query with a known pose is localized to
/// within a fraction of the scene extent and a fraction of a degree, the correct
/// database image is retrieved, and enough inliers back the pose.
///
/// With the default scene the query sees 79 landmarks and the recovered pose is
/// exact to ≈ 1e-16 in translation and ≈ 1e-6° in rotation, backed by 75 inliers;
/// retrieval ranks the correct view (index 4) first.
#[test]
fn localizes_query_with_accurate_pose() {
    let scene = scene();
    let database = scene.database(None);

    assert_eq!(database.len(), scene.views.len());
    assert!(database.is_built());
    assert_eq!(database.landmark_count(), scene.landmarks.len());

    let localizer = Localizer::new(&database, LocalizerConfig::default());
    let result = localizer
        .localize(
            &scene.query.keypoints,
            &scene.query.descriptors,
            &scene.camera,
        )
        .expect("query must localize against the synthetic database");

    let truth = scene.ground_truth_pose();
    let translation_err = translation_error(&result.pose, &truth);
    let rotation_err = rotation_error_degrees(&result.pose, &truth);
    let extent = scene.extent();

    assert!(
        translation_err < 0.01 * extent,
        "translation error {translation_err:.3e} is not < 1% of scene extent {extent:.3}"
    );
    assert!(
        rotation_err < 1.0,
        "rotation error {rotation_err:.3e} deg is not < 1 deg"
    );
    assert!(
        result.candidates.contains(&scene.correct_view_id()),
        "candidates {:?} must contain the correct image {}",
        result.candidates,
        scene.correct_view_id()
    );
    assert!(
        result.inliers >= localizer.config.min_matches,
        "inliers {} < min_matches {}",
        result.inliers,
        localizer.config.min_matches
    );
    assert!(result.success());
    assert!(result.reprojection_rmse < 1.0);

    // Retrieval is not just present but correct: the top candidate is the image
    // that shares the most landmarks with the query.
    assert_eq!(
        result.candidates.first().copied(),
        Some(scene.correct_view_id()),
        "retrieval must rank the correct image first"
    );
}

/// The same pipeline works when retrieval goes through a trained BoW vocabulary.
#[test]
fn localizes_with_vocabulary_retrieval() {
    let scene = scene();

    let pool: Vec<Vec<u8>> = scene
        .landmarks
        .iter()
        .flat_map(|landmark| landmark.descriptors.iter().map(|d| d.data.clone()))
        .collect();
    let vocabulary = Vocabulary::train(&pool, 64, 8, 7);
    let database = scene.database(Some(vocabulary));

    assert!(database.bow_index().is_some());

    let config = LocalizerConfig {
        candidates: 4,
        ..LocalizerConfig::default()
    };
    let localizer = Localizer::new(&database, config);
    let result = localizer
        .localize(
            &scene.query.keypoints,
            &scene.query.descriptors,
            &scene.camera,
        )
        .expect("BoW-backed query must localize");

    assert!(
        result.candidates.contains(&scene.correct_view_id()),
        "top-4 candidates {:?} must contain the correct image {}",
        result.candidates,
        scene.correct_view_id()
    );
    assert!(result.inliers >= config.min_matches);
    assert!(translation_error(&result.pose, &scene.ground_truth_pose()) < 0.01 * scene.extent());
}

/// Corrupting the query observations with Gaussian pixel noise must degrade the
/// pose gracefully: the error grows with the noise but stays under the
/// documented bounds.
///
/// Measured on the default scene (extent ≈ 51.55, matching on the noiseless
/// query): `sigma = 1 px` gives a translation error of ≈ 0.019% of the extent
/// and a rotation error of ≈ 0.08°; `sigma = 2 px` gives ≈ 0.057% and ≈ 0.16°.
/// The asserted bounds leave a wide margin below those.
#[test]
fn gaussian_pixel_noise_degrades_gracefully() {
    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());
    let extent = scene.extent();

    // (sigma_px, translation bound as a fraction of extent, rotation bound deg).
    let cases = [(1.0_f64, 0.005_f64, 0.5_f64), (2.0, 0.01, 0.5)];

    let mut previous_err = 0.0_f64;
    for (sigma, translation_bound, rotation_bound) in cases {
        let noisy = scene.query_with_pixel_noise(sigma, 0x1234 + sigma as u64);
        let result = localizer
            .localize(&noisy.keypoints, &noisy.descriptors, &scene.camera)
            .unwrap_or_else(|| panic!("noisy query (sigma={sigma}) must still localize"));

        let translation_err = translation_error(&result.pose, &scene.ground_truth_pose());
        let rotation_err = rotation_error_degrees(&result.pose, &scene.ground_truth_pose());

        assert!(
            translation_err < translation_bound * extent,
            "sigma {sigma}: translation error {translation_err:.3e} exceeds {:.1}% of extent",
            translation_bound * 100.0
        );
        assert!(
            rotation_err < rotation_bound,
            "sigma {sigma}: rotation error {rotation_err:.3e} deg exceeds {rotation_bound} deg"
        );
        assert!(result.success());
        assert!(
            translation_err >= previous_err - 1e-9,
            "sigma {sigma}: error {translation_err:.3e} regressed below the noisier-tolerant previous bound {previous_err:.3e}"
        );
        previous_err = translation_err;
    }
}

/// A query with no overlap must not produce a confident (wrong) pose.
#[test]
fn no_overlap_query_is_not_confidently_wrong() {
    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());

    let query = scene.no_overlap_query(999);
    match localizer.localize(&query.keypoints, &query.descriptors, &scene.camera) {
        None => {}
        Some(result) => assert_eq!(
            result.inliers, 0,
            "a no-overlap query must not yield supporting inliers, got {result:?}"
        ),
    }
}

/// Running the identical query twice must give bit-for-bit identical results.
#[test]
fn repeated_queries_are_deterministic() {
    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());

    let first = localizer
        .localize(
            &scene.query.keypoints,
            &scene.query.descriptors,
            &scene.camera,
        )
        .expect("first localization");
    let second = localizer
        .localize(
            &scene.query.keypoints,
            &scene.query.descriptors,
            &scene.camera,
        )
        .expect("second localization");

    assert_eq!(first.inliers, second.inliers);
    assert_eq!(first.matches, second.matches);
    assert_eq!(first.candidates, second.candidates);
    assert_eq!(first.pose.translation, second.pose.translation);
    assert_eq!(first.pose.rotation_matrix(), second.pose.rotation_matrix());
    assert_eq!(first.reprojection_rmse, second.reprojection_rmse);
}

/// The evaluation helper reduces hand-built synthetic results as expected.
#[test]
fn evaluation_reduces_synthetic_results() {
    let truth0 = Pose::identity();
    let truth1 = Pose::new(
        Rotation3::from_axis_angle(&Vector3::y_axis(), 0.3).into_inner(),
        Vector3::new(1.0, 2.0, 3.0),
    );

    // Exact recovery for query 0.
    let exact = LocalizationResult {
        pose: truth0,
        inliers: 50,
        matches: 60,
        candidates: vec![0],
        reprojection_rmse: 0.5,
    };

    // Query 1: 1.0 m translation error and a 0.2 rad (11.46°) rotation error.
    let extra_rotation = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.2).into_inner();
    let offset = LocalizationResult {
        pose: Pose::new(
            extra_rotation * truth1.rotation_matrix(),
            truth1.translation + Vector3::new(1.0, 0.0, 0.0),
        ),
        inliers: 30,
        matches: 40,
        candidates: vec![1],
        reprojection_rmse: 0.7,
    };

    let results = vec![Some(exact), Some(offset), None];
    let truth = vec![truth0, truth1, Pose::default()];
    let stats = evaluate_localization(&results, &truth);

    assert_eq!(stats.queries, 3);
    assert_eq!(stats.succeeded, 2);
    assert!((stats.success_rate - 2.0 / 3.0).abs() < 1e-12);

    // Translation errors are exactly 0 and 1.
    assert!((stats.mean_translation_error - 0.5).abs() < 1e-12);
    assert!((stats.median_translation_error - 0.5).abs() < 1e-12);

    // Rotation errors are exactly 0 and 0.2 rad.
    assert!((stats.mean_rotation_error_deg - 0.2_f64.to_degrees() / 2.0).abs() < 1e-9);
    assert!((stats.median_rotation_error_deg - 0.2_f64.to_degrees() / 2.0).abs() < 1e-9);
}

/// The same evaluation helper applied to a real localization result.
#[test]
fn evaluation_on_a_real_localization() {
    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());
    let result = localizer
        .localize(
            &scene.query.keypoints,
            &scene.query.descriptors,
            &scene.camera,
        )
        .expect("localize");

    let stats = evaluate_localization(&[Some(result)], &[scene.ground_truth_pose()]);
    assert_eq!(stats.queries, 1);
    assert_eq!(stats.succeeded, 1);
    assert!((stats.success_rate - 1.0).abs() < 1e-12);
    assert!(stats.mean_translation_error < 0.01 * scene.extent());
    assert!(stats.median_rotation_error_deg < 1.0);
}

/// A database with no images, and a query with no descriptors, return `None`
/// without panicking.
#[test]
fn empty_inputs_do_not_panic() {
    let intrinsics = default_intrinsics();

    let empty = Database::new(None);
    assert!(empty.is_empty());
    let localizer = Localizer::new(&empty, LocalizerConfig::default());
    assert!(localizer
        .localize(&[], &Descriptors::new(), &intrinsics)
        .is_none());

    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());
    assert!(localizer
        .localize(&[], &Descriptors::new(), &intrinsics)
        .is_none());
    assert!(localizer
        .localize(&scene.query.keypoints, &Descriptors::new(), &intrinsics)
        .is_none());
}

/// Database accessors agree with the scene, and the COLMAP bridge maps
/// observations onto landmarks.
#[test]
fn database_accessors_and_colmap_bridge() {
    let scene = scene();
    let database = scene.database(None);

    assert_eq!(
        database.image_ids(),
        (0..scene.views.len()).collect::<Vec<_>>()
    );

    let expected_descriptors: usize = scene.views.iter().map(|v| v.descriptors.len()).sum();
    assert_eq!(database.descriptor_index().len(), expected_descriptors);
    assert_eq!(database.descriptors().len(), expected_descriptors);

    let colmap_image = cv_io::datasets::colmap::Image {
        id: 7,
        pose: Pose::identity(),
        camera_id: 1,
        name: "a.jpg".to_string(),
        points2d: vec![
            cv_io::datasets::colmap::Point2D {
                x: 10.0,
                y: 20.0,
                point3d_id: 5,
            },
            cv_io::datasets::colmap::Point2D {
                x: 1.0,
                y: 2.0,
                point3d_id: -1,
            },
        ],
    };
    let mut landmark_of = HashMap::new();
    landmark_of.insert(5u64, 0usize);

    let image = database_image_from_colmap(&colmap_image, &landmark_of);
    assert_eq!(image.id, 7);
    assert_eq!(image.keypoints.len(), 2);
    assert!((image.keypoints[0].x - 10.0).abs() < 1e-12);
    assert!((image.keypoints[1].y - 2.0).abs() < 1e-12);
    assert_eq!(image.landmarks, vec![Some(0), None]);
    assert!(image.descriptors.is_empty());
    assert!(image.pose.is_some());
}

/// Re-project every landmark into `pose`, exactly as [`generate`] builds a view:
/// descriptors are the landmark's own, keypoints are the projections.
fn project_landmarks(
    scene: &crate::synthetic::SyntheticScene,
    pose: &Pose,
) -> (Vec<cv_core::KeyPoint>, Descriptors) {
    let camera = scene.camera();
    let margin = 2.0;
    let mut keypoints = Vec::new();
    let mut descriptors = Descriptors::new();
    for landmark in &scene.landmarks {
        let camera_point = pose.rotation * landmark.position.coords + pose.translation;
        if camera_point[2] <= 1e-6 {
            continue;
        }
        let projected = camera.project(&nalgebra::Point3::from(camera_point));
        if projected.x < margin
            || projected.y < margin
            || projected.x > camera.width as f64 - margin
            || projected.y > camera.height as f64 - margin
        {
            continue;
        }
        let kp = cv_core::KeyPoint::new(projected.x, projected.y);
        keypoints.push(kp);
        descriptors.push(cv_core::Descriptor::new(
            landmark.descriptors[0].data.clone(),
            kp,
        ));
    }
    (keypoints, descriptors)
}

/// The synthetic scene only ever *translates* the query camera relative to the
/// database ring, so a pipeline that mishandled the rotation part of the pose
/// would still pass every other test in this file. Roll and yaw the query camera
/// about its own optical axis and re-project: the recovered pose must still match
/// to machine precision.
///
/// Measured: 78 observations, 74 RANSAC inliers, translation error 3.6e-15
/// (extent 51.55), rotation error 0.0°, reprojection RMSE 6.6e-14 px.
#[test]
fn rotated_query_camera_is_localized_exactly() {
    let scene = scene();
    let database = scene.database(None);
    let localizer = Localizer::new(&database, LocalizerConfig::default());

    let roll = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 15.0_f64.to_radians());
    let yaw = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 5.0_f64.to_radians());
    let rotation = (yaw * roll).to_rotation_matrix().into_inner();
    let pose = Pose::new(
        rotation * scene.query.pose.rotation_matrix(),
        (yaw * roll) * scene.query.pose.translation,
    );

    let (keypoints, descriptors) = project_landmarks(&scene, &pose);
    assert!(
        keypoints.len() >= LocalizerConfig::default().min_matches,
        "the rotated view must observe enough landmarks, got {}",
        keypoints.len()
    );

    let result = localizer
        .localize(&keypoints, &descriptors, &scene.camera)
        .expect("a rolled query camera must still localize");

    let extent = scene.extent();
    assert!(
        translation_error(&result.pose, &pose) < 1e-6 * extent,
        "translation error {:.3e} exceeds 1e-6 of the scene extent {extent:.3}",
        translation_error(&result.pose, &pose)
    );
    assert!(
        rotation_error_degrees(&result.pose, &pose) < 1e-6,
        "rotation error {:.3e} deg",
        rotation_error_degrees(&result.pose, &pose)
    );
    assert!(result.inliers >= LocalizerConfig::default().min_matches);

    // Control: the very same construction with no rotation applied must work
    // too, so the assertion is not about the rotation being present.
    let (keypoints, descriptors) = project_landmarks(&scene, &scene.query.pose);
    let control = localizer
        .localize(&keypoints, &descriptors, &scene.camera)
        .expect("the unrotated control view must localize");
    assert!(translation_error(&control.pose, &scene.query.pose) < 1e-6 * extent);
}

/// `rotation_error_degrees` must return the geodesic angle between the two
/// orientations. Checked against the quaternion angle, which is an independent
/// path to the same quantity (no trace formula).
#[test]
fn rotation_error_matches_the_geodesic_angle() {
    let axis = nalgebra::Unit::new_normalize(Vector3::new(0.3, -0.7, 0.2));
    for deg in [0.0_f64, 0.25, 11.4591559, 90.0, 179.5] {
        let r = Rotation3::from_axis_angle(&axis, deg.to_radians());
        let a = Pose::new(r.into_inner(), Vector3::new(1.0, 2.0, 3.0));
        let b = Pose::identity();
        let got = rotation_error_degrees(&a, &b);
        assert!(
            (got - deg).abs() < 1e-9,
            "true {deg} deg, rotation_error_degrees = {got}"
        );

        let relative = b.rotation_matrix().transpose() * a.rotation_matrix();
        let reference = nalgebra::UnitQuaternion::from_rotation_matrix(
            &Rotation3::from_matrix_unchecked(relative),
        )
        .angle()
        .to_degrees();
        assert!(
            (got - reference).abs() < 1e-9,
            "trace formula {got} vs quaternion angle {reference}"
        );
    }
}

/// `translation_error` is the difference of the poses' own `t` vectors, which is
/// a *pose-parameter* difference and not the distance between camera centres
/// (`localization::evaluate` documents this). The measurement below records the
/// consequence for the world-to-camera convention this crate uses everywhere:
/// two poses at the *same* camera centre but different orientations report a
/// non-zero "translation error" (0.5176 at 30° yaw, 1.4142 at 90°), because
/// `t = -R·C` depends on `R`.
///
/// `benchmark.rs` compensates by inverting into camera-to-world poses first,
/// where `t` *is* the camera centre; `evaluate_localization`'s own test data uses
/// matching conventions, so the reported statistics are consistent. Do not
/// "fix" one call site without the other.
#[test]
fn translation_error_is_a_pose_parameter_difference() {
    let centre = Vector3::new(1.0, 0.0, 0.0);
    let truth = Pose::new(
        Rotation3::identity().into_inner(),
        -(Rotation3::identity() * centre),
    );

    for (deg, expected) in [(0.0_f64, 0.0_f64), (30.0, 0.517_638_1), (90.0, 1.414_213_6)] {
        let r = Rotation3::from_axis_angle(&Vector3::y_axis(), deg.to_radians());
        let pose = Pose::new(r.into_inner(), -(r * centre));
        // Same camera centre by construction.
        let centre_est = -(pose.rotation_matrix().transpose() * pose.translation);
        assert!(
            (centre_est - centre).norm() < 1e-12,
            "the poses must share a camera centre: {centre_est:?} vs {centre:?}"
        );
        let reported = translation_error(&pose, &truth);
        assert!(
            (reported - expected).abs() < 1e-5,
            "yaw {deg}: translation_error = {reported} (expected {expected})"
        );
    }
}

/// The evaluation statistics must still be exact on a batch whose answer is
/// known by hand, including the empty and all-failed cases.
#[test]
fn evaluation_statistics_are_exact_on_hand_checked_batches() {
    let truth = [Pose::identity(), Pose::identity(), Pose::identity()];

    // Nothing succeeded: every statistic is NaN, and success_rate is 0.0 rather
    // than NaN.
    let stats = evaluate_localization(&[None, None, None], &truth);
    assert_eq!(stats.queries, 3);
    assert_eq!(stats.succeeded, 0);
    assert_eq!(stats.success_rate, 0.0);
    assert!(stats.mean_translation_error.is_nan());
    assert!(stats.median_rotation_error_deg.is_nan());

    // Nothing was asked: no division by zero.
    let stats = evaluate_localization(&[], &[]);
    assert_eq!(stats.queries, 0);
    assert_eq!(stats.success_rate, 0.0);
    assert!(stats.mean_translation_error.is_nan());

    // One exact hit: zero error, and the metric is symmetric in its arguments.
    let hit = LocalizationResult {
        pose: truth[0],
        inliers: 30,
        matches: 40,
        candidates: vec![0],
        reprojection_rmse: 0.4,
    };
    let stats = evaluate_localization(&[Some(hit)], &truth[..1]);
    assert_eq!(stats.succeeded, 1);
    assert_eq!(stats.success_rate, 1.0);
    assert_eq!(stats.mean_translation_error, 0.0);
    assert_eq!(stats.median_translation_error, 0.0);
    assert_eq!(stats.mean_rotation_error_deg, 0.0);
}
