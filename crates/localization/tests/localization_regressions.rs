//! Regression tests for the `cv-localization` public API.
//!
//! These cover the boundaries the crate documents: metric behaviour on empty
//! and degenerate batches, non-finite inputs to the localizer, and the
//! invalidation contract of the built retrieval indices. They run under the
//! default feature set; `benchmark_run.rs` covers the `synthetic`-gated
//! benchmark module.

use cv_core::{CameraIntrinsics, Descriptor, Descriptors, KeyPoint, Pose};
use cv_localization::{
    Database, DatabaseImage, Landmark, LocalizationResult, Localizer, LocalizerConfig, Vocabulary,
    evaluate_localization, rotation_error_degrees, translation_error,
};
use nalgebra::Point3;
use std::time::{Duration, Instant};

/// Intrinsics shared by the hand-built scenes.
fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
}

/// A deterministic, deliberately **non-coplanar** scene: `n` landmarks in front
/// of an identity-pose camera with exact projections and unique descriptors.
///
/// Non-coplanarity matters: a coplanar target has a genuine two-fold pose
/// ambiguity, and `cv_calib3d::pnp::solve_pnp_dlt`'s homography path can return
/// the mirrored pose (all points behind the camera) for exact planar data, which
/// then yields zero RANSAC inliers. `z` below is not an affine function of
/// `(x, y)`.
fn tiny_scene(n: usize) -> (Database, Vec<KeyPoint>, Descriptors) {
    let camera = intrinsics();
    let mut db = Database::new(None);
    let mut keypoints = Vec::new();
    let mut descriptors = Descriptors::new();
    let mut landmark_of = Vec::new();

    for i in 0..n {
        let x = -1.0 + 0.4 * (i % 5) as f64;
        let y = -1.0 + 0.3 * ((i / 5) % 4) as f64;
        let z = 3.0 + 0.17 * (i as f64).sin() + 0.4 * (i % 3) as f64;
        db.add_landmark(Landmark {
            position: Point3::new(x, y, z),
            descriptors: vec![],
        });
        let kp = KeyPoint::new(camera.cx + camera.fx * x / z, camera.cy + camera.fy * y / z);
        let mut bytes = vec![0u8; 32];
        bytes[0] = i as u8;
        bytes[1] = (i * 31) as u8;
        bytes[2] = (i * 17 + 3) as u8;
        descriptors.push(Descriptor::new(bytes, kp));
        keypoints.push(kp);
        landmark_of.push(Some(i));
    }

    db.add_image(DatabaseImage {
        id: 0,
        pose: Some(Pose::identity()),
        keypoints: keypoints.clone(),
        descriptors: descriptors.clone(),
        landmarks: landmark_of,
    });
    db.build();
    (db, keypoints, descriptors)
}

/// The localizer must localize its own control scene: without this the guard
/// tests below could pass for the wrong reason.
#[test]
fn tiny_scene_control_localizes_with_a_finite_pose() {
    let (db, keypoints, descriptors) = tiny_scene(20);
    let localizer = Localizer::new(&db, LocalizerConfig::default());
    let result = localizer
        .localize(&keypoints, &descriptors, &intrinsics())
        .expect("the control scene must localize");
    assert!(result.inliers >= LocalizerConfig::default().min_matches);
    assert!(result.pose.translation.iter().all(|v| v.is_finite()));
    // The pose is world-to-camera for a camera at the origin: ~identity.
    assert!(result.pose.translation.norm() < 1e-9);
}

/// A non-finite intrinsic parameter used to make `Localizer::localize` never
/// return: a NaN focal length survives `try_inverse_matrix` and makes the DLT
/// design matrix non-finite, and nalgebra's `svd(true, true)` does not terminate
/// on non-finite input. The documented contract is `None`, never a panic (and
/// certainly never a hang), so the guard is at the caller boundary.
///
/// Before the fix this test did not fail - it hung; the audit measured
/// `timeout 45 cargo test --test ...` killing it with "entering localize with
/// NaN fx ..." as the last output.
#[test]
fn non_finite_intrinsics_return_none_without_hanging() {
    let (db, keypoints, descriptors) = tiny_scene(20);
    let localizer = Localizer::new(&db, LocalizerConfig::default());

    let bad: [(&str, CameraIntrinsics); 5] = [
        (
            "fx = NaN",
            CameraIntrinsics::new(f64::NAN, 500.0, 320.0, 240.0, 640, 480),
        ),
        (
            "fy = NaN",
            CameraIntrinsics::new(500.0, f64::NAN, 320.0, 240.0, 640, 480),
        ),
        (
            "cx = NaN",
            CameraIntrinsics::new(500.0, 500.0, f64::NAN, 240.0, 640, 480),
        ),
        (
            "cy = NaN",
            CameraIntrinsics::new(500.0, 500.0, 320.0, f64::NAN, 640, 480),
        ),
        (
            "fx = inf",
            CameraIntrinsics::new(f64::INFINITY, 500.0, 320.0, 240.0, 640, 480),
        ),
    ];

    for (name, bad_intrinsics) in bad {
        let started = Instant::now();
        let result = localizer.localize(&keypoints, &descriptors, &bad_intrinsics);
        assert!(result.is_none(), "{name}: expected None");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{name}: localize took {:?}, which means it entered the solver",
            started.elapsed()
        );
    }

    // Control: the identical call with finite intrinsics still succeeds.
    assert!(
        localizer
            .localize(&keypoints, &descriptors, &intrinsics())
            .is_some()
    );
}

/// Non-finite query data and landmarks must fail closed too (these already did:
/// the correspondence guard in `cv_calib3d::pnp` catches them, and this test
/// pins that the crate's boundary does not turn them into a pose).
#[test]
fn non_finite_correspondences_return_none() {
    let (db, keypoints, descriptors) = tiny_scene(20);
    let localizer = Localizer::new(&db, LocalizerConfig::default());

    let mut bad_keypoints = keypoints.clone();
    bad_keypoints[2].x = f64::NAN;
    assert!(
        localizer
            .localize(&bad_keypoints, &descriptors, &intrinsics())
            .is_none()
    );

    // A NaN landmark position in the database.
    let mut landmarks: Vec<Landmark> = db.landmarks().to_vec();
    landmarks[3].position = Point3::new(f64::NAN, 0.0, 1.0);
    let mut db_bad = Database::new(None);
    for landmark in landmarks {
        db_bad.add_landmark(landmark);
    }
    db_bad.add_image(db.images()[0].clone());
    db_bad.build();
    let localizer = Localizer::new(&db_bad, LocalizerConfig::default());
    assert!(
        localizer
            .localize(&keypoints, &descriptors, &intrinsics())
            .is_none()
    );
}

/// Success rate is `succeeded / queries`: a real 0.0 when queries were asked and
/// all failed, 1.0 when the only query succeeded, and NaN - not 0.0 - when
/// nothing was asked, so an empty batch cannot read as "everything failed".
#[test]
fn success_rate_has_no_value_for_an_empty_batch() {
    let exact = LocalizationResult {
        pose: Pose::identity(),
        inliers: 30,
        matches: 40,
        candidates: vec![0],
        reprojection_rmse: 0.4,
    };
    let truth = [Pose::identity(), Pose::identity(), Pose::identity()];

    // Zero queries: 0/0.
    let stats = evaluate_localization(&[], &[]);
    assert_eq!(stats.queries, 0);
    assert!(
        stats.success_rate.is_nan(),
        "empty batch reported success_rate {}",
        stats.success_rate
    );
    assert!(stats.mean_translation_error.is_nan());

    // Three queries, none successful: a measured 0.0.
    let stats = evaluate_localization(&[None, None, None], &truth);
    assert_eq!((stats.queries, stats.succeeded), (3, 0));
    assert_eq!(stats.success_rate, 0.0);

    // One query, successful: exactly 1.0 and exactly zero error.
    let stats = evaluate_localization(&[Some(exact.clone())], &truth[..1]);
    assert_eq!((stats.queries, stats.succeeded), (1, 1));
    assert_eq!(stats.success_rate, 1.0);
    assert_eq!(stats.mean_translation_error, 0.0);
    assert_eq!(stats.mean_rotation_error_deg, 0.0);

    // Three queries, two successful: the denominator is the query count.
    let stats = evaluate_localization(&[Some(exact.clone()), None, Some(exact)], &truth);
    assert_eq!((stats.queries, stats.succeeded), (3, 2));
    assert!((stats.success_rate - 2.0 / 3.0).abs() < 1e-15);
}

/// `Some` is not enough: a result with zero supporting inliers is a failure by
/// [`LocalizationResult::success`], and `Localizer::localize` can never produce
/// one (RANSAC needs at least one inlier, and the acceptance floor is
/// `min_matches`). Stored results can - and their unsupported pose must not be
/// counted as a success nor folded into the error statistics.
#[test]
fn zero_inlier_results_are_not_counted_as_successes() {
    let failed = LocalizationResult {
        pose: Pose::identity(),
        inliers: 0,
        matches: 0,
        candidates: vec![],
        reprojection_rmse: f64::INFINITY,
    };
    assert!(!failed.success());

    let stats = evaluate_localization(&[Some(failed.clone())], &[Pose::identity()]);
    assert_eq!(
        (stats.queries, stats.succeeded, stats.success_rate),
        (1, 0, 0.0),
        "a zero-inlier result must not be a success"
    );
    assert!(stats.mean_translation_error.is_nan());

    // Control: one inlier is a success, and the metric then reads 1.0.
    let supported = LocalizationResult {
        inliers: 1,
        matches: 1,
        ..failed
    };
    let stats = evaluate_localization(&[Some(supported)], &[Pose::identity()]);
    assert_eq!((stats.queries, stats.succeeded), (1, 1));
    assert_eq!(stats.success_rate, 1.0);
    assert_eq!(stats.mean_translation_error, 0.0);
}

/// A pose is at distance zero from itself (within the acos noise floor of the
/// trace formula), and the metric is symmetric.
#[test]
fn rotation_and_translation_error_are_zero_on_identical_poses() {
    let pose = Pose::new(
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.3).into_inner(),
        nalgebra::Vector3::new(1.0, 2.0, 3.0),
    );
    assert_eq!(translation_error(&pose, &pose), 0.0);
    let self_rotation = rotation_error_degrees(&pose, &pose);
    assert!(
        self_rotation < 1e-4,
        "self rotation error {self_rotation} deg"
    );

    let other = Pose::new(
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::x_axis(), 1.0).into_inner(),
        nalgebra::Vector3::zeros(),
    );
    assert_eq!(
        rotation_error_degrees(&pose, &other),
        rotation_error_degrees(&other, &pose)
    );
}

/// `add_image`/`add_landmark` must really invalidate the built state: the
/// accessors that document themselves as "empty until build" / "present only
/// after a build" must not keep serving a stale index.
#[test]
fn modification_invalidates_every_built_index() {
    let (scene_db, keypoints, descriptors) = tiny_scene(20);
    let pool: Vec<Vec<u8>> = descriptors.iter().map(|d| d.data.clone()).collect();
    let vocabulary = Vocabulary::train(&pool, 4, 8, 7);

    let mut db = Database::new(Some(vocabulary));
    for landmark in scene_db.landmarks() {
        db.add_landmark(landmark.clone());
    }
    for image in scene_db.images() {
        db.add_image(image.clone());
    }
    db.build();

    assert!(db.is_built());
    assert_eq!(db.descriptor_index().len(), 20);
    assert_eq!(db.descriptors().len(), 20);
    assert!(db.lsh_index().is_some());
    assert!(db.bow_index().is_some());

    let extra_kp = KeyPoint::new(10.0, 10.0);
    let mut extra_descriptors = Descriptors::new();
    extra_descriptors.push(Descriptor::new(vec![9u8; 32], extra_kp));
    db.add_image(DatabaseImage {
        id: 1,
        pose: Some(Pose::identity()),
        keypoints: vec![extra_kp],
        descriptors: extra_descriptors,
        landmarks: vec![Some(0)],
    });

    assert!(!db.is_built());
    assert!(
        db.descriptor_index().is_empty(),
        "descriptor_index() kept {} stale entries after add_image",
        db.descriptor_index().len()
    );
    assert!(db.descriptors().is_empty());
    assert!(db.lsh_index().is_none());
    assert!(db.bow_index().is_none());

    // A rebuild restores a table that matches the 20 + 1 keypoints and still
    // localizes the original query (control).
    db.build();
    assert!(db.is_built());
    assert_eq!(db.descriptor_index().len(), 21);
    assert_eq!(db.descriptors().len(), 21);
    assert!(db.lsh_index().is_some());
    let localizer = Localizer::new(&db, LocalizerConfig::default());
    assert!(
        localizer
            .localize(&keypoints, &descriptors, &intrinsics())
            .is_some()
    );

    // add_landmark invalidates as well.
    db.add_landmark(Landmark {
        position: Point3::new(0.0, 0.0, 0.0),
        descriptors: vec![],
    });
    assert!(!db.is_built());
    assert!(db.descriptor_index().is_empty());
    assert!(db.lsh_index().is_none());
}

/// `rank_by_descriptor_matches` returns an empty vector for the degenerate
/// cases and ranks an exact query first (control).
#[test]
fn rank_by_descriptor_matches_degenerate_cases_are_empty() {
    let (db, _, descriptors) = tiny_scene(20);

    assert!(db.rank_by_descriptor_matches(&descriptors, 0).is_empty());
    assert!(
        db.rank_by_descriptor_matches(&Descriptors::new(), 5)
            .is_empty()
    );
    assert!(
        Database::new(None)
            .rank_by_descriptor_matches(&descriptors, 5)
            .is_empty()
    );
    assert_eq!(db.rank_by_descriptor_matches(&descriptors, 5), vec![0]);
}

/// A localization must never return a pose that is not a pose.
///
/// Exactly-projectable data at some scene scales produced a *successful* result
/// — `inliers: 20`, `matches: 20` — whose rotation matrix was entirely `NaN`
/// and whose reprojection RMSE was `NaN`. Twenty inliers that fit the data
/// perfectly is the strongest possible claim, attached to a transform that
/// cannot map a single point.
///
/// The mechanism is in the refinement, not in the matching. `Localizer` refines
/// the RANSAC pose with `cv_calib3d::pnp::solve_pnp_refine`, which parameterises
/// rotation as `Rotation3::scaled_axis`. For a rotation matrix whose trace rounds
/// *above* 3.0, `scaled_axis` is `NaN` — `acos` of a value outside `[-1, 1]` —
/// and the refinement starts from that parameterisation, so it returns an
/// all-NaN rotation. `refine_pose` used to adopt whatever the refinement
/// returned, and `localize` returned it unchanged.
///
/// The scene below is a perspective grid seen by an identity-pose camera; the
/// landmark coordinates are uniformly scaled. Because the residual is exactly
/// zero, the RANSAC pose is already the right answer and the refinement can only
/// make it worse. Measured before the fix, over the scale sweep 0.25..=5.25 in
/// steps of 0.25: scales 0.75, 1.5 and 3 returned `rot_finite = false` with
/// `reprojection_rmse = NaN` and 20 inliers, while every other scale in the same
/// sweep returned the exact pose. That the same scene is correct at 1.0 and
/// poisoned at 0.75 is the tell: the defect is a rounding accident in the trace,
/// not a property of the input.
#[test]
fn a_localized_pose_is_always_finite() {
    let camera = intrinsics();
    let extent = 51.55;
    let mut refused = 0usize;

    for step in 0..=20 {
        let scale = 0.25 + 0.25 * step as f64;
        let (db, keypoints, descriptors) = scaled_tiny_scene(scale, 20);
        let localizer = Localizer::new(&db, LocalizerConfig::default());

        let Some(result) = localizer.localize(&keypoints, &descriptors, &camera) else {
            // Refusing is allowed; returning a fabricated pose is not. The
            // control below pins that refusal is not the normal answer.
            refused += 1;
            continue;
        };

        let rotation = result.pose.rotation_matrix();
        assert!(
            rotation.iter().all(|component| component.is_finite()),
            "scale {scale}: returned a non-finite rotation {rotation:?} with {} inliers",
            result.inliers
        );
        assert!(
            result.pose.translation.iter().all(|v| v.is_finite()),
            "scale {scale}: returned a non-finite translation {:?}",
            result.pose.translation
        );
        assert!(
            result.reprojection_rmse.is_finite(),
            "scale {scale}: finite pose but a non-finite reprojection RMSE {}",
            result.reprojection_rmse
        );
        // A rotation matrix, not merely finite numbers: orthonormal, det +1.
        let orthonormality = (rotation * rotation.transpose() - nalgebra::Matrix3::identity())
            .iter()
            .copied()
            .fold(0.0_f64, |worst, v| worst.max(v.abs()));
        assert!(
            orthonormality < 1e-9,
            "scale {scale}: returned rotation is not orthonormal (worst deviation {orthonormality:.3e})"
        );
        assert!(
            (rotation.determinant() - 1.0).abs() < 1e-9,
            "scale {scale}: returned rotation has determinant {}",
            rotation.determinant()
        );

        // The scene is exactly projectable from the identity pose, so a
        // returned pose must be that pose: a confidently wrong answer is no
        // better than a NaN one.
        let truth = Pose::identity();
        assert!(
            translation_error(&result.pose, &truth) < 1e-6 * extent,
            "scale {scale}: translation error {:.3e}",
            translation_error(&result.pose, &truth)
        );
        assert!(
            rotation_error_degrees(&result.pose, &truth) < 1e-6,
            "scale {scale}: rotation error {:.3e} deg",
            rotation_error_degrees(&result.pose, &truth)
        );
    }

    assert_eq!(
        refused, 0,
        "the guard must not turn the whole sweep into refusals; a clean scene \
         localizes"
    );

    // Control: the un-scaled scene (`scale = 1`) still localizes exactly, so the
    // assertions above cannot pass by rejecting everything.
    let (db, keypoints, descriptors) = scaled_tiny_scene(1.0, 20);
    let localizer = Localizer::new(&db, LocalizerConfig::default());
    let result = localizer
        .localize(&keypoints, &descriptors, &camera)
        .expect("the scale-1 control scene must localize");
    assert!(result.pose.rotation_matrix().iter().all(|v| v.is_finite()));
    assert!(result.reprojection_rmse < 1e-6);
}

/// [`tiny_scene`] with every landmark position multiplied by `scale`.
///
/// Scaling the world is a physical no-op — the camera, its intrinsics and the
/// projections are unchanged — so every scale must localize to the same pose.
fn scaled_tiny_scene(scale: f64, n: usize) -> (Database, Vec<KeyPoint>, Descriptors) {
    let camera = intrinsics();
    let mut db = Database::new(None);
    let mut keypoints = Vec::new();
    let mut descriptors = Descriptors::new();
    let mut landmark_of = Vec::new();

    for i in 0..n {
        let x = -1.0 + 0.4 * (i % 5) as f64;
        let y = -1.0 + 0.3 * ((i / 5) % 4) as f64;
        let z = 3.0 + 0.17 * (i as f64).sin() + 0.4 * (i % 3) as f64;
        db.add_landmark(Landmark {
            position: Point3::new(x * scale, y * scale, z * scale),
            descriptors: vec![],
        });
        let kp = KeyPoint::new(camera.cx + camera.fx * x / z, camera.cy + camera.fy * y / z);
        let mut bytes = vec![0u8; 32];
        bytes[0] = i as u8;
        bytes[1] = (i * 31) as u8;
        bytes[2] = (i * 17 + 3) as u8;
        descriptors.push(Descriptor::new(bytes, kp));
        keypoints.push(kp);
        landmark_of.push(Some(i));
    }

    db.add_image(DatabaseImage {
        id: 0,
        pose: Some(Pose::identity()),
        keypoints: keypoints.clone(),
        descriptors: descriptors.clone(),
        landmarks: landmark_of,
    });
    db.build();
    (db, keypoints, descriptors)
}
