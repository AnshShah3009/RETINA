//! Query-to-pose estimation on top of a [`Database`].

use crate::database::{Database, DatabaseImage};
use cv_calib3d::pnp::{solve_pnp_ransac, solve_pnp_refine};
use cv_core::{CameraIntrinsics, Descriptors, KeyPoint, Matches, Pose};
use cv_features::matcher::match_descriptors;
use cv_features::retrieval::{descriptors_to_bytes, BowVector};
use nalgebra::{Point2, Point3};

/// Tunable knobs for the localization pipeline.
#[derive(Debug, Clone, Copy)]
pub struct LocalizerConfig {
    /// Number of database images to consider, best retrieval score first.
    pub candidates: usize,
    /// Lowe ratio-test threshold applied to descriptor matches.
    pub ratio: f32,
    /// Minimum number of 2D–3D correspondences (and RANSAC inliers) required
    /// to accept a pose.
    pub min_matches: usize,
    /// Maximum RANSAC iterations for the PnP solver.
    pub ransac_iterations: usize,
    /// RANSAC inlier reprojection threshold in pixels.
    pub reprojection_threshold: f64,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            candidates: 10,
            ratio: 0.75,
            min_matches: 12,
            ransac_iterations: 1000,
            reprojection_threshold: 4.0,
        }
    }
}

/// The output of a successful [`Localizer::localize`] call.
#[derive(Debug, Clone)]
pub struct LocalizationResult {
    /// Recovered world-to-camera pose (`x_cam = R * X_world + t`).
    pub pose: Pose,
    /// Number of RANSAC inliers supporting the pose.
    pub inliers: usize,
    /// Number of 2D–3D correspondences that were fed to the solver.
    pub matches: usize,
    /// Candidate image ids considered, best retrieval score first.
    pub candidates: Vec<usize>,
    /// RMS reprojection error of the inliers, in pixels.
    pub reprojection_rmse: f64,
}

impl LocalizationResult {
    /// Whether the result carries at least one supporting inlier.
    ///
    /// [`Localizer::localize`] only ever returns a result with
    /// `inliers >= config.min_matches`, so a returned result is always
    /// successful by this definition; the predicate exists so callers can treat
    /// results uniformly whether they came from the localizer or from stored
    /// data.
    pub fn success(&self) -> bool {
        self.inliers > 0
    }
}

/// Estimates query poses against a landmark [`Database`].
pub struct Localizer<'a> {
    /// The database to localize against. Must have been built for BoW retrieval.
    pub database: &'a Database,
    /// Pipeline configuration.
    pub config: LocalizerConfig,
}

impl<'a> Localizer<'a> {
    /// Create a localizer with an explicit configuration.
    pub fn new(database: &'a Database, config: LocalizerConfig) -> Self {
        Self { database, config }
    }

    /// Localize a query image and return its camera pose.
    ///
    /// The pipeline is: retrieve candidate database images → ratio-test match
    /// the query descriptors to the best candidate → lift matches to 2D–3D
    /// correspondences via the candidate's landmark indices → PnP + RANSAC →
    /// refine on the inliers.
    ///
    /// Returns `None` (never panics) when the query or database is empty, no
    /// candidate yields enough correspondences (`< max(config.min_matches, 6)`),
    /// or the solver fails / produces too few inliers.
    pub fn localize(
        &self,
        query_keypoints: &[KeyPoint],
        query_descriptors: &Descriptors,
        intrinsics: &CameraIntrinsics,
    ) -> Option<LocalizationResult> {
        if query_descriptors.is_empty() || self.database.is_empty() {
            return None;
        }

        let candidates = self.retrieve_candidates(query_descriptors, self.config.candidates);
        if candidates.is_empty() {
            return None;
        }

        let min_correspondences = self.config.min_matches.max(6);

        for &image_id in &candidates {
            let Some(image) = self.database.image(image_id) else {
                continue;
            };
            if image.descriptors.is_empty() {
                continue;
            }

            let matches = match_descriptors(
                query_descriptors,
                &image.descriptors,
                Some(self.config.ratio),
            );
            let (object_points, image_points) =
                self.correspondences(&matches, image, query_keypoints);
            if object_points.len() < min_correspondences {
                continue;
            }

            let Ok((pose, inlier_mask)) = solve_pnp_ransac(
                &object_points,
                &image_points,
                intrinsics,
                None,
                self.config.reprojection_threshold,
                self.config.ransac_iterations,
            ) else {
                continue;
            };
            let inliers = inlier_mask.iter().filter(|&&inlier| inlier).count();
            if inliers < self.config.min_matches {
                continue;
            }

            let pose = refine_pose(
                pose,
                &object_points,
                &image_points,
                &inlier_mask,
                intrinsics,
            );
            let reprojection_rmse = reprojection_rmse(
                &pose,
                &object_points,
                &image_points,
                &inlier_mask,
                intrinsics,
            );

            return Some(LocalizationResult {
                pose,
                inliers,
                matches: object_points.len(),
                candidates: candidates.clone(),
                reprojection_rmse,
            });
        }

        None
    }

    /// Retrieve up to `k` candidate image ids, best first.
    fn retrieve_candidates(&self, query: &Descriptors, k: usize) -> Vec<usize> {
        if k == 0 || self.database.is_empty() {
            return Vec::new();
        }

        if let (Some(bow), Some(vocabulary)) =
            (self.database.bow_index(), self.database.vocabulary())
        {
            let bytes = descriptors_to_bytes(query);
            let bow_query = BowVector::from_descriptors(vocabulary, &bytes);
            let hits = bow.query(&bow_query, k);
            if !hits.is_empty() {
                return hits.into_iter().map(|(id, _)| id).collect();
            }
        }

        self.database.rank_by_descriptor_matches(query, k)
    }

    /// Lift descriptor matches to (object point, image point) correspondences.
    fn correspondences(
        &self,
        matches: &Matches,
        image: &DatabaseImage,
        query_keypoints: &[KeyPoint],
    ) -> (Vec<Point3<f64>>, Vec<Point2<f64>>) {
        let mut object_points = Vec::with_capacity(matches.matches.len());
        let mut image_points = Vec::with_capacity(matches.matches.len());

        for m in &matches.matches {
            let Some(query_idx) = usize::try_from(m.query_idx).ok() else {
                continue;
            };
            let Some(train_idx) = usize::try_from(m.train_idx).ok() else {
                continue;
            };
            let Some(query_kp) = query_keypoints.get(query_idx) else {
                continue;
            };
            let Some(Some(landmark_idx)) = image.landmarks.get(train_idx) else {
                continue;
            };
            let Some(landmark) = self.database.landmark(*landmark_idx) else {
                continue;
            };

            object_points.push(landmark.position);
            image_points.push(query_kp.pt());
        }

        (object_points, image_points)
    }
}

/// Re-run PnP refinement over the inliers (no-op with fewer than 6 of them).
fn refine_pose(
    pose: Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    inlier_mask: &[bool],
    intrinsics: &CameraIntrinsics,
) -> Pose {
    let (inlier_obj, inlier_img): (Vec<_>, Vec<_>) = object_points
        .iter()
        .zip(image_points.iter())
        .zip(inlier_mask.iter())
        .filter_map(|((obj, img), &inlier)| if inlier { Some((*obj, *img)) } else { None })
        .unzip();

    if inlier_obj.len() < 6 {
        return pose;
    }

    solve_pnp_refine(&pose, &inlier_obj, &inlier_img, intrinsics, None, 25).unwrap_or(pose)
}

/// RMS reprojection error (pixels) of the inlier correspondences under `pose`.
fn reprojection_rmse(
    pose: &Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    inlier_mask: &[bool],
    intrinsics: &CameraIntrinsics,
) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for i in 0..object_points.len() {
        if !inlier_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        let camera_point = pose.rotation * object_points[i].coords + pose.translation;
        if camera_point[2] <= 1e-6 {
            continue;
        }
        let projected = intrinsics.project(&Point3::from(camera_point));
        sum_sq +=
            (projected.x - image_points[i].x).powi(2) + (projected.y - image_points[i].y).powi(2);
        count += 1;
    }

    if count == 0 {
        f64::INFINITY
    } else {
        (sum_sq / count as f64).sqrt()
    }
}
