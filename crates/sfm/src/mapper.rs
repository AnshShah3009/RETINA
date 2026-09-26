//! Incremental Structure-from-Motion.
//!
//! [`map_views`] turns an *ordered* list of [`View`]s (keypoints + descriptors +
//! image size) and one shared set of [`CameraIntrinsics`] into a
//! [`Reconstruction`] — estimated camera poses and 3D points — **without ever
//! using a ground-truth pose**. Every other SfM entry point in this crate takes
//! poses as an input; this module is the piece that produces them.
//!
//! # Pipeline
//!
//! 1. **Pair selection** ([`PairSelection`]). By default every view is paired
//!    with the next `window` views in time order. With
//!    [`PairSelection::SequentialWithRetrieval`] a bag-of-words widening adds,
//!    for every view, its `neighbours` most similar views, which is how a real
//!    system keeps non-adjacent (loop-closing) pairs in the covisibility graph.
//!
//! 2. **Pair verification.** Descriptors are matched with Lowe's ratio test and
//!    a mutual-consistency (cross) check, the fundamental matrix is estimated
//!    from the matches with a deterministic RANSAC (see *Determinism* below),
//!    and a homography is fitted from those same matches. A normalized
//!    MSAC model-selection score identifies pairs explained significantly better
//!    by a homography; only the surviving fundamental-matrix inlier
//!    correspondences are kept. Degenerate pairs still contribute these matches
//!    to track building, but are excluded from seed selection.
//!
//! 3. **Track building.** A disjoint-set (union-find) over `(view, keypoint)`
//!    nodes merges the verified correspondences into tracks, enforcing at most
//!    one observation per view per track. This is the same shape the
//!    `cv-localization` TUM benchmark harness uses for its map builder.
//!
//! 4. **Initialization.** A seed pair is chosen from the verified pairs that
//!    have the most inliers *and* the most points that survive triangulation
//!    (in front of both cameras, above a parallax floor, reprojecting tightly).
//!    The relative pose is recovered from the essential matrix
//!    ([`cv_calib3d::find_essential_mat`] + [`cv_calib3d::recover_pose_from_essential`]).
//!    The first camera of the seed pair defines the world frame (its
//!    world-to-camera pose is the identity); the reconstruction is therefore
//!    monocular and **scale-free**, with the seed baseline fixed to unit length.
//!
//! 5. **Incremental registration.** Repeatedly, the unregistered view with the
//!    most 2D-3D correspondences (via the tracks) is solved with
//!    [`cv_calib3d::solve_pnp_ransac`], refined on its inliers by
//!    [`cv_calib3d::solve_pnp_refine`] and added. After every registration, all
//!    tracks that now have at least two registered observations and no 3D point
//!    yet are triangulated from their widest-baseline registered pair and
//!    filtered by cheirality, parallax and reprojection error.
//!
//! 6. **Refinement.** After every successful incremental registration, a local
//!    bundle adjustment is run on the new camera, its most covisible registered
//!    neighbours, and their landmarks. The existing
//!    [`crate::bundle_adjustment::bundle_adjust`] is also run globally every
//!    `ba_every` registrations and once more at the end. The parameters are
//!    validated after each call: a non-finite result is rejected and the
//!    previous state kept.
//!
//! 7. **Output.** [`Reconstruction`] holds the cameras in ascending view order,
//!    the 3D points in creation order and each point's surviving observations.
//!    [`MappingReport`] says which views registered and, for the ones that did
//!    not, why.
//!
//! # Conventions
//!
//! A [`Pose`] maps points from the *local* (camera) frame to the *parent*
//! (world) frame: `p_parent = R * p_local + t`. The reconstruction stores
//! **world-to-camera** poses (`p_cam = R * p_world + t`), i.e. the inverse of
//! that, because that is the convention
//! [`crate::bundle_adjustment::SfMState`] and
//! [`cv_calib3d::solve_pnp_ransac`] use. The camera centre in world
//! coordinates is therefore `pose.inverse().translation`; see
//! [`Reconstruction::camera_centers`].
//!
//! # Determinism
//!
//! The output does not depend on hash-map iteration order or on any
//! unseeded randomness:
//!
//! * pairs, tracks, seed candidates and views are always processed in a fixed
//!   ascending index order;
//! * the fundamental-matrix and homography RANSACs use an iteration-indexed,
//!   seeded linear congruential sampler (the same construction `cv-calib3d`'s PnP
//!   RANSAC uses), never `rand`;
//! * [`cv_calib3d::solve_pnp_ransac`] samples by iteration index already, so it
//!   is deterministic; and
//! * no `HashMap`/`HashSet` is iterated anywhere in this module.
//!
//! # Example
//!
//! ```rust,ignore
//! use cv_sfm::mapper::{map_views, MapperConfig, View};
//! # use cv_core::{CameraIntrinsics, Descriptors};
//! # fn views() -> Vec<View> { Vec::new() }
//! let mapping = map_views(&views(), &CameraIntrinsics::new(517.3, 516.5, 318.6, 255.3, 640, 480),
//!                         &MapperConfig::default());
//! for outcome in &mapping.report.outcomes {
//!     println!("view {} -> registered={} ({:?})", outcome.view, outcome.registered, outcome.failure);
//! }
//! ```

use cv_calib3d::{
    find_essential_mat, find_fundamental_mat, recover_pose_from_essential, solve_dlt_homography,
    solve_pnp_ransac, triangulate_points,
};
use cv_core::{CameraIntrinsics, Descriptors, KeyPoint, Pose};
use cv_features::matcher::{MatchType, Matcher};
use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, Vector3};

use crate::bundle_adjustment::{bundle_adjust, BundleAdjustmentConfig, SfMState};

/// A single input view: its keypoints, their descriptors and the image size.
///
/// `keypoints` must be parallel to `descriptors` (descriptor `i` was computed at
/// keypoint `i`). Views are supplied in temporal order; the mapper relies on that
/// order for its default pair selection and for the deterministic tie-breaking.
#[derive(Debug, Clone)]
pub struct View {
    /// Keypoints, parallel to `descriptors`.
    pub keypoints: Vec<KeyPoint>,
    /// Descriptors, parallel to `keypoints`.
    pub descriptors: Descriptors,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl View {
    /// Build a view from its keypoints, descriptors and image size.
    pub fn new(
        keypoints: Vec<KeyPoint>,
        descriptors: Descriptors,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            keypoints,
            descriptors,
            width,
            height,
        }
    }

    /// Number of usable features (the minimum of the two parallel lists).
    pub fn len(&self) -> usize {
        self.descriptors.len().min(self.keypoints.len())
    }

    /// Whether the view carries no usable features.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How candidate image pairs are chosen for verification.
#[derive(Debug, Clone, PartialEq)]
pub enum PairSelection {
    /// Temporal neighbours only: view `i` is paired with `i + 1 ..= i + window`.
    Sequential {
        /// Number of following views each view is paired with (at least 1).
        window: usize,
    },
    /// Temporal neighbours **plus** a bag-of-words widening, so non-adjacent
    /// views that look alike (loop closures) also enter the covisibility graph.
    SequentialWithRetrieval {
        /// Number of following views each view is paired with (at least 1).
        window: usize,
        /// Number of visual words the vocabulary is trained with (at least 1).
        vocab_size: usize,
        /// Number of most similar views added per view (at least 1).
        neighbours: usize,
        /// Deterministic seed for the vocabulary's k-means++ initialization.
        seed: u64,
    },
}

/// Every knob of [`map_views`].
///
/// The defaults are the ones used for the reported TUM RGB-D numbers; see the
/// crate's `tum_sfm` example.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Lowe ratio-test threshold for descriptor matching.
    pub ratio: f32,
    /// Candidate-pair strategy.
    pub pair_selection: PairSelection,
    /// Minimum number of ratio-test + cross-check matches for a pair to be tried.
    pub min_pair_matches: usize,
    /// Inlier threshold of the fundamental-matrix RANSAC, in pixels (Sampson).
    pub f_ransac_threshold_px: f64,
    /// Match a query view's descriptors against the map's landmarks to find
    /// 2D-3D correspondences, instead of relying only on track membership.
    /// Without this, registration cannot reach past the verified pair graph.
    pub map_matching: bool,
    /// Lowe ratio applied when matching a query view against the map.
    ///
    /// Kept separate from `ratio` so the two can be tuned independently, but
    /// measured at 0.75: raising it to 0.90 quadrupled the correspondences
    /// (38 -> 591 per query on ETH3D courtyard) while inliers stayed at ~1, so
    /// the extra matches are noise. A looser ratio trades precision for volume
    /// that PnP cannot use.
    pub map_ratio: f32,
    /// Maximum fundamental-matrix RANSAC iterations (an adaptive bound may stop it
    /// earlier; that bound is a deterministic function of the data).
    pub f_ransac_iters: usize,
    /// Homography RANSAC transfer-error threshold, in pixels. It defaults to the
    /// fundamental threshold because both scores below use the same residual
    /// scale and the same correspondences.
    pub h_ransac_threshold_px: f64,
    /// Maximum homography RANSAC iterations.
    pub h_ransac_iters: usize,
    /// Minimum normalized-score advantage required to classify a verified pair
    /// as planar/degenerate. A pair is rejected as a seed only when its
    /// homography score exceeds its essential score by at least this margin.
    pub planar_score_margin: f64,
    /// Minimum inliers for a pair to count as verified.
    pub min_pair_inliers: usize,
    /// Minimum inliers for a pair to be considered as a seed.
    pub min_seed_inliers: usize,
    /// Minimum number of cleanly triangulating correspondences for a pair to be
    /// considered as a seed (the same score used to rank candidate seeds).
    pub min_seed_points: usize,
    /// Number of best-scoring seed candidates actually evaluated.
    pub seed_candidates: usize,
    /// Number of candidate seeds for which the full incremental registration is
    /// attempted. Two-view initialization is ill-conditioned whenever the seed
    /// parallax is small, so several seeds are tried and the one that registers
    /// the most views wins. `1` disables the retry.
    pub seed_hypotheses: usize,
    /// Force a specific seed pair, bypassing the ranking. `None` lets the mapper
    /// choose. Useful for reproducing a run or for diagnosing initialization.
    pub seed_pair: Option<(usize, usize)>,
    /// Minimum parallax, in degrees, for a triangulated point to be accepted.
    pub min_parallax_deg: f64,
    /// Maximum reprojection error, in pixels, for an accepted observation.
    pub max_reproj_px: f64,
    /// Re-triangulate existing landmarks from the now-wider registered baseline
    /// after every registration, updating their point and observation set.
    pub retriangulate: bool,
    /// Minimum 2D-3D correspondences to attempt PnP on a view.
    pub min_pnp_correspondences: usize,
    /// PnP RANSAC reprojection threshold, in pixels.
    pub pnp_ransac_threshold_px: f64,
    /// PnP RANSAC iterations (`cv-calib3d` enforces a floor of 64).
    ///
    /// The minimal sample is 6 points, so this has to be large to have a chance
    /// of drawing a clean sample: for an inlier ratio `w` the required count for
    /// 99 % confidence is `ln(0.01) / ln(1 - w^6)`, i.e. ~1300 iterations at
    /// `w = 0.4` and ~11500 at `w = 0.3`.
    pub pnp_ransac_iters: usize,
    /// Minimum PnP inliers for a registration to be accepted.
    pub min_pnp_inliers: usize,
    /// Minimum PnP inlier ratio (`inliers / correspondences`) for a registration
    /// to be accepted. See the acceptance site for why the ratio matters even
    /// when the absolute count is satisfied.
    pub min_pnp_inlier_ratio: f64,
    /// Run bundle adjustment every this many registrations (`0` disables it).
    pub ba_every: usize,
    /// Number of registered cameras in each local bundle-adjustment problem,
    /// including the newly registered camera. `0` disables local adjustment.
    pub local_ba_window: usize,
    /// Minimum number of shared landmarks between the new camera and a candidate
    /// neighbour for that neighbour to enter the local problem.
    pub local_ba_min_overlap: usize,
    /// Maximum number of landmarks in one local bundle adjustment, taken in
    /// descending order of how many selected cameras observe them.
    ///
    /// Bounds the cost so a local solve stays local as the map grows. Measured
    /// on TUM fr1_desk, 150 views: uncapped the local problem reached 841
    /// points and 174 ms per call; 300 points is ~75 ms but rotation error
    /// degrades 3.8 -> 9.3 degrees, and 600 gives 6.5 degrees. 800 preserves
    /// the uncapped accuracy (3.85 degrees) at 195 s instead of 212 s.
    pub local_ba_max_points: usize,
    /// Run one final bundle adjustment after the last registration.
    pub ba_final: bool,
    /// Bundle-adjustment iterations per call.
    pub ba_max_iterations: usize,
    /// Bundle-adjustment sparsity flag (passed through to
    /// [`BundleAdjustmentConfig::use_sparsity`]).
    pub ba_use_sparsity: bool,
    /// Bundle-adjustment robust-kernel flag (passed through to
    /// [`BundleAdjustmentConfig::robust_kernel`]).
    pub ba_robust_kernel: bool,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            ratio: 0.75,
            pair_selection: PairSelection::Sequential { window: 3 },
            min_pair_matches: 20,
            f_ransac_threshold_px: 1.5,
            map_matching: true,
            map_ratio: 0.75,
            f_ransac_iters: 500,
            h_ransac_threshold_px: 1.5,
            h_ransac_iters: 500,
            planar_score_margin: 0.05,
            min_pair_inliers: 30,
            min_seed_inliers: 20,
            min_seed_points: 30,
            seed_candidates: 16,
            seed_hypotheses: 8,
            seed_pair: None,
            min_parallax_deg: 1.0,
            max_reproj_px: 4.0,
            retriangulate: true,
            min_pnp_correspondences: 12,
            pnp_ransac_threshold_px: 4.0,
            pnp_ransac_iters: 2000,
            min_pnp_inliers: 10,
            min_pnp_inlier_ratio: 0.25,
            ba_every: 10,
            local_ba_window: 6,
            local_ba_min_overlap: 10,
            local_ba_max_points: 800,
            ba_final: true,
            ba_max_iterations: 10,
            ba_use_sparsity: true,
            ba_robust_kernel: false,
        }
    }
}

/// Why a view could not be registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationFailure {
    /// The view never saw `min_pnp_correspondences` 3D points.
    InsufficientCorrespondences,
    /// The view is not connected to the reconstructed part of the scene at all.
    Unreachable,
    /// `solve_pnp_ransac` returned an error for the last correspondence set tried.
    PnpEstimationFailed,
    /// PnP returned a pose but with too few inliers.
    TooFewInliers,
}

impl RegistrationFailure {
    /// A short human-readable description, used by the report printers.
    pub fn description(self) -> &'static str {
        match self {
            RegistrationFailure::InsufficientCorrespondences => {
                "too few 2D-3D correspondences for PnP"
            }
            RegistrationFailure::Unreachable => "no 3D point visible in this view",
            RegistrationFailure::PnpEstimationFailed => "PnP RANSAC failed to return a pose",
            RegistrationFailure::TooFewInliers => "PnP pose had too few inliers",
        }
    }
}

/// What happened to one view.
#[derive(Debug, Clone)]
pub struct ViewOutcome {
    /// Index of the view in the input list.
    pub view: usize,
    /// Whether the view ended up registered.
    pub registered: bool,
    /// 2D-3D correspondences available at the last attempt (registration time for
    /// registered views).
    pub correspondences: usize,
    /// PnP inliers at the last attempt (`0` when no PnP was attempted).
    pub inliers: usize,
    /// Why the view failed, when it did.
    pub failure: Option<RegistrationFailure>,
}

/// The reconstructed scene.
///
/// Cameras are stored in **ascending view index** order and hold
/// *world-to-camera* poses. Points are in creation order (which is the ascending
/// track order of the triangulation pass, so it is deterministic). Every point
/// has at least two observations.
#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    /// `(view index, world-to-camera pose)`, ascending by view index.
    pub cameras: Vec<(usize, Pose)>,
    /// Triangulated 3D points.
    pub points: Vec<Point3<f64>>,
    /// `observations[p]` lists the `(view, keypoint)` observations of point `p`
    /// that survived the cheirality/parallax/reprojection filter, ascending by
    /// view.
    pub observations: Vec<Vec<(usize, usize)>>,
}

impl Reconstruction {
    /// Number of registered cameras.
    pub fn len(&self) -> usize {
        self.cameras.len()
    }

    /// Whether nothing was reconstructed.
    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    /// View indices of the registered cameras, ascending.
    pub fn view_indices(&self) -> Vec<usize> {
        self.cameras.iter().map(|&(view, _)| view).collect()
    }

    /// Camera centres in world coordinates, parallel to [`Self::cameras`].
    pub fn camera_centers(&self) -> Vec<Vector3<f64>> {
        self.cameras
            .iter()
            .map(|(_, pose)| pose.inverse().translation)
            .collect()
    }

    /// Total number of 2D observations across all points.
    pub fn observation_count(&self) -> usize {
        self.observations.iter().map(Vec::len).sum()
    }
}

/// What the mapper did, and what it could not do.
#[derive(Debug, Clone)]
pub struct MappingReport {
    /// Views handed to the mapper.
    pub views_supplied: usize,
    /// Views that ended up registered.
    pub registered: usize,
    /// `registered / views_supplied` (`0.0` when no view was supplied).
    pub registration_rate: f64,
    /// Candidate pairs produced by the pair-selection strategy.
    pub pairs_selected: usize,
    /// Candidate pairs that passed matching + fundamental-matrix verification.
    pub pairs_verified: usize,
    /// Verified pairs whose homography significantly outscored the essential
    /// model. Their correspondences remain in track building, but they are not
    /// seed candidates.
    pub pairs_planar: usize,
    /// Mean RMS Sampson error of the verified pairs' inliers, in pixels (`0.0`
    /// when no pair was verified). A sanity check on the covisibility graph.
    pub mean_pair_epipolar_rmse_px: f64,
    /// Per verified pair: fundamental and homography RANSAC results, normalized
    /// scores and whether the pair was classified planar. These diagnostics are
    /// populated for every pair that reached both model fits, including pairs
    /// that cannot be used as seeds.
    pub pair_model_diagnostics: Vec<PairModelDiagnostic>,
    /// Per candidate pair, in pair-selection order: `(view a, view b, descriptor
    /// matches, verified inliers)`. `None` inliers means the pair failed
    /// verification; a pair that produced too few matches reports them anyway, so
    /// the table explains where the covisibility graph came from.
    pub pair_diagnostics: Vec<(usize, usize, usize, Option<usize>)>,
    /// Per seed candidate: `(view a, view b, scoring correspondences)`. The score
    /// is the number of correspondences that triangulate cleanly in the
    /// candidate's own gauge; only the first [`MapperConfig::seed_hypotheses`]
    /// entries are actually run.
    pub seed_diagnostics: Vec<(usize, usize, usize)>,
    /// The `(view, view)` seed pair, when one was found.
    pub seed: Option<(usize, usize)>,
    /// Seed hypotheses for which the full incremental registration was run.
    pub seed_hypotheses_tried: usize,
    /// Number of 3D points in the reconstruction.
    pub num_points: usize,
    /// Total number of observations attached to those points.
    pub num_observations: usize,
    /// `num_observations / num_points` (`0.0` when there are no points).
    pub mean_track_length: f64,
    /// Number of successful full-reconstruction `bundle_adjust` calls (rejected/failed
    /// calls are not counted).
    pub ba_runs: usize,
    /// Number of successful local `bundle_adjust` calls (rejected/failed calls are
    /// not counted).
    pub local_ba_runs: usize,
    /// Per-view outcome, ascending by view index.
    pub outcomes: Vec<ViewOutcome>,
}

/// A reconstruction plus the report describing how it was obtained.
#[derive(Debug, Clone)]
pub struct Mapping {
    /// The reconstructed cameras and points.
    pub reconstruction: Reconstruction,
    /// What was registered and why anything failed.
    pub report: MappingReport,
}

/// Emit a stage/progress line to stderr when `CV_SFM_PROGRESS` is set.
///
/// The mapping report is only available once the whole run finishes, so a stage
/// that is slow or stuck is invisible from the outside. This writes to stderr
/// (unbuffered) and is a no-op unless the variable is set, so it costs nothing
/// in normal use.
fn progress(args: std::fmt::Arguments<'_>) {
    if std::env::var_os("CV_SFM_PROGRESS").is_some() {
        eprintln!("[sfm] {args}");
    }
}

/// Run the incremental mapper.
///
/// See the [module documentation](self) for the algorithm and the pose
/// convention. The function never panics on empty or degenerate input: it
/// returns an empty [`Reconstruction`] and a report explaining that no seed
/// could be found.
///
/// Two-view initialization is only as good as the seed pair: with a small
/// parallax, or a scene that is locally planar, the essential-matrix
/// decomposition can be badly wrong and no amount of downstream work repairs it.
/// The mapper therefore evaluates the best `seed_hypotheses` candidate seeds by
/// running the whole incremental registration for each and keeping the one that
/// registers the most views. The run stops early as soon as a hypothesis
/// registers every view.
pub fn map_views(views: &[View], intrinsics: &CameraIntrinsics, config: &MapperConfig) -> Mapping {
    let n = views.len();
    progress(format_args!("map_views: {n} views"));

    // ---- 1. Pair selection ----
    let pairs = select_pairs(views, config);
    progress(format_args!("pair selection: {} pairs", pairs.len()));

    // ---- 2. Pairwise verification ----
    let mut verified: Vec<VerifiedPair> = Vec::new();
    let mut pair_diagnostics: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
    let mut pair_model_diagnostics: Vec<PairModelDiagnostic> = Vec::new();
    if n >= 2 {
        let mut done = 0usize;
        for &(a, b) in &pairs {
            let (matches, pair) = verify_pair(a, b, views, intrinsics, config);
            pair_diagnostics.push((a, b, matches, pair.as_ref().map(|p| p.inliers.len())));
            if let Some(pair) = pair {
                pair_model_diagnostics.push(pair.model);
                verified.push(pair);
            }
            done += 1;
            if done % 10 == 0 {
                progress(format_args!(
                    "verify: {done}/{} pairs, {} verified",
                    pairs.len(),
                    verified.len()
                ));
            }
        }
    }
    progress(format_args!("verify done: {} verified", verified.len()));

    // ---- 3. Tracks ----
    let tracks = build_tracks(views, &verified);
    progress(format_args!("tracks: {}", tracks.len()));

    // ---- 4. Seed candidates ----
    let (candidates, seed_diagnostics) = seed_candidates(&verified, views, intrinsics, config);
    progress(format_args!("seed candidates: {}", candidates.len()));

    let mut report = MappingReport {
        views_supplied: n,
        registered: 0,
        registration_rate: 0.0,
        pairs_selected: pairs.len(),
        pairs_verified: verified.len(),
        pairs_planar: verified.iter().filter(|pair| pair.model.planar).count(),
        mean_pair_epipolar_rmse_px: if verified.is_empty() {
            0.0
        } else {
            verified
                .iter()
                .map(|pair| pair.epipolar_rmse_px)
                .sum::<f64>()
                / verified.len() as f64
        },
        pair_diagnostics,
        pair_model_diagnostics,
        seed_diagnostics,
        seed: None,
        seed_hypotheses_tried: 0,
        num_points: 0,
        num_observations: 0,
        mean_track_length: 0.0,
        ba_runs: 0,
        local_ba_runs: 0,
        outcomes: Vec::new(),
    };

    if candidates.is_empty() {
        // No usable seed: report every view as unreachable (nothing was built).
        let outcomes = (0..n)
            .map(|view| ViewOutcome {
                view,
                registered: false,
                correspondences: 0,
                inliers: 0,
                failure: Some(RegistrationFailure::Unreachable),
            })
            .collect();
        report.outcomes = outcomes;
        return Mapping {
            reconstruction: Reconstruction::default(),
            report,
        };
    }

    // ---- 5. Try the best seed hypotheses, keep the one that registers most ----
    let mut best: Option<Hypothesis> = None;
    for seed in candidates.iter().take(config.seed_hypotheses.max(1)) {
        report.seed_hypotheses_tried += 1;
        let hypothesis = run_incremental(seed, &tracks, views, intrinsics, config);
        let is_better = match &best {
            None => true,
            Some(current) => {
                hypothesis.est.poses.len() > current.est.poses.len()
                    || (hypothesis.est.poses.len() == current.est.poses.len()
                        && hypothesis.est.points.len() > current.est.points.len())
            }
        };
        if is_better {
            best = Some(hypothesis);
        }
        if best
            .as_ref()
            .is_some_and(|current| current.est.poses.len() == n)
        {
            break;
        }
    }
    let Some(best) = best else {
        report.outcomes = (0..n)
            .map(|view| ViewOutcome {
                view,
                registered: false,
                correspondences: 0,
                inliers: 0,
                failure: Some(RegistrationFailure::Unreachable),
            })
            .collect();
        return Mapping {
            reconstruction: Reconstruction::default(),
            report,
        };
    };

    let Hypothesis {
        mut est,
        mut outcomes,
        ba_runs,
        local_ba_runs,
        seed,
    } = best;
    report.seed = Some((seed.a, seed.b));
    report.ba_runs = ba_runs;
    report.local_ba_runs = local_ba_runs;

    // ---- 6. Report remaining failures ----
    let final_corr = gather_correspondences_matching(&est, views, config);
    for view in 0..n {
        if outcomes[view].is_none() {
            let count = final_corr[view].len();
            outcomes[view] = Some(ViewOutcome {
                view,
                registered: false,
                correspondences: count,
                inliers: 0,
                failure: Some(if count == 0 {
                    RegistrationFailure::Unreachable
                } else {
                    RegistrationFailure::InsufficientCorrespondences
                }),
            });
        }
    }

    // ---- 7. Build the deterministic output ----
    let mut cameras: Vec<(usize, Pose)> = est
        .views_of_cam
        .iter()
        .enumerate()
        .map(|(cam, &view)| (view, est.poses[cam]))
        .collect();
    cameras.sort_by_key(|&(view, _)| view);

    // Re-point the state at its output vectors so no copy of the point cloud is
    // made when the reconstruction is assembled.
    est.points.shrink_to_fit();
    let reconstruction = Reconstruction {
        cameras,
        points: est.points,
        observations: est.point_obs,
    };

    report.registered = reconstruction.cameras.len();
    report.registration_rate = if n == 0 {
        0.0
    } else {
        report.registered as f64 / n as f64
    };
    report.num_points = reconstruction.points.len();
    report.num_observations = reconstruction.observation_count();
    report.mean_track_length = if reconstruction.points.is_empty() {
        0.0
    } else {
        report.num_observations as f64 / reconstruction.points.len() as f64
    };
    report.outcomes = finish_outcomes(outcomes, n);

    Mapping {
        reconstruction,
        report,
    }
}

/// The result of one seed hypothesis.
struct Hypothesis {
    est: Est,
    outcomes: Vec<Option<ViewOutcome>>,
    ba_runs: usize,
    local_ba_runs: usize,
    seed: SeedChoice,
}

/// Initialize from one seed and register every view that becomes reachable.
///
/// `last_tried[view]` records the correspondence count of the last failed
/// attempt, so a view is only retried once new 3D points have become visible in
/// it. That keeps the loop bounded without ever giving up on a view permanently.
fn run_incremental(
    seed: &SeedChoice,
    tracks: &[Vec<(usize, usize)>],
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> Hypothesis {
    let n = views.len();
    let mut outcomes: Vec<Option<ViewOutcome>> = vec![None; n];
    let mut ba_runs = 0usize;
    let mut local_ba_runs = 0usize;

    // ---- Initialization: the world frame is the seed's first camera ----
    let mut est = Est::new(n, tracks.to_vec());
    est.add_camera(seed.a, Pose::identity());
    est.add_camera(seed.b, seed.pose_b);
    for view in [seed.a, seed.b] {
        outcomes[view] = Some(ViewOutcome {
            view,
            registered: true,
            correspondences: 0,
            inliers: 0,
            failure: None,
        });
    }
    triangulate_new_tracks(&mut est, views, intrinsics, config);

    // ---- Incremental registration ----
    // `attempts_left` bounds the loop. A view is retried only after the map has
    // grown, and the previous guard (`last_tried` holding the correspondence
    // count) was not sufficient: retriangulation can move that count back to a
    // previously seen value, which re-enabled the same failed view forever. The
    // loop therefore terminates after a bounded number of passes with no
    // registration, and each view gets at most `max_view_attempts` tries.
    const MAX_IDLE_PASSES: usize = 3;
    const MAX_VIEW_ATTEMPTS: usize = 3;
    let mut attempts: Vec<u8> = vec![0; n];
    let mut since_ba = 0usize;
    let mut idle_passes = 0usize;

    loop {
        if idle_passes >= MAX_IDLE_PASSES {
            progress(format_args!(
                "registration: stopping after {MAX_IDLE_PASSES} idle passes"
            ));
            break;
        }
        let corr = gather_correspondences_matching(&est, views, config);
        let mut order: Vec<usize> = (0..n)
            .filter(|&v| {
                est.cam_of_view[v].is_none()
                    && corr[v].len() >= config.min_pnp_correspondences
                    && (attempts[v] as usize) < MAX_VIEW_ATTEMPTS
            })
            .collect();
        // Most correspondences first; lowest view index breaks ties.
        order.sort_by(|&x, &y| corr[y].len().cmp(&corr[x].len()).then(x.cmp(&y)));
        progress(format_args!(
            "registration pass: {} views registered, {} candidates",
            est.cam_of_view.iter().filter(|c| c.is_some()).count(),
            order.len()
        ));

        let mut progressed = false;
        for v in order {
            let entries = &corr[v];
            attempts[v] = attempts[v].saturating_add(1);
            let object_points: Vec<Point3<f64>> =
                entries.iter().map(|&(p, _)| est.points[p]).collect();
            let image_points: Vec<Point2<f64>> = entries
                .iter()
                .map(|&(_, kp)| views[v].keypoints[kp].pt())
                .collect();

            let attempt = solve_pnp_ransac(
                &object_points,
                &image_points,
                intrinsics,
                None,
                config.pnp_ransac_threshold_px,
                config.pnp_ransac_iters,
            );
            match attempt {
                Ok((pose, inliers)) => {
                    let inlier_count = inliers.iter().filter(|&&flag| flag).count();
                    let inlier_ratio = inlier_count as f64 / entries.len().max(1) as f64;
                    // Both gates are needed. Measured on TUM fr1_desk: relaxing
                    // this to an absolute inlier count alone (COLMAP-style)
                    // registered one more view, but that view's pose was wrong
                    // (12 inliers of 73), and retriangulation then spread the
                    // error through the map — camera-centre RMSE went from
                    // 0.0102 m to 0.0604 m and the rotation error from 0.79 to
                    // 3.09 degrees. The ratio is what rejects a confident-looking
                    // pose built from mostly wrong correspondences.
                    if inlier_count >= config.min_pnp_inliers
                        && inlier_ratio >= config.min_pnp_inlier_ratio
                        && pose_is_finite(&pose)
                    {
                        est.add_camera(v, pose);
                        outcomes[v] = Some(ViewOutcome {
                            view: v,
                            registered: true,
                            correspondences: entries.len(),
                            inliers: inlier_count,
                            failure: None,
                        });
                        triangulate_new_tracks(&mut est, views, intrinsics, config);
                        if config.retriangulate {
                            retriangulate(&mut est, views, intrinsics, config);
                        }
                        since_ba += 1;
                        if refine_local(&mut est, v, views, intrinsics, config) {
                            local_ba_runs += 1;
                        }
                        if config.ba_every > 0 && since_ba >= config.ba_every {
                            if refine(&mut est, views, intrinsics, config) {
                                ba_runs += 1;
                            }
                            since_ba = 0;
                        }
                        progressed = true;
                        break;
                    }
                    outcomes[v] = Some(ViewOutcome {
                        view: v,
                        registered: false,
                        correspondences: entries.len(),
                        inliers: inlier_count,
                        failure: Some(RegistrationFailure::TooFewInliers),
                    });
                }
                Err(_) => {
                    outcomes[v] = Some(ViewOutcome {
                        view: v,
                        registered: false,
                        correspondences: entries.len(),
                        inliers: 0,
                        failure: Some(RegistrationFailure::PnpEstimationFailed),
                    });
                }
            }
        }

        if progressed {
            idle_passes = 0;
        } else {
            // No view registered this pass. Retry the remaining candidates after
            // the map has settled (a later view may add points an earlier one
            // could not see), but only a bounded number of times.
            idle_passes += 1;
        }
    }

    if config.ba_final
        && est.poses.len() >= 2
        && !est.points.is_empty()
        && refine(&mut est, views, intrinsics, config)
    {
        ba_runs += 1;
    }

    Hypothesis {
        est,
        outcomes,
        ba_runs,
        local_ba_runs,
        seed: *seed,
    }
}

/// Fill in any still-missing outcome (defensive; the loops above cover all views).
fn finish_outcomes(mut outcomes: Vec<Option<ViewOutcome>>, n: usize) -> Vec<ViewOutcome> {
    for (view, slot) in outcomes.iter_mut().enumerate().take(n) {
        if slot.is_none() {
            *slot = Some(ViewOutcome {
                view,
                registered: false,
                correspondences: 0,
                inliers: 0,
                failure: Some(RegistrationFailure::Unreachable),
            });
        }
    }
    outcomes.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// Working state
// ---------------------------------------------------------------------------

/// Mutable reconstruction state.
///
/// Cameras are indexed by *registration order* (`cam_of_view` maps a view index
/// to that camera index); points are indexed by creation order. Both are plain
/// vectors, so nothing depends on hash-map iteration order.
struct Est {
    /// World-to-camera pose per registered camera, in registration order.
    poses: Vec<Pose>,
    /// View index per registered camera, in registration order.
    views_of_cam: Vec<usize>,
    /// Camera index per view, or `None` while the view is unregistered.
    cam_of_view: Vec<Option<usize>>,
    /// 3D point per landmark.
    points: Vec<Point3<f64>>,
    /// Surviving observations per landmark, parallel to `points`.
    point_obs: Vec<Vec<(usize, usize)>>,
    /// Track index per landmark, parallel to `points`.
    point_tracks: Vec<usize>,
    /// Landmark index per track, or `None` while the track has no 3D point.
    track_point: Vec<Option<usize>>,
    /// All observations `(view, keypoint)` per track, ascending by view.
    track_obs: Vec<Vec<(usize, usize)>>,
}

impl Est {
    fn new(n_views: usize, track_obs: Vec<Vec<(usize, usize)>>) -> Self {
        let n_tracks = track_obs.len();
        Self {
            poses: Vec::new(),
            views_of_cam: Vec::new(),
            cam_of_view: vec![None; n_views],
            points: Vec::new(),
            point_obs: Vec::new(),
            point_tracks: Vec::new(),
            track_point: vec![None; n_tracks],
            track_obs,
        }
    }

    /// Register a camera for `view`; returns its camera index.
    fn add_camera(&mut self, view: usize, pose: Pose) -> usize {
        let cam = self.poses.len();
        self.poses.push(pose);
        self.views_of_cam.push(view);
        self.cam_of_view[view] = Some(cam);
        cam
    }

    /// World-to-camera pose of `view`, if it is registered.
    fn pose_of(&self, view: usize) -> Option<Pose> {
        self.cam_of_view[view].map(|cam| self.poses[cam])
    }
}

/// For every view, the `(landmark, keypoint)` pairs that link it to an existing
/// 3D point. Only unregistered views are interesting, but all are filled so the
/// final failure report can use the same structure.
/// 2D-3D correspondences for every unregistered view.
///
/// A view can only be offered points whose tracks already contain one of its
/// keypoints, which limits registration to the pair graph that happened to be
/// verified. On a wide-baseline sequence that graph does not reach far: measured
/// on ETH3D courtyard, views past the verified window reported "no 3D point
/// visible in this view" with zero correspondences while the map held thousands
/// of points.
///
/// So the query view's descriptors are also matched directly against the
/// descriptors the map already stores for its landmarks. A match gives the
/// (landmark, this view's keypoint) pair that PnP needs. Track membership is
/// tried first because it is exact; descriptor matching is the fallback that lets
/// the map grow past the verified pairs.
fn gather_correspondences_matching(
    est: &Est,
    views: &[View],
    config: &MapperConfig,
) -> Vec<Vec<(usize, usize)>> {
    let mut corr: Vec<Vec<(usize, usize)>> = vec![Vec::new(); est.cam_of_view.len()];
    for (point, &track) in est.point_tracks.iter().enumerate() {
        for &(view, kp) in &est.track_obs[track] {
            if est.cam_of_view[view].is_none() {
                corr[view].push((point, kp));
            }
        }
    }

    if !config.map_matching {
        return corr;
    }

    // Descriptors of every landmark, kept parallel to `points`, so a match index
    // maps straight back to a landmark.
    // Build the map's descriptor table by walking the tracks once. EVERY
    // registered observation of a landmark is included, not just the first:
    // a landmark's appearance drifts with viewpoint, so matching a distant view
    // against one view's descriptor produces mostly wrong correspondences
    // (measured: 59-101 correspondences, 0-1 inliers). Duplicates are harmless
    // because a match is deduplicated by landmark.
    let mut map_descs = Descriptors::with_capacity(est.points.len());
    let mut map_point: Vec<usize> = Vec::with_capacity(est.points.len());
    for (point, &track) in est.point_tracks.iter().enumerate() {
        for &(view, kp) in &est.track_obs[track] {
            if est.cam_of_view[view].is_some() {
                if let Some(d) = views[view].descriptors.descriptors.get(kp) {
                    map_descs.push(d.clone());
                    map_point.push(point);
                }
            }
        }
    }
    if map_descs.len() < 8 {
        return corr;
    }

    let matcher = Matcher::new(MatchType::BruteForce).with_ratio_test(config.map_ratio);
    for (view, entry) in corr.iter_mut().enumerate() {
        if est.cam_of_view[view].is_some() {
            continue;
        }
        if entry.len() >= config.min_pnp_correspondences {
            continue;
        }
        let matches = matcher.match_descriptors(&views[view].descriptors, &map_descs);
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for m in &matches.matches {
            let (Ok(q), Ok(t)) = (usize::try_from(m.query_idx), usize::try_from(m.train_idx))
            else {
                continue;
            };
            let Some(&point) = map_point.get(t) else {
                continue;
            };
            if seen.insert(point) {
                entry.push((point, q));
            }
        }
    }
    corr
}

// ---------------------------------------------------------------------------
// Pair selection
// ---------------------------------------------------------------------------

/// Candidate view pairs, sorted ascending and de-duplicated.
fn select_pairs(views: &[View], config: &MapperConfig) -> Vec<(usize, usize)> {
    let n = views.len();
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    let window = match &config.pair_selection {
        PairSelection::Sequential { window } => *window,
        PairSelection::SequentialWithRetrieval { window, .. } => *window,
    }
    .max(1);

    for i in 0..n {
        for j in i + 1..(i + 1 + window).min(n) {
            pairs.push((i, j));
        }
    }

    if let PairSelection::SequentialWithRetrieval {
        vocab_size,
        neighbours,
        seed,
        ..
    } = &config.pair_selection
    {
        pairs.extend(retrieval_pairs(views, *vocab_size, *neighbours, *seed));
    }

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Pairs added by the bag-of-words widening.
///
/// A vocabulary is trained on every view's descriptors, each view gets a TF-IDF
/// bag-of-words vector, and the `neighbours` most similar views (L1 similarity,
/// ties broken by ascending view index) are added as pairs on top of the
/// temporal ones.
fn retrieval_pairs(
    views: &[View],
    vocab_size: usize,
    neighbours: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    use cv_features::retrieval::{descriptors_to_bytes, BowVector, Vocabulary};

    let bytes: Vec<Vec<Vec<u8>>> = views
        .iter()
        .map(|view| descriptors_to_bytes(&view.descriptors))
        .collect();
    let pool: Vec<Vec<u8>> = bytes.iter().flatten().cloned().collect();
    if pool.is_empty() {
        return Vec::new();
    }
    let vocabulary = Vocabulary::train(&pool, vocab_size.max(1), 10, seed);
    let bows: Vec<BowVector> = bytes
        .iter()
        .map(|descriptors| BowVector::from_descriptors(&vocabulary, descriptors))
        .collect();

    let mut out: Vec<(usize, usize)> = Vec::new();
    for (i, query) in bows.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> = bows
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, candidate)| (BowVector::score_l1(query, candidate), j))
            .collect();
        // Highest similarity first, ascending index breaks ties.
        scored.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(&y.1))
        });
        for &(_, j) in scored.iter().take(neighbours.max(1)) {
            out.push(if i < j { (i, j) } else { (j, i) });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Pair verification
// ---------------------------------------------------------------------------

/// Diagnostic record for the two competing two-view models.
///
/// The scores are computed from the same set of descriptor matches and the same
/// pixel threshold. They are deliberately kept in the public report so a run
/// can explain why a pair was (or was not) considered planar.
#[derive(Debug, Clone, Copy)]
pub struct PairModelDiagnostic {
    /// Lower view index.
    pub a: usize,
    /// Higher view index.
    pub b: usize,
    /// Number of descriptor matches entering both fits.
    pub matches: usize,
    /// Number of fundamental-matrix inliers.
    pub fundamental_inliers: usize,
    /// Number of homography inliers.
    pub homography_inliers: usize,
    /// Normalized Sampson score of the essential model.
    pub essential_score: f64,
    /// Normalized transfer-error score of the homography model.
    pub homography_score: f64,
    /// `homography_score - essential_score` (zero when a fit was unavailable).
    pub score_margin: f64,
    /// Whether the pair is marked planar/degenerate and excluded from seeding.
    pub planar: bool,
}

/// A candidate pair whose epipolar geometry survived RANSAC.
struct VerifiedPair {
    /// Lower view index.
    a: usize,
    /// Higher view index.
    b: usize,
    /// Inlier keypoint correspondences as `(keypoint in a, keypoint in b)`.
    inliers: Vec<(usize, usize)>,
    /// RMS Sampson error of the inliers under the fitted fundamental matrix, in
    /// pixels. Reported as a sanity metric for the covisibility graph.
    epipolar_rmse_px: f64,
    /// Homography-vs-essential model-selection decision for this pair.
    model: PairModelDiagnostic,
}

/// Match, ratio-test, cross-check and RANSAC-verify one pair.
///
/// Returns the number of descriptor matches (after the ratio test and
/// cross-check) together with the verified pair, or `None` when the pair failed.
fn verify_pair(
    a: usize,
    b: usize,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> (usize, Option<VerifiedPair>) {
    let va = &views[a];
    let vb = &views[b];
    if va.is_empty() || vb.is_empty() {
        return (0, None);
    }

    let matcher = Matcher::new(MatchType::BruteForce)
        .with_ratio_test(config.ratio)
        .with_cross_check();
    let matches = matcher.match_descriptors(&va.descriptors, &vb.descriptors);
    let match_count = matches.matches.len();

    let min_required = config.min_pair_matches.max(8);
    if match_count < min_required {
        return (match_count, None);
    }

    let mut points_a: Vec<Point2<f64>> = Vec::with_capacity(match_count);
    let mut points_b: Vec<Point2<f64>> = Vec::with_capacity(match_count);
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(match_count);
    let lim_a = va.len();
    let lim_b = vb.len();
    for m in &matches.matches {
        let (Ok(ai), Ok(bi)) = (usize::try_from(m.query_idx), usize::try_from(m.train_idx)) else {
            continue;
        };
        if ai >= lim_a || bi >= lim_b {
            continue;
        }
        points_a.push(va.keypoints[ai].pt());
        points_b.push(vb.keypoints[bi].pt());
        indices.push((ai, bi));
    }
    if points_a.len() < min_required {
        return (match_count, None);
    }

    let Some((f, mask)) = robust_fundamental(
        &points_a,
        &points_b,
        config.f_ransac_threshold_px,
        config.f_ransac_iters,
        pair_seed(a, b),
    ) else {
        return (match_count, None);
    };

    let inliers: Vec<(usize, usize)> = indices
        .iter()
        .zip(mask.iter())
        .filter_map(|(&pair, &is_inlier)| if is_inlier { Some(pair) } else { None })
        .collect();
    if inliers.len() < config.min_pair_inliers.max(8) {
        return (match_count, None);
    }

    // Diagnostic: RMS Sampson error of the inliers under the fitted model.
    let error_sum: f64 = inliers
        .iter()
        .map(|&(ka, kb)| sampson_sq(&f, &va.keypoints[ka].pt(), &vb.keypoints[kb].pt()))
        .sum();
    let epipolar_rmse_px = (error_sum / inliers.len() as f64).max(0.0).sqrt();

    // Fit the competing homography on exactly the same descriptor matches. The
    // fundamental RANSAC mask is retained for the track filter, but is not used
    // as the homography input: a planar scene must not lose correspondences that
    // the epipolar model itself finds ambiguous. The essential hypothesis is
    // fitted on the F inliers, which are the correspondences already accepted by
    // the existing pair verification; this keeps both model scores comparable
    // without introducing a second random sampler.
    let (h, h_mask) = robust_homography(
        &points_a,
        &points_b,
        config.h_ransac_threshold_px,
        config.h_ransac_iters,
        pair_seed(a, b) ^ 0xD1B5_4A32_D192_ED03,
    )
    .unwrap_or((Matrix3::identity(), vec![false; points_a.len()]));
    let inlier_points_a: Vec<Point2<f64>> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| is_inlier.then_some(points_a[i]))
        .collect();
    let inlier_points_b: Vec<Point2<f64>> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| is_inlier.then_some(points_b[i]))
        .collect();
    let essential_score = if inlier_points_a.len() >= 8 {
        find_essential_mat(&inlier_points_a, &inlier_points_b, intrinsics)
            .map(|essential| {
                essential_model_score(
                    &essential,
                    &points_a,
                    &points_b,
                    intrinsics,
                    config.f_ransac_threshold_px,
                )
            })
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let homography_score =
        normalized_homography_score(&h, &points_a, &points_b, config.h_ransac_threshold_px);
    let score_margin = homography_score - essential_score;
    let margin = config.planar_score_margin.max(0.0);
    let planar = margin > 0.0 && homography_score > essential_score && score_margin >= margin;
    let model = PairModelDiagnostic {
        a,
        b,
        matches: match_count,
        fundamental_inliers: mask.iter().filter(|&&is_inlier| is_inlier).count(),
        homography_inliers: h_mask.iter().filter(|&&is_inlier| is_inlier).count(),
        essential_score,
        homography_score,
        score_margin,
        planar,
    };

    (
        match_count,
        Some(VerifiedPair {
            a,
            b,
            inliers,
            epipolar_rmse_px,
            model,
        }),
    )
}

/// Deterministic RANSAC seed for a view pair.
fn pair_seed(a: usize, b: usize) -> u64 {
    (a as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
}

/// Deterministic RANSAC for the fundamental matrix.
///
/// Samples are drawn by iteration index from a linear congruential generator
/// seeded with the pair, so the result is reproducible. The model is scored by
/// the Sampson distance and, once the best model is found, re-fitted up to three
/// times on its own inliers (only accepted when the inlier set does not shrink).
fn robust_fundamental(
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    threshold_px: f64,
    max_iters: usize,
    seed: u64,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    let n = points_a.len();
    if n < 8 || points_b.len() != n {
        return None;
    }
    let threshold = threshold_px * threshold_px;

    let mut best_model: Option<Matrix3<f64>> = None;
    let mut best_mask = vec![false; n];
    let mut best_count = 0usize;
    let mut best_error = f64::INFINITY;

    // Adaptive bound: a deterministic function of the best inlier ratio so far.
    let mut adaptive = max_iters as f64;

    for iteration in 0..max_iters {
        if iteration as f64 >= adaptive {
            break;
        }
        let sample = sample_unique_indices(n, 8, seed ^ (iteration as u64));
        let s1: Vec<Point2<f64>> = sample.iter().map(|&i| points_a[i]).collect();
        let s2: Vec<Point2<f64>> = sample.iter().map(|&i| points_b[i]).collect();
        let Ok(model) = find_fundamental_mat(&s1, &s2) else {
            continue;
        };

        let mut mask = vec![false; n];
        let mut count = 0usize;
        let mut error_sum = 0.0f64;
        for i in 0..n {
            let error = sampson_sq(&model, &points_a[i], &points_b[i]);
            if error <= threshold {
                mask[i] = true;
                count += 1;
                error_sum += error;
            }
        }
        if count == 0 {
            continue;
        }
        let mean_error = error_sum / count as f64;
        if count > best_count || (count == best_count && mean_error < best_error) {
            best_count = count;
            best_error = mean_error;
            best_model = Some(model);
            best_mask = mask;
            let w = count as f64 / n as f64;
            let w8 = w.powi(8);
            if w8 > 0.0 && w8 < 1.0 {
                let k = (1.0 - 0.99f64).ln() / (1.0 - w8).ln();
                adaptive = k.min(max_iters as f64);
            }
        }
    }

    let mut model = best_model?;
    let mut mask = best_mask;
    let mut count = best_count;

    for _ in 0..3 {
        let inlier_a: Vec<Point2<f64>> = (0..n).filter(|&i| mask[i]).map(|i| points_a[i]).collect();
        if inlier_a.len() < 8 {
            break;
        }
        let inlier_b: Vec<Point2<f64>> = (0..n).filter(|&i| mask[i]).map(|i| points_b[i]).collect();
        let Ok(refined) = find_fundamental_mat(&inlier_a, &inlier_b) else {
            break;
        };
        let mut next_mask = vec![false; n];
        let mut next_count = 0usize;
        for i in 0..n {
            if sampson_sq(&refined, &points_a[i], &points_b[i]) <= threshold {
                next_mask[i] = true;
                next_count += 1;
            }
        }
        if next_count >= count {
            model = refined;
            mask = next_mask;
            count = next_count;
        } else {
            break;
        }
    }

    Some((model, mask))
}

/// Deterministic RANSAC for a homography.
///
/// Minimal samples have four points. Scoring uses symmetric forward/inverse
/// transfer error, so a model that maps points to the other side of the image
/// cannot obtain a spuriously good score. Each sampled model is scored on every
/// correspondence; a final fit is attempted from the best inlier set.
fn robust_homography(
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    threshold_px: f64,
    max_iters: usize,
    seed: u64,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    let n = points_a.len();
    if n < 4 || points_b.len() != n || !threshold_px.is_finite() || threshold_px <= 0.0 {
        return None;
    }
    let threshold = threshold_px * threshold_px;

    let mut best_model: Option<Matrix3<f64>> = None;
    let mut best_mask = vec![false; n];
    let mut best_count = 0usize;
    let mut best_error = f64::INFINITY;
    let mut adaptive = max_iters as f64;

    for iteration in 0..max_iters {
        if iteration as f64 >= adaptive {
            break;
        }
        let sample = sample_unique_indices(n, 4, seed ^ (iteration as u64));
        let source: Vec<[f64; 2]> = sample
            .iter()
            .map(|&i| [points_a[i].x, points_a[i].y])
            .collect();
        let destination: Vec<[f64; 2]> = sample
            .iter()
            .map(|&i| [points_b[i].x, points_b[i].y])
            .collect();
        let Some(model) = solve_dlt_homography(&source, &destination) else {
            continue;
        };

        let Some((mask, count, error_sum)) =
            homography_inliers(&model, points_a, points_b, threshold)
        else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let mean_error = error_sum / count as f64;
        if count > best_count || (count == best_count && mean_error < best_error) {
            best_count = count;
            best_error = mean_error;
            best_model = Some(model);
            best_mask = mask;
            let w = count as f64 / n as f64;
            let w4 = w.powi(4);
            if w4 > 0.0 && w4 < 1.0 {
                let k = (1.0 - 0.99f64).ln() / (1.0 - w4).ln();
                adaptive = k.min(max_iters as f64);
            }
        }
    }

    let mut model = best_model?;
    let mut mask = best_mask;
    let mut count = best_count;
    for _ in 0..3 {
        let inlier_a: Vec<[f64; 2]> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_inlier)| is_inlier.then_some([points_a[i].x, points_a[i].y]))
            .collect();
        let inlier_b: Vec<[f64; 2]> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_inlier)| is_inlier.then_some([points_b[i].x, points_b[i].y]))
            .collect();
        let Some(refined) = solve_dlt_homography(&inlier_a, &inlier_b) else {
            break;
        };
        let Some((next_mask, next_count, _)) =
            homography_inliers(&refined, points_a, points_b, threshold)
        else {
            break;
        };
        if next_count >= count {
            model = refined;
            mask = next_mask;
            count = next_count;
        } else {
            break;
        }
    }
    Some((model, mask))
}

fn homography_inliers(
    model: &Matrix3<f64>,
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    threshold_sq: f64,
) -> Option<(Vec<bool>, usize, f64)> {
    let inverse = model.try_inverse()?;
    let mut mask = vec![false; points_a.len()];
    let mut count = 0usize;
    let mut error_sum = 0.0;
    for i in 0..points_a.len() {
        let forward = transfer_error(model, points_a[i], points_b[i]);
        let backward = transfer_error(&inverse, points_b[i], points_a[i]);
        let squared = 0.5 * (forward + backward);
        if squared.is_finite() && squared <= threshold_sq {
            mask[i] = true;
            count += 1;
            error_sum += squared;
        }
    }
    Some((mask, count, error_sum))
}

fn transfer_error(model: &Matrix3<f64>, source: Point2<f64>, target: Point2<f64>) -> f64 {
    let predicted = model * Vector3::new(source.x, source.y, 1.0);
    if !predicted.iter().all(|value| value.is_finite()) || predicted[2].abs() <= 1e-12 {
        return f64::INFINITY;
    }
    let residual = Vector3::new(
        predicted[0] / predicted[2] - target.x,
        predicted[1] / predicted[2] - target.y,
        0.0,
    );
    residual.norm_squared()
}

/// Higher-is-better MSAC score normalized by its ideal value.
///
/// For every correspondence this is `max(0, 1 - e^2 / t^2)`, so a zero-residual
/// inlier contributes one. Dividing by the number of all descriptor matches
/// rewards coverage as well as accuracy, and a model that also explains the
/// outliers remains penalized. SH-style 1-exp(-S/T^2) scores are algebraically
/// equivalent up to scale; because fundamental Sampson and symmetric homography
/// transfer errors are both squared pixels, the same threshold and the same
/// `2 / (t^2 n)` normalization make their difference directly meaningful.
fn normalized_msac_score(
    model: &Matrix3<f64>,
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    mask: &[bool],
    threshold_px: f64,
    homography: bool,
) -> f64 {
    let n = points_a.len().max(points_b.len());
    if n == 0 || points_a.len() != points_b.len() || mask.len() != n || threshold_px <= 0.0 {
        return 0.0;
    }
    let inverse = if homography {
        model.try_inverse()
    } else {
        None
    };
    let threshold_sq = threshold_px * threshold_px;
    let mut score = 0.0;
    for i in 0..n {
        let error = if homography {
            let Some(inverse) = inverse.as_ref() else {
                return 0.0;
            };
            0.5 * (transfer_error(model, points_a[i], points_b[i])
                + transfer_error(inverse, points_b[i], points_a[i]))
        } else {
            sampson_sq(model, &points_a[i], &points_b[i])
        };
        if mask[i] {
            score += (1.0 - error / threshold_sq).max(0.0);
        }
    }
    // Same normalisation as `essential_model_score`: MSAC is the mean of
    // (1 - err/threshold^2) over the correspondences, so both models are scored
    // on the same 0..1 scale. This function previously divided by
    // `threshold_sq` as well, which inflated it by 1/threshold^2 and made every
    // pair look planar regardless of the data.
    (score / n as f64).clamp(0.0, 1.0)
}

/// Calculate the normalized essential score on calibrated pixel correspondences.
fn essential_model_score(
    essential: &Matrix3<f64>,
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    threshold_px: f64,
) -> f64 {
    let k_inv = intrinsics.inverse_matrix();
    let focal = 0.5 * (intrinsics.fx + intrinsics.fy).max(1e-12);
    let threshold = threshold_px / focal;
    if points_a.len() != points_b.len() || points_a.is_empty() {
        return 0.0;
    }
    let normalize = |point: Point2<f64>| {
        let p = k_inv * Vector3::new(point.x, point.y, 1.0);
        Point2::new(p[0] / p[2], p[1] / p[2])
    };
    let threshold_sq = threshold * threshold;
    let mut score = 0.0;
    for (a, b) in points_a.iter().zip(points_b.iter()) {
        let a = normalize(*a);
        let b = normalize(*b);
        let error = sampson_sq(essential, &a, &b);
        if error <= threshold_sq {
            score += 1.0 - error / threshold_sq;
        }
    }
    (score / points_a.len() as f64).clamp(0.0, 1.0)
}

/// Calculate the normalized homography score on pixel correspondences.
fn normalized_homography_score(
    model: &Matrix3<f64>,
    points_a: &[Point2<f64>],
    points_b: &[Point2<f64>],
    threshold_px: f64,
) -> f64 {
    if points_a.len() != points_b.len() || points_a.is_empty() {
        return 0.0;
    }
    let Some(inverse) = model.try_inverse() else {
        return 0.0;
    };
    let threshold_sq = threshold_px * threshold_px;
    if threshold_sq <= 0.0 || !threshold_sq.is_finite() {
        return 0.0;
    }
    let mut score = 0.0;
    for (a, b) in points_a.iter().zip(points_b.iter()) {
        let forward = transfer_error(model, *a, *b);
        let backward = transfer_error(&inverse, *b, *a);
        let error = 0.5 * (forward + backward);
        if error <= threshold_sq {
            score += 1.0 - error / threshold_sq;
        }
    }
    (score / points_a.len() as f64).clamp(0.0, 1.0)
}

/// Sampson distance, squared, of a correspondonce under `f`.
fn sampson_sq(f: &Matrix3<f64>, p1: &Point2<f64>, p2: &Point2<f64>) -> f64 {
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let x2 = Vector3::new(p2.x, p2.y, 1.0);
    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;
    let numerator = x2.dot(&fx1);
    let denominator = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if denominator <= 1e-18 {
        f64::INFINITY
    } else {
        numerator * numerator / denominator
    }
}

/// Deterministic sample of `k` distinct indices in `[0, n)`.
///
/// Uses the same linear congruential construction `cv-calib3d`'s PnP RANSAC uses.
/// If the generator cannot cover the index range quickly enough the remaining
/// slots are filled by ascending index, so this always terminates.
fn sample_unique_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    if k >= n {
        return (0..n).collect();
    }
    let mut out = Vec::with_capacity(k);
    let mut used = vec![false; n];
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut attempts = 0usize;
    let budget = 64 * n + 64;
    while out.len() < k && attempts < budget {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = (state as usize) % n;
        if !used[idx] {
            used[idx] = true;
            out.push(idx);
        }
        attempts += 1;
    }
    if out.len() < k {
        for (idx, taken) in used.iter().enumerate() {
            if !taken {
                out.push(idx);
                if out.len() == k {
                    break;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tracks
// ---------------------------------------------------------------------------

/// Disjoint-set (union-find) with union by rank and path compression.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut current = x;
        while self.parent[current] != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

/// Merge verified correspondences into tracks of `(view, keypoint)` observations.
///
/// Node id for `(view, keypoint)` is `offset[view] + keypoint`. Nodes are scanned
/// in ascending id order, so a track is created when its smallest node is first
/// seen and its observations come out ascending by view. Tracks with fewer than
/// two observations are dropped (they cannot be triangulated).
fn build_tracks(views: &[View], verified: &[VerifiedPair]) -> Vec<Vec<(usize, usize)>> {
    let total: usize = views.iter().map(View::len).sum();
    if total == 0 || verified.is_empty() {
        return Vec::new();
    }

    let mut offset = Vec::with_capacity(views.len() + 1);
    offset.push(0usize);
    for view in views {
        offset.push(offset[offset.len() - 1] + view.len());
    }

    let mut node_view = vec![0usize; total];
    let mut node_keypoint = vec![0usize; total];
    for (view_index, view) in views.iter().enumerate() {
        for keypoint in 0..view.len() {
            node_view[offset[view_index] + keypoint] = view_index;
            node_keypoint[offset[view_index] + keypoint] = keypoint;
        }
    }

    let mut uf = UnionFind::new(total);
    for pair in verified {
        for &(ka, kb) in &pair.inliers {
            uf.union(offset[pair.a] + ka, offset[pair.b] + kb);
        }
    }

    let mut track_of_root = vec![usize::MAX; total];
    let mut tracks: Vec<Vec<(usize, usize)>> = Vec::new();
    for node in 0..total {
        let root = uf.find(node);
        let track = if track_of_root[root] == usize::MAX {
            track_of_root[root] = tracks.len();
            tracks.push(Vec::new());
            tracks.len() - 1
        } else {
            track_of_root[root]
        };
        let view = node_view[node];
        // Nodes are visited in ascending order, so the first node of each view in
        // this track has the lowest keypoint index; later ones collapse into it.
        if tracks[track].last().map(|&(v, _)| v) == Some(view) {
            continue;
        }
        tracks[track].push((view, node_keypoint[node]));
    }

    tracks.retain(|track| track.len() >= 2);
    tracks
}

// ---------------------------------------------------------------------------
// Seed selection and initialization
// ---------------------------------------------------------------------------

/// The chosen seed pair and the recovered relative pose.
#[derive(Debug, Clone, Copy)]
struct SeedChoice {
    a: usize,
    b: usize,
    /// World-to-camera pose of `b` with the world frame set to camera `a`.
    pose_b: Pose,
}

/// Rank the verified pairs as seed candidates, best first.
///
/// Every verified pair with at least `min_seed_inliers` inliers is scored by how
/// many of its correspondences triangulate into a finite point that is in front
/// of both cameras, has at least `min_parallax_deg` parallax and reprojects
/// within `max_reproj_px` in both. Candidates are ranked by that score and the
/// list is truncated to `seed_candidates`.
///
/// Ranking by *matches* would be a trap here: the pairs with the most inliers are
/// the near-duplicate pairs with the smallest baseline, and those are exactly the
/// pairs whose essential-matrix decomposition is worst conditioned, because the
/// translation direction is only observable through parallax. Ranking by cleanly
/// triangulating geometry prefers pairs with real baseline. Candidates scoring at
/// least `min_seed_points` come first; if none reaches that bar the rest are
/// returned anyway, so the caller can decide.
///
/// Even so, the score is measured *in the candidate's own gauge*, so a wrong
/// essential-matrix decomposition can look self-consistent. That is why
/// [`map_views`] runs the full incremental registration for several candidates
/// and keeps the one that registers the most views.
fn seed_candidates(
    verified: &[VerifiedPair],
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> (Vec<SeedChoice>, Vec<(usize, usize, usize)>) {
    let mut scored: Vec<(usize, SeedChoice)> = Vec::new();
    let mut diagnostics: Vec<(usize, usize, usize)> = Vec::new();
    let mut forced: Option<SeedChoice> = None;
    for pair in verified {
        // A pair may still contribute its robust matches to track building, but
        // its essential decomposition is not trusted when H significantly beats E.
        if pair.model.planar {
            continue;
        }
        if pair.inliers.len() < config.min_seed_inliers {
            continue;
        }
        let Some((good, choice)) = seed_from_pair(pair, views, intrinsics, config) else {
            continue;
        };
        diagnostics.push((pair.a, pair.b, good));
        if config.seed_pair == Some((pair.a, pair.b)) {
            forced = Some(choice);
        }
        if good > 0 {
            scored.push((good, choice));
        }
    }

    // Fallback: if model selection rejected every pair, the map cannot start at
    // all. On a shallow capture of a largely planar scene that is the common
    // case — measured on ETH3D courtyard, 13 of 13 verified pairs were flagged
    // planar, which excluded the whole pair graph and left a 6-view map out of
    // 23. Rather than never seed from a planar-flagged pair, fall back to the
    // best-scoring one only when nothing else is available: the essential
    // decomposition is checked by triangulation and cheirality downstream, so a
    // bad one is rejected there, whereas refusing to try means no map at all.
    if scored.is_empty() {
        for pair in verified {
            if pair.inliers.len() < config.min_seed_inliers {
                continue;
            }
            let Some((good, choice)) = seed_from_pair(pair, views, intrinsics, config) else {
                continue;
            };
            diagnostics.push((pair.a, pair.b, good));
            if good > 0 {
                scored.push((good, choice));
            }
        }
        if !scored.is_empty() {
            scored.sort_by(|left, right| right.0.cmp(&left.0));
            scored.truncate(1);
        }
    }

    // A forced seed may be a pair below `min_seed_inliers`; build it anyway.
    if let Some(requested) = config.seed_pair {
        if forced.is_none() {
            if let Some(pair) = verified.iter().find(|pair| {
                (pair.a, pair.b) == requested && !pair.model.planar && pair.inliers.len() >= 8
            }) {
                if let Some((good, choice)) = seed_from_pair(pair, views, intrinsics, config) {
                    diagnostics.push((pair.a, pair.b, good));
                    forced = Some(choice);
                }
            }
        }
    }

    let qualified: Vec<(usize, SeedChoice)> = scored
        .iter()
        .filter(|(good, _)| *good >= config.min_seed_points)
        .copied()
        .collect();
    let mut chosen = if qualified.is_empty() {
        scored
    } else {
        qualified
    };
    // Best geometry first; ascending view indices break ties.
    chosen.sort_by(|x, y| {
        y.0.cmp(&x.0)
            .then(x.1.a.cmp(&y.1.a))
            .then(x.1.b.cmp(&y.1.b))
    });
    let mut ordered: Vec<SeedChoice> = chosen
        .into_iter()
        .map(|(_, choice)| choice)
        .take(config.seed_candidates.max(1))
        .collect();

    if let Some(forced) = forced {
        // Put the requested seed first and drop the duplicate.
        ordered.retain(|choice| (choice.a, choice.b) != (forced.a, forced.b));
        ordered.insert(0, forced);
    }

    (ordered, diagnostics)
}

/// Score one verified pair as a seed and recover its relative pose.
///
/// The score is the number of correspondences that triangulate into a finite
/// point in front of both cameras, with at least `min_parallax_deg` parallax and
/// at most `max_reproj_px` reprojection error in both. The score is measured *in
/// the pair's own gauge*, so a wrong essential-matrix decomposition can still
/// score highly; see [`map_views`].
fn seed_from_pair(
    pair: &VerifiedPair,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> Option<(usize, SeedChoice)> {
    let identity = Pose::identity();
    let projection_a = projection_matrix(&identity, intrinsics);
    let centre_a = identity.inverse().translation;

    let va = &views[pair.a];
    let vb = &views[pair.b];
    let points_a: Vec<Point2<f64>> = pair
        .inliers
        .iter()
        .map(|&(ka, _)| va.keypoints[ka].pt())
        .collect();
    let points_b: Vec<Point2<f64>> = pair
        .inliers
        .iter()
        .map(|&(_, kb)| vb.keypoints[kb].pt())
        .collect();

    let essential = find_essential_mat(&points_a, &points_b, intrinsics).ok()?;
    let pose_b = recover_pose_from_essential(&essential, &points_a, &points_b, intrinsics).ok()?;
    if !pose_is_finite(&pose_b) {
        return None;
    }

    let projection_b = projection_matrix(&pose_b, intrinsics);
    let centre_b = pose_b.inverse().translation;
    let triangulated =
        triangulate_points(&projection_a, &projection_b, &points_a, &points_b).ok()?;

    let mut good = 0usize;
    for (i, point) in triangulated.iter().enumerate() {
        if !point_is_finite(point) || point.z <= 0.0 {
            continue;
        }
        let in_b = pose_b.rotation * point.coords + pose_b.translation;
        if in_b[2] <= 0.0 {
            continue;
        }
        if parallax_deg(point, &centre_a, &centre_b) < config.min_parallax_deg {
            continue;
        }
        let reprojected_a = intrinsics.project(point);
        let reprojected_b = intrinsics.project(&Point3::from(in_b));
        let error_a = (reprojected_a - points_a[i]).norm();
        let error_b = (reprojected_b - points_b[i]).norm();
        if error_a <= config.max_reproj_px && error_b <= config.max_reproj_px {
            good += 1;
        }
    }

    Some((
        good,
        SeedChoice {
            a: pair.a,
            b: pair.b,
            pose_b,
        },
    ))
}

// ---------------------------------------------------------------------------
// Triangulation
// ---------------------------------------------------------------------------

/// Triangulate every track that has no 3D point yet and at least two registered
/// observations. Returns how many points were added.
fn triangulate_new_tracks(
    est: &mut Est,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> usize {
    let mut added = 0usize;
    for track in 0..est.track_obs.len() {
        if est.track_point[track].is_some() {
            continue;
        }
        let registered: Vec<(usize, usize)> = est.track_obs[track]
            .iter()
            .copied()
            .filter(|&(view, _)| est.cam_of_view[view].is_some())
            .collect();
        if registered.len() < 2 {
            continue;
        }
        let Some((point, survivors)) =
            triangulate_track(&registered, est, views, intrinsics, config)
        else {
            continue;
        };

        let landmark = est.points.len();
        est.points.push(point);
        est.point_obs.push(survivors);
        est.point_tracks.push(track);
        est.track_point[track] = Some(landmark);
        added += 1;
    }
    added
}

/// Re-triangulate every existing landmark from the now-wider registered baseline,
/// returning how many landmarks were updated.
///
/// A landmark is first triangulated as soon as two of its views register, so it
/// initially carries the depth error of that (often short) baseline. Re-running
/// the triangulation after each registration uses the widest registered baseline
/// available so far and refreshes the observation set. A landmark whose
/// re-triangulation fails keeps its previous point and observations.
fn retriangulate(
    est: &mut Est,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> usize {
    let mut updated = 0usize;
    for landmark in 0..est.points.len() {
        let track = est.point_tracks[landmark];
        let registered: Vec<(usize, usize)> = est.track_obs[track]
            .iter()
            .copied()
            .filter(|&(view, _)| est.cam_of_view[view].is_some())
            .collect();
        if registered.len() < 2 {
            continue;
        }
        if let Some((point, survivors)) =
            triangulate_track(&registered, est, views, intrinsics, config)
        {
            est.points[landmark] = point;
            est.point_obs[landmark] = survivors;
            updated += 1;
        }
    }
    updated
}

/// Triangulate one track from a set of its registered observations.
///
/// The point is triangulated from the two registered views whose camera centres
/// are farthest apart (ties keep the earliest pair, because the scan order is
/// deterministic), then projected into every registered view of the track.
/// Observations that lie behind their camera or reproject more than
/// `max_reproj_px` away are dropped; a track needs at least two survivors to
/// yield a point.
fn triangulate_track(
    registered: &[(usize, usize)],
    est: &Est,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> Option<(Point3<f64>, Vec<(usize, usize)>)> {
    if registered.len() < 2 {
        return None;
    }

    // Widest-baseline pair among the registered observations.
    let mut best_pair = (0usize, 1usize);
    let mut best_baseline = -1.0f64;
    for p in 0..registered.len() {
        let Some(centre_p) = est
            .pose_of(registered[p].0)
            .map(|pose| pose.inverse().translation)
        else {
            continue;
        };
        for q in p + 1..registered.len() {
            let Some(centre_q) = est
                .pose_of(registered[q].0)
                .map(|pose| pose.inverse().translation)
            else {
                continue;
            };
            let baseline = (centre_p - centre_q).norm();
            if baseline > best_baseline {
                best_baseline = baseline;
                best_pair = (p, q);
            }
        }
    }

    let (fa, ka) = registered[best_pair.0];
    let (fb, kb) = registered[best_pair.1];
    let pose_a = est.pose_of(fa)?;
    let pose_b = est.pose_of(fb)?;
    let projection_a = projection_matrix(&pose_a, intrinsics);
    let projection_b = projection_matrix(&pose_b, intrinsics);
    let point_a = views[fa].keypoints[ka].pt();
    let point_b = views[fb].keypoints[kb].pt();

    let triangulated =
        triangulate_points(&projection_a, &projection_b, &[point_a], &[point_b]).ok()?;
    let point = *triangulated.first()?;
    if !point_is_finite(&point) || point.z <= 0.0 {
        return None;
    }
    let depth_in_b = pose_b.rotation * point.coords + pose_b.translation;
    if depth_in_b[2] <= 0.0 {
        return None;
    }
    let centre_a = pose_a.inverse().translation;
    let centre_b = pose_b.inverse().translation;
    if parallax_deg(&point, &centre_a, &centre_b) < config.min_parallax_deg {
        return None;
    }

    let mut survivors: Vec<(usize, usize)> = Vec::with_capacity(registered.len());
    for &(view, keypoint) in registered {
        let Some(pose) = est.pose_of(view) else {
            continue;
        };
        let Some(reprojected) = project_world(&pose, intrinsics, &point) else {
            continue;
        };
        let observed = views[view].keypoints[keypoint].pt();
        if (reprojected - observed).norm() <= config.max_reproj_px {
            survivors.push((view, keypoint));
        }
    }
    if survivors.len() < 2 {
        return None;
    }

    Some((point, survivors))
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

/// The deterministic camera and landmark subset used for local BA.
///
/// The indices refer to the mapper's `Est`, rather than to a sub-state. Keeping
/// those indices makes the writeback boundary explicit: cameras and landmarks
/// omitted here are never assigned to by local refinement.
struct LocalBaProblem {
    /// The new camera first, followed by its selected co-visible neighbours.
    cameras: Vec<usize>,
    /// Original `Est::points` indices observed by at least one selected camera.
    landmarks: Vec<usize>,
}

/// Construct the local BA problem for a newly registered view.
///
/// A co-visible camera shares at least `local_ba_min_overlap` surviving
/// landmarks with the new camera. Candidates are sorted by shared-landmark count
/// descending and camera index ascending, then the first
/// `local_ba_window - 1` are selected. All collections are vectors in
/// registration/landmark order, so selection does not depend on hash-map
/// iteration.
fn local_ba_problem(est: &Est, new_view: usize, config: &MapperConfig) -> Option<LocalBaProblem> {
    if config.local_ba_window < 2 {
        return None;
    }
    let new_camera = est.cam_of_view.get(new_view).copied().flatten()?;

    // Repeated observation pairs cannot occur in a valid track, so the inner loop
    // stays linear and follows fixed landmark/camera order.
    let mut shared_counts = vec![0usize; est.poses.len()];
    for observations in &est.point_obs {
        let mut shares_new = false;
        for &(view, _) in observations {
            let camera = est.cam_of_view[view]?;
            if camera == new_camera {
                shares_new = true;
            }
        }
        if !shares_new {
            continue;
        }
        for &(view, _) in observations {
            let camera = est.cam_of_view[view]?;
            if camera != new_camera {
                shared_counts[camera] += 1;
            }
        }
    }

    let mut neighbours: Vec<(usize, usize)> = shared_counts
        .iter()
        .enumerate()
        .filter_map(|(camera, &count)| {
            (count > 0 && camera != new_camera && count >= config.local_ba_min_overlap)
                .then_some((count, camera))
        })
        .collect();
    neighbours.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));

    let mut cameras = Vec::with_capacity(config.local_ba_window.min(est.poses.len()));
    cameras.push(new_camera);
    cameras.extend(
        neighbours
            .into_iter()
            .take(config.local_ba_window - 1)
            .map(|(_, camera)| camera),
    );
    if cameras.len() < 2 {
        return None;
    }

    let mut camera_to_local = vec![None; est.poses.len()];
    for (local, &camera) in cameras.iter().enumerate() {
        camera_to_local[camera] = Some(local);
    }
    // Landmarks visible to the selected cameras, ranked by how many of those
    // cameras observe each one. Without a cap this set grows with the whole map,
    // so a "local" solve became progressively more expensive as the
    // reconstruction grew (measured: 400 points at the start of a 60-view run,
    // 841 by the end, 79 ms -> 174 ms per call, ~90 calls). A local bundle
    // adjustment is meant to be local: the landmarks that anchor the new view
    // to its neighbours are the ones the adjustment is for. Ties are broken by
    // ascending landmark index so the selection is deterministic.
    let mut ranked: Vec<(usize, usize)> = est
        .point_obs
        .iter()
        .enumerate()
        .filter_map(|(landmark, observations)| {
            let count = observations
                .iter()
                .filter(|&&(view, _)| {
                    est.cam_of_view
                        .get(view)
                        .copied()
                        .flatten()
                        .is_some_and(|camera| camera_to_local[camera].is_some())
                })
                .count();
            (count > 0).then_some((count, landmark))
        })
        .collect();
    ranked.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    let landmarks: Vec<usize> = ranked
        .into_iter()
        .take(config.local_ba_max_points)
        .map(|(_, landmark)| landmark)
        .collect();

    (!landmarks.is_empty()).then_some(LocalBaProblem { cameras, landmarks })
}

/// Refine the local BA problem for `new_view` and write back only its subset.
///
/// The full state is still passed through the same settings as global BA. The
/// resulting sub-state is accepted only when its dimensions and all parameters
/// are valid. No filtering, retraction, or removal occurs here: a local landmark
/// is optimised using only observations from the selected cameras, and only the
/// selected camera poses and landmark positions are written back.
fn refine_local(
    est: &mut Est,
    new_view: usize,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> bool {
    let Some(problem) = local_ba_problem(est, new_view, config) else {
        return false;
    };

    let mut camera_to_local = vec![None; est.poses.len()];
    for (local, &camera) in problem.cameras.iter().enumerate() {
        camera_to_local[camera] = Some(local);
    }
    let mut state = SfMState::new(*intrinsics);
    for &camera in &problem.cameras {
        state.add_camera(est.poses[camera]);
    }
    for &landmark in &problem.landmarks {
        let observations = est.point_obs[landmark]
            .iter()
            .filter_map(|&(view, keypoint)| {
                let camera = est.cam_of_view[view]?;
                let local = camera_to_local[camera]?;
                Some((local, views[view].keypoints[keypoint].pt()))
            })
            .collect();
        state.add_landmark(est.points[landmark], observations);
    }

    bundle_adjust(&mut state, &mapper_ba_config(config));
    if state.cameras.len() != problem.cameras.len()
        || state.landmarks.len() != problem.landmarks.len()
        || !state.cameras.iter().all(pose_is_finite)
        || state
            .landmarks
            .iter()
            .any(|landmark| !point_is_finite(&landmark.position))
    {
        return false;
    }

    for (local, &camera) in problem.cameras.iter().enumerate() {
        est.poses[camera] = state.cameras[local];
    }
    for (local, &landmark) in problem.landmarks.iter().enumerate() {
        est.points[landmark] = state.landmarks[local].position;
    }
    true
}

fn mapper_ba_config(config: &MapperConfig) -> BundleAdjustmentConfig {
    BundleAdjustmentConfig {
        max_iterations: config.ba_max_iterations,
        convergence_threshold: 1e-6,
        lambda: 0.001,
        use_sparsity: config.ba_use_sparsity,
        robust_kernel: config.ba_robust_kernel,
    }
}

/// Run the crate's bundle adjustment on the current state.
///
/// Returns `true` when the refined parameters were accepted. A call that produces
/// a non-finite pose or point is discarded and the previous state is kept.
fn refine(
    est: &mut Est,
    views: &[View],
    intrinsics: &CameraIntrinsics,
    config: &MapperConfig,
) -> bool {
    if est.poses.len() < 2 || est.points.is_empty() {
        return false;
    }

    let mut state = SfMState::new(*intrinsics);
    for pose in &est.poses {
        state.add_camera(*pose);
    }
    for landmark in 0..est.points.len() {
        let observations: Vec<(usize, Point2<f64>)> = est.point_obs[landmark]
            .iter()
            .filter_map(|&(view, keypoint)| {
                est.cam_of_view[view].map(|camera| (camera, views[view].keypoints[keypoint].pt()))
            })
            .collect();
        state.add_landmark(est.points[landmark], observations);
    }

    bundle_adjust(&mut state, &mapper_ba_config(config));

    let poses_ok = state.cameras.iter().all(pose_is_finite)
        && state
            .landmarks
            .iter()
            .all(|landmark| landmark.position.coords.iter().all(|v| v.is_finite()));
    if !poses_ok
        || state.cameras.len() != est.poses.len()
        || state.landmarks.len() != est.points.len()
    {
        return false;
    }

    for (camera, pose) in state.cameras.iter().enumerate() {
        est.poses[camera] = *pose;
    }
    for (landmark, point) in state.landmarks.iter().enumerate() {
        est.points[landmark] = point.position;
    }
    true
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Pixel projection matrix `K * [R | t]` for a world-to-camera pose.
fn projection_matrix(pose_cw: &Pose, intrinsics: &CameraIntrinsics) -> Matrix3x4<f64> {
    let rt = pose_cw.matrix().fixed_view::<3, 4>(0, 0).into_owned();
    intrinsics.matrix() * rt
}

/// Project a world point through a world-to-camera pose, or `None` when it is
/// behind (or exactly on) the image plane.
fn project_world(
    pose_cw: &Pose,
    intrinsics: &CameraIntrinsics,
    point: &Point3<f64>,
) -> Option<Point2<f64>> {
    let camera = pose_cw.rotation * point.coords + pose_cw.translation;
    if camera[2] <= 1e-9 {
        return None;
    }
    Some(intrinsics.project(&Point3::from(camera)))
}

/// Parallax of a triangulated point as seen from two camera centres, in degrees.
fn parallax_deg(point: &Point3<f64>, centre_a: &Vector3<f64>, centre_b: &Vector3<f64>) -> f64 {
    let ray_a = point.coords - centre_a;
    let ray_b = point.coords - centre_b;
    let norm_a = ray_a.norm();
    let norm_b = ray_b.norm();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    let cosine = (ray_a.dot(&ray_b) / (norm_a * norm_b)).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

/// Whether every entry of the pose is finite.
fn pose_is_finite(pose: &Pose) -> bool {
    pose.translation.iter().all(|v| v.is_finite())
        && pose.rotation_matrix().iter().all(|v| v.is_finite())
}

/// Whether every coordinate of the point is finite.
fn point_is_finite(point: &Point3<f64>) -> bool {
    point.coords.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Descriptor;

    /// Build a synthetic scene: `n_views` cameras looking at a 3D point cloud,
    /// with each view's "descriptor" being a unique id per 3D point so that the
    /// matcher links the same point across views.
    ///
    /// Returns the views and the ground-truth camera-to-world poses.
    fn synthetic_scene(n_views: usize, n_points: usize) -> (Vec<View>, Vec<Pose>) {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };

        // 3D points in a box in front of the cameras (world == camera 0 frame).
        let points: Vec<Point3<f64>> = (0..n_points)
            .map(|_| Point3::new(next() * 0.8, next() * 0.6, 1.5 + next() * 0.6))
            .collect();

        // Cameras move along +x in small steps, looking roughly forward.
        let mut views = Vec::with_capacity(n_views);
        let mut ground_truth = Vec::with_capacity(n_views);
        for view in 0..n_views {
            let t = 0.02 * view as f64;
            let rotation =
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.01 * view as f64);
            let pose_wc = Pose::from_quat_translation(rotation, Vector3::new(t, 0.0, 0.0));
            let pose_cw = pose_wc.inverse();

            let mut keypoints = Vec::with_capacity(n_points);
            let mut descriptors = Descriptors::with_capacity(n_points);
            for (id, point) in points.iter().enumerate() {
                let camera = pose_cw.rotation * point.coords + pose_cw.translation;
                if camera[2] <= 1e-6 {
                    continue;
                }
                let projected = intrinsics.project(&Point3::from(camera));
                if projected.x < 0.0
                    || projected.y < 0.0
                    || projected.x >= 640.0
                    || projected.y >= 480.0
                {
                    continue;
                }
                // Descriptor bytes encode the point id, so descriptor distance is
                // zero exactly between observations of the same point.
                let mut data = vec![0u8; 32];
                data[0] = (id & 0xFF) as u8;
                data[1] = ((id >> 8) & 0xFF) as u8;
                keypoints.push(KeyPoint::new(projected.x, projected.y));
                descriptors.push(Descriptor::new(
                    data,
                    KeyPoint::new(projected.x, projected.y),
                ));
            }
            views.push(View::new(keypoints, descriptors, 640, 480));
            ground_truth.push(pose_wc);
        }
        (views, ground_truth)
    }

    /// Build exact descriptor matches for two fixed cameras. A descriptor stores
    /// its point id, so every point has exactly one zero-distance match in the
    /// other view and there is no random sampling or numerical noise in the test.
    fn exact_views(points: &[Point3<f64>], cameras: &[Pose]) -> Vec<View> {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cameras
            .iter()
            .map(|pose_cw| {
                let mut keypoints = Vec::with_capacity(points.len());
                let mut descriptors = Descriptors::with_capacity(points.len());
                for (id, point) in points.iter().enumerate() {
                    let camera = pose_cw.rotation * point.coords + pose_cw.translation;
                    assert!(camera[2] > 0.0, "synthetic point is behind its camera");
                    let projected = intrinsics.project(&Point3::from(camera));
                    assert!(
                        (0.0..640.0).contains(&projected.x) && (0.0..480.0).contains(&projected.y),
                        "synthetic point is outside the image: {projected:?}"
                    );
                    keypoints.push(KeyPoint::new(projected.x, projected.y));
                    let mut data = vec![0u8; 32];
                    data[0] = (id & 0xFF) as u8;
                    data[1] = ((id >> 8) & 0xFF) as u8;
                    descriptors.push(Descriptor::new(
                        data,
                        KeyPoint::new(projected.x, projected.y),
                    ));
                }
                View::new(keypoints, descriptors, 640, 480)
            })
            .collect()
    }

    fn model_selection_config() -> MapperConfig {
        MapperConfig {
            min_pair_matches: 8,
            min_pair_inliers: 8,
            f_ransac_threshold_px: 1.5,
            f_ransac_iters: 500,
            h_ransac_threshold_px: 1.5,
            h_ransac_iters: 500,
            planar_score_margin: 0.05,
            ba_every: 0,
            ba_final: false,
            ..MapperConfig::default()
        }
    }

    #[test]
    fn planar_pair_is_degenerate_but_keeps_track_matches() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let cameras = [
            Pose::identity(),
            Pose::from_quat_translation(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(0.30, 0.0, 0.0),
            ),
        ];
        let points: Vec<Point3<f64>> = (0..8)
            .flat_map(|y| {
                (0..8).map(move |x| Point3::new(-1.4 + 0.4 * x as f64, -1.05 + 0.3 * y as f64, 4.0))
            })
            .collect();
        let views = exact_views(&points, &cameras);
        let config = model_selection_config();

        let pair = verify_pair(0, 1, &views, &intrinsics, &config)
            .1
            .expect("the exact planar pair passes fundamental verification");
        assert!(pair.model.planar, "planar diagnostic: {:?}", pair.model);
        assert!(
            pair.model.homography_score > pair.model.essential_score + config.planar_score_margin,
            "H should significantly beat E: {:?}",
            pair.model
        );

        // Model selection is seed-only: the same inliers remain in the covisibility
        // graph and can become a multi-view track.
        let tracks = build_tracks(&views, std::slice::from_ref(&pair));
        assert_eq!(tracks.len(), points.len());
        assert!(tracks
            .iter()
            .all(|track| track.len() == 2 && track[0].0 == 0 && track[1].0 == 1));
        let (seeds, _) = seed_candidates(std::slice::from_ref(&pair), &views, &intrinsics, &config);
        assert!(
            seeds.is_empty(),
            "a planar pair must not initialize the map"
        );
    }

    #[test]
    fn volumetric_pair_with_significant_baseline_is_not_degenerate() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let cameras = [
            Pose::identity(),
            Pose::from_quat_translation(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(0.70, 0.0, 0.0),
            ),
        ];
        // Depth 4-8 m: close enough that a 500 px focal length keeps every point
        // inside a 640x480 image in both views, with real depth variation so the
        // scene is not a homography. The 0.70 m baseline gives a ~10 degree
        // parallax at 4 m, well above the degenerate regime.
        let points: Vec<Point3<f64>> = [4.0, 6.0, 8.0]
            .into_iter()
            .flat_map(|z| {
                [-0.36, -0.12, 0.12, 0.36]
                    .into_iter()
                    .flat_map(move |x| [-0.24, 0.0, 0.24].map(move |y| Point3::new(x, y, z)))
            })
            .collect();
        let views = exact_views(&points, &cameras);
        let config = model_selection_config();

        let pair = verify_pair(0, 1, &views, &intrinsics, &config)
            .1
            .expect("the exact volumetric pair passes fundamental verification");
        assert!(
            !pair.model.planar,
            "volumetric diagnostic: {:?}",
            pair.model
        );
        assert!(
            pair.model.essential_score > pair.model.homography_score,
            "E should explain the volumetric data better: {:?}",
            pair.model
        );
    }

    #[test]
    fn empty_input_does_not_panic() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let mapping = map_views(&[], &intrinsics, &MapperConfig::default());
        assert!(mapping.reconstruction.is_empty());
        assert_eq!(mapping.report.registered, 0);
        assert_eq!(mapping.report.registration_rate, 0.0);
        assert!(mapping.report.outcomes.is_empty());
    }

    #[test]
    fn views_without_features_do_not_seed() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let views: Vec<View> = (0..4)
            .map(|_| View::new(Vec::new(), Descriptors::new(), 640, 480))
            .collect();
        let mapping = map_views(&views, &intrinsics, &MapperConfig::default());
        assert!(mapping.reconstruction.is_empty());
        assert!(mapping.report.seed.is_none());
        assert_eq!(mapping.report.registered, 0);
        assert_eq!(mapping.report.registration_rate, 0.0);
        assert_eq!(mapping.report.outcomes.len(), 4);
        assert!(mapping
            .report
            .outcomes
            .iter()
            .all(|outcome| outcome.failure == Some(RegistrationFailure::Unreachable)));
    }

    #[test]
    fn recovers_synthetic_trajectory_up_to_similarity() {
        let (views, ground_truth) = synthetic_scene(8, 120);
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let config = MapperConfig {
            min_pair_matches: 8,
            min_pair_inliers: 8,
            min_seed_inliers: 20,
            ba_every: 0,
            ba_final: false,
            ..MapperConfig::default()
        };
        let mapping = map_views(&views, &intrinsics, &config);

        assert!(
            mapping.report.registered >= views.len() - 1,
            "expected nearly every view to register, got {}/{}",
            mapping.report.registered,
            views.len()
        );
        assert!(!mapping.reconstruction.points.is_empty());
        assert!(mapping.report.mean_track_length >= 2.0);

        // Cameras come out ascending by view index.
        let indices = mapping.reconstruction.view_indices();
        assert!(indices.windows(2).all(|w| w[0] < w[1]));

        // The reconstruction is only defined up to a similarity; with exact
        // synthetic observations the Sim(3)-aligned camera-centre RMSE should be
        // negligible relative to the trajectory span.
        let centres = mapping.reconstruction.camera_centers();
        let span = centres
            .iter()
            .map(|c| (*c - centres[0]).norm())
            .fold(0.0f64, f64::max);
        assert!(span > 0.0, "camera centres should not all coincide");

        let unit = vec![nalgebra::UnitQuaternion::identity(); centres.len()];
        let est = cv_eval::Trajectory::from_positions_and_quaternions(&centres, &unit);
        let gt_centres: Vec<Vector3<f64>> = mapping
            .reconstruction
            .view_indices()
            .iter()
            .map(|&view| ground_truth[view].translation)
            .collect();
        let unit_gt = vec![nalgebra::UnitQuaternion::identity(); gt_centres.len()];
        let gt = cv_eval::Trajectory::from_positions_and_quaternions(&gt_centres, &unit_gt);
        let ate = est.ate(&gt, cv_eval::Alignment::Sim3);
        assert!(
            ate.rmse < 0.02 * span.max(1e-9),
            "Sim(3) camera-centre RMSE {} too large for trajectory span {}",
            ate.rmse,
            span
        );
    }

    #[test]
    fn matching_pair_selection_includes_only_expected_pairs() {
        let views: Vec<View> = (0..5)
            .map(|_| View::new(Vec::new(), Descriptors::new(), 10, 10))
            .collect();
        let config = MapperConfig {
            pair_selection: PairSelection::Sequential { window: 2 },
            ..MapperConfig::default()
        };
        let pairs = select_pairs(&views, &config);
        assert_eq!(
            pairs,
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
    }

    #[test]
    fn sampling_is_deterministic_and_unique() {
        let a = sample_unique_indices(50, 8, 7);
        let b = sample_unique_indices(50, 8, 7);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 8);
        assert_eq!(sample_unique_indices(3, 5, 1), vec![0, 1, 2]);
    }

    #[test]
    fn local_ba_writes_back_only_the_selected_subset() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let poses = [
            Pose::identity(),
            Pose::from_quat_translation(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(0.05, 0.0, 0.0),
            ),
            Pose::from_quat_translation(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(-0.03, 0.01, 0.0),
            ),
            Pose::from_quat_translation(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(-0.07, 0.02, 0.0),
            ),
        ];
        let world_points = [
            Point3::new(-0.45, -0.20, 4.0),
            Point3::new(0.20, -0.10, 4.2),
            Point3::new(0.35, 0.22, 4.1),
            Point3::new(-0.20, 0.30, 4.3),
            // Seen only by camera 2, which is deliberately outside the window.
            Point3::new(0.80, 0.40, 4.0),
        ];

        let mut views = Vec::new();
        for pose in poses {
            let mut keypoints = Vec::with_capacity(world_points.len());
            for (point_index, point) in world_points.iter().enumerate() {
                let camera = pose.rotation * point.coords + pose.translation;
                let projected = intrinsics.project(&Point3::from(camera));
                if point_index == 4 {
                    // Only view 2 uses this point; other views receive a harmless
                    // placeholder so every observation has a valid keypoint index.
                    keypoints.push(KeyPoint::new(10.0, 10.0));
                } else {
                    keypoints.push(KeyPoint::new(projected.x, projected.y));
                }
            }
            views.push(View::new(keypoints, Descriptors::new(), 640, 480));
        }
        // A fifth view is present only to keep the state sized like a normal map;
        // it is unregistered and therefore cannot enter local BA.
        views.push(View::new(
            vec![KeyPoint::new(0.0, 0.0); world_points.len()],
            Descriptors::new(),
            640,
            480,
        ));

        let mut est = Est::new(views.len(), vec![Vec::new(); world_points.len()]);
        for (view, pose) in poses.into_iter().enumerate() {
            est.add_camera(view, pose);
        }
        est.points = world_points.to_vec();
        est.point_obs = vec![
            vec![(0, 0), (1, 0), (2, 0), (3, 0)],
            vec![(0, 1), (1, 1), (2, 1), (3, 1)],
            vec![(0, 2), (1, 2), (2, 2), (3, 2)],
            vec![(0, 3), (1, 3), (2, 3), (3, 3)],
            vec![(2, 4)],
        ];
        est.point_tracks = (0..world_points.len()).collect();
        est.track_point = (0..world_points.len()).map(Some).collect();

        let config = MapperConfig {
            local_ba_window: 3,
            local_ba_min_overlap: 2,
            ba_max_iterations: 1,
            ba_use_sparsity: false,
            ba_robust_kernel: false,
            ba_every: 0,
            ba_final: false,
            ..MapperConfig::default()
        };
        let problem = local_ba_problem(&est, 3, &config).expect("a co-visible local problem");
        let repeated = local_ba_problem(&est, 3, &config).expect("a repeated local problem");
        assert_eq!(problem.cameras, repeated.cameras);
        assert_eq!(problem.landmarks, repeated.landmarks);
        // Camera 2 ties with cameras 0 and 1, but the window is three cameras
        // and the ascending-index tie-break selects 0 then 1, leaving 2 outside.
        assert_eq!(problem.cameras, vec![3, 0, 1]);
        assert_eq!(problem.landmarks, vec![0, 1, 2, 3]);

        let outside_camera = est.cam_of_view[2].expect("camera 2 is registered");
        let outside_pose = est.poses[outside_camera];
        let outside_point = est.points[4];
        assert!(refine_local(&mut est, 3, &views, &intrinsics, &config));

        // These are the explicit writeback boundary checks: neither an omitted
        // camera nor a landmark outside the sub-problem is assigned to.
        assert_eq!(
            est.poses[outside_camera].translation,
            outside_pose.translation
        );
        assert_eq!(
            est.poses[outside_camera].rotation.into_inner(),
            outside_pose.rotation.into_inner()
        );
        assert_eq!(est.points[4], outside_point);

        // The full-state entry point remains available independently of local BA.
        assert!(refine(&mut est, &views, &intrinsics, &config));
        assert!(est.poses.iter().all(pose_is_finite));
        assert!(est.points.iter().all(point_is_finite));
    }

    #[test]
    fn sampson_error_is_zero_on_the_epipolar_constraint() {
        // F derived from a pure translation along x.
        let f = nalgebra::Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        // p2 lies on the epipolar line of p1 when y is preserved.
        let error = sampson_sq(&f, &Point2::new(100.0, 100.0), &Point2::new(120.0, 100.0));
        assert!(error < 1e-12, "expected zero Sampson error, got {error}");
    }
}
