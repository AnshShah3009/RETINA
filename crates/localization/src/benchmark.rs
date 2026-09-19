//! Real-data benchmark support for the localization pipeline.
//!
//! This module lives behind the `synthetic` feature so that the default build of
//! `cv-localization` stays free of the `image` decoder and the `cv-eval`
//! metrics. It contains the machinery behind the `tum_benchmark` example:
//!
//! 1. Load a TUM RGB-D sequence (`rgb.txt`, `groundtruth.txt` and the referenced
//!    `rgb/*.png` frames), associating frames with ground-truth poses through
//!    [`cv_io::datasets::tum::associate`].
//! 2. Select disjoint database/query frames (a second directory may be supplied
//!    for the cross-sequence case).
//! 3. Detect ORB features and descriptors on every selected frame.
//! 4. Triangulate a landmark map from consecutive database frames and lift every
//!    match into a 2D observation of a 3D [`Landmark`].
//! 5. Build a [`Database`] (optionally with a BoW [`Vocabulary`]) and localize
//!    every query frame.
//! 6. Aggregate plain localization, retrieval and timing metrics.
//!
//! Nothing here fabricates data: every reported number is measured from the
//! frames and ground-truth poses it is given, and an inability to localize is
//! reported as such rather than papered over.

use crate::localizer::{LocalizationResult, Localizer, LocalizerConfig};
use crate::{evaluate_localization, Database, DatabaseImage, Landmark};
use cv_calib3d::triangulate_points;
use cv_core::{CameraIntrinsics, Descriptors, KeyPoint, Pose};
use cv_eval::retrieval::{hit_rate_at_k, mean_average_precision, recall_at_k};
use cv_features::matcher::match_descriptors;
use cv_features::orb::orb_detect_and_compute;
use cv_features::retrieval::{descriptors_to_bytes, BowVector, Vocabulary};
use cv_io::datasets::tum;
use nalgebra::{Matrix3x4, Point3};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Every knob needed to reproduce a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Directory of the database TUM sequence (`rgb.txt`, `groundtruth.txt`,
    /// `rgb/*.png`).
    pub db_dir: PathBuf,
    /// Optional second sequence for the cross-sequence case. When `None` the
    /// query frames are taken from the second half of `db_dir`.
    pub query_dir: Option<PathBuf>,
    /// Number of database frames to select.
    pub db_frames: usize,
    /// Number of query frames to select.
    pub query_frames: usize,
    /// Index stride applied to each sequence before the even selection.
    pub stride: usize,
    /// Maximum number of ORB features (keypoints) per frame.
    pub features: usize,
    /// Lowe ratio-test threshold used both for map triangulation and for the
    /// localizer's descriptor matching.
    pub ratio: f32,
    /// Retrieval hit radius in metres.
    pub hit_radius: f64,
    /// Timestamp association tolerance in seconds.
    pub max_dt: f64,
    /// Triangulation reprojection tolerance in pixels, checked in *both* views.
    pub tri_reproj: f64,
    /// Optional vocabulary size. When set, a [`Vocabulary`] is trained on the
    /// database descriptors and used for BoW retrieval.
    pub vocab: Option<usize>,
    /// Deterministic seed for vocabulary training.
    pub seed: u64,
    /// Number of retrieval candidates the localizer considers per query.
    pub candidates: usize,
    /// Camera intrinsics used for every frame.
    pub intrinsics: CameraIntrinsics,
}

/// What was loaded and how large the reconstructed map turned out to be.
#[derive(Debug, Clone)]
pub struct DatasetSummary {
    /// Frames decoded in total (database + query selections).
    pub frames_loaded: usize,
    /// Frames associated with a ground-truth pose in the database sequence.
    pub db_sequence_frames: usize,
    /// Frames associated with a ground-truth pose in the query sequence.
    pub query_sequence_frames: usize,
    /// Database frames actually selected.
    pub db_frames: usize,
    /// Database frames requested.
    pub db_frames_requested: usize,
    /// Query frames actually selected.
    pub query_frames: usize,
    /// Query frames requested.
    pub query_frames_requested: usize,
    /// Features per frame requested.
    pub features_requested: usize,
    /// Mean number of descriptors actually extracted per frame.
    pub mean_features_per_frame: f64,
    /// Number of triangulated landmarks in the map.
    pub landmarks: usize,
    /// Mean number of keypoint observations per landmark.
    pub mean_observations_per_landmark: f64,
    /// Width of the loaded frames in pixels (`0` when nothing was loaded).
    pub image_width: u32,
    /// Height of the loaded frames in pixels (`0` when nothing was loaded).
    pub image_height: u32,
}

/// Aggregated pose-estimation accuracy over the query set.
#[derive(Debug, Clone)]
pub struct LocalizationReport {
    /// Query frames attempted.
    pub attempted: usize,
    /// Queries that returned a pose.
    pub succeeded: usize,
    /// `succeeded / attempted` (`0.0` when nothing was attempted).
    pub success_rate: f64,
    /// Mean camera-centre error over successful queries, in metres.
    pub mean_translation_m: f64,
    /// Median camera-centre error over successful queries, in metres.
    pub median_translation_m: f64,
    /// Mean rotation error over successful queries, in degrees.
    pub mean_rotation_deg: f64,
    /// Median rotation error over successful queries, in degrees.
    pub median_rotation_deg: f64,
    /// Mean RANSAC inliers over successful queries.
    pub mean_inliers: f64,
    /// Mean RMS reprojection error over successful queries, in pixels.
    pub mean_reprojection_rmse_px: f64,
}

/// Image-retrieval quality, computed independently of the PnP stage.
#[derive(Debug, Clone)]
pub struct RetrievalReport {
    /// Hit rate@1: fraction of queries whose top-1 candidate is within the hit
    /// radius. This is "recall@1" as place recognition papers use the term.
    pub hit_rate_at_1: f64,
    /// Hit rate@5, same convention.
    pub hit_rate_at_5: f64,
    /// Hit rate@10, same convention.
    pub hit_rate_at_10: f64,
    /// Mean recall@1 over all relevant frames (information-retrieval
    /// convention: what fraction of the frames within the radius were
    /// retrieved). Drops as the database gets denser, because the relevant set
    /// grows; prefer `hit_rate_at_*` when comparing configurations.
    pub recall_at_1: f64,
    /// Mean recall@5, IR convention.
    pub recall_at_5: f64,
    /// Mean recall@10, IR convention.
    pub recall_at_10: f64,
    /// Mean average precision over queries with a ground-truth hit.
    pub map: f64,
    /// Queries for which at least one database frame is within the hit radius.
    pub queries_with_hit: usize,
    /// Query frames considered.
    pub queries: usize,
}

/// Wall-clock timings for the run.
#[derive(Debug, Clone)]
pub struct TimingReport {
    /// Mean milliseconds spent per query in retrieval + matching + PnP.
    pub mean_query_ms: f64,
    /// Total wall time for the whole run, in seconds.
    pub total_wall_s: f64,
}

/// The complete benchmark result.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Dataset / map summary.
    pub summary: DatasetSummary,
    /// Pose-estimation accuracy.
    pub localization: LocalizationReport,
    /// Retrieval quality.
    pub retrieval: RetrievalReport,
    /// Timings.
    pub timing: TimingReport,
}

/// A frame of a sequence after timestamp association.
struct SequenceFrame {
    /// File name relative to the sequence directory.
    filename: String,
    /// Ground-truth camera-to-world pose (`x_world = R * x_cam + t`), so
    /// `pose.translation` is the camera centre in world coordinates.
    pose_wc: Pose,
}

/// A frame after feature extraction.
struct ExtractedFrame {
    /// Ground-truth camera-to-world pose.
    pose_wc: Pose,
    /// Ground-truth world-to-camera pose.
    pose_cw: Pose,
    /// Keypoints, parallel to `descriptors`.
    keypoints: Vec<KeyPoint>,
    /// Descriptors, parallel to `keypoints`.
    descriptors: Descriptors,
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
}

/// Run the benchmark described by `config`.
///
/// All failures (missing dataset files, unreadable frames, too few frames to
/// split) are returned as a human-readable error string; this function never
/// panics on malformed input.
pub fn run(config: &BenchmarkConfig) -> Result<BenchmarkReport, String> {
    let started = Instant::now();

    if config.db_frames < 2 {
        return Err("--db-frames must be at least 2 (a map needs frame pairs)".to_string());
    }
    if config.query_frames == 0 {
        return Err("--query-frames must be at least 1".to_string());
    }

    let query_dir = config
        .query_dir
        .clone()
        .unwrap_or_else(|| config.db_dir.clone());
    let cross_sequence = config.query_dir.is_some();

    // ---- 1. Load and associate the sequences ----
    let db_seq = load_sequence(&config.db_dir, config.max_dt, "database")?;
    let query_seq = if cross_sequence {
        load_sequence(&query_dir, config.max_dt, "query")?
    } else {
        Vec::new()
    };

    // ---- 2. Select database and query frame indices ----
    let (db_indices, query_indices) = if cross_sequence {
        let db_pool = strided_indices(0, db_seq.len(), config.stride);
        let query_pool = strided_indices(0, query_seq.len(), config.stride);
        (
            select_evenly(&db_pool, config.db_frames),
            select_evenly(&query_pool, config.query_frames),
        )
    } else {
        if db_seq.len() < 4 {
            return Err(format!(
                "sequence {} has only {} associated frames; need at least 4 to split into \
                 disjoint database/query halves",
                config.db_dir.display(),
                db_seq.len()
            ));
        }
        let mid = db_seq.len() / 2;
        let db_pool = strided_indices(0, mid, config.stride);
        let query_pool = strided_indices(mid, db_seq.len(), config.stride);
        (
            select_evenly(&db_pool, config.db_frames),
            select_evenly(&query_pool, config.query_frames),
        )
    };

    if db_indices.len() < 2 {
        return Err(format!(
            "only {} database frame(s) selected (requested {}); increase --db-frames, lower \
             --stride, or check --max-dt",
            db_indices.len(),
            config.db_frames
        ));
    }
    if query_indices.is_empty() {
        return Err(
            "no query frames selected; lower --stride or check the query directory".to_string(),
        );
    }

    // ---- 3. Extract ORB features on every selected frame ----
    let mut db_frames = Vec::with_capacity(db_indices.len());
    for &i in &db_indices {
        db_frames.push(extract_frame(&config.db_dir, &db_seq[i], config.features)?);
    }

    let mut query_frames = Vec::with_capacity(query_indices.len());
    if cross_sequence {
        for &i in &query_indices {
            query_frames.push(extract_frame(&query_dir, &query_seq[i], config.features)?);
        }
    } else {
        // Same directory: the query half is disjoint from the database half.
        for &i in &query_indices {
            query_frames.push(extract_frame(&config.db_dir, &db_seq[i], config.features)?);
        }
    }

    let (image_width, image_height) = db_frames
        .iter()
        .chain(query_frames.iter())
        .next()
        .map(|f| (f.width, f.height))
        .unwrap_or((0, 0));

    let total_descriptors: usize = db_frames
        .iter()
        .chain(query_frames.iter())
        .map(|f| f.descriptors.len())
        .sum();
    let num_frames = db_frames.len() + query_frames.len();
    let mean_features_per_frame = if num_frames == 0 {
        0.0
    } else {
        total_descriptors as f64 / num_frames as f64
    };

    // ---- 4. Triangulate the landmark map from consecutive database frames ----
    let (landmarks, per_image_landmarks, total_observations) = build_map(
        &db_frames,
        &config.intrinsics,
        config.ratio,
        config.tri_reproj,
    );
    let mean_observations_per_landmark = if landmarks.is_empty() {
        0.0
    } else {
        total_observations as f64 / landmarks.len() as f64
    };

    // ---- 5. Build the database and localize every query ----
    let vocabulary = config.vocab.map(|k| {
        let pool: Vec<Vec<u8>> = db_frames
            .iter()
            .flat_map(|f| descriptors_to_bytes(&f.descriptors))
            .collect();
        Vocabulary::train(&pool, k.max(1), 10, config.seed)
    });

    let mut database = Database::new(vocabulary);
    for landmark in &landmarks {
        database.add_landmark(landmark.clone());
    }
    for (id, frame) in db_frames.iter().enumerate() {
        database.add_image(DatabaseImage {
            id,
            pose: Some(frame.pose_cw),
            keypoints: frame.keypoints.clone(),
            descriptors: frame.descriptors.clone(),
            landmarks: per_image_landmarks[id].clone(),
        });
    }
    database.build();

    let localizer = Localizer::new(
        &database,
        LocalizerConfig {
            candidates: config.candidates,
            ratio: config.ratio,
            ..LocalizerConfig::default()
        },
    );

    let mut results: Vec<Option<LocalizationResult>> = Vec::with_capacity(query_frames.len());
    let mut query_time = Duration::ZERO;
    for query in &query_frames {
        let t0 = Instant::now();
        let result = localizer.localize(&query.keypoints, &query.descriptors, &config.intrinsics);
        query_time += t0.elapsed();
        results.push(result);
    }

    // ---- Localization accuracy against ground truth ----
    // The localizer returns a world-to-camera pose; inverting it yields a
    // camera-to-world pose whose translation *is* the camera centre, matching the
    // TUM ground-truth convention. The translation error is therefore the
    // Euclidean distance between the estimated and true camera centres, in
    // metres, and the rotation error is frame-convention independent.
    let inverted: Vec<Option<LocalizationResult>> = results
        .iter()
        .map(|result| {
            result.as_ref().map(|r| {
                let mut converted = r.clone();
                converted.pose = r.pose.inverse();
                converted
            })
        })
        .collect();
    let ground_truth: Vec<Pose> = query_frames.iter().map(|f| f.pose_wc).collect();
    let stats = evaluate_localization(&inverted, &ground_truth);

    let successes: Vec<&LocalizationResult> = results.iter().flatten().collect();
    let mean_inliers = mean(successes.iter().map(|r| r.inliers as f64));
    let mean_reprojection_rmse = mean(successes.iter().map(|r| r.reprojection_rmse));

    // ---- Retrieval metrics (independent of PnP) ----
    let all = database.len();
    let mut predictions: Vec<Vec<usize>> = Vec::with_capacity(query_frames.len());
    let mut relevant: Vec<Vec<usize>> = Vec::with_capacity(query_frames.len());
    for query in &query_frames {
        predictions.push(rank_images(&database, &query.descriptors, all));
        let centre = query.pose_wc.translation;
        let hits: Vec<usize> = db_frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| (frame.pose_wc.translation - centre).norm() <= config.hit_radius)
            .map(|(id, _)| id)
            .collect();
        relevant.push(hits);
    }
    let queries_with_hit = relevant.iter().filter(|ids| !ids.is_empty()).count();
    let retrieval = RetrievalReport {
        hit_rate_at_1: hit_rate_at_k(&predictions, &relevant, 1),
        hit_rate_at_5: hit_rate_at_k(&predictions, &relevant, 5),
        hit_rate_at_10: hit_rate_at_k(&predictions, &relevant, 10),
        recall_at_1: recall_at_k(&predictions, &relevant, 1),
        recall_at_5: recall_at_k(&predictions, &relevant, 5),
        recall_at_10: recall_at_k(&predictions, &relevant, 10),
        map: mean_average_precision(&predictions, &relevant),
        queries_with_hit,
        queries: query_frames.len(),
    };

    let mean_query_ms = if query_frames.is_empty() {
        f64::NAN
    } else {
        query_time.as_secs_f64() * 1000.0 / query_frames.len() as f64
    };

    // In the same-directory case the query half is a slice of the database
    // sequence, so both counts refer to the same associated frame list.
    let query_sequence_frames = if cross_sequence {
        query_seq.len()
    } else {
        db_seq.len()
    };

    Ok(BenchmarkReport {
        summary: DatasetSummary {
            frames_loaded: db_frames.len() + query_frames.len(),
            db_sequence_frames: db_seq.len(),
            query_sequence_frames,
            db_frames: db_frames.len(),
            db_frames_requested: config.db_frames,
            query_frames: query_frames.len(),
            query_frames_requested: config.query_frames,
            features_requested: config.features,
            mean_features_per_frame,
            landmarks: landmarks.len(),
            mean_observations_per_landmark,
            image_width,
            image_height,
        },
        localization: LocalizationReport {
            attempted: query_frames.len(),
            succeeded: stats.succeeded,
            success_rate: stats.success_rate,
            mean_translation_m: stats.mean_translation_error,
            median_translation_m: stats.median_translation_error,
            mean_rotation_deg: stats.mean_rotation_error_deg,
            median_rotation_deg: stats.median_rotation_error_deg,
            mean_inliers,
            mean_reprojection_rmse_px: mean_reprojection_rmse,
        },
        retrieval,
        timing: TimingReport {
            mean_query_ms,
            total_wall_s: started.elapsed().as_secs_f64(),
        },
    })
}

/// Load a TUM sequence and return its frames associated with ground-truth poses.
fn load_sequence(dir: &Path, max_dt: f64, label: &str) -> Result<Vec<SequenceFrame>, String> {
    if !dir.is_dir() {
        return Err(format!("{label} directory not found: {}", dir.display()));
    }
    let rgb_path = dir.join("rgb.txt");
    let gt_path = dir.join("groundtruth.txt");
    if !rgb_path.is_file() {
        return Err(format!(
            "{label} index file not found: {}",
            rgb_path.display()
        ));
    }
    if !gt_path.is_file() {
        return Err(format!(
            "{label} ground-truth file not found: {}",
            gt_path.display()
        ));
    }

    let rgb = tum::read_index(&rgb_path)
        .map_err(|e| format!("failed to read {}: {e}", rgb_path.display()))?;
    let groundtruth = tum::read_groundtruth(&gt_path)
        .map_err(|e| format!("failed to read {}: {e}", gt_path.display()))?;

    if rgb.is_empty() {
        return Err(format!("{} contains no entries", rgb_path.display()));
    }
    if groundtruth.is_empty() {
        return Err(format!("{} contains no poses", gt_path.display()));
    }

    // `associate` only consumes timestamps of the second list.
    let gt_index: Vec<tum::IndexEntry> = groundtruth
        .iter()
        .map(|pose| tum::IndexEntry {
            timestamp: pose.timestamp,
            filename: String::new(),
        })
        .collect();
    let associations = tum::associate(&rgb, &gt_index, max_dt);

    let frames = associations
        .into_iter()
        .map(|(rgb_idx, gt_idx)| SequenceFrame {
            filename: rgb[rgb_idx].filename.clone(),
            pose_wc: groundtruth[gt_idx].pose,
        })
        .collect::<Vec<_>>();

    if frames.is_empty() {
        return Err(format!(
            "no frames in {} within {} s of a ground-truth pose; check --max-dt",
            dir.display(),
            max_dt
        ));
    }

    Ok(frames)
}

/// Detect ORB features on one frame and load its descriptors.
fn extract_frame(
    dir: &Path,
    frame: &SequenceFrame,
    features: usize,
) -> Result<ExtractedFrame, String> {
    let path = dir.join(&frame.filename);
    let image =
        image::open(&path).map_err(|e| format!("failed to open frame {}: {e}", path.display()))?;
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());

    let (_keypoints, descriptors) = orb_detect_and_compute(&gray, features.max(1));
    // The descriptor carries the keypoint it was computed at; rebuilding the
    // keypoint list from the descriptors keeps the two parallel even though ORB
    // drops border keypoints for which no descriptor could be computed.
    let keypoints: Vec<KeyPoint> = descriptors.iter().map(|d| d.keypoint).collect();

    Ok(ExtractedFrame {
        pose_wc: frame.pose_wc,
        pose_cw: frame.pose_wc.inverse(),
        keypoints,
        descriptors,
        width,
        height,
    })
}

/// Triangulate a landmark map from consecutive database frames.
///
/// Returns the landmarks, per-image landmark assignments (parallel to each
/// frame's descriptors) and the total number of keypoint observations.
fn build_map(
    frames: &[ExtractedFrame],
    intrinsics: &CameraIntrinsics,
    ratio: f32,
    tri_reproj: f64,
) -> (Vec<Landmark>, Vec<Vec<Option<usize>>>, usize) {
    let mut landmarks: Vec<Landmark> = Vec::new();
    let mut per_image: Vec<Vec<Option<usize>>> = frames
        .iter()
        .map(|frame| vec![None; frame.descriptors.len()])
        .collect();

    if frames.len() < 2 {
        return (landmarks, per_image, 0);
    }

    for i in 0..frames.len() - 1 {
        let a = &frames[i];
        let b = &frames[i + 1];
        if a.descriptors.is_empty() || b.descriptors.is_empty() {
            continue;
        }

        let projection_a = projection_matrix(&a.pose_cw, intrinsics);
        let projection_b = projection_matrix(&b.pose_cw, intrinsics);
        let matches = match_descriptors(&a.descriptors, &b.descriptors, Some(ratio));

        for m in &matches.matches {
            let (Ok(ai), Ok(bi)) = (usize::try_from(m.query_idx), usize::try_from(m.train_idx))
            else {
                continue;
            };
            if ai >= a.keypoints.len() || bi >= b.keypoints.len() {
                continue;
            }

            let point_a = a.keypoints[ai].pt();
            let point_b = b.keypoints[bi].pt();
            let Ok(triangulated) =
                triangulate_points(&projection_a, &projection_b, &[point_a], &[point_b])
            else {
                continue;
            };
            let point = triangulated[0];
            if !point.coords.iter().all(|c| c.is_finite()) {
                continue;
            }

            // In front of both cameras.
            let camera_a = a.pose_cw.rotation * point.coords + a.pose_cw.translation;
            let camera_b = b.pose_cw.rotation * point.coords + b.pose_cw.translation;
            if camera_a[2] <= 1e-6 || camera_b[2] <= 1e-6 {
                continue;
            }

            // Reprojection error below tolerance in both views.
            let reprojected_a = intrinsics.project(&Point3::from(camera_a));
            let reprojected_b = intrinsics.project(&Point3::from(camera_b));
            let error_a = ((reprojected_a.x - point_a.x).powi(2)
                + (reprojected_a.y - point_a.y).powi(2))
            .sqrt();
            let error_b = ((reprojected_b.x - point_b.x).powi(2)
                + (reprojected_b.y - point_b.y).powi(2))
            .sqrt();
            if !(error_a <= tri_reproj && error_b <= tri_reproj) {
                continue;
            }

            let index = landmarks.len();
            landmarks.push(Landmark {
                position: point,
                descriptors: vec![
                    a.descriptors.descriptors[ai].clone(),
                    b.descriptors.descriptors[bi].clone(),
                ],
            });
            if per_image[i][ai].is_none() {
                per_image[i][ai] = Some(index);
            }
            if per_image[i + 1][bi].is_none() {
                per_image[i + 1][bi] = Some(index);
            }
        }
    }

    let observations = per_image
        .iter()
        .flat_map(|landmarks| landmarks.iter())
        .filter(|entry| entry.is_some())
        .count();

    (landmarks, per_image, observations)
}

/// Build the 3x4 pixel projection matrix `K * [R | t]` for a world-to-camera pose.
fn projection_matrix(pose_cw: &Pose, intrinsics: &CameraIntrinsics) -> Matrix3x4<f64> {
    let rt = pose_cw.matrix().fixed_view::<3, 4>(0, 0).into_owned();
    intrinsics.matrix() * rt
}

/// Rank database images for a query, mirroring the localizer's retrieval path.
fn rank_images(database: &Database, query: &Descriptors, k: usize) -> Vec<usize> {
    if let (Some(bow), Some(vocabulary)) = (database.bow_index(), database.vocabulary()) {
        let bytes = descriptors_to_bytes(query);
        let bow_vector = BowVector::from_descriptors(vocabulary, &bytes);
        let hits = bow.query(&bow_vector, k);
        if !hits.is_empty() {
            return hits.into_iter().map(|(id, _)| id).collect();
        }
    }
    database.rank_by_descriptor_matches(query, k)
}

/// Indices `start, start + stride, ...` in `[start, end)`.
fn strided_indices(start: usize, end: usize, stride: usize) -> Vec<usize> {
    let stride = stride.max(1);
    let mut indices = Vec::new();
    let mut i = start;
    while i < end {
        indices.push(i);
        i += stride;
    }
    indices
}

/// Pick up to `count` indices spread evenly across `pool`.
///
/// When `count` is zero the result is empty; when `count` is at least the pool
/// length the whole pool is returned.
fn select_evenly(pool: &[usize], count: usize) -> Vec<usize> {
    if count == 0 || pool.is_empty() {
        return Vec::new();
    }
    if count >= pool.len() {
        return pool.to_vec();
    }
    if count == 1 {
        return vec![pool[0]];
    }
    (0..count)
        .map(|k| {
            let idx = k * (pool.len() - 1) / (count - 1);
            pool[idx]
        })
        .collect()
}

/// Arithmetic mean, or `NaN` for an empty sequence.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
