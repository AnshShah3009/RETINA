//! `cv-bench` — command-line evaluation for the RETINA workspace.
//!
//! The binary ties the `cv-eval` metric layer to the `cv-io` dataset loaders so
//! a reconstruction or a trajectory can be measured from the command line:
//!
//! * `trajectory` — ATE / RPE of an estimate against ground truth (TUM, KITTI or
//!   EuRoC).
//! * `model` — summary of a COLMAP text model (registration rate, observations,
//!   track lengths).
//! * `retrieval` — recall@k / precision@k / mAP over ranked id lists.
//!
//! Every failure is reported as a message on stderr and a non-zero exit code;
//! bad input never panics.

#![forbid(unsafe_code)]

mod args;

use args::{Format, TrajectoryArgs};
use cv_core::Pose;
use cv_eval::{Alignment, Trajectory};
use cv_io::datasets::tum::IndexEntry;
use cv_io::datasets::{colmap, euroc, kitti, tum};
use nalgebra::{UnitQuaternion, Vector3};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("cv-bench: error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: &[String]) -> Result<(), String> {
    match argv.split_first() {
        None => {
            print_usage();
            Ok(())
        }
        Some((command, rest)) => match command.as_str() {
            "help" | "--help" | "-h" => {
                print_usage();
                Ok(())
            }
            "trajectory" => command_trajectory(rest),
            "model" => command_model(rest),
            "retrieval" => command_retrieval(rest),
            other => Err(format!(
                "unknown subcommand '{other}' (expected: trajectory, model, retrieval, help)"
            )),
        },
    }
}

fn print_usage() {
    println!("cv-bench - evaluate trajectories, reconstructions and retrieval\n");
    println!("USAGE:");
    println!("  cv-bench <SUBCOMMAND> [OPTIONS]\n");
    println!("SUBCOMMANDS:");
    println!("  trajectory   Compare an estimated trajectory against ground truth");
    println!("  model        Summarise a COLMAP text model");
    println!("  retrieval    Compute recall@k, precision@k and mAP");
    println!("  help         Print this message\n");
    println!("trajectory options:");
    println!("  --estimate <FILE>            estimated trajectory (required)");
    println!("  --ground-truth <FILE>        ground-truth trajectory (required)");
    println!("  --format <tum|kitti|euroc>   input format (required)");
    println!("  --align <none|se3|sim3>      alignment before ATE (default: se3)");
    println!("  --rpe-delta <N>              RPE frame gap (default: 1)");
    println!("  --max-dt <SECONDS>           TUM association tolerance (default: 0.02)\n");
    println!("model options:");
    println!("  --images <FILE>              COLMAP images.txt (required)");
    println!("  --points3d <FILE>            COLMAP points3D.txt (optional)");
    println!("  --cameras <FILE>             COLMAP cameras.txt (optional)\n");
    println!("retrieval options:");
    println!("  --predictions <FILE>         ranked ids, one query per line (required)");
    println!("  --ground-truth <FILE>        relevant ids, one query per line (required)");
    println!("  --k <N>                      rank cut-off (required)");
}

/// Loaded, aligned trajectories plus a note on how they were paired.
struct LoadedPair {
    estimate: Trajectory,
    ground_truth: Trajectory,
    paired: usize,
    method: String,
}

fn command_trajectory(argv: &[String]) -> Result<(), String> {
    let opts = args::parse_trajectory(argv)?;
    let pair = load_pair(&opts)?;

    let estimate = &pair.estimate;
    let ground_truth = &pair.ground_truth;
    let ate = estimate.ate(ground_truth, opts.align);
    let rpe = estimate.rpe(ground_truth, opts.rpe_delta);

    println!("Trajectory comparison");
    println!("  format:             {}", opts.format.as_str());
    println!("  alignment:          {}", alignment_name(opts.align));
    println!("  paired poses:       {} ({})", pair.paired, pair.method);
    println!("  estimate poses:     {}", estimate.len());
    println!("  ground-truth poses: {}", ground_truth.len());
    println!(
        "  path length (estimate):     {:.6} m",
        estimate.path_length()
    );
    println!(
        "  path length (ground truth): {:.6} m",
        ground_truth.path_length()
    );
    println!(
        "  camera-centre RMSE:         {:.6} m",
        estimate.camera_center_rmse(ground_truth)
    );
    println!("  ATE rmse:           {:.6} m", ate.rmse);
    println!("  ATE mean:           {:.6} m", ate.mean);
    println!("  ATE median:         {:.6} m", ate.median);
    println!("  ATE max:            {:.6} m", ate.max);
    println!("  ATE std:            {:.6} m", ate.std);
    println!(
        "  RPE translation rmse (delta={}): {:.6} m",
        rpe.delta_frames, rpe.translation.rmse
    );
    println!(
        "  RPE rotation rmse (delta={}):    {:.6} rad",
        rpe.delta_frames, rpe.rotation.rmse
    );
    Ok(())
}

fn load_pair(opts: &TrajectoryArgs) -> Result<LoadedPair, String> {
    match opts.format {
        Format::Tum => load_tum(opts),
        Format::Kitti => load_kitti(opts),
        Format::Euroc => load_euroc(opts),
    }
}

fn load_tum(opts: &TrajectoryArgs) -> Result<LoadedPair, String> {
    let estimate = tum::read_groundtruth(&opts.estimate).map_err(|e| e.to_string())?;
    let ground_truth = tum::read_groundtruth(&opts.ground_truth).map_err(|e| e.to_string())?;
    if estimate.is_empty() {
        return Err(format!(
            "estimate trajectory '{}' contains no poses",
            opts.estimate.display()
        ));
    }
    if ground_truth.is_empty() {
        return Err(format!(
            "ground-truth trajectory '{}' contains no poses",
            opts.ground_truth.display()
        ));
    }

    let matches = tum::associate(
        &tum_index(&estimate),
        &tum_index(&ground_truth),
        opts.max_dt,
    );
    if matches.is_empty() {
        return Err(format!(
            "no TUM poses could be associated within --max-dt = {} s \
             (estimate has {} poses, ground truth has {})",
            opts.max_dt,
            estimate.len(),
            ground_truth.len()
        ));
    }

    let estimate_poses: Vec<Pose> = matches.iter().map(|(i, _)| estimate[*i].pose).collect();
    let estimate_times: Vec<f64> = matches
        .iter()
        .map(|(i, _)| estimate[*i].timestamp)
        .collect();
    let truth_poses: Vec<Pose> = matches.iter().map(|(_, j)| ground_truth[*j].pose).collect();
    let truth_times: Vec<f64> = matches
        .iter()
        .map(|(_, j)| ground_truth[*j].timestamp)
        .collect();

    Ok(LoadedPair {
        estimate: build_trajectory(&estimate_poses, &estimate_times),
        ground_truth: build_trajectory(&truth_poses, &truth_times),
        paired: matches.len(),
        method: format!("timestamp association, max_dt = {} s", opts.max_dt),
    })
}

/// Build the `IndexEntry` view of a TUM trajectory that [`tum::associate`] needs.
fn tum_index(poses: &[tum::TumPose]) -> Vec<IndexEntry> {
    poses
        .iter()
        .map(|entry| IndexEntry {
            timestamp: entry.timestamp,
            filename: String::new(),
        })
        .collect()
}

/// KITTI `--max-dt` is ignored: poses are index-aligned. The two files are
/// normally the same length; a mismatch of at most [`kitti_tolerance`] poses is
/// truncated to the shorter with a warning, while a larger mismatch is an error.
fn load_kitti(opts: &TrajectoryArgs) -> Result<LoadedPair, String> {
    let estimate = kitti::read_poses(&opts.estimate).map_err(|e| e.to_string())?;
    let ground_truth = kitti::read_poses(&opts.ground_truth).map_err(|e| e.to_string())?;
    if estimate.is_empty() || ground_truth.is_empty() {
        return Err("a KITTI trajectory contains no poses (empty file?)".to_string());
    }

    let longer = estimate.len().max(ground_truth.len());
    let tolerance = kitti_tolerance(longer);
    let difference = estimate.len().abs_diff(ground_truth.len());
    if difference > tolerance {
        return Err(format!(
            "KITTI pose counts differ by {difference} (estimate {}, ground truth {}), \
             exceeding the tolerance of {tolerance} poses",
            estimate.len(),
            ground_truth.len()
        ));
    }
    if difference > 0 {
        eprintln!(
            "cv-bench: warning: KITTI pose counts differ (estimate {}, ground truth {}); \
             truncating to the shorter ({})",
            estimate.len(),
            ground_truth.len(),
            longer - difference
        );
    }

    let n = estimate.len().min(ground_truth.len());
    let times: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Ok(LoadedPair {
        estimate: build_trajectory(&estimate[..n], &times),
        ground_truth: build_trajectory(&ground_truth[..n], &times),
        paired: n,
        method: "index alignment (KITTI; --max-dt ignored)".to_string(),
    })
}

/// Number of pose-count mismatches tolerated for KITTI before it is an error.
fn kitti_tolerance(longer: usize) -> usize {
    (longer / 100).max(1)
}

/// EuRoC ground-truth states are index-aligned here; a length mismatch is
/// truncated to the shorter with a warning.
fn load_euroc(opts: &TrajectoryArgs) -> Result<LoadedPair, String> {
    let estimate = euroc::read_groundtruth(&opts.estimate).map_err(|e| e.to_string())?;
    let ground_truth = euroc::read_groundtruth(&opts.ground_truth).map_err(|e| e.to_string())?;
    if estimate.is_empty() || ground_truth.is_empty() {
        return Err("a EuRoC trajectory contains no poses (empty file?)".to_string());
    }
    if estimate.len() != ground_truth.len() {
        eprintln!(
            "cv-bench: warning: EuRoC state counts differ (estimate {}, ground truth {}); \
             truncating to the shorter ({})",
            estimate.len(),
            ground_truth.len(),
            estimate.len().min(ground_truth.len())
        );
    }

    let n = estimate.len().min(ground_truth.len());
    let estimate_poses: Vec<Pose> = estimate[..n].iter().map(|state| state.pose).collect();
    let estimate_times: Vec<f64> = estimate[..n]
        .iter()
        .map(|state| state.timestamp_ns as f64 * 1e-9)
        .collect();
    let truth_poses: Vec<Pose> = ground_truth[..n].iter().map(|state| state.pose).collect();
    let truth_times: Vec<f64> = ground_truth[..n]
        .iter()
        .map(|state| state.timestamp_ns as f64 * 1e-9)
        .collect();

    Ok(LoadedPair {
        estimate: build_trajectory(&estimate_poses, &estimate_times),
        ground_truth: build_trajectory(&truth_poses, &truth_times),
        paired: n,
        method: "index alignment (EuRoC)".to_string(),
    })
}

/// Assemble a [`Trajectory`] from poses and matching timestamps.
fn build_trajectory(poses: &[Pose], timestamps: &[f64]) -> Trajectory {
    let positions: Vec<Vector3<f64>> = poses.iter().map(|pose| pose.translation).collect();
    let quaternions: Vec<UnitQuaternion<f64>> = poses.iter().map(|pose| pose.rotation).collect();
    Trajectory::from_positions_and_quaternions_with_timestamps(&positions, &quaternions, timestamps)
}

fn alignment_name(alignment: Alignment) -> &'static str {
    match alignment {
        Alignment::None => "none",
        Alignment::Se3 => "se3",
        Alignment::Sim3 => "sim3",
    }
}

fn command_model(argv: &[String]) -> Result<(), String> {
    let opts = args::parse_model(argv)?;
    let images = colmap::read_images_text(&opts.images).map_err(|e| e.to_string())?;
    let cameras = match &opts.cameras {
        Some(path) => Some(colmap::read_cameras_text(path).map_err(|e| e.to_string())?),
        None => None,
    };
    let points = match &opts.points3d {
        Some(path) => Some(colmap::read_points3d_text(path).map_err(|e| e.to_string())?),
        None => None,
    };

    let image_count = images.len();
    let registered = images
        .iter()
        .filter(|image| !image.points2d.is_empty())
        .count();
    let total_observations: usize = images.iter().map(|image| image.points2d.len()).sum();
    let registered_observations: usize = images
        .iter()
        .filter(|image| !image.points2d.is_empty())
        .map(|image| image.points2d.len())
        .sum();
    let registration = cv_eval::registration_rate(registered, image_count);
    let mean_observations = if registered == 0 {
        0.0
    } else {
        registered_observations as f64 / registered as f64
    };
    let camera_count = match &cameras {
        Some(cameras) => cameras.len(),
        None => distinct_camera_ids(&images),
    };

    println!("COLMAP model");
    println!("  cameras:                                {camera_count}");
    println!("  images:                                 {image_count}");
    println!("  registered images:                      {registered}");
    println!("  registration rate:                      {registration:.6}");
    println!("  total observations:                     {total_observations}");
    println!("  mean observations per registered image: {mean_observations:.6}");

    if let Some(points) = points {
        let point_count = points.len();
        let mean_track = if point_count == 0 {
            0.0
        } else {
            points.iter().map(|point| point.track.len()).sum::<usize>() as f64 / point_count as f64
        };
        let mean_error = if point_count == 0 {
            0.0
        } else {
            points.iter().map(|point| point.error).sum::<f64>() / point_count as f64
        };
        println!("  points3D:                               {point_count}");
        println!("  mean track length:                      {mean_track:.6}");
        println!("  mean reprojection error:                {mean_error:.6}");
    }
    Ok(())
}

/// Number of distinct camera ids referenced by the images (used when the
/// optional `cameras.txt` is not supplied).
fn distinct_camera_ids(images: &[colmap::Image]) -> usize {
    let mut ids: Vec<u32> = images.iter().map(|image| image.camera_id).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

fn command_retrieval(argv: &[String]) -> Result<(), String> {
    let opts = args::parse_retrieval(argv)?;
    let predictions = read_ranked_ids(&opts.predictions)?;
    let ground_truth = read_ranked_ids(&opts.ground_truth)?;
    if predictions.len() != ground_truth.len() {
        return Err(format!(
            "query count mismatch: predictions file has {} line(s), ground-truth file has {}",
            predictions.len(),
            ground_truth.len()
        ));
    }

    let k = opts.k;
    let recall = cv_eval::recall_at_k(&predictions, &ground_truth, k);
    let precision = cv_eval::precision_at_k(&predictions, &ground_truth, k);
    let map = cv_eval::mean_average_precision(&predictions, &ground_truth);

    println!("Retrieval metrics");
    println!("  queries:      {}", predictions.len());
    println!("  k:            {k}");
    println!("  recall@{k}:    {recall:.6}");
    println!("  precision@{k}: {precision:.6}");
    println!("  mAP:          {map:.6}");
    Ok(())
}

/// Read one query per line, splitting ids on commas and/or whitespace.
fn read_ranked_ids(path: &Path) -> Result<Vec<Vec<usize>>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;

    let mut queries = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let mut ids = Vec::new();
        for token in raw.split(|c: char| c == ',' || c.is_whitespace()) {
            if token.is_empty() {
                continue;
            }
            let id = token.parse::<usize>().map_err(|_| {
                format!(
                    "{}: line {}: '{token}' is not a valid id",
                    path.display(),
                    index + 1
                )
            })?;
            ids.push(id);
        }
        queries.push(ids);
    }
    Ok(queries)
}
