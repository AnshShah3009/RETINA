//! Incremental Structure-from-Motion on a real TUM RGB-D sequence.
//!
//! This example runs [`cv_sfm::mapper`] on the RGB stream of a TUM RGB-D
//! sequence and reports `visloc`-comparable numbers. The mapper never sees the
//! ground-truth poses: they are used *only* here, in the example, to score the
//! result after the fact.
//!
//! ```text
//! cargo run -p cv-sfm --release --example tum_sfm -- \
//!     --dir /data/tum/rgbd_dataset_freiburg1_desk --frames 60 --stride 5
//! ```
//!
//! Reported:
//!
//! * **registration rate** — cameras registered / views supplied;
//! * **camera-centre RMSE after Sim(3) alignment** to the ground truth (the
//!   reconstruction is monocular, hence scale-free, so a similarity alignment is
//!   the only meaningful one) plus its mean/median/max and the recovered scale;
//! * **rotation RMSE** in degrees, measured after applying the alignment
//!   rotation to the estimated camera-to-world orientations;
//! * number of 3D points, mean track length, and wall time.
//!
//! Nothing is fabricated: if the mapper cannot seed, the example says so and
//! exits 0 with the diagnostic instead of inventing numbers.

use cv_core::{CameraIntrinsics, Pose};
use cv_eval::{Alignment, Trajectory};
use cv_features::orb::orb_detect_and_compute;
use cv_io::datasets::tum;
use cv_sfm::mapper::{map_views, MapperConfig, Mapping, PairSelection, View};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// TUM RGB-D published intrinsics for the 640x480 rgb stream (fr1/fr2 series are
/// within ~1 px of these values). Overridable with `--fx/--fy/--cx/--cy`.
const TUM_FX: f64 = 517.3;
const TUM_FY: f64 = 516.5;
const TUM_CX: f64 = 318.6;
const TUM_CY: f64 = 255.3;

fn usage() -> String {
    "\
cv-sfm incremental Structure-from-Motion on a TUM RGB-D sequence

USAGE:
    tum_sfm --dir <path> [OPTIONS]

REQUIRED:
    --dir <path>          sequence directory: a TUM sequence (rgb.txt,
                          groundtruth.txt, rgb/*.png) or a COLMAP/ETH3D scene
                          (images/*.png with a sparse/ text model)
    --format <F>          auto (default), tum, or colmap

OPTIONS:
    --frames <N>          number of views handed to the mapper   [default: 20]
    --stride <S>          index step between consecutive views   [default: 8]
    --features <K>        ORB features per frame                 [default: 1500]
    --ratio <R>           Lowe ratio-test threshold              [default: 0.75]
    --window <W>          sequential pair window: view i pairs with
                          i+1..=i+W                             [default: 3]
    --f-threshold <P>     fundamental RANSAC Sampson threshold, px
                          [default: 1.5]
    --h-threshold <P>     homography RANSAC transfer threshold, px [default: 1.5]
    --h-iters <N>         homography RANSAC iterations           [default: 500]
    --planar-margin <S>   homography-vs-essential score margin
                          required to exclude a seed pair          [default: 0.05]
    --retrieval <V>       enable the bag-of-words widening with a vocabulary of
                          V words (default: sequential pairs only)
    --neighbours <M>      retrieved neighbours per view when --retrieval is set
                                                                 [default: 8]
    --max-dt <T>          timestamp association tolerance, s     [default: 0.02]
    --min-seed-inliers <K> minimum inliers for a seed pair       [default: 20]
    --seed-hypotheses <H>  number of candidate seeds for which the incremental
                          registration is actually run          [default: 8]
    --seed-pair <A> <B>   force the seed pair (diagnostic)
    --min-parallax <D>    minimum triangulation parallax, degrees [default: 1.0]
    --max-reproj <P>      maximum observation reprojection, px    [default: 4.0]
    --no-retriangulate    keep each landmark at the depth of the first two views
                          that registered it (ablation switch)
    --pnp-iters <N>       PnP RANSAC iterations                  [default: 2000]
                                                                 [default: 0.25]
    --ba-every <N>        global bundle adjustment every N registrations (0 = never)
                                                                  [default: 10]
    --local-ba-window <N>  cameras in each local BA problem, including the new
                           camera; 0 disables local BA                [default: 6]
    --local-ba-max-points <N>     maximum landmarks in one local bundle
                                  adjustment [default: 800]
    --local-ba-min-overlap <N>
                           minimum shared landmarks for a co-visible camera
                                                                  [default: 10]
    --ba-iters <N>        local/global bundle-adjustment iterations per call
                                                                  [default: 10]
    --no-ba-final         skip the final bundle adjustment
    --ba-dense            run bundle adjustment on the dense sequential path
                          (robust kernel on, sparsity off)
    --seed <S>            deterministic seed for the vocabulary  [default: 0]
    --fx --fy --cx --cy   camera intrinsics overrides
                          [default: TUM RGB-D 640x480: 517.3 516.5 318.6 255.3]
    --repeat-check        run the mapper twice and report whether the two
                          reconstructions are bit-identical (determinism check)
    --outcomes            print the per-view registration outcome
    --pairs               print the per-pair matching/verification diagnostics
                          and the seed-candidate scores
    -h, --help            print this help

The reconstruction is monocular, so it is only defined up to a similarity
transform; the reported camera-centre error is the RMSE after a Sim(3) Umeyama
alignment to the ground-truth trajectory, and the rotation error is measured
after applying that alignment's rotation.
"
    .to_string()
}

/// Everything the run needs, parsed from the command line.
#[derive(Debug, Clone)]
struct Args {
    dir: PathBuf,
    format_name: String,
    frames: usize,
    stride: usize,
    features: usize,
    ratio: f32,
    window: usize,
    f_ransac_threshold_px: f64,
    h_ransac_threshold_px: f64,
    h_ransac_iters: usize,
    planar_score_margin: f64,
    retrieval: Option<usize>,
    neighbours: usize,
    max_dt: f64,
    min_seed_inliers: usize,
    seed_hypotheses: usize,
    seed_pair: Option<(usize, usize)>,
    min_parallax: f64,
    max_reproj: f64,
    retriangulate: bool,
    pnp_iters: usize,
    ba_every: usize,
    local_ba_window: usize,
    local_ba_min_overlap: usize,
    local_ba_max_points: usize,
    ba_iters: usize,
    ba_final: bool,
    ba_dense: bool,
    seed: u64,
    intrinsics: CameraIntrinsics,
    repeat_check: bool,
    outcomes: bool,
    pairs: bool,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("error: --dir is required\n");
        eprint!("{}", usage());
        std::process::exit(2);
    }
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return;
    }
    let mut args = match parse_args(&argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}\n");
            eprint!("{}", usage());
            std::process::exit(2);
        }
    };
    match run(&mut args) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

fn run(args: &mut Args) -> Result<(), String> {
    let total_started = Instant::now();

    // ---- 1. Load the sequence ----
    // "auto" picks COLMAP when a sparse model is present (ETH3D), else TUM.
    let colmap_like = match args.format_name.as_str() {
        "colmap" => true,
        "tum" => false,
        // "auto": a COLMAP model in any of the usual layouts means COLMAP.
        _ => [
            "sparse/0/images.txt",
            "sparse/images.txt",
            "colmap/sparse/0/images.txt",
            "dslr_calibration_undistorted/images.txt",
        ]
        .iter()
        .any(|rel| args.dir.join(rel).is_file()),
    };
    let (files, ground_truth, sequence_len, intrinsics_override) = if colmap_like {
        let (files, poses, intrinsics) = load_colmap_sequence(&args.dir)?;
        let n = files.len();
        (files, poses, n, Some(intrinsics))
    } else {
        let (files, poses, n) = load_sequence(&args.dir, args.max_dt)?;
        (files, poses, n, None)
    };
    if let Some(k) = intrinsics_override {
        // The COLMAP model carries the true intrinsics; the TUM defaults do not
        // apply to a different camera.
        println!(
            "intrinsics: from COLMAP model (fx={}, fy={}, cx={}, cy={})",
            k.fx, k.fy, k.cx, k.cy
        );
        args.intrinsics = k;
    }
    let indices: Vec<usize> = (0..args.frames)
        .map(|k| k * args.stride)
        .take_while(|&i| i < files.len())
        .collect();
    if indices.len() < 2 {
        return Err(format!(
            "only {} view(s) selected from {} associated frames; increase --frames, lower \
             --stride, or check --max-dt",
            indices.len(),
            files.len()
        ));
    }
    // Ground truth for the views actually handed to the mapper, parallel to
    // `views` (the sequence is strided, so `ground_truth[k]` is *not* the pose of
    // view `k`).
    let view_ground_truth: Vec<Pose> = indices.iter().map(|&index| ground_truth[index]).collect();

    // ---- 2. Extract ORB features on the selected frames ----
    let extract_started = Instant::now();
    let mut views: Vec<View> = Vec::with_capacity(indices.len());
    for &index in &indices {
        views.push(extract_view(&args.dir, &files[index], args.features)?);
    }
    let extract_time = extract_started.elapsed();

    let total_features: usize = views.iter().map(View::len).sum();
    let mean_features = total_features as f64 / views.len() as f64;

    // ---- 3. Map ----
    let config = MapperConfig {
        ratio: args.ratio,
        pair_selection: match args.retrieval {
            Some(vocab) => PairSelection::SequentialWithRetrieval {
                window: args.window,
                vocab_size: vocab,
                neighbours: args.neighbours,
                seed: args.seed,
            },
            None => PairSelection::Sequential {
                window: args.window,
            },
        },
        f_ransac_threshold_px: args.f_ransac_threshold_px,
        h_ransac_threshold_px: args.h_ransac_threshold_px,
        h_ransac_iters: args.h_ransac_iters,
        planar_score_margin: args.planar_score_margin,
        min_seed_inliers: args.min_seed_inliers,
        seed_hypotheses: args.seed_hypotheses,
        seed_pair: args.seed_pair,
        min_parallax_deg: args.min_parallax,
        max_reproj_px: args.max_reproj,
        retriangulate: args.retriangulate,
        pnp_ransac_iters: args.pnp_iters,
        ba_every: args.ba_every,
        local_ba_window: args.local_ba_window,
        local_ba_min_overlap: args.local_ba_min_overlap,
        local_ba_max_points: args.local_ba_max_points,
        ba_final: args.ba_final,
        ba_max_iterations: args.ba_iters,
        ba_use_sparsity: !args.ba_dense,
        ba_robust_kernel: args.ba_dense,
        ..MapperConfig::default()
    };

    let map_started = Instant::now();
    let mapping = map_views(&views, &args.intrinsics, &config);
    let map_time = map_started.elapsed();

    let repeat = if args.repeat_check {
        let started = Instant::now();
        let second = map_views(&views, &args.intrinsics, &config);
        let elapsed = started.elapsed();
        Some((second, elapsed))
    } else {
        None
    };

    print_config(args, sequence_len, indices.len(), mean_features, &config);
    print_report(&mapping, &view_ground_truth, &views, args);

    if let Some((second, elapsed)) = &repeat {
        println!();
        println!("determinism check (second run)");
        println!("  second run mapping : {:.3} s", elapsed.as_secs_f64());
        println!(
            "  bit-identical      : {}",
            if reconstructions_identical(&mapping, second) {
                "yes"
            } else {
                "NO"
            }
        );
    }

    println!();
    println!("timing");
    println!("  feature extraction : {:.3} s", extract_time.as_secs_f64());
    println!("  mapping            : {:.3} s", map_time.as_secs_f64());
    println!(
        "  total wall time    : {:.3} s",
        total_started.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Compare two reconstructions exactly (poses, points and observations).
fn reconstructions_identical(a: &Mapping, b: &Mapping) -> bool {
    if a.reconstruction.cameras.len() != b.reconstruction.cameras.len()
        || a.reconstruction.points.len() != b.reconstruction.points.len()
        || a.reconstruction.observations != b.reconstruction.observations
    {
        return false;
    }
    for ((view_a, pose_a), (view_b, pose_b)) in a
        .reconstruction
        .cameras
        .iter()
        .zip(b.reconstruction.cameras.iter())
    {
        if view_a != view_b {
            return false;
        }
        if pose_a.translation != pose_b.translation
            || pose_a.rotation.into_inner() != pose_b.rotation.into_inner()
        {
            return false;
        }
    }
    a.reconstruction
        .points
        .iter()
        .zip(b.reconstruction.points.iter())
        .all(|(pa, pb)| pa == pb)
}

/// Reprojection quality of the final reconstruction, in pixels.
struct ReprojectionQuality {
    /// Observations examined.
    total: usize,
    /// Observations whose point is behind (or exactly on) the image plane.
    behind: usize,
    /// RMS of the in-front observations, in pixels.
    rmse: f64,
    /// Median of the in-front observations, in pixels.
    median: f64,
}

/// Per-observation reprojection error of the final reconstruction.
///
/// Points behind their camera are counted separately rather than folded into the
/// RMS with a large penalty, which would swamp the statistic.
fn observation_reprojection(
    recon: &cv_sfm::mapper::Reconstruction,
    views: &[View],
    intrinsics: &CameraIntrinsics,
) -> ReprojectionQuality {
    let pose_of = |view: usize| -> Option<Pose> {
        recon
            .cameras
            .iter()
            .find(|&&(v, _)| v == view)
            .map(|&(_, pose)| pose)
    };

    let mut errors: Vec<f64> = Vec::new();
    let mut behind = 0usize;
    let mut total = 0usize;
    for (point, observations) in recon.points.iter().zip(recon.observations.iter()) {
        for &(view, keypoint) in observations {
            let Some(pose) = pose_of(view) else {
                continue;
            };
            total += 1;
            let camera = pose.rotation * point.coords + pose.translation;
            if camera[2] <= 1e-9 {
                behind += 1;
                continue;
            }
            let projected = intrinsics.project(&Point3::from(camera));
            let observed = views[view].keypoints[keypoint].pt();
            errors.push((projected - observed).norm());
        }
    }
    if errors.is_empty() {
        return ReprojectionQuality {
            total,
            behind,
            rmse: 0.0,
            median: 0.0,
        };
    }
    errors.sort_by(|a, b| a.total_cmp(b));
    let median = errors[errors.len() / 2];
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    ReprojectionQuality {
        total,
        behind,
        rmse,
        median,
    }
}

fn print_config(
    args: &Args,
    sequence_len: usize,
    selected: usize,
    mean_features: f64,
    config: &MapperConfig,
) {
    let pairs = match &config.pair_selection {
        PairSelection::Sequential { window } => format!("sequential, window {window}"),
        PairSelection::SequentialWithRetrieval {
            window,
            vocab_size,
            neighbours,
            seed,
        } => format!(
            "sequential window {window} + BoW retrieval ({vocab_size} words, {neighbours} \
             neighbours, seed {seed})"
        ),
    };
    let ba = format!(
        "local window {} / min overlap {} after every registration; global every {} \
         registrations{}, {} iters, {}",
        args.local_ba_window,
        args.local_ba_min_overlap,
        args.ba_every,
        if args.ba_final { " + final" } else { "" },
        args.ba_iters,
        if args.ba_dense {
            "dense sequential (robust kernel)"
        } else {
            "sparse/ctx path"
        }
    );

    println!("=== cv-sfm TUM incremental mapper ===");
    println!();
    println!("effective configuration");
    println!("  dir                : {}", args.dir.display());
    println!("  frames / stride    : {} / {}", args.frames, args.stride);
    println!(
        "  sequence frames    : {} associated, {} selected as views",
        sequence_len, selected
    );
    println!(
        "  features per frame : {} requested; {:.1} mean extracted",
        args.features, mean_features
    );
    println!("  ratio              : {:.3}", args.ratio);
    println!("  pair selection     : {pairs}");
    println!(
        "  fundamental RANSAC : {:.2} px (Sampson)",
        args.f_ransac_threshold_px
    );
    println!(
        "  homography RANSAC  : {:.2} px / {} iterations; planar margin {:.3}",
        args.h_ransac_threshold_px, args.h_ransac_iters, args.planar_score_margin
    );
    println!("  max-dt             : {:.4} s", args.max_dt);
    println!("  min-seed-inliers   : {}", args.min_seed_inliers);
    println!("  seed-hypotheses    : {}", args.seed_hypotheses);
    println!(
        "  seed-pair override : {}",
        match args.seed_pair {
            Some((a, b)) => format!("({a}, {b})"),
            None => "none".to_string(),
        }
    );
    println!("  min-parallax       : {:.2} deg", args.min_parallax);
    println!("  max-reproj         : {:.2} px", args.max_reproj);
    println!(
        "  retriangulate      : {}",
        if args.retriangulate { "yes" } else { "NO" }
    );
    println!(
        "  PnP                : {} iterations, min inliers {}",
        args.pnp_iters,
        cv_sfm::mapper::MapperConfig::default().min_pnp_inliers
    );
    println!("  bundle adjustment  : {ba}");
    println!(
        "  intrinsics         : fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
        args.intrinsics.fx, args.intrinsics.fy, args.intrinsics.cx, args.intrinsics.cy
    );
}

fn print_report(mapping: &Mapping, ground_truth: &[Pose], views: &[View], args: &Args) {
    let report = &mapping.report;
    let recon = &mapping.reconstruction;

    println!();
    println!("mapping");
    println!("  pairs selected     : {}", report.pairs_selected);
    println!("  pairs verified     : {}", report.pairs_verified);
    println!(
        "  planar/degenerate  : {} (kept in tracks, excluded from seeding)",
        report.pairs_planar
    );
    println!(
        "  pair epipolar RMS  : {:.3} px (mean over verified pairs)",
        report.mean_pair_epipolar_rmse_px
    );
    println!(
        "  seed pair          : {}",
        match report.seed {
            Some((a, b)) => format!("({a}, {b})"),
            None => "NONE — the reconstruction could not be seeded".to_string(),
        }
    );
    println!(
        "  seed hypotheses    : {} tried",
        report.seed_hypotheses_tried
    );
    println!(
        "  registered         : {} / {} views  ({:.2} %)",
        report.registered,
        report.views_supplied,
        report.registration_rate * 100.0
    );
    println!("  3D points          : {}", report.num_points);
    println!("  observations       : {}", report.num_observations);
    println!("  mean track length  : {:.3}", report.mean_track_length);
    println!("  local BA           : {} accepted", report.local_ba_runs);
    println!("  global BA          : {} accepted", report.ba_runs);

    // Failure breakdown.
    let mut unreachable = 0usize;
    let mut insufficient = 0usize;
    let mut pnp_failed = 0usize;
    let mut few_inliers = 0usize;
    for outcome in &report.outcomes {
        match outcome.failure {
            None => {}
            Some(cv_sfm::mapper::RegistrationFailure::Unreachable) => unreachable += 1,
            Some(cv_sfm::mapper::RegistrationFailure::InsufficientCorrespondences) => {
                insufficient += 1
            }
            Some(cv_sfm::mapper::RegistrationFailure::PnpEstimationFailed) => pnp_failed += 1,
            Some(cv_sfm::mapper::RegistrationFailure::TooFewInliers) => few_inliers += 1,
        }
    }
    println!();
    println!("failures");
    println!("  unreachable (no 3D point visible)     : {unreachable}");
    println!("  too few correspondences for PnP       : {insufficient}");
    println!("  PnP RANSAC failed                     : {pnp_failed}");
    println!("  PnP pose had too few inliers          : {few_inliers}");

    if args.outcomes {
        println!();
        println!("per-view outcomes");
        for outcome in &report.outcomes {
            match outcome.failure {
                None => println!(
                    "  view {:4}  registered  ({} correspondences, {} inliers)",
                    outcome.view, outcome.correspondences, outcome.inliers
                ),
                Some(reason) => println!(
                    "  view {:4}  FAILED      ({} correspondences, {} inliers) — {}",
                    outcome.view,
                    outcome.correspondences,
                    outcome.inliers,
                    reason.description()
                ),
            }
        }
    }

    if args.pairs {
        println!();
        println!("per-pair diagnostics (matches, F verification, and H-vs-E model selection)");
        let mut failed = 0usize;
        let mut match_sum = 0usize;
        let mut inlier_sum = 0usize;
        let mut verified_count = 0usize;
        for &(_, _, matches, inliers) in &report.pair_diagnostics {
            match_sum += matches;
            match inliers {
                Some(count) => {
                    inlier_sum += count;
                    verified_count += 1;
                }
                None => failed += 1,
            }
        }
        println!(
            "  {} pairs: {} verified, {} failed; mean matches {:.1}, mean verified inliers {:.1}",
            report.pair_diagnostics.len(),
            verified_count,
            failed,
            match_sum as f64 / report.pair_diagnostics.len().max(1) as f64,
            if verified_count == 0 {
                0.0
            } else {
                inlier_sum as f64 / verified_count as f64
            }
        );
        for &(a, b, matches, inliers) in &report.pair_diagnostics {
            match inliers {
                Some(count) => println!("  ({a:3}, {b:3})  matches {matches:5}  inliers {count:5}"),
                None => println!("  ({a:3}, {b:3})  matches {matches:5}  REJECTED"),
            }
        }
        for diagnostic in &report.pair_model_diagnostics {
            let a = diagnostic.a;
            let b = diagnostic.b;
            let matches = diagnostic.matches;
            let fundamental_inliers = diagnostic.fundamental_inliers;
            let homography_inliers = diagnostic.homography_inliers;
            let essential_score = diagnostic.essential_score;
            let homography_score = diagnostic.homography_score;
            let score_margin = diagnostic.score_margin;
            let decision = if diagnostic.planar {
                "PLANAR (tracks only)"
            } else {
                "seed-eligible"
            };
            println!(
                "  ({a:3}, {b:3})  matches {matches:5}  E/H inliers {fundamental_inliers:5}/\
                 {homography_inliers:5}  E/H score {essential_score:.4}/{homography_score:.4}  \
                 margin {score_margin:+.4}  {decision}"
            );
        }
        println!();
        println!("seed candidates (view a, view b, cleanly triangulating correspondences)");
        for &(a, b, good) in &report.seed_diagnostics {
            println!("  ({a:3}, {b:3})  score {good:5}");
        }
    }

    // ---- Accuracy against ground truth (Sim(3)-aligned) ----
    if recon.cameras.is_empty() {
        println!();
        println!("accuracy");
        println!("  not evaluated: no camera was registered");
        return;
    }

    // ---- Seed-pair check: how good is the relative pose that started the map?
    // Both sides are *relative* poses between the two seed cameras (cam a ->
    // cam b), which is the only comparison invariant to the gauge drift bundle
    // adjustment is free to introduce.
    if let Some((seed_a, seed_b)) = report.seed {
        let pose_of = |view: usize| -> Option<Pose> {
            recon
                .cameras
                .iter()
                .find(|&&(v, _)| v == view)
                .map(|&(_, pose)| pose)
        };
        if let (Some(pose_a), Some(pose_b)) = (pose_of(seed_a), pose_of(seed_b)) {
            // `pose_a`/`pose_b` are world-to-camera, so the cam-a -> cam-b
            // transform is `B ∘ A⁻¹`; the ground-truth cameras are stored
            // camera-to-world, where the same transform is `gt_b⁻¹ ∘ gt_a`.
            let estimated = pose_b.compose(&pose_a.inverse());
            let gt_relative = ground_truth[seed_b]
                .inverse()
                .compose(&ground_truth[seed_a]);
            let rotation_error = (gt_relative.rotation.inverse() * estimated.rotation)
                .angle()
                .to_degrees();
            let direction_error = estimated
                .translation
                .normalize()
                .dot(&gt_relative.translation.normalize())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            println!();
            println!("seed-pair check (relative pose vs ground truth)");
            println!("  seed pair          : ({seed_a}, {seed_b})");
            println!("  rotation error     : {rotation_error:.3} deg");
            println!("  translation dir.   : {direction_error:.3} deg");
            println!(
                "  gt baseline        : {:.4} m",
                gt_relative.translation.norm()
            );
        }
    }

    // ---- Reconstruction-internal reprojection quality (no ground truth) ----
    let reprojection = observation_reprojection(recon, views, &args.intrinsics);
    println!();
    println!("reconstruction-internal reprojection");
    println!("  observations       : {}", reprojection.total);
    println!(
        "  behind camera      : {} ({:.2} %)",
        reprojection.behind,
        reprojection.behind as f64 / reprojection.total.max(1) as f64 * 100.0
    );
    println!("  mean RMSE          : {:.3} px", reprojection.rmse);
    println!("  median error       : {:.3} px", reprojection.median);

    let est_poses: Vec<Pose> = recon
        .cameras
        .iter()
        .map(|&(_, pose)| pose.inverse())
        .collect();
    let gt_poses: Vec<Pose> = recon
        .cameras
        .iter()
        .map(|&(view, _)| ground_truth[view])
        .collect();
    let est_traj = Trajectory::from_poses(&est_poses);
    let gt_traj = Trajectory::from_poses(&gt_poses);
    let ate = est_traj.ate(&gt_traj, Alignment::Sim3);

    let align_rotation = ate
        .transform
        .map(|transform| transform.rotation)
        .unwrap_or_else(UnitQuaternion::identity);

    let mut rotation_sq = 0.0f64;
    for ((_, pose_cw), gt) in recon.cameras.iter().zip(gt_poses.iter()) {
        let est_wc = pose_cw.inverse();
        let error = (gt.rotation.inverse() * align_rotation * est_wc.rotation).angle();
        rotation_sq += error * error;
    }
    let rotation_rmse_deg = (rotation_sq / recon.cameras.len() as f64)
        .sqrt()
        .to_degrees();

    println!();
    println!("accuracy vs ground truth (Sim(3)-aligned)");
    println!("  cameras compared   : {}", recon.cameras.len());
    println!("  camera-centre RMSE : {:.4} m", ate.rmse);
    println!("  camera-centre mean : {:.4} m", ate.mean);
    println!("  camera-centre med. : {:.4} m", ate.median);
    println!("  camera-centre max  : {:.4} m", ate.max);
    println!("  rotation RMSE      : {:.4} deg", rotation_rmse_deg);
    println!(
        "  recovered scale    : {:.6} (estimated / ground truth)",
        ate.scale
    );
    println!(
        "  path length        : est {:.4} m vs gt {:.4} m",
        est_traj.path_length(),
        gt_traj.path_length()
    );

    // ---- 3D point accuracy against the ground-truth cameras ----
    // The alignment above is fitted to camera *centres* only, which leaves its
    // rotation underdetermined when the cameras are nearly collinear. Both the
    // rotation RMSE and the point check therefore also use a second, pose-aware
    // alignment that fits the rotation to the camera orientations directly and
    // only then solves for scale and translation from the centres.
    let pose_alignment = pose_alignment(&est_poses, &gt_poses);
    let align_rotation_pose = pose_alignment
        .map(|(rotation, _, _)| rotation)
        .unwrap_or_else(UnitQuaternion::identity);

    let mut rotation_pose_sq = 0.0f64;
    for ((_, pose_cw), gt) in recon.cameras.iter().zip(gt_poses.iter()) {
        let est_wc = pose_cw.inverse();
        let error = (gt.rotation.inverse() * align_rotation_pose * est_wc.rotation).angle();
        rotation_pose_sq += error * error;
    }
    let rotation_pose_rmse_deg = (rotation_pose_sq / recon.cameras.len() as f64)
        .sqrt()
        .to_degrees();

    // Project every estimated point through the *ground-truth* camera that
    // observed it, after mapping the point into the ground-truth gauge. This
    // separates "wrong points" from "wrong poses": a landmark whose track is
    // geometrically correct reprojects tightly here regardless of how good the
    // estimated poses are.
    let point_errors = pose_alignment.map(|(rotation, scale, translation)| {
        let mut errors: Vec<f64> = Vec::new();
        for (point, observations) in recon.points.iter().zip(recon.observations.iter()) {
            let aligned = scale * (rotation * point.coords) + translation;
            for &(view, keypoint) in observations {
                let gt = ground_truth[view];
                let camera = gt.rotation.inverse() * (aligned - gt.translation);
                if camera[2] <= 1e-9 {
                    continue;
                }
                let projected = args.intrinsics.project(&Point3::from(camera));
                let observed = views[view].keypoints[keypoint].pt();
                errors.push((projected - observed).norm());
            }
        }
        errors
    });

    println!();
    println!("pose-aware Sim(3) alignment (rotation fitted to orientations)");
    println!("  Rotation-only fit is used because the centre-based Umeyama rotation is");
    println!("  ill-conditioned for a near-collinear trajectory.");
    println!("  rotation RMSE      : {rotation_pose_rmse_deg:.4} deg");
    println!(
        "  scale / translation: {:.6} / ({:.4}, {:.4}, {:.4})",
        pose_alignment.map(|(_, s, _)| s).unwrap_or(1.0),
        pose_alignment.map(|(_, _, t)| t.x).unwrap_or(0.0),
        pose_alignment.map(|(_, _, t)| t.y).unwrap_or(0.0),
        pose_alignment.map(|(_, _, t)| t.z).unwrap_or(0.0)
    );

    if let Some(mut point_errors) = point_errors {
        if !point_errors.is_empty() {
            point_errors.sort_by(|a, b| a.total_cmp(b));
            let mean = point_errors.iter().sum::<f64>() / point_errors.len() as f64;
            let median = point_errors[point_errors.len() / 2];
            let within_3 = point_errors.iter().filter(|&&e| e <= 3.0).count();
            let within_5 = point_errors.iter().filter(|&&e| e <= 5.0).count();
            let total = point_errors.len() as f64;
            println!();
            println!("3D point accuracy through the ground-truth cameras (pose-aligned)");
            println!("  observations       : {}", point_errors.len());
            println!("  mean error         : {mean:.3} px");
            println!("  median error       : {median:.3} px");
            println!(
                "  within 3 px        : {within_3} ({:.2} %)",
                within_3 as f64 / total * 100.0
            );
            println!(
                "  within 5 px        : {within_5} ({:.2} %)",
                within_5 as f64 / total * 100.0
            );
        }
    }
}

/// Similarity transform `p -> s * R * p + t` mapping the estimated camera poses
/// onto the ground truth, with the rotation fitted to the orientations.
///
/// The rotation is the closest proper rotation to `mean_i R_gt_i R_est_i^T`
/// (a quaternion average, i.e. the principal eigenvector of `sum_i q_i q_iᵀ`).
/// Given that rotation, the scale and translation follow from the camera centres
/// by the usual Umeyama least-squares solution.
fn pose_alignment(
    estimated: &[Pose],
    ground_truth: &[Pose],
) -> Option<(UnitQuaternion<f64>, f64, Vector3<f64>)> {
    let n = estimated.len().min(ground_truth.len());
    if n == 0 {
        return None;
    }

    // Rotation: principal eigenvector of the quaternion scatter matrix.
    let mut scatter = nalgebra::Matrix4::<f64>::zeros();
    for (est, gt) in estimated[..n].iter().zip(ground_truth[..n].iter()) {
        let relative = gt.rotation * est.rotation.inverse();
        let q = relative.into_inner().coords;
        scatter += q * q.transpose();
    }
    let eigen = nalgebra::SymmetricEigen::new(scatter);
    let mut best = 0usize;
    for index in 1..4 {
        if eigen.eigenvalues[index] > eigen.eigenvalues[best] {
            best = index;
        }
    }
    let vector = eigen.eigenvectors.column(best);
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        vector[3], vector[0], vector[1], vector[2],
    ));

    // Scale and translation from the centres, with the rotation fixed.
    let mut mean_est = Vector3::zeros();
    let mut mean_gt = Vector3::zeros();
    for (est, gt) in estimated[..n].iter().zip(ground_truth[..n].iter()) {
        mean_est += est.translation;
        mean_gt += gt.translation;
    }
    mean_est /= n as f64;
    mean_gt /= n as f64;

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (est, gt) in estimated[..n].iter().zip(ground_truth[..n].iter()) {
        let rotated = rotation * (est.translation - mean_est);
        numerator += rotated.dot(&(gt.translation - mean_gt));
        denominator += rotated.norm_squared();
    }
    let scale = if denominator > 1e-15 {
        numerator / denominator
    } else {
        1.0
    };
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let translation = mean_gt - scale * (rotation * mean_est);
    Some((rotation, scale, translation))
}

// ---------------------------------------------------------------------------
// Dataset handling
// ---------------------------------------------------------------------------

/// Load a TUM sequence: the rgb file names, the associated ground-truth poses in
/// camera-to-world form, and the number of associated frames.
fn load_sequence(dir: &Path, max_dt: f64) -> Result<(Vec<String>, Vec<Pose>, usize), String> {
    if !dir.is_dir() {
        return Err(format!("directory not found: {}", dir.display()));
    }
    let rgb_path = dir.join("rgb.txt");
    let gt_path = dir.join("groundtruth.txt");
    let rgb = tum::read_index(&rgb_path)
        .map_err(|e| format!("failed to read {}: {e}", rgb_path.display()))?;
    let groundtruth = tum::read_groundtruth(&gt_path)
        .map_err(|e| format!("failed to read {}: {e}", gt_path.display()))?;

    let gt_index: Vec<tum::IndexEntry> = groundtruth
        .iter()
        .map(|pose| tum::IndexEntry {
            timestamp: pose.timestamp,
            filename: String::new(),
        })
        .collect();
    let associations = tum::associate(&rgb, &gt_index, max_dt);
    if associations.is_empty() {
        return Err(format!(
            "no frames in {} within {max_dt} s of a ground-truth pose; check --max-dt",
            dir.display()
        ));
    }

    let files: Vec<String> = associations
        .iter()
        .map(|&(rgb_index, _)| rgb[rgb_index].filename.clone())
        .collect();
    let poses: Vec<Pose> = associations
        .iter()
        .map(|&(_, gt_index)| groundtruth[gt_index].pose)
        .collect();
    let len = files.len();
    Ok((files, poses, len))
}

/// Load a COLMAP/ETH3D-style sequence: an image directory plus a COLMAP sparse
/// model in `sparse/0` (or `sparse/`), scoring against the registered poses.
///
/// ETH3D ships exactly this layout (`images/*.png` + a COLMAP text model), so the
/// same mapper and the same scoring work on it without a TUM index file. The
/// poses come from the model, so they are only used to score the reconstruction —
/// the mapper itself never sees them.
fn load_colmap_sequence(dir: &Path) -> Result<(Vec<String>, Vec<Pose>, CameraIntrinsics), String> {
    use cv_io::datasets::colmap;

    // ETH3D names the directory after the calibration rather than "sparse".
    let sparse = [
        "sparse/0",
        "sparse",
        "colmap/sparse/0",
        "dslr_calibration_undistorted",
    ]
    .iter()
    .map(|p| dir.join(p))
    .find(|p| p.join("images.txt").is_file())
    .ok_or_else(|| {
        format!(
            "no COLMAP text model found under {} (looked for sparse/0/images.txt)",
            dir.display()
        )
    })?;

    let cameras = colmap::read_cameras_text(sparse.join("cameras.txt"))
        .map_err(|e| format!("cameras.txt: {e}"))?;
    let images = colmap::read_images_text(sparse.join("images.txt"))
        .map_err(|e| format!("images.txt: {e}"))?;

    let by_id: std::collections::HashMap<u32, &colmap::Camera> =
        cameras.iter().map(|c| (c.id, c)).collect();
    // The mapper carries ONE camera model for the whole reconstruction, so a
    // scene whose views span several DSLRs cannot be reconstructed correctly
    // with a single intrinsics. ETH3D scenes have up to six cameras with
    // slightly different focal lengths, and silently using the first one's
    // values for every view produces wrong geometry (measured: electro
    // registered 17% with one camera's intrinsics applied to all views).
    // Restrict to the largest set of views sharing one camera.
    let mut per_camera: std::collections::HashMap<u32, Vec<&colmap::Image>> =
        std::collections::HashMap::new();
    for img in &images {
        per_camera.entry(img.camera_id).or_default().push(img);
    }
    let (chosen_camera, chosen_images) = per_camera
        .iter()
        .max_by_key(|(camera, imgs)| (imgs.len(), std::cmp::Reverse(**camera)))
        .map(|(camera, imgs)| (*camera, imgs.clone()))
        .ok_or_else(|| "images.txt contains no images".to_string())?;
    let camera = by_id
        .get(&chosen_camera)
        .ok_or_else(|| format!("camera {chosen_camera} referenced but absent from cameras.txt"))?;
    let intrinsics = camera
        .intrinsics
        .ok_or_else(|| format!("camera {chosen_camera} has no usable intrinsics"))?;
    if per_camera.len() > 1 {
        eprintln!(
            "note: scene spans {} cameras; using camera {chosen_camera} ({} of {} views)",
            per_camera.len(),
            chosen_images.len(),
            images.len()
        );
    }

    // COLMAP stores world-to-camera; the mapper and the scorer work in
    // camera-to-world, so invert.
    let mut entries: Vec<(String, Pose)> = chosen_images
        .iter()
        .filter_map(|img| {
            let pose_cw = img.pose.inverse();
            // ETH3D nests the frames one level deeper
            // (images/dslr_images_undistorted/*.JPG) while the COLMAP model
            // stores a bare file name, so search the usual layouts.
            let path = [
                dir.join("images").join(&img.name),
                dir.join("images/dslr_images_undistorted").join(&img.name),
                dir.join(&img.name),
            ]
            .into_iter()
            .find(|p| p.is_file());
            let path = path?;
            // `extract_view` resolves names against the sequence directory, so
            // return the path relative to that root.
            let relative = path
                .strip_prefix(dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| format!("images/{}", img.name));
            Some((relative, pose_cw))
        })
        .collect();
    // Deterministic order: by image name, so a rerun selects the same views.
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    if entries.is_empty() {
        return Err(format!(
            "no images from {} were found under {}/images",
            sparse.display(),
            dir.display()
        ));
    }
    let (files, poses) = entries.into_iter().unzip();
    Ok((files, poses, intrinsics))
}

/// Detect ORB features on one frame.
fn extract_view(dir: &Path, filename: &str, features: usize) -> Result<View, String> {
    let path = dir.join(filename);
    let image =
        image::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());
    let (_keypoints, descriptors) = orb_detect_and_compute(&gray, features.max(1));
    // Rebuild the keypoint list from the descriptors so the two stay parallel
    // even though ORB drops border keypoints with no descriptor.
    let keypoints = descriptors.iter().map(|d| d.keypoint).collect();
    Ok(View::new(keypoints, descriptors, width, height))
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut dir: Option<PathBuf> = None;
    let mut format_name: String = String::from("auto");
    let mut frames = 60usize;
    let mut stride = 5usize;
    let mut features = 1500usize;
    let mut ratio = 0.75f32;
    let mut window = 3usize;
    let mut f_ransac_threshold_px = 1.5f64;
    let mut h_ransac_threshold_px = 1.5f64;
    let mut h_ransac_iters = 500usize;
    let mut planar_score_margin = 0.05f64;
    let mut retrieval: Option<usize> = None;
    let mut neighbours = 5usize;
    let mut max_dt = 0.02f64;
    let mut min_seed_inliers = 20usize;
    let mut seed_hypotheses = 8usize;
    let mut seed_pair: Option<(usize, usize)> = None;
    let mut min_parallax = 1.0f64;
    let mut max_reproj = 4.0f64;
    let mut retriangulate = true;
    let mut pnp_iters = 2000usize;
    let mut ba_every = 10usize;
    let mut local_ba_window = 6usize;
    let mut local_ba_min_overlap = 10usize;
    let mut local_ba_max_points = 800usize;
    let mut ba_iters = 10usize;
    let mut ba_final = true;
    let mut ba_dense = false;
    let mut seed = 0u64;
    let mut fx = TUM_FX;
    let mut fy = TUM_FY;
    let mut cx = TUM_CX;
    let mut cy = TUM_CY;
    let mut repeat_check = false;
    let mut outcomes = false;
    let mut pairs = false;

    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--dir" => dir = Some(PathBuf::from(take(argv, &mut i, flag)?)),
            "--format" => format_name = take(argv, &mut i, flag)?,
            "--frames" => frames = parse(&take(argv, &mut i, flag)?, flag)?,
            "--stride" => stride = parse(&take(argv, &mut i, flag)?, flag)?,
            "--features" => features = parse(&take(argv, &mut i, flag)?, flag)?,
            "--ratio" => ratio = parse(&take(argv, &mut i, flag)?, flag)?,
            "--window" => window = parse(&take(argv, &mut i, flag)?, flag)?,
            "--f-threshold" => f_ransac_threshold_px = parse(&take(argv, &mut i, flag)?, flag)?,
            "--h-threshold" => h_ransac_threshold_px = parse(&take(argv, &mut i, flag)?, flag)?,
            "--h-iters" => h_ransac_iters = parse(&take(argv, &mut i, flag)?, flag)?,
            "--planar-margin" => planar_score_margin = parse(&take(argv, &mut i, flag)?, flag)?,
            "--retrieval" => retrieval = Some(parse(&take(argv, &mut i, flag)?, flag)?),
            "--neighbours" => neighbours = parse(&take(argv, &mut i, flag)?, flag)?,
            "--max-dt" => max_dt = parse(&take(argv, &mut i, flag)?, flag)?,
            "--min-seed-inliers" => min_seed_inliers = parse(&take(argv, &mut i, flag)?, flag)?,
            "--seed-hypotheses" => seed_hypotheses = parse(&take(argv, &mut i, flag)?, flag)?,
            "--seed-pair" => {
                let a = parse::<usize>(&take(argv, &mut i, flag)?, flag)?;
                let b = parse::<usize>(&take(argv, &mut i, flag)?, flag)?;
                seed_pair = Some((a, b));
            }
            "--min-parallax" => min_parallax = parse(&take(argv, &mut i, flag)?, flag)?,
            "--max-reproj" => max_reproj = parse(&take(argv, &mut i, flag)?, flag)?,
            "--no-retriangulate" => retriangulate = false,
            "--retriangulate" => retriangulate = true,
            "--pnp-iters" => pnp_iters = parse(&take(argv, &mut i, flag)?, flag)?,
            "--ba-every" => ba_every = parse(&take(argv, &mut i, flag)?, flag)?,
            "--local-ba-window" => local_ba_window = parse(&take(argv, &mut i, flag)?, flag)?,
            "--local-ba-max-points" => {
                local_ba_max_points = parse(&take(argv, &mut i, flag)?, flag)?
            }
            "--local-ba-min-overlap" => {
                local_ba_min_overlap = parse(&take(argv, &mut i, flag)?, flag)?
            }
            "--ba-iters" => ba_iters = parse(&take(argv, &mut i, flag)?, flag)?,
            "--no-ba-final" => ba_final = false,
            "--ba-dense" => ba_dense = true,
            "--seed" => seed = parse(&take(argv, &mut i, flag)?, flag)?,
            "--fx" => fx = parse(&take(argv, &mut i, flag)?, flag)?,
            "--fy" => fy = parse(&take(argv, &mut i, flag)?, flag)?,
            "--cx" => cx = parse(&take(argv, &mut i, flag)?, flag)?,
            "--cy" => cy = parse(&take(argv, &mut i, flag)?, flag)?,
            "--repeat-check" => repeat_check = true,
            "--outcomes" => outcomes = true,
            "--pairs" => pairs = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 1;
    }

    let dir = dir.ok_or_else(|| "--dir is required".to_string())?;
    if frames < 2 {
        return Err("--frames must be at least 2".to_string());
    }
    if stride == 0 {
        return Err("--stride must be at least 1".to_string());
    }
    if features == 0 {
        return Err("--features must be at least 1".to_string());
    }
    if !(ratio > 0.0 && ratio <= 1.0) {
        return Err("--ratio must be in (0, 1]".to_string());
    }
    if window == 0 {
        return Err("--window must be at least 1".to_string());
    }
    if !(h_ransac_threshold_px.is_finite() && h_ransac_threshold_px > 0.0) {
        return Err("--h-threshold must be positive".to_string());
    }
    if h_ransac_iters == 0 {
        return Err("--h-iters must be at least 1".to_string());
    }
    if !(planar_score_margin.is_finite() && planar_score_margin > 0.0) {
        return Err("--planar-margin must be positive".to_string());
    }
    if neighbours == 0 {
        return Err("--neighbours must be at least 1".to_string());
    }
    if !(max_dt.is_finite() && max_dt > 0.0) {
        return Err("--max-dt must be positive".to_string());
    }
    if min_seed_inliers < 8 {
        return Err("--min-seed-inliers must be at least 8".to_string());
    }
    if seed_hypotheses == 0 {
        return Err("--seed-hypotheses must be at least 1".to_string());
    }
    if pnp_iters < 64 {
        return Err("--pnp-iters must be at least 64".to_string());
    }
    if !(min_parallax.is_finite() && min_parallax >= 0.0) {
        return Err("--min-parallax must be non-negative".to_string());
    }
    if !(max_reproj.is_finite() && max_reproj > 0.0) {
        return Err("--max-reproj must be positive".to_string());
    }
    if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
        return Err("--fx and --fy must be positive".to_string());
    }
    if !(cx.is_finite() && cy.is_finite()) {
        return Err("--cx and --cy must be finite".to_string());
    }

    Ok(Args {
        dir,
        format_name,
        frames,
        stride,
        features,
        ratio,
        window,
        f_ransac_threshold_px,
        h_ransac_threshold_px,
        h_ransac_iters,
        planar_score_margin,
        retrieval,
        neighbours,
        max_dt,
        min_seed_inliers,
        seed_hypotheses,
        seed_pair,
        min_parallax,
        max_reproj,
        retriangulate,
        pnp_iters,
        ba_every,
        local_ba_window,
        local_ba_min_overlap,
        local_ba_max_points,
        ba_iters,
        ba_final,
        ba_dense,
        seed,
        intrinsics: CameraIntrinsics::new(fx, fy, cx, cy, 640, 480),
        repeat_check,
        outcomes,
        pairs,
    })
}

fn take(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value:?}"))
}
