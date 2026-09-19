//! Real-data benchmark for the visual localization pipeline on TUM RGB-D
//! sequences.
//!
//! This example is gated behind the `synthetic` feature (which pulls in the
//! `image` decoder and the `cv-eval` retrieval metrics used by the benchmark
//! support module):
//!
//! ```text
//! cargo run -p cv-localization --features synthetic --release \
//!     --example tum_benchmark -- \
//!     --db-dir /data/tum/rgbd_dataset_freiburg1_xyz \
//!     --db-frames 30 --query-frames 100 --stride 5 --features 1000
//! ```
//!
//! It reports plain, measured metrics — dataset summary, localization accuracy,
//! retrieval quality and timings — and prints the effective configuration at the
//! top so a run is reproducible. It never substitutes synthetic data and never
//! extrapolates: if fewer frames are available than requested, the actual counts
//! are reported.

use cv_core::CameraIntrinsics;
use cv_localization::benchmark::{run, BenchmarkConfig};
use std::path::PathBuf;

/// TUM RGB-D published intrinsics for the 640x480 rgb stream (fr1 / fr2 series
/// are within ~1 px of these values). Overridable with `--fx/--fy/--cx/--cy`.
const TUM_FX: f64 = 517.3;
const TUM_FY: f64 = 516.5;
const TUM_CX: f64 = 318.6;
const TUM_CY: f64 = 255.3;

fn usage() -> String {
    "\
cv-localization TUM benchmark

USAGE:
    tum_benchmark --db-dir <path> [OPTIONS]

REQUIRED:
    --db-dir <path>        TUM sequence directory containing rgb.txt,
                           groundtruth.txt and rgb/*.png

OPTIONS:
    --query-dir <path>     second sequence for the cross-sequence case
                           (defaults to --db-dir; same-sequence queries are then
                           drawn from the second half of the sequence)
    --db-frames <N>        database frames to select  [default: 20]
    --query-frames <M>     query frames to select      [default: 50]
    --stride <S>           index stride within each sequence [default: 1]
    --features <K>         ORB features per frame      [default: 1000]
    --ratio <R>            Lowe ratio-test threshold   [default: 0.75]
    --hit-radius <D>       retrieval hit radius, metres [default: 1.0]
    --max-dt <T>           timestamp association tolerance, seconds [default: 0.02]
    --tri-reproj <P>       triangulation reprojection tolerance, pixels [default: 3.0]
    --vocab <K>            train a BoW vocabulary of K words on the database
                           descriptors (default: no vocabulary)
    --seed <S>             deterministic vocabulary seed [default: 0]
    --candidates <C>       retrieval candidates the localizer tries per query [default: 10]
    --fx --fy --cx --cy    camera intrinsics overrides
                           [default: TUM RGB-D 640x480: 517.3 516.5 318.6 255.3]
    -h, --help             print this help

The report lists what each number means. Translation error is the Euclidean
distance between the estimated and ground-truth camera centres (metres); rotation
error is the relative camera rotation (degrees); success requires a PnP pose that
passed the localizer's inlier threshold.
"
    .to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("error: --db-dir is required\n");
        eprint!("{}", usage());
        std::process::exit(2);
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return;
    }

    let config = match parse_args(&args) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: {err}\n");
            eprint!("{}", usage());
            std::process::exit(2);
        }
    };

    match run(&config) {
        Ok(report) => print_report(&config, &report),
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

/// Parse the command line into a [`BenchmarkConfig`].
fn parse_args(args: &[String]) -> Result<BenchmarkConfig, String> {
    let mut db_dir: Option<PathBuf> = None;
    let mut query_dir: Option<PathBuf> = None;
    let mut db_frames = 20usize;
    let mut query_frames = 50usize;
    let mut stride = 1usize;
    let mut features = 1000usize;
    let mut ratio = 0.75f32;
    let mut hit_radius = 1.0f64;
    let mut max_dt = 0.02f64;
    let mut tri_reproj = 3.0f64;
    let mut vocab: Option<usize> = None;
    let mut seed = 0u64;
    let mut candidates = 10usize;
    let mut fx = TUM_FX;
    let mut fy = TUM_FY;
    let mut cx = TUM_CX;
    let mut cy = TUM_CY;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--db-dir" => db_dir = Some(PathBuf::from(take_value(args, &mut i, flag)?)),
            "--query-dir" => query_dir = Some(PathBuf::from(take_value(args, &mut i, flag)?)),
            "--db-frames" => {
                db_frames = parse_usize(&take_value(args, &mut i, flag)?, flag)?;
            }
            "--query-frames" => {
                query_frames = parse_usize(&take_value(args, &mut i, flag)?, flag)?;
            }
            "--stride" => stride = parse_usize(&take_value(args, &mut i, flag)?, flag)?,
            "--features" => features = parse_usize(&take_value(args, &mut i, flag)?, flag)?,
            "--ratio" => ratio = parse_f32(&take_value(args, &mut i, flag)?, flag)?,
            "--hit-radius" => hit_radius = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--max-dt" => max_dt = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--tri-reproj" => tri_reproj = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--vocab" => vocab = Some(parse_usize(&take_value(args, &mut i, flag)?, flag)?),
            "--seed" => seed = parse_u64(&take_value(args, &mut i, flag)?, flag)?,
            "--candidates" => candidates = parse_usize(&take_value(args, &mut i, flag)?, flag)?,
            "--fx" => fx = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--fy" => fy = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--cx" => cx = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            "--cy" => cy = parse_f64(&take_value(args, &mut i, flag)?, flag)?,
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 1;
    }

    let db_dir = db_dir.ok_or_else(|| "--db-dir is required".to_string())?;
    if db_frames < 2 {
        return Err("--db-frames must be at least 2".to_string());
    }
    if query_frames == 0 {
        return Err("--query-frames must be at least 1".to_string());
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
    if !(hit_radius.is_finite() && hit_radius > 0.0) {
        return Err("--hit-radius must be positive".to_string());
    }
    if !(max_dt.is_finite() && max_dt > 0.0) {
        return Err("--max-dt must be positive".to_string());
    }
    if !(tri_reproj.is_finite() && tri_reproj > 0.0) {
        return Err("--tri-reproj must be positive".to_string());
    }
    if vocab == Some(0) {
        return Err("--vocab must be at least 1".to_string());
    }
    if candidates == 0 {
        return Err("--candidates must be at least 1".to_string());
    }
    if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
        return Err("--fx and --fy must be positive".to_string());
    }
    if !(cx.is_finite() && cy.is_finite()) {
        return Err("--cx and --cy must be finite".to_string());
    }

    // The camera size is only known once frames are loaded; 640x480 is the TUM
    // RGB-D stream size and is used for the intrinsics metadata.
    let intrinsics = CameraIntrinsics::new(fx, fy, cx, cy, 640, 480);

    Ok(BenchmarkConfig {
        db_dir,
        query_dir,
        db_frames,
        query_frames,
        stride,
        features,
        ratio,
        hit_radius,
        max_dt,
        tri_reproj,
        vocab,
        seed,
        candidates,
        intrinsics,
    })
}

/// Consume the value that follows the flag at `args[*i]`, advancing `*i`.
fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value.parse().map_err(|_| {
        format!("invalid value for {flag}: {value:?} (expected a non-negative integer)")
    })
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value.parse().map_err(|_| {
        format!("invalid value for {flag}: {value:?} (expected a non-negative integer)")
    })
}

fn parse_f64(value: &str, flag: &str) -> Result<f64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value:?} (expected a number)"))
}

fn parse_f32(value: &str, flag: &str) -> Result<f32, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value:?} (expected a number)"))
}

/// Format a metric, rendering non-finite values (no successes) as `n/a`.
fn metric(value: f64, precision: usize, unit: &str) -> String {
    if !value.is_finite() {
        "n/a".to_string()
    } else if unit.is_empty() {
        format!("{:.*}", precision, value)
    } else {
        format!("{:.*} {}", precision, value, unit)
    }
}

fn print_report(config: &BenchmarkConfig, report: &cv_localization::benchmark::BenchmarkReport) {
    let summary = &report.summary;
    let localization = &report.localization;
    let retrieval = &report.retrieval;
    let timing = &report.timing;

    let query_description = match &config.query_dir {
        Some(dir) => format!("{} (cross-sequence)", dir.display()),
        None => format!(
            "{} (same sequence; query frames from the second half)",
            config.db_dir.display()
        ),
    };
    let vocab_description = match config.vocab {
        Some(k) => format!(
            "{k} words (trained on database descriptors, seed {})",
            config.seed
        ),
        None => "none (descriptor-count retrieval)".to_string(),
    };

    println!("=== cv-localization TUM benchmark report ===");
    println!();
    println!("effective configuration");
    println!("  db-dir             : {}", config.db_dir.display());
    println!("  query-dir          : {query_description}");
    println!("  db-frames          : {}", config.db_frames);
    println!("  query-frames       : {}", config.query_frames);
    println!("  stride             : {}", config.stride);
    println!("  features           : {}", config.features);
    println!("  ratio              : {:.3}", config.ratio);
    println!("  hit-radius         : {:.3} m", config.hit_radius);
    println!("  max-dt             : {:.4} s", config.max_dt);
    println!("  tri-reproj         : {:.3} px", config.tri_reproj);
    println!("  vocab              : {vocab_description}");
    println!("  candidates         : {}", config.candidates);
    println!("  seed               : {}", config.seed);
    println!(
        "  intrinsics         : fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
        config.intrinsics.fx, config.intrinsics.fy, config.intrinsics.cx, config.intrinsics.cy
    );
    println!(
        "  image size         : {}x{}",
        summary.image_width, summary.image_height
    );
    println!();

    println!("dataset");
    println!(
        "  frames loaded      : {} (decoded image files)",
        summary.frames_loaded
    );
    println!(
        "  associated frames  : {} database sequence, {} query sequence",
        summary.db_sequence_frames, summary.query_sequence_frames
    );
    println!(
        "  database frames    : {} / {} requested",
        summary.db_frames, summary.db_frames_requested
    );
    println!(
        "  query frames       : {} / {} requested",
        summary.query_frames, summary.query_frames_requested
    );
    println!(
        "  features per frame : {} requested; {:.1} mean extracted",
        summary.features_requested, summary.mean_features_per_frame
    );
    println!("  landmarks in map   : {}", summary.landmarks);
    println!(
        "  observations/land  : {:.2}",
        summary.mean_observations_per_landmark
    );
    println!();

    println!("localization");
    println!("  queries attempted  : {}", localization.attempted);
    println!("  successes          : {}", localization.succeeded);
    println!(
        "  success rate       : {:.2} %",
        localization.success_rate * 100.0
    );
    println!(
        "  translation error  : median {}, mean {}",
        metric(localization.median_translation_m, 4, "m"),
        metric(localization.mean_translation_m, 4, "m")
    );
    println!(
        "  rotation error     : median {}, mean {}",
        metric(localization.median_rotation_deg, 3, "deg"),
        metric(localization.mean_rotation_deg, 3, "deg")
    );
    println!(
        "  mean inliers       : {}",
        metric(localization.mean_inliers, 1, "")
    );
    println!(
        "  mean reproj. RMSE  : {}",
        metric(localization.mean_reprojection_rmse_px, 3, "px")
    );
    println!();

    println!("retrieval (independent of PnP)");
    println!(
        "  hit rate @1        : {:.4}   (top-1 contains a frame within the hit radius)",
        retrieval.hit_rate_at_1
    );
    println!("  hit rate @5        : {:.4}", retrieval.hit_rate_at_5);
    println!("  hit rate @10       : {:.4}", retrieval.hit_rate_at_10);
    println!(
        "  recall@1/5/10      : {:.4} / {:.4} / {:.4}   (IR convention: fraction of all relevant frames retrieved)",
        retrieval.recall_at_1, retrieval.recall_at_5, retrieval.recall_at_10
    );
    println!("  mAP                : {:.4}", retrieval.map);
    println!(
        "  queries with DB hit: {} / {}",
        retrieval.queries_with_hit, retrieval.queries
    );
    println!();

    println!("timing");
    println!(
        "  mean query         : {} (retrieval + match + PnP)",
        metric(timing.mean_query_ms, 2, "ms")
    );
    println!("  total wall time    : {:.3} s", timing.total_wall_s);
}
