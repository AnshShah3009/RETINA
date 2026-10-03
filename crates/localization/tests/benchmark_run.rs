//! Integration tests for the `synthetic`-gated benchmark runner
//! (`cv_localization::benchmark::run`).
//!
//! These build tiny TUM-format sequences on disk in a uniquely named temporary
//! directory (process id + counter), run the public entry point over them and
//! check the report against the inputs. Without the `synthetic` feature this
//! file compiles to an empty test binary, matching the module's own gate.

#![cfg(feature = "synthetic")]

use cv_core::CameraIntrinsics;
use cv_localization::benchmark::{run, BenchmarkConfig};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// What the frames look like.
#[derive(Clone, Copy)]
enum Frames {
    /// A constant grey image: ORB finds no features, so no map is built.
    Blank,
    /// A sliding window over a fixed random texture: consecutive frames share
    /// content, so ORB features match and a map is built. The window shift and
    /// the ground-truth translation are consistent (20 px at fx = 500 and the
    /// ground truth moving 0.04 m per frame is a 1 m deep scene).
    Panning,
}

/// A TUM-format sequence that deletes its directory on drop.
struct TempSequence {
    dir: PathBuf,
}

impl TempSequence {
    fn new(label: &str, frames: usize, kind: Frames) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cv_localization_bench_{}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
            label
        ));
        assert!(
            dir.starts_with(std::env::temp_dir()),
            "refusing to build a test sequence outside the temp directory"
        );
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(dir.join("rgb")).unwrap();

        let mut rgb = String::from("# timestamp filename\n");
        let mut groundtruth = String::from("# timestamp tx ty tz qx qy qz qw\n");
        for i in 0..frames {
            let timestamp = i as f64 * 0.1;
            let name = format!("rgb/{i:04}.png");
            rgb.push_str(&format!("{timestamp:.6} {name}\n"));
            groundtruth.push_str(&format!(
                "{timestamp:.6} {:.6} 0.0 0.0 0.0 0.0 0.0 1.0\n",
                0.04 * i as f64
            ));

            let image = match kind {
                Frames::Blank => image::GrayImage::from_pixel(64, 64, image::Luma([128u8])),
                Frames::Panning => {
                    let mut img = image::GrayImage::new(64, 64);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        *pixel = image::Luma([texture(x as usize + 20 * i, y as usize)]);
                    }
                    img
                }
            };
            image.save(dir.join(&name)).unwrap();
        }
        fs::write(dir.join("rgb.txt"), rgb).unwrap();
        fs::write(dir.join("groundtruth.txt"), groundtruth).unwrap();

        Self { dir }
    }

    fn config(&self, db_frames: usize, query_frames: usize) -> BenchmarkConfig {
        BenchmarkConfig {
            db_dir: self.dir.clone(),
            query_dir: None,
            db_frames,
            query_frames,
            stride: 1,
            features: 200,
            ratio: 0.75,
            hit_radius: 1.0,
            max_dt: 0.02,
            tri_reproj: 3.0,
            match_window: 2,
            vocab: None,
            seed: 0,
            candidates: 10,
            intrinsics: CameraIntrinsics::new(500.0, 500.0, 32.0, 24.0, 64, 64),
        }
    }
}

impl Drop for TempSequence {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Deterministic per-pixel noise (xorshift-style mix of the coordinates).
fn texture(x: usize, y: usize) -> u8 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h & 0xFF) as u8
}

/// No features means no landmarks: the observation statistics have no
/// denominator, so they must be NaN (the median always was; the mean reported a
/// measured-looking 0.0 before the fix). Everything else is pinned too: the
/// report counts what was loaded and attempted.
#[test]
fn blank_sequence_reports_an_empty_map_honestly() {
    let sequence = TempSequence::new("blank", 10, Frames::Blank);
    let report = run(&sequence.config(4, 2)).expect("blank sequence must run");

    assert_eq!(report.summary.frames_loaded, 6);
    assert_eq!(report.summary.db_frames, 4);
    assert_eq!(report.summary.query_frames, 2);
    assert_eq!(report.summary.image_width, 64);
    assert_eq!(report.summary.image_height, 64);
    assert_eq!(report.summary.landmarks, 0);
    assert_eq!(report.summary.triangulated_tracks, 0);
    assert_eq!(report.summary.rejected_tracks, 0);
    assert!(
        report.summary.mean_observations_per_landmark.is_nan(),
        "empty map reported a mean of {}",
        report.summary.mean_observations_per_landmark
    );
    assert!(report.summary.median_observations_per_landmark.is_nan());

    assert_eq!(report.localization.attempted, 2);
    assert_eq!(report.localization.succeeded, 0);
    assert_eq!(
        report.localization.success_rate, 0.0,
        "two queries were attempted and both failed, so 0.0 is the measured value"
    );
    assert!(report.localization.mean_translation_m.is_nan());
    assert!(report.localization.mean_rotation_deg.is_nan());

    assert_eq!(report.retrieval.queries, 2);
    // Every ground-truth pose is at the origin, so every database frame is
    // within the 1 m hit radius of every query.
    assert_eq!(report.retrieval.queries_with_hit, 2);
    assert_eq!(report.retrieval.hit_rate_at_1, 0.0);
    assert!(report.timing.total_wall_s >= 0.0);
}

/// The control for the test above: with a shared texture the map is non-empty,
/// the mean/median observation counts are real numbers, and the per-track
/// accounting is exact (`triangulated == landmarks + rejected`).
#[test]
fn panning_sequence_builds_a_map_and_reports_consistent_counts() {
    let sequence = TempSequence::new("panning", 10, Frames::Panning);
    let report = run(&sequence.config(4, 2)).expect("panning sequence must run");

    // Measured on this fixture: 31 landmarks, 31/0 tracks, 2.00 mean and median
    // observations/landmark, 66.8 features/frame, hit@1 = 0.5.

    assert_eq!(report.summary.frames_loaded, 6);
    assert_eq!(
        report.summary.landmarks,
        report.summary.triangulated_tracks - report.summary.rejected_tracks
    );
    assert!(
        report.summary.landmarks > 0,
        "the panning control must triangulate at least one landmark, got {}",
        report.summary.landmarks
    );
    assert!(report.summary.mean_observations_per_landmark >= 2.0);
    assert!(report.summary.median_observations_per_landmark >= 2.0);
    assert!(report.summary.mean_features_per_frame > 0.0);

    // Success rate is the measured quotient, whatever it happens to be.
    let expected = report.localization.succeeded as f64 / report.localization.attempted as f64;
    assert_eq!(report.localization.success_rate, expected);
    assert!(report.localization.attempted >= 1);
}

/// Non-finite intrinsics must not hang the whole run either: the map builder's
/// triangulation rejects them (non-finite projection matrices) and the localizer
/// rejects them at its boundary, so the report is "nothing built, nothing
/// localized" rather than a process that never returns.
#[test]
fn non_finite_intrinsics_do_not_hang_the_run() {
    let sequence = TempSequence::new("nan_intrinsics", 10, Frames::Panning);
    let mut config = sequence.config(4, 2);
    config.intrinsics = CameraIntrinsics::new(f64::NAN, 500.0, 32.0, 24.0, 64, 64);
    let report = run(&config).expect("run must return a report");

    assert_eq!(report.summary.landmarks, 0);
    assert_eq!(
        report.summary.rejected_tracks,
        report.summary.triangulated_tracks
    );
    assert_eq!(report.localization.succeeded, 0);
    assert_eq!(report.localization.success_rate, 0.0);
}

/// The documented error paths: every failure is a message, never a panic, and
/// never a report built from a partial load.
#[test]
fn run_reports_input_errors_without_panicking() {
    // Missing directory.
    let missing = std::env::temp_dir().join(format!(
        "cv_localization_bench_{}_does_not_exist",
        std::process::id()
    ));
    let sequence = TempSequence::new("errors", 10, Frames::Blank);
    let mut config = sequence.config(4, 2);
    config.db_dir = missing;
    let err = run(&config).expect_err("a missing directory must fail");
    assert!(err.contains("not found"), "error was {err:?}");

    // Too few database frames.
    let sequence = TempSequence::new("errors_db", 10, Frames::Blank);
    let err = run(&sequence.config(1, 2)).expect_err("db_frames = 1 must fail");
    assert!(err.contains("--db-frames"), "error was {err:?}");

    // No query frames requested.
    let sequence = TempSequence::new("errors_query", 10, Frames::Blank);
    let err = run(&sequence.config(4, 0)).expect_err("query_frames = 0 must fail");
    assert!(err.contains("--query-frames"), "error was {err:?}");

    // A sequence too short to split into non-empty halves.
    let sequence = TempSequence::new("errors_short", 2, Frames::Blank);
    let err = run(&sequence.config(4, 2)).expect_err("2 frames must fail");
    assert!(err.contains("at least 4"), "error was {err:?}");

    // rgb.txt present, groundtruth.txt missing.
    let sequence = TempSequence::new("errors_gt", 10, Frames::Blank);
    fs::remove_file(sequence.dir.join("groundtruth.txt")).unwrap();
    let err = run(&sequence.config(4, 2)).expect_err("missing ground truth must fail");
    assert!(err.contains("ground-truth"), "error was {err:?}");
}
