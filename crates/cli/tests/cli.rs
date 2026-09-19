//! End-to-end tests for the `cv-bench` binary.
//!
//! Each test writes small input files into a self-cleaning temporary directory,
//! runs the compiled binary with [`std::process::Command`] and asserts on a few
//! key output lines (never the whole transcript) plus the exit code.

use nalgebra::{UnitQuaternion, Vector3};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A unique temporary directory that removes itself on drop (std-only, so the
/// crate takes on no extra dependency).
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("cv_cli_{tag}_{}_{seq}_{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.path.join(name);
        std::fs::write(&path, contents).expect("write temp file");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cv-bench"))
        .args(args)
        .output()
        .expect("run cv-bench")
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Extract the numeric value printed on the line that starts with `label`.
fn value_after(output: &str, label: &str) -> f64 {
    for line in output.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix(label) {
            let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
            if let Some(token) = rest.split_whitespace().next() {
                if let Ok(value) = token.parse::<f64>() {
                    return value;
                }
            }
        }
    }
    panic!("no '{label}' value found in output:\n{output}");
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

#[test]
fn trajectory_se3_transformed_tum_reports_near_zero_ate() {
    let dir = TempDir::new("traj");

    let ground_truth_points = [
        (0.0_f64, 0.0_f64, 0.0_f64),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ];
    // A known rigid SE(3) transform the estimate is generated with.
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7);
    let translation = Vector3::new(1.0, -2.0, 3.0);

    let mut ground_truth = String::from("# timestamp tx ty tz qx qy qz qw\n");
    let mut estimate = String::from("# timestamp tx ty tz qx qy qz qw\n");
    for (index, (x, y, z)) in ground_truth_points.iter().enumerate() {
        let timestamp = 1.0 + index as f64 * 0.1;
        ground_truth.push_str(&format!("{timestamp:.6} {x:.6} {y:.6} {z:.6} 0 0 0 1\n"));

        let point = rotation * Vector3::new(*x, *y, *z) + translation;
        estimate.push_str(&format!(
            "{timestamp:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n",
            point.x, point.y, point.z, rotation.i, rotation.j, rotation.k, rotation.w
        ));
    }

    let ground_truth_path = dir.write("groundtruth.txt", &ground_truth);
    let estimate_path = dir.write("estimate.txt", &estimate);

    let output = run(&[
        "trajectory",
        "--estimate",
        path_str(&estimate_path),
        "--ground-truth",
        path_str(&ground_truth_path),
        "--format",
        "tum",
        "--align",
        "se3",
    ]);

    assert!(output.status.success(), "stderr: {}", stderr_of(&output));
    let stdout = stdout_of(&output);
    assert!(
        value_after(&stdout, "ATE rmse") < 1e-6,
        "SE(3)-aligned ATE should be ~0:\n{stdout}"
    );
    assert!(value_after(&stdout, "ATE max") < 1e-6, "{stdout}");
    assert_eq!(value_after(&stdout, "paired poses"), 5.0, "{stdout}");
}

#[test]
fn trajectory_default_alignment_matches_explicit_se3() {
    let dir = TempDir::new("traj_default");
    let ground_truth = "0.0 0.0 0.0 0.0 0 0 0 1\n0.1 1.0 0.0 0.0 0 0 0 1\n";
    let estimate = "0.0 5.0 6.0 -7.0 0 0 0 1\n0.1 6.0 6.0 -7.0 0 0 0 1\n";
    let ground_truth_path = dir.write("gt.txt", ground_truth);
    let estimate_path = dir.write("est.txt", estimate);

    let output = run(&[
        "trajectory",
        "--estimate",
        path_str(&estimate_path),
        "--ground-truth",
        path_str(&ground_truth_path),
        "--format",
        "tum",
    ]);

    assert!(output.status.success(), "stderr: {}", stderr_of(&output));
    let stdout = stdout_of(&output);
    // Default alignment is SE(3); a pure translation between the two is removed.
    assert!(value_after(&stdout, "ATE rmse") < 1e-6, "{stdout}");
    assert!(
        stdout.contains("alignment:          se3"),
        "default alignment should be se3:\n{stdout}"
    );
}

#[test]
fn model_reports_known_registration_rate_and_points() {
    let dir = TempDir::new("model");
    let cameras = dir.write(
        "cameras.txt",
        concat!(
            "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n",
            "1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n",
        ),
    );
    // Three images: 1 and 3 are registered (2 observations each), 2 is not.
    let images = dir.write(
        "images.txt",
        concat!(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
            "1 1 0 0 0 0 0 0 1 image1.jpg\n",
            "10.0 20.0 1 30.0 40.0 2\n",
            "2 1 0 0 0 1 0 0 1 image2.jpg\n",
            "\n",
            "3 1 0 0 0 2 0 0 1 image3.jpg\n",
            "50.0 60.0 1 70.0 80.0 -1\n",
        ),
    );
    // Two points: track lengths 2 and 0, errors 0.5 and 1.25.
    let points = dir.write(
        "points3D.txt",
        concat!(
            "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n",
            "1 0.5 -1.5 2.5 255 128 0 0.5 1 0 2 3\n",
            "2 -1.0 0.0 1.0 0 255 0 1.25\n",
        ),
    );

    let output = run(&[
        "model",
        "--images",
        path_str(&images),
        "--points3d",
        path_str(&points),
        "--cameras",
        path_str(&cameras),
    ]);

    assert!(output.status.success(), "stderr: {}", stderr_of(&output));
    let stdout = stdout_of(&output);

    assert_eq!(value_after(&stdout, "cameras"), 1.0, "{stdout}");
    assert_eq!(value_after(&stdout, "images"), 3.0, "{stdout}");
    assert_eq!(value_after(&stdout, "registered images"), 2.0, "{stdout}");
    assert!(
        (value_after(&stdout, "registration rate") - 2.0 / 3.0).abs() < 1e-6,
        "{stdout}"
    );
    assert_eq!(value_after(&stdout, "total observations"), 4.0, "{stdout}");
    assert!(
        (value_after(&stdout, "mean observations per registered image") - 2.0).abs() < 1e-9,
        "{stdout}"
    );
    assert_eq!(value_after(&stdout, "points3D"), 2.0, "{stdout}");
    assert!(
        (value_after(&stdout, "mean track length") - 1.0).abs() < 1e-9,
        "{stdout}"
    );
    assert!(
        (value_after(&stdout, "mean reprojection error") - 0.875).abs() < 1e-9,
        "{stdout}"
    );
}

#[test]
fn model_without_optional_files_still_reports() {
    let dir = TempDir::new("model_min");
    let images = dir.write(
        "images.txt",
        concat!("1 1 0 0 0 0 0 0 1 image1.jpg\n", "10.0 20.0 1\n",),
    );

    let output = run(&["model", "--images", path_str(&images)]);
    assert!(output.status.success(), "stderr: {}", stderr_of(&output));
    let stdout = stdout_of(&output);
    assert_eq!(value_after(&stdout, "registration rate"), 1.0, "{stdout}");
    assert!(
        !stdout.contains("points3D"),
        "no points3D line expected:\n{stdout}"
    );
}

#[test]
fn retrieval_matches_hand_computed_metrics() {
    let dir = TempDir::new("retrieval");
    let predictions = dir.write("predictions.txt", "0,1,2\n3,4,5\n6,7,8\n");
    let ground_truth = dir.write("ground_truth.txt", "0,1\n4\n\n");

    let output = run(&[
        "retrieval",
        "--predictions",
        path_str(&predictions),
        "--ground-truth",
        path_str(&ground_truth),
        "--k",
        "2",
    ]);

    assert!(output.status.success(), "stderr: {}", stderr_of(&output));
    let stdout = stdout_of(&output);
    assert!(
        (value_after(&stdout, "recall@2") - 1.0).abs() < 1e-9,
        "{stdout}"
    );
    assert!(
        (value_after(&stdout, "precision@2") - 0.5).abs() < 1e-9,
        "{stdout}"
    );
    assert!(
        (value_after(&stdout, "mAP") - 0.75).abs() < 1e-9,
        "{stdout}"
    );
}

#[test]
fn missing_input_file_exits_nonzero_with_clear_message() {
    let output = run(&[
        "trajectory",
        "--estimate",
        "/definitely/not/here/estimate.txt",
        "--ground-truth",
        "/definitely/not/here/groundtruth.txt",
        "--format",
        "tum",
    ]);

    assert!(!output.status.success(), "expected failure");
    assert!(output.stdout.is_empty(), "no report on failure");
    let stderr = stderr_of(&output);
    assert!(stderr.contains("error"), "stderr: {stderr}");
}

#[test]
fn unknown_flag_exits_nonzero_with_clear_message() {
    let output = run(&[
        "retrieval",
        "--predictions",
        "p",
        "--ground-truth",
        "g",
        "--bogus",
        "x",
    ]);

    assert!(!output.status.success(), "expected failure");
    let stderr = stderr_of(&output);
    assert!(stderr.contains("--bogus"), "stderr: {stderr}");
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let output = run(&["frobnicate"]);
    assert!(!output.status.success());
    assert!(stderr_of(&output).contains("unknown subcommand"));
}

#[test]
fn help_and_no_arguments_print_usage() {
    for argv in [Vec::<&str>::new(), vec!["help"]] {
        let output = run(&argv);
        assert!(output.status.success(), "stderr: {}", stderr_of(&output));
        let stdout = stdout_of(&output);
        for expected in ["trajectory", "model", "retrieval", "--max-dt", "--k"] {
            assert!(
                stdout.contains(expected),
                "usage should mention '{expected}':\n{stdout}"
            );
        }
    }
}
