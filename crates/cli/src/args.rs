//! Minimal hand-rolled argument parser for the `cv-bench` binary.
//!
//! Every option is written as `--name value` (or `--name=value`); there are no
//! positional arguments and no boolean flags. Parsing is deliberately split from
//! execution so the whole surface can be unit-tested without touching the
//! filesystem. A parse failure is a plain [`String`] describing the problem,
//! never a panic.

use cv_eval::Alignment;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Default timestamp tolerance (seconds) for TUM association.
pub const DEFAULT_MAX_DT: f64 = 0.02;
/// Default frame gap for the relative pose error.
pub const DEFAULT_RPE_DELTA: usize = 1;

/// Trajectory file formats accepted by `cv-bench trajectory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// TUM RGB-D `groundtruth.txt`-style timestamped poses, associated by time.
    Tum,
    /// KITTI odometry `poses.txt` (row-major 3x4 `[R|t]`), index-aligned.
    Kitti,
    /// EuRoC MAV `state_groundtruth_estimate0/data.csv`, index-aligned.
    Euroc,
}

impl Format {
    /// Parse a `--format` value.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "tum" => Ok(Format::Tum),
            "kitti" => Ok(Format::Kitti),
            "euroc" => Ok(Format::Euroc),
            other => Err(format!(
                "invalid --format '{other}' (expected one of: tum, kitti, euroc)"
            )),
        }
    }

    /// Lower-case name, for report output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Format::Tum => "tum",
            Format::Kitti => "kitti",
            Format::Euroc => "euroc",
        }
    }
}

/// Parse an `--align` value into a [`cv_eval::Alignment`].
pub fn parse_alignment(value: &str) -> Result<Alignment, String> {
    match value {
        "none" => Ok(Alignment::None),
        "se3" => Ok(Alignment::Se3),
        "sim3" => Ok(Alignment::Sim3),
        other => Err(format!(
            "invalid --align '{other}' (expected one of: none, se3, sim3)"
        )),
    }
}

/// Options for `cv-bench trajectory`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryArgs {
    /// Estimated trajectory file.
    pub estimate: PathBuf,
    /// Ground-truth trajectory file.
    pub ground_truth: PathBuf,
    /// Input format.
    pub format: Format,
    /// Alignment applied before the ATE (default [`Alignment::Se3`]).
    pub align: Alignment,
    /// RPE frame gap (default [`DEFAULT_RPE_DELTA`]).
    pub rpe_delta: usize,
    /// TUM association tolerance in seconds (default [`DEFAULT_MAX_DT`]).
    pub max_dt: f64,
}

/// Options for `cv-bench model`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelArgs {
    /// COLMAP `images.txt` (required).
    pub images: PathBuf,
    /// COLMAP `points3D.txt` (optional).
    pub points3d: Option<PathBuf>,
    /// COLMAP `cameras.txt` (optional).
    pub cameras: Option<PathBuf>,
}

/// Options for `cv-bench retrieval`.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalArgs {
    /// Ranked predictions, one query per line.
    pub predictions: PathBuf,
    /// Relevant ids, one query per line.
    pub ground_truth: PathBuf,
    /// Rank cut-off.
    pub k: usize,
}

/// Parse `cv-bench trajectory` arguments.
pub fn parse_trajectory(argv: &[String]) -> Result<TrajectoryArgs, String> {
    let flags = Flags::parse(
        argv,
        &[
            "estimate",
            "ground-truth",
            "format",
            "align",
            "rpe-delta",
            "max-dt",
        ],
    )?;
    let estimate = flags.path("estimate")?;
    let ground_truth = flags.path("ground-truth")?;
    let format = Format::parse(&flags.required("format")?)?;
    let align = match flags.values.get("align") {
        Some(value) => parse_alignment(value)?,
        None => Alignment::Se3,
    };
    let rpe_delta = flags.usize_or("rpe-delta", DEFAULT_RPE_DELTA)?;
    let max_dt = flags.f64_or("max-dt", DEFAULT_MAX_DT)?;
    Ok(TrajectoryArgs {
        estimate,
        ground_truth,
        format,
        align,
        rpe_delta,
        max_dt,
    })
}

/// Parse `cv-bench model` arguments.
pub fn parse_model(argv: &[String]) -> Result<ModelArgs, String> {
    let flags = Flags::parse(argv, &["images", "points3d", "cameras"])?;
    Ok(ModelArgs {
        images: flags.path("images")?,
        points3d: flags.optional_path("points3d"),
        cameras: flags.optional_path("cameras"),
    })
}

/// Parse `cv-bench retrieval` arguments.
pub fn parse_retrieval(argv: &[String]) -> Result<RetrievalArgs, String> {
    let flags = Flags::parse(argv, &["predictions", "ground-truth", "k"])?;
    let predictions = flags.path("predictions")?;
    let ground_truth = flags.path("ground-truth")?;
    let k = flags.usize_required("k")?;
    if k == 0 {
        return Err("option '--k' must be greater than zero".to_string());
    }
    Ok(RetrievalArgs {
        predictions,
        ground_truth,
        k,
    })
}

/// A validated set of `--name value` options.
struct Flags {
    values: BTreeMap<String, String>,
}

impl Flags {
    /// Tokenise `argv`, rejecting anything that is not an allowed option.
    fn parse(argv: &[String], allowed: &[&str]) -> Result<Self, String> {
        let mut values = BTreeMap::new();
        let mut index = 0;
        while index < argv.len() {
            let token = &argv[index];
            let body = token.strip_prefix("--").ok_or_else(|| {
                format!("unexpected argument '{token}' (options must start with '--')")
            })?;
            if body.is_empty() {
                return Err("unexpected bare '--' argument".to_string());
            }

            let (name, value) = match body.split_once('=') {
                Some((name, value)) => (name.to_string(), value.to_string()),
                None => {
                    index += 1;
                    let value = argv
                        .get(index)
                        .ok_or_else(|| format!("option '--{body}' is missing its value"))?;
                    (body.to_string(), value.clone())
                }
            };

            if name.is_empty() {
                return Err("option name must not be empty".to_string());
            }
            if !allowed.contains(&name.as_str()) {
                let allowed = allowed
                    .iter()
                    .map(|name| format!("--{name}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "unknown option '--{name}' for this subcommand (allowed: {allowed})"
                ));
            }
            if values.contains_key(&name) {
                return Err(format!("option '--{name}' given more than once"));
            }
            values.insert(name, value);
            index += 1;
        }
        Ok(Self { values })
    }

    fn required(&self, name: &str) -> Result<String, String> {
        self.values
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing required option '--{name}'"))
    }

    fn path(&self, name: &str) -> Result<PathBuf, String> {
        Ok(PathBuf::from(self.required(name)?))
    }

    fn optional_path(&self, name: &str) -> Option<PathBuf> {
        self.values.get(name).map(PathBuf::from)
    }

    fn usize_or(&self, name: &str, default: usize) -> Result<usize, String> {
        match self.values.get(name) {
            None => Ok(default),
            Some(value) => value.parse::<usize>().map_err(|_| {
                format!("option '--{name}' expects a non-negative integer, got '{value}'")
            }),
        }
    }

    fn usize_required(&self, name: &str) -> Result<usize, String> {
        let value = self.required(name)?;
        value
            .parse::<usize>()
            .map_err(|_| format!("option '--{name}' expects a non-negative integer, got '{value}'"))
    }

    fn f64_or(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.values.get(name) {
            None => Ok(default),
            Some(value) => {
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("option '--{name}' expects a number, got '{value}'"))?;
                if parsed.is_finite() {
                    Ok(parsed)
                } else {
                    Err(format!(
                        "option '--{name}' expects a finite number, got '{value}'"
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn trajectory_defaults_are_applied() {
        let parsed = parse_trajectory(&args(&[
            "--estimate",
            "e.txt",
            "--ground-truth",
            "g.txt",
            "--format",
            "tum",
        ]))
        .expect("valid trajectory args");

        assert_eq!(parsed.estimate, PathBuf::from("e.txt"));
        assert_eq!(parsed.ground_truth, PathBuf::from("g.txt"));
        assert_eq!(parsed.format, Format::Tum);
        assert_eq!(parsed.align, Alignment::Se3);
        assert_eq!(parsed.rpe_delta, DEFAULT_RPE_DELTA);
        assert_eq!(parsed.max_dt, DEFAULT_MAX_DT);
    }

    #[test]
    fn trajectory_missing_required_flag_is_error() {
        let err = parse_trajectory(&args(&["--ground-truth", "g.txt", "--format", "tum"]))
            .expect_err("missing --estimate must fail");
        assert!(err.contains("--estimate"), "{err}");
    }

    #[test]
    fn trajectory_unknown_flag_is_error() {
        let err = parse_trajectory(&args(&[
            "--estimate",
            "e.txt",
            "--ground-truth",
            "g.txt",
            "--format",
            "tum",
            "--bogus",
            "x",
        ]))
        .expect_err("unknown flag must fail");
        assert!(err.contains("--bogus"), "{err}");
    }

    #[test]
    fn bad_enum_values_are_errors() {
        let err = parse_trajectory(&args(&[
            "--estimate",
            "e",
            "--ground-truth",
            "g",
            "--format",
            "xyz",
        ]))
        .expect_err("bad format must fail");
        assert!(err.contains("--format"), "{err}");

        let err = parse_trajectory(&args(&[
            "--estimate",
            "e",
            "--ground-truth",
            "g",
            "--format",
            "tum",
            "--align",
            "xyz",
        ]))
        .expect_err("bad alignment must fail");
        assert!(err.contains("--align"), "{err}");
    }

    #[test]
    fn equals_syntax_and_duplicates() {
        let parsed = parse_trajectory(&args(&[
            "--estimate=e.txt",
            "--ground-truth=g.txt",
            "--format=tum",
            "--max-dt=0.1",
        ]))
        .expect("valid trajectory args");
        assert_eq!(parsed.max_dt, 0.1);

        let err = parse_trajectory(&args(&[
            "--estimate",
            "a",
            "--estimate",
            "b",
            "--ground-truth",
            "g",
            "--format",
            "tum",
        ]))
        .expect_err("duplicate flag must fail");
        assert!(err.contains("more than once"), "{err}");
    }

    #[test]
    fn non_numeric_numbers_are_rejected() {
        let err = parse_trajectory(&args(&[
            "--estimate",
            "e",
            "--ground-truth",
            "g",
            "--format",
            "tum",
            "--rpe-delta",
            "x",
        ]))
        .expect_err("bad --rpe-delta must fail");
        assert!(err.contains("--rpe-delta"), "{err}");

        let err = parse_trajectory(&args(&[
            "--estimate",
            "e",
            "--ground-truth",
            "g",
            "--format",
            "tum",
            "--max-dt",
            "soon",
        ]))
        .expect_err("bad --max-dt must fail");
        assert!(err.contains("--max-dt"), "{err}");
    }

    #[test]
    fn model_optionals_default_to_none() {
        let parsed = parse_model(&args(&["--images", "images.txt"])).expect("valid model args");
        assert_eq!(parsed.images, PathBuf::from("images.txt"));
        assert_eq!(parsed.points3d, None);
        assert_eq!(parsed.cameras, None);

        assert!(parse_model(&args(&[])).is_err());
    }

    #[test]
    fn retrieval_requires_positive_k() {
        let parsed = parse_retrieval(&args(&[
            "--predictions",
            "p",
            "--ground-truth",
            "g",
            "--k",
            "5",
        ]))
        .expect("valid retrieval args");
        assert_eq!(parsed.k, 5);

        let err = parse_retrieval(&args(&["--predictions", "p", "--ground-truth", "g"]))
            .expect_err("missing --k must fail");
        assert!(err.contains("--k"), "{err}");

        let err = parse_retrieval(&args(&[
            "--predictions",
            "p",
            "--ground-truth",
            "g",
            "--k",
            "0",
        ]))
        .expect_err("zero --k must fail");
        assert!(err.contains("--k"), "{err}");
    }

    #[test]
    fn positional_argument_is_rejected() {
        let err = parse_model(&args(&["images.txt"])).expect_err("positional must fail");
        assert!(err.contains("unexpected argument"), "{err}");
    }
}
