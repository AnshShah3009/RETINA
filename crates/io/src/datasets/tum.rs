//! TUM RGB-D benchmark dataset loaders.
//!
//! Source of the formats below: the **TUM RGB-D Benchmark**
//! (Technical University of Munich), <https://vision.in.tum.de/data/datasets/rgbd-dataset>.
//! The timestamp association reproduces the reference `associate.py` shipped
//! with the benchmark
//! (<https://vision.in.tum.de/data/datasets/rgbd-dataset/tools>).
//!
//! ## `read_index` — `rgb.txt` / `depth.txt` / `accel.txt`
//!
//! Whitespace-separated, one entry per line; lines beginning with `#` are
//! comments. The two fields are a timestamp in seconds and a file name relative
//! to the sequence folder.
//!
//! ```text
//! # color images
//! # file: rgb.txt
//! # ...
//! 1305031102.175304 rgb/1305031102.175304.png
//! ```
//!
//! ## `read_groundtruth` — `groundtruth.txt`
//!
//! Whitespace-separated, one pose per line, 8 fields. The quaternion is in
//! `(qx, qy, qz, qw)` order (i.e. **vector-first**), unlike EuRoC.
//!
//! ```text
//! # groundtruth trajectory
//! # file: groundtruth.txt
//! # ...
//! 1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0 1.0
//! ```
//!
//! ## `associate`
//!
//! [`associate`] implements the same greedy nearest-timestamp rule as
//! `associate.py`: every candidate pair at most `max_dt` apart is sorted by its
//! absolute timestamp difference (ties broken by the indices for determinism),
//! then pairs are consumed greedily so each entry in either list is used at most
//! once. The returned `(index_in_a, index_in_b)` list is sorted by the timestamp
//! of the first list, matching `associate.py`'s final `matches.sort()`.

use crate::datasets::{parse_f64, unit_quaternion_from_wxyz};
use cv_core::{Error, Pose, Result};
use nalgebra::Vector3;
use std::fs;
use std::path::Path;

/// One entry of an index file (`rgb.txt`, `depth.txt`, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexEntry {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// File name relative to the sequence folder.
    pub filename: String,
}

/// One ground-truth pose from `groundtruth.txt`.
#[derive(Debug, Clone, Copy)]
pub struct TumPose {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Pose (translation + unit-quaternion orientation) of the camera in the
    /// world frame.
    pub pose: Pose,
}

/// Read a TUM index file (`rgb.txt`, `depth.txt`, ...).
///
/// `#`-comment lines and blank lines are skipped. Every remaining line must
/// contain exactly two whitespace-separated fields, `timestamp filename`.
pub fn read_index<P: AsRef<Path>>(path: P) -> Result<Vec<IndexEntry>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut entries = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 2 whitespace-separated fields (timestamp filename), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp = parse_f64(fields[0], &ctx)?;
        if fields[1].is_empty() {
            return Err(Error::InvalidInput(format!(
                "{}: line {}: empty file name",
                path.display(),
                line_no
            )));
        }

        entries.push(IndexEntry {
            timestamp,
            filename: fields[1].to_owned(),
        });
    }

    Ok(entries)
}

/// Read a TUM `groundtruth.txt` file.
///
/// Every data line must contain exactly 8 whitespace-separated fields in the
/// order `timestamp tx ty tz qx qy qz qw`. The quaternion is normalised to unit
/// length.
pub fn read_groundtruth<P: AsRef<Path>>(path: P) -> Result<Vec<TumPose>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut poses = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 8 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 8 whitespace-separated fields \
                 (timestamp tx ty tz qx qy qz qw), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp = parse_f64(fields[0], &ctx)?;
        let tx = parse_f64(fields[1], &ctx)?;
        let ty = parse_f64(fields[2], &ctx)?;
        let tz = parse_f64(fields[3], &ctx)?;
        // TUM stores the quaternion vector-first: (qx, qy, qz, qw).
        let qx = parse_f64(fields[4], &ctx)?;
        let qy = parse_f64(fields[5], &ctx)?;
        let qz = parse_f64(fields[6], &ctx)?;
        let qw = parse_f64(fields[7], &ctx)?;

        let rotation = unit_quaternion_from_wxyz(qw, qx, qy, qz, &ctx)?;
        let pose = Pose::from_quat_translation(rotation, Vector3::new(tx, ty, tz));

        poses.push(TumPose { timestamp, pose });
    }

    Ok(poses)
}

/// Associate two timestamped index lists by nearest timestamp.
///
/// Returns `(index_in_a, index_in_b)` pairs. Candidate pairs whose timestamps
/// differ by more than `max_dt` (in the same units as [`IndexEntry::timestamp`],
/// normally seconds) are discarded; among the rest, the pair with the smallest
/// difference is taken first and both of its entries are removed, then the
/// process repeats. The result contains each `a` index and each `b` index at
/// most once and is sorted by the timestamp of `a`.
///
/// A non-finite or negative `max_dt` yields an empty association.
pub fn associate(a: &[IndexEntry], b: &[IndexEntry], max_dt: f64) -> Vec<(usize, usize)> {
    if !max_dt.is_finite() || max_dt < 0.0 {
        return Vec::new();
    }

    // All candidate pairs within the tolerance, tagged with their difference.
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, ea) in a.iter().enumerate() {
        for (j, eb) in b.iter().enumerate() {
            let diff = (ea.timestamp - eb.timestamp).abs();
            if diff.is_finite() && diff <= max_dt {
                candidates.push((diff, i, j));
            }
        }
    }

    // Greedy: smallest difference first; indices break ties deterministically.
    candidates.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));

    let mut used_a = vec![false; a.len()];
    let mut used_b = vec![false; b.len()];
    let mut matches: Vec<(usize, usize)> = Vec::new();

    for (_, i, j) in candidates {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            matches.push((i, j));
        }
    }

    // Match `associate.py`'s final ordering: by the first list's timestamp.
    matches.sort_by(|x, y| {
        a[x.0]
            .timestamp
            .total_cmp(&a[y.0].timestamp)
            .then(x.1.cmp(&y.1))
    });

    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::test_util::TempDir;

    fn entry(timestamp: f64, filename: &str) -> IndexEntry {
        IndexEntry {
            timestamp,
            filename: filename.to_string(),
        }
    }

    #[test]
    fn tum_read_index_parses_and_skips_comments() {
        let dir = TempDir::new("tum_index");
        let path = dir.write(
            "rgb.txt",
            concat!(
                "# color images\n",
                "# file: rgb.txt\n",
                "\n",
                "1305031102.175304 rgb/1305031102.175304.png\n",
                "1305031102.275326 rgb/1305031102.275326.png\n",
            ),
        );

        let entries = read_index(&path).expect("parse index");
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            entry(1305031102.175304, "rgb/1305031102.175304.png")
        );
        assert_eq!(entries[1].filename, "rgb/1305031102.275326.png");
    }

    #[test]
    fn tum_read_index_rejects_non_numeric() {
        let dir = TempDir::new("tum_index_bad");
        let path = dir.write("rgb.txt", "not-a-time rgb/x.png\n");
        assert!(read_index(&path).is_err());
    }

    #[test]
    fn tum_read_index_rejects_extra_columns() {
        let dir = TempDir::new("tum_index_cols");
        let path = dir.write("rgb.txt", "1305031102.175304 rgb/x.png extra\n");
        assert!(read_index(&path).is_err());
    }

    #[test]
    fn tum_read_groundtruth_parses_vector_first_quaternion() {
        let dir = TempDir::new("tum_gt");
        let path = dir.write(
            "groundtruth.txt",
            concat!(
                "# groundtruth trajectory\n",
                "1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n",
            ),
        );

        let poses = read_groundtruth(&path).expect("parse groundtruth");
        assert_eq!(poses.len(), 1);
        let p = &poses[0];
        assert_eq!(p.timestamp, 1305031102.175304);
        assert_eq!(p.pose.translation, Vector3::new(1.0, 2.0, 3.0));
        // (qx,qy,qz,qw) = (0,0,0,1) => identity rotation.
        assert!((p.pose.rotation.w - 1.0).abs() < 1e-15);
        assert!(p.pose.rotation.i.abs() < 1e-15);
    }

    #[test]
    fn tum_read_groundtruth_rejects_wrong_column_count() {
        let dir = TempDir::new("tum_gt_cols");
        let path = dir.write(
            "groundtruth.txt",
            "1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0\n",
        );
        let err = read_groundtruth(&path).expect_err("7 fields must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn tum_read_groundtruth_missing_file_is_err() {
        let dir = TempDir::new("tum_gt_missing");
        assert!(read_groundtruth(dir.missing("groundtruth.txt")).is_err());
    }

    #[test]
    fn tum_associate_pairs_nearest_timestamps() {
        let a = [entry(1.0, "a0"), entry(2.0, "a1"), entry(3.0, "a2")];
        let b = [entry(1.02, "b0"), entry(2.01, "b1"), entry(3.5, "b2")];

        let matches = associate(&a, &b, 0.05);
        assert_eq!(matches, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn tum_associate_enforces_one_to_one() {
        // a[0] and a[1] both lie near b[0]; only the closest may be consumed.
        let a = [entry(0.0, "a0"), entry(0.03, "a1")];
        let b = [entry(0.01, "b0")];

        let matches = associate(&a, &b, 0.05);
        assert_eq!(matches, vec![(0, 0)]);
    }

    #[test]
    fn tum_associate_discards_pairs_beyond_max_dt() {
        let a = [entry(0.0, "a0"), entry(10.0, "a1")];
        let b = [entry(0.01, "b0"), entry(10.5, "b1")];

        let matches = associate(&a, &b, 0.1);
        assert_eq!(matches, vec![(0, 0)]);
    }

    #[test]
    fn tum_associate_sorts_by_first_timestamp() {
        let a = [entry(5.0, "a0"), entry(1.0, "a1")];
        let b = [entry(5.01, "b0"), entry(1.02, "b1")];

        let matches = associate(&a, &b, 0.05);
        // Sorted by a's timestamp, not by input order.
        assert_eq!(matches, vec![(1, 1), (0, 0)]);
    }

    #[test]
    fn tum_associate_empty_and_invalid_tolerance() {
        let a = [entry(0.0, "a0")];
        let b = [entry(0.0, "b0")];
        assert!(associate(&a, &b, -1.0).is_empty());
        assert!(associate(&a, &b, f64::NAN).is_empty());
        assert!(associate(&[], &b, 1.0).is_empty());
    }
}
