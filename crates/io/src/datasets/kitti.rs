//! KITTI odometry dataset loaders.
//!
//! Source of the formats below: the **KITTI Vision Benchmark Suite — Odometry**
//! (Karlsruhe Institute of Technology),
//! <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>. The ground-truth
//! archive (`data_odometry_poses.zip`) contains one `poses.txt` per sequence and
//! the raw archive contains the matching `times.txt`.
//!
//! ## `read_poses` — `poses.txt`
//!
//! One pose per line, 12 whitespace-separated numbers: the row-major 3x4 matrix
//! `[R | t]` that maps a point in the (left) camera frame to the world frame.
//!
//! ```text
//! r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
//! ```
//!
//! ## `read_times` — `times.txt`
//!
//! One timestamp in seconds per line, whitespace-separated.
//!
//! ```text
//! 0.000000e+00
//! 1.033827e-01
//! ```

use crate::datasets::parse_f64;
use cv_core::{Error, Pose, Result};
use nalgebra::{Matrix3, Vector3};
use std::fs;
use std::path::Path;

/// Read a KITTI `poses.txt` file into [`cv_core::Pose`] values.
///
/// Each non-empty line must contain exactly 12 whitespace-separated numbers,
/// read row-major as a 3x4 `[R | t]` matrix. All values must be finite.
pub fn read_poses<P: AsRef<Path>>(path: P) -> Result<Vec<Pose>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut poses = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 12 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 12 whitespace-separated numbers (row-major 3x4 [R|t]), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let mut m = [0.0_f64; 12];
        for (slot, field) in m.iter_mut().zip(fields.iter()) {
            let v = parse_f64(field, &ctx)?;
            if !v.is_finite() {
                return Err(Error::ParseError(format!(
                    "{}: line {}: matrix entry {field:?} is not finite",
                    path.display(),
                    line_no
                )));
            }
            *slot = v;
        }

        // Row-major 3x4 [R | t]:
        //   m0  m1  m2  m3   (tx)
        //   m4  m5  m6  m7   (ty)
        //   m8  m9  m10 m11  (tz)
        let rotation = Matrix3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
        let translation = Vector3::new(m[3], m[7], m[11]);
        poses.push(Pose::new(rotation, translation));
    }

    Ok(poses)
}

/// Read a KITTI `times.txt` file.
///
/// Every non-empty line must contain exactly one whitespace-separated number.
pub fn read_times<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut times = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 1 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected a single timestamp, found {} fields",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        times.push(parse_f64(fields[0], &ctx)?);
    }

    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::test_util::TempDir;

    #[test]
    fn kitti_read_poses_parses_row_major_r_t() {
        let dir = TempDir::new("kitti_poses");
        let path = dir.write(
            "poses.txt",
            concat!(
                "1 0 0 0 0 1 0 0 0 0 1 0\n",
                "1 0 0 1.5 0 1 0 -2.5 0 0 1 3.5\n",
            ),
        );

        let poses = read_poses(&path).expect("parse poses");
        assert_eq!(poses.len(), 2);

        let identity = poses[0].rotation_matrix();
        assert!((identity - Matrix3::identity()).norm() < 1e-12);
        assert_eq!(poses[0].translation, Vector3::new(0.0, 0.0, 0.0));

        assert_eq!(poses[1].translation, Vector3::new(1.5, -2.5, 3.5));
        assert!((poses[1].rotation_matrix() - Matrix3::identity()).norm() < 1e-12);
    }

    #[test]
    fn kitti_read_poses_rejects_wrong_count() {
        let dir = TempDir::new("kitti_poses_cols");
        let path = dir.write("poses.txt", "1 0 0 0 0 1 0 0 0 0 1\n");
        let err = read_poses(&path).expect_err("11 numbers must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn kitti_read_poses_rejects_non_numeric() {
        let dir = TempDir::new("kitti_poses_nan");
        let path = dir.write("poses.txt", "1 0 0 0 0 1 0 0 0 0 1 x\n");
        assert!(read_poses(&path).is_err());
    }

    #[test]
    fn kitti_read_poses_rejects_non_finite() {
        let dir = TempDir::new("kitti_poses_inf");
        let path = dir.write("poses.txt", "1 0 0 0 0 1 0 0 0 0 1 inf\n");
        assert!(read_poses(&path).is_err());
    }

    #[test]
    fn kitti_read_poses_missing_file_is_err() {
        let dir = TempDir::new("kitti_poses_missing");
        assert!(read_poses(dir.missing("poses.txt")).is_err());
    }

    #[test]
    fn kitti_read_times_parses() {
        let dir = TempDir::new("kitti_times");
        let path = dir.write("times.txt", "0.000000e+00\n1.033827e-01\n2.067654e-01\n");

        let times = read_times(&path).expect("parse times");
        assert_eq!(times.len(), 3);
        assert_eq!(times[0], 0.0);
        assert!((times[1] - 0.1033827).abs() < 1e-12);
    }

    #[test]
    fn kitti_read_times_rejects_extra_tokens() {
        let dir = TempDir::new("kitti_times_bad");
        let path = dir.write("times.txt", "0.0 1.0\n");
        assert!(read_times(&path).is_err());
    }
}
