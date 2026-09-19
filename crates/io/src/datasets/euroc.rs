//! EuRoC MAV / ETH ASL dataset loaders.
//!
//! Source of the formats below: the **EuRoC MAV Dataset** (ETH Zurich Autonomous
//! Systems Lab), <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>.
//! Each downloaded sequence contains a `mav0/` folder with sensor streams. This
//! module reads the three text artefacts applications depend on.
//!
//! ## `read_csv` — generic `<sensor>/data.csv`
//!
//! A comma-separated table whose first line is a `#`-prefixed comment header.
//! Column 0 is a nanosecond timestamp; the remaining columns are the sensor
//! payload (gyro/accel for `imu0`, etc.).
//!
//! ```text
//! #timestamp [ns],w_RS_S_x [rad s^-1],...,a_RS_S_z [m s^-2]
//! 1403636579763555584,-0.0991,...,9.7913
//! ```
//!
//! ## `read_groundtruth` — `state_groundtruth_estimate0/data.csv`
//!
//! One sample per row, 17 comma-separated fields, with a `#`-prefixed header:
//!
//! ```text
//! t[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z,
//! b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z
//! ```
//!
//! `p` is the position and `q` the (scalar-first) orientation of the IMU frame
//! in the world; `v` the velocity; `b_w` / `b_a` the gyroscope and accelerometer
//! biases.
//!
//! ## `read_image_index` — `cam0/data.csv` (also `cam1/data.csv`)
//!
//! A `#`-prefixed header followed by `timestamp[ns],filename` rows, where
//! `filename` is relative to the image folder (`cam0/data/<filename>`).
//!
//! ```text
//! #timestamp [ns],filename
//! 1403636579763555584,1403636579763555584.png
//! ```

use crate::datasets::{parse_f64, parse_i64, unit_quaternion_from_wxyz};
use cv_core::{Error, Pose, Result};
use nalgebra::Vector3;
use std::fs;
use std::path::Path;

/// One ground-truth state sample from
/// `state_groundtruth_estimate0/data.csv`.
#[derive(Debug, Clone, Copy)]
pub struct GroundTruthState {
    /// Sample timestamp in nanoseconds.
    pub timestamp_ns: i64,
    /// Position (translation) and unit-quaternion orientation of the body in
    /// the world frame.
    pub pose: Pose,
    /// Linear velocity of the body in the world frame [m/s].
    pub velocity: Vector3<f64>,
    /// Gyroscope bias [rad/s].
    pub gyro_bias: Vector3<f64>,
    /// Accelerometer bias [m/s^2].
    pub accel_bias: Vector3<f64>,
}

/// One entry of an image stream index (`cam0/data.csv`, `cam1/data.csv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageEntry {
    /// Image timestamp in nanoseconds.
    pub timestamp_ns: i64,
    /// Image file name relative to the stream's `data/` folder.
    pub filename: String,
}

/// Read a EuRoC comma-separated `data.csv` file as a numeric table.
///
/// Lines starting with `#` and blank lines are skipped. Every data row must
/// contain the same number of comma-separated columns as the first data row, and
/// every field must parse as `f64`. The nanosecond timestamp of column 0 is
/// returned as an `f64` here (use [`read_groundtruth`] / [`read_image_index`] to
/// keep the exact integer timestamp); because 2^53 < 1e18 the raw `f64` cannot
/// represent every nanosecond exactly.
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f64>>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut expected_width: Option<usize> = None;

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        match expected_width {
            Some(width) if fields.len() != width => {
                return Err(Error::ParseError(format!(
                    "{}: line {}: expected {} comma-separated columns, found {}",
                    path.display(),
                    line_no,
                    width,
                    fields.len()
                )));
            }
            None => expected_width = Some(fields.len()),
            _ => {}
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let mut row = Vec::with_capacity(fields.len());
        for field in &fields {
            row.push(parse_f64(field, &ctx)?);
        }
        rows.push(row);
    }

    Ok(rows)
}

/// Read `state_groundtruth_estimate0/data.csv` into [`GroundTruthState`] values.
///
/// Each non-comment data row must have exactly 17 comma-separated fields in the
/// order `t[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, b_w_x, b_w_y,
/// b_w_z, b_a_x, b_a_y, b_a_z`. The quaternion is normalised to unit length.
pub fn read_groundtruth<P: AsRef<Path>>(path: P) -> Result<Vec<GroundTruthState>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut states = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 17 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 17 comma-separated fields \
                 (t[ns], p_x..p_z, q_w..q_z, v_x..v_z, b_w_x..b_w_z, b_a_x..b_a_z), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp_ns = parse_i64(fields[0], &ctx)?;

        let px = parse_f64(fields[1], &ctx)?;
        let py = parse_f64(fields[2], &ctx)?;
        let pz = parse_f64(fields[3], &ctx)?;

        let qw = parse_f64(fields[4], &ctx)?;
        let qx = parse_f64(fields[5], &ctx)?;
        let qy = parse_f64(fields[6], &ctx)?;
        let qz = parse_f64(fields[7], &ctx)?;

        let vx = parse_f64(fields[8], &ctx)?;
        let vy = parse_f64(fields[9], &ctx)?;
        let vz = parse_f64(fields[10], &ctx)?;

        let bw = Vector3::new(
            parse_f64(fields[11], &ctx)?,
            parse_f64(fields[12], &ctx)?,
            parse_f64(fields[13], &ctx)?,
        );
        let ba = Vector3::new(
            parse_f64(fields[14], &ctx)?,
            parse_f64(fields[15], &ctx)?,
            parse_f64(fields[16], &ctx)?,
        );

        let rotation = unit_quaternion_from_wxyz(qw, qx, qy, qz, &ctx)?;
        let pose = Pose::from_quat_translation(rotation, Vector3::new(px, py, pz));

        states.push(GroundTruthState {
            timestamp_ns,
            pose,
            velocity: Vector3::new(vx, vy, vz),
            gyro_bias: bw,
            accel_bias: ba,
        });
    }

    Ok(states)
}

/// Read a EuRoC camera image index (`cam0/data.csv`).
///
/// Lines starting with `#` and blank lines are skipped; every remaining row must
/// be exactly `timestamp[ns],filename`.
pub fn read_image_index<P: AsRef<Path>>(path: P) -> Result<Vec<ImageEntry>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut entries = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 2 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 2 comma-separated fields (timestamp[ns], filename), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp_ns = parse_i64(fields[0], &ctx)?;
        let filename = fields[1].trim().to_owned();
        if filename.is_empty() {
            return Err(Error::InvalidInput(format!(
                "{}: line {}: empty image filename",
                path.display(),
                line_no
            )));
        }

        entries.push(ImageEntry {
            timestamp_ns,
            filename,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::test_util::TempDir;

    const GROUNDTRUTH: &str = concat!(
        "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],",
        "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],",
        "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],",
        "b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],",
        "b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n",
        // quaternion given as (2,0,0,0): must be normalised to identity.
        "1403636579763555584,1.0,2.0,3.0,2.0,0.0,0.0,0.0,",
        "0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06\n",
    );

    #[test]
    fn euroc_read_csv_parses_table() {
        let dir = TempDir::new("euroc_csv");
        let path = dir.write(
            "data.csv",
            concat!(
                "#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z\n",
                "1403636579763555584,-0.0991,0.1542,-1.1201\n",
                "1403636579813555456,-0.0992,0.1543,-1.1200\n",
            ),
        );

        let rows = read_csv(&path).expect("parse csv");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 4);
        assert_eq!(rows[0][0], 1403636579763555584.0);
        assert_eq!(rows[0][1], -0.0991);
        assert_eq!(rows[0][2], 0.1542);
        assert_eq!(rows[0][3], -1.1201);
        assert_eq!(rows[1][1], -0.0992);
    }

    #[test]
    fn euroc_read_csv_rejects_wrong_column_count() {
        let dir = TempDir::new("euroc_csv_cols");
        let path = dir.write(
            "data.csv",
            "1403636579763555584,-0.0991,0.1542,-1.1201\n1403636579813555456,-0.0992,0.1543\n",
        );
        let err = read_csv(&path).expect_err("ragged row must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn euroc_read_csv_rejects_non_numeric() {
        let dir = TempDir::new("euroc_csv_nan");
        let path = dir.write("data.csv", "1403636579763555584,not-a-number\n");
        assert!(read_csv(&path).is_err());
    }

    #[test]
    fn euroc_read_csv_missing_file_is_err() {
        let dir = TempDir::new("euroc_csv_missing");
        let err = read_csv(dir.missing("nope.csv")).expect_err("missing file must fail");
        assert!(matches!(err, Error::IoError(_)));
    }

    #[test]
    fn euroc_read_groundtruth_parses_full_state() {
        let dir = TempDir::new("euroc_gt");
        let path = dir.write("data.csv", GROUNDTRUTH);

        let states = read_groundtruth(&path).expect("parse groundtruth");
        assert_eq!(states.len(), 1);
        let s = &states[0];
        assert_eq!(s.timestamp_ns, 1_403_636_579_763_555_584);
        assert_eq!(s.pose.translation, Vector3::new(1.0, 2.0, 3.0));
        // (2,0,0,0) normalised => identity rotation.
        assert!((s.pose.rotation.w - 1.0).abs() < 1e-15);
        assert!(s.pose.rotation.i.abs() < 1e-15);
        assert!(s.pose.rotation.j.abs() < 1e-15);
        assert!(s.pose.rotation.k.abs() < 1e-15);
        assert_eq!(s.velocity, Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(s.gyro_bias, Vector3::new(0.01, 0.02, 0.03));
        assert_eq!(s.accel_bias, Vector3::new(0.04, 0.05, 0.06));
    }

    #[test]
    fn euroc_read_groundtruth_normalises_quaternion() {
        let dir = TempDir::new("euroc_gt_quat");
        // 90 deg about z as a non-unit quaternion (scale 2).
        let s = 0.7071067811865476_f64 * 2.0;
        let content = format!("0,0,0,0,{s},0,0,{s},0,0,0,0,0,0,0,0,0\n");
        let path = dir.write("data.csv", &content);

        let states = read_groundtruth(&path).expect("parse groundtruth");
        let rot = states[0].pose.rotation;
        assert!((rot.w - 0.7071067811865476).abs() < 1e-12);
        assert!((rot.k - 0.7071067811865476).abs() < 1e-12);
        // Applying the rotation to +x must land on +y.
        let rotated = rot * Vector3::new(1.0, 0.0, 0.0);
        assert!((rotated.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn euroc_read_groundtruth_rejects_degenerate_quaternion() {
        let dir = TempDir::new("euroc_gt_degenerate");
        // 17 zero fields: the quaternion (w,x,y,z) = (0,0,0,0) is degenerate.
        let path = dir.write("data.csv", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n");
        let err = read_groundtruth(&path).expect_err("zero quaternion must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn euroc_read_groundtruth_rejects_truncated_line() {
        let dir = TempDir::new("euroc_gt_trunc");
        let path = dir.write("data.csv", "1403636579763555584,1.0,2.0,3.0,1.0,0.0,0.0\n");
        let err = read_groundtruth(&path).expect_err("truncated line must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn euroc_read_image_index_parses_entries() {
        let dir = TempDir::new("euroc_cam");
        let path = dir.write(
            "data.csv",
            concat!(
                "#timestamp [ns],filename\n",
                "1403636579763555584,1403636579763555584.png\n",
                "1403636580811214592,1403636580811214592.png\n",
            ),
        );
        let entries = read_image_index(&path).expect("parse index");
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            ImageEntry {
                timestamp_ns: 1403636579763555584,
                filename: "1403636579763555584.png".to_string(),
            }
        );
        assert_eq!(entries[1].filename, "1403636580811214592.png");
    }

    #[test]
    fn euroc_read_image_index_rejects_bad_columns() {
        let dir = TempDir::new("euroc_cam_bad");
        let path = dir.write("data.csv", "1403636579763555584\n");
        assert!(read_image_index(&path).is_err());
    }

    #[test]
    fn euroc_read_image_index_missing_file_is_err() {
        let dir = TempDir::new("euroc_cam_missing");
        assert!(read_image_index(dir.missing("data.csv")).is_err());
    }
}
