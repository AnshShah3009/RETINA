//! Standard dataset loaders for visual odometry / SLAM benchmarking.
//!
//! This module parses the on-disk text formats of the datasets that visual
//! localization, structure-from-motion and SLAM systems are most commonly
//! evaluated on. Each submodule documents the exact layout it consumes and
//! cites the upstream source of that layout:
//!
//! * [`euroc`] — EuRoC MAV / ETH ASL `data.csv`, ground-truth state and camera
//!   image index.
//! * [`tum`] — TUM RGB-D benchmark `rgb.txt` / `depth.txt` / `groundtruth.txt`
//!   plus the standard greedy timestamp association.
//! * [`kitti`] — KITTI odometry `poses.txt` (row-major 3x4 `[R|t]`) and
//!   `times.txt`.
//! * [`colmap`] — COLMAP text model (`cameras.txt`, `images.txt`,
//!   `points3D.txt`).
//!
//! Every loader validates the structure and numerics of its input and reports a
//! descriptive [`cv_core::Error`] on malformed data. None of them panic on bad
//! input: truncated lines, non-numeric fields, wrong column/field counts and
//! degenerate quaternions all produce an error, and an unreadable path surfaces
//! as [`cv_core::Error::IoError`].
//!
//! Only the plain-text variants are supported (COLMAP's binary model and the
//! dataset image/point-cloud binaries themselves are out of scope).

pub mod colmap;
pub mod euroc;
pub mod kitti;
pub mod tum;

use crate::Result;

/// Parse a single (already whitespace-split) token as `f64`.
///
/// `ctx` is a pre-formatted location string such as `"file.txt: line 3"` used
/// only to build a descriptive error message; the token itself is trimmed
/// before parsing.
pub(crate) fn parse_f64(field: &str, ctx: &str) -> Result<f64> {
    field.trim().parse::<f64>().map_err(|e| {
        cv_core::Error::ParseError(format!("{ctx}: cannot parse {field:?} as f64: {e}"))
    })
}

/// Parse a single (already whitespace-split) token as `i64`.
pub(crate) fn parse_i64(field: &str, ctx: &str) -> Result<i64> {
    field.trim().parse::<i64>().map_err(|e| {
        cv_core::Error::ParseError(format!("{ctx}: cannot parse {field:?} as i64: {e}"))
    })
}

/// Build a *normalised* unit quaternion from raw `(w, x, y, z)` components.
///
/// Dataset files frequently store quaternions with a norm that differs from one
/// by a few ULPs, so the components are rescaled to unit length instead of being
/// trusted as-is. Degenerate inputs (zero-length or non-finite) are rejected
/// with an error rather than silently producing NaNs.
pub(crate) fn unit_quaternion_from_wxyz(
    w: f64,
    x: f64,
    y: f64,
    z: f64,
    ctx: &str,
) -> Result<nalgebra::UnitQuaternion<f64>> {
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err(cv_core::Error::ParseError(format!(
            "{ctx}: degenerate quaternion ({w}, {x}, {y}, {z}) (norm = {norm})"
        )));
    }
    Ok(nalgebra::UnitQuaternion::from_quaternion(
        nalgebra::Quaternion::new(w / norm, x / norm, y / norm, z / norm),
    ))
}

/// Test-only helpers shared by the per-format unit tests.
///
/// Creates a unique directory under [`std::env::temp_dir`] that is removed when
/// the guard is dropped, so the tests are hermetic and leave nothing behind.
/// Only `std` is used (the crate intentionally has no `tempfile` dependency).
#[cfg(test)]
pub(crate) mod test_util {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique temporary directory that deletes itself on drop.
    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        /// Create a fresh unique temporary directory tagged with `tag`.
        pub fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut path = std::env::temp_dir();
            path.push(format!(
                "cv_io_datasets_{tag}_{}_{seq}_{nanos}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        /// Directory path.
        pub fn path(&self) -> &Path {
            &self.path
        }

        /// Write `contents` to `name` inside the directory and return its path.
        pub fn write(&self, name: &str, contents: &str) -> PathBuf {
            let p = self.path.join(name);
            std::fs::write(&p, contents).expect("write temp file");
            p
        }

        /// Path for a file that does **not** exist (for missing-file tests).
        pub fn missing(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
