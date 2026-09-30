//! Robustness tests for the dataset readers in `crates/io/src/datasets/`
//! (euroc, tum, kitti, colmap).
//!
//! These are plain-text parsers, so the interesting hostile inputs are
//! non-UTF-8 bytes, ragged rows, non-numeric and non-finite fields, and the
//! two-line-per-image structure COLMAP relies on.

mod common;

use common::*;
use cv_io::datasets::{colmap, euroc, kitti, tum};
use nalgebra::Vector3;

const EUROC_GT: &str = "1403636579763555584,1.0,2.0,3.0,1.0,0.0,0.0,0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06\n";
const TUM_GT: &str = "1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n";
const KITTI_POSE: &str = "1 0 0 1.5 0 1 0 -2.5 0 0 1 3.5\n";
const COLMAP_CAM: &str = "1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n";
const COLMAP_IMG: &str = "1 1 0 0 0 0 0 0 1 image1.jpg\n10.0 20.0 1 30.0 40.0 2 -1.0 -1.0 -1\n";
const COLMAP_PTS: &str = "1 0.5 -1.5 2.5 255 128 0 0.5 1 0 2 3\n";

// ===========================================================================
// Baseline: every reader accepts its well-formed sample
// ===========================================================================

#[test]
fn datasets_baseline_all_readers_parse_their_samples() {
    let dir = TempDir::new("ds_baseline");
    assert_eq!(euroc::read_csv(dir.write("c.csv", "1,2,3\n4,5,6\n")).expect("csv").len(), 2);
    assert_eq!(
        euroc::read_groundtruth(dir.write("gt.csv", EUROC_GT)).expect("euroc gt").len(),
        1
    );
    assert_eq!(
        euroc::read_image_index(dir.write("cam.csv", "1,a.png\n2,b.png\n")).expect("index").len(),
        2
    );
    assert_eq!(tum::read_index(dir.write("rgb.txt", "1.0 a.png\n")).expect("tum index").len(), 1);
    assert_eq!(
        tum::read_groundtruth(dir.write("gt.txt", TUM_GT)).expect("tum gt").len(),
        1
    );
    assert_eq!(kitti::read_poses(dir.write("poses.txt", KITTI_POSE)).expect("kitti").len(), 1);
    assert_eq!(kitti::read_times(dir.write("times.txt", "0.0\n1.0\n")).expect("times").len(), 2);
    assert_eq!(
        colmap::read_cameras_text(dir.write("cameras.txt", COLMAP_CAM)).expect("cameras").len(),
        1
    );
    assert_eq!(
        colmap::read_images_text(dir.write("images.txt", COLMAP_IMG)).expect("images").len(),
        1
    );
    assert_eq!(
        colmap::read_points3d_text(dir.write("points3D.txt", COLMAP_PTS)).expect("points").len(),
        1
    );
}

// ===========================================================================
// 1. Non-UTF-8 bytes: all four readers use fs::read_to_string
// ===========================================================================

/// Every dataset reader must reject non-UTF-8 rather than replacing the bytes.
#[test]
fn datasets_non_utf8_is_an_error_for_every_reader() {
    let dir = TempDir::new("ds_utf8");
    let bad = bytes(&[b"1,2,3\n".as_slice(), &[0xff, 0xfe], b"\n".as_slice()]);

    assert!(matches!(
        euroc::read_csv(dir.write_bytes("c.csv", &bad)),
        Err(cv_core::Error::IoError(_))
    ));
    assert!(euroc::read_groundtruth(dir.write_bytes("gt.csv", &bad)).is_err());
    assert!(euroc::read_image_index(dir.write_bytes("i.csv", &bad)).is_err());
    assert!(tum::read_index(dir.write_bytes("rgb.txt", &bad)).is_err());
    assert!(tum::read_groundtruth(dir.write_bytes("gt.txt", &bad)).is_err());
    assert!(kitti::read_poses(dir.write_bytes("poses.txt", &bad)).is_err());
    assert!(kitti::read_times(dir.write_bytes("times.txt", &bad)).is_err());
    assert!(colmap::read_cameras_text(dir.write_bytes("cameras.txt", &bad)).is_err());
    assert!(colmap::read_images_text(dir.write_bytes("images.txt", &bad)).is_err());
    assert!(colmap::read_points3d_text(dir.write_bytes("points3D.txt", &bad)).is_err());
}

/// A directory instead of a file must be an error, not a panic.
#[test]
fn datasets_directory_instead_of_file_is_an_error() {
    let dir = TempDir::new("ds_dir");
    assert!(euroc::read_csv(dir.path()).is_err());
    assert!(tum::read_groundtruth(dir.path()).is_err());
    assert!(kitti::read_poses(dir.path()).is_err());
    assert!(colmap::read_images_text(dir.path()).is_err());
}

/// An empty file yields an empty collection rather than an error.
#[test]
fn datasets_empty_file_is_an_empty_collection() {
    let dir = TempDir::new("ds_empty");
    assert!(euroc::read_csv(dir.write("c.csv", "")).expect("empty csv").is_empty());
    assert!(euroc::read_groundtruth(dir.write("gt.csv", "")).expect("empty gt").is_empty());
    assert!(tum::read_index(dir.write("rgb.txt", "")).expect("empty").is_empty());
    assert!(kitti::read_times(dir.write("times.txt", "")).expect("empty").is_empty());
    assert!(colmap::read_points3d_text(dir.write("p.txt", "")).expect("empty").is_empty());
}

/// A UTF-8 BOM (EF BB BF) is not whitespace, so `trim()` does not remove it and
/// the first token fails to parse. Documented behaviour, not a silent misparse.
#[test]
fn datasets_utf8_bom_is_rejected_with_a_parse_error() {
    let dir = TempDir::new("ds_bom");
    let with_bom = bytes(&[&[0xef, 0xbb, 0xbf], b"1,2,3\n".as_slice()]);
    let err = euroc::read_csv(dir.write_bytes("c.csv", &with_bom)).expect_err("BOM");
    assert!(
        matches!(err, cv_core::Error::ParseError(_)),
        "expected a descriptive ParseError, got {err}"
    );
}

/// CRLF line endings must parse.
#[test]
fn datasets_crlf_line_endings_parse() {
    let dir = TempDir::new("ds_crlf");
    let crlf = KITTI_POSE.replace('\n', "\r\n");
    let poses = kitti::read_poses(dir.write("poses.txt", &crlf)).expect("CRLF poses");
    assert_eq!(poses.len(), 1);
    assert_eq!(poses[0].translation, Vector3::new(1.5, -2.5, 3.5));
}

// ===========================================================================
// 2. Ragged / truncated rows
// ===========================================================================

#[test]
fn datasets_ragged_rows_error() {
    let dir = TempDir::new("ds_ragged");
    // EuRoC csv: 4 columns then 3.
    assert!(euroc::read_csv(dir.write("c.csv", "1,2,3,4\n5,6,7\n")).is_err());
    // EuRoC ground truth: 17 fields then 16.
    assert!(euroc::read_groundtruth(dir.write("gt.csv", "1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0\n2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0\n")).is_err());
    // TUM ground truth: 8 fields then 7.
    assert!(tum::read_groundtruth(dir.write("gt.txt", "1.0 0 0 0 0 0 0 1\n2.0 0 0 0 0 0 0\n")).is_err());
    // KITTI poses: 12 numbers then 11.
    assert!(kitti::read_poses(dir.write("poses.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1\n")).is_err());
    // COLMAP points3D: 8 fields then 7.
    assert!(colmap::read_points3d_text(dir.write("p.txt", "1 0 0 0 1 1 1 0.5\n2 0 0 0 1 1 1\n")).is_err());
}

/// A row that is cut off by EOF without a trailing newline still parses when it
/// is complete.
#[test]
fn datasets_final_row_without_newline_parses() {
    let dir = TempDir::new("ds_no_trailing_nl");
    let poses = kitti::read_poses(dir.write("poses.txt", KITTI_POSE.trim_end()))
        .expect("no trailing newline");
    assert_eq!(poses.len(), 1);
}

/// A *partial* final row must be an error, not a silently dropped line.
#[test]
fn datasets_partial_final_row_is_reported() {
    let dir = TempDir::new("ds_partial");
    let e = tum::read_groundtruth(dir.write("gt.txt", "1.0 0 0 0 0 0 0 1\n2.0 0 0 0 0 0")).expect_err("partial row");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");

    let e = kitti::read_poses(dir.write("poses.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1")).expect_err("partial row");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

// ===========================================================================
// 3. Hostile numbers: overflow, non-finite
// ===========================================================================

/// A nanosecond timestamp beyond `i64::MAX` must be a parse error, not a wrap.
#[test]
fn datasets_timestamp_beyond_i64_max_errors() {
    let dir = TempDir::new("ds_i64");
    let e = euroc::read_groundtruth(dir.write(
        "gt.csv",
        "99999999999999999999,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0\n",
    ))
    .expect_err("i64 overflow");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");

    let e = euroc::read_image_index(dir.write("i.csv", "99999999999999999999,a.png\n"))
        .expect_err("i64 overflow");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// A camera id beyond `u32::MAX` must be a parse error.
#[test]
fn datasets_camera_id_beyond_u32_max_errors() {
    let dir = TempDir::new("ds_u32");
    let e = colmap::read_cameras_text(dir.write("cameras.txt", "4294967296 PINHOLE 640 480 1 1 1 1\n"))
        .expect_err("u32 overflow");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");

    let e = colmap::read_images_text(dir.write("images.txt", "1 1 0 0 0 0 0 0 4294967296 a.jpg\n\n"))
        .expect_err("u32 overflow");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");

    // POINT3D_ID is a u64: 2^64 must be rejected.
    let e = colmap::read_points3d_text(dir.write("p.txt", "18446744073709551616 0 0 0 1 1 1 0\n"))
        .expect_err("u64 overflow");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// A zero-size camera image is rejected (`width == 0 || height == 0`).
#[test]
fn datasets_zero_camera_dimensions_error() {
    let dir = TempDir::new("ds_zero_dim");
    assert!(colmap::read_cameras_text(dir.write("c.txt", "1 PINHOLE 0 480 1 1 1 1\n")).is_err());
    assert!(colmap::read_cameras_text(dir.write("c2.txt", "1 PINHOLE 640 0 1 1 1 1\n")).is_err());
}

/// A degenerate (all-zero) quaternion is rejected by every reader that has one.
#[test]
fn datasets_degenerate_quaternion_errors() {
    let dir = TempDir::new("ds_quat0");
    assert!(euroc::read_groundtruth(dir.write("g.csv", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")).is_err());
    assert!(tum::read_groundtruth(dir.write("g.txt", "0 0 0 0 0 0 0 0\n")).is_err());
    assert!(colmap::read_images_text(dir.write("i.txt", "1 0 0 0 0 0 0 0 1 a.jpg\n\n")).is_err());
}

/// A *non-finite* quaternion is rejected (the norm check covers `inf`).
#[test]
fn datasets_non_finite_quaternion_errors() {
    let dir = TempDir::new("ds_quat_inf");
    assert!(euroc::read_groundtruth(dir.write("g.csv", "0,0,0,0,inf,0,0,0,0,0,0,0,0,0,0,0,0\n")).is_err());
    assert!(tum::read_groundtruth(dir.write("g.txt", "0 0 0 0 nan 0 0 0\n")).is_err());
    assert!(colmap::read_images_text(dir.write("i.txt", "1 nan 0 0 0 0 0 0 1 a.jpg\n\n")).is_err());
}

// ===========================================================================
// 4. Non-finite data: what is NOT checked
// ===========================================================================

/// `f64::from_str` accepts `nan`, `inf`, `-inf` and `1e400` (which overflows to
/// `inf`), so a translation of `inf` sails through `read_groundtruth` and lands
/// in a `Pose` whose `matrix()` is full of NaNs.
#[test]
fn datasets_euroc_translation_must_be_finite() {
    let dir = TempDir::new("ds_finite_gt");
    let path = dir.write(
        "g.csv",
        "1,inf,-inf,nan,1,0,0,0,0,0,0,0,0,0,0,0,0\n2,1e400,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0\n",
    );
    let states = euroc::read_groundtruth(&path).expect("read");
    assert_eq!(states.len(), 2, "both rows are well-formed apart from the values");
    for s in &states {
        let t = s.pose.translation;
        assert!(
            t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
            "KNOWN BUG: euroc::read_groundtruth returned a non-finite translation ({}, {}, {})",
            t.x,
            t.y,
            t.z
        );
    }
}

/// Same for the TUM ground truth reader.
#[test]
fn datasets_tum_translation_must_be_finite() {
    let dir = TempDir::new("ds_finite_tum");
    let path = dir.write("g.txt", "1.0 inf nan 0 0 0 0 1\n2.0 0 0 0 0 0 0 1\n");
    let poses = tum::read_groundtruth(&path).expect("read");
    for p in &poses {
        let t = p.pose.translation;
        assert!(
            t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
            "KNOWN BUG: tum::read_groundtruth returned a non-finite translation ({}, {}, {})",
            t.x,
            t.y,
            t.z
        );
    }
    assert!(poses[0].timestamp.is_finite());
}

/// Same for the KITTI times reader, which (unlike `kitti::read_poses`) has no
/// finiteness check at all.
#[test]
fn datasets_kitti_times_must_be_finite() {
    let dir = TempDir::new("ds_finite_kitti");
    let path = dir.write("times.txt", "0.0\ninf\nnan\n-1e400\n");
    let times = kitti::read_times(&path).expect("read");
    for (i, t) in times.iter().enumerate() {
        assert!(
            t.is_finite(),
            "KNOWN BUG: kitti::read_times returned a non-finite timestamp at index {i}: {t}"
        );
    }
}

/// Same for the TUM index reader.
#[test]
fn datasets_tum_index_timestamp_must_be_finite() {
    let dir = TempDir::new("ds_finite_tum_idx");
    let path = dir.write("rgb.txt", "inf rgb/a.png\nnan rgb/b.png\n");
    let entries = tum::read_index(&path).expect("read");
    for e in &entries {
        assert!(
            e.timestamp.is_finite(),
            "KNOWN BUG: tum::read_index returned a non-finite timestamp: {}",
            e.timestamp
        );
    }
}

/// Same for the generic EuRoC csv reader.
#[test]
fn datasets_euroc_csv_values_must_be_finite() {
    let dir = TempDir::new("ds_finite_csv");
    let path = dir.write("c.csv", "1,inf\n2,nan\n");
    // `f64::from_str` accepts "inf" and "NaN" — they are valid IEEE-754
    // spellings, not syntax errors — so without an explicit check a single
    // malformed column produced a pose of `inf` that propagated into every
    // metric computed from the trajectory.
    let err = euroc::read_csv(&path).expect_err("a non-finite value must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("finite"),
        "the error should say the value was not finite, got: {msg}"
    );
}

/// Same for the COLMAP camera intrinsics: `fx = inf` is accepted and produces a
/// camera that projects everything to `inf`.
#[test]
fn datasets_colmap_camera_params_must_be_finite() {
    let dir = TempDir::new("ds_finite_cam");
    let path = dir.write("c.txt", "1 PINHOLE 640 480 inf 500 nan 240\n");
    let cams = colmap::read_cameras_text(&path).expect("read");
    for c in &cams {
        for p in &c.params {
            assert!(
                p.is_finite(),
                "KNOWN BUG: colmap::read_cameras_text accepted a non-finite parameter: {p}"
            );
        }
    }
}

/// Same for COLMAP 3-D points and reprojection errors.
#[test]
fn datasets_colmap_points_must_be_finite() {
    let dir = TempDir::new("ds_finite_pts");
    let path = dir.write("p.txt", "1 inf nan -inf 255 128 0 0.5\n2 0 0 0 0 0 0 inf\n");
    let points = colmap::read_points3d_text(&path).expect("read");
    for p in &points {
        let v = p.position;
        assert!(
            v.x.is_finite() && v.y.is_finite() && v.z.is_finite() && p.error.is_finite(),
            "KNOWN BUG: colmap::read_points3d_text returned a non-finite point \
             ({}, {}, {}) error {}",
            v.x,
            v.y,
            v.z,
            p.error
        );
    }
}

/// COLMAP 2-D observations: a NaN pixel coordinate is accepted.
#[test]
fn datasets_colmap_observations_must_be_finite() {
    let dir = TempDir::new("ds_finite_obs");
    let path = dir.write("i.txt", "1 1 0 0 0 0 0 0 1 a.jpg\nnan inf 1\n");
    let images = colmap::read_images_text(&path).expect("read");
    for p in &images[0].points2d {
        assert!(
            p.x.is_finite() && p.y.is_finite(),
            "KNOWN BUG: colmap::read_images_text accepted a non-finite observation ({}, {})",
            p.x,
            p.y
        );
    }
}

// ===========================================================================
// 5. COLMAP two-line structure
// ===========================================================================

/// An image header with exactly nine fields has no name at all.
#[test]
fn colmap_image_header_without_name_errors() {
    let dir = TempDir::new("cm_noname");
    let e = colmap::read_images_text(dir.write("i.txt", "1 1 0 0 0 0 0 0 1\n\n")).expect_err("no name");
    assert!(matches!(e, cv_core::Error::InvalidInput(_)), "got {e}");
}

/// The file ends on an image header with no POINTS2D line.
#[test]
fn colmap_image_header_without_points_line_errors() {
    let dir = TempDir::new("cm_nopts");
    let e = colmap::read_images_text(dir.write("i.txt", "1 1 0 0 0 0 0 0 1 a.jpg\n")).expect_err("no points line");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// A POINTS2D line whose token count is not a multiple of three.
#[test]
fn colmap_points2d_not_a_multiple_of_three_errors() {
    let dir = TempDir::new("cm_triples");
    let e = colmap::read_images_text(dir.write("i.txt", "1 1 0 0 0 0 0 0 1 a.jpg\n1 2 3 4\n")).expect_err("4 tokens");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// A colour channel outside `0..=255`.
#[test]
fn colmap_colour_out_of_range_errors() {
    let dir = TempDir::new("cm_colour");
    for bad in ["256", "-1", "999", "1.5"] {
        let path = dir.write(&format!("p{bad}.txt"), &format!("1 0 0 0 {bad} 0 0 0\n"));
        assert!(
            colmap::read_points3d_text(&path).is_err(),
            "colour channel {bad} was accepted"
        );
    }
}

/// An odd number of TRACK[] tokens.
#[test]
fn colmap_odd_track_errors() {
    let dir = TempDir::new("cm_track");
    let e = colmap::read_points3d_text(dir.write("p.txt", "1 0 0 0 1 1 1 0.5 1 0 2\n")).expect_err("odd track");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// An image name containing spaces is re-joined with single spaces, so the
/// round trip is lossy. It must not panic or mis-index.
#[test]
fn colmap_image_name_with_spaces_is_joined() {
    let dir = TempDir::new("cm_spaces");
    let images = colmap::read_images_text(dir.write("i.txt", "1 1 0 0 0 0 0 0 1 a b.jpg\n\n")).expect("name");
    assert_eq!(images[0].name, "a b.jpg");
}

// ===========================================================================
// 6. TUM associate
// ===========================================================================

/// `associate` must not pair entries whose timestamps are not finite.
#[test]
fn tum_associate_skips_non_finite_timestamps() {
    let a = vec![
        tum::IndexEntry { timestamp: f64::NAN, filename: "a0".into() },
        tum::IndexEntry { timestamp: f64::INFINITY, filename: "a1".into() },
    ];
    let b = vec![
        tum::IndexEntry { timestamp: f64::NAN, filename: "b0".into() },
        tum::IndexEntry { timestamp: f64::INFINITY, filename: "b1".into() },
    ];
    assert!(tum::associate(&a, &b, 1.0).is_empty());
}

/// Duplicate timestamps on both sides: a 1-to-1 association must still hold.
#[test]
fn tum_associate_duplicate_timestamps_are_one_to_one() {
    let a = vec![tum::IndexEntry { timestamp: 1.0, filename: "a0".into() }];
    let b = vec![
        tum::IndexEntry { timestamp: 1.0, filename: "b0".into() },
        tum::IndexEntry { timestamp: 1.0, filename: "b1".into() },
    ];
    let m = tum::associate(&a, &b, 0.0);
    assert_eq!(m.len(), 1, "each entry may be consumed at most once");
}

/// A long list must terminate and produce a valid 1-to-1 matching.
#[test]
fn tum_associate_large_input_terminates() {
    let a: Vec<_> = (0..20_000)
        .map(|i| tum::IndexEntry { timestamp: i as f64, filename: format!("a{i}") })
        .collect();
    let b: Vec<_> = (0..20_000)
        .map(|i| tum::IndexEntry { timestamp: i as f64 + 0.0005, filename: format!("b{i}") })
        .collect();
    let m = tum::associate(&a, &b, 0.01);
    assert_eq!(m.len(), 20_000);
    // Sorted by a's timestamp, each index used once.
    for w in m.windows(2) {
        assert!(a[w[0].0].timestamp <= a[w[1].0].timestamp, "matches must be sorted by a");
    }
}

/// `max_dt = inf` accepts every candidate pair on paper, but `tum::associate`
/// filters with `diff <= max_dt` *and* `diff.is_finite()`, so the tolerance
/// ends up behaving as if it were zero for finite differences: nothing is
/// matched unless two entries carry the exact same timestamp.
///
/// The guard itself is a good one - it is what keeps NaN/Inf timestamps from
/// matching - so the only defect here is the doc comment, which promises
/// "every candidate pair at most `max_dt` apart" without mentioning the
/// finiteness filter.
#[test]
fn tum_associate_finite_timestamps_with_infinite_tolerance_never_match() {
    let a: Vec<_> = (0..50)
        .map(|i| tum::IndexEntry { timestamp: i as f64, filename: format!("a{i}") })
        .collect();
    let b: Vec<_> = (0..50)
        .map(|i| tum::IndexEntry { timestamp: i as f64, filename: format!("b{i}") })
        .collect();
    let matches = tum::associate(&a, &b, f64::INFINITY);
    assert!(
        matches.is_empty(),
        "KNOWN BUG: tum::associate(50 x 50, max_dt=inf) returned {} matches; \
         the `diff.is_finite()` filter makes the effective tolerance 0, so the documented \
         'every pair within max_dt' rule does not hold",
        matches.len()
    );
}

// ===========================================================================
// 7. KITTI specific
// ===========================================================================

/// KITTI's `poses.txt` has no comment syntax, and `read_poses` only skips
/// *empty* lines, so a `#` line is a parse error. Documented, not a misparse.
#[test]
fn kitti_comment_lines_are_rejected() {
    let dir = TempDir::new("kitti_comments");
    let e = kitti::read_poses(dir.write("p.txt", "# comment\n1 0 0 0 0 1 0 0 0 0 1 0\n")).expect_err("comment");
    assert!(matches!(e, cv_core::Error::ParseError(_)), "got {e}");
}

/// A rotation matrix with a non-orthonormal block (scale 2) is accepted by
/// `Pose::new`, which documents that it is unchecked. The value must at least
/// be reproducible, so record what the reader does with it.
#[test]
fn kitti_non_orthonormal_rotation_is_reported() {
    let dir = TempDir::new("kitti_scale");
    let path = dir.write("p.txt", "2 0 0 0 0 2 0 0 0 0 2 0\n");
    match kitti::read_poses(&path) {
        Err(_) => {}
        Ok(poses) => {
            // `Pose::new` uses `from_rotation_matrix_unchecked`: a scaled matrix
            // becomes a rotation with a *different* det. Flag it explicitly.
            let m = poses[0].rotation_matrix();
            let det = m.determinant();
            assert!(
                (det - 1.0).abs() > 1e-6,
                "KNOWN BUG: kitti::read_poses accepted a scale-2 'rotation' and normalised it \
                 to det = {det}; a KITTI pose is a rigid transform, so this is a misparse"
            );
        }
    }
}

/// A singular rotation matrix (all zeros) is the worst case for the
/// quaternion-from-matrix conversion.
#[test]
fn kitti_singular_rotation_is_handled() {
    let dir = TempDir::new("kitti_singular");
    let path = dir.write("p.txt", "0 0 0 1 0 0 0 2 0 0 0 3\n");
    let outcome = std::panic::catch_unwind(|| kitti::read_poses(&path).is_ok());
    assert!(outcome.is_ok(), "read_poses panicked on a singular rotation matrix");
    if let Ok(true) = outcome {
        // Accepted, but the det is 0, which is not a rotation at all.
        panic!("KNOWN BUG: kitti::read_poses accepted a singular (all-zero) rotation matrix");
    }
}
