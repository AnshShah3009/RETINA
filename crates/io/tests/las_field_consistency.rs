#![forbid(unsafe_code)]
//! Regression tests for three confirmed defects in `crates/io/src/las_io.rs`.
//!
//! All three are the same family: parallel per-point arrays that can drift out
//! of index-parity, and a value taken from the file without validation.
//! `LasData` indexes `colors`, `intensities`, `classifications`,
//! `return_numbers`, `number_of_returns` and `gps_times` by point index, so a
//! vector of the wrong length does not crash - it silently attributes one
//! point's attribute to a different point, which is far worse.
//!
//! | test | defect |
//! |---|---|
//! | `mask_shorter_than_points_*` | `filter_by_mask` `zip` truncation (defect 1) |
//! | `mask_longer_than_points_*` | `filter_by_mask` `zip` truncation (defect 1) |
//! | `classifications_shorter_than_points_*` | same path via `filter_by_classification` (defect 1) |
//! | `all_parallel_arrays_are_index_parallel_*` | per-point field detection (defect 2) |
//! | `header_bounds_are_not_adopted_verbatim_*` | unvalidated header box (defect 3) |
//!
//! Every test opens with a CONTROL: a well-formed LAS file written through the
//! crate's own `write_las`, read back, and asserted to have every parallel
//! array the same length, a valid box, and the true extent. Without that, a
//! parser that failed closed on everything would pass all the negative tests
//! vacuously.
//!
//! Temporary file names are unique per call (a distinct tag, the process id and
//! an atomic counter) because the suite runs its tests concurrently in a single
//! process and a shared path would let them clobber each other.

#![cfg(feature = "las")]

use cv_io::las_io::{read_las, write_las, LasData};
use nalgebra::Point3;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A temporary path unique to `(tag, call)` within this process.
fn temp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "cv_io_las_field_consistency_{}_{}_{}.las",
        tag,
        std::process::id(),
        n
    ));
    path
}

/// Three points spanning a non-degenerate extent: (0,0,0), (1,2,3), (2,4,6).
const TRUE_MIN: (f64, f64, f64) = (0.0, 0.0, 0.0);
const TRUE_MAX: (f64, f64, f64) = (2.0, 4.0, 6.0);

/// A `LasData` with every optional field populated, so a defect that shows up
/// as a length mismatch has the maximum surface to show up on.
fn fully_populated() -> LasData {
    LasData {
        points: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(2.0, 4.0, 6.0),
        ],
        colors: Some(vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]),
        intensities: Some(vec![0.1, 0.2, 0.3]),
        classifications: Some(vec![2, 3, 2]),
        return_numbers: Some(vec![1, 1, 2]),
        number_of_returns: Some(vec![1, 2, 2]),
        gps_times: Some(vec![100.0, 200.0, 300.0]),
        bounds: (0.0, 0.0, 0.0, 2.0, 4.0, 6.0),
        num_points: 3,
    }
}

/// Write `data` to a fresh temp file and return the path.
fn write_temp(tag: &str, data: &LasData) -> PathBuf {
    let path = temp_path(tag);
    write_las(&path, data).unwrap_or_else(|e| panic!("write_las({}): {e}", path.display()));
    path
}

/// Read a LAS file back, panicking on error - only well-formed files go through
/// this, so a failure is the test's own bug, not the parser's.
fn read_ok(path: &std::path::Path) -> LasData {
    read_las(path).unwrap_or_else(|e| panic!("read_las({}): {e}", path.display()))
}

/// `min <= max` and all finite, for a box a consumer could size an ROI from.
fn assert_valid_bounds(b: (f64, f64, f64, f64, f64, f64), what: &str) {
    let (min_x, min_y, min_z, max_x, max_y, max_z) = b;
    assert!(
        min_x <= max_x && min_y <= max_y && min_z <= max_z,
        "{what}: inverted bounding box {b:?} - min must not exceed max on any axis"
    );
    assert!(
        [min_x, min_y, min_z, max_x, max_y, max_z]
            .iter()
            .all(|v| v.is_finite()),
        "{what}: bounding box {b:?} contains a non-finite value"
    );
}

/// THE CONTROL. A well-formed file must read back with every parallel array the
/// same length as `points`, a valid box, and the extent of the points written.
///
/// This is what stops the negative tests passing vacuously: if `read_las`
/// regressed into dropping every point, or into never populating a field, the
/// negative assertions below would still "pass" while the parser was broken.
fn assert_well_formed(data: &LasData) {
    let path = write_temp("control", data);
    let r = read_ok(&path);
    let _ = std::fs::remove_file(&path);

    let n = r.points.len();
    assert_eq!(r.num_points, n, "CONTROL: num_points disagrees with points");
    assert_eq!(n, 3, "CONTROL: expected all 3 points back");

    for (name, len) in [
        ("colors", r.colors.as_ref().map(|v| v.len())),
        ("intensities", r.intensities.as_ref().map(|v| v.len())),
        (
            "classifications",
            r.classifications.as_ref().map(|v| v.len()),
        ),
        ("return_numbers", r.return_numbers.as_ref().map(|v| v.len())),
        (
            "number_of_returns",
            r.number_of_returns.as_ref().map(|v| v.len()),
        ),
        ("gps_times", r.gps_times.as_ref().map(|v| v.len())),
    ] {
        if let Some(len) = len {
            assert_eq!(len, n, "CONTROL: {name} is not index-parallel with points");
        }
    }

    assert_valid_bounds(r.bounds, "CONTROL");
    assert_eq!(
        (r.bounds.0, r.bounds.1, r.bounds.2),
        TRUE_MIN,
        "CONTROL: min of the control file's extent"
    );
    assert_eq!(
        (r.bounds.3, r.bounds.4, r.bounds.5),
        TRUE_MAX,
        "CONTROL: max of the control file's extent"
    );
}

/// Byte offsets of the six bounding-box doubles in a LAS public header block.
/// Fixed by the ASPRS spec (6 signature + 2 + 2 + 16 guid + 1 + 1 + 32 + 32 +
/// 2 + 2 + 2 + 2 + 4 + 4 + 4 + 1 + 2 + 4 + 15*4 + 6*8 == 179), and identical
/// for every point format and version, so these are safe to patch in place.
const OFF_MAX_X: usize = 179;
const OFF_MIN_X: usize = 187;
const OFF_MAX_Y: usize = 195;
const OFF_MIN_Y: usize = 203;
const OFF_MAX_Z: usize = 211;
const OFF_MIN_Z: usize = 219;

fn put_f64(bytes: &mut [u8], off: usize, v: f64) {
    bytes[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Write a well-formed file, then overwrite its header's bounding box with
/// `(min, max)` - i.e. produce a file whose *declared* extent is the given one
/// while its point records still hold the true extent. This is a valid LAS
/// file in every structural respect; only the box is a lie.
fn write_with_header_bounds(
    tag: &str,
    data: &LasData,
    min: (f64, f64, f64),
    max: (f64, f64, f64),
) -> PathBuf {
    let path = write_temp(tag, data);
    let mut bytes = std::fs::read(&path).expect("read back temp las");
    put_f64(&mut bytes, OFF_MIN_X, min.0);
    put_f64(&mut bytes, OFF_MIN_Y, min.1);
    put_f64(&mut bytes, OFF_MIN_Z, min.2);
    put_f64(&mut bytes, OFF_MAX_X, max.0);
    put_f64(&mut bytes, OFF_MAX_Y, max.1);
    put_f64(&mut bytes, OFF_MAX_Z, max.2);
    std::fs::write(&path, &bytes).expect("rewrite temp las");
    path
}

// ---------------------------------------------------------------------------
// DEFECT 1 - `filter_by_mask` / `filter_by_classification` truncate silently
// ---------------------------------------------------------------------------

/// A mask one entry short must not silently drop the trailing point.
///
/// `points.iter().zip(mask.iter())` stops at the shorter side, so with 3 points
/// and a 2-entry mask the third point vanished with no error and the result
/// claimed to be a valid 2-point cloud.
#[test]
fn mask_shorter_than_points_is_not_silently_truncated() {
    assert_well_formed(&fully_populated()); // CONTROL

    let data = fully_populated();
    let short = vec![true, true]; // 2 entries for 3 points

    let out = cv_io::las_io::filter_by_mask(&data, &short);

    // The mask is a caller error, not a partial request. Either report it or
    // keep everything; silently returning 2 of 3 points is the bug.
    assert_ne!(
        out.points.len(),
        2,
        "KNOWN BUG: a 2-entry mask over 3 points yielded a 2-point cloud - the \
         third point was dropped by zip with no error"
    );
    assert_eq!(
        out.points.len(),
        out.num_points,
        "num_points must describe the points actually kept"
    );
    // Whatever the outcome, the result must not be a lie about its contents.
    for (name, len) in [
        ("colors", out.colors.as_ref().map(|v| v.len())),
        ("intensities", out.intensities.as_ref().map(|v| v.len())),
        (
            "classifications",
            out.classifications.as_ref().map(|v| v.len()),
        ),
        ("gps_times", out.gps_times.as_ref().map(|v| v.len())),
    ] {
        if let Some(len) = len {
            assert_eq!(len, out.points.len(), "{name} desynchronized by filtering");
        }
    }
    assert_valid_bounds(out.bounds, "mask_shorter_than_points");
}

/// A mask one entry too long must not have its tail ignored.
#[test]
fn mask_longer_than_points_is_not_silently_ignored() {
    assert_well_formed(&fully_populated()); // CONTROL

    let data = fully_populated();
    let long = vec![true, true, true, true, true]; // 5 entries for 3 points

    let out = cv_io::las_io::filter_by_mask(&data, &long);

    // The mask does not describe the point list, so no subset of it is a
    // meaningful answer. Silently ignoring the tail happens to give the "right"
    // 3 here, but that is the bug in a lucky direction: a tail of `false` over
    // 3 points, or a mask of `[true, false] ++ 3 trues`, produces a wrong cloud
    // with no error. The mismatch must be reported rather than absorbed, so the
    // result cannot claim to be a normal 3-point filter.
    assert_ne!(
        out.num_points, 3,
        "KNOWN BUG: a 5-entry mask over 3 points was accepted as a normal filter \
         and the 2 extra entries were discarded with no error"
    );
    assert_eq!(
        out.points.len(),
        out.num_points,
        "num_points must describe the points actually kept"
    );
    for (name, len) in [
        ("colors", out.colors.as_ref().map(|v| v.len())),
        ("intensities", out.intensities.as_ref().map(|v| v.len())),
        (
            "classifications",
            out.classifications.as_ref().map(|v| v.len()),
        ),
        ("gps_times", out.gps_times.as_ref().map(|v| v.len())),
    ] {
        if let Some(len) = len {
            assert_eq!(len, out.points.len(), "{name} desynchronized by filtering");
        }
    }
    assert_valid_bounds(out.bounds, "mask_longer_than_points");
}

/// An exactly-sized mask must still work - the length check must not have
/// broken the ordinary path.
#[test]
fn an_exactly_sized_mask_still_filters() {
    let data = fully_populated();
    let out = cv_io::las_io::filter_by_mask(&data, &[true, false, true]);

    assert_eq!(out.num_points, 2, "two points were flagged true");
    assert_eq!(out.points.len(), 2);
    // Kept points 0 and 2, so the box must span them, not all three.
    assert_valid_bounds(out.bounds, "exact mask");
    assert_eq!((out.bounds.0, out.bounds.1, out.bounds.2), (0.0, 0.0, 0.0));
    assert_eq!((out.bounds.3, out.bounds.4, out.bounds.5), (2.0, 4.0, 6.0));
    // colours[1] was the green point; keeping indices 0 and 2 keeps red and blue.
    let colors = out.colors.expect("colors dropped by filtering");
    assert_eq!(colors.len(), 2, "colors desynchronized by filtering");
    assert!((colors[0].x - 1.0).abs() < 1e-3, "wrong first colour kept");
    assert!((colors[1].z - 1.0).abs() < 1e-3, "wrong second colour kept");
}

/// The same defect reached through `filter_by_classification`, where the mask is
/// derived from `classifications`. A `LasData` whose `classifications` is
/// shorter than its `points` used to lose points with no error.
#[test]
fn classifications_shorter_than_points_does_not_drop_points() {
    assert_well_formed(&fully_populated()); // CONTROL

    let mut data = fully_populated();
    data.classifications = Some(vec![2, 3]); // 2 entries for 3 points

    let out = cv_io::las_io::filter_by_classification(&data, 2);

    assert_eq!(
        out.points.len(),
        out.num_points,
        "num_points must describe the points actually kept"
    );
    // The mask is one entry short of the point list, so filtering is unsound:
    // the result must not be presented as a normal per-class subset.
    assert!(
        out.points.len() == 0,
        "KNOWN BUG: a 2-entry `classifications` over 3 points produced a \
         {}-point cloud; the mask was built from a vector that is not \
         index-parallel with points, so the third point was dropped silently",
        out.points.len()
    );
    assert_valid_bounds(out.bounds, "classifications_shorter_than_points");
}

/// A well-formed `LasData` must still filter normally, so the guard added for
/// the mismatch case has not broken the real path.
#[test]
fn a_consistent_classification_filter_still_works() {
    let data = fully_populated(); // classifications = [2, 3, 2]
    let ground = cv_io::las_io::filter_by_classification(&data, 2);

    assert_eq!(ground.num_points, 2, "two ground points");
    assert_eq!(ground.points.len(), 2);
    assert_eq!(
        ground.classifications.as_ref().map(|v| v.as_slice()),
        Some([2u8, 2u8].as_slice()),
        "the kept classifications must be the ground ones"
    );
    assert_eq!(
        ground.gps_times.as_ref().map(|v| v.len()),
        Some(2),
        "gps_times desynchronized by filtering"
    );
}

// ---------------------------------------------------------------------------
// DEFECT 2 - per-point optional-field detection can desynchronise the vectors
// ---------------------------------------------------------------------------

/// CONTROL for defect 2: a file carrying colour and GPS time must come back
/// with both vectors the same length as `points`.
#[test]
fn a_colour_and_gps_file_reads_back_index_parallel() {
    let data = fully_populated(); // has colors AND gps_times
    let path = write_temp("d2_control", &data);
    let r = read_ok(&path);
    let _ = std::fs::remove_file(&path);

    assert_eq!(r.points.len(), 3);
    assert_eq!(
        r.colors.as_ref().map(|v| v.len()),
        Some(3),
        "CONTROL: colors must be index-parallel with points"
    );
    assert_eq!(
        r.gps_times.as_ref().map(|v| v.len()),
        Some(3),
        "CONTROL: gps_times must be index-parallel with points"
    );
    assert_eq!(
        r.classifications.as_ref().map(|v| v.len()),
        Some(3),
        "CONTROL: classifications must be index-parallel with points"
    );
}

/// Every point format the writer can emit must come back index-parallel.
///
/// `read_las` used to decide whether colour and GPS time exist from the FIRST
/// record only and reuse that answer for all later records, so any record that
/// failed to supply a field skipped its push and left `points` longer than
/// `colors`/`gps_times`. That is the same shape as the `read_pcd` bug where
/// `colors` was attached with no `len() == points.len()` gate. Whether a
/// deviation is reachable today depends on the `las` crate's reader (see the
/// notes in the commit report: presence is currently derived from the
/// file-wide point format, so it is uniform), which is exactly why the
/// invariant is locked here rather than left to chance.
///
/// This sweeps both field layouts the crate can write: format 3 (XYZ + GPS +
/// RGB, both optional fields present) and format 0 (XYZ only, both absent).
#[test]
fn all_parallel_arrays_are_index_parallel_for_both_field_layouts() {
    // CONTROL, colour + GPS layout.
    assert_well_formed(&fully_populated());

    // XYZ-only layout: neither optional field is present in the file at all,
    // so both must come back absent rather than partially filled.
    let mut bare = fully_populated();
    bare.colors = None;
    bare.gps_times = None;
    let path = write_temp("d2_bare", &bare);
    let r = read_ok(&path);
    let _ = std::fs::remove_file(&path);

    assert_eq!(r.points.len(), 3, "all points must survive");
    for (name, len) in [
        ("colors", r.colors.as_ref().map(|v| v.len())),
        ("gps_times", r.gps_times.as_ref().map(|v| v.len())),
    ] {
        if let Some(len) = len {
            assert_eq!(
                len,
                r.points.len(),
                "{name} is {len} long for {} points - the per-point vectors are \
                 not index-parallel",
                r.points.len()
            );
        }
    }
    // Mandatory fields are still populated for every record.
    for (name, len) in [
        ("intensities", r.intensities.as_ref().map(|v| v.len())),
        (
            "classifications",
            r.classifications.as_ref().map(|v| v.len()),
        ),
        ("return_numbers", r.return_numbers.as_ref().map(|v| v.len())),
        (
            "number_of_returns",
            r.number_of_returns.as_ref().map(|v| v.len()),
        ),
    ] {
        assert_eq!(
            len,
            Some(r.points.len()),
            "{name} must be populated for every record"
        );
    }
}

/// A file whose point records all carry colour and GPS time must have those
/// values attached to the right point indices - the observable consequence of
/// the vectors being index-parallel.
#[test]
fn colours_and_gps_times_land_on_the_right_point_indices() {
    let data = fully_populated();
    let path = write_temp("d2_indices", &data);
    let r = read_ok(&path);
    let _ = std::fs::remove_file(&path);

    let gps = r.gps_times.expect("gps_times missing");
    assert_eq!(gps.len(), r.points.len());
    for (i, expected) in [100.0f64, 200.0, 300.0].iter().enumerate() {
        assert!(
            (gps[i] - expected).abs() < 1e-3,
            "gps_times[{i}] = {} but point {i} was written with {expected}",
            gps[i]
        );
    }
    let colors = r.colors.expect("colors missing");
    assert_eq!(colors.len(), r.points.len());
    assert!(
        (colors[0].x - 1.0).abs() < 1e-3,
        "colours are not in point order"
    );
    assert!(
        (colors[1].y - 1.0).abs() < 1e-3,
        "colours are not in point order"
    );
    assert!(
        (colors[2].z - 1.0).abs() < 1e-3,
        "colours are not in point order"
    );
}

// ---------------------------------------------------------------------------
// DEFECT 3 - the file's bounding box is adopted without validation
// ---------------------------------------------------------------------------

/// CONTROL for defect 3: a well-formed file must report the extent of its own
/// points.
#[test]
fn a_well_formed_file_reports_its_true_extent() {
    let data = fully_populated();
    let path = write_temp("d3_control", &data);
    let r = read_ok(&path);
    let _ = std::fs::remove_file(&path);

    assert_valid_bounds(r.bounds, "CONTROL");
    assert_eq!((r.bounds.0, r.bounds.1, r.bounds.2), TRUE_MIN);
    assert_eq!((r.bounds.3, r.bounds.4, r.bounds.5), TRUE_MAX);
}

/// A header declaring `min > max` on every axis must not produce an inverted
/// box. `header.bounds()` was copied straight into `LasData::bounds`, and that
/// is what every consumer uses to size an ROI or a voxel grid - an inverted box
/// makes both meaningless.
#[test]
fn an_inverted_header_box_is_not_adopted() {
    assert_well_formed(&fully_populated()); // CONTROL

    let path = write_with_header_bounds(
        "d3_inverted",
        &fully_populated(),
        (1000.0, 1000.0, 1000.0),
        (-1000.0, -1000.0, -1000.0),
    );

    match read_las(&path) {
        Ok(r) => assert_valid_bounds(
            r.bounds,
            "KNOWN BUG: read_las adopted an inverted header box verbatim",
        ),
        // Rejecting the file is an acceptable outcome; adopting the box is not.
        Err(_) => {}
    }
    let _ = std::fs::remove_file(&path);
}

/// A header box of `min = 0, max = f64::MAX` must not propagate - a consumer
/// sizing a voxel grid from it would allocate for the whole float range.
#[test]
fn a_saturated_header_box_is_not_adopted() {
    assert_well_formed(&fully_populated()); // CONTROL

    let path = write_with_header_bounds(
        "d3_saturated",
        &fully_populated(),
        (0.0, 0.0, 0.0),
        (f64::MAX, f64::MAX, f64::MAX),
    );

    match read_las(&path) {
        Ok(r) => {
            assert_valid_bounds(
                r.bounds,
                "KNOWN BUG: read_las adopted a saturated (max = f64::MAX) header box",
            );
            assert!(
                r.bounds.3 < 1e6 && r.bounds.4 < 1e6 && r.bounds.5 < 1e6,
                "KNOWN BUG: reported max {:?} is the file's f64::MAX, not the extent \
                 of the {} points read",
                (r.bounds.3, r.bounds.4, r.bounds.5),
                r.points.len()
            );
        }
        Err(_) => {}
    }
    let _ = std::fs::remove_file(&path);
}

/// The subtle case: the header box is *valid* but simply does not describe the
/// points - here it under-reports the extent by three orders of magnitude. A
/// "validate that min <= max" check would pass this, so it is what settles
/// validate-vs-recompute: only recomputing from the points gets it right.
#[test]
fn a_stale_but_valid_header_box_is_not_adopted() {
    assert_well_formed(&fully_populated()); // CONTROL

    let path = write_with_header_bounds(
        "d3_stale",
        &fully_populated(),
        (0.0, 0.0, 0.0),
        (0.001, 0.001, 0.001),
    );

    let r = read_las(&path).unwrap_or_else(|e| panic!("read_las: {e}"));
    let _ = std::fs::remove_file(&path);

    assert_valid_bounds(r.bounds, "stale header box");
    assert_eq!(
        (r.bounds.0, r.bounds.1, r.bounds.2),
        TRUE_MIN,
        "min should be the true min of the points read"
    );
    assert_eq!(
        (r.bounds.3, r.bounds.4, r.bounds.5),
        TRUE_MAX,
        "KNOWN BUG: the header's stale under-reported max {:?} was adopted; the \
         points actually extend to {:?}",
        (r.bounds.3, r.bounds.4, r.bounds.5),
        TRUE_MAX
    );
}
