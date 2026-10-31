#![forbid(unsafe_code)]
//! Regression tests for six confirmed correctness defects in
//! `crates/io/src/pcd.rs`.
//!
//! Each defect is a case where `read_pcd` (or `write_pcd`) returned `Ok` with a
//! result that is not the file's data: a silently empty cloud, a colour vector
//! too short to index, a fabricated zero normal, a timestamp handed back as a
//! coordinate, a coordinate silently zeroed, and a panic on write. Every test
//! below writes a real file to a temporary path and reads it back through the
//! public `read_pcd` API, and every test starts with a CONTROL assertion - a
//! well-formed PCD of the same shape that must still parse to exactly the
//! values it declares - so a test cannot pass because the parser failed
//! closed on everything.
//!
//! | test                                    | defect                                    |
//! |-----------------------------------------|-------------------------------------------|
//! | `width_times_height_overflow_is_an_error`| unguarded `width * height`               |
//! | `binary_rgb_field_of_wrong_size_*`      | `colors` desynchronised from `points`    |
//! | `ascii_short_row_is_not_padded_or_dropped` | ASCII short rows                      |
//! | `fields_without_xyz_is_rejected_*`      | positional coordinate fallback            |
//! | `unsupported_type_size_pair_is_an_error`| `_ => 0.0` catch-all decoder             |
//! | `write_pcd_on_short_colors_does_not_panic`| writer indexes `colors[i]` unguarded    |
//!
//! Temporary file names are unique per test (a distinct tag plus the process id
//! plus a per-call counter) because the suite runs its tests concurrently in a
//! single process and a shared path would let them clobber each other.

use cv_core::PointCloud;
use cv_io::pcd::{read_pcd, write_pcd, write_pcd_binary, write_pcd_binary_compressed};
use nalgebra::{Point3, Vector3};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
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
        "cv_io_pcd_defect_{}_{}_{}.pcd",
        tag,
        std::process::id(),
        n
    ));
    path
}

/// Write `contents` to a fresh file and return its path.
fn write_temp(tag: &str, contents: &[u8]) -> PathBuf {
    let path = temp_path(tag);
    let mut f = File::create(&path).expect("create temp pcd");
    f.write_all(contents).expect("write temp pcd");
    f.flush().expect("flush temp pcd");
    path
}

/// Read a PCD back from a file. Panics on error - only well-formed files go
/// through this, and a failure there is the test's own bug, not the parser's.
fn read_file(path: &std::path::Path) -> PointCloud {
    let f = File::open(path).expect("open temp pcd");
    read_pcd(BufReader::new(f)).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The error message from reading `path`, panicking if the parse succeeded.
fn read_file_err(path: &std::path::Path) -> String {
    let f = File::open(path).expect("open temp pcd");
    match read_pcd(BufReader::new(f)) {
        Ok(cloud) => panic!(
            "KNOWN BUG: {} parsed as Ok with {} point(s) = {:?}",
            path.display(),
            cloud.len(),
            cloud.points
        ),
        Err(e) => format!("{e}"),
    }
}

/// Every attribute vector that is present must cover every point.
fn assert_attributes_cover_points(cloud: &PointCloud) {
    if let Some(c) = &cloud.colors {
        assert_eq!(
            c.len(),
            cloud.len(),
            "colors has {} entries for {} points",
            c.len(),
            cloud.len()
        );
    }
    if let Some(n) = &cloud.normals {
        assert_eq!(
            n.len(),
            cloud.len(),
            "normals has {} entries for {} points",
            n.len(),
            cloud.len()
        );
    }
}

fn f32le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Defect 1: `points_count = width * height` with no overflow check
// ---------------------------------------------------------------------------

/// `POINTS` is an optional header line, so the reader falls back to
/// `WIDTH * HEIGHT` - and both operands come from the file.
///
/// `WIDTH 9223372036854775808` (2^63) times `HEIGHT 2` is 2^64, one past
/// `usize::MAX`. In this repo's release profile (no `overflow-checks`) the
/// product WRAPS to 0, so `count` became 0, the binary loop never ran, and the
/// reader returned `Ok` with **zero points** while the complete, valid
/// two-point body below sat unread - a silently empty cloud, indistinguishable
/// from an empty file. With overflow checks on, the same line panicked.
///
/// The control: the identical file with a `POINTS 2` line parses to both
/// points, so the body is known-good and only the product is at fault.
#[test]
fn width_times_height_overflow_is_an_error_not_a_silent_empty_cloud() {
    // CONTROL: the same header, with POINTS, and the same two records.
    let control = write_temp(
        "wh_control",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n",
    );
    // 8 = SPACE, 9 = TAB, 10 = LF
    let mut control_bytes = std::fs::read(&control).expect("read control");
    control_bytes.extend(f32le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    std::fs::write(&control, &control_bytes).expect("append control body");
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control body must parse to two points");
    assert_eq!(cloud.points[1], Point3::new(4.0, 5.0, 6.0));

    // The defect: WIDTH * HEIGHT does not fit in a usize.
    let path = write_temp(
        "wh_overflow",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 9223372036854775808\nHEIGHT 2\nVIEWPOINT 0 0 0 1 0 0 0\nDATA binary\n",
    );
    let mut bytes = std::fs::read(&path).expect("read");
    bytes.extend(f32le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    std::fs::write(&path, &bytes).expect("append body");
    let err = read_file_err(&path);
    assert!(
        err.contains("WIDTH") && err.contains("HEIGHT"),
        "the error should name the product that did not fit, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Defect 2: a declared `rgb` whose SIZE is not 4 desynchronises `colors`
// ---------------------------------------------------------------------------

/// The binary colour branch is guarded by `if size == 4`, but `colors` was
/// created from the mere presence of an `rgb` field and then assigned onto the
/// cloud with no length check. So `FIELDS x y z rgb / SIZE 4 4 4 1` returned
/// `Ok` with 2 points and `colors = Some([])`: every consumer that indexes
/// `colors[i]` then panics, and this crate's own `write_pcd` demonstrably
/// panics on exactly that cloud.
///
/// A packed rgb field is a u32, so a 1-byte rgb is not decodable and the file
/// has no colour data at all - the correct result is Ok with the two points and
/// no colour vector, never a half-populated one.
#[test]
fn binary_rgb_field_of_wrong_size_does_not_desync_colors() {
    // CONTROL: identical except SIZE 4 on the rgb field, which is a packed u32
    // and decodes normally.
    let control = write_temp(
        "rgb_size_control",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\n\
          COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut control_bytes = std::fs::read(&control).expect("read control");
    control_bytes.extend(f32le(&[1.0, 2.0, 3.0]));
    // 0x00FF0000 as a bit pattern = red; stored in a 4-byte F slot.
    control_bytes.extend(f32::from_bits(0x00FF_0000).to_le_bytes());
    control_bytes.extend(f32le(&[4.0, 5.0, 6.0]));
    // 0x0000FF00 as a bit pattern = green; stored in a 4-byte F slot.
    control_bytes.extend(f32::from_bits(0x0000_FF00).to_le_bytes());
    std::fs::write(&control, &control_bytes).expect("append control body");
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control must parse to two points");
    let colors = cloud.colors.as_ref().expect("control has colors");
    assert_eq!(colors.len(), 2, "control colors must cover both points");
    assert!((colors[0].x - 1.0).abs() < 0.01, "control point 0 is red");
    assert!((colors[1].y - 1.0).abs() < 0.01, "control point 1 is green");

    // The defect: rgb declared with SIZE 1, so the colour branch is skipped.
    let path = write_temp(
        "rgb_size_1",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 1\nTYPE F F F F\n\
          COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut bytes = std::fs::read(&path).expect("read");
    // 2 records of (x, y, z, 1-byte rgb) = 13 bytes each.
    bytes.extend(f32le(&[1.0, 2.0, 3.0]));
    bytes.push(0xFF);
    bytes.extend(f32le(&[4.0, 5.0, 6.0]));
    bytes.push(0xFF);
    std::fs::write(&path, &bytes).expect("append body");

    let f = File::open(&path).expect("open");
    let cloud = read_pcd(BufReader::new(f)).expect("geometry is intact, so Ok is correct");
    assert_eq!(cloud.len(), 2, "both records are valid geometry");
    assert_eq!(cloud.points[0], Point3::new(1.0, 2.0, 3.0));
    assert_eq!(cloud.points[1], Point3::new(4.0, 5.0, 6.0));
    // The heart of the defect: `Some` but shorter than `points`.
    assert_attributes_cover_points(&cloud);
    assert!(
        cloud.colors.is_none(),
        "an undecodable rgb field must not publish a colour vector, got {:?}",
        cloud.colors
    );
    // The writer must not panic on whatever the reader produced.
    let mut out = Vec::new();
    write_pcd(&mut out, &cloud).expect("write_pcd on a cloud with no colours");
    write_pcd_binary(&mut out, &cloud).expect("write_pcd_binary ditto");
}

/// The same defect reached through a record whose field offsets/sizes make the
/// colour branch produce nothing, this time with a *partial* fill: the guard
/// must be a length check, not a truthiness check.
#[test]
fn binary_short_colors_vector_is_dropped_not_published() {
    // CONTROL: a well-formed cloud with two colours round-trips.
    let mut control_cloud =
        PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
    control_cloud.colors = Some(vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)]);
    let control = write_temp("short_colors_control", b"");
    {
        let mut w = BufWriter::new(File::create(&control).expect("create"));
        write_pcd(&mut w, &control_cloud).expect("control write");
        w.flush().expect("flush");
    }
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2);
    assert_eq!(cloud.colors.as_ref().expect("colors").len(), 2);

    // The defect: build the cloud the way the reader did when the colour branch
    // was skipped - `colors` is `Some` but empty.
    let mut broken = PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
    broken.colors = Some(Vec::new());

    let mut out = Vec::new();
    // Before the fix this panicked with "index out of bounds: the len is 0 but
    // the index is 0" at `colors[i]`.
    write_pcd(&mut out, &broken).expect("write_pcd must not panic on a short colors vector");
    write_pcd_binary(&mut out, &broken).expect("write_pcd_binary must not panic either");
    write_pcd_binary_compressed(&mut out, &broken).expect("write_pcd_binary_compressed too");
    assert!(
        !out.is_empty(),
        "the writer still has to produce a file for the points it does have"
    );
}

// ---------------------------------------------------------------------------
// Defect 3: ASCII short rows
// ---------------------------------------------------------------------------

/// A truncated row used to be padded out with `unwrap_or(0.0)`, so a declared
/// 9-column FIELDS with one short body row returned `Ok` and reported the
/// truncated point with a ZERO NORMAL and a BLACK COLOUR - fabricated values
/// that are exactly what a normal-estimation or shading consumer treats as
/// valid input.
#[test]
fn ascii_short_row_is_not_padded_with_zero_normal_and_black_colour() {
    // CONTROL: the full 9-column body parses to three points with real normals
    // and a real packed colour.
    let control = write_temp(
        "ascii_short_control",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z rgb\n\
          SIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\n\
          WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n\
          1 2 3 0 0 1 2.341805152028776e-38\n\
          4 5 6 1 0 0 2.341805152028776e-38\n\
          7 8 9 0 1 0 2.341805152028776e-38\n",
    );
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 3, "control must parse to three points");
    assert_eq!(cloud.points[2], Point3::new(7.0, 8.0, 9.0));
    let normals = cloud.normals.as_ref().expect("control has normals");
    assert_eq!(
        normals.len(),
        3,
        "control normals must cover all three points"
    );
    assert_eq!(normals[0], Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(normals[2], Vector3::new(0.0, 1.0, 0.0));
    let colors = cloud.colors.as_ref().expect("control has colors");
    assert_eq!(colors.len(), 3);
    assert!(
        (colors[0].x - 1.0).abs() < 0.01,
        "control point 0 must decode to red, got {:?}",
        colors[0]
    );

    // The defect: the second row is missing its normal and colour columns.
    let path = write_temp(
        "ascii_short_row",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z rgb\n\
          SIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\n\
          WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n\
          1 2 3 0 0 1 2.341805152028776e-38\n\
          4 5 6\n\
          7 8 9 0 1 0 2.341805152028776e-38\n",
    );
    let err = read_file_err(&path);
    assert!(
        err.contains("column") || err.contains("FIELDS"),
        "the error should describe the short row, got: {err}"
    );
    assert!(
        !err.contains("0 0 0"),
        "the error must not present the missing values as zeros, got: {err}"
    );
}

/// The other half of the same defect: `if values.len() < 3 { continue }` dropped
/// a short row on the floor, so a three-row body with a two-column middle line
/// returned `Ok` with two points and the caller could not tell that a point was
/// lost.
#[test]
fn ascii_two_column_row_is_not_silently_dropped() {
    // CONTROL: the same three rows, all complete.
    let control = write_temp(
        "ascii_drop_control",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n1 2 3\n4 5 6\n7 8 9\n",
    );
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 3, "control must parse to three points");
    assert_eq!(cloud.points[2], Point3::new(7.0, 8.0, 9.0));

    // The defect: the middle row is two columns wide.
    let path = write_temp(
        "ascii_drop_row",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n1 2 3\n4 5\n7 8 9\n",
    );
    let err = read_file_err(&path);
    assert!(
        err.contains("column") || err.contains("FIELDS"),
        "the error should describe the short row, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Defect 4: FIELDS without x/y/z
// ---------------------------------------------------------------------------

/// The coordinate indices fell back to "column 0 is x, column 1 is y, column 2
/// is z", so `FIELDS intensity ring time` returned `Ok` with
/// `[[10,20,30],[40,50,60]]` - intensity, ring index and timestamp handed back
/// as geometry. Nothing downstream can tell those points from a real scan, and
/// every bound, centroid and normal estimate computed from them is wrong.
#[test]
fn fields_without_xyz_is_rejected_ascii() {
    // CONTROL: adding the three coordinate columns to the same rows parses.
    let control = write_temp(
        "no_xyz_control_ascii",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity ring time\n\
          SIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\n\
          WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n\
          1 2 3 10 0 0.5\n4 5 6 20 1 0.5\n",
    );
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control must parse to two points");
    assert_eq!(cloud.points[0], Point3::new(1.0, 2.0, 3.0));
    assert_eq!(cloud.points[1], Point3::new(4.0, 5.0, 6.0));

    // The defect: the same rows, but the header never says where x, y, z are.
    let path = write_temp(
        "no_xyz_ascii",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS intensity ring time\nSIZE 4 4 4\n\
          TYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n\
          10 20 30\n40 50 60\n",
    );
    let err = read_file_err(&path);
    assert!(
        err.contains('x') && err.contains('y') && err.contains('z'),
        "the error should name the missing coordinate fields, got: {err}"
    );
}

/// The binary path had the same positional fallback.
#[test]
fn fields_without_xyz_is_rejected_binary() {
    // CONTROL: the same three records with x, y, z declared.
    let control = write_temp(
        "no_xyz_control_bin",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity ring time\n\
          SIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\n\
          WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut control_bytes = std::fs::read(&control).expect("read control");
    control_bytes.extend(f32le(&[1.0, 2.0, 3.0, 10.0, 0.0, 0.5]));
    control_bytes.extend(f32le(&[4.0, 5.0, 6.0, 20.0, 1.0, 0.5]));
    std::fs::write(&control, &control_bytes).expect("append control body");
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control must parse to two points");
    assert_eq!(cloud.points[0], Point3::new(1.0, 2.0, 3.0));

    // The defect: the header does not declare x, y or z at all.
    let path = write_temp(
        "no_xyz_bin",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS intensity ring time\nSIZE 4 4 4\n\
          TYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut bytes = std::fs::read(&path).expect("read");
    bytes.extend(f32le(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]));
    std::fs::write(&path, &bytes).expect("append body");
    let err = read_file_err(&path);
    assert!(
        err.contains('x') && err.contains('y') && err.contains('z'),
        "the error should name the missing coordinate fields, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Defect 5: unsupported (TYPE, SIZE) pairs decode as 0.0
// ---------------------------------------------------------------------------

/// `read_field_as_f32` ended in `_ => 0.0`, so any (TYPE, SIZE) pair outside
/// the nine decodable ones silently produced a zero. `SIZE 16 4 4` (x declared
/// as a 16-byte float) decoded x as 0.0 while y and z decoded correctly, so the
/// point looked like real data sitting on the x axis - no error, no warning,
/// nothing to distinguish it from a scan that really was at the origin.
#[test]
fn unsupported_type_size_pair_is_an_error() {
    // CONTROL: SIZE 4 4 4 is decodable and the same records parse exactly.
    let control = write_temp(
        "type_size_control",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut control_bytes = std::fs::read(&control).expect("read control");
    control_bytes.extend(f32le(&[1.5, 2.5, 3.5, -4.5, 0.25, 100.0]));
    std::fs::write(&control, &control_bytes).expect("append control body");
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control must parse to two points");
    assert_eq!(cloud.points[0], Point3::new(1.5, 2.5, 3.5));
    assert_eq!(cloud.points[1], Point3::new(-4.5, 0.25, 100.0));

    // The defect: x is declared as a 16-byte field. The stride is 16+4+4 = 24,
    // so the body below is 48 bytes of well-formed records - only the declared
    // type of x is undecodable.
    let path = write_temp(
        "type_size_16",
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 16 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
          WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n",
    );
    let mut bytes = std::fs::read(&path).expect("read");
    bytes.extend(f32le(&[1.5, 0.0, 0.0, 0.0, 2.5, 3.5]));
    bytes.extend(f32le(&[-4.5, 0.0, 0.0, 0.0, 0.25, 100.0]));
    std::fs::write(&path, &bytes).expect("append body");
    let err = read_file_err(&path);
    assert!(
        err.contains("unsupported") && err.contains("16"),
        "the error should name the unsupported (TYPE, SIZE) pair, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Defect 6: the writers index `normals[i]` / `colors[i]` unguarded
// ---------------------------------------------------------------------------

/// `write_pcd` (and its two siblings) indexed the attribute vectors with no
/// length check, so a cloud whose `colors`/`normals` are shorter than its
/// `points` panicked mid-write - after the header had already gone to the
/// caller's writer. Such a cloud is exactly what the reader produced for
/// defect 2, so the round trip panicked on its own output.
#[test]
fn write_pcd_on_short_colors_does_not_panic() {
    // CONTROL: a well-formed cloud writes and reads back identically.
    let control_cloud =
        PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
    let control = write_temp("writer_control", b"");
    {
        let mut w = BufWriter::new(File::create(&control).expect("create"));
        write_pcd(&mut w, &control_cloud).expect("control write");
        w.flush().expect("flush");
    }
    let cloud = read_file(&control);
    assert_eq!(cloud.len(), 2, "control round-trips through the file");
    assert_eq!(cloud.points[1], Point3::new(4.0, 5.0, 6.0));

    // The defect: `colors` is `Some` but empty, `normals` is one entry short.
    let mut broken = PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
    broken.colors = Some(Vec::new());
    broken.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0)]);

    // Before the fix: "index out of bounds: the len is 0 but the index is 0".
    let mut out = Vec::new();
    write_pcd(&mut out, &broken).expect("write_pcd must not panic on short attributes");
    let mut bin = Vec::new();
    write_pcd_binary(&mut bin, &broken).expect("write_pcd_binary must not panic either");
    let mut comp = Vec::new();
    write_pcd_binary_compressed(&mut comp, &broken)
        .expect("and neither must the compressed writer");

    // The file the writer produced must still be a parseable PCD of its two
    // points, and must not claim attributes the body does not carry.
    let path = write_temp("writer_short_attrs", &out);
    let cloud = read_file(&path);
    assert_eq!(cloud.len(), 2, "the points are still written");
    assert_eq!(cloud.points[0], Point3::new(1.0, 2.0, 3.0));
    assert_attributes_cover_points(&cloud);
}
