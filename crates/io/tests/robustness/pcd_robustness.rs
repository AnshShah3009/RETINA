//! Robustness tests for `crates/io/src/pcd.rs`.
//!
//! Every input is constructed inline so the exact bytes that trigger a failure
//! are visible in the test. The categories covered are the ones that matter for
//! an untrusted-input parser: truncation, header/body count disagreement,
//! hostile counts, overflow in the size computation, zero strides, non-finite
//! data, and binary garbage where text is expected.
//!
//! ## Running the allocation tests
//!
//! Several tests here measure the process's address space (`/proc/self/status`,
//! `VmSize`) across a parse, which is only meaningful when one test runs at a
//! time. The `.cargo/config.toml` in this repo already pins `test-threads` for
//! nextest, and these tests are written to be run either way:
//!
//! ```text
//! cargo test -p cv-io --test parser_robustness                 # whole suite
//! cargo test -p cv-io --test parser_robustness pcd_robustness   # one module
//! ```
//!
//! If a `pcd_ascii_*` allocation test reports a surprising number of bytes, re-run
//! it on its own before believing it - the reservation is touched by concurrent
//! threads in the same process.

mod common;

use common::*;

// ===========================================================================
// 1. Truncated files: a valid header followed by nothing
// ===========================================================================

/// `DATA binary` and then EOF: `read_exact` must report the shortfall instead
/// of returning a cloud of zeroed points.
#[test]
fn pcd_binary_truncated_after_data_line_errors() {
    let file = pcd_binary("1", "", &[]);
    let err = parse_err(run_pcd, &file, "truncated body");
    assert!(
        err.to_lowercase().contains("failed to read"),
        "error should name the failed read, got: {err}"
    );
}

/// Header cut in the middle of the `DATA` line, with no newline at all.
#[test]
fn pcd_header_cut_before_data_line_errors() {
    let file = b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDAT".to_vec();
    let err = parse_err(run_pcd, &file, "cut header");
    assert!(!err.is_empty());
}

/// The `DATA binary` line is present but the `binary_compressed` size words are
/// missing.
#[test]
fn pcd_binary_compressed_truncated_size_header_errors() {
    let head = b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary_compressed\n";
    let err = parse_err(run_pcd, &bytes(&[head, &[0u8; 3]]), "truncated size header");
    assert!(err.contains("size header"), "got: {err}");
}

/// A half-written binary record (7 of 12 bytes).
#[test]
fn pcd_binary_truncated_mid_record_errors() {
    let file = pcd_binary("1", "", &f32s(&[1.0, 2.0]));
    let err = parse_err(run_pcd, &file, "half record");
    assert!(err.contains("12 bytes"), "error should name the byte count: {err}");
}

// ===========================================================================
// 2. Counts that disagree with the data
// ===========================================================================

/// The classic silent misparse: the header claims 1000 points, the body holds
/// 3.
///
/// `parse_pcd_binary` does `read_exact(&mut data)` for `stride * count` bytes
/// (pcd.rs:339) and therefore refuses a short body outright. That is the right
/// call for binary PCD - a partial record cannot be decoded - but it is worth
/// pinning down, because the alternative (zero-filling the tail) would hand back
/// 1000 points of which 997 are fabricated.
#[test]
fn pcd_binary_header_says_1000_body_has_3() {
    let body = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let file = pcd_binary("1000", "", &body);
    let err = parse_err(run_pcd, &file, "3 points of body for a 1000-point header");
    assert!(
        err.contains("12000") || err.contains("failed to read"),
        "the error should name the 12000 bytes it wanted, got: {err}"
    );
}

/// The same disagreement in ASCII, where the reader is supposed to stop at the
/// header count. The body has *more* rows than `POINTS` claims; the extra row
/// must be ignored, not appended.
#[test]
fn pcd_ascii_body_longer_than_declared_is_not_appended() {
    let file = pcd_ascii("2", "POINTS 2\n", "1 2 3\n4 5 6\n7 8 9\n");
    let cloud = parse(run_pcd, &file).expect_ok("ascii");
    assert_eq!(cloud.len(), 2, "extra ASCII rows beyond POINTS must be ignored");
    assert_eq!(cloud[1], nalgebra::Point3::new(4.0, 5.0, 6.0));
}

/// ASCII with fewer rows than `POINTS` claims: a short cloud is fine, a padded
/// one is not.
#[test]
fn pcd_ascii_fewer_rows_than_declared_is_not_padded() {
    let file = pcd_ascii("1000", "POINTS 1000\n", "1.0 2.0 3.0\n");
    let cloud = parse(run_pcd, &file).expect_ok("ascii");
    assert_eq!(cloud.len(), 1, "missing ASCII rows must not become (0,0,0) points");
}

/// Same shape, but written `1 2 3`. A fix that *rejects* rows with fewer than
/// three columns outright would be too strict, because the row is valid data;
/// the only requirement is that the reader never invents points.
#[test]
fn pcd_ascii_fewer_rows_than_declared_does_not_panic() {
    let file = pcd_ascii("1000", "POINTS 1000\n", "1 2 3\n");
    let cloud = parse(run_pcd, &file).expect_ok("ascii");
    assert!(cloud.len() <= 1, "one row of data must yield at most one point");
}

/// `WIDTH`/`HEIGHT` disagree with `POINTS`; `POINTS` is authoritative, so the
/// cloud must have `POINTS` points.
#[test]
fn pcd_points_overrides_width_times_height() {
    let file = pcd_binary("2", "POINTS 2\n", &f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let cloud = parse(run_pcd, &file).expect_ok("points");
    assert_eq!(cloud.len(), 2);
}

// ===========================================================================
// 3. Hostile counts - must not allocate gigabytes or loop forever
// ===========================================================================

/// `POINTS 4000000000` with a 12-byte stride is 48 GB on the strength of one
/// header line. The reader has a byte cap, so this must be refused.
#[test]
fn pcd_binary_hostile_points_is_rejected_promptly() {
    let file = pcd_binary("4000000000", "", &[]);
    let err = parse_hostile_err(run_pcd, &file, "4e9 points x 12 bytes");
    assert!(err.contains("refusing"), "got: {err}");
}

/// The plain-binary path's cap holds, but the *ASCII* path clamps the
/// reservation to 200M points - 2.4 GB - and commits it before looking at the
/// body, so a 90-byte file with `POINTS 40000000000` costs 2.4 GB of memory to
/// discover that it is empty. Not a crash, but a cheap amplification.
#[test]
fn pcd_ascii_hostile_points_reserves_2_4gb_for_an_empty_body() {
    let file = pcd_ascii("40000000000", "POINTS 40000000000\n", "");
    let before = address_space_bytes();
    let cloud = parse_hostile(run_pcd, &file).expect_ok("empty ascii body");
    assert_eq!(cloud.len(), 0, "no data lines means no points");
    let grew = address_space_bytes()
        .zip(before)
        .map(|(after, before)| after.saturating_sub(before))
        .unwrap_or(0);
    assert!(
        grew < 64 * 1024 * 1024,
        "KNOWN BUG: `POINTS 40000000000` on an empty ASCII body grew the address space by \
         {grew} bytes (the reservation is clamped to 200M points, still 2.4 GB); a 90-byte file \
         should not cost gigabytes"
    );
}

/// `POINTS` = `usize::MAX` overflows `stride * count`; the `checked_mul` guard
/// must turn that into an error rather than a wrapped, too-small buffer.
#[test]
fn pcd_binary_overflowing_points_is_rejected() {
    let file = pcd_binary("18446744073709551615", "", &[]);
    parse_err(run_pcd, &file, "usize::MAX points");
}

/// A count large enough to fit in `usize` and to pass the byte cap only because
/// the stride is small - then the `vec![0u8; total]` and the `read_exact` are
/// reached, but the call must still return.
#[test]
fn pcd_binary_small_stride_huge_count_does_not_hang() {
    // A single 4-byte field keeps `stride * count` under the 8 GiB cap.
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x\nSIZE 4\nTYPE F\nCOUNT 1\nWIDTH 1000000000\nHEIGHT 1\nPOINTS 1000000000\nDATA binary\n";
    let file = head.as_bytes().to_vec();
    match parse_hostile(run_pcd, &file) {
        Outcome::Error(_) => {}
        Outcome::Returned(cloud) => panic!(
            "returned Ok with {} points for a 1e9-point claim with an empty body",
            cloud.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG: a 1e9-point claim with an empty body did not return"),
    }
}

/// The address-space probe: whatever the outcome, a hostile header must not
/// commit a 48 GB virtual allocation. (Belt and braces for the byte cap.)
#[test]
fn pcd_hostile_points_does_not_commit_gigabytes() {
    let before = address_space_bytes();
    let _ = parse_hostile(run_pcd, &pcd_binary("4000000000", "", &[]));
    if let (Some(before), Some(after)) = (before, address_space_bytes()) {
        let grew = after.saturating_sub(before);
        assert!(
            grew < 1024 * 1024 * 1024,
            "address space grew by {} bytes for a hostile POINTS header",
            grew
        );
    }
}

// ===========================================================================
// 4. Arithmetic overflow in the size / offset computation
// ===========================================================================

/// `WIDTH`/`HEIGHT` are multiplied at `pcd.rs:142` with no overflow check:
/// `points_count = width * height`. Both operands are file-controlled, the
/// product feeds a `vec![0u8; total]`, and a `POINTS` line is optional, so a
/// header of
///
/// ```text
/// WIDTH 18446744073709551615
/// HEIGHT 2
/// DATA binary
/// ```
///
/// is enough.
///
/// Consequences, both verified by the standalone reproducer
/// `scratchpad/pcd_width_height_overflow_repro.rs`:
///
/// * overflow checks on (debug, and any profile with `overflow-checks = true`):
///   panic "attempt to multiply with overflow" from inside the parser;
/// * overflow checks off (release): the product wraps to `2^64 - 2`, `stride *
///   count` wraps again to `2^64 - 1`, `checked_mul` at pcd.rs:297 reports no
///   overflow, the 8 GiB cap rejects it - so the wrapped value happens to be
///   caught *here*, by luck of the byte cap rather than by a guard.
///
/// The unguarded multiply itself is the defect: it is a panic on one line of
/// input in a library that promises errors, and the safety of the surrounding
/// code depends on an unrelated 8 GiB cap. A 32-bit target wraps far sooner and
/// is not saved by the cap.
#[test]
fn pcd_width_times_height_multiply_is_unguarded() {
    let source = include_str!("../../src/pcd.rs");
    let line_no = source
        .lines()
        .position(|l| l.contains("points_count = width * height"))
        .expect("the width*height expression moved; re-check this test");
    let line = source.lines().nth(line_no).unwrap();
    assert!(
        !line.contains("checked_mul")
            && !line.contains("saturating_mul")
            && !line.contains("wrapping_mul"),
        "pcd.rs:{} looked unguarded before but is now: {line:?} - re-evaluate this test",
        line_no + 1
    );
}

// ===========================================================================
// 5. Division by zero / zero-size records
// ===========================================================================

/// `SIZE 0 0 0` gives a zero stride. The binary reader has an explicit guard;
/// check it, and check the ASCII/compressed paths do not divide by it either.
#[test]
fn pcd_binary_zero_point_stride_errors() {
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 0 0 0\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n";
    let err = parse_err(run_pcd, &head.as_bytes().to_vec(), "zero stride");
    assert!(err.contains("stride"), "got: {err}");
}

/// `SIZE 0` for a single-field file.
#[test]
fn pcd_binary_single_zero_size_field_errors() {
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x\nSIZE 0\nTYPE F\nCOUNT 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n";
    parse_err(run_pcd, &head.as_bytes().to_vec(), "zero-size single field");
}

/// `FIELDS` with no `SIZE` line at all falls back to 4 bytes/field - verify the
/// default path is a clean short-read error, not a panic.
#[test]
fn pcd_binary_missing_size_line_uses_default_stride() {
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nTYPE F F F\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n";
    let err = parse_err(run_pcd, &head.as_bytes().to_vec(), "no SIZE line, empty body");
    assert!(err.contains("12 bytes"), "got: {err}");
}

// ===========================================================================
// 6. Infinite loops
// ===========================================================================

/// The header loop at `pcd.rs:80` exits only on a `DATA` line. A file that never
/// contains one must fail on EOF.
#[test]
fn pcd_header_without_data_line_errors() {
    let file = b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 4\nHEIGHT 4\n".to_vec();
    let err = parse_err(run_pcd, &file, "no DATA line");
    assert!(err.contains("EOF"), "got: {err}");
}

/// A single 1 MiB line with no `DATA` keyword: the loop must not spin.
#[test]
fn pcd_giant_unterminated_header_line_terminates() {
    let mut file = b"# .PCD v0.7\nVERSION 0.7\nVERSION ".to_vec();
    file.extend(std::iter::repeat(b'A').take(1 << 20));
    file.push(b'\n');
    let err = parse_err(run_pcd, &file, "1 MiB header line with no DATA");
    assert!(err.contains("EOF"), "got: {err}");
}

// ===========================================================================
// 7. Garbage where text is expected, and vice versa
// ===========================================================================

/// A header that is valid UTF-8 but contains non-numeric junk in the body.
///
/// This is a real silent misparse: the ASCII reader maps every token through
/// `unwrap_or(0.0)` (pcd.rs:223), so the three columns of `hello world there`
/// become the point `(0, 0, 0)` and a one-point cloud is returned.
#[test]
fn pcd_ascii_non_numeric_field_is_rejected() {
    let file = pcd_ascii("1", "POINTS 1\n", "hello world there\n");
    match parse(run_pcd, &file) {
        Outcome::Returned(cloud) => panic!(
            "KNOWN BUG: a non-numeric ASCII row is coerced to 0.0 per column \
             (pcd.rs `s.parse().unwrap_or(0.0)`), so read_pcd returned Ok with {} point(s) = {:?}",
            cloud.len(),
            cloud
        ),
        other => assert!(other.is_ok_or_err()),
    }
}

/// Non-UTF-8 bytes in a text PCD body: `read_line` into a `String` must be an
/// error, not a panic.
#[test]
fn pcd_non_utf8_in_ascii_body_errors() {
    let mut file = pcd_ascii("1", "POINTS 1\n", "");
    file.extend_from_slice(&[0xff, 0xfe, b' ', b'1', b' ', b'2', b'\n']);
    let err = parse_err(run_pcd, &file, "non-UTF8 in an ASCII PCD body");
    assert!(err.to_lowercase().contains("utf"), "got: {err}");
}

/// Non-UTF-8 bytes in the *header*.
#[test]
fn pcd_non_utf8_in_header_errors() {
    let file = bytes(&[
        b"# .PCD v0.7\nVERSION 0.7\n".as_slice(),
        &[0xff],
        b"\nFIELDS x y z\nDATA ascii\n".as_slice(),
    ]);
    let err = parse_err(run_pcd, &file, "non-UTF8 in a PCD header");
    assert!(err.to_lowercase().contains("utf"), "got: {err}");
}

/// Binary garbage where a text PCD was expected.
#[test]
fn pcd_binary_garbage_in_ascii_body_is_rejected() {
    let file = bytes(&[
        pcd_ascii("1", "POINTS 1\n", "").as_slice(),
        &[0u8, 0xff, 0x7f, 0x80, b'\n'],
    ]);
    match parse(run_pcd, &file) {
        Outcome::Returned(cloud) => assert_all_finite(&cloud, "binary garbage in ASCII PCD"),
        other => assert!(other.is_ok_or_err()),
    }
}

// ===========================================================================
// 8. Non-finite data
// ===========================================================================

/// A NaN/Inf coordinate must either be rejected or flagged - it must not be
/// handed back inside a successful `PointCloud`, where every downstream
/// consumer silently produces garbage.
#[test]
fn pcd_binary_nan_coordinates_are_rejected() {
    let file = pcd_binary("1", "", &f32s(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]));
    match parse(run_pcd, &file) {
        Outcome::Returned(cloud) => assert_all_finite(&cloud, "read_pcd accepted a NaN coordinate"),
        other => assert!(other.is_ok_or_err()),
    }
}

/// The same for the ASCII path. `NaN`, `inf` and `1e40` (which overflows to
/// `inf` in f32) all parse successfully in Rust, so these rows are accepted by
/// `f32::from_str` and reach the cloud.
#[test]
fn pcd_ascii_nan_coordinates_are_rejected() {
    let file = pcd_ascii("2", "POINTS 2\n", "nan inf -inf\n1e40 0 0\n");
    match parse(run_pcd, &file) {
        Outcome::Returned(cloud) => {
            assert_all_finite(&cloud, "read_pcd (ascii) accepted a non-finite coordinate")
        }
        other => assert!(other.is_ok_or_err()),
    }
}

// ===========================================================================
// 9. binary_compressed: the LZF path
// ===========================================================================

/// `compressed_size` = 0xFFFF_FFFF is a file-controlled 4 GiB allocation at
/// `pcd.rs:475` (`vec![0u8; compressed_size]`), guarded only by how much data the
/// file actually contains. A ~120-byte file commits 4 GB of memory before the
/// `read_exact` fails, and the plain-binary path's 8 GiB byte cap does not
/// cover this buffer.
#[test]
fn pcd_binary_compressed_hostile_compressed_size_allocates_4gb() {
    let before = address_space_bytes();
    let file = pcd_binary_compressed("1", &[], 0xFFFF_FFFF);
    match parse_hostile(run_pcd, &file) {
        Outcome::Returned(_) => panic!("returned Ok for compressed_size = 0xFFFFFFFF"),
        other => assert!(other.is_ok_or_err()),
    }
    let grew = address_space_bytes()
        .zip(before)
        .map(|(after, before)| after.saturating_sub(before))
        .unwrap_or(0);
    assert!(
        grew < 512 * 1024 * 1024,
        "KNOWN BUG: compressed_size = 0xFFFFFFFF grew the address space by {grew} bytes; \
         pcd.rs allocates the compressed buffer straight from the file-controlled 4-byte field \
         with no cap"
    );
}

/// A valid literal-run LZF stream that decompresses to 12 bytes, with the
/// header claiming 10 points: the de-interleaver must refuse the mismatch
/// rather than reading past the buffer.
#[test]
fn pcd_binary_compressed_decompressed_too_small_errors() {
    // literal run of 12 bytes
    let compressed = [11u8, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3];
    let file = pcd_binary_compressed("10", &compressed, 12);
    match parse(run_pcd, &file) {
        Outcome::Returned(cloud) => {
            assert_eq!(cloud.len(), 0, "no decompressed data means no points")
        }
        other => assert!(other.is_ok_or_err()),
    }
}

/// Back-reference that points before the start of the output: LZF must reject
/// it rather than indexing out of bounds.
#[test]
fn pcd_binary_compressed_bad_backreference_errors() {
    let compressed = [0xffu8, 0xff, 0x00];
    let file = pcd_binary_compressed("1", &compressed, 12);
    parse_err(run_pcd, &file, "bad back-reference");
}

/// A literal run that runs off the end of the compressed input.
#[test]
fn pcd_binary_compressed_truncated_literal_run_errors() {
    // ctrl = 31 => literal run of 32 bytes, but only 2 follow.
    let compressed = [31u8, 0xaa, 0xbb];
    let file = pcd_binary_compressed("1", &compressed, 32);
    parse_err(run_pcd, &file, "truncated literal run");
}

/// `SIZE` shorter than `FIELDS`, which previously made the stride and the
/// per-field offsets disagree.
#[test]
fn pcd_binary_size_list_shorter_than_fields() {
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n";
    let file = bytes(&[head.as_bytes(), &f32s(&[1.0, 2.0, 3.0]), &0u32.to_le_bytes()]);
    let cloud = parse(run_pcd, &file).expect_ok("SIZE shorter than FIELDS");
    assert_eq!(cloud.len(), 1);
    assert_eq!(cloud[0], nalgebra::Point3::new(1.0, 2.0, 3.0));
}

/// `FIELDS` containing an entry with a huge `COUNT`: the stride computation must
/// saturate rather than overflow, and the read must fail cleanly.
#[test]
fn pcd_binary_huge_field_count_does_not_overflow_stride() {
    let head = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z pad\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 4000000000\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n";
    let file = head.as_bytes().to_vec();
    match parse_hostile(run_pcd, &file) {
        Outcome::Returned(cloud) => panic!(
            "returned {} points for a header whose stride computation cannot be satisfied",
            cloud.len()
        ),
        other => assert!(other.is_ok_or_err()),
    }
}
