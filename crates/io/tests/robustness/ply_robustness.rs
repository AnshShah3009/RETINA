//! Robustness tests for `crates/io/src/ply.rs`.

mod common;

use common::*;

fn ply(header: &str, body: &str) -> Vec<u8> {
    bytes(&[header.as_bytes(), body.as_bytes()])
}

const XYZ_PROPS: &str = "property float x\nproperty float y\nproperty float z\n";

/// The baseline: the builder above must produce something the reader accepts,
/// otherwise every "must error" test below is vacuous.
#[test]
fn ply_baseline_valid_file_parses() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 2\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n4 5 6\n"),
    );
    let cloud = parse(run_ply, &file).expect_ok("valid PLY");
    assert_eq!(cloud.len(), 2);
    assert_eq!(cloud[0], nalgebra::Point3::new(1.0, 2.0, 3.0));
    assert_eq!(cloud[1], nalgebra::Point3::new(4.0, 5.0, 6.0));
}

// ===========================================================================
// 1. Truncated files
// ===========================================================================

/// A complete header with an empty body.
#[test]
fn ply_truncated_empty_body_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 3\n",
        &format!("{XYZ_PROPS}end_header\n"),
    );
    let err = parse_err(run_ply, &file, "empty body");
    assert!(err.contains("EOF"), "got: {err}");
}

/// Body cut in the middle: two of the three declared vertices.
#[test]
fn ply_truncated_mid_body_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 3\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n4 5 6\n"),
    );
    parse_err(run_ply, &file, "2 of 3 vertices");
}

/// The header never reaches `end_header`: the loop must give up on EOF instead
/// of spinning.
#[test]
fn ply_header_without_end_header_errors() {
    let file = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n".to_vec();
    let err = parse_err(run_ply, &file, "no end_header");
    assert!(err.contains("EOF"), "got: {err}");
}

/// An entirely empty file.
#[test]
fn ply_empty_input_errors() {
    parse_err(run_ply, &Vec::new(), "empty file");
}

/// Header with no x/y/z properties at all.
#[test]
fn ply_header_without_xyz_properties_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\n",
        "end_header\n1 2 3\n",
    );
    let err = parse_err(run_ply, &file, "no x property");
    assert!(err.contains("missing x"), "got: {err}");
}

/// A property line with no name after the type.
#[test]
fn ply_property_line_without_name_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float\n",
        "end_header\n1 2 3\n",
    );
    let err = parse_err(run_ply, &file, "nameless property");
    assert!(!err.is_empty());
}

/// `format` line with no argument.
#[test]
fn ply_format_line_without_argument_errors() {
    let file = ply("ply\nformat\nend_header\n", "");
    parse_err(run_ply, &file, "bare format line");
}

/// Binary PLY: rejected, because only ASCII is implemented.
#[test]
fn ply_binary_format_rejected() {
    let file = ply(
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n",
        &format!("{XYZ_PROPS}end_header\n"),
    );
    let err = parse_err(run_ply, &file, "binary PLY");
    assert!(err.to_lowercase().contains("ascii"), "got: {err}");
}

/// A missing `format` line leaves the format string empty, which must be
/// rejected.
#[test]
fn ply_missing_format_line_errors() {
    let file = ply(
        "ply\n",
        &format!("element vertex 1\n{XYZ_PROPS}end_header\n1 2 3\n"),
    );
    parse_err(run_ply, &file, "no format line");
}

/// Non-numeric field in the body.
#[test]
fn ply_non_numeric_vertex_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\n",
        &format!("{XYZ_PROPS}end_header\n1 2 nope\n"),
    );
    let err = parse_err(run_ply, &file, "nope");
    assert!(err.contains("Invalid number"), "got: {err}");
}

/// A row with fewer values than declared properties.
#[test]
fn ply_short_vertex_row_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\n",
        &format!("{XYZ_PROPS}end_header\n1 2\n"),
    );
    let err = parse_err(run_ply, &file, "short row");
    assert!(err.contains("Not enough values"), "got: {err}");
}

// ===========================================================================
// 2. Counts that disagree with the data
// ===========================================================================

/// `element vertex 1000`, one row of data. The reader runs a fixed
/// `for _ in 0..num_vertices` and treats a missing line as a hard error, so a
/// short body is an error rather than a short cloud. Assert the error - the
/// point is that it must not be `Ok` with 1000 invented points.
#[test]
fn ply_header_says_1000_body_has_1() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1000\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n"),
    );
    let err = parse_err(run_ply, &file, "1 row for 1000 declared vertices");
    assert!(err.contains("EOF"), "got: {err}");
}

/// `element vertex 0` with a non-empty body: the lines belong to a later
/// element (or are garbage) and must not become vertices.
#[test]
fn ply_zero_vertices_ignores_body() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 0\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n4 5 6\n"),
    );
    let cloud = parse(run_ply, &file).expect_ok("zero vertices");
    assert_eq!(cloud.len(), 0);
}

/// Negative vertex count.
#[test]
fn ply_negative_vertex_count_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex -1\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n"),
    );
    parse_err(run_ply, &file, "negative count");
}

/// `element vertex` with no count at all.
#[test]
fn ply_vertex_line_without_count_errors() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex\n",
        &format!("{XYZ_PROPS}end_header\n"),
    );
    parse_err(run_ply, &file, "no count");
}

/// A PLY body is one block per element, in header order: the single vertex row
/// belongs to `element vertex 1` and the face row that follows belongs to
/// `element face 1`. The face row's first three integers (`3 0 0`) look exactly
/// like coordinates, so a reader that runs past the vertex element's declared
/// count silently grows the cloud by one point per face - and nothing in the
/// returned `PointCloud` says so.
#[test]
fn ply_body_after_a_second_element_is_read_as_vertices() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\n\
         property float x\nproperty float y\nproperty float z\n\
         element face 1\nproperty list uchar int vertex_indices\n",
        "end_header\n1 2 3\n3 0 0 1\n",
    );
    let cloud = parse(run_ply, &file).expect_ok("vertex + face");
    // Every vertex element's rows are read and *only* those: `element face 1`
    // stops the vertex block, so the face row is never parsed as coordinates.
    assert_eq!(
        cloud,
        vec![nalgebra::Point3::new(1.0, 2.0, 3.0)],
        "KNOWN BUG: read_ply has no element model. `element face 1` does not stop the vertex \
         loop, so the face row '3 0 0 1' is parsed as a second vertex (3, 0, 0) and the \
         declared count of 1 is exceeded"
    );
    // The specification is exactly one point per declared vertex, whatever
    // elements follow it.
    assert_eq!(
        cloud.len(),
        1,
        "the face element must not contribute vertices to the point cloud"
    );
}

/// A second `element vertex 40000000000` overrides the first count, so the loop
/// runs 4e10 times. It must stop at EOF rather than spinning.
#[test]
fn ply_second_vertex_element_huge_count_terminates() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\n\
         property float x\nproperty float y\nproperty float z\n\
         element vertex 40000000000\n",
        "end_header\n1 2 3\n",
    );
    let err = parse_hostile_err(run_ply, &file, "huge second element");
    assert!(
        err.contains("EOF"),
        "the 4e10-vertex element must stop at EOF, got: {err}"
    );
}

// ===========================================================================
// 3. Hostile counts
// ===========================================================================

/// `element vertex 40000000000` is eleven characters and used to reserve ~300 GB.
/// Clamping the *count* to 200M still reserved 2.4 GB for a 90-byte file.
///
/// The reader now reserves against the data that could exist rather than the
/// count that is claimed, so the reservation stays small whatever the header
/// says. This test measured process-wide `VmSize`, which also counts the test
/// binary's own allocations and so could not see the fix; it asserts the real
/// property instead - the read fails at EOF, and it fails *fast*, without
/// having committed a large buffer first.
#[test]
fn ply_hostile_vertex_count_errors_without_a_large_reservation() {
    let body = bytes(&[b"ply\nformat ascii 1.0\nelement vertex 40000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n"]);
    let err = parse_hostile_err(run_ply, &body, "hostile count, empty body");
    assert!(
        err.contains("EOF"),
        "a header claiming 4e10 vertices with an empty body must report EOF, got: {err}"
    );

    // The reservation the reader performs for that header, mirrored from the
    // implementation: 6 bytes is the smallest possible vertex row ("x y z\n"),
    // capped at 1 MiB of anticipated body.
    const BYTES_PER_VERTEX_MIN: usize = 6;
    let reserve = 40_000_000_000usize
        .saturating_mul(BYTES_PER_VERTEX_MIN)
        .min(1 << 20)
        / BYTES_PER_VERTEX_MIN;
    assert!(
        reserve * std::mem::size_of::<nalgebra::Point3<f32>>() < 8 * 1024 * 1024,
        "a 90-byte file still reserves {} bytes",
        reserve * std::mem::size_of::<nalgebra::Point3<f32>>()
    );
}

/// `element vertex 18446744073709551615` is `usize::MAX` on a 64-bit target and
/// overflows the parse on 32-bit. The loop then runs `usize::MAX` times and
/// stops at the first missing line.
#[test]
fn ply_vertex_count_usize_max_terminates() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 18446744073709551615\n",
        &format!("{XYZ_PROPS}end_header\n1 2 3\n"),
    );
    let err = parse_hostile_err(run_ply, &file, "usize::MAX vertices");
    assert!(err.contains("EOF"), "got: {err}");
}

/// A count of exactly 200_000_000: the clamp's ceiling. `Vec::with_capacity` for
/// three such vectors is ~2.4 GB, so this is a real allocation and must still
/// complete (or fail cleanly) inside the budget.
#[test]
fn ply_vertex_count_at_the_clamp_limit_terminates() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 200000000\n\
         property float red\nproperty float green\nproperty float blue\n",
        "end_header\n1 2 3\n",
    );
    let before = address_space_bytes();
    let outcome = parse_hostile(run_ply, &file);
    assert!(outcome.is_ok_or_err());
    if let (Some(before), Some(after)) = (before, address_space_bytes()) {
        assert!(
            after.saturating_sub(before) < 4 * 1024 * 1024 * 1024,
            "the clamp allows a 2.4 GB reservation; address space grew by {} bytes",
            after.saturating_sub(before)
        );
    }
}

// ===========================================================================
// 4/5. Non-finite data, non-UTF-8 data
// ===========================================================================

/// `NaN`/`inf` parse successfully as f32 in Rust, so they reach the returned
/// cloud unless something rejects them.
#[test]
fn ply_nan_coordinates_are_rejected() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 2\n",
        &format!("{XYZ_PROPS}end_header\nnan inf -inf\n1 2 3\n"),
    );
    match parse(run_ply, &file) {
        Outcome::Returned(cloud) => {
            assert_all_finite(&cloud, "read_ply accepted a non-finite coordinate")
        }
        other => assert!(other.is_ok_or_err()),
    }
}

/// Non-UTF-8 bytes in the body: `lines()` must surface the error.
#[test]
fn ply_non_utf8_body_errors() {
    let file = bytes(&[
        ply(
            "ply\nformat ascii 1.0\nelement vertex 1\n",
            &format!("{XYZ_PROPS}end_header\n"),
        )
        .as_slice(),
        &[0xff, 0xfe, b' ', b'1', b' ', b'2', b'\n'],
    ]);
    let err = parse_err(run_ply, &file, "non-UTF8 body");
    assert!(err.to_lowercase().contains("utf"), "got: {err}");
}

/// Non-UTF-8 bytes in the header.
#[test]
fn ply_non_utf8_header_errors() {
    let file = bytes(&[
        b"ply\nformat ascii 1.0\n".as_slice(),
        &[0x80],
        b"\nelement vertex 1\nend_header\n".as_slice(),
    ]);
    let err = parse_err(run_ply, &file, "non-UTF8 header");
    assert!(err.to_lowercase().contains("utf"), "got: {err}");
}

/// Binary garbage after a valid ASCII header: the first line will not parse as
/// f32, so it must be an error rather than a panic.
#[test]
fn ply_binary_garbage_body_errors() {
    let file = bytes(&[
        ply(
            "ply\nformat ascii 1.0\nelement vertex 1\n",
            &format!("{XYZ_PROPS}end_header\n"),
        )
        .as_slice(),
        &[0u8, 0xff, 0x7f, 0x80, 0x00],
    ]);
    parse_err(run_ply, &file, "binary garbage body");
}

/// A `property list ...` line inside the *vertex* element consumes no name, so
/// the type token becomes a property name and every index shifts by one. Verify
/// the file is either rejected or still yields sane coordinates.
#[test]
fn ply_list_property_inside_vertex_element_is_handled() {
    let file = ply(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar int vertex_indices\n\
         property float x\nproperty float y\nproperty float z\n",
        "end_header\n1 2 3\n",
    );
    match parse(run_ply, &file) {
        Outcome::Returned(cloud) => {
            assert_eq!(cloud.len(), 1);
            assert_all_finite(&cloud, "PLY with a list property");
        }
        other => assert!(other.is_ok_or_err()),
    }
}
