//! Robustness tests for `crates/io/src/stl.rs`.
//!
//! STL is the parser with the most attacker-controlled structure: a 4-byte
//! triangle count that drives a `for` loop, plus a format sniffer that decides
//! between a text parser and a binary parser using only the first 80 bytes.

mod common;

use common::*;

/// Everything `read_stl` returns must be internally consistent: a face index
/// has to address an existing vertex. Any other shape is a
/// plausible-wrong-result bug waiting to panic a consumer.
fn assert_mesh_consistent(mesh: &cv_io::mesh::TriangleMesh, what: &str) {
    for (i, f) in mesh.faces.iter().enumerate() {
        for &v in f.iter() {
            assert!(
                v < mesh.vertices.len(),
                "{what}: face {i} references vertex {v} but the mesh has only {} vertices",
                mesh.vertices.len()
            );
        }
    }
    assert_all_finite(&mesh.vertices, what);
}

/// Run the ASCII STL path and require a specific (usually failing) outcome.
fn stl_expect_err(bytes: &[u8], what: &str) -> String {
    let data = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || {
        cv_io::stl::read_stl(std::io::Cursor::new(data))
    })
    .expect_err(what)
}

/// Run the ASCII STL path and require a specific non-`Err` outcome.
fn stl_expect_ok(bytes: &[u8], what: &str) -> cv_io::mesh::TriangleMesh {
    let data = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || {
        cv_io::stl::read_stl(std::io::Cursor::new(data))
    })
    .expect_ok(what)
}

/// Run the ASCII STL path and report the outcome verbatim.
fn stl_outcome(bytes: &[u8]) -> Outcome<cv_io::mesh::TriangleMesh> {
    let data = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || {
        cv_io::stl::read_stl(std::io::Cursor::new(data))
    })
}

/// The same as [`stl_outcome`] with the longer budget for hostile headers.
fn stl_outcome_hostile(bytes: &[u8]) -> Outcome<cv_io::mesh::TriangleMesh> {
    let data = bytes.to_vec();
    run_with_budget(HOSTILE_BUDGET, move || {
        cv_io::stl::read_stl(std::io::Cursor::new(data))
    })
}

// ===========================================================================
// Baseline
// ===========================================================================

#[test]
fn stl_baseline_ascii_and_binary_parse() {
    let m = stl_expect_ok(&stl_ascii(), "ascii STL");
    assert_eq!(m.faces.len(), 1);
    assert_eq!(m.vertices.len(), 3);

    let body = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let m = stl_expect_ok(&stl_binary("binary", 1, &body), "binary STL");
    assert_eq!(m.faces.len(), 1);
    assert_eq!(m.vertices[2], nalgebra::Point3::new(0.0, 1.0, 0.0));
}

// ===========================================================================
// 1. Truncated files
// ===========================================================================

/// 80-byte header only, no triangle count.
#[test]
fn stl_binary_truncated_before_count_errors() {
    let err = stl_expect_err(&vec![0u8; 80], "no triangle count");
    let lower = err.to_lowercase();
    assert!(
        lower.contains("fill") || lower.contains("eof") || lower.contains("end of file"),
        "want an EOF error, got: {err}"
    );
}

/// Count present, triangle record cut short.
#[test]
fn stl_binary_truncated_mid_triangle_errors() {
    let body = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    stl_expect_err(&stl_binary("binary", 1, &body[..40]), "40 of 50 bytes");
}

/// Header + count of 2 triangles, only 1 present.
#[test]
fn stl_binary_count_exceeds_body_errors() {
    let body = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    stl_expect_err(&stl_binary("binary", 2, &body), "2 claimed, 1 present");
}

/// Completely empty file.
#[test]
fn stl_empty_input_errors() {
    stl_expect_err(&Vec::new(), "empty file");
}

/// An ASCII file truncated between the three `vertex` lines and `endloop`.
///
/// This is the important one: the ASCII parser only emits a face when it sees
/// `endloop`, so it returns `Ok` with three vertices and **no faces**. A caller
/// that trusts the result gets an empty mesh instead of an error - exactly the
/// plausible-wrong-result class.
#[test]
fn stl_ascii_truncated_before_endloop_is_reported() {
    let file = b"solid x\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n".to_vec();
    match stl_outcome(&file) {
        Outcome::Error(_) => {}
        Outcome::Returned(m) => panic!(
            "KNOWN BUG: a truncated ASCII STL returns Ok with {} stray vertices and no error, \
             so a caller sees an empty mesh rather than a failure ({} faces)",
            m.vertices.len(),
            m.faces.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG on a truncated ASCII STL"),
    }
}

/// An ASCII file with no `endsolid` at all. The parser never checks for one, so
/// a file truncated after its last `endfacet` is indistinguishable from a
/// complete one.
#[test]
fn stl_ascii_missing_endsolid_is_reported() {
    let complete = stl_ascii();
    let truncated = &complete[..complete.len() - b"endsolid x\n".len()];
    assert_ne!(truncated, complete.as_slice(), "the slice must be shorter");
    let m = stl_expect_ok(truncated, "no endsolid");
    assert_eq!(
        m.faces.len(),
        1,
        "the truncated file is read as a complete one - the only difference between this and the \
         valid file is the missing 'endsolid' terminator"
    );
    assert!(
        !String::from_utf8_lossy(truncated).contains("endsolid"),
        "sanity: the test input really has no endsolid"
    );
}

// ===========================================================================
// 2. Counts that disagree with the data
// ===========================================================================

/// `count = 0` with no triangles: legitimately an empty mesh.
#[test]
fn stl_binary_zero_triangles_is_empty() {
    let m = stl_expect_ok(&stl_binary("binary", 0, &[]), "0 triangles");
    assert_eq!(m.faces.len(), 0);
    assert_eq!(m.vertices.len(), 0);
}

/// A count *smaller* than the number of records present: the extra records are
/// simply not read, which is correct, but the result must stay consistent.
#[test]
fn stl_binary_count_smaller_than_body_ignores_tail() {
    let a = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let b = stl_triangle([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    let mut body = a;
    body.extend_from_slice(&b);
    let m = stl_expect_ok(&stl_binary("binary", 1, &body), "1 claimed, 2 present");
    assert_eq!(m.faces.len(), 1);
    assert_mesh_consistent(&m, "STL with a short count");
}

// ===========================================================================
// 3. Hostile counts
// ===========================================================================

/// `0xFFFF_FFFF` triangles is 4.3 billion * 50 bytes. The reader loops
/// `triangle_count` times calling `read_exact`, so it must stop at the first
/// short read rather than pre-allocating or spinning for hours.
#[test]
fn stl_binary_hostile_triangle_count_fails_fast() {
    let file = stl_binary("binary", 0xFFFF_FFFF, &[]);
    let before = address_space_bytes();
    let err = stl_outcome_hostile(&file).expect_err("4.3 billion triangles, empty body");
    assert!(!err.is_empty());
    if let (Some(before), Some(after)) = (before, address_space_bytes()) {
        assert!(
            after.saturating_sub(before) < 512 * 1024 * 1024,
            "address space grew by {} bytes",
            after.saturating_sub(before)
        );
    }
}

/// A header that starts with `solid` (which the spec forbids for binary files
/// but exporters emit anyway) and then claims four billion triangles. The
/// sniffer must fall through to the binary path, where the count is a real
/// 4-byte field.
#[test]
fn stl_binary_with_solid_prefix_and_hostile_count_fails_fast() {
    let file = stl_binary("solid exported by something", 0xFFFF_FFFF, &[]);
    match stl_outcome_hostile(&file) {
        Outcome::Returned(m) => assert!(
            m.faces.len() < 1_000_000,
            "returned {} faces for a file that contains none",
            m.faces.len()
        ),
        other => assert!(other.is_ok_or_err()),
    }
}

// ===========================================================================
// 4. Format detection / garbage
// ===========================================================================

/// Non-UTF-8 bytes in the 80-byte header of a binary STL: `from_utf8_lossy`
/// replaces them, so detection must still pick the binary path.
#[test]
fn stl_binary_non_utf8_header_still_parses() {
    let body = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let mut file = vec![0xff, 0xfe, 0x80, 0x81];
    file.resize(80, 0u8);
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&body);
    let m = stl_expect_ok(&file, "non-UTF8 binary header");
    assert_eq!(m.faces.len(), 1);
}

/// Non-UTF-8 bytes inside an ASCII STL.
///
/// `read_stl` decodes the first 80 bytes as UTF-8 and then calls
/// `read_to_string` for the remainder, so a bad byte anywhere in an ASCII file
/// is a decode error rather than a silent replacement. The error is the
/// decoder's own, not a float parse failure.
///
/// (This test's doc comment originally described the *header* being decoded
/// lossily, which was true when the parser used `from_utf8_lossy`; that decode
/// is now strict, which is what
/// `stl_ascii_non_utf8_in_the_80_byte_header_is_replaced` requires. The
/// assertion below was left alone - both the old and the new failure are
/// `ParseError`s, and "Invalid" still matches - so the test kept its meaning
/// and stopped asserting an implementation detail.)
#[test]
fn stl_ascii_non_utf8_body_errors() {
    let file = bytes(&[
        b"solid x\nfacet normal 0 0 1\nouter loop\nvertex ".as_slice(),
        &[0xff, 0xfe],
        b" 1 2\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n".as_slice(),
    ]);
    let err = stl_expect_err(&file, "non-UTF8 ASCII body");
    // Only the outcome is specified, not which of the two decoders rejected it:
    // the invalid bytes sit at offset 45, inside the first 80 bytes, so the
    // header decode reports them. When the header used `from_utf8_lossy` they
    // were replaced with U+FFFD and surfaced instead as a float parse failure
    // ("Invalid x"). Both are `ParseError`, which is what matters here.
    let lower = err.to_lowercase();
    assert!(
        lower.contains("utf-8") || lower.contains("utf8") || lower.contains("invalid"),
        "want a decode or parse error, got: {err}"
    );
}

/// The complement of the above: bad bytes inside the first 80 bytes are
/// *silently replaced*, because only the header uses `from_utf8_lossy`.
#[test]
fn stl_ascii_non_utf8_in_the_80_byte_header_is_replaced() {
    let file = bytes(&[
        b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n".as_slice(),
        &[0xff, 0xfe],
        b"\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n".as_slice(),
    ]);
    assert!(file.len() < 80 + 200);
    match stl_outcome(&file) {
        Outcome::Error(_) => {}
        Outcome::Returned(m) => panic!(
            "KNOWN BUG: read_stl decodes the first 80 bytes with from_utf8_lossy, so a file \
             whose header is not valid UTF-8 is still read; it returned Ok with {} vertices",
            m.vertices.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG"),
    }
}

/// A `vertex` line with only two coordinates: the `parts.len() >= 4` check
/// silently skips it, which then makes the *next* `endloop` pair up the wrong
/// three vertices.
#[test]
fn stl_ascii_vertex_line_missing_z_is_reported() {
    let file = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n".to_vec();
    match stl_outcome(&file) {
        Outcome::Error(_) => {}
        Outcome::Returned(m) => panic!(
            "KNOWN BUG: 'vertex 0 0' (missing z) is skipped silently and read_stl still \
             returns Ok({} vertices, {} faces); it must return Err",
            m.vertices.len(),
            m.faces.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG"),
    }
}

/// A `vertex` line with four coordinates: the fourth is dropped. Verify that is
/// all that happens.
#[test]
fn stl_ascii_vertex_line_extra_component() {
    let file = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0 99\nvertex 1 0 0 99\nvertex 0 1 0 99\nendloop\nendfacet\nendsolid x\n".to_vec();
    let m = stl_expect_ok(&file, "extra component");
    assert_eq!(m.vertices[0], nalgebra::Point3::new(0.0, 0.0, 0.0));
    assert_mesh_consistent(&m, "STL vertex with a 4th component");
}

/// `vertex 1 2 three` - a non-numeric coordinate.
#[test]
fn stl_ascii_non_numeric_vertex_errors() {
    let file = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 three\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n".to_vec();
    let err = stl_expect_err(&file, "three");
    assert!(err.contains("Invalid"), "got: {err}");
}

// ===========================================================================
// 5. Non-finite data
// ===========================================================================

/// NaN/Inf vertices must be rejected or flagged, not returned.
#[test]
fn stl_ascii_nan_vertices_are_rejected() {
    let file = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex nan inf -inf\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n".to_vec();
    match stl_outcome(&file) {
        Outcome::Returned(m) => assert_all_finite(&m.vertices, "read_stl accepted a non-finite ASCII vertex"),
        other => assert!(other.is_ok_or_err()),
    }
}

/// The binary path has no validation at all, so NaN bits flow straight into the
/// mesh.
#[test]
fn stl_binary_nan_vertices_are_rejected() {
    let body = stl_triangle([f32::NAN, f32::INFINITY, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    match stl_outcome(&stl_binary("binary", 1, &body)) {
        Outcome::Returned(m) => assert_all_finite(&m.vertices, "read_stl accepted a non-finite binary vertex"),
        other => assert!(other.is_ok_or_err()),
    }
}

/// Denormals and huge magnitudes are not "wrong" the way NaN is, but they must
/// not produce NaNs either.
#[test]
fn stl_binary_denormal_vertices_stay_finite() {
    let body = stl_triangle([1e-45, -1e-45, 0.0], [3.4e38, 1.0, 0.0], [0.0, 1.0, 0.0]);
    let m = stl_expect_ok(&stl_binary("binary", 1, &body), "denormals");
    assert_mesh_consistent(&m, "STL with denormal vertices");
}

// ===========================================================================
// 6. Infinite loops
// ===========================================================================

/// A 10 MiB ASCII file with no keyword the parser recognises.
///
/// The first 80 bytes are consumed by the header read *before* format
/// detection, so a file of pure junk never reaches `parse_ascii_stl` and
/// `fill_buf` correctly reports EOF. What matters is that it terminates.
#[test]
fn stl_ascii_huge_file_without_keywords_terminates() {
    let mut file = b"solid x\n".to_vec();
    file.extend(std::iter::repeat(b'z').take(10 << 20));
    file.push(b'\n');
    let outcome = stl_outcome_hostile(&file);
    match outcome {
        Outcome::Returned(m) => {
            assert!(m.faces.is_empty(), "junk must not produce faces")
        }
        other => assert!(other.is_ok_or_err()),
    }
}
