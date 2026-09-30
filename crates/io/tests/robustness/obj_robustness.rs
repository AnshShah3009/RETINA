//! Robustness tests for `crates/io/src/obj.rs`.
//!
//! `read_obj` only extracts vertices, so most malformed input is either ignored
//! or turned into a parse error. `ObjMesh::read` is the interesting one: it
//! converts faces to `TriangleMesh` indices, and that `i - 1` conversion is
//! where a file-controlled value becomes a memory-safety-relevant index.

mod common;

use common::*;
use cv_io::obj::ObjMesh;
use std::io::Cursor;

fn assert_mesh_consistent(mesh: &cv_io::mesh::TriangleMesh, what: &str) {
    for (i, f) in mesh.faces.iter().enumerate() {
        for &v in f.iter() {
            assert!(
                v < mesh.vertices.len(),
                "{what}: face {i} references vertex {v} but the mesh has only {} vertices \
                 (indices must be in range, or the mesh is a plausible-wrong-result)",
                mesh.vertices.len()
            );
        }
    }
    assert_all_finite(&mesh.vertices, what);
}

/// Parse an OBJ mesh, converting a panic or a hang into a reported failure.
fn obj_mesh(bytes: &[u8]) -> Outcome<cv_io::mesh::TriangleMesh> {
    let data = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || {
        ObjMesh::read(Cursor::new(data)).map(|m| m.to_triangle_mesh())
    })
}

/// The raw `ObjMesh` (before fan triangulation), for inspecting face indices.
fn obj_raw(bytes: &[u8]) -> Outcome<ObjMesh> {
    let data = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || ObjMesh::read(Cursor::new(data)))
}

// ===========================================================================
// Baseline
// ===========================================================================

#[test]
fn obj_baseline_parses() {
    let file = obj_ascii();
    let cloud = parse(run_obj, &file).expect_ok("valid OBJ");
    assert_eq!(cloud.len(), 3);

    let m = obj_mesh(&file).expect_ok("valid OBJ mesh");
    assert_eq!(m.faces.len(), 1);
    assert_mesh_consistent(&m, "baseline OBJ");
}

// ===========================================================================
// 1. Truncated / malformed text
// ===========================================================================

/// An empty file is a legitimately empty cloud, not an error.
#[test]
fn obj_empty_input_is_empty_cloud() {
    let cloud = parse(run_obj, &Vec::new()).expect_ok("empty OBJ");
    assert_eq!(cloud.len(), 0);
}

/// A `v` line cut off after two coordinates: the `parts.len() >= 4` guard skips
/// it silently, so the vertex simply disappears.
#[test]
fn obj_short_vertex_line_is_reported() {
    let file = b"v 1 2\nv 1 2 3\n".to_vec();
    match parse(run_obj, &file) {
        Outcome::Error(_) => {}
        Outcome::Returned(cloud) => panic!(
            "KNOWN BUG: 'v 1 2' (missing z) is dropped silently and read_obj returns Ok with \
             {} vertex/vertices; it must return Err",
            cloud.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG"),
    }
}

/// A non-numeric coordinate is an error.
#[test]
fn obj_non_numeric_vertex_errors() {
    let file = b"v one 2 3\n".to_vec();
    let err = parse_err(run_obj, &file, "one");
    assert!(err.contains("Invalid"), "got: {err}");
}

/// `v` with no space after the keyword: not a vertex line at all.
#[test]
fn obj_vertex_without_space_is_reported() {
    let file = b"v1.0 2.0 3.0\nv 1 2 3\n".to_vec();
    match parse(run_obj, &file) {
        Outcome::Error(_) => {}
        Outcome::Returned(cloud) => panic!(
            "KNOWN BUG: 'v1.0 2.0 3.0' is not recognised as a vertex (the parser requires a \
             space after 'v'), so read_obj silently returns {} of the 2 vertices present",
            cloud.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG"),
    }
}

/// Non-UTF-8 bytes anywhere in an OBJ: `lines()` must surface the error.
#[test]
fn obj_non_utf8_errors() {
    let file = bytes(&[b"v 1 2 3\n".as_slice(), &[0xff, 0xfe, b'\n']]);
    let err = parse_err(run_obj, &file, "non-UTF8");
    assert!(err.to_lowercase().contains("utf"), "got: {err}");
}

/// Non-finite coordinates.
#[test]
fn obj_nan_vertices_are_rejected() {
    let file = b"v nan inf -inf\nv 1e40 0 0\n".to_vec();
    match parse(run_obj, &file) {
        Outcome::Returned(cloud) => assert_all_finite(&cloud, "read_obj accepted a non-finite coordinate"),
        other => assert!(other.is_ok_or_err()),
    }
}

/// ~2.5 MiB of lines that are neither comments nor vertices: must terminate.
#[test]
fn obj_huge_file_without_vertices_terminates() {
    let mut file = Vec::new();
    for i in 0..200_000 {
        file.extend_from_slice(format!("vn {} {} {}\n", i, i, i).as_bytes());
    }
    let cloud = run_with_budget(HOSTILE_BUDGET, move || {
        cv_io::obj::read_obj(Cursor::new(file)).map(|c| c.points)
    })
    .expect_ok("200k `vn` lines");
    assert_eq!(cloud.len(), 0);
}

// ===========================================================================
// 2. Face indices: the 1-based to 0-based conversion
// ===========================================================================

/// A face index far beyond the vertex list. `i - 1` succeeds and the index is
/// stored as-is; `to_triangle_mesh` then hands downstream code an out-of-range
/// index.
#[test]
fn obj_out_of_range_face_index_is_rejected() {
    let file = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999999\n".to_vec();
    let m = obj_mesh(&file).expect_ok("out-of-range face");
    if let Err(payload) = std::panic::catch_unwind(|| {
        assert_mesh_consistent(&m, "obj_robustness::obj_out_of_range_face_index_is_rejected")
    }) {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&'static str>().map(|s| s.to_string()))
            .unwrap_or_default();
        panic!("KNOWN BUG (obj face indices are not bounds-checked): {msg}");
    }
    panic!(
        "KNOWN BUG: ObjMesh::read accepted face index 999999 for a 3-vertex mesh; the face \
         {:?} refers to vertex 999998, which does not exist",
        m.faces[0]
    );
}

/// Index 0 is explicitly rejected (OBJ is 1-based), which is the one case that
/// is handled.
#[test]
fn obj_face_index_zero_is_rejected() {
    let file = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n".to_vec();
    let err = obj_raw(&file).expect_err("index 0");
    assert!(err.contains("1-based"), "got: {err}");
}

/// A non-numeric face index.
#[test]
fn obj_non_numeric_face_index_errors() {
    let file = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n".to_vec();
    let err = obj_raw(&file).expect_err("x");
    assert!(err.contains("Invalid face index"), "got: {err}");
}

/// `f 1/2/3 4/5/6 7/8/9` (v/vt/vn form) parses; a negative *relative* index -
/// legal OBJ - is rejected by the `usize` parse.
#[test]
fn obj_face_with_slashes_and_negative_index() {
    let file = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n".to_vec();
    let m = obj_raw(&file).expect_ok("v/vt/vn");
    assert_eq!(m.faces[0], vec![0, 1, 2]);

    let rel = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n".to_vec();
    obj_raw(&rel).expect_err("relative index");
}

/// A face index of `usize::MAX` becomes `usize::MAX - 1` after the 1-based
/// conversion.
#[test]
fn obj_face_index_near_usize_max_is_rejected() {
    let file = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 18446744073709551615\n".to_vec();
    let m = obj_mesh(&file).expect_ok("usize::MAX-1 index");
    if let Err(payload) = std::panic::catch_unwind(|| {
        assert_mesh_consistent(&m, "obj_robustness::obj_face_index_near_usize_max_is_rejected")
    }) {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&'static str>().map(|s| s.to_string()))
            .unwrap_or_default();
        panic!("KNOWN BUG (obj face indices are not bounds-checked): {msg}");
    }
    panic!("KNOWN BUG: face index usize::MAX survived into the mesh as {:?}", m.faces[0]);
}

/// A `v` line that appears *after* an `f` line: the face refers to a vertex that
/// has not been read yet, and forward references are accepted.
#[test]
fn obj_forward_face_reference_is_rejected() {
    let file = b"f 1 2 3\nv 0 0 0\nv 1 0 0\nv 0 1 0\n".to_vec();
    let m = obj_mesh(&file).expect_ok("forward reference");
    if let Err(payload) = std::panic::catch_unwind(|| {
        assert_mesh_consistent(&m, "obj_robustness::obj_forward_face_reference_is_rejected")
    }) {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&'static str>().map(|s| s.to_string()))
            .unwrap_or_default();
        panic!("KNOWN BUG (obj face indices are not bounds-checked): {msg}");
    }
    panic!("KNOWN BUG: a forward face reference was accepted: {:?}", m.faces[0]);
}

/// An n-gon: fan triangulation must produce `len - 2` triangles, all with
/// in-range indices.
#[test]
fn obj_ngon_fan_triangulation_is_consistent() {
    let mut file = String::new();
    for i in 0..8 {
        file.push_str(&format!("v {i} 0 0\n"));
    }
    file.push_str("f 1 2 3 4 5 6 7 8\n");
    let m = obj_mesh(&file.as_bytes()).expect_ok("n-gon");
    assert_eq!(m.faces.len(), 6, "an 8-gon fans into 6 triangles");
    assert_mesh_consistent(&m, "8-gon");
}

/// A two-element "face" (`f 1 2`) is skipped by the `parts.len() >= 4` guard.
#[test]
fn obj_two_element_face_is_reported() {
    let file = b"v 0 0 0\nv 1 0 0\nf 1 2\n".to_vec();
    match obj_raw(&file) {
        Outcome::Error(_) => {}
        Outcome::Returned(m) => panic!(
            "KNOWN BUG: 'f 1 2' (a 2-vertex face) is dropped silently and ObjMesh::read returns \
             Ok with {} face(s); it must return Err",
            m.faces.len()
        ),
        Outcome::Panicked(m) => panic!("PANIC: {m}"),
        Outcome::TimedOut => panic!("HANG"),
    }
}
