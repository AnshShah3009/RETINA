//! Two confirmed defects in `crates/io/src/stl.rs` and `crates/io/src/gltf_io.rs`.
//!
//! **Defect 1 - STL: truncation at a facet boundary reads as a complete mesh.**
//!
//! `parse_ascii_stl` guarded truncation with
//! `if saw_facet && faces.len() * 3 != vertices.len()`. Both halves of that guard
//! are wrong:
//!
//! * a file cut *between* two facets already satisfies `faces.len() * 3 ==
//!   vertices.len()`, so it passed, and
//! * `endsolid` - the only terminator the format has - was never required, and
//!   `saw_facet` was never actually required either, so the invariant itself
//!   was only checked on files that happened to have a `facet normal` line.
//!
//! **Defect 2 - glTF: face indices are never range-checked.**
//!
//! `read_gltf` did `idx.chunks(3).map(|c| [c[0], c[1], c[2]])`, so an `indices`
//! accessor naming a vertex the POSITION accessor does not contain produced a
//! `GltfMesh` with out-of-range faces. `TriangleMesh` stores no bound, so the
//! panic surfaces later in `cv_3d::mesh::TriangleMesh::compute_face_normals`
//! (`self.vertices[face[0]]`), i.e. in someone else's crate.
//!
//! `read_obj` already does the bound check the right way; `gltf_to_triangle_mesh`
//! and `write_glb` are exercised here too, because the second defect is only
//! half-fixed by a reader-side check.

#![forbid(unsafe_code)]

use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use cv_io::stl::{read_stl, write_stl_ascii, write_stl_binary};

// ===========================================================================
// Per-test temp files
// ===========================================================================

/// This suite runs concurrently inside one test binary, and a shared path has
/// broken this repo twice before, so every file name carries the process id, a
/// per-process counter and the nanosecond clock.
static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempPath(PathBuf);

impl TempPath {
    fn new(tag: &str, ext: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("cv_io_mesh_format_robustness");
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir.join(format!("{tag}_{}_{seq}_{nanos}.{ext}", std::process::id())))
    }

    fn write(&self, bytes: &[u8]) -> &Self {
        fs::write(&self.0, bytes).expect("write temp file");
        self
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// A temp path that already holds `bytes`.
fn temp(tag: &str, ext: &str, bytes: &[u8]) -> TempPath {
    let p = TempPath::new(tag, ext);
    p.write(bytes);
    p
}

/// A triangle in the XY plane, shared by the STL and glTF controls.
fn triangle() -> (Vec<nalgebra::Point3<f32>>, Vec<[usize; 3]>) {
    (
        vec![
            nalgebra::Point3::new(0.0, 0.0, 0.0),
            nalgebra::Point3::new(1.0, 0.0, 0.0),
            nalgebra::Point3::new(0.0, 1.0, 0.0),
        ],
        vec![[0usize, 1, 2]],
    )
}

// ===========================================================================
// DEFECT 1 - STL truncation
// ===========================================================================

/// The control: a complete ASCII STL, written by this crate's own writer, reads
/// back as three vertices and one face.
#[test]
fn control_a_complete_ascii_stl_reads_back() {
    let file = TempPath::new("control_ascii", "stl");
    let (verts, faces) = triangle();
    let mut bytes = Vec::new();
    write_stl_ascii(
        &mut bytes,
        &cv_io::TriangleMesh::with_vertices_and_faces(verts, faces),
    )
    .expect("write ASCII STL");
    file.write(&bytes);

    let mesh = read_stl(BufReader::new(fs::File::open(file.path()).expect("open")))
        .unwrap_or_else(|e| panic!("a complete ASCII STL must parse, got: {e}"));

    assert_eq!(mesh.vertices.len(), 3, "vertex count");
    assert_eq!(mesh.faces.len(), 1, "face count");
    assert_eq!(mesh.faces[0], [0, 1, 2], "face indices");
    assert_eq!(mesh.vertices[2], nalgebra::Point3::new(0.0, 1.0, 0.0));
}

/// The control: a complete binary STL, and a binary file whose declared count
/// is 10 million while only one triangle is present.
///
/// The binary path differs from the ASCII one: it has a real declared count and
/// reads each record with `read_exact`, so the truncation is caught by the short
/// read rather than by an end-of-file terminator. Pinned here so the ASCII fix
/// cannot be mistaken for the thing that already worked.
#[test]
fn control_binary_stl_count_truncation_is_already_rejected() {
    let (verts, faces) = triangle();
    let mut good = Vec::new();
    write_stl_binary(
        &mut good,
        &cv_io::TriangleMesh::with_vertices_and_faces(verts, faces),
    )
    .expect("write binary STL");
    assert_eq!(good.len(), 80 + 4 + 50, "binary STL size");

    // The control: the complete binary file parses.
    let good_file = temp("control_binary", "stl", &good);
    let mesh = read_stl(BufReader::new(
        fs::File::open(good_file.path()).expect("open"),
    ))
    .unwrap_or_else(|e| panic!("a complete binary STL must parse, got: {e}"));
    assert_eq!(mesh.faces.len(), 1, "face count");

    // ...and the same file with the declared count inflated to 10 million does not.
    let mut inflated = good.clone();
    inflated[80..84].copy_from_slice(&10_000_000u32.to_le_bytes());
    let bad_file = temp("binary_count_mismatch", "stl", &inflated);
    let err = read_stl(BufReader::new(
        fs::File::open(bad_file.path()).expect("open"),
    ))
    .expect_err("a declared count of 10 million with one triangle present must be rejected");
    assert!(
        err.to_string().to_lowercase().contains("eof")
            || err.to_string().to_lowercase().contains("failed to fill")
            || err.to_string().to_lowercase().contains("end of file"),
        "want an EOF error, got: {err}"
    );
}

/// The defect, case 1: the file stops exactly between two facets.
///
/// There is no partial facet here, so the old invariant
/// `faces.len() * 3 == vertices.len()` already held and the file was accepted
/// as a complete one-facet mesh. The only thing missing is the `endsolid`
/// terminator.
#[test]
fn defect1_an_ascii_stl_cut_between_facets_is_rejected() {
    let truncated = b"solid x\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\n";

    let file = temp("cut_between_facets", "stl", truncated);
    assert!(
        !std::str::from_utf8(truncated).unwrap().contains("endsolid"),
        "sanity: the input really has no terminator"
    );

    match read_stl(BufReader::new(fs::File::open(file.path()).expect("open"))) {
        Err(_) => {}
        Ok(m) => panic!(
            "DEFECT 1: an ASCII STL with no 'endsolid' is accepted as a complete mesh \
             (Ok, {} vertices, {} faces). The only difference from a valid file is the \
             missing terminator.",
            m.vertices.len(),
            m.faces.len()
        ),
    }
}

/// The defect, case 2: the file stops before any facet, with the terminator
/// missing as well.
#[test]
fn defect1_an_ascii_stl_stopped_before_its_first_facet_is_rejected() {
    let truncated = b"solid x\n";

    let file = temp("before_first_facet", "stl", truncated);
    match read_stl(BufReader::new(fs::File::open(file.path()).expect("open"))) {
        Err(_) => {}
        Ok(m) => panic!(
            "DEFECT 1: 'solid x' with nothing after it is accepted as a complete mesh \
             (Ok, {} vertices, {} faces).",
            m.vertices.len(),
            m.faces.len()
        ),
    }
}

/// The defect, case 3: the declared-vs-actual mismatch the old guard skipped.
///
/// Vertices are read but no facet ever closes, and the file *does* end with
/// `endsolid`. `saw_facet` is false here, so the old
/// `if saw_facet && faces.len() * 3 != vertices.len()` never ran and three stray
/// vertices were returned with no faces - a caller sees a broken mesh rather
/// than a parse failure.
#[test]
fn defect1_vertices_without_a_closed_facet_are_rejected() {
    let file_bytes =
        b"solid x\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\nendsolid x\n";

    let file = temp("no_closed_facet", "stl", file_bytes);
    match read_stl(BufReader::new(fs::File::open(file.path()).expect("open"))) {
        Err(_) => {}
        Ok(m) => panic!(
            "DEFECT 1: {} vertices for 0 closed facets are accepted because the mismatch \
             check was gated on 'saw_facet'.",
            m.vertices.len()
        ),
    }
}

/// An ASCII STL with no `facet normal` lines is legal (`normal` is optional in
/// the format), so the count check must not depend on having seen one - but such
/// a file must still parse. This is the control for case 3.
#[test]
fn control_ascii_stl_without_facet_normals_still_parses() {
    let file_bytes = b"solid x\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n";

    let file = temp("no_facet_normal", "stl", file_bytes);
    let mesh = read_stl(BufReader::new(fs::File::open(file.path()).expect("open")))
        .unwrap_or_else(|e| panic!("a facet-normal-less ASCII STL is legal, got: {e}"));

    assert_eq!(mesh.vertices.len(), 3, "vertex count");
    assert_eq!(mesh.faces.len(), 1, "face count");
}

// ===========================================================================
// DEFECT 2 - glTF face indices
// ===========================================================================

#[cfg(feature = "gltf")]
mod gltf {
    use cv_io::gltf_io::{gltf_to_triangle_mesh, read_gltf, write_glb};
    use nalgebra::Point3;
    use std::fs;

    use super::{temp, triangle, TempPath};

    /// Byte offset of the BIN chunk's payload inside a GLB written by
    /// `write_glb`.
    ///
    /// Only the container framing is decoded here - the container itself is
    /// produced by the crate's own writer, not hand-rolled.
    fn bin_chunk_offset(glb: &[u8]) -> usize {
        assert_eq!(&glb[..4], b"glTF", "GLB magic");
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json_type = &glb[16..20];
        assert_eq!(json_type, b"JSON", "first chunk is JSON");
        12 + 8 + json_len + 8
    }

    /// Replace one u32 in the BIN chunk without changing the file length, so no
    /// accessor or chunk length has to be touched.
    fn patch_u32(glb: &mut [u8], index: usize, value: u32) {
        let at = bin_chunk_offset(glb) + index * 4;
        glb[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// A GLB whose POSITION accessor holds one vertex while its `indices`
    /// accessor names vertex 5000 - the file from the defect report.
    fn glb_with_out_of_range_index() -> Vec<u8> {
        let file = TempPath::new("out_of_range", "glb");
        write_glb(
            file.path(),
            &[Point3::new(1.0, 2.0, 3.0)],
            &[[0, 0, 0]],
            None,
        )
        .expect("write_glb");
        let mut bytes = fs::read(file.path()).expect("the GLB should exist");

        // `write_glb` puts the index buffer first, so the indices are the first
        // 12 bytes of the BIN chunk.
        patch_u32(&mut bytes, 1, 5000);
        bytes
    }

    /// The control: the very same file *before* the index is patched is
    /// well-formed and must round-trip.
    #[test]
    fn control_a_well_formed_glb_round_trips() {
        let (verts, faces) = triangle();
        let file = TempPath::new("control_glb", "glb");
        write_glb(file.path(), &verts, &faces, None).expect("write_glb");

        let meshes = read_gltf(file.path())
            .unwrap_or_else(|e| panic!("a well-formed GLB must read, got: {e}"));
        assert_eq!(meshes.len(), 1, "mesh count");
        assert_eq!(meshes[0].vertices.len(), 3, "vertex count");
        assert_eq!(meshes[0].faces, vec![[0usize, 1, 2]], "faces");

        // And the control's faces index real vertices downstream.
        let normals = gltf_to_triangle_mesh(&meshes[0]).compute_face_normals();
        assert_eq!(normals.len(), 1, "normal count");
        assert!((normals[0].z - 1.0).abs() < 1e-5, "normal {:?}", normals[0]);
    }

    /// The defect: `read_gltf` returns a mesh whose face addresses a vertex that
    /// does not exist, which panics in `cv_3d` when the face is used.
    #[test]
    fn defect2_an_out_of_range_face_index_is_rejected() {
        let bytes = glb_with_out_of_range_index();
        let file = temp("out_of_range", "glb", &bytes);

        match read_gltf(file.path()) {
            Err(_) => {}
            Ok(meshes) => {
                let m = &meshes[0];
                let bad: Vec<[usize; 3]> = m
                    .faces
                    .iter()
                    .copied()
                    .filter(|f| f.iter().any(|&v| v >= m.vertices.len()))
                    .collect();
                panic!(
                    "DEFECT 2: read_gltf accepted faces {bad:?} against a POSITION accessor of {} \
                     vertices; compute_face_normals would index vertices[5000] out of bounds.",
                    m.vertices.len()
                );
            }
        }
    }

    /// The control for the reject-or-fix question at the writer end: indices are
    /// written as `i as u32` with no check, so a caller that hands `write_glb` a
    /// mesh with one vertex and a face pointing at vertex 2 gets a file on disk
    /// that no reader can accept.
    #[test]
    fn defect2_write_glb_rejects_an_out_of_range_face_index() {
        let file = TempPath::new("writer_bad_face", "glb");
        let verts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];

        let result = write_glb(file.path(), &verts, &[[0usize, 1, 2]], None);

        match result {
            Err(_) => {}
            Ok(()) => {
                // If it wrote the file anyway, it must at least not have
                // produced a mesh whose faces are addressable.
                let meshes = read_gltf(file.path())
                    .unwrap_or_else(|e| panic!("the written file should still parse: {e}"));
                let m = &meshes[0];
                assert!(
                    m.faces.iter().flatten().all(|&v| v < m.vertices.len()),
                    "DEFECT 2: write_glb wrote face {:?} against {} vertices unchecked",
                    m.faces,
                    m.vertices.len()
                );
            }
        }
    }

    /// The writer's control: a valid mesh is written and reads back unchanged.
    #[test]
    fn control_write_glb_writes_only_in_range_faces() {
        let (verts, faces) = triangle();
        let file = TempPath::new("writer_control", "glb");
        write_glb(file.path(), &verts, &faces, None).expect("a valid mesh must be writable");

        let meshes = read_gltf(file.path()).expect("read back");
        assert_eq!(meshes[0].faces, faces, "faces round-trip");
        assert!(
            meshes[0]
                .faces
                .iter()
                .flatten()
                .all(|&v| v < meshes[0].vertices.len()),
            "every written face must index a written vertex"
        );
    }
}
