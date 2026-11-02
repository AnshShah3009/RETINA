//! Every GLB `write_glb` produces must be readable by `read_gltf`.
//!
//! Two inputs produced files that violate glTF 2.0's schema, which the reader
//! (the `gltf` crate's validator) then refused:
//!
//! * **no faces** (a vertex-only mesh - what a point cloud is): the index
//!   accessor was written with `"count": 0` over a bufferView with
//!   `"byteLength": 0`. The schema requires `count >= 1` and
//!   `byteLength >= 1`, so `write_glb` returned `Ok` for a file nothing can
//!   load. glTF's way to say "no indices" is to omit the `indices` accessor and
//!   use the `POINTS` mode for vertices alone.
//! * **a normal count that differs from the vertex count**: the NORMAL accessor
//!   declared `count = vertices.len()` over a bufferView holding
//!   `normals.len()` normals, so a short normal list made the accessor overrun
//!   its view. There is no correct encoding of such a mesh, so it is reported.

#![cfg(feature = "gltf")]

use cv_io::{read_gltf, write_glb};
use nalgebra::{Point3, Vector3};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn temp_glb(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "cv_io_glb_writer_{tag}_{}_{}.glb",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

fn quad() -> Vec<Point3<f32>> {
    vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]
}

#[test]
fn a_mesh_without_faces_writes_a_loadable_glb() {
    let path = temp_glb("no_faces");
    write_glb(&path, &quad(), &[], None).expect("write_glb");
    let read = read_gltf(&path);
    let _ = std::fs::remove_file(&path);
    let meshes = read.unwrap_or_else(|e| panic!("write_glb produced an unreadable GLB: {e}"));
    // A vertex-only primitive is not a triangle mesh, so the reader skips it.
    assert!(
        meshes.iter().all(|m| m.faces.is_empty()),
        "no faces were written, got {meshes:?}"
    );
}

#[test]
fn a_normal_count_that_does_not_match_the_vertices_is_rejected() {
    let path = temp_glb("short_normals");
    let normals = vec![Vector3::new(0.0f32, 0.0, 1.0); 3];
    let result = write_glb(&path, &quad(), &[[0, 1, 2], [0, 2, 3]], Some(&normals));
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_err(),
        "4 vertices with 3 normals has no valid glTF encoding, but write_glb returned Ok"
    );
}

/// Control: a mesh with faces and one normal per vertex round-trips.
#[test]
fn a_mesh_with_faces_and_normals_round_trips() {
    let path = temp_glb("control");
    let normals = vec![Vector3::new(0.0f32, 0.0, 1.0); 4];
    write_glb(&path, &quad(), &[[0, 1, 2], [0, 2, 3]], Some(&normals)).expect("write_glb");
    let read = read_gltf(&path);
    let _ = std::fs::remove_file(&path);
    let meshes = read.expect("read_gltf");
    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0].faces, vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(meshes[0].normals.as_ref().map(Vec::len), Some(4));
}

/// The reader side of the same schema rule: an accessor with `"count": 0`
/// (glTF requires `count >= 1`) made `read_gltf` panic inside the `gltf`
/// crate's accessor iterator (`attempt to subtract with overflow` on
/// `count - 1`), so a malformed file aborted the caller instead of returning
/// an error.
#[test]
fn a_zero_count_accessor_is_an_error_not_a_panic() {
    let dir = std::env::temp_dir().join(format!(
        "cv_io_glb_zero_count_{}_{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let gltf = r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
    {"bufferView": 1, "componentType": 5125, "count": 0, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 4}
  ],
  "buffers": [{"byteLength": 40, "uri": "mesh.bin"}]
}
"#;
    std::fs::write(dir.join("model.gltf"), gltf).expect("write gltf");
    let mut bin = Vec::new();
    for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    bin.extend_from_slice(&0u32.to_le_bytes());
    std::fs::write(dir.join("mesh.bin"), bin).expect("write bin");

    let path = dir.join("model.gltf");
    let outcome = std::panic::catch_unwind(|| read_gltf(&path).map(|m| m.len()));
    let _ = std::fs::remove_dir_all(&dir);
    match outcome {
        Err(_) => panic!("read_gltf panicked on a zero-count accessor"),
        Ok(result) => assert!(result.is_err(), "zero-count accessor accepted: {result:?}"),
    }
}
