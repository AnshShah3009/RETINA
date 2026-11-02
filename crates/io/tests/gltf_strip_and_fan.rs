//! `TRIANGLE_STRIP` (mode 5) and `TRIANGLE_FAN` (mode 6) primitives are
//! triangle meshes and must be read as such.
//!
//! `read_gltf` skipped every primitive whose mode was not `TRIANGLES`, with the
//! point and line modes, and still returned `Ok`: a strip-encoded quad came back
//! as **no mesh at all**. glTF 2.0 §3.7.2.1 defines the triangles of both modes:
//!
//! * strip: `p_i = {v_i, v_{i+(1+i%2)}, v_{i+(2-i%2)}}`
//! * fan:   `p_i = {v_{i+1}, v_{i+2}, v_0}`

#![cfg(feature = "gltf")]

use cv_io::read_gltf;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "cv_io_gltf_topology_{tag}_{}_{}_{nanos}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A non-indexed primitive of four vertices in the z = 0 plane.
fn load(mode: u32, xy: [[f32; 2]; 4], tag: &str) -> Vec<cv_io::GltfMesh> {
    let dir = TempDir::new(tag);
    let gltf = format!(
        r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "mode": {mode}}}]}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}}
  ],
  "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 48}}],
  "buffers": [{{"byteLength": 48, "uri": "mesh.bin"}}]
}}
"#
    );
    fs::write(dir.0.join("model.gltf"), gltf).expect("write gltf");
    let mut bin = Vec::new();
    for [x, y] in xy {
        for v in [x, y, 0.0f32] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    fs::write(dir.0.join("mesh.bin"), bin).expect("write bin");
    read_gltf(dir.0.join("model.gltf")).expect("load gltf")
}

/// Every face must wind counter-clockwise seen from +z (positive signed area).
fn assert_ccw(mesh: &cv_io::GltfMesh) {
    for f in &mesh.faces {
        let (a, b, c) = (
            mesh.vertices[f[0]],
            mesh.vertices[f[1]],
            mesh.vertices[f[2]],
        );
        let z = (b - a).cross(&(c - a)).z;
        assert!(z > 0.0, "face {f:?} is not counter-clockwise (z = {z})");
    }
}

#[test]
fn triangle_strip_is_read_as_two_triangles() {
    // Strip order for a unit quad: (0,0) (1,0) (0,1) (1,1).
    let meshes = load(5, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "strip");
    assert_eq!(meshes.len(), 1, "the strip primitive was dropped");
    assert_eq!(meshes[0].faces, vec![[0, 1, 2], [1, 3, 2]]);
    assert_ccw(&meshes[0]);
}

#[test]
fn triangle_fan_is_read_as_two_triangles() {
    // Fan around (0,0): (0,0) (1,0) (1,1) (0,1).
    let meshes = load(6, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "fan");
    assert_eq!(meshes.len(), 1, "the fan primitive was dropped");
    assert_eq!(meshes[0].faces, vec![[1, 2, 0], [2, 3, 0]]);
    assert_ccw(&meshes[0]);
}

/// Control: point primitives are still not triangle meshes.
#[test]
fn points_primitive_is_still_skipped() {
    let meshes = load(
        0,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "points",
    );
    assert!(meshes.is_empty());
}
