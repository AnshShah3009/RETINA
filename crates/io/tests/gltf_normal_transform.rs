//! glTF normals must be transformed by the inverse-transpose of the node.
//!
//! glTF 2.0 says normal vectors are transformed "by the inverse transpose of
//! the node's world matrix" and then normalized. `read_gltf` applied the linear
//! part directly, which is only equivalent when that part is a similarity
//! transform. Under a non-uniform scale it tilts the normal away from the
//! surface: the transformed surface is still perpendicular to the
//! inverse-transpose, but not to the directly multiplied vector.
//!
//! A small input shows it. One vertex with normal `(1, 1, 1)` under a node
//! scaled by `(2, 1, 1)` must come back along `(0.5, 1, 1)`; the direct linear
//! transform gives `(2, 1, 1)` instead.

#![cfg(feature = "gltf")]

use cv_io::read_gltf;
use nalgebra::Vector3;
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
            "cv_io_gltf_normal_{tag}_{}_{}_{nanos}",
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

#[test]
fn read_gltf_uses_inverse_transpose_for_normals_under_non_uniform_scale() {
    let dir = TempDir::new("non_uniform_scale");
    let gltf = r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0, "scale": [2.0, 1.0, 1.0]}],
  "meshes": [{"name": "scaled", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
    {"bufferView": 1, "componentType": 5126, "count": 1, "type": "VEC3"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 12},
    {"buffer": 0, "byteOffset": 12, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 24, "uri": "mesh.bin"}]
}
"#;
    fs::write(dir.0.join("model.gltf"), gltf).expect("write gltf");

    // POSITION (0, 0, 0) followed by NORMAL (1, 1, 1).
    let mut bin = Vec::new();
    for v in [0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0] {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(dir.0.join("mesh.bin"), bin).expect("write bin");

    let meshes = read_gltf(dir.0.join("model.gltf")).expect("load gltf");
    assert_eq!(meshes.len(), 1);
    let normals = meshes[0].normals.as_ref().expect("normals missing");

    // Inverse-transpose of diag(2, 1, 1) is diag(0.5, 1, 1), so the normal
    // (1, 1, 1) maps to (0.5, 1, 1). Direct multiplication (the old behaviour)
    // gives (2, 1, 1), which is not perpendicular to the transformed surface.
    let expected = Vector3::new(0.5f32, 1.0, 1.0);
    assert!(
        (normals[0] - expected).norm() < 1e-6,
        "normal should be the inverse-transpose {expected:?}, got {:?}",
        normals[0]
    );

    // Sanity: the transformed normal is perpendicular to a tangent of the
    // original surface that is itself perpendicular to (1, 1, 1).
    let tangent = Vector3::new(1.0f32, -1.0, 0.0);
    let transformed_tangent = Vector3::new(2.0 * tangent.x, tangent.y, tangent.z);
    assert!(
        transformed_tangent.dot(&normals[0]).abs() < 1e-6,
        "transformed normal is not perpendicular to the transformed tangent"
    );
}
