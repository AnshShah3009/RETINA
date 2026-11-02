//! A node whose world transform mirrors (negative determinant) reverses the
//! winding of its triangles, and the reader must undo that.
//!
//! glTF 2.0, section 3.7.2.1 ("Meshes"): "When a mesh primitive uses any
//! triangle-based topology, the determinant of the node's global transform
//! defines the winding order of that primitive. If the determinant is a
//! positive value, the winding order triangle faces is counterclockwise; in the
//! opposite case, the winding order is clockwise."
//!
//! `read_gltf` bakes the node transform into the positions but kept the stored
//! index order, so a mirrored instance came back with every triangle facing
//! inward: the face normal from the vertex order pointed opposite the
//! (correctly inverse-transposed) vertex normals. Mirrored instances are how
//! exporters write symmetric models (left/right wheels, wings), and anything
//! that derives orientation from winding - face normals, back-face culling,
//! inside/outside tests, ray casting - saw them inside out.

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
            "cv_io_gltf_mirror_{tag}_{}_{}_{nanos}",
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

/// One counter-clockwise triangle in the z = 0 plane with vertex normals
/// `+z`, placed by a node with the given `scale`. `indexed` selects an index
/// accessor or the non-indexed path.
fn load(
    scale: [f32; 3],
    indexed: bool,
    tag: &str,
) -> (Vec<Vector3<f32>>, Vec<[usize; 3]>, Vec<Vector3<f32>>) {
    let dir = TempDir::new(tag);
    let indices = if indexed { r#", "indices": 2"# } else { "" };
    let index_accessor = if indexed {
        r#",
    {"bufferView": 2, "componentType": 5125, "count": 3, "type": "SCALAR"}"#
    } else {
        ""
    };
    let index_view = if indexed {
        r#",
    {"buffer": 0, "byteOffset": 72, "byteLength": 12}"#
    } else {
        ""
    };
    let byte_length = if indexed { 84 } else { 72 };
    let gltf = format!(
        r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0, "scale": [{}, {}, {}]}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "NORMAL": 1}}{indices}, "mode": 4}}]}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}}{index_accessor}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 36}}{index_view}
  ],
  "buffers": [{{"byteLength": {byte_length}, "uri": "mesh.bin"}}]
}}
"#,
        scale[0], scale[1], scale[2]
    );
    fs::write(dir.0.join("model.gltf"), gltf).expect("write gltf");

    let mut bin = Vec::new();
    // POSITION: (0,0,0), (1,0,0), (0,1,0) - counter-clockwise seen from +z.
    for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    // NORMAL: +z for each vertex.
    for _ in 0..3 {
        for v in [0.0f32, 0.0, 1.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    if indexed {
        for i in [0u32, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
    }
    fs::write(dir.0.join("mesh.bin"), bin).expect("write bin");

    let meshes = read_gltf(dir.0.join("model.gltf")).expect("load gltf");
    assert_eq!(meshes.len(), 1);
    let m = &meshes[0];
    let verts = m.vertices.iter().map(|p| p.coords).collect();
    (verts, m.faces.clone(), m.normals.clone().expect("normals"))
}

/// Dot product of the winding normal of face 0 with its first vertex normal.
fn winding_agreement(
    verts: &[Vector3<f32>],
    faces: &[[usize; 3]],
    normals: &[Vector3<f32>],
) -> f32 {
    let [a, b, c] = faces[0];
    let face_normal = (verts[b] - verts[a]).cross(&(verts[c] - verts[a]));
    face_normal.dot(&normals[a])
}

#[test]
fn mirrored_node_keeps_triangles_facing_their_normals() {
    for indexed in [true, false] {
        let (verts, faces, normals) = load([-1.0, 1.0, 1.0], indexed, "mirror");
        assert_eq!(faces.len(), 1);
        let agreement = winding_agreement(&verts, &faces, &normals);
        assert!(
            agreement > 0.0,
            "indexed={indexed}: mirrored triangle faces away from its own vertex normal \
             (winding normal . vertex normal = {agreement}); faces {faces:?}, vertices {verts:?}, \
             normals {normals:?}"
        );
    }
}

/// Control: an unmirrored node (including a 180-degree turn made of two
/// negative scales, determinant +1) keeps the stored winding.
#[test]
fn unmirrored_nodes_keep_the_stored_winding() {
    for scale in [[1.0f32, 1.0, 1.0], [-1.0, -1.0, 1.0], [2.0, 3.0, 0.5]] {
        let (verts, faces, normals) = load(scale, true, "plain");
        assert_eq!(faces, vec![[0, 1, 2]], "scale {scale:?}");
        assert!(
            winding_agreement(&verts, &faces, &normals) > 0.0,
            "scale {scale:?}"
        );
    }
}
