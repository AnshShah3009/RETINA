//! glTF 2.0 mesh I/O.
//!
//! Reads triangle meshes from glTF/GLB files. Feature-gated behind `gltf`.
//!
//! # Example
//! ```ignore
//! use cv_io::gltf_io::read_gltf;
//! let meshes = read_gltf("model.glb")?;
//! for mesh in &meshes {
//!     println!("{} vertices, {} faces", mesh.vertices.len(), mesh.faces.len());
//! }
//! ```

use cv_core::Result;
use nalgebra::{Point3, Vector3};
use std::path::Path;

/// A simple triangle mesh extracted from glTF.
#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name: String,
    pub vertices: Vec<Point3<f32>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub faces: Vec<[usize; 3]>,
    pub tex_coords: Option<Vec<[f32; 2]>>,
}

/// Read all triangle meshes from a glTF or GLB file.
pub fn read_gltf<P: AsRef<Path>>(path: P) -> Result<Vec<GltfMesh>> {
    let (document, buffers, _images) = gltf::import(path.as_ref())
        .map_err(|e| cv_core::Error::IoError(format!("Failed to load glTF: {}", e)))?;

    let mut meshes = Vec::new();

    for mesh in document.meshes() {
        let mesh_name = mesh
            .name()
            .unwrap_or(&format!("mesh_{}", mesh.index()))
            .to_string();

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Positions (required)
            let positions: Vec<Point3<f32>> = match reader.read_positions() {
                Some(iter) => iter.map(|p| Point3::new(p[0], p[1], p[2])).collect(),
                None => continue,
            };

            // Normals (optional)
            let normals: Option<Vec<Vector3<f32>>> = reader
                .read_normals()
                .map(|iter| iter.map(|n| Vector3::new(n[0], n[1], n[2])).collect());

            // Tex coords (optional)
            let tex_coords: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect());

            // Indices
            let faces: Vec<[usize; 3]> = if let Some(indices) = reader.read_indices() {
                let idx: Vec<usize> = indices.into_u32().map(|i| i as usize).collect();
                idx.chunks(3)
                    .filter(|c| c.len() == 3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect()
            } else {
                // Non-indexed: every 3 vertices form a triangle
                (0..positions.len() / 3)
                    .map(|i| [i * 3, i * 3 + 1, i * 3 + 2])
                    .collect()
            };

            meshes.push(GltfMesh {
                name: mesh_name.clone(),
                vertices: positions,
                normals,
                faces,
                tex_coords,
            });
        }
    }

    Ok(meshes)
}

/// Convert a GltfMesh to RETINA's TriangleMesh.
pub fn gltf_to_triangle_mesh(gltf_mesh: &GltfMesh) -> crate::TriangleMesh {
    let mut mesh = crate::TriangleMesh::with_vertices_and_faces(
        gltf_mesh.vertices.clone(),
        gltf_mesh.faces.clone(),
    );
    if let Some(ref normals) = gltf_mesh.normals {
        mesh.normals = Some(normals.clone());
    }
    mesh
}

/// Write a TriangleMesh to a GLB (binary glTF) file.
pub fn write_glb<P: AsRef<Path>>(
    path: P,
    vertices: &[Point3<f32>],
    faces: &[[usize; 3]],
    normals: Option<&[Vector3<f32>]>,
) -> Result<()> {
    use std::io::Write;

    // Build minimal binary glTF manually (no gltf crate write support needed)
    let n_verts = vertices.len();
    let n_faces = faces.len();

    // Position data
    let mut pos_data = Vec::with_capacity(n_verts * 12);
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for v in vertices {
        let coords = [v.x, v.y, v.z];
        for (i, &c) in coords.iter().enumerate() {
            min_pos[i] = min_pos[i].min(c);
            max_pos[i] = max_pos[i].max(c);
        }
        for &c in &coords {
            pos_data.extend_from_slice(&c.to_le_bytes());
        }
    }

    // Normal data
    let mut norm_data = Vec::new();
    if let Some(norms) = normals {
        for n in norms {
            for &c in &[n.x, n.y, n.z] {
                norm_data.extend_from_slice(&c.to_le_bytes());
            }
        }
    }

    // Index data (u32)
    let mut idx_data = Vec::with_capacity(n_faces * 12);
    for f in faces {
        for &i in f {
            idx_data.extend_from_slice(&(i as u32).to_le_bytes());
        }
    }

    // Pad to 4-byte alignment
    fn pad4(data: &mut Vec<u8>) {
        while data.len() % 4 != 0 {
            data.push(0);
        }
    }

    // Build buffer: indices + positions + normals
    let mut bin_buffer = Vec::new();
    let idx_offset = 0;
    let idx_len = idx_data.len();
    bin_buffer.extend_from_slice(&idx_data);
    pad4(&mut bin_buffer);

    let pos_offset = bin_buffer.len();
    let pos_len = pos_data.len();
    bin_buffer.extend_from_slice(&pos_data);
    pad4(&mut bin_buffer);

    let norm_offset = bin_buffer.len();
    let norm_len = norm_data.len();
    if !norm_data.is_empty() {
        bin_buffer.extend_from_slice(&norm_data);
        pad4(&mut bin_buffer);
    }

    let total_bin_len = bin_buffer.len();

    // Build JSON
    let has_normals = normals.is_some() && !norm_data.is_empty();
    let attributes = if has_normals {
        r#""POSITION": 1, "NORMAL": 2"#
    } else {
        r#""POSITION": 1"#
    };

    let mut accessors = format!(
        r#"[
    {{"bufferView": 0, "componentType": 5125, "count": {}, "type": "SCALAR"}},
    {{"bufferView": 1, "componentType": 5126, "count": {}, "type": "VEC3", "min": [{}, {}, {}], "max": [{}, {}, {}]}}"#,
        n_faces * 3,
        n_verts,
        min_pos[0],
        min_pos[1],
        min_pos[2],
        max_pos[0],
        max_pos[1],
        max_pos[2]
    );
    if has_normals {
        accessors += &format!(
            r#",
    {{"bufferView": 2, "componentType": 5126, "count": {}, "type": "VEC3"}}"#,
            n_verts
        );
    }
    accessors += "\n  ]";

    let mut buffer_views = format!(
        r#"[
    {{"buffer": 0, "byteOffset": {}, "byteLength": {}, "target": 34963}},
    {{"buffer": 0, "byteOffset": {}, "byteLength": {}, "target": 34962}}"#,
        idx_offset, idx_len, pos_offset, pos_len
    );
    if has_normals {
        buffer_views += &format!(
            r#",
    {{"buffer": 0, "byteOffset": {}, "byteLength": {}, "target": 34962}}"#,
            norm_offset, norm_len
        );
    }
    buffer_views += "\n  ]";

    let json = format!(
        r#"{{
  "asset": {{"version": "2.0", "generator": "RETINA cv-io"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{{}}}, "indices": 0}}]}}],
  "accessors": {},
  "bufferViews": {},
  "buffers": [{{"byteLength": {}}}]
}}"#,
        attributes, accessors, buffer_views, total_bin_len
    );

    let json_bytes = json.as_bytes();
    let mut json_padded = json_bytes.to_vec();
    while json_padded.len() % 4 != 0 {
        json_padded.push(b' ');
    }

    // GLB header: magic + version + length
    let total_length = 12 + 8 + json_padded.len() + 8 + total_bin_len;

    let mut file = std::fs::File::create(path.as_ref())
        .map_err(|e| cv_core::Error::IoError(format!("Failed to create GLB: {}", e)))?;

    // Header
    file.write_all(&0x46546C67u32.to_le_bytes())?; // magic "glTF"
    file.write_all(&2u32.to_le_bytes())?; // version
    file.write_all(&(total_length as u32).to_le_bytes())?;

    // JSON chunk
    file.write_all(&(json_padded.len() as u32).to_le_bytes())?;
    file.write_all(&0x4E4F534Au32.to_le_bytes())?; // "JSON"
    file.write_all(&json_padded)?;

    // BIN chunk
    file.write_all(&(total_bin_len as u32).to_le_bytes())?;
    file.write_all(&0x004E4942u32.to_le_bytes())?; // "BIN\0"
    file.write_all(&bin_buffer)?;

    Ok(())
}
