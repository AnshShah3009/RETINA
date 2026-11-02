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

use cv_core::{Error, Result};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use std::collections::HashMap;
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

/// Read triangle meshes from a glTF or GLB file.
///
/// One [`GltfMesh`] is returned per triangle primitive (`TRIANGLES`,
/// `TRIANGLE_STRIP` or `TRIANGLE_FAN`; point and line primitives are skipped)
/// of each node that instantiates a document mesh. The node's world transform — its own `node.transform()`
/// composed with every ancestor — is applied to the positions and normals
/// (normals by its inverse transpose, then renormalised to unit length), so
/// a mesh authored at the origin with the translation on the node comes back
/// in world coordinates. A mesh referenced by several nodes is returned once
/// per node, so none of those transforms is dropped. A mesh that no node
/// references is returned once, in mesh-local coordinates.
pub fn read_gltf<P: AsRef<Path>>(path: P) -> Result<Vec<GltfMesh>> {
    let (document, buffers, _images) = gltf::import(path.as_ref())
        .map_err(|e| cv_core::Error::IoError(format!("Failed to load glTF: {}", e)))?;

    let placements = mesh_world_transforms(&document);
    let mut meshes = Vec::new();

    for mesh in document.meshes() {
        let mesh_name = mesh
            .name()
            .unwrap_or(&format!("mesh_{}", mesh.index()))
            .to_string();
        let identity = [Matrix4::identity()];
        let worlds = match placements.get(&mesh.index()) {
            Some(list) if !list.is_empty() => list.as_slice(),
            _ => &identity,
        };

        for world in worlds {
            for primitive in mesh.primitives() {
                // Triangle strips and fans are triangle topologies too and are
                // unpacked below. They used to be skipped with the point and
                // line modes, so a strip-encoded mesh vanished from the result
                // while the call still returned `Ok`.
                let mode = primitive.mode();
                if !matches!(
                    mode,
                    gltf::mesh::Mode::Triangles
                        | gltf::mesh::Mode::TriangleStrip
                        | gltf::mesh::Mode::TriangleFan
                ) {
                    continue;
                }

                // glTF 2.0 requires every accessor's `count` to be at least 1.
                // A zero count is not merely empty: the `gltf` crate's accessor
                // iterator computes `count - 1` and panicked ("attempt to
                // subtract with overflow"), so a malformed file aborted the
                // caller. Reject it here, before any accessor is read.
                if let Some(accessor) = primitive
                    .attributes()
                    .map(|(_, accessor)| accessor)
                    .chain(primitive.indices())
                    .find(|accessor| accessor.count() == 0)
                {
                    return Err(cv_core::Error::ParseError(format!(
                        "glTF: accessor {} of mesh {:?} has count 0; glTF 2.0 requires count >= 1",
                        accessor.index(),
                        mesh_name
                    )));
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // Positions (required). `world` is the node's transform in the
                // scene; a mesh left at the origin with the translation on the
                // node is the usual export, and the bytes alone are mesh-local.
                let positions: Vec<Point3<f32>> = match reader.read_positions() {
                    Some(iter) => iter
                        .map(|p| transform_position(world, Point3::new(p[0], p[1], p[2])))
                        .collect(),
                    None => continue,
                };

                // Normals (optional). The linear part of the same node transform,
                // so a rotated node does not leave them in mesh-local space.
                let normals: Option<Vec<Vector3<f32>>> = reader.read_normals().map(|iter| {
                    iter.map(|n| transform_normal(world, Vector3::new(n[0], n[1], n[2])))
                        .collect()
                });

                // Tex coords (optional)
                let tex_coords: Option<Vec<[f32; 2]>> = reader
                    .read_tex_coords(0)
                    .map(|iter| iter.into_f32().collect());

                // Indices
                //
                // A primitive's `indices` accessor is a *separate* accessor from
                // `POSITION`, and nothing in the file ties it to the vertex count:
                // an index of 5000 against a POSITION accessor of one vertex is a
                // perfectly loadable file. The conversion below did no check at
                // all, so such an index survived into `GltfMesh::faces`, and from
                // there into `TriangleMesh`, which stores no bound either - the
                // panic surfaced much later in `cv_3d`, at
                // `TriangleMesh::compute_face_normals`'s `self.vertices[face[0]]`.
                // `read_obj` already rejects out-of-range face indices; do the same.
                let sequence: Vec<usize> = if let Some(indices) = reader.read_indices() {
                    let idx: Vec<usize> = indices.into_u32().map(|i| i as usize).collect();
                    for &i in &idx {
                        if i >= positions.len() {
                            return Err(cv_core::Error::ParseError(format!(
                                "glTF: face index {i} is out of range: mesh {:?} has only {} \
                             vertex/vertices in its POSITION accessor",
                                mesh_name,
                                positions.len()
                            )));
                        }
                    }
                    idx
                } else {
                    // Non-indexed: the vertices themselves, in order.
                    (0..positions.len()).collect()
                };
                let mut faces = triangles_from_sequence(mode, &sequence);

                // glTF 2.0 §3.7.2.1: a node whose global transform has a
                // negative determinant (a mirror) makes its triangles wind
                // clockwise. The transform is baked into the positions above,
                // so restore counter-clockwise winding by swapping two corners.
                // Without this a mirrored instance came back inside out: the
                // winding normal pointed opposite the (inverse-transposed)
                // vertex normals.
                if world.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
                    for face in &mut faces {
                        face.swap(1, 2);
                    }
                }

                meshes.push(GltfMesh {
                    name: mesh_name.clone(),
                    vertices: positions,
                    normals,
                    faces,
                    tex_coords,
                });
            }
        }
    }

    Ok(meshes)
}

/// Triangles of a triangle-topology primitive whose vertex sequence (its
/// indices, or `0..count` without an index accessor) is `seq`, following
/// glTF 2.0 §3.7.2.1:
///
/// * `TRIANGLES`: `{v[3i], v[3i+1], v[3i+2]}`;
/// * `TRIANGLE_STRIP`: `{v[i], v[i + (1 + i % 2)], v[i + (2 - i % 2)]}`, which
///   keeps every triangle counter-clockwise;
/// * `TRIANGLE_FAN`: `{v[i + 1], v[i + 2], v[0]}`.
fn triangles_from_sequence(mode: gltf::mesh::Mode, seq: &[usize]) -> Vec<[usize; 3]> {
    match mode {
        gltf::mesh::Mode::TriangleStrip => (0..seq.len().saturating_sub(2))
            .map(|i| [seq[i], seq[i + 1 + i % 2], seq[i + 2 - i % 2]])
            .collect(),
        gltf::mesh::Mode::TriangleFan => (0..seq.len().saturating_sub(2))
            .map(|i| [seq[i + 1], seq[i + 2], seq[0]])
            .collect(),
        _ => seq.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
    }
}

/// Column-major glTF matrix (`m[column][row]`) as a nalgebra matrix.
fn gltf_to_matrix(m: [[f32; 4]; 4]) -> Matrix4<f32> {
    Matrix4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}

fn transform_position(world: &Matrix4<f32>, p: Point3<f32>) -> Point3<f32> {
    let h = world * p.to_homogeneous();
    Point3::new(h.x, h.y, h.z)
}

fn transform_normal(world: &Matrix4<f32>, n: Vector3<f32>) -> Vector3<f32> {
    // glTF 2.0: a normal is transformed "by the inverse transpose of the node's
    // world matrix". The linear part used to be applied directly, which is only
    // equivalent when that part is a similarity transform: under a non-uniform
    // scale the directly multiplied normal no longer stays perpendicular to the
    // transformed surface. With scale `(2, 1, 1)` the normal `(1, 1, 1)` of a
    // plane came back as `(2, 1, 1)`; its inverse transpose is `(0.5, 1, 1)`,
    // and only that one is perpendicular to the transformed tangent `(2, -1, 0)`
    // (dot = 0).
    let linear = Matrix3::new(
        world[(0, 0)],
        world[(0, 1)],
        world[(0, 2)],
        world[(1, 0)],
        world[(1, 1)],
        world[(1, 2)],
        world[(2, 0)],
        world[(2, 1)],
        world[(2, 2)],
    );
    // A singular linear part (a zero scale) has no inverse transpose. Keep the
    // previous direct transform rather than failing to load the mesh; the
    // geometry itself is degenerate in that case.
    if linear == Matrix3::identity() {
        return n;
    }
    let normal_matrix = linear
        .try_inverse()
        .map(|inverse| inverse.transpose())
        .unwrap_or(linear);
    // ...and renormalised: glTF 2.0 defines a normal as "a unit XYZ vector"
    // and `NORMAL` as "Normalized XYZ vertex normals", and the inverse
    // transpose of any scale changes the length (scale 3 turned (0, 0, 1)
    // into (0, 0, 0.333)). A zero vector has no direction and is kept.
    let transformed = normal_matrix * n;
    let len = transformed.norm();
    if len > 0.0 && len.is_finite() {
        transformed / len
    } else {
        transformed
    }
}

/// World transform of every node that instantiates a mesh, keyed by mesh index.
///
/// A node may sit under a transformed parent. The matrix stored here is the
/// product of the ancestor chain and the node's own transform, so a translation
/// on either is applied. Nodes are visited in document order.
fn mesh_world_transforms(document: &gltf::Document) -> HashMap<usize, Vec<Matrix4<f32>>> {
    let nodes: Vec<_> = document.nodes().collect();
    let mut local: HashMap<usize, Matrix4<f32>> = HashMap::new();
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    for node in &nodes {
        local.insert(node.index(), gltf_to_matrix(node.transform().matrix()));
        for child in node.children() {
            parent_of.insert(child.index(), node.index());
        }
    }

    let mut out: HashMap<usize, Vec<Matrix4<f32>>> = HashMap::new();
    for node in &nodes {
        let Some(mesh) = node.mesh() else {
            continue;
        };
        let world = world_matrix(node.index(), &local, &parent_of);
        out.entry(mesh.index()).or_default().push(world);
    }
    out
}

fn world_matrix(
    index: usize,
    local: &HashMap<usize, Matrix4<f32>>,
    parent_of: &HashMap<usize, usize>,
) -> Matrix4<f32> {
    let mut chain = Vec::new();
    let mut cursor = index;
    loop {
        chain.push(cursor);
        if chain.len() > local.len().saturating_add(1) {
            break;
        }
        match parent_of.get(&cursor).copied() {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    let mut world = Matrix4::identity();
    for idx in chain.iter().rev() {
        if let Some(m) = local.get(idx) {
            world *= *m;
        }
    }
    world
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
    // Build minimal binary glTF manually (no gltf crate write support needed)
    let n_verts = vertices.len();
    let n_faces = faces.len();

    // glTF accessors count in u32, and so does the index buffer. A mesh past
    // either limit used to be written anyway with `mesh.faces.len() as u32`,
    // which wraps: the accessor would declare a small count while the BIN chunk
    // carried gigabytes of index data, and every later offset in the file would
    // be wrong. Reject instead of truncating.
    if n_verts > u32::MAX as usize || n_faces > u32::MAX as usize / 3 {
        return Err(Error::ParseError(format!(
            "glTF: mesh is too large to write: {} vertices / {} faces exceed the u32 index and \
             accessor-count range of glTF 2.0",
            n_verts, n_faces
        )));
    }

    // The NORMAL accessor is per vertex. A list of another length used to be
    // written anyway, with the accessor declaring `n_verts` normals over a
    // bufferView holding `normals.len()`: a short list overran its view. There
    // is no valid encoding of such a mesh, so report it.
    if let Some(norms) = normals
        && norms.len() != n_verts
    {
        return Err(Error::ParseError(format!(
            "glTF: {} normals for {} vertices; NORMAL is a per-vertex attribute",
            norms.len(),
            n_verts
        )));
    }

    // An empty mesh has no valid encoding *as a mesh*: the glTF 2.0 schema
    // requires accessor `count >= 1`, bufferView and buffer `byteLength >= 1`,
    // at least one primitive per mesh and at least one mesh in `meshes` when
    // present. It used to be written as a POSITION accessor of count 0 over
    // 0-byte views. It is now a valid asset whose scene holds one node with no
    // mesh (and no BIN chunk), which `read_gltf` reads back as no meshes. (A
    // scene without `nodes` is valid glTF too, but the `gltf` crate rejects it.)
    if n_verts == 0 {
        if let Some(face) = faces.first() {
            return Err(Error::ParseError(format!(
                "glTF: face index {} is out of range: only 0 vertex/vertices were given",
                face[0]
            )));
        }
        let json = r#"{
  "asset": {"version": "2.0", "generator": "RETINA cv-io"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{}]
}"#;
        return write_glb_container(path.as_ref(), json, &[]);
    }

    // Position data
    let mut pos_data = Vec::with_capacity(n_verts * 12);
    // Seeded from the first vertex rather than from sentinels.
    //
    // `f32::MIN` is the most *negative* finite f32, so a `max` seeded with it can
    // never rise. With no vertices the sentinels survived into the file: measured,
    // `write_glb` on an empty mesh wrote
    //
    //     "min": [3.4028235e38, ...], "max": [-3.4028235e38, ...]
    //
    // into the POSITION accessor - min greater than max on every axis, in a
    // 876-byte file. No importer can read that as an extent.
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for (n, v) in vertices.iter().enumerate() {
        let coords = [v.x, v.y, v.z];
        for (i, &c) in coords.iter().enumerate() {
            min_pos[i] = min_pos[i].min(c);
            max_pos[i] = max_pos[i].max(c);
        }
        for &c in &coords {
            pos_data.extend_from_slice(&c.to_le_bytes());
        }
        if n == 0 {
            // The first vertex *is* the seed, so nothing is left to accumulate.
            min_pos = coords;
            max_pos = coords;
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
    //
    // This used to be a silent `i as u32` per index, with no check that the
    // index addresses a written vertex at all. Such a file has no correct
    // encoding - the reader has to reject it - so reject it here too, where the
    // caller can still see which face was wrong.
    let mut idx_data = Vec::with_capacity(n_faces * 12);
    for f in faces {
        for &i in f {
            if i >= n_verts {
                return Err(Error::ParseError(format!(
                    "glTF: face index {i} is out of range: only {n_verts} vertex/vertices were \
                     given"
                )));
            }
            idx_data.extend_from_slice(&(i as u32).to_le_bytes());
        }
    }

    // Pad to 4-byte alignment
    fn pad4(data: &mut Vec<u8>) {
        while data.len() % 4 != 0 {
            data.push(0);
        }
    }

    // glTF accessors need `count >= 1` and bufferViews `byteLength >= 1`, so a
    // mesh without faces has no index accessor at all: its vertices are
    // written as a POINTS primitive. An index accessor of count 0 over an
    // empty bufferView used to be written instead, and `read_gltf` panicked on
    // the file (`count - 1` in the gltf crate).
    let has_indices = n_faces > 0;
    // Accessor / bufferView numbering: [indices], POSITION, [NORMAL].
    let first = usize::from(has_indices);

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
        format!(r#""POSITION": {}, "NORMAL": {}"#, first, first + 1)
    } else {
        format!(r#""POSITION": {first}"#)
    };
    let primitive_topology = if has_indices {
        r#""indices": 0"#
    } else {
        // POINTS: vertices only.
        r#""mode": 0"#
    };

    let mut accessors = Vec::new();
    let mut views = Vec::new();
    if has_indices {
        accessors.push(format!(
            r#"{{"bufferView": 0, "componentType": 5125, "count": {}, "type": "SCALAR"}}"#,
            n_faces * 3
        ));
        views.push(format!(
            r#"{{"buffer": 0, "byteOffset": {idx_offset}, "byteLength": {idx_len}, "target": 34963}}"#
        ));
    }
    accessors.push(format!(
        r#"{{"bufferView": {first}, "componentType": 5126, "count": {}, "type": "VEC3", "min": [{}, {}, {}], "max": [{}, {}, {}]}}"#,
        n_verts,
        min_pos[0],
        min_pos[1],
        min_pos[2],
        max_pos[0],
        max_pos[1],
        max_pos[2]
    ));
    views.push(format!(
        r#"{{"buffer": 0, "byteOffset": {pos_offset}, "byteLength": {pos_len}, "target": 34962}}"#
    ));
    if has_normals {
        accessors.push(format!(
            r#"{{"bufferView": {}, "componentType": 5126, "count": {}, "type": "VEC3"}}"#,
            first + 1,
            n_verts
        ));
        views.push(format!(
            r#"{{"buffer": 0, "byteOffset": {norm_offset}, "byteLength": {norm_len}, "target": 34962}}"#
        ));
    }
    let accessors = format!("[\n    {}\n  ]", accessors.join(",\n    "));
    let buffer_views = format!("[\n    {}\n  ]", views.join(",\n    "));

    let json = format!(
        r#"{{
  "asset": {{"version": "2.0", "generator": "RETINA cv-io"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{{}}}, {}}}]}}],
  "accessors": {},
  "bufferViews": {},
  "buffers": [{{"byteLength": {}}}]
}}"#,
        attributes, primitive_topology, accessors, buffer_views, total_bin_len
    );

    write_glb_container(path.as_ref(), &json, &bin_buffer)
}

/// Write the GLB container: header, JSON chunk and, when `bin` is not empty,
/// the BIN chunk (GLB 2.0 makes it optional; a buffer of `byteLength` 0 is
/// not allowed, so an asset without binary data has none).
fn write_glb_container(path: &Path, json: &str, bin: &[u8]) -> Result<()> {
    use std::io::Write;

    let mut json_padded = json.as_bytes().to_vec();
    while json_padded.len() % 4 != 0 {
        json_padded.push(b' ');
    }

    // GLB header: magic + version + length
    let bin_chunk_len = if bin.is_empty() { 0 } else { 8 + bin.len() };
    let total_length = 12 + 8 + json_padded.len() + bin_chunk_len;

    // The GLB header stores its own length in a u32. A larger mesh would wrap
    // and produce a file whose declared length disagrees with its contents.
    let total_length = u32::try_from(total_length).map_err(|_| {
        Error::ParseError(format!(
            "glTF: GLB would be {total_length} bytes, past the 4 GiB the container can describe"
        ))
    })?;

    let mut file = std::fs::File::create(path)
        .map_err(|e| cv_core::Error::IoError(format!("Failed to create GLB: {}", e)))?;

    // Header
    file.write_all(&0x46546c67u32.to_le_bytes())?; // magic "glTF"
    file.write_all(&2u32.to_le_bytes())?; // version
    file.write_all(&total_length.to_le_bytes())?;

    // JSON chunk
    file.write_all(&(json_padded.len() as u32).to_le_bytes())?;
    file.write_all(&0x4e4f534au32.to_le_bytes())?; // "JSON"
    file.write_all(&json_padded)?;

    // BIN chunk
    if !bin.is_empty() {
        file.write_all(&(bin.len() as u32).to_le_bytes())?;
        file.write_all(&0x004e4942u32.to_le_bytes())?; // "BIN\0"
        file.write_all(bin)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "cv_io_gltf_{tag}_{}_{}_{nanos}",
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

    /// A mesh whose only vertex is the origin, with the translation `(1, 2, 3)`
    /// on the node. Blender exports this way: the POSITION accessor stays in
    /// mesh-local coordinates and the node carries the placement. The point
    /// returned is the transformed one.
    #[test]
    fn read_gltf_applies_node_translation_to_an_origin_vertex() {
        let dir = TempDir::new("node_translation");
        let gltf = r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0, 1]}],
  "nodes": [
    {"mesh": 0, "translation": [1.0, 2.0, 3.0]},
    {"mesh": 0, "translation": [4.0, 5.0, 6.0]}
  ],
  "meshes": [{"name": "origin", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
    {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 12},
    {"buffer": 0, "byteOffset": 12, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 24, "uri": "mesh.bin"}]
}
"#;
        fs::write(dir.0.join("model.gltf"), gltf).expect("write gltf");
        fs::write(dir.0.join("mesh.bin"), [0u8; 24]).expect("write bin");

        let meshes = read_gltf(dir.0.join("model.gltf")).expect("load gltf");
        // One result per node. Both reference the same document mesh, and each
        // node's translation is kept.
        assert_eq!(meshes.len(), 2);
        assert_eq!(meshes[0].vertices, vec![Point3::new(1.0, 2.0, 3.0)]);
        assert_eq!(meshes[1].vertices, vec![Point3::new(4.0, 5.0, 6.0)]);
    }
}
