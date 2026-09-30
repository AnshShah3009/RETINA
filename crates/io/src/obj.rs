//! OBJ (Wavefront Object) I/O
//!
//! OBJ is a common format for storing 3D mesh geometry.

use crate::mesh::TriangleMesh;
use crate::Result;
use cv_core::point_cloud::PointCloud;
use cv_core::Error;
use nalgebra::Point3;
use std::io::{BufRead, Write};

/// OBJ keywords that must be followed by whitespace to be read as a statement.
///
/// **Order matters:** the longest prefixes come first, so that `vn 1 1 1` is
/// matched as `vn` and not as an unspaced `v`.
const KEYWORDS: &[&str] = &["vn", "vt", "vp", "v", "f"];

/// Does `line` begin with the OBJ statement `keyword`?
///
/// The character after the keyword must be whitespace or the line must end
/// there; anything else means the token is not the keyword at all.
fn starts_with_keyword(line: &str, keyword: &str) -> bool {
    match line.strip_prefix(keyword) {
        Some(rest) => rest.is_empty() || rest.starts_with(char::is_whitespace),
        None => false,
    }
}

/// Return the statement keyword `line` starts with, if it is one of the known
/// OBJ keywords written without the separating space.
///
/// OBJ's grammar is line oriented, and exporters emit `v0.0 0.0 0.0` with no
/// space at all. The parser needs the space to tell the keyword from the first
/// coordinate, so such a line used to be skipped silently - which is worse than
/// an error, because every later `f` index then refers to a *different*
/// vertex. Reporting it keeps the mistake visible instead of shifting the whole
/// file by one.
fn unspaced_keyword(line: &str) -> Option<&'static str> {
    // Longest keyword the line actually begins with. `vn 1 1 1` must resolve to
    // `vn`, never to `v`, so the table is ordered longest-prefix-first and the
    // *first* prefix match wins - whether or not it is properly spaced.
    let keyword = KEYWORDS
        .iter()
        .copied()
        .find(|kw| line.starts_with(kw))?;
    let rest = &line[keyword.len()..];
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        // A well-formed statement.
        return None;
    }
    Some(keyword)
}

/// Read vertex positions from an OBJ file
pub fn read_obj<R: BufRead>(reader: R) -> Result<PointCloud> {
    let mut points = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(kw) = unspaced_keyword(line) {
            return Err(Error::ParseError(format!(
                "OBJ statement '{kw}' is not followed by a space: {line:?}"
            )));
        }

        // Parse vertex lines (v x y z)
        if starts_with_keyword(line, "v") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // A `v` line with fewer than three coordinates is malformed. The
            // old `parts.len() >= 4` guard dropped it silently, so the vertex
            // vanished and every later `f` index referred to a different
            // vertex - a plausible-wrong-result rather than a report.
            if parts.len() < 4 {
                return Err(Error::ParseError(format!(
                    "OBJ vertex line needs 3 coordinates, got {}: {line:?}",
                    parts.len().saturating_sub(1)
                )));
            }
            points.push(Point3::new(
                parse_obj_coordinate(parts[1], "x", line)?,
                parse_obj_coordinate(parts[2], "y", line)?,
                parse_obj_coordinate(parts[3], "z", line)?,
            ));
        }
    }

    Ok(PointCloud::new(points))
}

/// Parse one OBJ coordinate, rejecting non-finite values.
///
/// `f32::from_str` accepts "NaN", "inf" and "infinity": valid IEEE-754
/// spellings, not syntax errors. A single malformed coordinate therefore
/// produced a point at infinity that is indistinguishable from real data once
/// it is inside a `PointCloud`, and every bound, centroid and RANSAC threshold
/// computed from the cloud is then garbage.
fn parse_obj_coordinate(token: &str, axis: &str, line: &str) -> Result<f32> {
    let value: f32 = token
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid {axis} coordinate: {token} in {line:?}")))?;
    if !value.is_finite() {
        return Err(Error::ParseError(format!(
            "Non-finite {axis} coordinate: {token} in {line:?}"
        )));
    }
    Ok(value)
}

/// Write point cloud to OBJ format (vertex positions only)
pub fn write_obj<W: Write>(writer: &mut W, cloud: &PointCloud) -> Result<()> {
    for point in &cloud.points {
        writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
    }
    Ok(())
}

/// Mesh data structure for OBJ with faces (supports polygons, not just triangles)
#[derive(Debug, Clone)]
pub struct ObjMesh {
    pub vertices: Vec<Point3<f32>>,
    pub faces: Vec<Vec<usize>>, // Face indices (0-based), supports n-gons
}

impl ObjMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Convert to TriangleMesh (triangulates n-gons using fan triangulation)
    pub fn to_triangle_mesh(&self) -> TriangleMesh {
        let mut triangles: Vec<[usize; 3]> = Vec::new();

        for face in &self.faces {
            if face.len() >= 3 {
                // Fan triangulation for n-gons
                for i in 1..(face.len() - 1) {
                    triangles.push([face[0], face[i], face[i + 1]]);
                }
            }
        }

        TriangleMesh::with_vertices_and_faces(self.vertices.clone(), triangles)
    }

    /// Read a mesh with faces from OBJ
    pub fn read<R: BufRead>(reader: R) -> Result<Self> {
        let mut mesh = Self::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(kw) = unspaced_keyword(line) {
                return Err(Error::ParseError(format!(
                    "OBJ statement '{kw}' is not followed by a space: {line:?}"
                )));
            }

            if starts_with_keyword(line, "v") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    return Err(Error::ParseError(format!(
                        "OBJ vertex line needs 3 coordinates, got {}: {line:?}",
                        parts.len().saturating_sub(1)
                    )));
                }
                let x = parse_obj_coordinate(parts[1], "x", line)?;
                let y = parse_obj_coordinate(parts[2], "y", line)?;
                let z = parse_obj_coordinate(parts[3], "z", line)?;
                mesh.vertices.push(Point3::new(x, y, z));
            } else if starts_with_keyword(line, "f") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                // A face needs at least three corners. The old
                // `parts.len() >= 4` guard dropped a shorter one silently, so
                // the mesh simply came out with fewer faces than the file
                // declares and nothing said which line went missing.
                if parts.len() < 4 {
                    return Err(Error::ParseError(format!(
                        "OBJ face line needs at least 3 indices, got {}: {line:?}",
                        parts.len().saturating_sub(1)
                    )));
                }
                // Parse face indices (handle v/vt/vn format)
                let face: Vec<usize> = parts[1..]
                    .iter()
                    .map(|p| {
                        let idx_str = p.split('/').next().unwrap_or(p);
                        let i = idx_str.parse::<usize>().map_err(|_| {
                            Error::ParseError(format!("Invalid face index: {}", p))
                        })?;
                        if i == 0 {
                            return Err(Error::ParseError(format!(
                                "OBJ face index is 1-based, got 0 in: {}",
                                p
                            )));
                        }
                        let idx = i - 1; // OBJ uses 1-based indexing
                        // Bound-check against the vertices read *so far*.
                        // Nothing else does: `ObjMesh` and `TriangleMesh` both
                        // store faces as `usize`, so a file-controlled index
                        // that addresses no vertex survives into a mesh whose
                        // every consumer indexes `vertices[face[k]]` - ICP,
                        // normals, rasterisation - and panics or reads
                        // unrelated memory. That includes a *forward*
                        // reference, which OBJ does not allow, and an index
                        // near `usize::MAX`.
                        if idx >= mesh.vertices.len() {
                            return Err(Error::ParseError(format!(
                                "OBJ face index {i} is out of range: only {} vertex/vertices \
                                 have been read so far (in: {line:?})",
                                mesh.vertices.len()
                            )));
                        }
                        Ok(idx)
                    })
                    .collect::<Result<Vec<_>>>()?;
                mesh.faces.push(face);
            }
        }

        Ok(mesh)
    }

    /// Write mesh to OBJ format
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        for v in &self.vertices {
            writeln!(writer, "v {} {} {}", v.x, v.y, v.z)?;
        }

        for face in &self.faces {
            write!(writer, "f")?;
            for &idx in face {
                // OBJ uses 1-based indexing
                write!(writer, " {}", idx + 1)?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }
}

impl Default for ObjMesh {
    fn default() -> Self {
        Self::new()
    }
}
