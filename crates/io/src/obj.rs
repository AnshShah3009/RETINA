//! OBJ (Wavefront Object) I/O
//!
//! OBJ is a common format for storing 3D mesh geometry.

use crate::Result;
use crate::mesh::TriangleMesh;
use cv_core::Error;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
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
    let keyword = KEYWORDS.iter().copied().find(|kw| line.starts_with(kw))?;
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
    let value: f32 = token.parse().map_err(|_| {
        Error::ParseError(format!("Invalid {axis} coordinate: {token} in {line:?}"))
    })?;
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

    /// Convert to TriangleMesh, triangulating n-gons by ear clipping.
    ///
    /// An OBJ face is an arbitrary planar polygon, and only *convex* polygons
    /// can be split by a fan from their first vertex. For a concave face the fan
    /// covers area the polygon does not contain, so the mesh describes the
    /// face's convex hull: a U-shaped face of area 10 (a 4x4 square with a 2x3
    /// notch) came back as 19, and a ray fired down the middle of the notch -
    /// where there is no surface at all - hit a triangle at z = 0.
    ///
    /// That is not visible in `ObjMesh` itself, which stores the face's own
    /// indices: the geometry goes wrong at the boundary where `cv-3d` measures
    /// and ray-casts the `TriangleMesh` this function produces.
    ///
    /// Ear clipping is used for every polygon with four or more corners. The
    /// face is projected onto the plane of its Newell normal (the best-fit
    /// plane, which is exact for a planar face and stable for the slightly
    /// non-planar ones exporters do emit) and clipped in 2-D, with the winding
    /// direction taken from the projected signed area so a clockwise face is
    /// triangulated exactly like a counter-clockwise one. Emitted triangles keep
    /// the face's own winding. A degenerate face - collinear corners, or a
    /// projection with zero area, where no ear exists - falls back to the fan,
    /// which at least preserves the previous behaviour for input that has no
    /// correct triangulation.
    pub fn to_triangle_mesh(&self) -> TriangleMesh {
        let mut triangles: Vec<[usize; 3]> = Vec::new();

        for face in &self.faces {
            if face.len() >= 3 {
                triangles.extend(triangulate_face(&self.vertices, face));
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
                        let i = idx_str
                            .parse::<i128>()
                            .map_err(|_| Error::ParseError(format!("Invalid face index: {}", p)))?;
                        if i == 0 {
                            return Err(Error::ParseError(format!(
                                "OBJ face index is 1-based, got 0 in: {}",
                                p
                            )));
                        }
                        // OBJ indices are 1-based, and a *negative* index is
                        // relative to the end of the vertex list read so far
                        // (`-1` is the latest vertex). Streaming exporters
                        // write `f -3 -2 -1`; it used to be rejected as an
                        // invalid index.
                        let idx = if i > 0 {
                            usize::try_from(i - 1).unwrap_or(usize::MAX)
                        } else {
                            match usize::try_from(i.unsigned_abs())
                                .ok()
                                .and_then(|back| mesh.vertices.len().checked_sub(back))
                            {
                                Some(idx) => idx,
                                None => {
                                    return Err(Error::ParseError(format!(
                                        "OBJ relative face index {i} reaches before the first \
                                         vertex: only {} read so far (in: {line:?})",
                                        mesh.vertices.len()
                                    )));
                                }
                            }
                        };
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

/// Triangulate one OBJ face into `[usize; 3]` triangles of `vertices`.
///
/// Ear clipping, as described on [`ObjMesh::to_triangle_mesh`]. Returns the
/// fan of the face whenever it cannot do better - a degenerate face, a repeated
/// index, an out-of-range index, or a polygon for which no ear exists within a
/// bounded number of passes - so the result is never worse than the fan that
/// was the only triangulation before, and never an infinite loop.
fn triangulate_face(vertices: &[Point3<f32>], face: &[usize]) -> Vec<[usize; 3]> {
    let fan = |face: &[usize]| -> Vec<[usize; 3]> {
        (1..face.len().saturating_sub(1))
            .map(|i| [face[0], face[i], face[i + 1]])
            .collect()
    };

    let n = face.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[face[0], face[1], face[2]]];
    }
    // `faces` is `pub`, so an index here need not have come from `read`'s
    // validated parse. `TriangleMesh` stores no bound, so an out-of-range index
    // would panic in the first consumer that indexes `vertices[face[k]]`.
    let mut pts: Vec<Point3<f32>> = Vec::with_capacity(n);
    for &i in face {
        match vertices.get(i) {
            Some(p) if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() => pts.push(*p),
            _ => return fan(face),
        }
    }
    // A repeated corner has no correct ear-clipping order; the fan at least
    // terminates.
    for i in 0..n {
        if face[i + 1..].contains(&face[i]) {
            return fan(face);
        }
    }

    // Newell's method: the best-fit plane normal, exact for a planar face.
    let mut normal: Vector3<f32> = Vector3::zeros();
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        normal.x += (a.y - b.y) * (a.z + b.z);
        normal.y += (a.z - b.z) * (a.x + b.x);
        normal.z += (a.x - b.x) * (a.y + b.y);
    }

    // Project onto the coordinate plane the normal faces least - the choice with
    // the largest projected area, so a face nearly parallel to a coordinate
    // plane keeps its shape and its winding.
    let (ax, ay) = {
        let (nx, ny, nz) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
        if nz >= nx && nz >= ny {
            (0usize, 1usize) // drop z
        } else if nx >= ny {
            (1, 2) // drop x
        } else {
            (2, 0) // drop y
        }
    };
    let coord = |p: &Point3<f32>, axis: usize| -> f32 {
        match axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        }
    };
    let flat: Vec<(f32, f32)> = pts.iter().map(|p| (coord(p, ax), coord(p, ay))).collect();

    // Signed area of the projection. Zero means the face has no in-plane
    // extent (collinear corners), so there is nothing to clip.
    let signed_area = {
        let mut s = 0.0f64;
        for i in 0..n {
            let (x0, y0) = flat[i];
            let (x1, y1) = flat[(i + 1) % n];
            s += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
        }
        s * 0.5
    };
    // Scale-relative, so a millimetre-unit face is judged the same as a
    // metre-unit one.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &flat {
        min_x = min_x.min(p.0);
        min_y = min_y.min(p.1);
        max_x = max_x.max(p.0);
        max_y = max_y.max(p.1);
    }
    let span = (max_x - min_x).max(max_y - min_y).abs();
    if !signed_area.is_finite() || signed_area.abs() <= 1e-12 * (span as f64) * (span as f64) {
        return fan(face);
    }
    // The sign of the projected area is the winding as seen in this projection,
    // which is all the clipping needs: a clockwise face then behaves exactly
    // like a counter-clockwise one.
    let orient = if signed_area > 0.0 { 1.0f64 } else { -1.0f64 };

    let cross2 = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| -> f64 {
        ((b.0 - a.0) as f64 * (c.1 - a.1) as f64) - ((b.1 - a.1) as f64 * (c.0 - a.0) as f64)
    };

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut out: Vec<[usize; 3]> = Vec::with_capacity(n - 2);
    while remaining.len() > 3 {
        let m = remaining.len();
        let mut clipped = false;
        for k in 0..m {
            let ip = remaining[(k + m - 1) % m];
            let ic = remaining[k];
            let inx = remaining[(k + 1) % m];
            let (a, b, c) = (flat[ip], flat[ic], flat[inx]);

            // Reflex (or collinear) corners are not ears.
            if cross2(a, b, c) * orient <= 0.0 {
                continue;
            }
            // No other remaining corner may lie inside the candidate triangle.
            let blocked = remaining.iter().any(|&j| {
                j != ip
                    && j != ic
                    && j != inx
                    && cross2(a, b, flat[j]) * orient >= 0.0
                    && cross2(b, c, flat[j]) * orient >= 0.0
                    && cross2(c, a, flat[j]) * orient >= 0.0
            });
            if blocked {
                continue;
            }

            out.push([face[ip], face[ic], face[inx]]);
            remaining.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            // No ear left, and the next pass would examine exactly the same
            // corners and reach the same conclusion: a self-intersecting or
            // otherwise non-simple face, for which there is no triangulation to
            // be found. Keep the fan rather than guess, and terminate.
            return fan(face);
        }
    }
    out.push([face[remaining[0]], face[remaining[1]], face[remaining[2]]]);
    out
}
