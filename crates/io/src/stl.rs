//! STL (STereoLithography) I/O
//!
//! STL is a common format for 3D printing and CAD.

use crate::mesh::TriangleMesh;
use crate::Result;
use cv_core::Error;
use nalgebra::Point3;
use std::io::{BufRead, Write};

/// Read an STL file (ASCII or Binary)
pub fn read_stl<R: BufRead>(mut reader: R) -> Result<TriangleMesh> {
    // Try to detect format by reading first 80 bytes
    let mut header = [0u8; 80];
    let bytes_read = reader.read(&mut header)?;

    // Check for ASCII STL signature
    let header_str = String::from_utf8_lossy(&header[..bytes_read]);
    let mut is_ascii = header_str.trim_start().starts_with("solid");

    if is_ascii {
        // Many binary STL exporters begin the 80-byte header with "solid"
        // despite the spec forbidding it. Genuine ASCII files always contain
        // "facet normal" (or at least "endsolid") shortly after, so look for
        // those markers before committing to the text parser.
        let marker_in_header = header_str.contains("facet") || header_str.contains("endsolid");
        let marker_in_peek = if marker_in_header {
            false
        } else {
            let peek = reader.fill_buf()?;
            let peek_str = String::from_utf8_lossy(&peek[..peek.len().min(512)]);
            peek_str.contains("facet") || peek_str.contains("endsolid")
        };
        is_ascii = marker_in_header || marker_in_peek;
    }

    if is_ascii {
        // ASCII format
        // Prepend the header bytes already consumed, then read the rest
        //
        // The header must be decoded as UTF-8 *strictly*. It used to go through
        // `from_utf8_lossy`, which silently replaced every invalid byte with
        // U+FFFD and then parsed the result as text - so a file that is not
        // valid UTF-8 at all was accepted and its mangled bytes were discarded
        // as if they were a comment. The remaining bytes are read the same way
        // (`read_to_string` rejects invalid UTF-8), so this only closes the gap
        // the 80-byte header read had opened.
        let header_prefix = std::str::from_utf8(&header[..bytes_read])
            .map_err(|e| Error::ParseError(format!("STL ASCII header is not valid UTF-8: {e}")))?;
        let mut rest = String::new();
        reader.read_to_string(&mut rest)?;
        let content = header_prefix.to_owned() + &rest;
        parse_ascii_stl(&content)
    } else {
        // Binary format
        parse_binary_stl(&header, reader)
    }
}

fn parse_ascii_stl(content: &str) -> Result<TriangleMesh> {
    let mut vertices: Vec<Point3<f32>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut saw_endsolid = false;
    let mut loop_open = false;
    let mut loop_vertices: Vec<Point3<f32>> = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("solid ") || line == "solid" {
            i += 1;
            continue;
        }

        if line.starts_with("facet normal") {
            i += 1;
            continue;
        }

        if line.starts_with("outer loop") {
            if loop_open {
                return Err(Error::ParseError(
                    "STL ASCII: 'outer loop' inside an open loop".to_string(),
                ));
            }
            loop_open = true;
            loop_vertices.clear();
            i += 1;
            continue;
        }

        if line.starts_with("vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // A `vertex` line must carry exactly three coordinates. The old
            // `parts.len() >= 4` guard dropped a short line *silently*, which
            // did not lose the vertex - it shifted every later vertex down by
            // one, so the following `endloop` closed a triangle over three
            // unrelated vertices and the mesh was quietly wrong.
            if parts.len() < 4 {
                return Err(Error::ParseError(format!(
                    "STL ASCII: vertex line needs 3 coordinates, got {}: {line:?}",
                    parts.len().saturating_sub(1)
                )));
            }
            let x = parse_stl_coordinate(parts[1], line)?;
            let y = parse_stl_coordinate(parts[2], line)?;
            let z = parse_stl_coordinate(parts[3], line)?;
            let point = Point3::new(x, y, z);
            loop_vertices.push(point);
            vertices.push(point);
            i += 1;
            continue;
        }

        if line.starts_with("endloop") {
            if !loop_open {
                return Err(Error::ParseError(
                    "STL ASCII: 'endloop' without an 'outer loop'".to_string(),
                ));
            }
            // A triangle is exactly three vertices. Any other count means the
            // facet was truncated or malformed, and emitting a face anyway (as
            // the old `vertices.len() >= 3` over the whole file did) paired
            // whatever three vertices happened to be lying around.
            if loop_vertices.len() != 3 {
                return Err(Error::ParseError(format!(
                    "STL ASCII: a facet must have exactly 3 vertices, got {}",
                    loop_vertices.len()
                )));
            }
            let n = vertices.len();
            faces.push([n - 3, n - 2, n - 1]);
            loop_open = false;
            loop_vertices.clear();
            i += 1;
            continue;
        }

        if line.starts_with("endfacet") {
            if loop_open {
                return Err(Error::ParseError(
                    "STL ASCII: 'endfacet' inside an open loop".to_string(),
                ));
            }
            i += 1;
            continue;
        }

        if line.starts_with("endsolid") {
            saw_endsolid = true;
            break;
        }

        i += 1;
    }

    // A file that stops in the middle of a facet - no `endloop`, no `endfacet`,
    // no `endsolid` - used to return `Ok` with the three vertices it had read
    // and *no* faces, so a caller saw an empty mesh rather than a failure. An
    // unfinished loop is the clearest signal of truncation, so it is reported
    // first; an `endsolid` in the middle of a facet is reported after it.
    if loop_open {
        return Err(Error::ParseError(
            "STL ASCII: unexpected EOF inside a loop (no 'endloop')".to_string(),
        ));
    }

    // `endsolid` is the only terminator the format has, so its absence is
    // truncation - and it is the *only* signal for a file cut cleanly *between*
    // two facets, where every other invariant still holds:
    //
    //     solid x / facet / outer loop / 3 vertices / endloop / endfacet / <EOF>
    //
    // is byte-for-byte a complete one-facet mesh apart from the missing
    // `endsolid`, and the old check could not tell them apart.
    //
    // A binary STL declaring `triangle_count = 10 000 000` while carrying one
    // triangle reads the same way: declared and actual agree with what was
    // found, and only the terminator's absence says the file is incomplete.
    if !saw_endsolid {
        return Err(Error::ParseError(format!(
            "STL ASCII: unexpected EOF: no 'endsolid' terminator after {} facets",
            faces.len()
        )));
    }

    // Declared-vs-actual, independent of whether a `facet normal` line was ever
    // seen. The old guard was `if saw_facet && faces.len() * 3 != vertices.len()`,
    // so `saw_facet` was never actually required and the invariant was checked
    // only on files that happened to spell out their normals. That let three
    // orphan `vertex` lines - no facet, no loop - through as a mesh with zero
    // faces.
    if faces.len() * 3 != vertices.len() {
        return Err(Error::ParseError(format!(
            "STL ASCII: {} vertices for {} closed facets",
            vertices.len(),
            faces.len()
        )));
    }

    Ok(TriangleMesh::with_vertices_and_faces(vertices, faces))
}

/// Parse one STL coordinate, rejecting non-finite values.
///
/// `f32::from_str` accepts "NaN", "inf" and "infinity" - valid IEEE-754
/// spellings, not syntax errors - so a single malformed coordinate used to
/// produce a vertex at infinity, which poisons every bound, normal and
/// distance computed from the mesh instead of being reported here.
fn parse_stl_coordinate(token: &str, line: &str) -> Result<f32> {
    let value: f32 = token
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid coordinate: {token} in {line:?}")))?;
    if !value.is_finite() {
        return Err(Error::ParseError(format!(
            "Non-finite coordinate: {token} in {line:?}"
        )));
    }
    Ok(value)
}

fn parse_binary_stl<R: BufRead>(_header: &[u8; 80], mut reader: R) -> Result<TriangleMesh> {
    // Header is already read
    let mut vertices: Vec<Point3<f32>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Read triangle count (u32, little endian)
    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let triangle_count = u32::from_le_bytes(count_bytes) as usize;

    for _ in 0..triangle_count {
        // Each triangle: normal (3 floats), vertices (9 floats), attribute (2 bytes)
        let mut triangle_data = [0u8; 50]; // 12 * 4 + 2
        reader.read_exact(&mut triangle_data)?;

        // Parse vertices (skip normal)
        let mut float_bytes = [0u8; 4];

        for v in 0..3 {
            let offset = 12 + v * 12; // Skip normal (12 bytes), then 3 floats per vertex

            float_bytes.copy_from_slice(&triangle_data[offset..offset + 4]);
            let x = f32::from_le_bytes(float_bytes);

            float_bytes.copy_from_slice(&triangle_data[offset + 4..offset + 8]);
            let y = f32::from_le_bytes(float_bytes);

            float_bytes.copy_from_slice(&triangle_data[offset + 8..offset + 12]);
            let z = f32::from_le_bytes(float_bytes);

            // The binary path decoded the floats with no validation at all, so
            // the NaN and infinity *bit patterns* a malformed (or hostile) file
            // carries went straight into the mesh. A vertex at infinity is not a
            // vertex: it poisons every normal, bound and distance derived from
            // the mesh instead of being reported where it occurred.
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                return Err(Error::ParseError(format!(
                    "STL binary: non-finite vertex {v} of triangle {}: ({x}, {y}, {z})",
                    faces.len()
                )));
            }

            vertices.push(Point3::new(x, y, z));
        }

        let n = vertices.len();
        faces.push([n - 3, n - 2, n - 1]);
    }

    Ok(TriangleMesh::with_vertices_and_faces(vertices, faces))
}

/// Write mesh to ASCII STL format
pub fn write_stl_ascii<W: Write>(writer: &mut W, mesh: &TriangleMesh) -> Result<()> {
    writeln!(writer, "solid model")?;

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0]];
        let v1 = mesh.vertices[face[1]];
        let v2 = mesh.vertices[face[2]];

        // Compute face normal
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(&e2).normalize();

        writeln!(
            writer,
            "  facet normal {} {} {}",
            normal.x, normal.y, normal.z
        )?;
        writeln!(writer, "    outer loop")?;
        writeln!(writer, "      vertex {} {} {}", v0.x, v0.y, v0.z)?;
        writeln!(writer, "      vertex {} {} {}", v1.x, v1.y, v1.z)?;
        writeln!(writer, "      vertex {} {} {}", v2.x, v2.y, v2.z)?;
        writeln!(writer, "    endloop")?;
        writeln!(writer, "  endfacet")?;
    }

    writeln!(writer, "endsolid model")?;
    Ok(())
}

/// Write mesh to Binary STL format
pub fn write_stl_binary<W: Write>(writer: &mut W, mesh: &TriangleMesh) -> Result<()> {
    // Write 80-byte header
    let mut header = [0u8; 80];
    let msg = b"Binary STL generated by rust-cv-native";
    header[..msg.len()].copy_from_slice(msg);
    writer.write_all(&header)?;

    // Write triangle count (u32, little endian)
    let triangle_count = mesh.faces.len() as u32;
    writer.write_all(&triangle_count.to_le_bytes())?;

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0]];
        let v1 = mesh.vertices[face[1]];
        let v2 = mesh.vertices[face[2]];

        // Compute face normal
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(&e2).normalize();

        // Write normal (3 floats)
        writer.write_all(&normal.x.to_le_bytes())?;
        writer.write_all(&normal.y.to_le_bytes())?;
        writer.write_all(&normal.z.to_le_bytes())?;

        // Write vertices (9 floats)
        writer.write_all(&v0.x.to_le_bytes())?;
        writer.write_all(&v0.y.to_le_bytes())?;
        writer.write_all(&v0.z.to_le_bytes())?;
        writer.write_all(&v1.x.to_le_bytes())?;
        writer.write_all(&v1.y.to_le_bytes())?;
        writer.write_all(&v1.z.to_le_bytes())?;
        writer.write_all(&v2.x.to_le_bytes())?;
        writer.write_all(&v2.y.to_le_bytes())?;
        writer.write_all(&v2.z.to_le_bytes())?;

        // Write attribute byte count (u16, always 0)
        writer.write_all(&[0u8; 2])?;
    }

    Ok(())
}

/// Write mesh to STL format (auto-detects ASCII vs Binary based on extension preference)
pub fn write_stl<W: Write>(writer: &mut W, mesh: &TriangleMesh, binary: bool) -> Result<()> {
    if binary {
        write_stl_binary(writer, mesh)
    } else {
        write_stl_ascii(writer, mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_stl_ascii_write() {
        let mut mesh = TriangleMesh::new();
        mesh.vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        mesh.faces = vec![[0, 1, 2]];

        let mut buffer = Vec::new();
        write_stl_ascii(&mut buffer, &mesh).expect("Write ASCII failed");

        // Verify output is not empty and contains expected markers
        assert!(!buffer.is_empty());
        let content = String::from_utf8_lossy(&buffer);
        assert!(content.contains("solid"));
        assert!(content.contains("vertex"));
    }

    #[test]
    fn test_stl_ascii_detect_format() {
        let ascii_content = "solid test\nfacet normal 0 0 1\nouter loop\n\
                             vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n\
                             endloop\nendfacet\nendsolid test\n";
        let cursor = Cursor::new(ascii_content.as_bytes().to_vec());
        let result = read_stl(cursor);

        assert!(result.is_ok());
    }

    #[test]
    fn test_stl_multiple_facets_write() {
        let mut mesh = TriangleMesh::new();
        mesh.vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        mesh.faces = vec![[0, 1, 2], [1, 2, 3]];

        let mut buffer = Vec::new();
        write_stl_ascii(&mut buffer, &mesh).expect("Write failed");

        // Verify that multiple facets are written to buffer
        let content = String::from_utf8_lossy(&buffer);
        // Count the number of "facet" occurrences to verify multiple facets
        let facet_count = content.matches("facet").count();
        // Should have at least 2 facet entries for 2 triangles
        assert!(facet_count >= 2);
    }

    #[test]
    fn test_stl_empty_mesh() {
        let mesh = TriangleMesh::new();

        let mut buffer = Vec::new();
        write_stl_ascii(&mut buffer, &mesh).expect("Write failed");

        let cursor = Cursor::new(buffer);
        let loaded = read_stl(cursor).expect("Read failed");

        assert_eq!(loaded.faces.len(), 0);
    }

    #[test]
    fn test_stl_write_unified_ascii() {
        let mut mesh = TriangleMesh::new();
        mesh.vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        mesh.faces = vec![[0, 1, 2]];

        let mut buffer = Vec::new();
        write_stl(&mut buffer, &mesh, false).expect("Write failed");

        let written = String::from_utf8(buffer).expect("UTF-8 conversion failed");
        assert!(written.contains("solid"));
        assert!(written.contains("facet"));
    }

    #[test]
    fn test_stl_ascii_normal_computation() {
        let mut mesh = TriangleMesh::new();
        // Triangle in XY plane
        mesh.vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        mesh.faces = vec![[0, 1, 2]];

        let mut buffer = Vec::new();
        write_stl_ascii(&mut buffer, &mesh).expect("Write failed");

        let written = String::from_utf8(buffer).expect("UTF-8 conversion failed");
        // Should compute normal as (0, 0, 1) for XY plane triangle
        assert!(written.contains("facet normal"));
    }
}
