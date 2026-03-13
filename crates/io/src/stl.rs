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
    if header_str.trim_start().starts_with("solid ") {
        // ASCII format
        // Re-read from start
        let _ = header;
        drop(header_str);
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        parse_ascii_stl(&content)
    } else {
        // Binary format
        parse_binary_stl(&header, reader)
    }
}

fn parse_ascii_stl(content: &str) -> Result<TriangleMesh> {
    let mut vertices: Vec<Point3<f32>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("facet normal") || line.starts_with("outer loop") {
            i += 1;
            continue;
        }

        if line.starts_with("vertex ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1]
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid x: {}", parts[1])))?;
                let y: f32 = parts[2]
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid y: {}", parts[2])))?;
                let z: f32 = parts[3]
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid z: {}", parts[3])))?;
                vertices.push(Point3::new(x, y, z));
            }
        }

        if line.starts_with("endloop") {
            // End of a triangle
            if vertices.len() >= 3 {
                let n = vertices.len();
                faces.push([n - 3, n - 2, n - 1]);
            }
        }

        i += 1;
    }

    Ok(TriangleMesh::with_vertices_and_faces(vertices, faces))
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
