//! OBJ (Wavefront Object) I/O
//!
//! OBJ is a common format for storing 3D mesh geometry.

use crate::mesh::TriangleMesh;
use crate::Result;
use cv_core::point_cloud::PointCloud;
use cv_core::Error;
use nalgebra::Point3;
use std::io::{BufRead, Write};

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

        // Parse vertex lines (v x y z)
        if line.starts_with("v ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().map_err(|_| {
                    Error::ParseError(format!("Invalid x coordinate: {}", parts[1]))
                })?;
                let y: f32 = parts[2].parse().map_err(|_| {
                    Error::ParseError(format!("Invalid y coordinate: {}", parts[2]))
                })?;
                let z: f32 = parts[3].parse().map_err(|_| {
                    Error::ParseError(format!("Invalid z coordinate: {}", parts[3]))
                })?;

                points.push(Point3::new(x, y, z));
            }
        }
    }

    Ok(PointCloud::new(points))
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

            if line.starts_with("v ") {
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
                    mesh.vertices.push(Point3::new(x, y, z));
                }
            } else if line.starts_with("f ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    // Parse face indices (handle v/vt/vn format)
                    let face: Vec<usize> = parts[1..]
                        .iter()
                        .map(|p| {
                            let idx_str = p.split('/').next().unwrap_or(p);
                            idx_str
                                .parse::<usize>()
                                .map(|i| if i > 0 { i - 1 } else { 0 }) // OBJ uses 1-based indexing
                                .map_err(|_| {
                                    Error::ParseError(format!("Invalid face index: {}", p))
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    mesh.faces.push(face);
                }
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
