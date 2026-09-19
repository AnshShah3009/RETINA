//! PLY (Polygon File Format) I/O
//!
//! PLY is a flexible format for storing 3D data with arbitrary properties.

use crate::Result;
use cv_core::point_cloud::PointCloud;
use cv_core::Error;
use nalgebra::{Point3, Vector3};
use std::io::{BufRead, Write};

/// Read a PLY file from a reader
pub fn read_ply<R: BufRead>(reader: R) -> Result<PointCloud> {
    let mut lines = reader.lines();

    // Parse header
    let mut in_header = true;
    let mut format = String::new();
    let mut num_vertices = 0usize;
    // Vertex properties in declared order — PLY allows any property order,
    // so data must be indexed by name rather than assumed position.
    let mut props: Vec<String> = Vec::new();
    let mut in_vertex_element = false;

    while in_header {
        let line = lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in header".to_string()))??;

        let line = line.trim();

        if line.starts_with("format ") {
            format = line
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| Error::ParseError("Invalid format line".to_string()))?
                .to_string();
        } else if line.starts_with("element ") {
            // A new element switches the property scope; only vertex
            // properties are relevant here.
            in_vertex_element = line.starts_with("element vertex");
            if in_vertex_element {
                num_vertices = line
                    .split_whitespace()
                    .nth(2)
                    .ok_or_else(|| Error::ParseError("Invalid vertex count".to_string()))?
                    .parse()
                    .map_err(|_| Error::ParseError("Invalid vertex count number".to_string()))?;
            }
        } else if in_vertex_element && line.starts_with("property ") {
            let name = line
                .split_whitespace()
                .last()
                .ok_or_else(|| Error::ParseError("Invalid property line".to_string()))?
                .to_string();
            props.push(name);
        } else if line == "end_header" {
            in_header = false;
        }
    }

    if format != "ascii" {
        return Err(Error::InvalidInput(format!(
            "PLY format '{}' not supported, only ASCII",
            format
        )));
    }

    let pos_of = |names: &[&str]| -> Option<usize> {
        props.iter().position(|p| names.contains(&p.as_str()))
    };

    let xi = pos_of(&["x"]).ok_or_else(|| Error::ParseError("PLY: missing x property".to_string()))?;
    let yi = pos_of(&["y"]).ok_or_else(|| Error::ParseError("PLY: missing y property".to_string()))?;
    let zi = pos_of(&["z"]).ok_or_else(|| Error::ParseError("PLY: missing z property".to_string()))?;

    let nxi = pos_of(&["nx", "normal_x"]);
    let nyi = pos_of(&["ny", "normal_y"]);
    let nzi = pos_of(&["nz", "normal_z"]);
    let has_normals = nxi.is_some() && nyi.is_some() && nzi.is_some();

    // Colors either as packed rgb/rgba float or as separate channels.
    let rgb_i = pos_of(&["rgb", "rgba"]);
    let ri = pos_of(&["r", "red"]);
    let gi = pos_of(&["g", "green"]);
    let bi = pos_of(&["b", "blue"]);
    let has_colors =
        rgb_i.is_some() || (ri.is_some() && gi.is_some() && bi.is_some());

    // Parse data
    let mut points = Vec::with_capacity(num_vertices);
    let mut colors = if has_colors {
        Some(Vec::with_capacity(num_vertices))
    } else {
        None
    };
    let mut normals = if has_normals {
        Some(Vec::with_capacity(num_vertices))
    } else {
        None
    };
    let width = props.len();

    for _ in 0..num_vertices {
        let line = lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in data".to_string()))??;

        let values: Vec<f32> = line
            .split_whitespace()
            .map(|s| {
                s.parse()
                    .map_err(|_| Error::ParseError(format!("Invalid number: {}", s)))
            })
            .collect::<Result<Vec<_>>>()?;

        if values.len() < width.max(3) {
            return Err(Error::InvalidInput(
                "Not enough values for vertex".to_string(),
            ));
        }

        points.push(Point3::new(values[xi], values[yi], values[zi]));

        if has_normals {
            normals.as_mut().unwrap().push(Vector3::new(
                values[nxi.unwrap()],
                values[nyi.unwrap()],
                values[nzi.unwrap()],
            ));
        }

        if let Some(packed_idx) = rgb_i {
            // PLY rgb is stored as a float whose bit pattern packs u8 channels.
            let packed: u32 = values[packed_idx].to_bits();
            let r = ((packed >> 16) & 0xFF) as f32 / 255.0;
            let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
            let b = (packed & 0xFF) as f32 / 255.0;
            colors.as_mut().unwrap().push(Point3::new(r, g, b));
        } else if has_colors {
            let r = values[ri.unwrap()];
            let g = values[gi.unwrap()];
            let b = values[bi.unwrap()];
            // Normalize if stored as 0-255.
            let norm = |v: f32| if v > 1.0 { v / 255.0 } else { v };
            colors
                .as_mut()
                .unwrap()
                .push(Point3::new(norm(r), norm(g), norm(b)));
        }
    }

    let mut pc = PointCloud::new(points);

    // Attach optional attributes only when complete — an empty or partial
    // vector would desynchronize downstream consumers that index by point.
    if let Some(c) = colors {
        if c.len() == pc.len() {
            pc.colors = Some(c);
        }
    }

    if let Some(n) = normals {
        if n.len() == pc.len() {
            pc.normals = Some(n);
        }
    }

    Ok(pc)
}

/// Write a point cloud to PLY format
pub fn write_ply<W: Write>(writer: &mut W, cloud: &PointCloud) -> Result<()> {
    let num_points = cloud.len();
    let has_colors = cloud.colors.is_some();
    let has_normals = cloud.normals.is_some();

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "element vertex {}", num_points)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if has_colors {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
    }

    writeln!(writer, "end_header")?;

    // Write data
    for i in 0..num_points {
        let p = cloud.points[i];
        write!(writer, "{} {} {}", p.x, p.y, p.z)?;

        if let Some(ref normals) = cloud.normals {
            let n = normals[i];
            write!(writer, " {} {} {}", n.x, n.y, n.z)?;
        }

        if let Some(ref colors) = cloud.colors {
            let c = colors[i];
            let r = (c.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (c.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (c.z.clamp(0.0, 1.0) * 255.0) as u8;
            write!(writer, " {} {} {}", r, g, b)?;
        }

        writeln!(writer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_ply_round_trip_basic() {
        let cloud = PointCloud::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, -2.0, -3.0),
        ]);

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_ply(reader).expect("read failed");

        assert_eq!(read_cloud.len(), 3);
        assert!((read_cloud.points[1].y - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_ply_round_trip_with_normals() {
        let mut cloud =
            PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
        cloud.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
        ]);

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_ply(reader).expect("read failed");

        assert!(read_cloud.normals.is_some());
    }

    #[test]
    fn test_ply_empty_cloud() {
        let cloud = PointCloud::new(vec![]);

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_ply(reader).expect("read failed");

        assert_eq!(read_cloud.len(), 0);
    }

    #[test]
    fn test_ply_large_point_cloud() {
        let n = 300;
        let points: Vec<_> = (0..n)
            .map(|i| Point3::new(i as f32, i as f32 * 2.0, i as f32 * 3.0))
            .collect();
        let cloud = PointCloud::new(points);

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_ply(reader).expect("read failed");

        assert_eq!(read_cloud.len(), n);
    }
}
