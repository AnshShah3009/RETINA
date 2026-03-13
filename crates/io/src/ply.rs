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
    let mut has_colors = false;
    let mut has_normals = false;
    let mut num_vertices = 0;

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
        } else if line.starts_with("element vertex ") {
            num_vertices = line
                .split_whitespace()
                .nth(2)
                .ok_or_else(|| Error::ParseError("Invalid vertex count".to_string()))?
                .parse()
                .map_err(|_| Error::ParseError("Invalid vertex count number".to_string()))?;
        } else if line.contains("property") && line.contains("red") {
            has_colors = true;
        } else if line.contains("property") && line.contains("nx") {
            has_normals = true;
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

        if values.len() < 3 {
            return Err(Error::InvalidInput(
                "Not enough values for vertex".to_string(),
            ));
        }

        points.push(Point3::new(values[0], values[1], values[2]));

        let mut idx = 3;

        if has_normals && values.len() >= idx + 3 {
            normals.as_mut().unwrap().push(Vector3::new(
                values[idx],
                values[idx + 1],
                values[idx + 2],
            ));
            idx += 3;
        }

        if has_colors && values.len() >= idx + 3 {
            let r = values[idx] / 255.0;
            let g = values[idx + 1] / 255.0;
            let b = values[idx + 2] / 255.0;
            colors.as_mut().unwrap().push(Point3::new(r, g, b));
        }
    }

    let mut pc = PointCloud::new(points);

    if let Some(c) = colors {
        pc.colors = Some(c);
    }

    if let Some(n) = normals {
        pc.normals = Some(n);
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
