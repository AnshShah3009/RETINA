//! PCD (Point Cloud Data) I/O
//!
//! PCD is the native format for Point Cloud Library (PCL).

use crate::Result;
use cv_core::point_cloud::PointCloud;
use cv_core::Error;
use nalgebra::{Point3, Vector3};
use std::io::{BufRead, Write};

/// PCD data format
#[derive(Debug, Clone)]
pub enum PcdData {
    Ascii,
    Binary,
    BinaryCompressed,
}

/// Read a PCD file
pub fn read_pcd<R: BufRead>(reader: R) -> Result<PointCloud> {
    let mut lines = reader.lines();

    // Parse header
    let mut _version = String::new();
    let mut fields: Vec<String> = Vec::new();
    let mut _sizes: Vec<usize> = Vec::new();
    let mut _types: Vec<char> = Vec::new();
    let mut _counts: Vec<usize> = Vec::new();
    let mut width = 0;
    let mut height = 0;
    let mut viewpoint = [0.0f32; 7]; // tx, ty, tz, qw, qx, qy, qz
    let mut points_count = 0;
    let mut data_format = PcdData::Ascii;

    let mut in_header = true;

    while in_header {
        let line = lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in header".to_string()))??;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "VERSION" => {
                _version = parts.get(1).unwrap_or(&"0.7").to_string();
            }
            "FIELDS" => {
                fields = parts[1..].iter().map(|s| s.to_string()).collect();
            }
            "SIZE" => {
                _sizes = parts[1..].iter().map(|s| s.parse().unwrap_or(4)).collect();
            }
            "TYPE" => {
                _types = parts[1..].iter().filter_map(|s| s.chars().next()).collect();
            }
            "COUNT" => {
                _counts = parts[1..].iter().map(|s| s.parse().unwrap_or(1)).collect();
            }
            "WIDTH" => {
                width = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "HEIGHT" => {
                height = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "VIEWPOINT" => {
                for (i, &val) in parts[1..].iter().enumerate().take(7) {
                    viewpoint[i] = val.parse().unwrap_or(0.0);
                }
            }
            "POINTS" => {
                points_count = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "DATA" => {
                data_format = match parts.get(1).copied() {
                    Some("binary") => PcdData::Binary,
                    Some("binary_compressed") => PcdData::BinaryCompressed,
                    _ => PcdData::Ascii,
                };
                in_header = false;
            }
            _ => {}
        }
    }

    if points_count == 0 {
        points_count = width * height;
    }

    // Parse data
    match data_format {
        PcdData::Ascii => parse_pcd_ascii(lines, points_count, &fields),
        PcdData::Binary => Err(Error::InvalidInput(
            "Binary PCD not yet implemented".to_string(),
        )),
        PcdData::BinaryCompressed => Err(Error::InvalidInput(
            "Binary compressed PCD not yet implemented".to_string(),
        )),
    }
}

fn parse_pcd_ascii<I>(lines: I, count: usize, fields: &[String]) -> Result<PointCloud>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let mut points = Vec::with_capacity(count);
    let mut normals: Option<Vec<Vector3<f32>>> = None;
    let mut colors: Option<Vec<Point3<f32>>> = None;

    // Check for normal and color fields
    let has_normals =
        fields.contains(&"normal_x".to_string()) || fields.contains(&"nx".to_string());
    let has_colors = fields.contains(&"rgb".to_string())
        || fields.contains(&"rgba".to_string())
        || (fields.contains(&"r".to_string())
            && fields.contains(&"g".to_string())
            && fields.contains(&"b".to_string()));

    if has_normals {
        normals = Some(Vec::with_capacity(count));
    }
    if has_colors {
        colors = Some(Vec::with_capacity(count));
    }

    // Get field indices
    let x_idx = fields.iter().position(|f| f == "x").unwrap_or(0);
    let y_idx = fields.iter().position(|f| f == "y").unwrap_or(1);
    let z_idx = fields.iter().position(|f| f == "z").unwrap_or(2);

    let nx_idx = fields.iter().position(|f| f == "normal_x" || f == "nx");
    let ny_idx = fields.iter().position(|f| f == "normal_y" || f == "ny");
    let nz_idx = fields.iter().position(|f| f == "normal_z" || f == "nz");

    let rgb_idx = fields.iter().position(|f| f == "rgb" || f == "rgba");
    let r_idx = fields.iter().position(|f| f == "r");
    let g_idx = fields.iter().position(|f| f == "g");
    let b_idx = fields.iter().position(|f| f == "b");

    for line in lines {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let values: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();

        if values.len() < 3 {
            continue;
        }

        // Read point
        let x = values.get(x_idx).copied().unwrap_or(0.0);
        let y = values.get(y_idx).copied().unwrap_or(0.0);
        let z = values.get(z_idx).copied().unwrap_or(0.0);
        points.push(Point3::new(x, y, z));

        // Read normals
        if let Some(ref mut n) = normals {
            let nx = nx_idx.and_then(|i| values.get(i)).copied().unwrap_or(0.0);
            let ny = ny_idx.and_then(|i| values.get(i)).copied().unwrap_or(0.0);
            let nz = nz_idx.and_then(|i| values.get(i)).copied().unwrap_or(0.0);
            n.push(Vector3::new(nx, ny, nz));
        }

        // Read colors
        if let Some(ref mut c) = colors {
            if let Some(idx) = rgb_idx {
                // Packed RGB/RGBA: the float is a bit-reinterpreted u32, not a numeric value
                let float_value = values.get(idx).copied().unwrap_or(0.0);
                let packed: u32 = float_value.to_bits();
                let r = ((packed >> 16) & 0xFF) as f32 / 255.0;
                let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
                let b = (packed & 0xFF) as f32 / 255.0;
                c.push(Point3::new(r, g, b));
            } else if let (Some(ri), Some(gi), Some(bi)) = (r_idx, g_idx, b_idx) {
                // Separate R, G, B fields
                let r = values.get(ri).copied().unwrap_or(0.0);
                let g = values.get(gi).copied().unwrap_or(0.0);
                let b = values.get(bi).copied().unwrap_or(0.0);

                // Assume 0-255 range if values are large
                let r_norm = if r > 1.0 { r / 255.0 } else { r };
                let g_norm = if g > 1.0 { g / 255.0 } else { g };
                let b_norm = if b > 1.0 { b / 255.0 } else { b };

                c.push(Point3::new(r_norm, g_norm, b_norm));
            }
        }

        if points.len() >= count {
            break;
        }
    }

    let mut cloud = PointCloud::new(points);
    cloud.normals = normals;
    cloud.colors = colors;

    Ok(cloud)
}

/// Write point cloud to PCD format (ASCII)
pub fn write_pcd<W: Write>(writer: &mut W, cloud: &PointCloud) -> Result<()> {
    let num_points = cloud.len();
    let has_normals = cloud.normals.is_some();
    let has_colors = cloud.colors.is_some();

    // Write header
    writeln!(writer, "# .PCD v0.7 - Point Cloud Data file format")?;
    writeln!(writer, "VERSION 0.7")?;

    // Fields
    write!(writer, "FIELDS x y z")?;
    if has_normals {
        write!(writer, " normal_x normal_y normal_z")?;
    }
    if has_colors {
        write!(writer, " rgb")?;
    }
    writeln!(writer)?;

    // Sizes
    write!(writer, "SIZE 4 4 4")?;
    if has_normals {
        write!(writer, " 4 4 4")?;
    }
    if has_colors {
        write!(writer, " 4")?;
    }
    writeln!(writer)?;

    // Types
    write!(writer, "TYPE F F F")?;
    if has_normals {
        write!(writer, " F F F")?;
    }
    if has_colors {
        write!(writer, " F")?;
    }
    writeln!(writer)?;

    // Count
    write!(writer, "COUNT 1 1 1")?;
    if has_normals {
        write!(writer, " 1 1 1")?;
    }
    if has_colors {
        write!(writer, " 1")?;
    }
    writeln!(writer)?;

    writeln!(writer, "WIDTH {}", num_points)?;
    writeln!(writer, "HEIGHT 1")?;
    writeln!(writer, "VIEWPOINT 0 0 0 1 0 0 0")?;
    writeln!(writer, "POINTS {}", num_points)?;
    writeln!(writer, "DATA ascii")?;

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
            let r = (c.x.clamp(0.0, 1.0) * 255.0) as u32;
            let g = (c.y.clamp(0.0, 1.0) * 255.0) as u32;
            let b = (c.z.clamp(0.0, 1.0) * 255.0) as u32;
            let packed: u32 = (r << 16) | (g << 8) | b;
            write!(writer, " {}", packed)?;
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
    fn test_pcd_round_trip_basic() {
        let cloud = PointCloud::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, -2.0, -3.0),
        ]);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert_eq!(read_cloud.len(), 3);
        assert!((read_cloud.points[0].x - 0.0).abs() < 0.001);
        assert!((read_cloud.points[1].y - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_pcd_round_trip_with_normals() {
        let mut cloud =
            PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
        cloud.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
        ]);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert!(read_cloud.normals.is_some());
        let normals = read_cloud.normals.unwrap();
        assert_eq!(normals.len(), 2);
    }

    #[test]
    fn test_pcd_round_trip_with_colors() {
        let mut cloud = PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0)]);
        cloud.colors = Some(vec![Point3::new(1.0, 0.5, 0.0)]);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert!(read_cloud.colors.is_some());
    }

    #[test]
    fn test_pcd_empty_cloud() {
        let cloud = PointCloud::new(vec![]);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert_eq!(read_cloud.len(), 0);
    }

    #[test]
    fn test_pcd_large_point_cloud() {
        let n = 500;
        let points: Vec<_> = (0..n)
            .map(|i| Point3::new(i as f32, i as f32 * 2.0, i as f32 * 3.0))
            .collect();
        let cloud = PointCloud::new(points);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert_eq!(read_cloud.len(), n);
        assert_eq!(read_cloud.points[100].x, 100.0);
    }
}
