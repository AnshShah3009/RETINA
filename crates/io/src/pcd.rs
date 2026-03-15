//! PCD (Point Cloud Data) I/O
//!
//! PCD is the native format for Point Cloud Library (PCL).
//! Supports ASCII and binary DATA formats.

use crate::Result;
use cv_core::point_cloud::PointCloud;
use cv_core::Error;
use nalgebra::{Point3, Vector3};
use std::io::{BufRead, Read, Write};

/// PCD data format
#[derive(Debug, Clone, PartialEq)]
pub enum PcdData {
    Ascii,
    Binary,
    BinaryCompressed,
}

/// Parsed PCD header
#[derive(Debug, Clone)]
struct PcdHeader {
    fields: Vec<String>,
    sizes: Vec<usize>,
    types: Vec<char>,
    counts: Vec<usize>,
    #[allow(dead_code)]
    width: usize,
    #[allow(dead_code)]
    height: usize,
    points_count: usize,
    data_format: PcdData,
}

impl PcdHeader {
    /// Total byte size of a single point record
    fn point_stride(&self) -> usize {
        self.sizes
            .iter()
            .zip(self.counts.iter())
            .map(|(s, c)| s * c)
            .sum()
    }
}

/// Read a PCD file (ASCII or binary)
pub fn read_pcd<R: BufRead>(mut reader: R) -> Result<PointCloud> {
    let header = parse_header(&mut reader)?;

    match header.data_format {
        PcdData::Ascii => {
            let lines = reader.lines();
            parse_pcd_ascii(lines, header.points_count, &header.fields)
        }
        PcdData::Binary => parse_pcd_binary(reader, &header),
        PcdData::BinaryCompressed => Err(Error::InvalidInput(
            "Binary compressed PCD requires LZF decompression which is not yet supported. \
             Use DATA ascii or DATA binary instead."
                .to_string(),
        )),
    }
}

/// Parse the PCD header from a reader, leaving the reader positioned right after
/// the DATA line (including its newline).
fn parse_header<R: BufRead>(reader: &mut R) -> Result<PcdHeader> {
    let mut _version = String::new();
    let mut fields: Vec<String> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut types: Vec<char> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut width = 0usize;
    let mut height = 0usize;
    let mut _viewpoint = [0.0f32; 7];
    let mut points_count = 0usize;
    let data_format;

    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        let bytes_read = reader.read_line(&mut line_buf)?;
        if bytes_read == 0 {
            return Err(Error::ParseError(
                "Unexpected EOF in PCD header".to_string(),
            ));
        }

        let line = line_buf.trim();
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
                sizes = parts[1..].iter().map(|s| s.parse().unwrap_or(4)).collect();
            }
            "TYPE" => {
                types = parts[1..].iter().filter_map(|s| s.chars().next()).collect();
            }
            "COUNT" => {
                counts = parts[1..].iter().map(|s| s.parse().unwrap_or(1)).collect();
            }
            "WIDTH" => {
                width = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "HEIGHT" => {
                height = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "VIEWPOINT" => {
                for (i, &val) in parts[1..].iter().enumerate().take(7) {
                    _viewpoint[i] = val.parse().unwrap_or(0.0);
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
                break;
            }
            _ => {}
        }
    }

    if points_count == 0 {
        points_count = width * height;
    }

    // Default counts to 1 if not specified
    if counts.is_empty() {
        counts = vec![1; fields.len()];
    }

    Ok(PcdHeader {
        fields,
        sizes,
        types,
        counts,
        width,
        height,
        points_count,
        data_format,
    })
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

/// Parse binary PCD data after the header has been read.
fn parse_pcd_binary<R: Read>(mut reader: R, header: &PcdHeader) -> Result<PointCloud> {
    let count = header.points_count;
    let stride = header.point_stride();

    if stride == 0 {
        return Err(Error::ParseError(
            "PCD binary: point stride is zero (missing SIZE info)".to_string(),
        ));
    }

    // Compute byte offsets for each field within a point record
    let field_offsets = compute_field_offsets(header);

    // Find field indices in the header
    let x_field = header.fields.iter().position(|f| f == "x");
    let y_field = header.fields.iter().position(|f| f == "y");
    let z_field = header.fields.iter().position(|f| f == "z");

    let nx_field = header
        .fields
        .iter()
        .position(|f| f == "normal_x" || f == "nx");
    let ny_field = header
        .fields
        .iter()
        .position(|f| f == "normal_y" || f == "ny");
    let nz_field = header
        .fields
        .iter()
        .position(|f| f == "normal_z" || f == "nz");

    let rgb_field = header.fields.iter().position(|f| f == "rgb" || f == "rgba");
    let r_field = header.fields.iter().position(|f| f == "r");
    let g_field = header.fields.iter().position(|f| f == "g");
    let b_field = header.fields.iter().position(|f| f == "b");

    let has_normals = nx_field.is_some() && ny_field.is_some() && nz_field.is_some();
    let has_rgb = rgb_field.is_some();
    let has_separate_rgb = r_field.is_some() && g_field.is_some() && b_field.is_some();
    let has_colors = has_rgb || has_separate_rgb;

    // Read all binary data at once
    let total_bytes = stride * count;
    let mut data = vec![0u8; total_bytes];
    reader.read_exact(&mut data).map_err(|e| {
        Error::ParseError(format!(
            "PCD binary: failed to read {} bytes of point data: {}",
            total_bytes, e
        ))
    })?;

    let mut points = Vec::with_capacity(count);
    let mut normals: Option<Vec<Vector3<f32>>> = if has_normals {
        Some(Vec::with_capacity(count))
    } else {
        None
    };
    let mut colors: Option<Vec<Point3<f32>>> = if has_colors {
        Some(Vec::with_capacity(count))
    } else {
        None
    };

    for i in 0..count {
        let base = i * stride;
        let point_data = &data[base..base + stride];

        // Read x, y, z
        let x = read_field_as_f32(
            point_data,
            &field_offsets,
            &header.sizes,
            &header.types,
            x_field.unwrap_or(0),
        );
        let y = read_field_as_f32(
            point_data,
            &field_offsets,
            &header.sizes,
            &header.types,
            y_field.unwrap_or(1),
        );
        let z = read_field_as_f32(
            point_data,
            &field_offsets,
            &header.sizes,
            &header.types,
            z_field.unwrap_or(2),
        );
        points.push(Point3::new(x, y, z));

        // Read normals
        if let Some(ref mut norms) = normals {
            let nx = read_field_as_f32(
                point_data,
                &field_offsets,
                &header.sizes,
                &header.types,
                nx_field.unwrap(),
            );
            let ny = read_field_as_f32(
                point_data,
                &field_offsets,
                &header.sizes,
                &header.types,
                ny_field.unwrap(),
            );
            let nz = read_field_as_f32(
                point_data,
                &field_offsets,
                &header.sizes,
                &header.types,
                nz_field.unwrap(),
            );
            norms.push(Vector3::new(nx, ny, nz));
        }

        // Read colors
        if let Some(ref mut cols) = colors {
            if let Some(idx) = rgb_field {
                // Packed RGB stored as float (bit-reinterpreted u32)
                let offset = field_offsets[idx];
                let size = header.sizes[idx];
                if size == 4 {
                    let bytes: [u8; 4] =
                        point_data[offset..offset + 4].try_into().unwrap_or([0; 4]);
                    let packed = u32::from_le_bytes(bytes);
                    let r = ((packed >> 16) & 0xFF) as f32 / 255.0;
                    let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
                    let b = (packed & 0xFF) as f32 / 255.0;
                    cols.push(Point3::new(r, g, b));
                }
            } else if let (Some(ri), Some(gi), Some(bi)) = (r_field, g_field, b_field) {
                let r =
                    read_field_as_f32(point_data, &field_offsets, &header.sizes, &header.types, ri);
                let g =
                    read_field_as_f32(point_data, &field_offsets, &header.sizes, &header.types, gi);
                let b =
                    read_field_as_f32(point_data, &field_offsets, &header.sizes, &header.types, bi);
                // Normalize if in 0-255 range
                let r_norm = if r > 1.0 { r / 255.0 } else { r };
                let g_norm = if g > 1.0 { g / 255.0 } else { g };
                let b_norm = if b > 1.0 { b / 255.0 } else { b };
                cols.push(Point3::new(r_norm, g_norm, b_norm));
            }
        }
    }

    let mut cloud = PointCloud::new(points);
    cloud.normals = normals;
    cloud.colors = colors;

    Ok(cloud)
}

/// Compute byte offsets for each field within a point record.
fn compute_field_offsets(header: &PcdHeader) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(header.fields.len());
    let mut offset = 0usize;
    for i in 0..header.fields.len() {
        offsets.push(offset);
        let size = header.sizes.get(i).copied().unwrap_or(4);
        let count = header.counts.get(i).copied().unwrap_or(1);
        offset += size * count;
    }
    offsets
}

/// Read a single field value from the point record and return as f32.
fn read_field_as_f32(
    point_data: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    types: &[char],
    field_idx: usize,
) -> f32 {
    let offset = offsets.get(field_idx).copied().unwrap_or(0);
    let size = sizes.get(field_idx).copied().unwrap_or(4);
    let typ = types.get(field_idx).copied().unwrap_or('F');

    if offset + size > point_data.len() {
        return 0.0;
    }

    let bytes = &point_data[offset..offset + size];

    match (typ, size) {
        ('F', 4) => {
            let arr: [u8; 4] = bytes.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(arr)
        }
        ('F', 8) => {
            let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
            f64::from_le_bytes(arr) as f32
        }
        ('U', 1) => bytes[0] as f32,
        ('U', 2) => {
            let arr: [u8; 2] = bytes.try_into().unwrap_or([0; 2]);
            u16::from_le_bytes(arr) as f32
        }
        ('U', 4) => {
            let arr: [u8; 4] = bytes.try_into().unwrap_or([0; 4]);
            u32::from_le_bytes(arr) as f32
        }
        ('U', 8) => {
            let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
            u64::from_le_bytes(arr) as f32
        }
        ('I', 1) => bytes[0] as i8 as f32,
        ('I', 2) => {
            let arr: [u8; 2] = bytes.try_into().unwrap_or([0; 2]);
            i16::from_le_bytes(arr) as f32
        }
        ('I', 4) => {
            let arr: [u8; 4] = bytes.try_into().unwrap_or([0; 4]);
            i32::from_le_bytes(arr) as f32
        }
        ('I', 8) => {
            let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
            i64::from_le_bytes(arr) as f32
        }
        _ => 0.0,
    }
}

/// Write a PCD header to the writer. Returns the number of fields written.
fn write_pcd_header<W: Write>(writer: &mut W, cloud: &PointCloud, data_format: &str) -> Result<()> {
    let num_points = cloud.len();
    let has_normals = cloud.normals.is_some();
    let has_colors = cloud.colors.is_some();

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
    writeln!(writer, "DATA {}", data_format)?;

    Ok(())
}

/// Write point cloud to PCD format (ASCII)
pub fn write_pcd<W: Write>(writer: &mut W, cloud: &PointCloud) -> Result<()> {
    let num_points = cloud.len();

    write_pcd_header(writer, cloud, "ascii")?;

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

/// Write point cloud to PCD format (binary)
///
/// The header is ASCII, but point data is written as raw little-endian bytes.
/// All fields are written as f32 (4 bytes each). Colors are packed as a u32
/// stored in the same 4-byte slot.
pub fn write_pcd_binary<W: Write>(writer: &mut W, cloud: &PointCloud) -> Result<()> {
    let num_points = cloud.len();

    write_pcd_header(writer, cloud, "binary")?;

    // Write binary data
    for i in 0..num_points {
        let p = cloud.points[i];
        writer.write_all(&p.x.to_le_bytes())?;
        writer.write_all(&p.y.to_le_bytes())?;
        writer.write_all(&p.z.to_le_bytes())?;

        if let Some(ref normals) = cloud.normals {
            let n = normals[i];
            writer.write_all(&n.x.to_le_bytes())?;
            writer.write_all(&n.y.to_le_bytes())?;
            writer.write_all(&n.z.to_le_bytes())?;
        }

        if let Some(ref colors) = cloud.colors {
            let c = colors[i];
            let r = (c.x.clamp(0.0, 1.0) * 255.0) as u32;
            let g = (c.y.clamp(0.0, 1.0) * 255.0) as u32;
            let b = (c.z.clamp(0.0, 1.0) * 255.0) as u32;
            let packed: u32 = (r << 16) | (g << 8) | b;
            // In binary PCD, rgb is stored as a float whose bits represent the packed u32
            let float_bits = f32::from_bits(packed);
            writer.write_all(&float_bits.to_le_bytes())?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

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

    // ---- Binary PCD tests ----

    #[test]
    fn test_pcd_binary_round_trip_basic() {
        let cloud = PointCloud::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, -2.0, -3.0),
        ]);

        let mut buffer = Vec::new();
        write_pcd_binary(&mut buffer, &cloud).expect("binary write failed");

        let reader = BufReader::new(Cursor::new(buffer));
        let read_cloud = read_pcd(reader).expect("binary read failed");

        assert_eq!(read_cloud.len(), 3);
        assert!((read_cloud.points[0].x - 0.0).abs() < 1e-6);
        assert!((read_cloud.points[1].x - 1.0).abs() < 1e-6);
        assert!((read_cloud.points[1].y - 2.0).abs() < 1e-6);
        assert!((read_cloud.points[1].z - 3.0).abs() < 1e-6);
        assert!((read_cloud.points[2].x - (-1.0)).abs() < 1e-6);
        assert!((read_cloud.points[2].y - (-2.0)).abs() < 1e-6);
        assert!((read_cloud.points[2].z - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pcd_binary_round_trip_with_normals() {
        let mut cloud =
            PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
        cloud.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
        ]);

        let mut buffer = Vec::new();
        write_pcd_binary(&mut buffer, &cloud).expect("binary write failed");

        let reader = BufReader::new(Cursor::new(buffer));
        let read_cloud = read_pcd(reader).expect("binary read failed");

        assert_eq!(read_cloud.len(), 2);
        assert!(read_cloud.normals.is_some());
        let normals = read_cloud.normals.unwrap();
        assert_eq!(normals.len(), 2);
        assert!((normals[0].z - 1.0).abs() < 1e-6);
        assert!((normals[1].x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pcd_binary_round_trip_with_colors() {
        let mut cloud =
            PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]);
        cloud.colors = Some(vec![
            Point3::new(1.0, 0.0, 0.0), // red
            Point3::new(0.0, 1.0, 0.0), // green
        ]);

        let mut buffer = Vec::new();
        write_pcd_binary(&mut buffer, &cloud).expect("binary write failed");

        let reader = BufReader::new(Cursor::new(buffer));
        let read_cloud = read_pcd(reader).expect("binary read failed");

        assert!(read_cloud.colors.is_some());
        let colors = read_cloud.colors.unwrap();
        assert_eq!(colors.len(), 2);
        // Colors go through 255-quantization, so check with tolerance
        assert!((colors[0].x - 1.0).abs() < 0.01); // red
        assert!((colors[0].y - 0.0).abs() < 0.01);
        assert!((colors[1].y - 1.0).abs() < 0.01); // green
    }

    #[test]
    fn test_pcd_binary_round_trip_large() {
        let n = 1000;
        let points: Vec<_> = (0..n)
            .map(|i| Point3::new(i as f32 * 0.1, i as f32 * -0.2, i as f32 * 0.3))
            .collect();
        let cloud = PointCloud::new(points);

        let mut buffer = Vec::new();
        write_pcd_binary(&mut buffer, &cloud).expect("binary write failed");

        let reader = BufReader::new(Cursor::new(buffer));
        let read_cloud = read_pcd(reader).expect("binary read failed");

        assert_eq!(read_cloud.len(), n);
        // Verify exact f32 round-trip (no precision loss in binary)
        for i in 0..n {
            assert_eq!(read_cloud.points[i].x, cloud.points[i].x);
            assert_eq!(read_cloud.points[i].y, cloud.points[i].y);
            assert_eq!(read_cloud.points[i].z, cloud.points[i].z);
        }
    }

    #[test]
    fn test_pcd_binary_vs_ascii_equivalence() {
        // Write as binary, read back, then write as ASCII, read back, compare
        let mut cloud = PointCloud::new(vec![
            Point3::new(1.5, 2.5, 3.5),
            Point3::new(-0.5, 0.0, 100.0),
        ]);
        cloud.normals = Some(vec![
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.577, 0.577, 0.577),
        ]);

        // Binary round-trip
        let mut bin_buf = Vec::new();
        write_pcd_binary(&mut bin_buf, &cloud).expect("binary write failed");
        let bin_cloud = read_pcd(BufReader::new(Cursor::new(bin_buf))).expect("binary read failed");

        assert_eq!(bin_cloud.len(), 2);
        // Binary preserves exact f32 values
        assert_eq!(bin_cloud.points[0].x, 1.5);
        assert_eq!(bin_cloud.points[1].z, 100.0);
        assert!(bin_cloud.normals.is_some());
    }

    #[test]
    fn test_pcd_binary_empty_cloud() {
        let cloud = PointCloud::new(vec![]);

        let mut buffer = Vec::new();
        write_pcd_binary(&mut buffer, &cloud).expect("binary write failed");

        let reader = BufReader::new(Cursor::new(buffer));
        let read_cloud = read_pcd(reader).expect("binary read failed");

        assert_eq!(read_cloud.len(), 0);
    }

    #[test]
    fn test_pcd_binary_compressed_error_message() {
        // Construct a minimal PCD with DATA binary_compressed header
        let header = "# .PCD v0.7\n\
                       VERSION 0.7\n\
                       FIELDS x y z\n\
                       SIZE 4 4 4\n\
                       TYPE F F F\n\
                       COUNT 1 1 1\n\
                       WIDTH 1\n\
                       HEIGHT 1\n\
                       POINTS 1\n\
                       DATA binary_compressed\n";

        let reader = BufReader::new(Cursor::new(header.as_bytes().to_vec()));
        let result = read_pcd(reader);

        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("LZF"),
            "Error should mention LZF decompression: {}",
            err_msg
        );
    }

    #[test]
    fn test_pcd_ascii_write_read_back() {
        let mut cloud = PointCloud::new(vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(4.0, 5.0, 6.0),
            Point3::new(7.0, 8.0, 9.0),
        ]);
        cloud.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ]);
        cloud.colors = Some(vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]);

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud).expect("write failed");

        let reader = Cursor::new(buffer);
        let read_cloud = read_pcd(reader).expect("read failed");

        assert_eq!(read_cloud.len(), 3);
        assert!(read_cloud.normals.is_some());
        assert!(read_cloud.colors.is_some());
        assert!((read_cloud.points[2].x - 7.0).abs() < 0.001);
    }
}
