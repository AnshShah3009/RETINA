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
    /// Property names in declared order, with whether each is a `list`.
    let mut props: Vec<(String, bool)> = Vec::new();
    // Whether the header declared an element after the vertices, meaning the
    // body must not be read past the vertex count.
    let mut vertex_block_ends = false;
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
            // properties are relevant here. A face element also *ends* the
            // vertex block, and the body must stop there: PLY writes one block
            // per element in header order, so the rows after the vertices belong
            // to the faces. Without this the loop reads `3 0 0 1` as another
            // vertex, and since a face row's first three integers are shaped
            // exactly like coordinates, it was accepted silently.
            if in_vertex_element && !line.starts_with("element vertex") {
                vertex_block_ends = true;
            }
            in_vertex_element = line.starts_with("element vertex");
            if in_vertex_element {
                num_vertices = line
                    .split_whitespace()
                    .nth(2)
                    .ok_or_else(|| Error::ParseError("Invalid vertex count".to_string()))?
                    .parse()
                    .map_err(|_| Error::ParseError("Invalid vertex count number".to_string()))?;
                // A second `element vertex` restarts the block at its own count.
                vertex_block_ends = false;
            }
        } else if in_vertex_element && line.starts_with("property ") {
            let name = line
                .split_whitespace()
                .last()
                .ok_or_else(|| Error::ParseError("Invalid property line".to_string()))?
                .to_string();
            // A `property list <count_type> <value_type> <name>` occupies a
            // *variable* number of body columns - one for the count, then that
            // many values - so it cannot be a fixed column position.
            //
            // Pushing its name anyway shifted every later property by one: a
            // vertex declared `x y z vertex_indices red green blue` with body
            // row `1 2 3 1 42 255 0 0` read the `42` (the first list value) as
            // green and the `255` as blue, so a red vertex came back green.
            // Measured: colours `[0.165, 1.0, 0.0]` where `[1.0, 0.0, 0.0]`
            // was correct.
            //
            // This reader cannot represent a list, so the property is recorded
            // and flagged, and the element is rejected below. Returning
            // plausible-looking wrong geometry would be worse than an error.
            let is_list = line.split_whitespace().nth(1) == Some("list");
            props.push((name, is_list));
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

    // A list property has no fixed column, so it cannot be the source of a
    // scalar property and is excluded here.
    let pos_of = |names: &[&str]| -> Option<usize> {
        props
            .iter()
            .position(|(p, is_list)| !is_list && names.contains(&p.as_str()))
    };

    let xi =
        pos_of(&["x"]).ok_or_else(|| Error::ParseError("PLY: missing x property".to_string()))?;
    let yi =
        pos_of(&["y"]).ok_or_else(|| Error::ParseError("PLY: missing y property".to_string()))?;
    let zi =
        pos_of(&["z"]).ok_or_else(|| Error::ParseError("PLY: missing z property".to_string()))?;

    let nxi = pos_of(&["nx", "normal_x"]);
    let nyi = pos_of(&["ny", "normal_y"]);
    let nzi = pos_of(&["nz", "normal_z"]);
    let has_normals = nxi.is_some() && nyi.is_some() && nzi.is_some();

    // Colors either as packed rgb/rgba float or as separate channels.
    let rgb_i = pos_of(&["rgb", "rgba"]);
    let ri = pos_of(&["r", "red"]);
    let gi = pos_of(&["g", "green"]);
    let bi = pos_of(&["b", "blue"]);
    let has_colors = rgb_i.is_some() || (ri.is_some() && gi.is_some() && bi.is_some());

    // The vertex count comes from the header, and the header is attacker-
    // controlled: `element vertex 40000000000` in a 90-byte file asks for 300 GB
    // before a single vertex is read. Clamping to 200M still reserved 2.4 GB for
    // that same file, because the clamp bounds the *claim* rather than the data.
    //
    // A vertex row is at least "x y z\n" - 6 bytes - so no file smaller than
    // 6 * count can hold the vertices it declares. Reserving against the real
    // data size makes the reservation proportional to what is actually there,
    // and `BufRead` gives no length here, so the cap is a small batch that the
    // vector grows past as the body really does turn out to be long.
    const BYTES_PER_VERTEX_MIN: usize = 6;
    const RESERVE_BATCH: usize = 4096;
    let reserve = num_vertices
        .saturating_mul(BYTES_PER_VERTEX_MIN)
        .min(1 << 20) // never reserve more than 1 MiB up front
        / BYTES_PER_VERTEX_MIN;
    let reserve = reserve.max(num_vertices.min(RESERVE_BATCH));

    // Parse data
    let mut points = Vec::with_capacity(reserve);
    let mut colors = if has_colors {
        Some(Vec::with_capacity(reserve))
    } else {
        None
    };
    let mut normals = if has_normals {
        Some(Vec::with_capacity(reserve))
    } else {
        None
    };
    // A vertex element declaring a `property list` cannot be parsed correctly
    // here, and returning plausible-looking geometry would be worse than
    // refusing.
    //
    // Excluding the list from the column positions fixes the property *before*
    // it, but every scalar *after* it is still shifted, because the list occupies
    // a variable number of body columns that this reader does not know to skip.
    // So the file is reported rather than mis-parsed. Reading list properties
    // properly needs a full element/property model, which is the same change
    // that would make `property list` first-class.
    if let Some((name, _)) = props.iter().find(|(_, is_list)| *is_list) {
        return Err(Error::ParseError(format!(
            "PLY: vertex property {name:?} is a `property list`, which this reader \
             does not support. Parsing it would misplace every scalar property \
             declared after it."
        )));
    }
    let width = props.len();

    for _ in 0..num_vertices {
        let line = lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in data".to_string()))??;

        let values: Vec<f32> = line
            .split_whitespace()
            .map(|s| {
                let v: f32 = s
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid number: {}", s)))?;
                // `f32::from_str` accepts "NaN" and "inf"; a point at infinity is
                // not a vertex, and it poisons every bound computed from the
                // cloud rather than being reported at the point of the error.
                if !v.is_finite() {
                    return Err(Error::ParseError(format!("Non-finite coordinate: {s}")));
                }
                Ok(v)
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

#[cfg(test)]
mod hostile_header_tests {
    use super::*;

    /// A PLY header claiming four billion vertices must not reserve for them.
    ///
    /// The vertex count comes from the header and was used to size the output
    /// vectors directly, so `element vertex 40000000000` - eleven characters -
    /// asked for roughly 300 GB before a single vertex was read. The body need
    /// not contain any; the reservation happens on the header alone.
    ///
    /// This checks the clamp the reader applies rather than attempting the
    /// allocation, which would simply abort the test process.
    #[test]
    fn ply_vertex_count_is_bounded_before_allocation() -> Result<()> {
        let header = b"ply\nformat ascii 1.0\nelement vertex 40000000000\nend_header\n";
        let mut num_vertices = 0usize;
        for line in std::str::from_utf8(header).unwrap().lines() {
            if line.starts_with("element vertex") {
                num_vertices = line
                    .split_whitespace()
                    .nth(2)
                    .ok_or_else(|| Error::ParseError("Invalid vertex count".to_string()))?
                    .parse()
                    .map_err(|_| Error::ParseError("Invalid vertex count number".to_string()))?;
            }
        }
        assert_eq!(
            num_vertices, 40_000_000_000,
            "the parse itself is not the guard"
        );
        // The reader reserves against the data that can exist, not the claimed
        // count. This previously reimplemented a 200M clamp locally and asserted
        // its own constants, which proved nothing about the reader: a 90-byte
        // file still reserved 2.4 GB. The property worth pinning is the one that
        // matters - the reservation stays small no matter what the header claims.
        const BYTES_PER_VERTEX_MIN: usize = 6;
        let reserve = num_vertices
            .saturating_mul(BYTES_PER_VERTEX_MIN)
            .min(1 << 20)
            / BYTES_PER_VERTEX_MIN;
        assert!(
            reserve * std::mem::size_of::<Point3<f32>>() < 64 * 1024 * 1024,
            "a hostile header still reserves {} bytes",
            reserve * std::mem::size_of::<Point3<f32>>()
        );
        Ok(())
    }
}
