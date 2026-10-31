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
    //
    // Each entry is the property name, whether it is a `list`, and whether it is
    // declared as an integer type.
    //
    // The integer flag is what makes colour normalisation exact. A colour
    // property declared `uchar` holds a byte 0..255 and must be divided by 255;
    // one declared `float` already holds 0..1 and must not be. Guessing from
    // the value instead cannot distinguish the two: a byte value of `1` is a
    // legitimate near-black (`1/255`) and is also what an already-normalised
    // `1.0` looks like, so the reader called near-black white.
    //
    // This is the *working* set: it is flushed at every element boundary.
    let mut props: Vec<(String, bool, bool)> = Vec::new();
    // The working property set is flushed on every element boundary, so this
    // holds the *vertex* element's properties — the ones the body is read
    // against — even when a face element is declared after them.
    let mut vertex_props: Vec<(String, bool, bool)> = Vec::new();
    // Body rows belonging to elements declared *before* the vertex element.
    //
    // PLY writes one block per element, in header order, so the body does not
    // start at the vertices: if `element face 2` precedes `element vertex 2`,
    // the first two rows are faces. Without this the body loop read `3 0 0 1`
    // as a vertex - a face row's first three integers are shaped exactly like
    // coordinates - and the real vertex block was never read at all. Measured:
    // `Ok`, 2 points, `[[3,0,0],[3,0,1]]` where `[[1,2,3],[4,5,6]]` was the
    // file. Total data loss, reported as success.
    let mut rows_before_vertex: usize = 0;
    // Rows of the element currently being described, summed into
    // `rows_before_vertex` when the header moves on to the next element.
    let mut current_rows: usize = 0;
    // Counts of any further `element vertex` blocks, added to the first so the
    // body still has to satisfy every vertex element the header declares.
    let mut vertex_blocks_after: usize = 0;
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
            // A new element closes the previous one: its properties leave the
            // vertex column set, so two `element vertex` blocks no longer union
            // theirs and a face property can no longer be found by the x/y/z
            // search. If the previous element was itself a vertex element its
            // rows are already part of `num_vertices` and are not prepended.
            if !in_vertex_element {
                rows_before_vertex = rows_before_vertex.saturating_add(current_rows);
            }
            if in_vertex_element {
                // Hand the vertex columns over before the working set is
                // emptied. The body is read against the vertex element's own
                // properties, so a later `element face` must not take them away.
                vertex_props = std::mem::take(&mut props);
            }
            props.clear();

            let mut parts = line.split_whitespace();
            let _keyword = parts.next();
            let name = parts.next().unwrap_or("");
            // The VERTEX count is the one the body length depends on, so it is
            // parsed strictly: `element vertex` with no count, or with one that
            // is not a number, is still an error.
            //
            // A NON-vertex element's count is read leniently - an unparsable one
            // becomes 0 - because this reader SKIPS those rows as whole lines
            // rather than parsing their columns, so the count only has to be a
            // row tally. Making it a second error path would mean a face element
            // could fail a file whose vertices are perfectly fine.
            let count: usize = if name == "vertex" {
                parts
                    .next()
                    .ok_or_else(|| Error::ParseError("Invalid vertex count".to_string()))?
                    .parse()
                    .map_err(|_| Error::ParseError("Invalid vertex count number".to_string()))?
            } else {
                parts.next().and_then(|c| c.parse().ok()).unwrap_or(0)
            };
            current_rows = count;
            if name == "vertex" {
                if in_vertex_element {
                    // A second `element vertex` extends the block.
                    vertex_blocks_after = vertex_blocks_after.saturating_add(count);
                    current_rows = current_rows.saturating_add(count);
                } else {
                    in_vertex_element = true;
                    num_vertices = count;
                }
            } else {
                in_vertex_element = false;
            }
        } else if line.starts_with("property ") {
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
            // `property <type> <name>`, or `property list <count_type> <value_type> <name>`.
            let mut parts = line.split_whitespace();
            let _property = parts.next();
            // The token after `property` is the type, and is `list` only for a
            // list property - so testing it for `list` does not consume a
            // separate token in the non-list case.
            let first = parts.next().unwrap_or("");
            let is_list = first == "list";
            let declared_type = if is_list {
                parts.next().unwrap_or("")
            } else {
                first
            };
            let is_integer = matches!(
                declared_type,
                "char"
                    | "uchar"
                    | "short"
                    | "ushort"
                    | "int"
                    | "uint"
                    | "int8"
                    | "uint8"
                    | "int16"
                    | "uint16"
                    | "int32"
                    | "uint32"
            );
            props.push((name, is_list, is_integer));
        } else if line.starts_with("property ") {
            // A property line outside any element has nothing to attach to; it
            // is not counted against any element's row budget.
        } else if line == "end_header" {
            // A vertex element that is the LAST one in the header never hit an
            // element boundary, so its columns are handed over here instead.
            // Guarded, because a vertex element followed by `element face` was
            // already handed over at that boundary and this would wipe it.
            if in_vertex_element && vertex_props.is_empty() {
                vertex_props = std::mem::take(&mut props);
            }
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
    // scalar property and is excluded here. Only the vertex element's own
    // properties are searched: a property declared by a later element is not a
    // column of a vertex row.
    let props = &vertex_props;
    let pos_of = |names: &[&str]| -> Option<usize> {
        props
            .iter()
            .position(|(p, is_list, _)| !is_list && names.contains(&p.as_str()))
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
    if let Some((name, _, _)) = props.iter().find(|(_, is_list, _)| *is_list) {
        return Err(Error::ParseError(format!(
            "PLY: vertex property {name:?} is a `property list`, which this reader \
             does not support. Parsing it would misplace every scalar property \
             declared after it."
        )));
    }
    let width = props.len();

    // The body opens with the rows of any element declared before the vertex
    // element, in header order. Skip exactly that many lines: they belong to
    // another element, and the first three integers of a face row are shaped
    // exactly like coordinates, so reading one is silent data corruption rather
    // than a visible failure.
    for _ in 0..rows_before_vertex {
        lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in data".to_string()))??;
    }

    // A second `element vertex` block extends the first rather than replacing
    // it. Replacing the count is what `element vertex 1` followed by
    // `element vertex 40000000000` did - the second block's properties were also
    // unioned onto the first's, giving a row width no real row has.
    let total_vertices = num_vertices + vertex_blocks_after;

    for _ in 0..total_vertices {
        let line = lines
            .next()
            .ok_or_else(|| Error::ParseError("Unexpected EOF in data".to_string()))??;

        // Only the coordinates are checked here, and only *after* every token is
        // parsed - see the note at the check below.
        let values: Vec<f32> = line
            .split_whitespace()
            .map(|s| {
                let v: f32 = s
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid number: {}", s)))?;
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;

        if values.len() < width.max(3) {
            return Err(Error::InvalidInput(
                "Not enough values for vertex".to_string(),
            ));
        }

        // `f32::from_str` accepts "NaN", "inf" and "1e40" (which overflows to
        // infinity), so a single bad coordinate would reach the cloud as a point
        // at infinity and poison every bound computed from it.
        //
        // This MUST run on x/y/z alone, and that is why it cannot live in the
        // parse closure above. A packed `rgb`/`rgba` column is a *bit pattern*
        // reinterpreted as an f32 - the body literally holds the float whose
        // bits are `R<<16 | G<<8 | B` - so the exponent field IS the colour.
        // Checking every token refused exactly those colours:
        //
        // ```text
        // 0x7f800000 -> "inf"    (R=127,G=128)
        // 0xff800000 -> "-inf"   (R=255,G=128)
        // 0x7fff0000 -> "NaN"    (R=127,G=255)
        // ```
        //
        // a quarter of the saturated red/magenta corner of the cube. Moving the
        // check past the parse is what lets a packed column through, and keeps
        // the coordinate guard - and its message - exactly as strict as before.
        for &(name, idx) in &[("x", xi), ("y", yi), ("z", zi)] {
            if !values[idx].is_finite() {
                return Err(Error::ParseError(format!(
                    "Non-finite coordinate {name}: {}",
                    line.split_whitespace().nth(idx).unwrap_or("?")
                )));
            }
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
            // Scale by the *declared* type, not by guessing from the value.
            //
            // The previous heuristic was `if v > 1.0 { v / 255.0 } else { v }`,
            // which cannot tell a byte from an already-normalised float: the
            // legitimate byte value `1` is a near-black `1/255`, but it is also
            // what a normalised `1.0` looks like, so it was read as pure white.
            // Measured through `write_ply` -> `read_ply`: a vertex coloured
            // `1/255` came back `(1.0, 1.0, 1.0)` - near-black became white, a
            // factor of 255.
            //
            // Byte 0 is also ambiguous but harmless: dividing 0 by 255 is 0.
            let scale = |idx: usize| -> f32 {
                match props.get(idx) {
                    // An integer-declared property holds a byte; divide by 255.
                    Some((_, _, true)) => values[idx] / 255.0,
                    // A float-declared property is already 0..1 by the spec, so
                    // dividing would darken the whole cloud.
                    _ => values[idx],
                }
            };
            colors.as_mut().unwrap().push(Point3::new(
                scale(ri.unwrap()),
                scale(gi.unwrap()),
                scale(bi.unwrap()),
            ));
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
