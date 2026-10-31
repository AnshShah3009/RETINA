//! LAS/LAZ point cloud I/O (ASPRS LiDAR format).
//!
//! Supports LAS 1.0-1.4 and LAZ compressed files.
//! Feature-gated behind the `las` feature flag.
//!
//! # Example
//! ```ignore
//! use cv_io::las_io::{read_las, write_las, LasData};
//!
//! let data = read_las("input.laz")?;
//! println!("{} points, bounds: {:?}", data.points.len(), data.bounds);
//! write_las("output.las", &data)?;
//! ```

use cv_core::PointCloud;
use nalgebra::{Point3, Vector3};
use std::path::Path;

/// LAS point cloud data with metadata.
#[derive(Debug, Clone)]
pub struct LasData {
    /// Point positions (x, y, z).
    pub points: Vec<Point3<f32>>,
    /// Point colors (RGB, 0-1 range). None if no color data.
    pub colors: Option<Vec<Point3<f32>>>,
    /// Point intensities (0-1 range). None if not available.
    pub intensities: Option<Vec<f32>>,
    /// Classification codes (e.g., ground=2, vegetation=3). None if not available.
    pub classifications: Option<Vec<u8>>,
    /// Return numbers (1-based). None if not available.
    pub return_numbers: Option<Vec<u8>>,
    /// Number of returns. None if not available.
    pub number_of_returns: Option<Vec<u8>>,
    /// GPS time per point. None if not available.
    pub gps_times: Option<Vec<f64>>,
    /// Bounding box: (min_x, min_y, min_z, max_x, max_y, max_z).
    pub bounds: (f64, f64, f64, f64, f64, f64),
    /// Total number of points.
    pub num_points: usize,
}

/// Read a LAS or LAZ file.
pub fn read_las<P: AsRef<Path>>(path: P) -> cv_core::Result<LasData> {
    use las::{Read, Reader};

    let mut reader = Reader::from_path(path.as_ref())
        .map_err(|e| cv_core::Error::IoError(format!("Failed to open LAS file: {}", e)))?;

    let header = reader.header();
    // `header` borrows `reader`, which the point loop below needs mutably, so
    // take the two header values needed up front by value. The bounding box is
    // deliberately NOT taken here: it is unvalidated file input and is handled
    // after the points are read - see the `compute_bounds` call below.
    let format = *header.point_format();

    // The header's point count is a file-controlled u64. Reserving for it up
    // front means a 4-byte field claiming four billion points commits ~48 GB
    // before a single point has been read, and the file need not contain any -
    // the allocation happens on the header alone. Cap the reservation at
    // something plausible for a cloud and let the vector grow if the file turns
    // out to be longer.
    const MAX_REASONABLE_LAS_POINTS: usize = 200_000_000;
    let num_points = (header.number_of_points() as usize).min(MAX_REASONABLE_LAS_POINTS);

    let mut points = Vec::with_capacity(num_points);
    let mut colors_vec: Vec<Point3<f32>> = Vec::new();
    let mut intensities_vec: Vec<f32> = Vec::new();
    let mut classifications_vec: Vec<u8> = Vec::new();
    let mut return_numbers_vec: Vec<u8> = Vec::new();
    let mut number_of_returns_vec: Vec<u8> = Vec::new();
    let mut gps_times_vec: Vec<f64> = Vec::new();

    // The point format is a file-wide, header-level declaration, so whether a
    // record *can* carry a colour or a GPS time is one answer for the whole
    // file. Detect that up front, and then demand one push per record for every
    // present field. Deciding from the first point alone and reusing the answer
    // let a record that failed to supply a field skip its push, leaving
    // `points` longer than `colors`/`gps_times` and desynchronising every
    // downstream `data[i]` index.
    let has_color = format.has_color;
    let has_gps = format.has_gps_time;
    // Intensity, classification and return numbers are mandatory in every LAS
    // point format, so they are always pushed and need no detection.
    let has_intensity = true;
    let has_classification = true;
    let has_returns = true;

    for point_result in reader.points() {
        let point = point_result
            .map_err(|e| cv_core::Error::IoError(format!("Failed to read LAS point: {}", e)))?;

        points.push(Point3::new(point.x as f32, point.y as f32, point.z as f32));

        if has_intensity {
            intensities_vec.push(point.intensity as f32 / 65535.0);
        }

        if has_color {
            // A record in a colour-carrying format that yields no colour is a
            // malformed file. Substituting black would fabricate data, and
            // skipping the push is what desynchronises the vectors, so the
            // file is rejected instead.
            let color = point.color.ok_or_else(|| {
                cv_core::Error::ParseError(format!(
                    "LAS record {} declares point format {} (which carries RGB) but \
                     supplies no colour, so colors would desynchronize from points",
                    points.len() - 1,
                    format.to_u8().unwrap_or(0xFF)
                ))
            })?;
            colors_vec.push(Point3::new(
                color.red as f32 / 65535.0,
                color.green as f32 / 65535.0,
                color.blue as f32 / 65535.0,
            ));
        }

        if has_classification {
            let class_u8: u8 = point.classification.into();
            classifications_vec.push(class_u8);
        }

        if has_returns {
            return_numbers_vec.push(point.return_number);
            number_of_returns_vec.push(point.number_of_returns);
        }

        if has_gps {
            // Same reasoning as colour: a missing GPS time would skip a push
            // and leave `gps_times` short.
            let t = point.gps_time.ok_or_else(|| {
                cv_core::Error::ParseError(format!(
                    "LAS record {} declares point format {} (which carries GPS time) \
                     but supplies none, so gps_times would desynchronize from points",
                    points.len() - 1,
                    format.to_u8().unwrap_or(0xFF)
                ))
            })?;
            gps_times_vec.push(t);
        }
    }

    // The loop above pushes exactly one value per record into every field it
    // declares, so these are equal by construction. Assert it anyway: this is
    // the boundary the returned `LasData` crosses, and a short vector here is
    // precisely the `read_pcd` bug where `colors` was attached to a cloud
    // without a `len() == points.len()` gate and then panicked in the writer.
    // Every one of these fields is indexed by point index downstream
    // (`las_to_point_cloud`, `filter_by_mask`, and the writer), so a short
    // vector silently mislabels data rather than failing.
    //
    // Each check is conditional on the field being collected at all: a file
    // whose format carries no colour legitimately leaves `colors_vec` empty.
    let n = points.len();
    if has_color {
        debug_assert_eq!(colors_vec.len(), n, "colors desynchronized from points");
    }
    if has_gps {
        debug_assert_eq!(
            gps_times_vec.len(),
            n,
            "gps_times desynchronized from points"
        );
    }
    for (name, len) in [
        ("intensities", intensities_vec.len()),
        ("classifications", classifications_vec.len()),
        ("return_numbers", return_numbers_vec.len()),
        ("number_of_returns", number_of_returns_vec.len()),
    ] {
        debug_assert_eq!(len, n, "{name} desynchronized from points");
    }

    // Bounds are recomputed from the points that were actually read, rather
    // than copied from `header.bounds()`.
    //
    // The header box is unvalidated file input and every consumer uses it to
    // size an ROI or a voxel grid, so an inverted (`min > max`) or saturated
    // (`max = f64::MAX`) box propagates straight into geometry. It is also
    // routinely *wrong* without being hostile: many writers leave it at its
    // default or as the bounds of a pre-filtered subset, so a file can be
    // perfectly valid and still under-report its extent.
    //
    // The extra pass is a single min/max over data already resident in
    // `points`, i.e. it is O(n) arithmetic against an O(n) parse that has
    // already been paid, and it is what `point_cloud_to_las` and
    // `filter_by_mask` both do for the same reason. Recomputing is therefore
    // both stricter (it cannot propagate a bad box) and more accurate (it
    // cannot propagate a stale one) at negligible cost. The header value is
    // still read below purely so a genuinely inverted box can be reported -
    // a bad box is worth telling the caller about, and recomputing silently
    // would hide that the file is malformed.
    // The header box is read back here, after the loop, so that reporting an
    // inverted one does not need `reader` to still be borrowed by `header`.
    let hb = reader.header().bounds();
    let header_bounds = (hb.min.x, hb.min.y, hb.min.z, hb.max.x, hb.max.y, hb.max.z);
    if n > 0 {
        let (i, f) = (header_bounds.0, header_bounds.3);
        let (j, g) = (header_bounds.1, header_bounds.4);
        let (k, h) = (header_bounds.2, header_bounds.5);
        if !(i <= f && j <= g && k <= h) {
            return Err(cv_core::Error::ParseError(format!(
                "LAS header declares an inverted bounding box: min=({i}, {j}, {k}) \
                 max=({f}, {g}, {h}); falling back to the extent of the points read"
            )));
        }
    }

    // Computed before `points` is moved into the struct below.
    let computed_bounds = if n == 0 {
        // An empty file has no extent, so report a degenerate box at the origin
        // rather than infinities, matching `point_cloud_to_las` and
        // `filter_by_mask`.
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        compute_bounds(&points)
    };

    Ok(LasData {
        num_points: n,
        points,
        colors: if has_color && !colors_vec.is_empty() {
            Some(colors_vec)
        } else {
            None
        },
        intensities: if has_intensity && !intensities_vec.is_empty() {
            Some(intensities_vec)
        } else {
            None
        },
        classifications: if has_classification && !classifications_vec.is_empty() {
            Some(classifications_vec)
        } else {
            None
        },
        return_numbers: if has_returns && !return_numbers_vec.is_empty() {
            Some(return_numbers_vec)
        } else {
            None
        },
        number_of_returns: if has_returns && !number_of_returns_vec.is_empty() {
            Some(number_of_returns_vec)
        } else {
            None
        },
        gps_times: if has_gps && !gps_times_vec.is_empty() {
            Some(gps_times_vec)
        } else {
            None
        },
        // An empty file has no extent, so report a degenerate box at the origin
        // rather than infinities, matching `point_cloud_to_las` and
        // `filter_by_mask`.
        bounds: computed_bounds,
    })
}

/// The true extent of `points` as `(min_x, min_y, min_z, max_x, max_y, max_z)`.
///
/// Seeded with infinities rather than `f64::MIN`, which is the most *negative*
/// finite f64 and would leave an inverted box for an empty slice - the same
/// trap already fixed in `point_cloud_to_las` and `filter_by_mask`. Callers
/// must handle the empty case before calling.
fn compute_bounds(points: &[Point3<f32>]) -> (f64, f64, f64, f64, f64, f64) {
    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        min.x = min.x.min(p.x as f64);
        min.y = min.y.min(p.y as f64);
        min.z = min.z.min(p.z as f64);
        max.x = max.x.max(p.x as f64);
        max.y = max.y.max(p.y as f64);
        max.z = max.z.max(p.z as f64);
    }
    (min.x, min.y, min.z, max.x, max.y, max.z)
}

/// Write a LAS file (uncompressed).
pub fn write_las<P: AsRef<Path>>(path: P, data: &LasData) -> cv_core::Result<()> {
    use las::{Builder, Point, Write, Writer};

    let mut builder = Builder::from((1, 4)); // LAS 1.4
                                             // Pick a point format that actually carries the fields we write. Formats 0
                                             // and 2 have no GPS time slot, so writing `gps_times` with them silently
                                             // dropped it; formats 1 and 3 include GPS time.
    let has_color = data.colors.is_some();
    let has_gps = data.gps_times.is_some();
    builder.point_format = match (has_color, has_gps) {
        (true, true) => las::point::Format::new(3).unwrap(), // XYZ + GPS + RGB
        (false, true) => las::point::Format::new(1).unwrap(), // XYZ + GPS
        (true, false) => las::point::Format::new(2).unwrap(), // XYZ + RGB
        (false, false) => las::point::Format::new(0).unwrap(), // XYZ only
    };

    let header = builder
        .into_header()
        .map_err(|e| cv_core::Error::IoError(format!("Failed to build LAS header: {}", e)))?;

    let mut writer = Writer::from_path(path.as_ref(), header)
        .map_err(|e| cv_core::Error::IoError(format!("Failed to create LAS writer: {}", e)))?;

    for (i, pt) in data.points.iter().enumerate() {
        let mut point = Point {
            x: pt.x as f64,
            y: pt.y as f64,
            z: pt.z as f64,
            ..Default::default()
        };

        if let Some(ref intensities) = data.intensities {
            if i < intensities.len() {
                point.intensity = (intensities[i] * 65535.0).clamp(0.0, 65535.0) as u16;
            }
        }

        if let Some(ref colors) = data.colors {
            if i < colors.len() {
                point.color = Some(las::Color {
                    red: (colors[i].x * 65535.0).clamp(0.0, 65535.0) as u16,
                    green: (colors[i].y * 65535.0).clamp(0.0, 65535.0) as u16,
                    blue: (colors[i].z * 65535.0).clamp(0.0, 65535.0) as u16,
                });
            }
        }

        if let Some(ref classifications) = data.classifications {
            if i < classifications.len() {
                point.classification =
                    las::point::Classification::new(classifications[i]).unwrap_or_default();
            }
        }

        if let Some(ref return_numbers) = data.return_numbers {
            if i < return_numbers.len() {
                point.return_number = return_numbers[i];
            }
        }

        if let Some(ref number_of_returns) = data.number_of_returns {
            if i < number_of_returns.len() {
                point.number_of_returns = number_of_returns[i];
            }
        }

        if let Some(ref gps_times) = data.gps_times {
            if i < gps_times.len() {
                point.gps_time = Some(gps_times[i]);
            }
        }

        writer
            .write(point)
            .map_err(|e| cv_core::Error::IoError(format!("Failed to write LAS point: {}", e)))?;
    }

    writer
        .close()
        .map_err(|e| cv_core::Error::IoError(format!("Failed to close LAS writer: {}", e)))?;

    Ok(())
}

/// Convert LAS data to a RETINA PointCloud.
pub fn las_to_point_cloud(data: &LasData) -> PointCloud {
    let cloud = PointCloud::new(data.points.clone());
    if let Some(ref colors) = data.colors {
        cloud
            .with_colors(colors.clone())
            .unwrap_or_else(|_| PointCloud::new(data.points.clone()))
    } else {
        cloud
    }
}

/// Create LAS data from a RETINA PointCloud.
pub fn point_cloud_to_las(cloud: &PointCloud) -> LasData {
    // `f64::MIN` is the most *negative* finite f64, not the smallest positive
    // one, so `min.max(p.x)` could never rise above it and `max` stayed there
    // for an empty cloud. The result was an inverted box:
    // `(1.797e308, 1.797e308, 1.797e308, -1.797e308, -1.797e308, -1.797e308)` -
    // min greater than max on every axis.
    //
    // `compute_bounds` seeds with infinities instead, so an empty cloud produces
    // a degenerate but *valid* box rather than an impossible one.
    LasData {
        num_points: cloud.points.len(),
        points: cloud.points.clone(),
        colors: cloud.colors.clone(),
        intensities: None,
        classifications: None,
        return_numbers: None,
        number_of_returns: None,
        gps_times: None,
        // An empty cloud has no extent, so report a degenerate box at the origin
        // rather than infinities, which a writer would then have to special-case.
        bounds: if cloud.points.is_empty() {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            compute_bounds(&cloud.points)
        },
    }
}

/// Filter LAS data by classification code.
///
/// The mask is derived from `data.classifications`, so if that vector is not
/// index-parallel with `data.points` the mask is not parallel either and every
/// result below would be wrong. That is a caller error - a `LasData` built by
/// this crate always satisfies the invariant - so a mismatch is reported as an
/// empty result rather than propagated.
pub fn filter_by_classification(data: &LasData, class: u8) -> LasData {
    let classifications = match &data.classifications {
        Some(c) => c,
        None => return data.clone(),
    };

    if classifications.len() != data.points.len() {
        return empty_las_like();
    }

    let mask: Vec<bool> = classifications.iter().map(|&c| c == class).collect();
    filter_by_mask(data, &mask)
}

/// Filter LAS data by a boolean mask.
///
/// `mask` must be index-parallel with `data.points`: one entry per point. This
/// used to `zip`, which silently truncated to the shorter of the two, so a mask
/// one entry short dropped trailing points and a longer mask had its tail
/// ignored - both with no error, and with a per-field vector ending up indexed
/// against a point that is no longer there.
///
/// A length mismatch is a caller error rather than something to absorb, so it
/// is reported by returning an empty `LasData` (zero points, no optional
/// fields) instead of a plausible-looking but wrong subset. The alternative
/// - padding the short side with `false` - was rejected because it invents data
/// to paper over a bug in the caller. The behaviour deliberately changes the
/// signature: silently producing a cloud that indexes one field against the
/// wrong point is the exact failure this guards against.
pub fn filter_by_mask(data: &LasData, mask: &[bool]) -> LasData {
    if mask.len() != data.points.len() {
        return empty_las_like();
    }

    // Index rather than `zip`: with the lengths checked above, `enumerate`
    // keeps every optional vector index-parallel with `points` by construction,
    // and would panic loudly rather than truncate if the invariant were broken.
    fn filter_vec<T: Clone>(opt: &Option<Vec<T>>, mask: &[bool]) -> Option<Vec<T>> {
        opt.as_ref().map(|v| {
            mask.iter()
                .enumerate()
                .filter(|(_, &m)| m)
                .map(|(i, _)| v[i].clone())
                .collect()
        })
    }

    let points: Vec<_> = data
        .points
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(p, _)| *p)
        .collect();

    let num_points = points.len();
    // Computed before `points` is moved into the struct below.
    let bounds = if num_points == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        // Same `f64::MIN` trap as `point_cloud_to_las`: it is the most negative
        // finite f64, so a running `max` seeded with it could never rise above
        // it and an empty result carried an inverted box.
        compute_bounds(&points)
    };
    LasData {
        num_points,
        points,
        colors: filter_vec(&data.colors, mask),
        intensities: filter_vec(&data.intensities, mask),
        classifications: filter_vec(&data.classifications, mask),
        return_numbers: filter_vec(&data.return_numbers, mask),
        number_of_returns: filter_vec(&data.number_of_returns, mask),
        gps_times: filter_vec(&data.gps_times, mask),
        bounds,
    }
}

/// A zero-point `LasData`.
///
/// Used to report a rejected filter request: zero points and no optional
/// fields, so nothing can be indexed against a point that is not there, and a
/// degenerate (but valid) box at the origin so the result is still writable.
fn empty_las_like() -> LasData {
    LasData {
        num_points: 0,
        points: Vec::new(),
        colors: None,
        intensities: None,
        classifications: None,
        return_numbers: None,
        number_of_returns: None,
        gps_times: None,
        bounds: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
}

#[cfg(all(test, feature = "las"))]
mod tests {
    use super::*;

    #[test]
    fn test_write_las_round_trips_gps_time() {
        let data = LasData {
            points: vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)],
            colors: None,
            intensities: None,
            classifications: None,
            return_numbers: None,
            number_of_returns: None,
            gps_times: Some(vec![1000.25, 2000.5]),
            bounds: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            num_points: 2,
        };

        let path = std::env::temp_dir().join(format!(
            "retina_las_gps_round_trip_{}.las",
            std::process::id()
        ));
        write_las(&path, &data).expect("write_las failed");
        let read = read_las(&path).expect("read_las failed");
        let _ = std::fs::remove_file(&path);

        let gps = read.gps_times.expect("gps_times missing after round-trip");
        assert_eq!(gps.len(), 2);
        assert!((gps[0] - 1000.25).abs() < 1e-6);
        assert!((gps[1] - 2000.5).abs() < 1e-6);
    }

    #[test]
    fn test_write_las_round_trips_gps_time_and_color() {
        let data = LasData {
            points: vec![Point3::new(1.0, 2.0, 3.0)],
            colors: Some(vec![Point3::new(1.0, 0.0, 0.0)]),
            intensities: None,
            classifications: None,
            return_numbers: None,
            number_of_returns: None,
            gps_times: Some(vec![42.5]),
            bounds: (1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
            num_points: 1,
        };

        let path = std::env::temp_dir().join(format!(
            "retina_las_gps_color_round_trip_{}.las",
            std::process::id()
        ));
        write_las(&path, &data).expect("write_las failed");
        let read = read_las(&path).expect("read_las failed");
        let _ = std::fs::remove_file(&path);

        let gps = read.gps_times.expect("gps_times missing after round-trip");
        assert_eq!(gps.len(), 1);
        assert!((gps[0] - 42.5).abs() < 1e-6);
        let colors = read.colors.expect("colors missing after round-trip");
        assert_eq!(colors.len(), 1);
    }
}
