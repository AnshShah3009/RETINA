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
    let bounds = header.bounds();
    let num_points = header.number_of_points() as usize;

    let mut points = Vec::with_capacity(num_points);
    let mut colors_vec: Vec<Point3<f32>> = Vec::new();
    let mut intensities_vec: Vec<f32> = Vec::new();
    let mut classifications_vec: Vec<u8> = Vec::new();
    let mut return_numbers_vec: Vec<u8> = Vec::new();
    let mut number_of_returns_vec: Vec<u8> = Vec::new();
    let mut gps_times_vec: Vec<f64> = Vec::new();

    let mut has_color = false;
    let mut has_intensity = false;
    let mut has_classification = false;
    let mut has_returns = false;
    let mut has_gps = false;
    let mut first = true;

    for point_result in reader.points() {
        let point = point_result
            .map_err(|e| cv_core::Error::IoError(format!("Failed to read LAS point: {}", e)))?;

        points.push(Point3::new(point.x as f32, point.y as f32, point.z as f32));

        // Detect available fields from first point
        if first {
            has_color = point.color.is_some();
            has_intensity = true; // intensity is always present in LAS
            has_classification = true;
            has_returns = true;
            has_gps = point.gps_time.is_some();
            first = false;
        }

        if has_intensity {
            intensities_vec.push(point.intensity as f32 / 65535.0);
        }

        if has_color {
            if let Some(color) = point.color {
                colors_vec.push(Point3::new(
                    color.red as f32 / 65535.0,
                    color.green as f32 / 65535.0,
                    color.blue as f32 / 65535.0,
                ));
            }
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
            if let Some(t) = point.gps_time {
                gps_times_vec.push(t);
            }
        }
    }

    Ok(LasData {
        num_points: points.len(),
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
        bounds: (
            bounds.min.x,
            bounds.min.y,
            bounds.min.z,
            bounds.max.x,
            bounds.max.y,
            bounds.max.z,
        ),
    })
}

/// Write a LAS file (uncompressed).
pub fn write_las<P: AsRef<Path>>(path: P, data: &LasData) -> cv_core::Result<()> {
    use las::{Builder, Point, Write, Writer};

    let mut builder = Builder::from((1, 4)); // LAS 1.4
    builder.point_format = if data.colors.is_some() {
        las::point::Format::new(2).unwrap() // Format 2: XYZ + RGB
    } else {
        las::point::Format::new(0).unwrap() // Format 0: XYZ only
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
    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in &cloud.points {
        min.x = min.x.min(p.x as f64);
        min.y = min.y.min(p.y as f64);
        min.z = min.z.min(p.z as f64);
        max.x = max.x.max(p.x as f64);
        max.y = max.y.max(p.y as f64);
        max.z = max.z.max(p.z as f64);
    }

    LasData {
        num_points: cloud.points.len(),
        points: cloud.points.clone(),
        colors: cloud.colors.clone(),
        intensities: None,
        classifications: None,
        return_numbers: None,
        number_of_returns: None,
        gps_times: None,
        bounds: (min.x, min.y, min.z, max.x, max.y, max.z),
    }
}

/// Filter LAS data by classification code.
pub fn filter_by_classification(data: &LasData, class: u8) -> LasData {
    let classifications = match &data.classifications {
        Some(c) => c,
        None => return data.clone(),
    };

    let mask: Vec<bool> = classifications.iter().map(|&c| c == class).collect();
    filter_by_mask(data, &mask)
}

/// Filter LAS data by a boolean mask.
pub fn filter_by_mask(data: &LasData, mask: &[bool]) -> LasData {
    let points: Vec<_> = data
        .points
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(p, _)| *p)
        .collect();

    fn filter_vec<T: Clone>(opt: &Option<Vec<T>>, mask: &[bool]) -> Option<Vec<T>> {
        opt.as_ref().map(|v| {
            v.iter()
                .zip(mask.iter())
                .filter(|(_, &m)| m)
                .map(|(val, _)| val.clone())
                .collect()
        })
    }

    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in &points {
        min.x = min.x.min(p.x as f64);
        min.y = min.y.min(p.y as f64);
        min.z = min.z.min(p.z as f64);
        max.x = max.x.max(p.x as f64);
        max.y = max.y.max(p.y as f64);
        max.z = max.z.max(p.z as f64);
    }

    LasData {
        num_points: points.len(),
        points,
        colors: filter_vec(&data.colors, mask),
        intensities: filter_vec(&data.intensities, mask),
        classifications: filter_vec(&data.classifications, mask),
        return_numbers: filter_vec(&data.return_numbers, mask),
        number_of_returns: filter_vec(&data.number_of_returns, mask),
        gps_times: filter_vec(&data.gps_times, mask),
        bounds: (min.x, min.y, min.z, max.x, max.y, max.z),
    }
}
