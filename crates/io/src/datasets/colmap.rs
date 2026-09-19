//! COLMAP text-model dataset loader.
//!
//! Source of the formats below: the **COLMAP** documentation, "Output Format —
//! Text", <https://colmap.github.io/format.html>. Only the plain-text model is
//! supported; headers/comments begin with `#` and are skipped.
//!
//! ## `read_cameras_text` — `cameras.txt`
//!
//! ```text
//! CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
//! ```
//!
//! `MODEL` is one of the COLMAP camera models (`SIMPLE_PINHOLE`, `PINHOLE`,
//! `SIMPLE_RADIAL`, `OPENCV`, `RADIAL`, `OPENCV_FISHEYE`, `FULL_OPENCV`, `FOV`,
//! ...). `SIMPLE_PINHOLE` (`f, cx, cy`) and `PINHOLE` (`fx, fy, cx, cy`) are
//! mapped to [`cv_core::CameraIntrinsics`]; for every other model the raw model
//! name and parameter vector are preserved and `intrinsics` is `None`.
//!
//! ```text
//! 1 PINHOLE 640 480 500.0 500.0 320.0 240.0
//! ```
//!
//! ## `read_images_text` — `images.txt`
//!
//! Two lines per image. The first holds the registered pose (a world-to-camera
//! transform, i.e. COLMAP projects `x = R * X + t`) and camera/name metadata;
//! the second holds the 2-D observations as `X Y POINT3D_ID` triples where
//! `POINT3D_ID == -1` means "no triangulated 3-D point". The second line is
//! present even when it is empty, and a missing one is an error.
//!
//! ```text
//! IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
//! X Y POINT3D_ID [X Y POINT3D_ID ...]
//! ```
//!
//! ## `read_points3d_text` — `points3D.txt`
//!
//! ```text
//! POINT3D_ID X Y Z R G B ERROR TRACK[]
//! ```
//!
//! `TRACK[]` is a list of `IMAGE_ID POINT2D_IDX` pairs (possibly empty).

use crate::datasets::{parse_f64, unit_quaternion_from_wxyz};
use cv_core::{CameraIntrinsics, Error, Pose, Result};
use nalgebra::{Point3, Vector3};
use std::fs;
use std::path::Path;

/// A camera from `cameras.txt`.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera id.
    pub id: u32,
    /// Raw COLMAP model name (e.g. `PINHOLE`, `OPENCV`).
    pub model: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw model parameters, in the order defined by the COLMAP model.
    pub params: Vec<f64>,
    /// Pinhole intrinsics, populated for `SIMPLE_PINHOLE` and `PINHOLE` only.
    pub intrinsics: Option<CameraIntrinsics>,
}

/// A 2-D observation stored on an image's `POINTS2D[]` line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// Pixel x coordinate.
    pub x: f64,
    /// Pixel y coordinate.
    pub y: f64,
    /// Triangulated 3-D point id, or `-1` when the feature is not triangulated.
    pub point3d_id: i64,
}

/// An image from `images.txt`.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image id.
    pub id: u32,
    /// Registered pose (world-to-camera transform as stored by COLMAP).
    pub pose: Pose,
    /// Id of the camera this image was taken with.
    pub camera_id: u32,
    /// Image name (usually the file name).
    pub name: String,
    /// 2-D observations on this image.
    pub points2d: Vec<Point2D>,
}

/// A 3-D point from `points3D.txt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3D {
    /// Point id.
    pub id: u64,
    /// 3-D position in the world frame.
    pub position: Point3<f64>,
    /// RGB colour.
    pub color: [u8; 3],
    /// Mean reprojection error.
    pub error: f64,
    /// Observations as `(image_id, point2d_idx)` pairs.
    pub track: Vec<(u32, u32)>,
}

/// Read `cameras.txt` from a COLMAP text model.
///
/// Each non-comment line must contain at least `CAMERA_ID MODEL WIDTH HEIGHT`;
/// `SIMPLE_PINHOLE` requires 3 parameters and `PINHOLE` 4, otherwise an error is
/// returned. `WIDTH`/`HEIGHT` must be positive.
pub fn read_cameras_text<P: AsRef<Path>>(path: P) -> Result<Vec<Camera>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut cameras = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(Error::ParseError(format!(
                "{}: expected at least 4 fields (CAMERA_ID MODEL WIDTH HEIGHT), found {}",
                ctx,
                fields.len()
            )));
        }

        let id = fields[0].parse::<u32>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid camera id {:?}: {e}", fields[0]))
        })?;
        let model = fields[1].to_owned();
        let width = fields[2].parse::<u32>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid camera width {:?}: {e}", fields[2]))
        })?;
        let height = fields[3].parse::<u32>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid camera height {:?}: {e}", fields[3]))
        })?;
        if width == 0 || height == 0 {
            return Err(Error::InvalidInput(format!(
                "{ctx}: camera dimensions must be positive, got {width}x{height}"
            )));
        }

        let mut params = Vec::with_capacity(fields.len().saturating_sub(4));
        for field in &fields[4..] {
            params.push(parse_f64(field, &ctx)?);
        }

        let intrinsics = match model.as_str() {
            "SIMPLE_PINHOLE" => {
                if params.len() < 3 {
                    return Err(Error::ParseError(format!(
                        "{ctx}: SIMPLE_PINHOLE needs 3 params (f, cx, cy), found {}",
                        params.len()
                    )));
                }
                Some(CameraIntrinsics::new(
                    params[0], params[0], params[1], params[2], width, height,
                ))
            }
            "PINHOLE" => {
                if params.len() < 4 {
                    return Err(Error::ParseError(format!(
                        "{ctx}: PINHOLE needs 4 params (fx, fy, cx, cy), found {}",
                        params.len()
                    )));
                }
                Some(CameraIntrinsics::new(
                    params[0], params[1], params[2], params[3], width, height,
                ))
            }
            _ => None,
        };

        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
            intrinsics,
        });
    }

    Ok(cameras)
}

/// Read `images.txt` from a COLMAP text model.
///
/// The file alternates between an image header line
/// (`IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`) and a `POINTS2D[]` line of
/// `X Y POINT3D_ID` triples (possibly empty). A header without a following
/// points line is an error, as is a points line whose token count is not a
/// multiple of three.
pub fn read_images_text<P: AsRef<Path>>(path: P) -> Result<Vec<Image>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();

    let mut images = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        let line_no = i + 1;
        let ctx = format!("{}: line {}", path.display(), line_no);
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 9 {
            return Err(Error::ParseError(format!(
                "{}: expected at least 9 fields \
                 (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME), found {}",
                ctx,
                fields.len()
            )));
        }

        let id = fields[0].parse::<u32>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid image id {:?}: {e}", fields[0]))
        })?;
        let qw = parse_f64(fields[1], &ctx)?;
        let qx = parse_f64(fields[2], &ctx)?;
        let qy = parse_f64(fields[3], &ctx)?;
        let qz = parse_f64(fields[4], &ctx)?;
        let tx = parse_f64(fields[5], &ctx)?;
        let ty = parse_f64(fields[6], &ctx)?;
        let tz = parse_f64(fields[7], &ctx)?;
        let camera_id = fields[8].parse::<u32>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid camera id {:?}: {e}", fields[8]))
        })?;
        let name = fields[9..].join(" ");
        if name.is_empty() {
            return Err(Error::InvalidInput(format!("{ctx}: empty image name")));
        }

        let rotation = unit_quaternion_from_wxyz(qw, qx, qy, qz, &ctx)?;
        let pose = Pose::from_quat_translation(rotation, Vector3::new(tx, ty, tz));

        // The very next physical line is the POINTS2D[] line (possibly empty).
        if i + 1 >= lines.len() {
            return Err(Error::ParseError(format!(
                "{}: image {} (line {}) is missing its POINTS2D[] line",
                path.display(),
                id,
                line_no
            )));
        }
        let points_line = lines[i + 1];
        let points2d = parse_points2d(points_line, path, i + 2)?;

        images.push(Image {
            id,
            pose,
            camera_id,
            name,
            points2d,
        });

        i += 2;
    }

    Ok(images)
}

/// Parse one `POINTS2D[]` line into observations.
fn parse_points2d(line: &str, path: &Path, line_no: usize) -> Result<Vec<Point2D>> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    if !tokens.len().is_multiple_of(3) {
        return Err(Error::ParseError(format!(
            "{}: line {}: POINTS2D[] must be triples (X Y POINT3D_ID), found {} tokens",
            path.display(),
            line_no,
            tokens.len()
        )));
    }

    let ctx = format!("{}: line {}", path.display(), line_no);
    let mut points = Vec::with_capacity(tokens.len() / 3);
    for triple in tokens.as_chunks::<3>().0 {
        let x = parse_f64(triple[0], &ctx)?;
        let y = parse_f64(triple[1], &ctx)?;
        let point3d_id = triple[2].parse::<i64>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid POINT3D_ID {:?}: {e}", triple[2]))
        })?;
        points.push(Point2D { x, y, point3d_id });
    }
    Ok(points)
}

/// Read `points3D.txt` from a COLMAP text model.
///
/// Each non-comment line must contain at least
/// `POINT3D_ID X Y Z R G B ERROR`, optionally followed by an even number of
/// `IMAGE_ID POINT2D_IDX` track entries. `R`, `G`, `B` must be in `0..=255`.
pub fn read_points3d_text<P: AsRef<Path>>(path: P) -> Result<Vec<Point3D>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut points = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 8 {
            return Err(Error::ParseError(format!(
                "{}: expected at least 8 fields \
                 (POINT3D_ID X Y Z R G B ERROR), found {}",
                ctx,
                fields.len()
            )));
        }

        let id = fields[0].parse::<u64>().map_err(|e| {
            Error::ParseError(format!("{ctx}: invalid POINT3D_ID {:?}: {e}", fields[0]))
        })?;
        let x = parse_f64(fields[1], &ctx)?;
        let y = parse_f64(fields[2], &ctx)?;
        let z = parse_f64(fields[3], &ctx)?;
        let mut color = [0u8; 3];
        for (slot, field) in color.iter_mut().zip(fields[4..7].iter()) {
            *slot = field.parse::<u8>().map_err(|e| {
                Error::ParseError(format!("{ctx}: invalid colour channel {field:?}: {e}"))
            })?;
        }
        let error = parse_f64(fields[7], &ctx)?;

        let track_fields = &fields[8..];
        if !track_fields.len().is_multiple_of(2) {
            return Err(Error::ParseError(format!(
                "{}: TRACK[] must be IMAGE_ID POINT2D_IDX pairs, found {} trailing tokens",
                ctx,
                track_fields.len()
            )));
        }
        let mut track = Vec::with_capacity(track_fields.len() / 2);
        for pair in track_fields.as_chunks::<2>().0 {
            let image_id = pair[0].parse::<u32>().map_err(|e| {
                Error::ParseError(format!("{ctx}: invalid track image id {:?}: {e}", pair[0]))
            })?;
            let point2d_idx = pair[1].parse::<u32>().map_err(|e| {
                Error::ParseError(format!(
                    "{ctx}: invalid track point2D idx {:?}: {e}",
                    pair[1]
                ))
            })?;
            track.push((image_id, point2d_idx));
        }

        points.push(Point3D {
            id,
            position: Point3::new(x, y, z),
            color,
            error,
            track,
        });
    }

    Ok(points)
}

/// Write a COLMAP text-model `cameras.txt`.
///
/// Emits the same comment header COLMAP writes, followed by one line per camera
/// in `CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]` order. [`Camera::model`] and
/// [`Camera::params`] are written verbatim; the derived [`Camera::intrinsics`]
/// field is not stored (it is reconstructed by [`read_cameras_text`]). Scalars
/// use the round-trip-exact `{:?}` representation. The file ends with a trailing
/// newline.
///
/// # Errors
///
/// Returns [`cv_core::Error::IoError`] if `path` cannot be written.
pub fn write_cameras_text<P: AsRef<Path>>(path: P, cameras: &[Camera]) -> Result<()> {
    let mut out = String::new();
    out.push_str("# Camera list with one line of data per camera:\n");
    out.push_str("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    for camera in cameras {
        let (id, model, width, height) = (
            camera.id,
            camera.model.as_str(),
            camera.width,
            camera.height,
        );
        out.push_str(&format!("{id} {model} {width} {height}"));
        for param in &camera.params {
            out.push_str(&format!(" {param:?}"));
        }
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

/// Write a COLMAP text-model `images.txt`.
///
/// Emits the same comment header COLMAP writes, then two lines per image: the
/// pose/metadata header (`IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`) and the
/// alternating `POINTS2D[]` line of `X Y POINT3D_ID` triples. An image with no
/// observations produces an empty (blank) `POINTS2D[]` line, matching COLMAP.
/// Quaternion components are written in COLMAP's `(QW, QX, QY, QZ)` order. The
/// file ends with a trailing newline.
///
/// # Errors
///
/// Returns [`cv_core::Error::IoError`] if `path` cannot be written.
pub fn write_images_text<P: AsRef<Path>>(path: P, images: &[Image]) -> Result<()> {
    let mut out = String::new();
    out.push_str("# Image list with two lines of data per image:\n");
    out.push_str("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
    out.push_str("#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
    for image in images {
        let rotation = &image.pose.rotation;
        let translation = &image.pose.translation;
        let (id, camera_id, name) = (image.id, image.camera_id, image.name.as_str());
        let (qw, qx, qy, qz) = (rotation.w, rotation.i, rotation.j, rotation.k);
        let (tx, ty, tz) = (translation.x, translation.y, translation.z);
        out.push_str(&format!(
            "{id} {qw:?} {qx:?} {qy:?} {qz:?} {tx:?} {ty:?} {tz:?} {camera_id} {name}\n"
        ));
        for (idx, point) in image.points2d.iter().enumerate() {
            if idx > 0 {
                out.push(' ');
            }
            let (x, y, point3d_id) = (point.x, point.y, point.point3d_id);
            out.push_str(&format!("{x:?} {y:?} {point3d_id}"));
        }
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

/// Write a COLMAP text-model `points3D.txt`.
///
/// Emits the same comment header COLMAP writes, followed by one line per point:
/// `POINT3D_ID X Y Z R G B ERROR TRACK[]`, where `R G B` are integers in
/// `0..=255` and `TRACK[]` is the `IMAGE_ID POINT2D_IDX` pair list (empty when
/// [`Point3D::track`] is empty). The file ends with a trailing newline.
///
/// # Errors
///
/// Returns [`cv_core::Error::IoError`] if `path` cannot be written.
pub fn write_points3d_text<P: AsRef<Path>>(path: P, points: &[Point3D]) -> Result<()> {
    let mut out = String::new();
    out.push_str("# 3D point list with one line of data per point:\n");
    out.push_str("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    for point in points {
        let id = point.id;
        let (x, y, z) = (point.position.x, point.position.y, point.position.z);
        let [r, g, b] = point.color;
        let error = point.error;
        out.push_str(&format!("{id} {x:?} {y:?} {z:?} {r} {g} {b} {error:?}"));
        for (image_id, point2d_idx) in &point.track {
            out.push_str(&format!(" {image_id} {point2d_idx}"));
        }
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::test_util::TempDir;
    use crate::datasets::unit_quaternion_from_wxyz;

    #[test]
    fn colmap_read_cameras_maps_pinhole_models() {
        let dir = TempDir::new("colmap_cameras");
        let path = dir.write(
            "cameras.txt",
            concat!(
                "# Camera list with one line of data per camera:\n",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
                "1 SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n",
                "2 PINHOLE 1280 720 800.0 810.0 640.0 360.0\n",
            ),
        );

        let cameras = read_cameras_text(&path).expect("parse cameras");
        assert_eq!(cameras.len(), 2);

        let simple = &cameras[0];
        assert_eq!(simple.id, 1);
        assert_eq!(simple.model, "SIMPLE_PINHOLE");
        assert_eq!(simple.width, 640);
        assert_eq!(simple.height, 480);
        assert_eq!(simple.params, vec![500.0, 320.0, 240.0]);
        let intr = simple.intrinsics.expect("simple pinhole intrinsics");
        assert_eq!(
            (intr.fx, intr.fy, intr.cx, intr.cy),
            (500.0, 500.0, 320.0, 240.0)
        );
        assert_eq!((intr.width, intr.height), (640, 480));

        let pinhole = &cameras[1];
        let intr = pinhole.intrinsics.expect("pinhole intrinsics");
        assert_eq!(
            (intr.fx, intr.fy, intr.cx, intr.cy),
            (800.0, 810.0, 640.0, 360.0)
        );
    }

    #[test]
    fn colmap_read_cameras_keeps_raw_model_for_others() {
        let dir = TempDir::new("colmap_cameras_raw");
        let path = dir.write(
            "cameras.txt",
            "3 OPENCV 640 480 500.0 500.0 320.0 240.0 0.1 -0.2 0.001 0.002\n",
        );

        let cameras = read_cameras_text(&path).expect("parse cameras");
        assert_eq!(cameras.len(), 1);
        assert_eq!(cameras[0].model, "OPENCV");
        assert_eq!(cameras[0].params.len(), 8);
        assert!(cameras[0].intrinsics.is_none());
    }

    #[test]
    fn colmap_read_cameras_rejects_too_few_fields() {
        let dir = TempDir::new("colmap_cameras_short");
        let path = dir.write("cameras.txt", "1 PINHOLE 640\n");
        let err = read_cameras_text(&path).expect_err("short line must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn colmap_read_cameras_rejects_wrong_param_count() {
        let dir = TempDir::new("colmap_cameras_params");
        // PINHOLE with only 3 params.
        let path = dir.write("cameras.txt", "1 PINHOLE 640 480 500.0 320.0 240.0\n");
        assert!(read_cameras_text(&path).is_err());
    }

    #[test]
    fn colmap_read_images_parses_pose_and_points() {
        let dir = TempDir::new("colmap_images");
        let path = dir.write(
            "images.txt",
            concat!(
                "# Image list with two lines of data per image:\n",
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
                "1 1 0 0 0 0 0 0 1 image1.jpg\n",
                "10.0 20.0 1 30.0 40.0 2 -1.0 -1.0 -1\n",
                "2 1 0 0 0 1 2 3 1 image2.jpg\n",
                "\n",
            ),
        );

        let images = read_images_text(&path).expect("parse images");
        assert_eq!(images.len(), 2);

        assert_eq!(images[0].id, 1);
        assert_eq!(images[0].camera_id, 1);
        assert_eq!(images[0].name, "image1.jpg");
        assert_eq!(images[0].points2d.len(), 3);
        assert_eq!(
            images[0].points2d[0],
            Point2D {
                x: 10.0,
                y: 20.0,
                point3d_id: 1
            }
        );
        assert_eq!(images[0].points2d[2].point3d_id, -1);

        assert_eq!(images[1].name, "image2.jpg");
        assert_eq!(images[1].pose.translation, Vector3::new(1.0, 2.0, 3.0));
        assert!(images[1].points2d.is_empty());
    }

    #[test]
    fn colmap_read_images_rejects_missing_points_line() {
        let dir = TempDir::new("colmap_images_missing");
        // The second image has no POINTS2D[] line before EOF.
        let path = dir.write(
            "images.txt",
            concat!(
                "1 1 0 0 0 0 0 0 1 image1.jpg\n",
                "10.0 20.0 1\n",
                "2 1 0 0 0 1 2 3 1 image2.jpg\n",
            ),
        );
        let err = read_images_text(&path).expect_err("missing points line must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn colmap_read_images_rejects_non_multiple_of_three() {
        let dir = TempDir::new("colmap_images_triples");
        let path = dir.write(
            "images.txt",
            concat!("1 1 0 0 0 0 0 0 1 image1.jpg\n", "10.0 20.0 1 30.0 40.0\n",),
        );
        assert!(read_images_text(&path).is_err());
    }

    #[test]
    fn colmap_read_images_missing_file_is_err() {
        let dir = TempDir::new("colmap_images_missing_file");
        assert!(read_images_text(dir.missing("images.txt")).is_err());
    }

    #[test]
    fn colmap_read_points3d_parses_track() {
        let dir = TempDir::new("colmap_points3d");
        let path = dir.write(
            "points3D.txt",
            concat!(
                "# 3D point list with one line of data per point:\n",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
                "1 0.5 -1.5 2.5 255 128 0 0.5 1 0 2 3\n",
                "2 -1.0 0.0 1.0 0 255 0 1.25\n",
            ),
        );

        let points = read_points3d_text(&path).expect("parse points3D");
        assert_eq!(points.len(), 2);

        let p = &points[0];
        assert_eq!(p.id, 1);
        assert_eq!(p.position, Point3::new(0.5, -1.5, 2.5));
        assert_eq!(p.color, [255, 128, 0]);
        assert_eq!(p.error, 0.5);
        assert_eq!(p.track, vec![(1, 0), (2, 3)]);

        assert_eq!(points[1].color, [0, 255, 0]);
        assert!(points[1].track.is_empty());
    }

    #[test]
    fn colmap_read_points3d_rejects_odd_track() {
        let dir = TempDir::new("colmap_points3d_odd");
        let path = dir.write("points3D.txt", "1 0 0 0 255 255 255 1.0 1 0 2\n");
        let err = read_points3d_text(&path).expect_err("odd track must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn colmap_read_points3d_rejects_bad_colour() {
        let dir = TempDir::new("colmap_points3d_colour");
        let path = dir.write("points3D.txt", "1 0 0 0 999 255 255 1.0\n");
        assert!(read_points3d_text(&path).is_err());
    }

    #[test]
    fn colmap_read_points3d_missing_file_is_err() {
        let dir = TempDir::new("colmap_points3d_missing");
        assert!(read_points3d_text(dir.missing("points3D.txt")).is_err());
    }

    /// A `SIMPLE_PINHOLE` plus a `PINHOLE` camera, with the `intrinsics` values
    /// the reader derives for them, so a round trip can assert full equality.
    fn sample_cameras() -> Vec<Camera> {
        vec![
            Camera {
                id: 1,
                model: "SIMPLE_PINHOLE".to_owned(),
                width: 640,
                height: 480,
                params: vec![500.0, 320.0, 240.0],
                intrinsics: Some(CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)),
            },
            Camera {
                id: 2,
                model: "PINHOLE".to_owned(),
                width: 1280,
                height: 720,
                params: vec![800.0, 810.0, 640.0, 360.0],
                intrinsics: Some(CameraIntrinsics::new(800.0, 810.0, 640.0, 360.0, 1280, 720)),
            },
        ]
    }

    #[test]
    fn colmap_write_cameras_text_round_trips() {
        let dir = TempDir::new("colmap_write_cameras");
        let path = dir.missing("cameras.txt");
        let original = sample_cameras();

        write_cameras_text(&path, &original).expect("write cameras");

        let text = fs::read_to_string(&path).expect("read raw cameras");
        assert!(text.starts_with(
            "# Camera list with one line of data per camera:\n\
             #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        ));
        assert!(
            text.ends_with('\n'),
            "file must end with a trailing newline"
        );

        let parsed = read_cameras_text(&path).expect("parse cameras");
        assert_eq!(parsed.len(), original.len());
        for (got, want) in parsed.iter().zip(original.iter()) {
            assert_eq!(got.id, want.id);
            assert_eq!(got.model, want.model);
            assert_eq!(got.width, want.width);
            assert_eq!(got.height, want.height);
            assert_eq!(got.params, want.params);
            match (got.intrinsics, want.intrinsics) {
                (Some(g), Some(w)) => assert_eq!(
                    (g.fx, g.fy, g.cx, g.cy, g.width, g.height),
                    (w.fx, w.fy, w.cx, w.cy, w.width, w.height)
                ),
                (None, None) => {}
                _ => panic!("intrinsics presence mismatch after round trip"),
            }
        }
    }

    /// One image with several observations (including a `-1` point id) and one
    /// image with zero observations (so its `POINTS2D[]` line must be blank).
    fn sample_images() -> Vec<Image> {
        vec![
            Image {
                id: 1,
                pose: Pose::from_quat_translation(
                    unit_quaternion_from_wxyz(0.5, 0.5, 0.5, 0.5, "test").unwrap(),
                    Vector3::new(1.0, 2.0, 3.0),
                ),
                camera_id: 1,
                name: "image.jpg".to_owned(),
                points2d: vec![
                    Point2D {
                        x: 100.0,
                        y: 200.0,
                        point3d_id: 1,
                    },
                    Point2D {
                        x: 300.0,
                        y: 400.0,
                        point3d_id: -1,
                    },
                    Point2D {
                        x: 12.5,
                        y: -7.25,
                        point3d_id: 42,
                    },
                ],
            },
            Image {
                id: 2,
                pose: Pose::from_quat_translation(
                    unit_quaternion_from_wxyz(1.0, 0.0, 0.0, 0.0, "test").unwrap(),
                    Vector3::new(0.0, 0.0, 0.0),
                ),
                camera_id: 1,
                name: "empty.png".to_owned(),
                points2d: vec![],
            },
        ]
    }

    #[test]
    fn colmap_write_images_text_round_trips() {
        let dir = TempDir::new("colmap_write_images");
        let path = dir.missing("images.txt");
        let original = sample_images();

        write_images_text(&path, &original).expect("write images");

        let text = fs::read_to_string(&path).expect("read raw images");
        assert!(text.starts_with(
            "# Image list with two lines of data per image:\n\
             #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
             #   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        ));
        assert!(
            text.ends_with('\n'),
            "file must end with a trailing newline"
        );

        // 3 comment lines + 2 lines per image; the last image has no
        // observations, so its POINTS2D[] line is blank.
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3 + 2 * original.len());
        assert_eq!(lines[3], "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 image.jpg");
        assert_eq!(lines[4], "100.0 200.0 1 300.0 400.0 -1 12.5 -7.25 42");
        assert_eq!(lines[5], "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 empty.png");
        assert_eq!(
            lines[6], "",
            "an image with no observations gets a blank line"
        );

        let parsed = read_images_text(&path).expect("parse images");
        assert_eq!(parsed.len(), original.len());
        for (got, want) in parsed.iter().zip(original.iter()) {
            assert_eq!(got.id, want.id);
            assert_eq!(got.camera_id, want.camera_id);
            assert_eq!(got.name, want.name);
            assert_eq!(got.points2d, want.points2d);
            let got_q = (
                got.pose.rotation.w,
                got.pose.rotation.i,
                got.pose.rotation.j,
                got.pose.rotation.k,
            );
            let want_q = (
                want.pose.rotation.w,
                want.pose.rotation.i,
                want.pose.rotation.j,
                want.pose.rotation.k,
            );
            assert_eq!(got_q, want_q);
            assert_eq!(got.pose.translation, want.pose.translation);
        }
    }

    #[test]
    fn colmap_write_points3d_text_round_trips() {
        let dir = TempDir::new("colmap_write_points3d");
        let path = dir.missing("points3D.txt");
        let original = vec![
            Point3D {
                id: 1,
                position: Point3::new(1.0, 2.0, 3.0),
                color: [255, 128, 0],
                error: 0.5,
                track: vec![(1, 0), (2, 4), (3, 2)],
            },
            Point3D {
                id: 2,
                position: Point3::new(-1.25, 0.0, 4.5),
                color: [0, 0, 255],
                error: 1.75,
                track: vec![],
            },
        ];

        write_points3d_text(&path, &original).expect("write points3D");

        let text = fs::read_to_string(&path).expect("read raw points3D");
        assert!(text.starts_with(
            "# 3D point list with one line of data per point:\n\
             #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, \
             TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        ));
        assert!(
            text.ends_with('\n'),
            "file must end with a trailing newline"
        );

        let parsed = read_points3d_text(&path).expect("parse points3D");
        assert_eq!(parsed, original);
    }

    #[test]
    fn colmap_write_unwritable_path_is_err() {
        let dir = TempDir::new("colmap_write_err");
        // The parent directory does not exist, so the write must fail cleanly.
        let unwritable = dir.path().join("no_such_dir").join("model.txt");

        assert!(matches!(
            write_cameras_text(&unwritable, &sample_cameras()),
            Err(Error::IoError(_))
        ));
        assert!(matches!(
            write_images_text(&unwritable, &sample_images()),
            Err(Error::IoError(_))
        ));
        assert!(matches!(
            write_points3d_text(
                &unwritable,
                &[Point3D {
                    id: 1,
                    position: Point3::new(0.0, 0.0, 0.0),
                    color: [0, 0, 0],
                    error: 0.0,
                    track: vec![],
                }]
            ),
            Err(Error::IoError(_))
        ));
    }
}
