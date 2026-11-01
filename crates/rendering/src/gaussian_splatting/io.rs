//! Gaussian-splat PLY input/output in the INRIA 3DGS layout.
//!
//! The reference implementation (Kerbl et al. 2023, `GaussianModel.save_ply`)
//! writes one `vertex` element with the properties
//!
//! ```text
//! x y z nx ny nz f_dc_0..2 f_rest_0..(3*((D+1)^2-1)-1) opacity scale_0..2 rot_0..3
//! ```
//!
//! as `binary_little_endian` floats, storing every parameter *before* its
//! activation:
//!
//! | file column | stored as                           | this crate's [`Gaussian`]                    |
//! |-------------|-------------------------------------|----------------------------------------------|
//! | `opacity`   | logit                               | `opacity = sigmoid(raw)`                     |
//! | `scale_i`   | natural log                         | `scale = exp(raw)`                           |
//! | `f_dc_c`    | degree-0 SH coefficient             | colour `0.5 + C0 * raw` in `coeffs[0..3]`    |
//! | `f_rest_k`  | channel-major (all R, all G, all B) | interleaved `[R1 G1 B1 R2 ...]` from `coeffs[3]` |
//! | `rot_0..3`  | quaternion `w, x, y, z`             | `rotation = (x, y, z, w)`                    |
//!
//! `C0 = 0.28209479` is the degree-0 SH basis constant, and `+ 0.5` is the
//! offset INRIA adds to the evaluated SH before using it as an RGB colour. This
//! crate renders `SphericalHarmonics::dc()` directly as the colour, so the reader
//! stores the *activated* colour there and the writer inverts it.
//!
//! The reader maps columns **by property name**, so any column order and any
//! scalar property type is accepted, in `ascii`, `binary_little_endian` and
//! `binary_big_endian`. A file that lacks one of the 3DGS properties, is
//! truncated, or declares an unknown format is an error rather than an empty or
//! garbled cloud.

use std::path::Path;

use nalgebra::{Point3, Vector3, Vector4};

use super::types::{Gaussian, GaussianCloud, SphericalHarmonics, unit_quaternion};

/// Degree-0 real spherical-harmonic basis constant, `1 / (2 sqrt(pi))`.
pub const SH_C0: f32 = 0.282_094_8;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Clone, Copy, Debug)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            other => return Err(format!("Unknown PLY property type '{other}'")),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug)]
enum Property {
    Scalar(ScalarType, String),
    List(ScalarType, ScalarType),
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

/// A stream of property values, either ASCII tokens or binary scalars.
trait ValueSource {
    fn next_value(&mut self, ty: ScalarType) -> Result<f64, String>;
}

struct AsciiSource<'a> {
    tokens: std::str::SplitAsciiWhitespace<'a>,
}

impl ValueSource for AsciiSource<'_> {
    fn next_value(&mut self, _ty: ScalarType) -> Result<f64, String> {
        let tok = self
            .tokens
            .next()
            .ok_or("PLY body ended before all declared elements were read")?;
        tok.parse::<f64>()
            .map_err(|_| format!("Invalid PLY value '{tok}'"))
    }
}

struct BinarySource<'a> {
    data: &'a [u8],
    pos: usize,
    little_endian: bool,
}

impl ValueSource for BinarySource<'_> {
    fn next_value(&mut self, ty: ScalarType) -> Result<f64, String> {
        let n = ty.size();
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(format!(
                "PLY binary body truncated: needed {} bytes at offset {}, have {}",
                n,
                self.pos,
                self.data.len()
            ));
        }
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(&self.data[self.pos..end]);
        if !self.little_endian {
            b[..n].reverse();
        }
        self.pos = end;
        Ok(match ty {
            ScalarType::I8 => b[0] as i8 as f64,
            ScalarType::U8 => b[0] as f64,
            ScalarType::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            ScalarType::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            ScalarType::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            ScalarType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            ScalarType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            ScalarType::F64 => f64::from_le_bytes(b),
        })
    }
}

/// Parse the header: format, elements, and the byte offset of the body.
fn parse_header(bytes: &[u8]) -> Result<(Format, Vec<Element>, usize), String> {
    let mut pos = 0;
    let mut next_line = || -> Option<String> {
        if pos >= bytes.len() {
            return None;
        }
        let rest = &bytes[pos..];
        let len = rest.iter().position(|&c| c == b'\n').unwrap_or(rest.len());
        let line = String::from_utf8_lossy(&rest[..len])
            .trim_end_matches('\r')
            .to_string();
        pos += (len + 1).min(rest.len());
        Some(line)
    };

    match next_line() {
        Some(l) if l.trim() == "ply" => {}
        _ => return Err("Not a PLY file".to_string()),
    }

    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        let line = next_line().ok_or("PLY header has no end_header")?;
        let mut words = line.split_whitespace();
        match words.next() {
            Some("format") => {
                format = Some(match words.next() {
                    Some("ascii") => Format::Ascii,
                    Some("binary_little_endian") => Format::BinaryLittleEndian,
                    Some("binary_big_endian") => Format::BinaryBigEndian,
                    other => return Err(format!("Unsupported PLY format {other:?}")),
                });
            }
            Some("element") => {
                let name = words.next().ok_or("Invalid element line")?.to_string();
                let count = words
                    .next()
                    .ok_or("Invalid element line")?
                    .parse()
                    .map_err(|_| "Invalid element count")?;
                elements.push(Element {
                    name,
                    count,
                    properties: Vec::new(),
                });
            }
            Some("property") => {
                let el = elements
                    .last_mut()
                    .ok_or("PLY property declared before any element")?;
                let ty = words.next().ok_or("Invalid property line")?;
                if ty == "list" {
                    let count_ty = ScalarType::parse(words.next().ok_or("Invalid list property")?)?;
                    let item_ty = ScalarType::parse(words.next().ok_or("Invalid list property")?)?;
                    el.properties.push(Property::List(count_ty, item_ty));
                } else {
                    let ty = ScalarType::parse(ty)?;
                    let name = words.next().ok_or("Invalid property line")?.to_string();
                    el.properties.push(Property::Scalar(ty, name));
                }
            }
            Some("end_header") => break,
            // `comment`, `obj_info` and blank lines carry no structure.
            _ => {}
        }
    }

    let format = format.ok_or("PLY header has no format line")?;
    Ok((format, elements, pos))
}

/// Indices of the 3DGS properties within one vertex record.
struct Columns {
    position: [usize; 3],
    normal: Option<[usize; 3]>,
    f_dc: [usize; 3],
    f_rest: Vec<usize>,
    opacity: usize,
    scale: [usize; 3],
    rot: [usize; 4],
}

impl Columns {
    fn resolve(vertex: &Element) -> Result<Self, String> {
        // Index among *scalar* properties, which is how a record is collected.
        let names: Vec<&str> = vertex
            .properties
            .iter()
            .filter_map(|p| match p {
                Property::Scalar(_, n) => Some(n.as_str()),
                Property::List(..) => None,
            })
            .collect();
        let find = |n: &str| names.iter().position(|&x| x == n);
        let need = |n: &str| {
            find(n).ok_or_else(|| format!("Gaussian PLY is missing vertex property '{n}'"))
        };
        let normal = match (find("nx"), find("ny"), find("nz")) {
            (Some(a), Some(b), Some(c)) => Some([a, b, c]),
            _ => None,
        };
        let mut f_rest = Vec::new();
        while let Some(i) = find(&format!("f_rest_{}", f_rest.len())) {
            f_rest.push(i);
        }
        if f_rest.len() % 3 != 0 {
            return Err(format!(
                "Gaussian PLY has {} f_rest properties, not a multiple of 3",
                f_rest.len()
            ));
        }
        Ok(Self {
            position: [need("x")?, need("y")?, need("z")?],
            normal,
            f_dc: [need("f_dc_0")?, need("f_dc_1")?, need("f_dc_2")?],
            f_rest,
            opacity: need("opacity")?,
            scale: [need("scale_0")?, need("scale_1")?, need("scale_2")?],
            rot: [
                need("rot_0")?,
                need("rot_1")?,
                need("rot_2")?,
                need("rot_3")?,
            ],
        })
    }

    fn gaussian(&self, row: &[f32]) -> Gaussian {
        let v = |i: usize| row[i];
        let position = Point3::new(
            v(self.position[0]),
            v(self.position[1]),
            v(self.position[2]),
        );
        let features = self
            .normal
            .map(|[a, b, c]| Vector3::new(v(a), v(b), v(c)))
            .unwrap_or_else(Vector3::zeros);

        let rest_per_channel = self.f_rest.len() / 3;
        let mut coeffs = Vec::with_capacity(3 * (1 + rest_per_channel));
        for c in 0..3 {
            coeffs.push(0.5 + SH_C0 * v(self.f_dc[c]));
        }
        for j in 0..rest_per_channel {
            for c in 0..3 {
                coeffs.push(v(self.f_rest[c * rest_per_channel + j]));
            }
        }
        // (degree + 1)^2 basis functions per channel. A count that is not a
        // square leaves the extra coefficients stored but unused, which is what
        // `SphericalHarmonics::eval` does with a short buffer as well.
        let degree = ((rest_per_channel + 1) as f64).sqrt().floor() as usize - 1;

        // File order is (w, x, y, z); this crate's `Vector4` is (x, y, z, w).
        let rotation = unit_quaternion(&Vector4::new(
            v(self.rot[1]),
            v(self.rot[2]),
            v(self.rot[3]),
            v(self.rot[0]),
        ));

        Gaussian {
            position,
            scale: Vector3::new(
                v(self.scale[0]).exp(),
                v(self.scale[1]).exp(),
                v(self.scale[2]).exp(),
            ),
            rotation,
            opacity: 1.0 / (1.0 + (-v(self.opacity)).exp()),
            spherical_harmonics: SphericalHarmonics { coeffs, degree },
            features,
        }
    }
}

/// Read a Gaussian-splat PLY (see the module docs for the layout).
pub fn read_ply_gaussian_cloud<P: AsRef<Path>>(path: P) -> Result<GaussianCloud, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to open file: {}", e))?;
    parse_ply_gaussian_cloud(&bytes)
}

/// Parse an in-memory Gaussian-splat PLY (see the module docs for the layout).
pub fn parse_ply_gaussian_cloud(bytes: &[u8]) -> Result<GaussianCloud, String> {
    let (format, elements, body_start) = parse_header(bytes)?;
    let vertex_pos = elements
        .iter()
        .position(|e| e.name == "vertex")
        .ok_or("PLY has no vertex element")?;
    let columns = Columns::resolve(&elements[vertex_pos])?;

    let body = &bytes[body_start..];
    let mut ascii;
    let mut binary;
    let source: &mut dyn ValueSource = match format {
        Format::Ascii => {
            let text = std::str::from_utf8(body).map_err(|_| "ASCII PLY body is not UTF-8")?;
            ascii = AsciiSource {
                tokens: text.split_ascii_whitespace(),
            };
            &mut ascii
        }
        Format::BinaryLittleEndian | Format::BinaryBigEndian => {
            binary = BinarySource {
                data: body,
                pos: 0,
                little_endian: format == Format::BinaryLittleEndian,
            };
            &mut binary
        }
    };

    let mut cloud = GaussianCloud::new();
    let mut row: Vec<f32> = Vec::new();
    // Elements before `vertex` are read and discarded; those after it are
    // never reached.
    for (ei, element) in elements.iter().enumerate().take(vertex_pos + 1) {
        let is_vertex = ei == vertex_pos;
        for _ in 0..element.count {
            row.clear();
            for prop in &element.properties {
                match prop {
                    Property::Scalar(ty, _) => {
                        let v = source.next_value(*ty)?;
                        if is_vertex {
                            row.push(v as f32);
                        }
                    }
                    Property::List(count_ty, item_ty) => {
                        let n = source.next_value(*count_ty)?;
                        if !(n >= 0.0 && n.fract() == 0.0) {
                            return Err(format!("Invalid PLY list length {n}"));
                        }
                        for _ in 0..n as usize {
                            source.next_value(*item_ty)?;
                        }
                    }
                }
            }
            if is_vertex {
                cloud.push(columns.gaussian(&row));
            }
        }
    }

    Ok(cloud)
}

/// Higher-order SH coefficients per colour channel written for `cloud`.
fn rest_per_channel(cloud: &GaussianCloud) -> usize {
    cloud
        .gaussians
        .iter()
        .map(|g| g.spherical_harmonics.coeffs.len().saturating_sub(3) / 3)
        .max()
        .unwrap_or(0)
}

/// One record, in the column order of [`ply_header`].
fn gaussian_record(g: &Gaussian, rest: usize) -> Vec<f32> {
    let coeffs = &g.spherical_harmonics.coeffs;
    let coeff = |i: usize| coeffs.get(i).copied().unwrap_or(0.0);
    let mut row = Vec::with_capacity(17 + 3 * rest);
    row.extend_from_slice(&[g.position.x, g.position.y, g.position.z]);
    row.extend_from_slice(&[g.features.x, g.features.y, g.features.z]);
    for c in 0..3 {
        row.push((coeff(c) - 0.5) / SH_C0);
    }
    // Channel-major, as INRIA stores `f_rest`.
    for c in 0..3 {
        for j in 0..rest {
            row.push(coeff(3 * (j + 1) + c));
        }
    }
    // Opacity is stored as its logit, `ln(p / (1 - p))` (an earlier version
    // wrote `ln((1 - p) / p)`, complementing it on read). Clamped strictly
    // inside (0, 1) because the logit is undefined at the endpoints.
    let p = g.opacity.clamp(1e-4, 1.0 - 1e-4);
    row.push((p / (1.0 - p)).ln());
    row.extend_from_slice(&[g.scale.x.ln(), g.scale.y.ln(), g.scale.z.ln()]);
    // (w, x, y, z) in the file.
    let q = unit_quaternion(&g.rotation);
    row.extend_from_slice(&[q.w, q.x, q.y, q.z]);
    row
}

fn ply_header(count: usize, rest: usize) -> String {
    let mut h = format!("ply\nformat ascii 1.0\nelement vertex {count}\n");
    let mut prop = |n: &str| h.push_str(&format!("property float {n}\n"));
    for n in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ] {
        prop(n);
    }
    for k in 0..3 * rest {
        prop(&format!("f_rest_{k}"));
    }
    for n in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        prop(n);
    }
    h.push_str("end_header\n");
    h
}

/// Write `cloud` as an ASCII PLY in the INRIA 3DGS layout.
pub fn write_ply_gaussian_cloud<P: AsRef<Path>>(
    cloud: &GaussianCloud,
    path: P,
) -> Result<(), String> {
    std::fs::write(path, gaussian_cloud_to_ply_string(cloud))
        .map_err(|e| format!("Failed to write file: {}", e))
}

/// The ASCII PLY text that [`write_ply_gaussian_cloud`] writes.
pub fn gaussian_cloud_to_ply_string(cloud: &GaussianCloud) -> String {
    let rest = rest_per_channel(cloud);
    let mut output = ply_header(cloud.num_gaussians(), rest);
    for g in &cloud.gaussians {
        let row: Vec<String> = gaussian_record(g, rest)
            .iter()
            .map(|v| v.to_string())
            .collect();
        output.push_str(&row.join(" "));
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_cloud() -> GaussianCloud {
        let mut cloud = GaussianCloud::new();

        let g1 = Gaussian::new(
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.8, 0.5, 0.3),
        )
        .with_opacity(0.7);

        let g2 = Gaussian::new(
            Point3::new(4.0, 5.0, 6.0),
            Vector3::new(0.4, 0.5, 0.6),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.2, 0.6, 0.9),
        )
        .with_opacity(0.3);

        cloud.push(g1);
        cloud.push(g2);
        cloud
    }

    #[test]
    fn test_gaussian_cloud_to_ply_string() {
        let cloud = create_test_cloud();
        let ply_string = gaussian_cloud_to_ply_string(&cloud);

        // Check header
        assert!(ply_string.contains("ply"));
        assert!(ply_string.contains("format ascii 1.0"));
        assert!(ply_string.contains("element vertex 2"));
        assert!(ply_string.contains("property float x"));
        assert!(ply_string.contains("end_header"));

        // Check that we have data lines
        let lines: Vec<&str> = ply_string.lines().collect();
        assert!(lines.len() > 10); // Header + data
    }

    #[test]
    fn test_write_and_read_ply_roundtrip() {
        let cloud = create_test_cloud();

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Write to file
        write_ply_gaussian_cloud(&cloud, &path).unwrap();

        // Read back
        let cloud_read = read_ply_gaussian_cloud(&path).unwrap();

        // Verify
        assert_eq!(cloud_read.num_gaussians(), cloud.num_gaussians());

        // Check first Gaussian (with some tolerance due to log conversions)
        let g1_orig = &cloud.gaussians[0];
        let g1_read = &cloud_read.gaussians[0];

        assert!((g1_read.position.x - g1_orig.position.x).abs() < 1e-5);
        assert!((g1_read.position.y - g1_orig.position.y).abs() < 1e-5);
        assert!((g1_read.position.z - g1_orig.position.z).abs() < 1e-5);
    }

    #[test]
    fn test_read_ply_invalid_file() {
        let result = read_ply_gaussian_cloud("/nonexistent/path.ply");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_ply_not_ply_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "This is not a PLY file").unwrap();

        let result = read_ply_gaussian_cloud(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_ply_with_higher_order_sh() {
        // A real 3DGS PLY carries f_rest_* (higher-order SH) after the 3 DC
        // terms; reading one must not panic and must keep all of them.
        let mut names: Vec<String> = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        names.extend((0..9).map(|i| format!("f_rest_{i}")));
        names.extend(
            [
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "ply").unwrap();
        writeln!(temp_file, "format ascii 1.0").unwrap();
        writeln!(temp_file, "element vertex 1").unwrap();
        for n in &names {
            writeln!(temp_file, "property float {n}").unwrap();
        }
        writeln!(temp_file, "end_header").unwrap();
        let values: Vec<String> = (0..names.len())
            .map(|i| format!("{}", i as f32 * 0.1))
            .collect();
        writeln!(temp_file, "{}", values.join(" ")).unwrap();
        temp_file.flush().unwrap();

        let cloud = read_ply_gaussian_cloud(temp_file.path()).expect("should parse");
        assert_eq!(cloud.num_gaussians(), 1);
        // 3 DC + 9 higher-order coefficients.
        assert_eq!(cloud.gaussians[0].spherical_harmonics.coeffs.len(), 12);
        assert_eq!(cloud.gaussians[0].spherical_harmonics.degree, 1);
    }

    #[test]
    fn test_ply_without_gaussian_properties_is_an_error() {
        // A plain point-cloud PLY is not a Gaussian cloud; it used to parse as
        // garbage (or as nothing) instead of failing.
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(
            temp_file,
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n\
             property float y\nproperty float z\nend_header\n0 0 0\n"
        )
        .unwrap();
        temp_file.flush().unwrap();
        let Err(err) = read_ply_gaussian_cloud(temp_file.path()) else {
            panic!("a PLY without 3DGS properties must not parse");
        };
        assert!(err.contains("f_dc_0"), "{err}");
    }
}
