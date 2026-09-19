use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use nalgebra::{Point3, Vector3, Vector4};

use super::types::{Gaussian, GaussianCloud, SphericalHarmonics};

pub fn read_ply_gaussian_cloud<P: AsRef<Path>>(path: P) -> Result<GaussianCloud, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header_line = lines
        .next()
        .ok_or("Empty file".to_string())?
        .map_err(|e| e.to_string())?;

    if !header_line.trim().starts_with("ply") {
        return Err("Not a PLY file".to_string());
    }

    let mut vertex_count = 0;
    let mut properties: Vec<String> = Vec::new();

    for line in lines.by_ref() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();

        if line.starts_with("element vertex") {
            vertex_count = line
                .split_whitespace()
                .nth(2)
                .ok_or("Invalid element line")?
                .parse()
                .map_err(|_| "Invalid vertex count")?;
        } else if line.starts_with("property") {
            properties.push(line.to_string());
        } else if line == "end_header" {
            break;
        }
    }

    let mut cloud = GaussianCloud::new();

    for _ in 0..vertex_count {
        if let Some(Ok(line)) = lines.next() {
            let values: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if values.len() < 14 {
                continue;
            }

            let position = Point3::new(values[0], values[1], values[2]);

            let nx = values.get(3).copied().unwrap_or(0.0);
            let ny = values.get(4).copied().unwrap_or(0.0);
            let nz = values.get(5).copied().unwrap_or(1.0);
            let scale0 = values.get(6).copied().unwrap_or(0.01).exp();
            let scale1 = values.get(7).copied().unwrap_or(0.01).exp();
            let scale2 = values.get(8).copied().unwrap_or(0.01).exp();

            let rot_dc_a = values.get(9).copied().unwrap_or(1.0);
            let rot_dc_b = values.get(10).copied().unwrap_or(0.0);
            let rot_dc_c = values.get(11).copied().unwrap_or(0.0);
            let rot_dc_d = values.get(12).copied().unwrap_or(0.0);

            let opacity = values.get(13).copied().unwrap_or(0.0);

            let dc = Vector3::new(
                values.get(14).copied().unwrap_or(0.8),
                values.get(15).copied().unwrap_or(0.8),
                values.get(16).copied().unwrap_or(0.8),
            );

            let mut sh = SphericalHarmonics::from_dc(dc);

            if values.len() > 17 {
                let copy_len = (values.len() - 17).min(sh.coeffs.len());
                sh.coeffs[3..3 + copy_len].copy_from_slice(&values[17..17 + copy_len]);
            }

            let rotation = Vector4::new(rot_dc_a, rot_dc_b, rot_dc_c, rot_dc_d);
            let rotation = if rotation.norm() > 0.001 {
                rotation.normalize()
            } else {
                Vector4::new(0.0, 0.0, 0.0, 1.0)
            };

            let gaussian = Gaussian {
                position,
                scale: Vector3::new(scale0, scale1, scale2),
                rotation,
                opacity: 1.0 / (1.0 + (-opacity).exp()),
                spherical_harmonics: sh,
                features: Vector3::new(nx, ny, nz),
            };

            cloud.push(gaussian);
        }
    }

    Ok(cloud)
}

pub fn write_ply_gaussian_cloud<P: AsRef<Path>>(
    cloud: &GaussianCloud,
    path: P,
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

    writeln!(file, "ply").map_err(|e| e.to_string())?;
    writeln!(file, "format ascii 1.0").map_err(|e| e.to_string())?;
    writeln!(file, "element vertex {}", cloud.num_gaussians()).map_err(|e| e.to_string())?;

    writeln!(file, "property float x").map_err(|e| e.to_string())?;
    writeln!(file, "property float y").map_err(|e| e.to_string())?;
    writeln!(file, "property float z").map_err(|e| e.to_string())?;
    writeln!(file, "property float nx").map_err(|e| e.to_string())?;
    writeln!(file, "property float ny").map_err(|e| e.to_string())?;
    writeln!(file, "property float nz").map_err(|e| e.to_string())?;
    writeln!(file, "property float scale_0").map_err(|e| e.to_string())?;
    writeln!(file, "property float scale_1").map_err(|e| e.to_string())?;
    writeln!(file, "property float scale_2").map_err(|e| e.to_string())?;
    writeln!(file, "property float rot_0").map_err(|e| e.to_string())?;
    writeln!(file, "property float rot_1").map_err(|e| e.to_string())?;
    writeln!(file, "property float rot_2").map_err(|e| e.to_string())?;
    writeln!(file, "property float rot_3").map_err(|e| e.to_string())?;
    writeln!(file, "property float opacity").map_err(|e| e.to_string())?;
    writeln!(file, "property float f_dc_0").map_err(|e| e.to_string())?;
    writeln!(file, "property float f_dc_1").map_err(|e| e.to_string())?;
    writeln!(file, "property float f_dc_2").map_err(|e| e.to_string())?;

    writeln!(file, "end_header").map_err(|e| e.to_string())?;

    for gaussian in &cloud.gaussians {
        let log_opacity =
            ((1.0 - gaussian.opacity.max(0.0001)) / gaussian.opacity.max(0.0001)).ln();
        let log_scale0 = gaussian.scale.x.ln();
        let log_scale1 = gaussian.scale.y.ln();
        let log_scale2 = gaussian.scale.z.ln();

        writeln!(
            file,
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            gaussian.position.x,
            gaussian.position.y,
            gaussian.position.z,
            gaussian.features.x,
            gaussian.features.y,
            gaussian.features.z,
            log_scale0,
            log_scale1,
            log_scale2,
            gaussian.rotation.x,
            gaussian.rotation.y,
            gaussian.rotation.z,
            gaussian.rotation.w,
            log_opacity,
            gaussian.spherical_harmonics.coeffs.first().unwrap_or(&0.8),
            gaussian.spherical_harmonics.coeffs.get(1).unwrap_or(&0.8),
            gaussian.spherical_harmonics.coeffs.get(2).unwrap_or(&0.8),
        )
        .map_err(|e| e.to_string())?;
    }

    Ok(())
}

pub fn gaussian_cloud_to_ply_string(cloud: &GaussianCloud) -> String {
    let mut output = String::new();

    output.push_str("ply\n");
    output.push_str("format ascii 1.0\n");
    output.push_str(&format!("element vertex {}\n", cloud.num_gaussians()));

    output.push_str("property float x\n");
    output.push_str("property float y\n");
    output.push_str("property float z\n");
    output.push_str("property float nx\n");
    output.push_str("property float ny\n");
    output.push_str("property float nz\n");
    output.push_str("property float scale_0\n");
    output.push_str("property float scale_1\n");
    output.push_str("property float scale_2\n");
    output.push_str("property float rot_0\n");
    output.push_str("property float rot_1\n");
    output.push_str("property float rot_2\n");
    output.push_str("property float rot_3\n");
    output.push_str("property float opacity\n");
    output.push_str("property float f_dc_0\n");
    output.push_str("property float f_dc_1\n");
    output.push_str("property float f_dc_2\n");

    output.push_str("end_header\n");

    for gaussian in &cloud.gaussians {
        let log_opacity =
            ((1.0 - gaussian.opacity.max(0.0001)) / gaussian.opacity.max(0.0001)).ln();
        let log_scale0 = gaussian.scale.x.ln();
        let log_scale1 = gaussian.scale.y.ln();
        let log_scale2 = gaussian.scale.z.ln();

        output.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            gaussian.position.x,
            gaussian.position.y,
            gaussian.position.z,
            gaussian.features.x,
            gaussian.features.y,
            gaussian.features.z,
            log_scale0,
            log_scale1,
            log_scale2,
            gaussian.rotation.x,
            gaussian.rotation.y,
            gaussian.rotation.z,
            gaussian.rotation.w,
            log_opacity,
            gaussian.spherical_harmonics.coeffs.first().unwrap_or(&0.8),
            gaussian.spherical_harmonics.coeffs.get(1).unwrap_or(&0.8),
            gaussian.spherical_harmonics.coeffs.get(2).unwrap_or(&0.8),
        ));
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
}
