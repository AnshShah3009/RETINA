use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, Vector4};

use super::types::{Gaussian, GaussianCloud, ProjectedGaussian, SphericalHarmonics};

#[derive(Clone, Debug)]
pub struct Camera {
    pub view_matrix: Matrix3x4<f32>,
    pub projection_matrix: Matrix3x4<f32>,
    pub focal_length: f32,
    pub width: u32,
    pub height: u32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        rotation: Vector4<f32>,
        focal_length: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let rot_mat = Self::rotation_to_matrix(&rotation);
        let px = position.x;
        let py = position.y;
        let pz = position.z;

        let t0 = -(rot_mat[(0, 0)] * px + rot_mat[(0, 1)] * py + rot_mat[(0, 2)] * pz);
        let t1 = -(rot_mat[(1, 0)] * px + rot_mat[(1, 1)] * py + rot_mat[(1, 2)] * pz);
        let t2 = -(rot_mat[(2, 0)] * px + rot_mat[(2, 1)] * py + rot_mat[(2, 2)] * pz);

        let view = Matrix3x4::new(
            rot_mat[(0, 0)],
            rot_mat[(0, 1)],
            rot_mat[(0, 2)],
            t0,
            rot_mat[(1, 0)],
            rot_mat[(1, 1)],
            rot_mat[(1, 2)],
            t1,
            rot_mat[(2, 0)],
            rot_mat[(2, 1)],
            rot_mat[(2, 2)],
            t2,
        );

        let aspect = width as f32 / height as f32;
        let fov = 2.0 * (focal_length / width as f32).atan();
        let proj = Self::perspective_matrix(fov, aspect, 0.01, 100.0);

        Self {
            view_matrix: view,
            projection_matrix: proj,
            focal_length,
            width,
            height,
            near: 0.01,
            far: 100.0,
        }
    }

    fn rotation_to_matrix(q: &Vector4<f32>) -> Matrix3<f32> {
        let x = q[0];
        let y = q[1];
        let z = q[2];
        let w = q[3];

        Matrix3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    }

    fn perspective_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> Matrix3x4<f32> {
        let tan_half_fov = (fov_y / 2.0).tan();
        let z_range = far - near;

        Matrix3x4::new(
            1.0 / (aspect * tan_half_fov),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / tan_half_fov,
            0.0,
            0.0,
            0.0,
            0.0,
            -(far + near) / z_range,
            -2.0 * far * near / z_range,
        )
    }

    pub fn view_projection(&self) -> Matrix3x4<f32> {
        let mut result = Matrix3x4::zeros();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..3 {
                    result[(i, j)] += self.projection_matrix[(i, k)] * self.view_matrix[(k, j)];
                }
            }
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub gaussian_ids: Vec<usize>,
}

pub struct GaussianRasterizer {
    pub camera: Camera,
    pub tile_width: u32,
    pub tile_height: u32,
}

impl GaussianRasterizer {
    pub fn new(camera: Camera, tile_width: u32, tile_height: u32) -> Self {
        Self {
            camera,
            tile_width,
            tile_height,
        }
    }

    pub fn num_tiles(&self) -> (u32, u32) {
        let tiles_x = self.camera.width.div_ceil(self.tile_width);
        let tiles_y = self.camera.height.div_ceil(self.tile_height);
        (tiles_x, tiles_y)
    }

    pub fn project_gaussians(&self, cloud: &GaussianCloud) -> Vec<ProjectedGaussian> {
        let vp = self.camera.view_projection();
        cloud
            .gaussians
            .iter()
            .map(|g| g.project(&vp, self.camera.focal_length))
            .collect()
    }

    pub fn compute_tile_bounds(&self, pg: &ProjectedGaussian) -> (u32, u32, u32, u32) {
        if !pg.is_valid() {
            return (0, 0, 0, 0);
        }

        let cov_2d = Matrix3::new(
            pg.covariance[(0, 0)],
            pg.covariance[(0, 1)],
            0.0,
            pg.covariance[(1, 0)],
            pg.covariance[(1, 1)],
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let det = cov_2d[(0, 0)] * cov_2d[(1, 1)] - cov_2d[(0, 1)] * cov_2d[(1, 0)];
        let det_abs = det.abs();
        let radius = (3.0 * det_abs.sqrt()).ceil() as u32;
        let radius = radius.max(1);

        let (tiles_x, tiles_y) = self.num_tiles();
        let tile_x = (pg.center.x / self.tile_width as f32).floor().max(0.0) as u32;
        let tile_y = (pg.center.y / self.tile_height as f32).floor().max(0.0) as u32;

        let min_x = tile_x.saturating_sub(radius).min(tiles_x);
        let min_y = tile_y.saturating_sub(radius).min(tiles_y);
        let max_x = (tile_x + radius).min(tiles_x);
        let max_y = (tile_y + radius).min(tiles_y);

        (min_x, min_y, max_x, max_y)
    }

    pub fn rasterize(&self, cloud: &GaussianCloud) -> RasterizationResult {
        let projected: Vec<ProjectedGaussian> = self.project_gaussians(cloud);
        let (tiles_x, tiles_y) = self.num_tiles();
        let num_tiles = (tiles_x * tiles_y) as usize;

        let mut tile_gaussians: Vec<Vec<usize>> = vec![Vec::new(); num_tiles];
        let mut depths: Vec<f32> = vec![f32::MAX; num_tiles];

        for (idx, pg) in projected.iter().enumerate() {
            if !pg.is_valid() {
                continue;
            }

            let (min_x, min_y, max_x, max_y) = self.compute_tile_bounds(pg);
            for y in min_y..max_y {
                for x in min_x..max_x {
                    let tile_idx = (y * tiles_x + x) as usize;
                    tile_gaussians[tile_idx].push(idx);
                    depths[tile_idx] = depths[tile_idx].min(pg.depth);
                }
            }
        }

        let mut sorted_tiles: Vec<(usize, f32)> = depths
            .iter()
            .enumerate()
            .filter(|(_, &d)| d < f32::MAX)
            .map(|(i, &d)| (i, d))
            .collect();
        sorted_tiles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut output_image =
            vec![Vector3::zeros(); (self.camera.width * self.camera.height) as usize];
        let mut output_alpha = vec![0.0f32; (self.camera.width * self.camera.height) as usize];
        let mut output_depth = vec![f32::MAX; (self.camera.width * self.camera.height) as usize];

        for (tile_idx, _) in sorted_tiles {
            let tile_x = (tile_idx as u32 % tiles_x) * self.tile_width;
            let tile_y = (tile_idx as u32 / tiles_x) * self.tile_height;

            let _tile_pixels: Vec<(usize, Vector3<f32>, f32)> = Vec::new();

            for &gauss_idx in &tile_gaussians[tile_idx] {
                let pg = &projected[gauss_idx];

                for py in tile_y..(tile_y + self.tile_height).min(self.camera.height) {
                    for px in tile_x..(tile_x + self.tile_width).min(self.camera.width) {
                        let pixel_x = px as f32;
                        let pixel_y = py as f32;

                        let dx = pixel_x - pg.center.x;
                        let dy = pixel_y - pg.center.y;
                        let dist_sq = dx * dx + dy * dy;

                        let inv_cov_2d = pg.inv_cov_2d();
                        let mahalanobis = inv_cov_2d[(0, 0)] * dist_sq;

                        let weight: f32 = (-0.5 * mahalanobis).exp() * pg.opacity;

                        if weight < 0.0001 {
                            continue;
                        }

                        let pixel_idx = (py * self.camera.width + px) as usize;
                        let color = pg.color;

                        output_image[pixel_idx] += color * weight;
                        output_alpha[pixel_idx] += weight;
                        output_depth[pixel_idx] = output_depth[pixel_idx].min(pg.depth);
                    }
                }
            }
        }

        for i in 0..output_image.len() {
            if output_alpha[i] > 0.0 {
                output_image[i] /= output_alpha[i].min(1.0);
            }
            output_alpha[i] = 1.0 - (-output_alpha[i] as f32).exp();
        }

        RasterizationResult {
            color: output_image,
            alpha: output_alpha,
            depth: output_depth,
            width: self.camera.width,
            height: self.camera.height,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RasterizationResult {
    pub color: Vec<Vector3<f32>>,
    pub alpha: Vec<f32>,
    pub depth: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl RasterizationResult {
    pub fn to_image(&self) -> Vec<u8> {
        self.color
            .iter()
            .flat_map(|c| {
                let r = (c.x.clamp(0.0, 1.0) * 255.0) as u8;
                let g = (c.y.clamp(0.0, 1.0) * 255.0) as u8;
                let b = (c.z.clamp(0.0, 1.0) * 255.0) as u8;
                [r, g, b]
            })
            .collect()
    }
}

pub struct DifferentiableRasterizer {
    pub rasterizer: GaussianRasterizer,
    pub background: Vector3<f32>,
}

impl DifferentiableRasterizer {
    pub fn new(camera: Camera, background: Vector3<f32>) -> Self {
        Self {
            rasterizer: GaussianRasterizer::new(camera, 16, 16),
            background,
        }
    }

    pub fn render(&self, cloud: &GaussianCloud) -> RasterizationResult {
        self.rasterizer.rasterize(cloud)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn compute_loss(&self, rendered: &RasterizationResult, target: &[Vector3<f32>]) -> f32 {
        let mut loss = 0.0;
        let mut count = 0;

        for i in 0..rendered.color.len() {
            if target[i].x < 0.0 && target[i].y < 0.0 && target[i].z < 0.0 {
                continue;
            }
            let diff = rendered.color[i] - target[i];
            loss += diff.dot(&diff);
            count += 1;
        }

        if count > 0 {
            loss / count as f32
        } else {
            0.0
        }
    }

    pub fn backward(
        &self,
        cloud: &GaussianCloud,
        _rendered: &RasterizationResult,
        _target: &[Vector3<f32>],
    ) -> Vec<GaussianGradient> {
        let _projected: Vec<ProjectedGaussian> = self.rasterizer.project_gaussians(cloud);

        cloud
            .gaussians
            .iter()
            .enumerate()
            .map(|(idx, _)| GaussianGradient {
                idx,
                position_grad: Vector3::zeros(),
                scale_grad: Vector3::zeros(),
                rotation_grad: Vector4::zeros(),
                opacity_grad: 0.0,
                color_grad: Vector3::zeros(),
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct GaussianGradient {
    pub idx: usize,
    pub position_grad: Vector3<f32>,
    pub scale_grad: Vector3<f32>,
    pub rotation_grad: Vector4<f32>,
    pub opacity_grad: f32,
    pub color_grad: Vector3<f32>,
}

pub fn create_dummy_gaussian_cloud(num_gaussians: usize) -> GaussianCloud {
    let mut cloud = GaussianCloud::new();
    for i in 0..num_gaussians {
        let theta = (i as f32 / num_gaussians as f32) * std::f32::consts::PI * 2.0;
        let phi = (i as f32 / num_gaussians as f32) * std::f32::consts::PI;
        let radius = 5.0;

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        let gaussian = Gaussian {
            position: Point3::new(x, y, z),
            scale: Vector3::new(0.1, 0.1, 0.1),
            rotation: Vector4::new(0.0, 0.0, 0.0, 1.0),
            opacity: 0.5,
            spherical_harmonics: SphericalHarmonics::from_dc(Vector3::new(0.8, 0.5, 0.3)),
            features: Vector3::zeros(),
        };
        cloud.push(gaussian);
    }
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_new() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        assert_eq!(camera.width, 800);
        assert_eq!(camera.height, 600);
        assert_eq!(camera.focal_length, 1000.0);
    }

    #[test]
    fn test_gaussian_rasterizer_new() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        let rasterizer = GaussianRasterizer::new(camera, 16, 16);
        assert_eq!(rasterizer.tile_width, 16);
        assert_eq!(rasterizer.tile_height, 16);
    }

    #[test]
    fn test_gaussian_rasterizer_num_tiles() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        let rasterizer = GaussianRasterizer::new(camera, 16, 16);
        let (tiles_x, tiles_y) = rasterizer.num_tiles();
        assert_eq!(tiles_x, 50); // 800/16 = 50
        assert_eq!(tiles_y, 38); // 600/16 = 37.5, rounded up
    }

    #[test]
    fn test_project_gaussians() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        let mut cloud = GaussianCloud::new();
        let gaussian = Gaussian::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.8, 0.5, 0.3),
        );
        cloud.push(gaussian);

        let rasterizer = GaussianRasterizer::new(camera, 16, 16);
        let projected = rasterizer.project_gaussians(&cloud);

        assert_eq!(projected.len(), 1);
        assert!(projected[0].is_valid());
    }

    #[test]
    fn test_rasterization_result_to_image() {
        let result = RasterizationResult {
            color: vec![Vector3::new(1.0, 0.5, 0.0), Vector3::new(0.0, 1.0, 0.5)],
            alpha: vec![1.0, 1.0],
            depth: vec![1.0, 2.0],
            width: 2,
            height: 1,
        };

        let image = result.to_image();
        assert_eq!(image.len(), 6); // 2 pixels * 3 channels
        assert_eq!(image[0], 255); // Red channel of first pixel
        assert_eq!(image[1], 127); // Green channel of first pixel (0.5 * 255 = 127.5 -> 127)
        assert_eq!(image[2], 0); // Blue channel of first pixel
    }

    #[test]
    fn test_differentiable_rasterizer_new() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        let background = Vector3::new(0.0, 0.0, 0.0);
        let rasterizer = DifferentiableRasterizer::new(camera, background);

        assert_eq!(rasterizer.background, background);
    }

    #[test]
    fn test_create_dummy_gaussian_cloud() {
        let cloud = create_dummy_gaussian_cloud(10);
        assert_eq!(cloud.num_gaussians(), 10);

        // Check that all gaussians are valid
        for gaussian in &cloud.gaussians {
            assert!(gaussian.opacity > 0.0);
            assert!(gaussian.scale.x > 0.0);
        }
    }
}
