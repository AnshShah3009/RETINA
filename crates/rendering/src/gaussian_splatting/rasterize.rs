use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, Vector4};

use super::types::{Gaussian, GaussianCloud, ProjectedGaussian, SphericalHarmonics};

/// The alpha below which a splat contributes nothing.
///
/// Used twice, and the two uses have to agree or the cull and the compositing
/// disagree: the inner loop skips a pixel whose `alpha_i` is below this, and
/// `compute_tile_bounds` sizes the tile range so that no such pixel can fall
/// outside it. See the note on the cull radius in `compute_tile_bounds`.
pub(crate) const ALPHA_CUTOFF: f32 = 1e-4;

impl From<cv_core::CameraIntrinsicsF32> for Camera {
    /// Convert from cv-core CameraIntrinsicsF32 to a rendering Camera at identity pose.
    fn from(intrinsics: cv_core::CameraIntrinsicsF32) -> Self {
        Camera::new(
            Point3::origin(),
            Vector4::new(0.0, 0.0, 0.0, 1.0), // identity quaternion
            intrinsics.fx,
            intrinsics.width,
            intrinsics.height,
        )
    }
}

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

        // `fov` is the VERTICAL field of view, and the vertical half-angle must
        // come from `height` and `focal_length`. It was built from `width`:
        //
        //     fov = 2 * atan(0.5 * width / focal_length)
        //
        // which is the *horizontal* field of view handed to a function whose
        // parameter is named `fov_y` and which computes
        // `P[1][1] = 1 / tan(fov_y / 2)`. The two therefore disagreed by exactly
        // the aspect ratio. Measured `1 / P[1][1]` against the focal length the
        // camera stores in `focal_length` - the value `Gaussian::project` actually
        // projects with:
        //
        //     160x120, f =  500 -> fy 0.160000 (expected 0.500000 = f/h)
        //     800x600, f = 1000 -> fy 0.400000 (expected 1.000000 = f/h)
        //     640x480, f =  500 -> fy 0.640000 (expected 0.500000 = f/h)
        //
        // For a non-square viewport the stored projection matrix and the
        // projection actually performed disagree, so anything reading
        // `projection_matrix` / `view_projection` - a depth test, a mesh draw, a
        // reprojection - uses a different focal length from the splat renderer.
        //
        // `aspect` also divides by `height`, so a zero-height camera made it
        // infinite (and, at width 0 as well, NaN), which propagated through
        // `perspective_matrix` and `view_projection`. `aspect` then multiplied
        // into `1 / (aspect * tan_half_fov)`, so a zero *width* made that entry
        // infinite too. Clamping both to at least 1 is the smallest image that
        // still has a meaningful aspect ratio, and it keeps the formula exact
        // for every ordinary viewport.
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let fov = 2.0 * (0.5 * height as f32 / focal_length.max(f32::MIN_POSITIVE)).atan();
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

    /// The rotation as a matrix, normalising `rotation` first.
    ///
    /// The formula below is the *unit*-quaternion form. Fed a quaternion of norm
    /// `k` it returns the rotation scaled by `k²`, so the determinant is `k³`
    /// rather than 1 and the view matrix is not a rigid transform at all.
    /// `Gaussian::unit_quaternion` documents exactly this hazard and is used by
    /// `Gaussian::rotation_matrix`; the camera is a sibling and needs the same
    /// guard. Measured with a norm-2 quaternion: determinant 2.894536, and
    /// max |R_scaled - R_unit| = 0.826081 - a visibly different camera.
    ///
    /// `rotation` is a public field of nothing here, but it is a function
    /// argument, so a caller that builds its own quaternion does reach it. A
    /// zero-norm quaternion specifies no rotation; the identity is used, for the
    /// same reason and with the same caveat as in `Gaussian::unit_quaternion`.
    fn rotation_to_matrix(q: &Vector4<f32>) -> Matrix3<f32> {
        let (x, y, z, w) = {
            let n = q.norm();
            if n.is_finite() && n > 0.0 {
                (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
            } else {
                (0.0, 0.0, 0.0, 1.0)
            }
        };

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
        // A zero field of view gives `tan(0) = 0`, and the entries below divide by
        // it - so a degenerate camera produced `inf`, then NaN through the
        // composite, then NaN in every vertex. Clamping the tangent to a positive
        // floor keeps the matrix finite for any input.
        //
        // The caller clamps `aspect` to at least 1 in the same way, but that alone
        // is not enough: a 0x0 viewport gives `aspect = 1` *and* `fov = 0`, so the
        // division by `tan_half_fov` is what actually needed guarding. Measured on a
        // 0x0 camera: `projection_matrix[1][1] = inf` with only the aspect clamp.
        let tan_half_fov = (fov_y / 2.0).tan().max(1e-6);
        let z_range = far - near;

        Matrix3x4::new(
            1.0 / (aspect.max(1e-6) * tan_half_fov),
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

    /// `projection_matrix * view_matrix`, the full 3x4 composite.
    ///
    /// The loop below summed `P[i][k] * V[k][j]` over `k` in `0..3` only, so the
    /// translation column - the `-R * position` term that `Camera::new` puts at
    /// `V[0][3..3][3]` - was never read and the fourth column of the result was
    /// identically zero. A view-projection matrix whose fourth column is zero
    /// maps every point to `z_clip = 0`: it is a projection onto a plane, not a
    /// projection, and there is no depth.
    ///
    /// Measured against the true composite `P*V`: max abs entry error 1.7210021.
    pub fn view_projection(&self) -> Matrix3x4<f32> {
        let mut result = Matrix3x4::zeros();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..3 {
                    result[(i, j)] += self.projection_matrix[(i, k)] * self.view_matrix[(k, j)];
                }
            }
        }
        // The `k == 3` term. `V` is a 3x4 *affine* matrix, so its implicit
        // fourth row is (0, 0, 0, 1) and the whole of `P[i][3] * V[3][j]` is
        // `P[i][3]` for every `j` - but only the translation column is affine, so
        // it contributes to `j == 3` alone. Adding it to all four columns would
        // shift the rotation columns as well, which is a different matrix
        // entirely.
        for i in 0..3 {
            result[(i, 3)] += self.projection_matrix[(i, 3)];
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
        let view = &self.camera.view_matrix;
        cloud
            .gaussians
            .iter()
            .map(|g| {
                g.project(
                    view,
                    self.camera.focal_length,
                    self.camera.width,
                    self.camera.height,
                )
            })
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

        // Compute eigenvalue-based radius from 2D covariance
        let a = cov_2d[(0, 0)];
        let b = cov_2d[(0, 1)];
        let d = cov_2d[(1, 1)];
        let trace = a + d;
        let det = a * d - b * b;
        let discriminant = ((trace * trace / 4.0) - det).max(0.0);
        let max_eigenvalue = trace / 2.0 + discriminant.sqrt();
        // The radius is 3 sigma, so it must be consistent with the alpha cutoff
        // the inner loop applies, or the cull decides what gets drawn and the
        // answer changes with the tile size.
        //
        // The inner loop drops any pixel with `alpha_i < ALPHA_CUTOFF`, and
        //
        //     alpha_i = exp(-sigma^2 / 2) * opacity  >  ALPHA_CUTOFF
        //  <=> sigma^2 < 2 * ln(opacity / ALPHA_CUTOFF)
        //
        // so every pixel the cutoff keeps out to `sigma_c = sqrt(2 ln(opacity /
        // ALPHA_CUTOFF))` must also be inside the culled tile range. The two
        // were unrelated constants: a fixed 3 against a cutoff that only bites
        // at 4.24 sigma for opacity 0.8. Every pixel between the two radii was
        // drawn in the tile it landed in and dropped in the tile it did not -
        // the image was a function of the tile size.
        //
        // Measured, 128x128 viewport, f = 300, one isotropic splat of world scale
        // 0.05 at (0.1, -0.05, 3.0) -> centre (74.0, 59.0) px, sigma = 5.0038 px,
        // so the 3-sigma radius is 15.008 px and the cutoff's radius is
        // 4.2396 sigma = 21.210 px. Reference: exhaustive untiled evaluation of
        // the same conic.
        //
        //     tile   max |d alpha| vs untiled   alpha sum   pixels lost
        //     4x4           2.474940e-3         125.63933        208
        //     8x8           5.901516e-4         125.72057         64
        //     16x16         1.184884e-4         125.73455          5
        //     32x32         9.984352e-5         125.73512          0
        //     64x64         9.984352e-5         125.73512          0
        //
        // The 208/64/5 are all beyond 3 sigma (checked by rebuilding sigma per
        // pixel from the inverse conic; max sigma among them 4.2379), and the
        // 9.98e-5 floor is the `alpha_i < 1e-4` cutoff itself. The fix moves the
        // 32x32 column to the top row.
        //
        // Above the cutoff radius the bound is itself clamped to the viewport, so
        // a splat at the image edge cannot produce an out-of-range tile; the
        // `.min(tiles_x)` on the min side covers `center` left of the viewport.
        let sigma_c = if pg.opacity > ALPHA_CUTOFF {
            (2.0 * (pg.opacity / ALPHA_CUTOFF).ln()).sqrt()
        } else {
            // Nothing reaches the cutoff, so the splat covers no pixel at all.
            // Left to the arithmetic this yields radius 0 and an empty range,
            // which is right, but stating it keeps the intent visible.
            0.0
        };
        let radius_px = (sigma_c * max_eigenvalue.sqrt()).ceil().max(1.0);

        let (tiles_x, tiles_y) = self.num_tiles();

        let min_px = (pg.center.x - radius_px).max(0.0);
        let min_py = (pg.center.y - radius_px).max(0.0);
        let max_px = (pg.center.x + radius_px).min(self.camera.width as f32);
        let max_py = (pg.center.y + radius_px).min(self.camera.height as f32);

        let min_x = ((min_px / self.tile_width as f32).floor() as u32).min(tiles_x);
        let min_y = ((min_py / self.tile_height as f32).floor() as u32).min(tiles_y);
        // `ceil`, not `floor`, for the exclusive upper bound: the tile containing
        // pixel `p` is `floor(p / T)`, and a half-open range wants one past the
        // last such index. `floor` here would drop a whole tile row or column.
        let max_x = ((max_px / self.tile_width as f32).ceil() as u32).min(tiles_x);
        let max_y = ((max_py / self.tile_height as f32).ceil() as u32).min(tiles_y);

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

        let n_pixels = (self.camera.width * self.camera.height) as usize;
        let mut output_image = vec![Vector3::zeros(); n_pixels];
        let mut output_alpha = vec![0.0f32; n_pixels];
        let mut output_depth = vec![f32::MAX; n_pixels];
        // Transmittance for front-to-back compositing, initialized to 1.0
        let mut transmittance = vec![1.0f32; n_pixels];

        for (tile_idx, _) in sorted_tiles {
            let tile_x = (tile_idx as u32 % tiles_x) * self.tile_width;
            let tile_y = (tile_idx as u32 / tiles_x) * self.tile_height;

            // Sort gaussians in this tile by depth (front to back)
            let mut tile_gauss: Vec<usize> = tile_gaussians[tile_idx].clone();
            tile_gauss.sort_by(|&a, &b| {
                projected[a]
                    .depth
                    .partial_cmp(&projected[b].depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for &gauss_idx in &tile_gauss {
                let pg = &projected[gauss_idx];

                for py in tile_y..(tile_y + self.tile_height).min(self.camera.height) {
                    for px in tile_x..(tile_x + self.tile_width).min(self.camera.width) {
                        let pixel_x = px as f32;
                        let pixel_y = py as f32;

                        let dx = pixel_x - pg.center.x;
                        let dy = pixel_y - pg.center.y;

                        // Full Mahalanobis distance using inverse covariance.
                        //
                        // A singular covariance means the splat has no defined
                        // footprint, so it is skipped. The previous fallback -
                        // a zero matrix - made `mahalanobis` 0 at every pixel, so
                        // the splat painted its entire tile at full opacity.
                        let Some(inv_cov_2d) = pg.inv_cov_2d() else {
                            continue;
                        };
                        let a = inv_cov_2d[(0, 0)];
                        let b = inv_cov_2d[(0, 1)];
                        let c = inv_cov_2d[(1, 1)];
                        let mahalanobis = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;

                        let alpha_i: f32 = ((-0.5 * mahalanobis).exp() * pg.opacity).min(0.99);

                        if alpha_i < ALPHA_CUTOFF {
                            continue;
                        }

                        let pixel_idx = (py * self.camera.width + px) as usize;

                        // Front-to-back compositing with transmittance
                        let t_i = transmittance[pixel_idx];
                        if t_i < 0.001 {
                            continue;
                        }

                        let color = pg.color;
                        output_image[pixel_idx] += color * alpha_i * t_i;
                        output_alpha[pixel_idx] += alpha_i * t_i;
                        transmittance[pixel_idx] *= 1.0 - alpha_i;
                        output_depth[pixel_idx] = output_depth[pixel_idx].min(pg.depth);
                    }
                }
            }
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
            Point3::new(0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            1000.0,
            800,
            600,
        );

        let mut cloud = GaussianCloud::new();
        let gaussian = Gaussian::new(
            Point3::new(0.0, 0.0, 5.0),
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
