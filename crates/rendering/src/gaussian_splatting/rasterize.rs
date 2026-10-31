use nalgebra::{Matrix2, Matrix2x3, Matrix3, Matrix3x4, Point3, Vector2, Vector3, Vector4};

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

    /// Project, cull and bin every Gaussian: the shared front half of the
    /// forward and backward passes.
    ///
    /// Returns the projected Gaussians, each one's 2-D conic `(a, b, c)` (the
    /// inverse screen covariance `[[a, b], [b, c]]`, `None` for a splat that is
    /// invalid or has a singular footprint and is therefore skipped), and per
    /// tile the indices of the splats overlapping it, sorted front to back.
    fn prepare(&self, cloud: &GaussianCloud) -> Prepared {
        let projected: Vec<ProjectedGaussian> = self.project_gaussians(cloud);
        let (tiles_x, tiles_y) = self.num_tiles();
        let num_tiles = (tiles_x * tiles_y) as usize;

        let conics: Vec<Option<[f32; 3]>> = projected
            .iter()
            .map(|pg| {
                if !pg.is_valid() {
                    return None;
                }
                // A singular covariance means the splat has no defined footprint,
                // so it is skipped. (A zero-matrix fallback once made the
                // Mahalanobis distance 0 everywhere and painted the whole tile.)
                pg.inv_cov_2d()
                    .map(|inv| [inv[(0, 0)], inv[(0, 1)], inv[(1, 1)]])
            })
            .collect();

        let mut tiles: Vec<Vec<usize>> = vec![Vec::new(); num_tiles];
        for (idx, pg) in projected.iter().enumerate() {
            if conics[idx].is_none() {
                continue;
            }
            let (min_x, min_y, max_x, max_y) = self.compute_tile_bounds(pg);
            for y in min_y..max_y {
                for x in min_x..max_x {
                    tiles[(y * tiles_x + x) as usize].push(idx);
                }
            }
        }
        for list in &mut tiles {
            list.sort_by(|&a, &b| {
                projected[a]
                    .depth
                    .partial_cmp(&projected[b].depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Prepared {
            projected,
            conics,
            tiles,
            tiles_x,
        }
    }

    /// Front-to-back compositing of one pixel. Calls `visit` for every splat
    /// that contributes, in order, with its [`Contribution`], and returns the
    /// final transmittance `T_N`.
    fn composite_pixel<F>(&self, prep: &Prepared, px: u32, py: u32, mut visit: F) -> f32
    where
        F: FnMut(&Contribution),
    {
        let tile_x = px / self.tile_width;
        let tile_y = py / self.tile_height;
        let list = &prep.tiles[(tile_y * prep.tiles_x + tile_x) as usize];
        let mut t = 1.0f32;
        for &idx in list {
            // Once the pixel is (almost) opaque nothing behind it is drawn.
            if t < 0.001 {
                break;
            }
            let pg = &prep.projected[idx];
            let Some([a, b, c]) = prep.conics[idx] else {
                continue;
            };
            let dx = px as f32 - pg.center.x;
            let dy = py as f32 - pg.center.y;
            let mahalanobis = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;
            let gauss = (-0.5 * mahalanobis).exp();
            let raw = gauss * pg.opacity;
            let alpha = raw.min(0.99);
            if alpha < ALPHA_CUTOFF {
                continue;
            }
            visit(&Contribution {
                idx,
                dx,
                dy,
                gauss,
                alpha,
                clamped: raw > 0.99,
                transmittance: t,
            });
            t *= 1.0 - alpha;
        }
        t
    }

    pub fn rasterize(&self, cloud: &GaussianCloud) -> RasterizationResult {
        self.rasterize_over(cloud, Vector3::zeros())
    }

    /// Render `cloud` composited over a constant `background` colour:
    ///
    /// ```text
    /// C = sum_i c_i alpha_i T_i + T_N * background,   T_i = prod_{j<i} (1 - alpha_j)
    /// ```
    pub fn rasterize_over(
        &self,
        cloud: &GaussianCloud,
        background: Vector3<f32>,
    ) -> RasterizationResult {
        let prep = self.prepare(cloud);

        let n_pixels = (self.camera.width * self.camera.height) as usize;
        let mut output_image = vec![Vector3::zeros(); n_pixels];
        let mut output_alpha = vec![0.0f32; n_pixels];
        let mut output_depth = vec![f32::MAX; n_pixels];

        for py in 0..self.camera.height {
            for px in 0..self.camera.width {
                let pixel_idx = (py * self.camera.width + px) as usize;
                let mut color = Vector3::zeros();
                let mut alpha = 0.0f32;
                let mut depth = f32::MAX;
                let t_final = self.composite_pixel(&prep, px, py, |c| {
                    let pg = &prep.projected[c.idx];
                    let w = c.alpha * c.transmittance;
                    color += pg.color * w;
                    alpha += w;
                    depth = depth.min(pg.depth);
                });
                output_image[pixel_idx] = color + background * t_final;
                output_alpha[pixel_idx] = alpha;
                output_depth[pixel_idx] = depth;
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

/// Projected, culled and binned splats. See [`GaussianRasterizer::prepare`].
struct Prepared {
    projected: Vec<ProjectedGaussian>,
    conics: Vec<Option<[f32; 3]>>,
    tiles: Vec<Vec<usize>>,
    tiles_x: u32,
}

/// One splat's contribution to one pixel during compositing.
struct Contribution {
    idx: usize,
    /// Pixel minus projected centre.
    dx: f32,
    dy: f32,
    /// `exp(-0.5 * mahalanobis)`.
    gauss: f32,
    /// `min(gauss * opacity, 0.99)`.
    alpha: f32,
    /// Whether the 0.99 cap was active (the alpha then has zero gradient).
    clamped: bool,
    /// Transmittance in front of this splat, `T_i`.
    transmittance: f32,
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

/// Rasterizer with an analytic backward pass, for fitting a [`GaussianCloud`]
/// to images.
///
/// The forward pass is [`GaussianRasterizer::rasterize_over`] with
/// [`background`](Self::background), and the loss is [`compute_loss`]
/// (masked mean squared error). [`backward`](Self::backward) returns the exact
/// gradient of that loss with respect to every parameter the forward pass
/// reads, following the reference CUDA rasterizer of Kerbl et al. 2023
/// (`backward.cu`): back-to-front through the alpha compositing, through the
/// 2-D Gaussian (conic and centre), through the EWA projection
/// `Sigma_2D = J W Sigma W^T J^T` including the dependence of the Jacobian `J`
/// on the mean, and through `Sigma = R S S^T R^T` to scale and quaternion.
///
/// [`compute_loss`]: Self::compute_loss
pub struct DifferentiableRasterizer {
    pub rasterizer: GaussianRasterizer,
    /// Colour composited behind the splats, weighted by the final
    /// transmittance of each pixel.
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
        self.rasterizer.rasterize_over(cloud, self.background)
    }

    /// Mean squared error over the pixels whose target is not masked out. A
    /// target pixel with all three channels negative is masked.
    #[allow(clippy::needless_range_loop)]
    pub fn compute_loss(&self, rendered: &RasterizationResult, target: &[Vector3<f32>]) -> f32 {
        let mut loss = 0.0;
        let mut count = 0;

        for i in 0..rendered.color.len() {
            if is_masked(&target[i]) {
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

    /// Gradient of `compute_loss(rendered, target)` with respect to each
    /// Gaussian's parameters, where `rendered` is `self.render(cloud)`.
    ///
    /// Gradients are with respect to the parameters as stored on [`Gaussian`]:
    /// `position`, `scale` (not log-scale), `rotation` (the raw stored
    /// quaternion, through its normalisation), `opacity` (not its logit), and
    /// the DC colour `spherical_harmonics.coeffs[0..3]`. Higher-order SH
    /// coefficients get no gradient because the forward pass does not read
    /// them: `Gaussian::project` colours a splat with `dc()` only, so the loss
    /// is constant in them and zero is their true derivative.
    ///
    /// The loss is piecewise smooth, as in the reference: the depth ordering,
    /// the `ALPHA_CUTOFF` skip, the 0.99 alpha cap and the early stop at
    /// transmittance 0.001 are not differentiated through. A splat whose alpha
    /// is capped receives no gradient through that alpha.
    ///
    /// Each [`GaussianGradient`] also carries `mean2d_grad` /
    /// `mean2d_grad_ndc`, the gradient with respect to the projected centre in
    /// pixels and in NDC units, and `visible`, whether the
    /// splat contributed to any pixel - the inputs 3DGS densification
    /// accumulates.
    pub fn backward(
        &self,
        cloud: &GaussianCloud,
        rendered: &RasterizationResult,
        target: &[Vector3<f32>],
    ) -> Vec<GaussianGradient> {
        let r = &self.rasterizer;
        let prep = r.prepare(cloud);
        let n = cloud.gaussians.len();

        // dL/dC per pixel for the masked MSE.
        let count = rendered
            .color
            .iter()
            .zip(target)
            .filter(|(_, t)| !is_masked(t))
            .count();
        let mut acc = vec![Accum::default(); n];
        if count == 0 {
            return finish(cloud, &prep, r, &acc);
        }
        let norm = 2.0 / count as f32;

        let mut stack: Vec<Contribution> = Vec::new();
        for py in 0..r.camera.height {
            for px in 0..r.camera.width {
                let pixel = (py * r.camera.width + px) as usize;
                let Some(t) = target.get(pixel) else {
                    continue;
                };
                if is_masked(t) {
                    continue;
                }
                let dl_dc = (rendered.color[pixel] - t) * norm;

                stack.clear();
                let t_final = r.composite_pixel(&prep, px, py, |c| {
                    stack.push(Contribution { ..*c });
                });

                // Back to front. `behind` is everything composited after splat
                // i, already attenuated by the splats in front of it:
                //     behind_i = sum_{j>i} c_j a_j T_j + T_N * bg
                // so dC/da_i = c_i T_i - behind_i / (1 - a_i).
                let mut behind = self.background * t_final;
                for c in stack.iter().rev() {
                    let pg = &prep.projected[c.idx];
                    let a = &mut acc[c.idx];
                    let w = c.alpha * c.transmittance;
                    a.visible = true;
                    a.color += dl_dc * w;

                    let dc_dalpha = pg.color * c.transmittance - behind / (1.0 - c.alpha);
                    behind += pg.color * w;
                    if c.clamped {
                        continue;
                    }
                    let dl_dalpha = dl_dc.dot(&dc_dalpha);
                    // alpha = G * o,  G = exp(-m / 2)
                    a.opacity += dl_dalpha * c.gauss;
                    let dl_dm = -0.5 * dl_dalpha * pg.opacity * c.gauss;
                    // m = d^T Q d with d = pixel - centre.
                    let [qa, qb, qc] = prep.conics[c.idx].unwrap_or([0.0; 3]);
                    a.center.x -= dl_dm * 2.0 * (qa * c.dx + qb * c.dy);
                    a.center.y -= dl_dm * 2.0 * (qb * c.dx + qc * c.dy);
                    a.conic[0] += dl_dm * c.dx * c.dx;
                    a.conic[1] += dl_dm * c.dx * c.dy;
                    a.conic[2] += dl_dm * c.dy * c.dy;
                }
            }
        }

        finish(cloud, &prep, r, &acc)
    }
}

fn is_masked(t: &Vector3<f32>) -> bool {
    t.x < 0.0 && t.y < 0.0 && t.z < 0.0
}

/// Per-Gaussian screen-space gradient sums.
#[derive(Clone, Default)]
struct Accum {
    visible: bool,
    color: Vector3<f32>,
    opacity: f32,
    /// dL/d(centre) in pixels.
    center: Vector2<f32>,
    /// dL/dQ for the symmetric conic `Q = [[a, b], [b, c]]`, as the full-matrix
    /// gradient `(g00, g01 = g10, g11)`.
    conic: [f32; 3],
}

/// Chain the screen-space sums back to the 3-D parameters.
fn finish(
    cloud: &GaussianCloud,
    prep: &Prepared,
    r: &GaussianRasterizer,
    acc: &[Accum],
) -> Vec<GaussianGradient> {
    cloud
        .gaussians
        .iter()
        .enumerate()
        .map(|(idx, g)| {
            let a = &acc[idx];
            let mut grad = GaussianGradient {
                idx,
                position_grad: Vector3::zeros(),
                scale_grad: Vector3::zeros(),
                rotation_grad: Vector4::zeros(),
                opacity_grad: a.opacity,
                color_grad: a.color,
                mean2d_grad: a.center,
                mean2d_grad_ndc: Vector2::new(
                    a.center.x * 0.5 * r.camera.width as f32,
                    a.center.y * 0.5 * r.camera.height as f32,
                ),
                visible: a.visible,
            };
            if !a.visible {
                return grad;
            }
            let Some([qa, qb, qc]) = prep.conics[idx] else {
                return grad;
            };
            let (dpos, dscale, drot) = project_backward(
                g,
                &r.camera,
                &prep.projected[idx],
                a.center,
                Matrix2::new(qa, qb, qb, qc),
                Matrix2::new(a.conic[0], a.conic[1], a.conic[1], a.conic[2]),
            );
            grad.position_grad = dpos;
            grad.scale_grad = dscale;
            grad.rotation_grad = drot;
            grad
        })
        .collect()
}

/// Backward of `Gaussian::project` for the centre and the 2-D covariance.
///
/// `q` is the conic (inverse 2-D covariance) and `dl_dq` its full-matrix
/// gradient. Returns dL/d(position, scale, stored rotation).
fn project_backward(
    g: &Gaussian,
    camera: &Camera,
    pg: &ProjectedGaussian,
    dl_dcenter: Vector2<f32>,
    q: Matrix2<f32>,
    dl_dq: Matrix2<f32>,
) -> (Vector3<f32>, Vector3<f32>, Vector4<f32>) {
    let f = camera.focal_length;
    let vm = &camera.view_matrix;
    let rv = vm.fixed_view::<3, 3>(0, 0).into_owned();
    let v = rv * g.position.coords + vm.fixed_view::<3, 1>(0, 3).into_owned();
    let (x, y, z) = (v.x, v.y, v.z);
    debug_assert!((z - pg.depth).abs() <= 1e-4 * z.abs().max(1.0));

    // Q = Sigma2^-1  =>  dL/dSigma2 = -Q dL/dQ Q.
    let dl_dsigma2 = -(q * dl_dq * q);

    // Sigma2 = T Sigma T^T with T = J2 Rv, J2 the top two rows of the
    // perspective Jacobian.
    let z2 = z * z;
    let j2 = Matrix2x3::new(f / z, 0.0, -f * x / z2, 0.0, f / z, -f * y / z2);
    let t = j2 * rv;
    let sigma = g.covariance();
    let dl_dsigma = t.transpose() * dl_dsigma2 * t;
    let dl_dt = 2.0 * dl_dsigma2 * t * sigma;
    let dl_dj = dl_dt * rv.transpose();

    // Centre: p = f (x, y) / z + (W, H) / 2.
    let z3 = z2 * z;
    let mut dl_dv = Vector3::new(
        dl_dcenter.x * f / z,
        dl_dcenter.y * f / z,
        -(dl_dcenter.x * f * x + dl_dcenter.y * f * y) / z2,
    );
    // Jacobian entries' dependence on the camera-space mean.
    dl_dv.x += dl_dj[(0, 2)] * (-f / z2);
    dl_dv.y += dl_dj[(1, 2)] * (-f / z2);
    dl_dv.z += (dl_dj[(0, 0)] + dl_dj[(1, 1)]) * (-f / z2)
        + dl_dj[(0, 2)] * (2.0 * f * x / z3)
        + dl_dj[(1, 2)] * (2.0 * f * y / z3);
    let dpos = rv.transpose() * dl_dv;

    // Sigma = M M^T, M = R S.
    let rot = g.rotation_matrix();
    let s = g.scale;
    let m = rot * Matrix3::from_diagonal(&s);
    let dl_dm = 2.0 * dl_dsigma * m;
    let mut dscale = Vector3::zeros();
    let mut dl_dr = Matrix3::zeros();
    for i in 0..3 {
        for k in 0..3 {
            dscale[k] += dl_dm[(i, k)] * rot[(i, k)];
            dl_dr[(i, k)] = dl_dm[(i, k)] * s[k];
        }
    }

    // R(q) for the unit quaternion (x, y, z, w); see `Gaussian::rotation_matrix`.
    let norm = g.rotation.norm();
    let drot = if norm.is_finite() && norm > 0.0 {
        let qn = g.rotation / norm;
        let (qx, qy, qz, qw) = (qn[0], qn[1], qn[2], qn[3]);
        let dr_dx = Matrix3::new(
            0.0,
            2.0 * qy,
            2.0 * qz,
            2.0 * qy,
            -4.0 * qx,
            -2.0 * qw,
            2.0 * qz,
            2.0 * qw,
            -4.0 * qx,
        );
        let dr_dy = Matrix3::new(
            -4.0 * qy,
            2.0 * qx,
            2.0 * qw,
            2.0 * qx,
            0.0,
            2.0 * qz,
            -2.0 * qw,
            2.0 * qz,
            -4.0 * qy,
        );
        let dr_dz = Matrix3::new(
            -4.0 * qz,
            -2.0 * qw,
            2.0 * qx,
            2.0 * qw,
            -4.0 * qz,
            2.0 * qy,
            2.0 * qx,
            2.0 * qy,
            0.0,
        );
        let dr_dw = Matrix3::new(
            0.0,
            -2.0 * qz,
            2.0 * qy,
            2.0 * qz,
            0.0,
            -2.0 * qx,
            -2.0 * qy,
            2.0 * qx,
            0.0,
        );
        let dl_dqn = Vector4::new(
            dl_dr.component_mul(&dr_dx).sum(),
            dl_dr.component_mul(&dr_dy).sum(),
            dl_dr.component_mul(&dr_dz).sum(),
            dl_dr.component_mul(&dr_dw).sum(),
        );
        // Through qn = q / |q|.
        (dl_dqn - qn * qn.dot(&dl_dqn)) / norm
    } else {
        // A zero quaternion is replaced by the identity; it has no derivative.
        Vector4::zeros()
    };

    (dpos, dscale, drot)
}

#[derive(Clone, Debug)]
pub struct GaussianGradient {
    pub idx: usize,
    pub position_grad: Vector3<f32>,
    pub scale_grad: Vector3<f32>,
    pub rotation_grad: Vector4<f32>,
    pub opacity_grad: f32,
    pub color_grad: Vector3<f32>,
    /// dL/d(projected centre), in pixels. 3DGS densification accumulates its
    /// norm (see `optimize::GaussianOptimizer`).
    pub mean2d_grad: Vector2<f32>,
    /// The same gradient with respect to the centre in normalised device
    /// coordinates, `(W/2, H/2) * mean2d_grad`: the units of the reference
    /// implementation's view-space gradient, which `grad_threshold` (0.0002)
    /// is calibrated in.
    pub mean2d_grad_ndc: Vector2<f32>,
    /// Whether the splat contributed to at least one pixel of this render.
    pub visible: bool,
}

impl GaussianGradient {
    /// Squared norm of all parameter gradients.
    pub fn norm_squared(&self) -> f32 {
        self.position_grad.norm_squared()
            + self.scale_grad.norm_squared()
            + self.rotation_grad.norm_squared()
            + self.opacity_grad * self.opacity_grad
            + self.color_grad.norm_squared()
    }
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
