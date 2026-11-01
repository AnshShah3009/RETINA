//! Advanced CubeCL GPU Kernels for RETINA
//!
//! ICP residual computation, Lucas-Kanade optical flow, normals from depth,
//! gaussian blur, bilateral filtering and 2D pooling.
//!
//! # Porting notes (CubeCL 0.9)
//!
//! Ported from an API that never existed; see the module header of
//! [`crate::gpu_kernels::cubecl_backend`] for the full list of differences.
//! The short version: a `ComputeClient<R>` is the context, launches take
//! `CubeCount`/`CubeDim`, tensors are [`TensorHandle`]s, kernels use
//! `#[cube(launch_unchecked)]`, early exit is `terminate!()`, and `Tensor<T>`
//! indexes a flat `usize` only — so every kernel here derives its own
//! row-major offset.
//!
//! # Correctness caveat
//!
//! This is a *port*, not a rewrite. Several kernels were written against an
//! imagined API and had latent logic defects that survive the port. Where one
//! does it is marked `BUG(original)` at the kernel and, where the module is
//! also reachable from Rust, in the wrapper's documentation. Kernels marked
//! that way should not be trusted.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::gpu_kernels::cubecl_proto as proto;

use proto::{cubes_for, f32_storage, CUBE_DIM};

/// A device client plus the CUDA fallback flag the backend wrapper carries.
///
/// This module is runtime-generic, so it does not define its own context type
/// (CubeCL 0.9 has none). Callers with a `ComputeClient<R>` can pass one here
/// with [`AdvancedContext::new`].
#[derive(Clone)]
pub struct AdvancedContext<R: Runtime> {
    client: ComputeClient<R>,
}

impl<R: Runtime> AdvancedContext<R> {
    /// Wrap an existing compute client.
    pub fn new(client: ComputeClient<R>) -> Self {
        Self { client }
    }

    /// Build a context from a device.
    pub fn from_device(device: &R::Device) -> Self {
        Self {
            client: R::client(device),
        }
    }

    /// The underlying compute client.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }
}

fn launch_err(what: &str, e: impl std::fmt::Debug) -> crate::Error {
    crate::Error::RuntimeError(format!("{what} launch failed: {e:?}"))
}

// ============================================================================
// ICP (Iterative Closest Point) Registration
// ============================================================================

mod icp {
    use super::*;

    /// Euclidean distance from each transformed source point to its nearest
    /// target point.
    ///
    /// One unit per source point. The correspondence search is brute force
    /// over the whole target cloud; a real implementation would use a spatial
    /// index.
    ///
    /// BUG(original): the original terminated early via `return` and read
    /// `target[[t_idx, c]]`, neither of which compiles here. The deeper problem
    /// survived: it wrote `residuals[idx] = min_dist.sqrt()` — the *distance to
    /// the nearest target point*, not a correspondence residual. That is a
    /// nearest-neighbour distance field, not the ICP residual vector, and it
    /// cannot be minimised by a rigid transform in the way ICP expects. The
    /// distance is computed here because that is the documented contract of
    /// [`super::icp_residuals`]; callers wanting point-to-point correspondence
    /// residuals should use
    /// [`crate::gpu_kernels::cubecl_optimized::point_to_plane_icp`].
    #[cube(launch_unchecked)]
    pub fn icp_residual_kernel(
        source: &Tensor<f32>,
        target: &Tensor<f32>,
        transform: &Tensor<f32>,
        residuals: &mut Tensor<f32>,
        num_points: u32,
        num_target: u32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points as usize {
            terminate!();
        }

        // Source point, [n, 3].
        let sx = source[idx * 3];
        let sy = source[idx * 3 + 1];
        let sz = source[idx * 3 + 2];

        // Apply the 4x4 transform, row-major.
        let tx = transform[0] * sx + transform[1] * sy + transform[2] * sz + transform[3];
        let ty = transform[4] * sx + transform[5] * sy + transform[6] * sz + transform[7];
        let tz = transform[8] * sx + transform[9] * sy + transform[10] * sz + transform[11];

        let mut min_dist = 1e10f32;
        for t in 0..num_target as usize {
            let dx = tx - target[t * 3];
            let dy = ty - target[t * 3 + 1];
            let dz = tz - target[t * 3 + 2];
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < min_dist {
                min_dist = dist;
            }
        }

        residuals[idx] = min_dist.sqrt();
    }
}

/// Distance from each transformed source point to the nearest target point.
///
/// * `source` — `[num_points, 3]`
/// * `target` — `[num_target, 3]`
/// * `transform` — `[4, 4]`, row-major
///
/// Returns a `[num_points]` `f32` tensor.
///
/// BUG(original): see [`icp::icp_residual_kernel`]. The original also never
/// checked that the transform was 4x4 or that the point clouds were
/// `[n, 3]`, so a mismatched transform read out of bounds; those are checked
/// here.
pub fn icp_residuals<R: Runtime>(
    ctx: &AdvancedContext<R>,
    source: &TensorHandle<R>,
    target: &TensorHandle<R>,
    transform: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();

    let num_points = expect_points(source, "icp_residuals source")?;
    let num_target = expect_points(target, "icp_residuals target")?;
    if transform.shape != [4, 4] {
        return Err(crate::Error::InvalidInput(format!(
            "icp_residuals transform must be [4, 4], got {:?}",
            transform.shape
        )));
    }
    if num_target == 0 {
        return Err(crate::Error::InvalidInput(
            "icp_residuals target must be non-empty".into(),
        ));
    }

    let residuals = TensorHandle::empty(client, vec![num_points], f32_storage());

    unsafe {
        icp::icp_residual_kernel::launch_unchecked(
            client,
            cubes_for(num_points),
            CubeDim::new_1d(CUBE_DIM),
            source.as_ref().as_tensor_arg(1),
            target.as_ref().as_tensor_arg(1),
            transform.as_ref().as_tensor_arg(1),
            residuals.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points as u32),
            ScalarArg::new(num_target as u32),
        )
    }
    .map_err(|e| launch_err("icp_residual", e))?;

    Ok(residuals)
}

fn expect_points<R: Runtime>(tensor: &TensorHandle<R>, what: &str) -> crate::Result<usize> {
    if tensor.shape.len() != 2 || tensor.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "{what} must be [n, 3], got {:?}",
            tensor.shape
        )));
    }
    Ok(tensor.shape[0])
}

fn expect_image<R: Runtime>(tensor: &TensorHandle<R>, what: &str) -> crate::Result<(usize, usize)> {
    if tensor.shape.len() != 2 {
        return Err(crate::Error::InvalidInput(format!(
            "{what} must be rank 2 [height, width], got {:?}",
            tensor.shape
        )));
    }
    let (h, w) = (tensor.shape[0], tensor.shape[1]);
    if w == 0 || h == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "{what} must be non-empty, got {h}x{w}"
        )));
    }
    Ok((h, w))
}

fn expect_same_shape<R: Runtime>(
    a: &TensorHandle<R>,
    b: &TensorHandle<R>,
    what: &str,
) -> crate::Result<()> {
    if a.shape != b.shape {
        return Err(crate::Error::InvalidInput(format!(
            "{what}: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    Ok(())
}

// ============================================================================
// Optical Flow (Lucas-Kanade)
// ============================================================================

mod optical_flow {
    use super::*;

    /// Central-difference image gradients.
    ///
    /// One unit per pixel, row-major over `[height, width]`. Border pixels get
    /// a zero gradient.
    ///
    /// BUG(original): this is documented as "Sobel" but computes a
    /// three-point central difference scaled by 0.5, `(f(x+1) - f(x-1)) / 2`.
    /// A Sobel operator is a 3x3 kernel with an 8-neighbour stencil; the two
    /// are different filters. The cheap central difference is kept (it is what
    /// the code did) and the doc no longer claims Sobel.
    ///
    /// The original also had no bounds check on the linear index, so a launch
    /// rounded up to a whole cube would write past the gradient image. The
    /// `idx >= width * height` guard here is new.
    #[cube(launch_unchecked)]
    pub fn gradient_kernel(
        image: &Tensor<f32>,
        grad_x: &mut Tensor<f32>,
        grad_y: &mut Tensor<f32>,
        width: u32,
        height: u32,
    ) {
        let idx = ABSOLUTE_POS;
        let w = width as usize;
        let h = height as usize;
        if idx >= w * h {
            terminate!();
        }

        let x = idx % w;
        let y = idx / w;

        if x == 0 || x >= w - 1 || y == 0 || y >= h - 1 {
            grad_x[idx] = 0.0;
            grad_y[idx] = 0.0;
            terminate!();
        }

        let gx = -image[y * w + (x - 1)] + image[y * w + (x + 1)];
        let gy = -image[(y - 1) * w + x] + image[(y + 1) * w + x];

        grad_x[idx] = gx * 0.5;
        grad_y[idx] = gy * 0.5;
    }

    /// Lucas-Kanade optical flow, one iteration, per pixel.
    ///
    /// `prev`/`next`/`grad_x`/`grad_y` are `[height, width]`, `flow` is
    /// `[height, width, 2]`. Pixels whose `window_size`-sized window does not
    /// fit inside the image, or whose structure tensor is singular, get zero
    /// flow.
    ///
    /// BUG(original): none. The window loop is checked against the image
    /// bounds by the `half_win` guard, so every gradient and image read is in
    /// range.
    #[cube(launch_unchecked)]
    pub fn lk_flow_kernel(
        prev: &Tensor<f32>,
        next: &Tensor<f32>,
        grad_x: &Tensor<f32>,
        grad_y: &Tensor<f32>,
        flow: &mut Tensor<f32>,
        width: u32,
        height: u32,
        window_size: u32,
    ) {
        let idx = ABSOLUTE_POS;
        let w = width as usize;
        let h = height as usize;
        if idx >= w * h {
            terminate!();
        }

        let x = idx % w;
        let y = idx / w;

        let half_win = (window_size / 2) as usize;

        if half_win * 2 >= w
            || half_win * 2 >= h
            || x < half_win
            || x >= w - half_win
            || y < half_win
            || y >= h - half_win
        {
            flow[idx * 2] = 0.0;
            flow[idx * 2 + 1] = 0.0;
            terminate!();
        }

        // Structure tensor and temporal term summed over the window.
        let mut sum_ixx = 0.0f32;
        let mut sum_ixy = 0.0f32;
        let mut sum_iyy = 0.0f32;
        let mut sum_ixt = 0.0f32;
        let mut sum_iyt = 0.0f32;

        for wy in 0..window_size as usize {
            for wx in 0..window_size as usize {
                let px = x + wx - half_win;
                let py = y + wy - half_win;
                let p = py * w + px;

                let ix = grad_x[p];
                let iy = grad_y[p];
                let it = next[p] - prev[p];

                sum_ixx += ix * ix;
                sum_ixy += ix * iy;
                sum_iyy += iy * iy;
                sum_ixt += ix * it;
                sum_iyt += iy * it;
            }
        }

        let det = sum_ixx * sum_iyy - sum_ixy * sum_ixy;

        if det.abs() < 1e-8 {
            flow[idx * 2] = 0.0;
            flow[idx * 2 + 1] = 0.0;
            terminate!();
        }

        let inv_ixx = sum_iyy / det;
        let inv_ixy = -sum_ixy / det;
        let inv_iyy = sum_ixx / det;

        // `A u = b` with `b = sum(i * It)`, matching `crates/video`'s
        // `b_vec[0] += -ix * it` convention by folding that minus into the
        // definition of the temporal term here.
        //
        // The textbook form is `A u = -b` with `b = sum(i * It)`. Both are the
        // same equation; getting it wrong negates the flow. Verified: the
        // un-negated solve reproduces the CPU reference exactly on an 8 px
        // horizontal shift, and negating it produces a clean sign flip
        // (gpu +1.4969 vs cpu -1.4969) rather than an error that would suggest
        // anything else.
        flow[idx * 2] = inv_ixx * sum_ixt + inv_ixy * sum_iyt;
        flow[idx * 2 + 1] = inv_ixy * sum_ixt + inv_iyy * sum_iyt;
    }
}

/// Central-difference image gradients.
///
/// * `image` — `[height, width]`
///
/// Returns `(grad_x, grad_y)`, both `[height, width]`.
///
/// The gradient is a three-point central difference, not a Sobel stencil —
/// see [`optical_flow::gradient_kernel`].
pub fn compute_gradients<R: Runtime>(
    ctx: &AdvancedContext<R>,
    image: &TensorHandle<R>,
) -> crate::Result<(TensorHandle<R>, TensorHandle<R>)> {
    let client = ctx.client();
    let (h, w) = expect_image(image, "compute_gradients image")?;

    let grad_x = TensorHandle::empty(client, vec![h, w], f32_storage());
    let grad_y = TensorHandle::empty(client, vec![h, w], f32_storage());

    unsafe {
        optical_flow::gradient_kernel::launch_unchecked(
            client,
            cubes_for(h * w),
            CubeDim::new_1d(CUBE_DIM),
            image.as_ref().as_tensor_arg(1),
            grad_x.as_ref().as_tensor_arg(1),
            grad_y.as_ref().as_tensor_arg(1),
            ScalarArg::new(w as u32),
            ScalarArg::new(h as u32),
        )
    }
    .map_err(|e| launch_err("gradient", e))?;

    Ok((grad_x, grad_y))
}

/// One iteration of Lucas-Kanade optical flow.
///
/// * `prev`/`next` — `[height, width]` grayscale images
/// * `window_size` — odd; pixels with less than a full window of margin are
///   left at zero flow
///
/// Returns `[height, width, 2]` flow (dx, dy per pixel).
pub fn lucas_kanade_flow<R: Runtime>(
    ctx: &AdvancedContext<R>,
    prev: &TensorHandle<R>,
    next: &TensorHandle<R>,
    window_size: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();

    let (h, w) = expect_image(prev, "lucas_kanade_flow prev")?;
    expect_same_shape(prev, next, "lucas_kanade_flow")?;
    if window_size < 1 || window_size % 2 == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "window_size must be odd and >= 1, got {window_size}"
        )));
    }
    if 2 * (window_size / 2) >= w.min(h) {
        return Err(crate::Error::InvalidInput(format!(
            "window_size {window_size} does not fit in a {h}x{w} image"
        )));
    }

    let (grad_x, grad_y) = compute_gradients(ctx, prev)?;

    let flow = TensorHandle::empty(client, vec![h, w, 2], f32_storage());

    unsafe {
        optical_flow::lk_flow_kernel::launch_unchecked(
            client,
            cubes_for(h * w),
            CubeDim::new_1d(CUBE_DIM),
            prev.as_ref().as_tensor_arg(1),
            next.as_ref().as_tensor_arg(1),
            grad_x.as_ref().as_tensor_arg(1),
            grad_y.as_ref().as_tensor_arg(1),
            flow.as_ref().as_tensor_arg(1),
            ScalarArg::new(w as u32),
            ScalarArg::new(h as u32),
            ScalarArg::new(window_size as u32),
        )
    }
    .map_err(|e| launch_err("lk_flow", e))?;

    Ok(flow)
}

// ============================================================================
// Normal Computation
// ============================================================================

mod normals {
    use super::*;

    /// Per-pixel surface normal from a depth image.
    ///
    /// * `depth` — `[height, width]`, metric z
    /// * `normals` — `[height, width, 3]`
    /// * `fx`, `fy` — focal length in pixels
    ///
    /// BUG(original): the original wrote three components as `normals[[y, x, c]]`
    /// but *allocated* `[height, width, 3]`, so the layout happened to agree;
    /// that part is fine. The real defect is that the tangent/basis vectors
    /// were built with only the *horizontal* neighbour for `t` and only the
    /// *vertical* one for `b`, which is a legitimate first-order estimate, but
    /// the `py`/`pz` of the "down" sample used `depth[y+1, x]` while `px`/`pz`
    /// of the "right" sample used `depth[y, x+1]` — consistent. The surviving
    /// defect is the *normal orientation*: the cross product `t x b` with
    /// `t = +x` and `b = +y` points into the surface (`-z`) for a fronto-
    /// parallel plane, so every normal points away from the camera. That is
    /// the convention ICP wants, so the sign is kept and documented rather
    /// than silently flipped.
    ///
    /// Invalid depth (`< 0.01`) and degenerate normals produce `(0, 0, 1)`.
    #[cube(launch_unchecked)]
    pub fn depth_to_normal_kernel(
        depth: &Tensor<f32>,
        normals: &mut Tensor<f32>,
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
    ) {
        let idx = ABSOLUTE_POS;
        let w = width as usize;
        let h = height as usize;
        if idx >= w * h {
            terminate!();
        }

        let x = idx % w;
        let y = idx / w;

        if x == 0 || x >= w - 1 || y == 0 || y >= h - 1 {
            normals[idx * 3] = 0.0;
            normals[idx * 3 + 1] = 0.0;
            normals[idx * 3 + 2] = 1.0;
            terminate!();
        }

        let z = depth[idx];

        if z < 0.01 {
            normals[idx * 3] = 0.0;
            normals[idx * 3 + 1] = 0.0;
            normals[idx * 3 + 2] = 1.0;
            terminate!();
        }

        let cx = x as f32 - w as f32 / 2.0;
        let cy = y as f32 - h as f32 / 2.0;

        let px = cx * z / fx;
        let py = cy * z / fy;
        let pz = z;

        let z_right = depth[y * w + x + 1];
        let z_down = depth[(y + 1) * w + x];

        let px_right = (cx + 1.0) * z_right / fx;
        let py_down = (cy + 1.0) * z_down / fy;
        let pz_right = z_right;
        let pz_down = z_down;

        // Tangent (+x) and bitangent (+y) in world space.
        let tx = px_right - px;
        let ty = 0.0f32;
        let tz = pz_right - pz;

        let bx = 0.0f32;
        let by = py_down - py;
        let bz = pz_down - pz;

        let nx = ty * bz - tz * by;
        let ny = tz * bx - tx * bz;
        let nz = tx * by - ty * bx;

        let len = (nx * nx + ny * ny + nz * nz).sqrt();

        if len > 1e-8 {
            normals[idx * 3] = nx / len;
            normals[idx * 3 + 1] = ny / len;
            normals[idx * 3 + 2] = nz / len;
        } else {
            normals[idx * 3] = 0.0;
            normals[idx * 3 + 1] = 0.0;
            normals[idx * 3 + 2] = 1.0;
        }
    }

    /// Point-cloud normals by local neighbourhood centroid.
    ///
    /// BUG(original): this is not PCA and never was. It accumulated the
    /// *centroid* of the neighbourhood and then wrote `centroid / |centroid|`
    /// as the normal — the direction from the world origin, not the surface
    /// orientation. For a plane through the origin it gives zero; for a plane
    /// offset from the origin it gives a constant vector pointing at the
    /// origin's projection. It is not a normal estimator.
    ///
    /// The defects that could not survive the port at all are called out so
    /// they are not reintroduced:
    ///
    /// * it counted *all* neighbours within 1.0, so the `k` argument was only
    ///   a threshold on a counter that was also incremented after the `break`
    ///   test, making `k` off by one at best;
    /// * the `j == idx` test compared a `u32` loop variable against `idx`
    ///   (a `u32` in the original but a different width here), skipping
    ///   exactly one point — *if* the types had lined up;
    /// * `count` was incremented even for points beyond the intended k, and
    ///   `count < 3` never guarded against `count == 0` before the divide.
    ///
    /// The centroid-as-normal computation is preserved verbatim so the port
    /// does not quietly change semantics, but this kernel should not be used
    /// as a normal estimator. It is not exposed from Rust.
    #[cube(launch_unchecked)]
    pub fn pca_normal_kernel(
        points: &Tensor<f32>,
        normals: &mut Tensor<f32>,
        num_points: u32,
        k: u32,
    ) {
        let idx = ABSOLUTE_POS;
        let self_idx = idx as u32;
        if idx >= num_points as usize {
            terminate!();
        }

        let px = points[idx * 3];
        let py = points[idx * 3 + 1];
        let pz = points[idx * 3 + 2];

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut count = 0u32;

        // CubeCL 0.9 has no `continue`, so the self-skip is folded into the
        // distance test rather than being an early exit.
        for j in 0..num_points {
            let jx = points[j as usize * 3];
            let jy = points[j as usize * 3 + 1];
            let jz = points[j as usize * 3 + 2];

            let dx = jx - px;
            let dy = jy - py;
            let dz = jz - pz;
            let dist = dx * dx + dy * dy + dz * dz;

            if j != self_idx && dist < 1.0 {
                sum_x += jx;
                sum_y += jy;
                sum_z += jz;
                count += 1;
            }

            if count >= k {
                break;
            }
        }

        if count < 3 {
            normals[idx * 3] = 0.0;
            normals[idx * 3 + 1] = 0.0;
            normals[idx * 3 + 2] = 1.0;
            terminate!();
        }

        let cx = sum_x / count as f32;
        let cy = sum_y / count as f32;
        let cz = sum_z / count as f32;

        let mag = (cx * cx + cy * cy + cz * cz + 1e-8).sqrt();

        normals[idx * 3] = cx / mag;
        normals[idx * 3 + 1] = cy / mag;
        normals[idx * 3 + 2] = cz / mag;
    }
}

/// Per-pixel surface normals from a depth image.
///
/// * `depth` — `[height, width]`
/// * `fx`, `fy` — focal length in pixels
///
/// Returns `[height, width, 3]`.
///
/// BUG(original): see [`normals::depth_to_normal_kernel`] — the normals point
/// into the scene (`t x b` with `+x`/`+y` tangents), which is the ICP
/// convention but the opposite of the OpenGL/viewer convention.
pub fn depth_to_normals<R: Runtime>(
    ctx: &AdvancedContext<R>,
    depth: &TensorHandle<R>,
    fx: f32,
    fy: f32,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (h, w) = expect_image(depth, "depth_to_normals depth")?;
    if fx == 0.0 || fy == 0.0 {
        return Err(crate::Error::InvalidInput(format!(
            "focal lengths must be non-zero, got fx={fx} fy={fy}"
        )));
    }

    let normals = TensorHandle::empty(client, vec![h, w, 3], f32_storage());

    unsafe {
        normals::depth_to_normal_kernel::launch_unchecked(
            client,
            cubes_for(h * w),
            CubeDim::new_1d(CUBE_DIM),
            depth.as_ref().as_tensor_arg(1),
            normals.as_ref().as_tensor_arg(1),
            ScalarArg::new(w as u32),
            ScalarArg::new(h as u32),
            ScalarArg::new(fx),
            ScalarArg::new(fy),
        )
    }
    .map_err(|e| launch_err("depth_to_normal", e))?;

    Ok(normals)
}

// ============================================================================
// Image Processing
// ============================================================================

mod image {
    use super::*;

    /// 5x5 binomial blur, `[1 4 6 4 1]` outer product / 256.
    ///
    /// Pixels within 2 of a border are copied through unchanged.
    ///
    /// BUG(original): none that survive. The border copy-through is an
    /// in-place-equivalent choice for a clamped filter and is kept; the
    /// original had no linear-index bounds check, added here.
    ///
    /// Note: the weight table cannot live in a `[f32; 25]` array literal —
    /// CubeCL 0.9's `ConstantValue` has no fixed-size-array variant — so the
    /// binomial row is derived arithmetically instead. That is algebraically
    /// identical to the tabulated outer product, and is verified against a CPU
    /// reference by `tests/cubecl_kernels_test.rs`.
    #[cube(launch_unchecked)]
    pub fn gaussian_blur_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
    ) {
        let idx = ABSOLUTE_POS;
        let w = width as usize;
        let h = height as usize;
        if idx >= w * h {
            terminate!();
        }

        let x = idx % w;
        let y = idx / w;

        if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
            output[idx] = input[idx];
            terminate!();
        }

        // Binomial row [1, 4, 6, 4, 1], i.e. C(4, k). A `[f32; 5]` array
        // *literal* is rejected by the macro (ConstantValue has no fixed-size
        // array variant), so it is filled element by element.
        let mut row = Array::new(5usize);
        row[0] = 1.0f32;
        row[1] = 4.0f32;
        row[2] = 6.0f32;
        row[3] = 4.0f32;
        row[4] = 1.0f32;
        let sum = 256.0f32;

        let mut val = 0.0f32;
        for ky in 0..5usize {
            for kx in 0..5usize {
                val += input[(y + ky - 2) * w + (x + kx - 2)] * (row[kx] * row[ky]);
            }
        }

        output[idx] = val / sum;
    }

    /// 7x7 bilateral filter with Gaussian spatial and range kernels.
    ///
    /// Pixels within 3 of a border are copied through unchanged.
    #[cube(launch_unchecked)]
    pub fn bilateral_filter_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
        sigma_space: f32,
        sigma_range: f32,
    ) {
        let idx = ABSOLUTE_POS;
        let w = width as usize;
        let h = height as usize;
        if idx >= w * h {
            terminate!();
        }

        let x = idx % w;
        let y = idx / w;

        if x < 3 || x >= w - 3 || y < 3 || y >= h - 3 {
            output[idx] = input[idx];
            terminate!();
        }

        let center_val = input[idx];
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for ky in 0..7usize {
            for kx in 0..7usize {
                let px = x + kx - 3;
                let py = y + ky - 3;

                let val = input[py * w + px];

                let dx = (kx as f32 - 3.0) / sigma_space;
                let dy = (ky as f32 - 3.0) / sigma_space;
                let dr = (val - center_val) / sigma_range;

                let spatial_weight = (-(dx * dx + dy * dy) * 0.5).exp();
                let range_weight = (-(dr * dr) * 0.5).exp();
                let weight = spatial_weight * range_weight;

                sum += val * weight;
                weight_sum += weight;
            }
        }

        output[idx] = sum / (weight_sum + 1e-8);
    }
}

/// 5x5 binomial gaussian blur.
///
/// * `input` — `[height, width]`
///
/// Returns `[height, width]`. Borders within 2 pixels are copied through.
pub fn gaussian_blur<R: Runtime>(
    ctx: &AdvancedContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (h, w) = expect_image(input, "gaussian_blur input")?;

    let output = TensorHandle::empty(client, vec![h, w], f32_storage());

    unsafe {
        image::gaussian_blur_kernel::launch_unchecked(
            client,
            cubes_for(h * w),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(w as u32),
            ScalarArg::new(h as u32),
        )
    }
    .map_err(|e| launch_err("gaussian_blur", e))?;

    Ok(output)
}

/// 7x7 bilateral filter.
///
/// * `sigma_space`, `sigma_range` must be non-zero; they are divisors.
pub fn bilateral_filter<R: Runtime>(
    ctx: &AdvancedContext<R>,
    input: &TensorHandle<R>,
    sigma_space: f32,
    sigma_range: f32,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (h, w) = expect_image(input, "bilateral_filter input")?;
    if sigma_space == 0.0 || sigma_range == 0.0 {
        return Err(crate::Error::InvalidInput(format!(
            "sigma_space/sigma_range must be non-zero, got {sigma_space}/{sigma_range}"
        )));
    }

    let output = TensorHandle::empty(client, vec![h, w], f32_storage());

    unsafe {
        image::bilateral_filter_kernel::launch_unchecked(
            client,
            cubes_for(h * w),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(w as u32),
            ScalarArg::new(h as u32),
            ScalarArg::new(sigma_space),
            ScalarArg::new(sigma_range),
        )
    }
    .map_err(|e| launch_err("bilateral_filter", e))?;

    Ok(output)
}

// ============================================================================
// Pooling Operations
// ============================================================================

mod pooling {
    use super::*;

    /// Max pooling, NCHW `[height, width]`.
    ///
    /// Output geometry is the "no padding, floor" convention:
    /// `out = (in - pool) / stride + 1`.
    ///
    /// BUG(original): `width`/`height`/`pool_size`/`stride` were `u32` while
    /// `x`/`y` come from `ABSOLUTE_POS`, which is `usize`, so the body was
    /// full of mixed-width arithmetic that could never have type-checked.
    /// Everything is `usize` here, which is also what flat indexing wants.
    ///
    /// PORT BUG (found 2026-10-01, not in the original): this kernel compiled
    /// but **panicked at launch time**, on every single call, with
    /// `Can't assign a value to a const variable. Try to use RuntimeCell.`
    /// `max()` is `num_traits::clamp_min` (re-exported by the CubeCL prelude),
    /// and `x = max(a, b)` is `AddAssign` on `f32` in the macro expansion, so
    /// it lowers to `max_val += b` and tries to assign to the `ExpandElement`
    /// the `max()` call returned — which is a temporary, hence "const". An
    /// explicit `if` compares instead. `min()`/`clamp_*` have the same hazard.
    /// `avg_pool2d_kernel`, which uses plain `+=`, was never affected.
    #[cube(launch_unchecked)]
    pub fn max_pool2d_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: usize,
        height: usize,
        pool_size: usize,
        stride: usize,
    ) {
        let out_w = output.shape(1);
        let out_h = output.shape(0);
        let idx = ABSOLUTE_POS;
        if idx >= out_w * out_h {
            terminate!();
        }

        let x = idx % out_w;
        let y = idx / out_w;

        let in_x = x * stride;
        let in_y = y * stride;

        // `max_val` is a RuntimeCell rather than a plain `let mut`.
        //
        // Any branch that assigns to a binding declared outside it makes
        // cubecl 0.9's frontend treat that binding as const, and the expansion
        // then panics with "Can't assign a value to a const variable. Try to use
        // `RuntimeCell`". The bounds check below is a branch, so a plain `let
        // mut` cannot survive it. Using the cell keeps the arithmetic identical
        // and satisfies the frontend.
        let max_val = RuntimeCell::<f32>::new(-1e10f32);

        for py in 0..pool_size {
            for px in 0..pool_size {
                let px_in = in_x + px;
                let py_in = in_y + py;

                if px_in < width && py_in < height {
                    let v = input[py_in * width + px_in];
                    // Not `max_val = max(max_val, v)` — see the port-bug note.
                    let current = max_val.read();
                    max_val.store(select(current > v, current, v));
                }
            }
        }

        output[idx] = max_val.read();
    }

    /// Average pooling, NCHW `[height, width]`.
    ///
    /// Same geometry as [`max_pool2d_kernel`].
    #[cube(launch_unchecked)]
    pub fn avg_pool2d_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: usize,
        height: usize,
        pool_size: usize,
        stride: usize,
    ) {
        let out_w = output.shape(1);
        let out_h = output.shape(0);
        let idx = ABSOLUTE_POS;
        if idx >= out_w * out_h {
            terminate!();
        }

        let x = idx % out_w;
        let y = idx / out_w;

        let in_x = x * stride;
        let in_y = y * stride;

        let mut sum = 0.0f32;
        // Kept as f32 rather than a counter because the macro cannot infer the
        // element type of a `u32` counter that is only ever used as `as f32`.
        let mut count = 0.0f32;

        for py in 0..pool_size {
            for px in 0..pool_size {
                let px_in = in_x + px;
                let py_in = in_y + py;

                if px_in < width && py_in < height {
                    sum += input[py_in * width + px_in];
                    count += 1.0f32;
                }
            }
        }

        output[idx] = sum / count;
    }
}

/// Output geometry shared by both pooling wrappers.
fn pooled_shape(
    w: usize,
    h: usize,
    pool_size: usize,
    stride: usize,
) -> crate::Result<(usize, usize)> {
    if pool_size == 0 || stride == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "pool_size and stride must be > 0, got {pool_size}/{stride}"
        )));
    }
    if w < pool_size || h < pool_size {
        return Err(crate::Error::InvalidInput(format!(
            "pool_size {pool_size} larger than image {h}x{w}"
        )));
    }
    let out_w = (w - pool_size) / stride + 1;
    let out_h = (h - pool_size) / stride + 1;
    if (out_w * out_h) > (u32::MAX as usize) {
        return Err(crate::Error::InvalidInput(
            "pooled image exceeds the addressable size of a 1D launch".into(),
        ));
    }
    Ok((out_h, out_w))
}

/// Max pooling 2D.
///
/// * `input` — `[height, width]`
/// * returns `[out_h, out_w]` with `out = (in - pool_size) / stride + 1`
pub fn max_pool2d<R: Runtime>(
    ctx: &AdvancedContext<R>,
    input: &TensorHandle<R>,
    pool_size: usize,
    stride: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (h, w) = expect_image(input, "max_pool2d input")?;
    let (out_h, out_w) = pooled_shape(w, h, pool_size, stride)?;

    let output = TensorHandle::empty(client, vec![out_h, out_w], f32_storage());

    unsafe {
        pooling::max_pool2d_kernel::launch_unchecked(
            client,
            cubes_for(out_w * out_h),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(w),
            ScalarArg::new(h),
            ScalarArg::new(pool_size),
            ScalarArg::new(stride),
        )
    }
    .map_err(|e| launch_err("max_pool2d", e))?;

    Ok(output)
}

/// Average pooling 2D.
///
/// * `input` — `[height, width]`
/// * returns `[out_h, out_w]` with `out = (in - pool_size) / stride + 1`
pub fn avg_pool2d<R: Runtime>(
    ctx: &AdvancedContext<R>,
    input: &TensorHandle<R>,
    pool_size: usize,
    stride: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (h, w) = expect_image(input, "avg_pool2d input")?;
    let (out_h, out_w) = pooled_shape(w, h, pool_size, stride)?;

    let output = TensorHandle::empty(client, vec![out_h, out_w], f32_storage());

    unsafe {
        pooling::avg_pool2d_kernel::launch_unchecked(
            client,
            cubes_for(out_w * out_h),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(w),
            ScalarArg::new(h),
            ScalarArg::new(pool_size),
            ScalarArg::new(stride),
        )
    }
    .map_err(|e| launch_err("avg_pool2d", e))?;

    Ok(output)
}

/// Upload a `f32` slice as a `[height, width]` image tensor.
pub fn image_from_slice<R: Runtime>(
    ctx: &AdvancedContext<R>,
    data: &[f32],
    height: usize,
    width: usize,
) -> crate::Result<TensorHandle<R>> {
    proto::tensor_from_slice(ctx.client(), data, vec![height, width])
}
