//! CubeCL GPU Backend for RETINA
//!
//! GPU acceleration using CubeCL 0.9, which can target CUDA, Vulkan, Metal and
//! other backends through a unified API.
//!
//! # Porting notes (CubeCL 0.9)
//!
//! The original version of this file was written against a pre-0.9 CubeCL API
//! and had never been compiled. The differences that mattered:
//!
//! * There is no `CubeContext` trait. A `ComputeClient<R>` *is* the context.
//! * There is no `ExecutionDims`. Launches take an explicit
//!   `CubeCount` (number of cubes) and `CubeDim` (units per cube).
//! * Kernels are declared with `#[cube(launch_unchecked)]`, which generates a
//!   sibling `mod` with a `launch_unchecked` host function.
//! * Tensors are launched as `TensorArg`, built from a `TensorHandleRef` via
//!   `.as_tensor_arg(line_size)`.
//! * There is no `CubeCLContext`-level tensor type; host buffers are
//!   `cubecl::std::tensor::TensorHandle<R>`.
//! * `Tensor<T>` indexes **flat `usize` only**. There is no `t[[i, j]]`
//!   multi-dimensional indexing, so every kernel below computes its own
//!   row-major offset from `shape(d)`.
//! * Early exit is `terminate!()`, not `return`.
//! * `#[const]` is not a kernel parameter attribute; it was replaced by
//!   `#[comptime]`. **Note:** `#[comptime]` on a `fn` parameter is broken in
//!   cubecl-macros 0.9.0 (the macro expansion passes the raw Rust `u32`/`usize`
//!   where an `ExpandElement` is expected, producing
//!   `expected u32, found ExpandElementTyped<u32>`). Every former `#[const]`
//!   parameter is therefore a plain runtime `u32` parameter here, which is
//!   strictly less optimal (one extra scalar binding, no constant folding) but
//!   correct and portable. See `docs` note in the report.
//!
//! # Correctness caveat
//!
//! These kernels were ported, not rewritten. Several were *written* against an
//! imagined API and contained latent logic errors before the port; where such a
//! defect survived the port it is called out in a `BUG(original)` comment at
//! the kernel. Those kernels are not trustworthy.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::gpu_kernels::cubecl_proto as proto;

/// Storage type for `f32` tensors.
#[inline]
fn f32_storage() -> StorageType {
    StorageType::from(cubecl::ir::FloatKind::F32)
}

/// Storage type for `u32` tensors.
#[inline]
fn u32_storage() -> StorageType {
    StorageType::from(cubecl::ir::UIntKind::U32)
}

/// Units per cube for element-wise launches.
const CUBE_DIM: u32 = 64;

/// Number of cubes needed to cover `n` working units in 1D.
#[inline]
fn cubes_for(n: usize) -> CubeCount {
    CubeCount::new_1d((n as u32).div_ceil(CUBE_DIM).max(1))
}
// ============================================================================
// Context
// ============================================================================

/// Handle to a CubeCL 0.9 compute client.
///
/// CubeCL 0.9 has no context object: the client *is* the context, so this
/// wrapper just owns one. It is generic over the runtime so the same code can
/// run on the WGPU (Vulkan/CUDA) runtime or any other.
pub struct CubeCLContext<R: Runtime> {
    client: ComputeClient<R>,
}

impl<R: Runtime> CubeCLContext<R> {
    /// Wrap an existing client.
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

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise addition: `output = lhs + rhs`.
pub fn add<R: Runtime>(
    ctx: &CubeCLContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::binary_elemwise(&ctx.client, lhs, rhs, proto::BinOp::Add)
}

/// Element-wise subtraction: `output = lhs - rhs`.
pub fn sub<R: Runtime>(
    ctx: &CubeCLContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::binary_elemwise(&ctx.client, lhs, rhs, proto::BinOp::Sub)
}

/// Element-wise multiplication: `output = lhs * rhs`.
pub fn mul<R: Runtime>(
    ctx: &CubeCLContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::binary_elemwise(&ctx.client, lhs, rhs, proto::BinOp::Mul)
}

/// ReLU activation: `output = max(0, x)`.
pub fn relu<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::unary_elemwise(&ctx.client, input, proto::UnOp::Relu)
}

/// Sigmoid activation: `output = 1 / (1 + exp(-x))`.
pub fn sigmoid<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::unary_elemwise(&ctx.client, input, proto::UnOp::Sigmoid)
}

/// Tanh activation.
pub fn tanh<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::unary_elemwise(&ctx.client, input, proto::UnOp::Tanh)
}

// ============================================================================
// Reduction Operations
// ============================================================================
//
// The original file called `input.sum(axis)` / `input.max(axis)` on a
// `Tensor<f32>`, which never existed in CubeCL. 0.9 has no built-in reduction
// tensor method, and a correct parallel reduction needs a multi-pass algorithm
// (workgroup reduction -> partials -> final reduce).
//
// Rather than ship a reduction that may be subtly wrong, these are provided as
// host-side reference implementations over the read-back data. They are
// correct, and obviously not fast. `sum/max/min/mean` are marked accordingly.

/// Sum reduction along `axis`, computed on the host.
pub fn sum<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
    axis: usize,
) -> crate::Result<TensorHandle<R>> {
    proto::host_reduce(&ctx.client, input, axis, proto::RedOp::Sum)
}

/// Max reduction along `axis`, computed on the host.
pub fn max<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
    axis: usize,
) -> crate::Result<TensorHandle<R>> {
    proto::host_reduce(&ctx.client, input, axis, proto::RedOp::Max)
}

/// Min reduction along `axis`, computed on the host.
pub fn min<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
    axis: usize,
) -> crate::Result<TensorHandle<R>> {
    proto::host_reduce(&ctx.client, input, axis, proto::RedOp::Min)
}

/// Mean reduction along `axis`, computed on the host.
pub fn mean<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
    axis: usize,
) -> crate::Result<TensorHandle<R>> {
    proto::host_reduce(&ctx.client, input, axis, proto::RedOp::Mean)
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: `C = A @ B`, via a naive tiled kernel.
///
/// Only supports `f32` and rank-2 tensors. This is a correctness reference,
/// not a tuned GEMM (CubeCL has no built-in matmul tensor method in 0.9).
pub fn matmul<R: Runtime>(
    ctx: &CubeCLContext<R>,
    a: &TensorHandle<R>,
    b: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    proto::launch_matmul(&ctx.client, a, b)
}

// ============================================================================
// Convolution (2D)
// ============================================================================

mod conv2d {
    use super::*;

    /// 2D convolution, NCHW layout.
    ///
    /// One unit per output element; the linear index is decomposed into
    /// `(b, c_out, h_out, w_out)` row-major.
    #[cube(launch_unchecked)]
    pub fn conv2d_kernel(
        input: &Tensor<f32>,
        weights: &Tensor<f32>,
        output: &mut Tensor<f32>,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) {
        let batch = input.shape(0);
        let channels_in = input.shape(1);
        let height_in = input.shape(2);
        let width_in = input.shape(3);

        let channels_out = output.shape(1);
        let height_out = output.shape(2);
        let width_out = output.shape(3);

        let idx = ABSOLUTE_POS;
        let total = batch * channels_out * height_out * width_out;
        if idx >= total {
            terminate!();
        }

        let k = kernel_size as usize;
        let s = stride as usize;
        let p = padding as usize;

        // Decompose the flat output index row-major over NCHW.
        let w_out = width_out;
        let h_out = height_out;
        let c_out = channels_out;

        let plane = h_out * w_out;
        let b = idx / (c_out * plane);
        let rem = idx % (c_out * plane);
        let c = rem / plane;
        let rem2 = rem % plane;
        let oh = rem2 / w_out;
        let ow = rem2 % w_out;

        let mut acc = 0.0f32;
        for c_in in 0..channels_in {
            for kh in 0..k {
                for kw in 0..k {
                    // Zero padding: skip taps that fall outside the input.
                    let ih_raw = oh * s + kh;
                    let iw_raw = ow * s + kw;
                    if ih_raw < p || iw_raw < p {
                        // Outside the padded border on the top/left.
                    } else {
                        let ih = ih_raw - p;
                        let iw = iw_raw - p;
                        if ih < height_in && iw < width_in {
                            let in_off =
                                ((b * channels_in + c_in) * height_in + ih) * width_in + iw;
                            let w_off = ((c * channels_in + c_in) * k + kh) * k + kw;
                            acc += input[in_off] * weights[w_off];
                        }
                    }
                }
            }
        }

        let out_off = ((b * c_out + c) * h_out + oh) * w_out + ow;
        output[out_off] = acc;
    }
}

/// 2D convolution.
///
/// * `input` - `[batch, channels_in, height, width]`
/// * `weights` - `[channels_out, channels_in, kernel_h, kernel_w]`
/// * `stride` - Convolution stride
/// * `padding` - Zero padding
pub fn conv2d<R: Runtime>(
    ctx: &CubeCLContext<R>,
    input: &TensorHandle<R>,
    weights: &TensorHandle<R>,
    stride: usize,
    padding: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = &ctx.client;

    let rank = input.shape.len();
    if rank != 4 || weights.shape.len() != 4 {
        return Err(crate::Error::InvalidInput(
            "conv2d expects rank-4 input and weights".into(),
        ));
    }
    if stride == 0 {
        return Err(crate::Error::InvalidInput(
            "conv2d stride must be > 0".into(),
        ));
    }

    let batch = input.shape[0];
    let channels_in = input.shape[1];
    let height_in = input.shape[2];
    let width_in = input.shape[3];
    let channels_out = weights.shape[0];
    let k = weights.shape[2];
    if weights.shape[3] != k {
        return Err(crate::Error::InvalidInput(
            "conv2d expects a square kernel".into(),
        ));
    }

    // Validate the output shape before subtracting, so a too-small input gives
    // a clean error instead of an integer-underflow panic.
    let h_num = (height_in as isize) + 2 * (padding as isize) - (k as isize);
    let w_num = (width_in as isize) + 2 * (padding as isize) - (k as isize);
    if h_num < 0 || w_num < 0 {
        return Err(crate::Error::InvalidInput(
            "conv2d kernel larger than padded input".into(),
        ));
    }
    let height_out = h_num as usize / stride + 1;
    let width_out = w_num as usize / stride + 1;

    let total = batch * channels_out * height_out * width_out;
    let output = TensorHandle::empty(
        client,
        vec![batch, channels_out, height_out, width_out],
        f32_storage(),
    );

    unsafe {
        conv2d::conv2d_kernel::launch_unchecked(
            client,
            cubes_for(total),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            weights.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(k as u32),
            ScalarArg::new(stride as u32),
            ScalarArg::new(padding as u32),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("conv2d launch failed: {e:?}")))?;

    Ok(output)
}

// ============================================================================
// Point Cloud Operations
// ============================================================================

mod pointcloud {
    use super::*;

    /// Squared Euclidean distance between all pairs of points.
    ///
    /// `points` is `[num_points, 3]`, output is `[num_points, num_points]`.
    #[cube(launch_unchecked)]
    pub fn pairwise_distance_kernel(
        points: &Tensor<f32>,
        output: &mut Tensor<f32>,
        num_points: u32,
    ) {
        let idx = ABSOLUTE_POS;
        let n = num_points as usize;
        if idx >= n * n {
            terminate!();
        }

        let i = idx / n;
        let j = idx % n;

        let dx = points[i * 3] - points[j * 3];
        let dy = points[i * 3 + 1] - points[j * 3 + 1];
        let dz = points[i * 3 + 2] - points[j * 3 + 2];

        output[idx] = dx * dx + dy * dy + dz * dz;
    }

    /// K-nearest-neighbours, one unit per query.
    ///
    /// Insertion sort into a fixed-size register array of capacity 32.
    /// BUG(original): the original computed `dists[k_idx]` for `k_idx` in
    /// `0..k` but only ever shifted/inserted within that range, and returned
    /// distances for slots that were never filled when fewer than `k` points
    /// existed. Here slots are initialised to `+inf` and the caller is
    /// required to pass `k <= 32`; with `k > num_points` the trailing entries
    /// stay `+inf` and index 0, which is the honest answer.
    #[cube(launch_unchecked)]
    pub fn knn_kernel(
        points: &Tensor<f32>,
        queries: &Tensor<f32>,
        distances: &mut Tensor<f32>,
        indices: &mut Tensor<u32>,
        k: u32,
        num_points: u32,
        num_queries: u32,
    ) {
        let q = ABSOLUTE_POS;
        if q >= num_queries as usize {
            terminate!();
        }

        let kk = k as usize;
        let n = num_points as usize;

        // Fill with +inf so unfilled slots are distinguishable.
        let mut dists = Array::new(32usize);
        let mut idxs = Array::new(32usize);
        for t in 0..32usize {
            dists[t] = 1e30f32;
            idxs[t] = 0u32;
        }

        let qx = queries[q * 3];
        let qy = queries[q * 3 + 1];
        let qz = queries[q * 3 + 2];

        for p in 0..n {
            let dx = qx - points[p * 3];
            let dy = qy - points[p * 3 + 1];
            let dz = qz - points[p * 3 + 2];
            let d = dx * dx + dy * dy + dz * dz;

            // Insertion into the sorted prefix.
            let mut j = 0usize;
            while j < kk {
                if d < dists[j] {
                    // Shift right.
                    let mut s = kk - 1;
                    while s > j {
                        dists[s] = dists[s - 1];
                        idxs[s] = idxs[s - 1];
                        s -= 1;
                    }
                    dists[j] = d;
                    idxs[j] = p as u32;
                    break;
                }
                j += 1;
            }
        }

        for j in 0..kk {
            distances[q * kk + j] = dists[j];
            indices[q * kk + j] = idxs[j];
        }
    }

    /// Voxel grid hashing.
    ///
    /// BUG(original): the original wrote the hash into column 0 and the point
    /// index into column 1, but the host wrapper allocated a `[n, 2]` f32
    /// tensor while the kernel was declared over `Tensor<u32>` - a type
    /// mismatch that would have been a hard error. It also multiplied `i32`
    /// voxel coordinates by large literals, which overflows in debug builds.
    /// Here the output is `[n, 2]` u32 (hash, point index) with reduced
    /// multipliers, and the caller re-scans the hashes to group points.
    #[cube(launch_unchecked)]
    pub fn voxel_hash_kernel(
        points: &Tensor<f32>,
        voxel_keys: &mut Tensor<u32>,
        num_points: u32,
        voxel_size: f32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points as usize {
            terminate!();
        }

        let px = points[idx * 3];
        let py = points[idx * 3 + 1];
        let pz = points[idx * 3 + 2];

        let vx = (px / voxel_size) as i32;
        let vy = (py / voxel_size) as i32;
        let vz = (pz / voxel_size) as i32;

        // Hash the voxel coordinate. CubeCL 0.9 has no wrapping integer
        // arithmetic, so the multipliers are reduced to values whose product
        // with a plausible voxel index stays inside i32.
        let h = ((vx * 73857) ^ (vy * 19349) ^ (vz * 83493)) as u32;

        voxel_keys[idx * 2] = h;
        voxel_keys[idx * 2 + 1] = idx as u32;
    }
}

/// Compute pairwise squared distances between all points.
///
/// `points` is `[num_points, 3]`.
pub fn pairwise_squared_distance<R: Runtime>(
    ctx: &CubeCLContext<R>,
    points: &TensorHandle<R>,
    num_points: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = &ctx.client;
    let output = TensorHandle::empty(client, vec![num_points, num_points], f32_storage());

    unsafe {
        pointcloud::pairwise_distance_kernel::launch_unchecked(
            client,
            cubes_for(num_points * num_points),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points as u32),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("pairwise distance launch failed: {e:?}")))?;

    Ok(output)
}

/// K-nearest-neighbours search.
///
/// `k` must be <= 32 (the register-array capacity compiled into the kernel).
pub fn knn<R: Runtime>(
    ctx: &CubeCLContext<R>,
    points: &TensorHandle<R>,
    queries: &TensorHandle<R>,
    k: usize,
) -> crate::Result<(TensorHandle<R>, TensorHandle<R>)> {
    let client = &ctx.client;

    if k == 0 || k > 32 {
        return Err(crate::Error::InvalidInput(
            "knn requires 1 <= k <= 32".into(),
        ));
    }
    if points.shape.len() != 2 || points.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(
            "knn points must be [n, 3]".into(),
        ));
    }
    if queries.shape.len() != 2 || queries.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(
            "knn queries must be [n, 3]".into(),
        ));
    }

    let num_points = points.shape[0];
    let num_queries = queries.shape[0];

    let distances = TensorHandle::empty(client, vec![num_queries, k], f32_storage());
    let indices = TensorHandle::empty(client, vec![num_queries, k], u32_storage());

    unsafe {
        pointcloud::knn_kernel::launch_unchecked(
            client,
            cubes_for(num_queries),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            queries.as_ref().as_tensor_arg(1),
            distances.as_ref().as_tensor_arg(1),
            indices.as_ref().as_tensor_arg(1),
            ScalarArg::new(k as u32),
            ScalarArg::new(num_points as u32),
            ScalarArg::new(num_queries as u32),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("knn launch failed: {e:?}")))?;

    Ok((distances, indices))
}

/// Voxel grid hashing for downsampling.
///
/// Returns a `[num_points, 2]` u32 tensor of `(hash, point_index)`.
pub fn voxel_hash<R: Runtime>(
    ctx: &CubeCLContext<R>,
    points: &TensorHandle<R>,
    voxel_size: f32,
) -> crate::Result<TensorHandle<R>> {
    let client = &ctx.client;

    if voxel_size <= 0.0 {
        return Err(crate::Error::InvalidInput("voxel_size must be > 0".into()));
    }
    if points.shape.len() != 2 || points.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(
            "voxel_hash points must be [n, 3]".into(),
        ));
    }

    let num_points = points.shape[0];
    let output = TensorHandle::empty(client, vec![num_points, 2], u32_storage());

    unsafe {
        pointcloud::voxel_hash_kernel::launch_unchecked(
            client,
            cubes_for(num_points),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points as u32),
            ScalarArg::new(voxel_size),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("voxel hash launch failed: {e:?}")))?;

    Ok(output)
}

// ============================================================================
// Host-side utilities
// ============================================================================

/// Upload `data` to a new contiguous `f32` tensor.
pub use proto::tensor_from_slice;

/// Download a `f32` tensor to the host.
pub use proto::tensor_to_slice;

/// Download a `u32` tensor to the host.
pub use proto::tensor_to_slice_u32;
