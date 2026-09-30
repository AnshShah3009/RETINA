//! Optimized CubeCL GPU Kernels for RETINA
//!
//! Everything here was originally written against a CubeCL API that does not
//! exist, and nothing in this file had ever been compiled or run. See the
//! module header of [`crate::gpu_kernels::cubecl_backend`] for the porting
//! notes; the short version is that a `ComputeClient<R>` *is* the context,
//! launches take `CubeCount`/`CubeDim`, tensors are [`TensorHandle`]s, kernels
//! use `#[cube(launch_unchecked)]`, early exit is `terminate!()`, and
//! `Tensor<T>` indexes a flat `usize` only.
//!
//! # What "optimized" meant, and what it means now
//!
//! The original file claimed five things it did not do:
//!
//! | Claim | Reality after the port |
//! |---|---|
//! | shared-memory tiling for conv | **false** — no shared memory is declared anywhere; the code comments that out and launches a scalar per output element |
//! | warp-shuffle reductions | **false** — [`reduction_opt::warp_reduce_sum`] returns its argument unchanged; it is now a *correct* (useless) identity, not a lie |
//! | atomic histogram | **false** — was `bins[bin] += 1` from every unit, i.e. a write race that loses counts. Now uses [`reduction_opt::histogram_kernel`] with real `Atomic<u32>::fetch_add` |
//! | fused kernels | **true**, trivially — see [`element_wise_opt`] |
//! | fp16 / tensor-core paths | **false** — `half::from_f32` is not a CubeCL type and `Tensor<half>` cannot be instantiated. Removed; see [`fp16_is_not_implemented`] |
//!
//! So the surviving kernels are correct scalar reference implementations. They
//! are *not* faster than the ones in [`crate::gpu_kernels::cubecl_backend`].
//! Where a defect in the original survived the port it is called out with a
//! `BUG(original)` comment, and where it is reachable from Rust the wrapper
//! carries the same marker.

use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::gpu_kernels::cubecl_proto as proto;

use proto::{cubes_for, f32_storage, u32_storage, CUBE_DIM};

// ============================================================================
// Context with Performance Settings
// ============================================================================

/// Runtime-generic wrapper around a [`ComputeClient`] plus the precision flag
/// the original `OptimizedCubeCLContext` carried.
///
/// The original pinned itself to `WgpuDevice`, which is both a backend-specific
/// type and (on this crate's wgpu version) not `Clone`, so the `#[derive(Clone,
/// Debug)]` on it could never have compiled. Both traits are kept; the device
/// binding is not.
#[derive(Clone)]
pub struct OptimizedContext<R: Runtime> {
    client: ComputeClient<R>,
    use_fp16: bool,
    tile_size: usize,
}

impl<R: Runtime> OptimizedContext<R> {
    /// Wrap an existing compute client.
    ///
    /// `use_fp16` is retained only so callers keep their configuration; see
    /// [`fp16_is_not_implemented`] — nothing in this module honours it yet.
    pub fn new(client: ComputeClient<R>, use_fp16: bool) -> Self {
        Self {
            client,
            use_fp16,
            tile_size: 16,
        }
    }

    /// Build a context from a device.
    pub fn from_device(device: &R::Device, use_fp16: bool) -> Self {
        Self {
            client: R::client(device),
            use_fp16,
            tile_size: 16,
        }
    }

    /// The underlying compute client.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Whether the caller asked for fp16. Currently advisory only.
    pub fn use_fp16(&self) -> bool {
        self.use_fp16
    }

    /// Tile edge length this context was configured with. Currently advisory
    /// only — no kernel here uses shared memory.
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }
}

/// Whether this device can do `Atomic<f32>` add.
fn has_float_atomic_add<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client
        .properties()
        .type_usage(StorageType::Atomic(ElemType::Float(
            cubecl::ir::FloatKind::F32,
        )))
        .contains(cubecl::ir::features::TypeUsage::AtomicAdd)
}

fn has_uint_atomic_add<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client
        .properties()
        .type_usage(StorageType::Atomic(ElemType::UInt(
            cubecl::ir::UIntKind::U32,
        )))
        .contains(cubecl::ir::features::TypeUsage::AtomicAdd)
}

fn launch_err(what: &str, e: impl std::fmt::Debug) -> crate::Error {
    crate::Error::RuntimeError(format!("{what} launch failed: {e:?}"))
}

// ============================================================================
// Element-wise Operations (fused)
// ============================================================================
//
// These are the only claims in this file that were honest. They are written as
// `#[device]` helpers and inlined into the element-wise kernels below: a fused
// scalar helper called once per element beats two passes over memory, which was
// the actual intent.

mod element_wise_opt {
    use super::*;

    /// `max(0, lhs + rhs)`
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn add_relu(lhs: f32, rhs: f32) -> f32 {
        max(lhs + rhs, 0.0)
    }

    /// `1 / (1 + exp(-(lhs + rhs)))`
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn add_sigmoid(lhs: f32, rhs: f32) -> f32 {
        let sum = lhs + rhs;
        1.0 / (1.0 + (-sum).exp())
    }

    /// Fused multiply-add, `a * b + c`. Exists to show the fusion point; on a
    /// GPU backend `a * b + c` already lowers to a single FMA instruction, so
    /// this is not faster than writing the expression inline.
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn mul_add(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }

    /// Batch-norm followed by ReLU.
    ///
    /// Note the argument order: `(conv_out, scale, bias, mean, var, eps)`. The
    /// original called this "conv + batch norm + relu" but took the batch-norm
    /// output as an argument, so no convolution happens here and never did.
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn conv_bn_relu(
        conv_out: f32,
        scale: f32,
        bias: f32,
        running_mean: f32,
        running_var: f32,
        epsilon: f32,
    ) -> f32 {
        let normalized = (conv_out - running_mean) / (running_var + epsilon).sqrt();
        max(normalized * scale + bias, 0.0)
    }

    /// Leaky ReLU. `alpha` must be in `(0, 1]` to be a leaky ReLU at all.
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            x * alpha
        }
    }

    /// Exponential linear unit.
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn elu(x: f32, alpha: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    }
}

/// Which fused element-wise op to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedOp {
    AddRelu,
    AddSigmoid,
}

mod fused {
    use super::*;

    /// Dispatch the fused binary element-wise ops.
    #[cube(launch_unchecked)]
    pub fn fused_binary_kernel(
        lhs: &Tensor<f32>,
        rhs: &Tensor<f32>,
        output: &mut Tensor<f32>,
        n: usize,
        op: u32,
    ) {
        let i = ABSOLUTE_POS;
        if i >= n {
            terminate!();
        }
        let a = lhs[i];
        let b = rhs[i];
        // 0 = add_relu, 1 = add_sigmoid
        if op == 0 {
            output[i] = element_wise_opt::add_relu(a, b);
        } else {
            output[i] = element_wise_opt::add_sigmoid(a, b);
        }
    }

    /// Dispatch the fused unary element-wise ops.
    #[cube(launch_unchecked)]
    pub fn fused_unary_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>, n: usize, op: u32) {
        let i = ABSOLUTE_POS;
        if i >= n {
            terminate!();
        }
        let x = input[i];
        // 0 = leaky_relu(0.01), 1 = elu(1.0)
        if op == 0 {
            output[i] = element_wise_opt::leaky_relu(x, 0.01);
        } else {
            output[i] = element_wise_opt::elu(x, 1.0);
        }
    }
}

/// `output = max(0, lhs + rhs)` — one pass over memory instead of two.
pub fn add_relu<R: Runtime>(
    ctx: &OptimizedContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    fused_binary(ctx, lhs, rhs, FusedOp::AddRelu)
}

/// `output = sigmoid(lhs + rhs)` — one pass over memory instead of two.
pub fn add_sigmoid<R: Runtime>(
    ctx: &OptimizedContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    fused_binary(ctx, lhs, rhs, FusedOp::AddSigmoid)
}

/// Leaky ReLU with the conventional `alpha = 0.01`.
pub fn leaky_relu<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    fused_unary(ctx, input, 0)
}

/// ELU with the conventional `alpha = 1.0`.
pub fn elu<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    fused_unary(ctx, input, 1)
}

fn fused_binary<R: Runtime>(
    ctx: &OptimizedContext<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    op: FusedOp,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    if lhs.shape != rhs.shape {
        return Err(crate::Error::InvalidInput(format!(
            "shape mismatch: {:?} vs {:?}",
            lhs.shape, rhs.shape
        )));
    }
    let n: usize = lhs.shape.iter().product();
    let output = TensorHandle::empty(client, lhs.shape.clone(), f32_storage());

    let op_code = match op {
        FusedOp::AddRelu => 0u32,
        FusedOp::AddSigmoid => 1u32,
    };

    unsafe {
        fused::fused_binary_kernel::launch_unchecked(
            client,
            cubes_for(n),
            CubeDim::new_1d(CUBE_DIM),
            lhs.as_ref().as_tensor_arg(1),
            rhs.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(n),
            ScalarArg::new(op_code),
        )
    }
    .map_err(|e| launch_err("fused_binary", e))?;

    Ok(output)
}

fn fused_unary<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    op: u32,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let n: usize = input.shape.iter().product();
    let output = TensorHandle::empty(client, input.shape.clone(), f32_storage());

    unsafe {
        fused::fused_unary_kernel::launch_unchecked(
            client,
            cubes_for(n),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(n),
            ScalarArg::new(op),
        )
    }
    .map_err(|e| launch_err("fused_unary", e))?;

    Ok(output)
}

// ============================================================================
// Convolution
// ============================================================================
//
// "With shared memory tiling", in the original. There was no shared memory:
// the declaration was commented out. These are the same scalar
// one-unit-per-output-element loops as `cubecl_backend::conv2d`, so read that
// for the flat-indexing convention.

/// Geometry check shared by the two conv wrappers: computes the output extent
/// from the input, rejecting configurations that would underflow.
fn conv_output_extent(
    extent: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> crate::Result<usize> {
    if stride == 0 {
        return Err(crate::Error::InvalidInput("stride must be > 0".into()));
    }
    if k == 0 {
        return Err(crate::Error::InvalidInput("kernel_size must be > 0".into()));
    }
    let num = extent as isize + 2 * (padding as isize) - (k as isize);
    if num < 0 {
        return Err(crate::Error::InvalidInput(format!(
            "kernel {k} larger than padded input {extent}"
        )));
    }
    Ok(num as usize / stride + 1)
}

mod conv2d_opt {
    use super::*;

    /// Plain 2D convolution, NCHW, one unit per output element.
    ///
    /// * `input` — `[batch, channels_in, height, width]`
    /// * `weights` — `[channels_out, channels_in, k, k]`
    /// * `output` — `[batch, channels_out, out_h, out_w]`
    ///
    /// BUG(original): the original computed `b`, `oy`, `ox` from a single flat
    /// index divided by `(out_height * out_width)`, so every unit in every
    /// channel of a batch wrote to channel 0 — the output tensor was `1`
    /// everywhere except `out[0]`/`out[1]`/`out[2]` for channels 0, 1, 2.
    /// It then indexed `input[[b, c_in, in_y, in_x]]` and
    /// `kernel[[b, c_in, ky, kx]]`, using the *batch* index into the weight
    /// tensor (which has no batch axis at all) and ignoring `channels_out`
    /// entirely. Both are fixed here: the output index is decomposed row-major
    /// over NCHW and the weight offset is `((c_out * channels_in + c_in) * k +
    /// ky) * k + kx`, matching `cubecl_backend::conv2d`.
    #[cube(launch_unchecked)]
    pub fn conv2d_kernel(
        input: &Tensor<f32>,
        weights: &Tensor<f32>,
        output: &mut Tensor<f32>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) {
        let batch = input.shape(0);
        let channels_in = input.shape(1);
        let height_in = input.shape(2);
        let width_in = input.shape(3);

        let channels_out = output.shape(1);
        let height_out = output.shape(2);
        let width_out = output.shape(3);

        let idx = ABSOLUTE_POS;
        let plane = height_out * width_out;
        let total = batch * channels_out * plane;
        if idx >= total {
            terminate!();
        }

        let b = idx / (channels_out * plane);
        let rem = idx % (channels_out * plane);
        let c_out = rem / plane;
        let rem2 = rem % plane;
        let oy = rem2 / width_out;
        let ox = rem2 % width_out;

        let mut sum = 0.0f32;

        for c_in in 0..channels_in {
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let in_y = oy * stride + ky;
                    let in_x = ox * stride + kx;

                    // Zero padding on all four sides.
                    if in_y >= padding && in_x >= padding {
                        let iy = in_y - padding;
                        let ix = in_x - padding;
                        if iy < height_in && ix < width_in {
                            let in_off =
                                ((b * channels_in + c_in) * height_in + iy) * width_in + ix;
                            let w_off = ((c_out * channels_in + c_in) * kernel_size + ky)
                                * kernel_size
                                + kx;
                            sum += input[in_off] * weights[w_off];
                        }
                    }
                }
            }
        }

        output[idx] = sum;
    }

    /// Depthwise convolution, NCHW, one unit per output element.
    ///
    /// * `input` — `[batch, channels, height, width]`
    /// * `weights` — `[channels, k, k]`
    ///
    /// BUG(original): none that survive. The indexing here was already
    /// row-major over NCHW and the weight layout matches. The unsigned
    /// arithmetic around the padding offset (`oy * stride - padding` where
    /// `padding` could exceed `oy * stride`) would have wrapped around rather
    /// than going negative, but the `padding > oy*stride` case only arises when
    /// padding exceeds the input extent, which the wrapper now rejects.
    #[cube(launch_unchecked)]
    pub fn depthwise_conv2d_kernel(
        input: &Tensor<f32>,
        weights: &Tensor<f32>,
        output: &mut Tensor<f32>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) {
        let batch = input.shape(0);
        let channels = input.shape(1);
        let height_in = input.shape(2);
        let width_in = input.shape(3);

        let out_height = output.shape(2);
        let out_width = output.shape(3);

        let idx = ABSOLUTE_POS;
        let plane = channels * out_height * out_width;
        let total = batch * plane;
        if idx >= total {
            terminate!();
        }

        let b = idx / plane;
        let rem = idx % plane;
        let c = rem / (out_height * out_width);
        let rem2 = rem % (out_height * out_width);
        let oy = rem2 / out_width;
        let ox = rem2 % out_width;

        let in_y_start = oy * stride;
        let in_x_start = ox * stride;

        let mut sum = 0.0f32;

        for ky in 0..kernel_size {
            for kx in 0..kernel_size {
                let raw_y = in_y_start + ky;
                let raw_x = in_x_start + kx;

                if raw_y >= padding && raw_x >= padding {
                    let iy = raw_y - padding;
                    let ix = raw_x - padding;
                    if iy < height_in && ix < width_in {
                        sum += input[((b * channels + c) * height_in + iy) * width_in + ix]
                            * weights[(c * kernel_size + ky) * kernel_size + kx];
                    }
                }
            }
        }

        output[idx] = sum;
    }

    /// Transposed convolution (gather formulation), NCHW.
    ///
    /// * `input` — `[batch, channels_in, height, width]`
    /// * `weights` — `[channels_in, channels_out, k, k]`
    ///
    /// Output extent follows the standard definition
    /// `out = (in - 1) * stride - 2 * padding + kernel_size + output_padding`.
    ///
    /// BUG(original): this function did not compute a transposed convolution.
    /// `in_y_start`/`in_x_start` were a single scalar computed from the
    /// *output* coordinate with no reference to `ky`/`kx`, so for every tap it
    /// read the same input pixel and the sum degenerated to `k * k` copies of
    /// one product; it also looped `kx` over the weight index in the *y* guard
    /// (copy-paste bug), read `input[[b, c_out, ...]]` using the output channel
    /// to index the input channels, read `kernel[[c_out, 0, ky, kx]]` — a
    /// second output-channel mix-up — and could underflow the unsigned
    /// expression `(kernel_size - padding)` when `padding > kernel_size`.
    ///
    /// It is replaced with the textbook gather:
    ///
    /// ```text
    /// out[b, co, oy, ox] = sum_{ci, ky, kx}
    ///     w[ci, co, ky, kx] * in[b, ci, oy + padding - ky, ox + padding - kx]
    /// ```
    ///
    /// This is verified against a CPU reference by
    /// `tests/cubecl_kernels_test.rs`.
    #[cube(launch_unchecked)]
    pub fn transposed_conv2d_kernel(
        input: &Tensor<f32>,
        weights: &Tensor<f32>,
        output: &mut Tensor<f32>,
        kernel_size: usize,
        padding: usize,
    ) {
        let batch = input.shape(0);
        let channels_in = input.shape(1);
        let height_in = input.shape(2);
        let width_in = input.shape(3);
        let channels_out = weights.shape(1);

        let out_height = output.shape(2);
        let out_width = output.shape(3);

        let idx = ABSOLUTE_POS;
        let plane = channels_out * out_height * out_width;
        let total = batch * plane;
        if idx >= total {
            terminate!();
        }

        let b = idx / plane;
        let rem = idx % plane;
        let c_out = rem / (out_height * out_width);
        let rem2 = rem % (out_height * out_width);
        let oy = rem2 / out_width;
        let ox = rem2 % out_width;

        let mut sum = 0.0f32;

        for c_in in 0..channels_in {
            for ky in 0..kernel_size {
                let iy_raw = oy + padding;
                if ky > iy_raw {
                    // Falls in the zero padding above the input.
                } else {
                    let iy = iy_raw - ky;
                    if iy < height_in {
                        for kx in 0..kernel_size {
                            let ix_raw = ox + padding;
                            if kx <= ix_raw {
                                let ix = ix_raw - kx;
                                if ix < width_in {
                                    let in_off =
                                        ((b * channels_in + c_in) * height_in + iy) * width_in + ix;
                                    let w_off = ((c_in * channels_out + c_out) * kernel_size + ky)
                                        * kernel_size
                                        + kx;
                                    sum += input[in_off] * weights[w_off];
                                }
                            }
                        }
                    }
                }
            }
        }

        output[idx] = sum;
    }
}

fn expect_rank4<R: Runtime>(
    tensor: &TensorHandle<R>,
    what: &str,
) -> crate::Result<(usize, usize, usize, usize)> {
    if tensor.shape.len() != 4 {
        return Err(crate::Error::InvalidInput(format!(
            "{what} must be rank 4, got {:?}",
            tensor.shape
        )));
    }
    Ok((
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    ))
}

/// 2D convolution.
///
/// * `input` — `[batch, channels_in, height, width]`
/// * `weights` — `[channels_out, channels_in, k, k]`
///
/// BUG(original): see [`conv2d_opt::conv2d_kernel`] — the original collapsed
/// every output channel onto channel 0 and indexed the weight tensor with the
/// batch index. Fixed here, and now numerically verified.
pub fn conv2d<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    weights: &TensorHandle<R>,
    stride: usize,
    padding: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (batch, c_in, h, w) = expect_rank4(input, "conv2d input")?;
    let (c_out, wc_in, k, kw) = expect_rank4(weights, "conv2d weights")?;
    if kw != k {
        return Err(crate::Error::InvalidInput(
            "conv2d expects a square kernel".into(),
        ));
    }
    if wc_in != c_in {
        return Err(crate::Error::InvalidInput(format!(
            "weight channels {wc_in} != input channels {c_in}"
        )));
    }

    let out_h = conv_output_extent(h, k, stride, padding)?;
    let out_w = conv_output_extent(w, k, stride, padding)?;

    let output = TensorHandle::empty(client, vec![batch, c_out, out_h, out_w], f32_storage());

    unsafe {
        conv2d_opt::conv2d_kernel::launch_unchecked(
            client,
            cubes_for(batch * c_out * out_h * out_w),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            weights.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(k),
            ScalarArg::new(stride),
            ScalarArg::new(padding),
        )
    }
    .map_err(|e| launch_err("conv2d", e))?;

    Ok(output)
}

/// Depthwise-separable convolution.
///
/// * `input` — `[batch, channels, height, width]`
/// * `weights` — `[channels, k, k]`
pub fn depthwise_conv2d<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    weights: &TensorHandle<R>,
    stride: usize,
    padding: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (batch, channels, h, w) = expect_rank4(input, "depthwise_conv2d input")?;
    if weights.shape.len() != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "depthwise weights must be [channels, k, k], got {:?}",
            weights.shape
        )));
    }
    if weights.shape[0] != channels {
        return Err(crate::Error::InvalidInput(format!(
            "depthwise weights have {} channels, input has {channels}",
            weights.shape[0]
        )));
    }
    let k = weights.shape[1];
    if weights.shape[2] != k {
        return Err(crate::Error::InvalidInput(
            "depthwise_conv2d expects a square kernel".into(),
        ));
    }

    let out_h = conv_output_extent(h, k, stride, padding)?;
    let out_w = conv_output_extent(w, k, stride, padding)?;

    let output = TensorHandle::empty(client, vec![batch, channels, out_h, out_w], f32_storage());

    unsafe {
        conv2d_opt::depthwise_conv2d_kernel::launch_unchecked(
            client,
            cubes_for(batch * channels * out_h * out_w),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            weights.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(k),
            ScalarArg::new(stride),
            ScalarArg::new(padding),
        )
    }
    .map_err(|e| launch_err("depthwise_conv2d", e))?;

    Ok(output)
}

/// Transposed convolution (deconvolution).
///
/// * `input` — `[batch, channels_in, height, width]`
/// * `weights` — `[channels_in, channels_out, k, k]`
///
/// BUG(original): the original was not a transposed convolution at all; see
/// [`conv2d_opt::transposed_conv2d_kernel`]. It is replaced with the textbook
/// gather and numerically verified.
pub fn transposed_conv2d<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    weights: &TensorHandle<R>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let (batch, c_in, h, w) = expect_rank4(input, "transposed_conv2d input")?;
    let (wc_in, c_out, k, kw) = expect_rank4(weights, "transposed_conv2d weights")?;
    if kw != k {
        return Err(crate::Error::InvalidInput(
            "transposed_conv2d expects a square kernel".into(),
        ));
    }
    if wc_in != c_in {
        return Err(crate::Error::InvalidInput(format!(
            "weight input channels {wc_in} != input channels {c_in}"
        )));
    }
    if stride == 0 {
        return Err(crate::Error::InvalidInput("stride must be > 0".into()));
    }

    let out_h = (h - 1) * stride + k;
    let out_w = (w - 1) * stride + k;
    if out_h < 2 * padding + output_padding || out_w < 2 * padding + output_padding {
        return Err(crate::Error::InvalidInput(format!(
            "output_padding {output_padding} / padding {padding} exceed the {out_h}x{out_w} output"
        )));
    }
    let out_h = out_h - 2 * padding + output_padding;
    let out_w = out_w - 2 * padding + output_padding;

    let output = TensorHandle::empty(client, vec![batch, c_out, out_h, out_w], f32_storage());

    unsafe {
        conv2d_opt::transposed_conv2d_kernel::launch_unchecked(
            client,
            cubes_for(batch * c_out * out_h * out_w),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            weights.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(k),
            ScalarArg::new(padding),
        )
    }
    .map_err(|e| launch_err("transposed_conv2d", e))?;

    Ok(output)
}

// ============================================================================
// Reductions
// ============================================================================

mod reduction_opt {
    use super::*;

    /// The "warp reduction" from the original file.
    ///
    /// BUG(original): it was a no-op dressed up as one. It took a value,
    /// stored it in `result`, and returned `result`, while claiming to perform
    /// a shuffle reduction. Anything built on it computed a per-element partial
    /// and called it a sum.
    ///
    /// The port keeps it as the identity it actually was, but with an honest
    /// name and an honest comment: CubeCL 0.9 exposes no cross-unit reduction
    /// primitive, so there is nothing to call. Building an atomic accumulator
    /// on top of it is what [`histogram_kernel`] does for a histogram; a
    /// general float sum would need the same and is *not* worth it.
    /// Inlined into the calling kernel by the `#[cube]` macro.
    #[cube]
    pub fn reduce_partial(value: f32) -> f32 {
        // Deliberately the identity. See the doc comment: the original
        // claimed to be a warp-shuffle reduction and was not one.
        value
    }

    /// Per-cube partial sums: each unit sums `block_size` consecutive elements.
    ///
    /// `output` is `[ceil(len / block_size)]` — a partial-sum vector, **not** a
    /// full reduction. Summing `output` on the host completes it. See
    /// [`partial_block_sum`] for the documented wrapper.
    #[cube(launch_unchecked)]
    pub fn block_sum_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        len: usize,
        block_size: usize,
    ) {
        let idx = ABSOLUTE_POS;
        let blocks = (len + block_size - 1) / block_size;
        if idx >= blocks {
            terminate!();
        }

        let mut sum = 0.0f32;
        for i in 0..block_size {
            let pos = idx * block_size + i;
            if pos < len {
                sum += input[pos];
            }
        }

        output[idx] = sum;
    }

    /// Histogram with real `Atomic<u32>` increments.
    ///
    /// The original was `bins[bin] += 1` executed by every unit independently —
    /// a lost-update race, so the result depended on how the driver happened to
    /// schedule writes and was wrong whenever two units landed in the same bin.
    /// The comment "Atomic increment would be used here" was an aspiration, not
    /// an implementation; `bins` is `&mut Tensor<u32>` and CubeCL 0.9's
    /// `Tensor<T>` has no atomic accessors, so it could not be fixed by adding a
    /// keyword. The kernel now takes `&Tensor<Atomic<u32>>` and uses
    /// `fetch_add`.
    ///
    /// One unit per input element.
    #[cube(launch_unchecked)]
    pub fn histogram_kernel(
        input: &Tensor<f32>,
        bins: &Tensor<Atomic<u32>>,
        len: usize,
        num_bins: usize,
        min_val: f32,
        max_val: f32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= len {
            terminate!();
        }

        let val = input[idx];
        let span = max_val - min_val;
        if span <= 0.0 {
            terminate!();
        }

        let scaled = (val - min_val) / span * num_bins as f32;
        // Values at or above `max_val` belong in the last bin; anything below
        // `min_val` is clamped into the first.
        let bin = scaled as usize;
        if bin < num_bins {
            bins[bin].fetch_add(1u32);
        } else if bin == num_bins {
            bins[num_bins - 1].fetch_add(1u32);
        }
    }

    /// Per-element index of the maximum value.
    ///
    /// BUG(original): this is not argmax and never was. It ran **one unit per
    /// input element** and wrote `values[i] = input[i]`,
    /// `indices[i] = i`, i.e. it was a copy of the input plus `0, 1, 2, ...`.
    /// There was no cross-element comparison anywhere, so there was no maximum
    /// to report and "track global max" never happened.
    ///
    /// Making it a true argmax needs a reduction, and there is no atomic float
    /// add that works (sums need a separate count) and no portable float
    /// `fetch_max` path through `Tensor`. So the honest port keeps the
    /// per-element behaviour under an accurate name and exposes it as
    /// [`element_values_and_indices`]; a real argmax is left for a reduction
    /// pass that does not exist in this crate.
    #[cube(launch_unchecked)]
    pub fn values_and_indices_kernel(
        input: &Tensor<f32>,
        values: &mut Tensor<f32>,
        indices: &mut Tensor<u32>,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= input.len() {
            terminate!();
        }
        values[idx] = input[idx];
        indices[idx] = idx as u32;
    }
}

/// Per-cube partial sums over `len` elements, `block_size` elements per unit.
///
/// Returns a `[ceil(len / block_size)]` tensor of partial sums. This is **not**
/// a full reduction — sum the result on the host, or feed it to another stage.
pub fn partial_block_sum<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    block_size: usize,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    if block_size == 0 {
        return Err(crate::Error::InvalidInput("block_size must be > 0".into()));
    }
    let len: usize = input.shape.iter().product();
    if len == 0 {
        return Err(crate::Error::InvalidInput(
            "partial_block_sum input must be non-empty".into(),
        ));
    }
    let blocks = (len + block_size - 1) / block_size;

    let output = TensorHandle::empty(client, vec![blocks], f32_storage());

    unsafe {
        reduction_opt::block_sum_kernel::launch_unchecked(
            client,
            cubes_for(blocks),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(len),
            ScalarArg::new(block_size),
        )
    }
    .map_err(|e| launch_err("block_sum", e))?;

    Ok(output)
}

/// Allocate a zeroed `u32` tensor — `TensorHandle::empty` is uninitialised
/// memory, which a histogram over `fetch_add` cannot tolerate.
fn zeroed_u32<R: Runtime>(client: &ComputeClient<R>, len: usize) -> TensorHandle<R> {
    let bytes = vec![0u8; len * 4];
    let alloc = client.create_tensor_from_slice(&bytes, &[len], 4);
    TensorHandle::new_contiguous(vec![len], alloc.handle, u32_storage())
}

/// Histogram of `input` into `num_bins` equal-width bins over `[min, max]`.
///
/// Values are clamped into range; `max` falls into the last bin. The
/// accumulator starts zeroed — `u32` counters, because an f32 atomic add would
/// make the counts inexact past 2^24.
///
/// Requires the device to support `Atomic<u32>` add; otherwise returns
/// [`crate::Error::NotSupported`] rather than silently producing a racy result.
pub fn histogram<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
    num_bins: usize,
    min_val: f32,
    max_val: f32,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    if num_bins == 0 {
        return Err(crate::Error::InvalidInput("num_bins must be > 0".into()));
    }
    if !(max_val > min_val) {
        return Err(crate::Error::InvalidInput(format!(
            "max ({max_val}) must exceed min ({min_val})"
        )));
    }
    let len: usize = input.shape.iter().product();
    if len == 0 {
        return Err(crate::Error::InvalidInput(
            "histogram input must be non-empty".into(),
        ));
    }
    if !has_uint_atomic_add(client) {
        return Err(crate::Error::NotSupported(
            "device does not support Atomic<u32> add; a non-atomic histogram would race".into(),
        ));
    }

    let bins = zeroed_u32(client, num_bins);

    unsafe {
        reduction_opt::histogram_kernel::launch_unchecked(
            client,
            cubes_for(len),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            bins.as_ref().as_tensor_arg(1),
            ScalarArg::new(len),
            ScalarArg::new(num_bins),
            ScalarArg::new(min_val),
            ScalarArg::new(max_val),
        )
    }
    .map_err(|e| launch_err("histogram", e))?;

    Ok(bins)
}

/// Copy every element into `values` and its flat index into `indices`.
///
/// BUG(original): this function was called `argmax_kernel` and did exactly
/// this. It is not argmax and does not claim to be — see
/// [`reduction_opt::values_and_indices_kernel`]. It is kept only so the port is
/// complete; nothing should depend on it.
pub fn element_values_and_indices<R: Runtime>(
    ctx: &OptimizedContext<R>,
    input: &TensorHandle<R>,
) -> crate::Result<(TensorHandle<R>, TensorHandle<R>)> {
    let client = ctx.client();
    let len: usize = input.shape.iter().product();
    if len == 0 {
        return Err(crate::Error::InvalidInput("input must be non-empty".into()));
    }

    let values = TensorHandle::empty(client, vec![len], f32_storage());
    let indices = TensorHandle::empty(client, vec![len], u32_storage());

    unsafe {
        reduction_opt::values_and_indices_kernel::launch_unchecked(
            client,
            cubes_for(len),
            CubeDim::new_1d(CUBE_DIM),
            input.as_ref().as_tensor_arg(1),
            values.as_ref().as_tensor_arg(1),
            indices.as_ref().as_tensor_arg(1),
        )
    }
    .map_err(|e| launch_err("values_and_indices", e))?;

    Ok((values, indices))
}

// ============================================================================
// Point Cloud Operations
// ============================================================================

mod pointcloud_opt {
    use super::*;

    /// Pairwise squared distance, one unit per `(i, j)` pair.
    ///
    /// BUG(original): the "tiled" kernel did not tile. It launched
    /// `num_points * num_points` units and computed
    /// `block_i = idx / block_size`, `block_j = idx % block_size`, i.e. it
    /// computed a *within-block* index pair and ignored `num_points` almost
    /// entirely. Every unit past the first `block_size` produced
    /// `output[block_i * num_points + block_j]` for `block_i >= 1`, and because
    /// `block_j` repeated every `block_size` units the writes collided: the
    /// vast majority of the output was never written and the rest was written
    /// many times with whichever unit happened to land last. It is also the
    /// same computation as
    /// [`crate::gpu_kernels::cubecl_backend::pairwise_squared_distance`], just
    /// more slowly.
    ///
    /// Replaced with the straightforward `n^2` mapping, which is correct.
    #[cube(launch_unchecked)]
    pub fn pairwise_distance_kernel(
        points: &Tensor<f32>,
        output: &mut Tensor<f32>,
        num_points: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points * num_points {
            terminate!();
        }

        let i = idx / num_points;
        let j = idx % num_points;

        let dx = points[i * 3] - points[j * 3];
        let dy = points[i * 3 + 1] - points[j * 3 + 1];
        let dz = points[i * 3 + 2] - points[j * 3 + 2];

        output[idx] = dx * dx + dy * dy + dz * dz;
    }

    /// Morton (Z-order) code for each point, normalised into `[0, 2^bits)`.
    ///
    /// `bits <= 10` — the interleaved code is `3 * bits` wide and must fit in a
    /// `u32`.
    ///
    /// BUG(original): none that survive, once the bit index types agree. The
    /// original shifted by `3 * i` for `i in 0..num_bits` with `num_bits`
    /// unchecked, so `num_bits >= 11` shifted past bit 30 and produced
    /// garbage-or-wrapped codes; the check is now on the host.
    #[cube(launch_unchecked)]
    pub fn morton_code_kernel(
        points: &Tensor<f32>,
        morton_codes: &mut Tensor<u32>,
        num_points: usize,
        min_bound: f32,
        max_bound: f32,
        num_bits: u32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points {
            terminate!();
        }

        let px = points[idx * 3];
        let py = points[idx * 3 + 1];
        let pz = points[idx * 3 + 2];

        let range = max_bound - min_bound;
        let levels = ((1u32 << num_bits) - 1) as f32;

        let nx = (((px - min_bound) / range) * levels) as u32;
        let ny = (((py - min_bound) / range) * levels) as u32;
        let nz = (((pz - min_bound) / range) * levels) as u32;

        let mut code = 0u32;
        let mut i = 0u32;
        while i < num_bits {
            code = code | (((nx >> i) & 1u32) << (3u32 * i));
            code = code | (((ny >> i) & 1u32) << (3u32 * i + 1u32));
            code = code | (((nz >> i) & 1u32) << (3u32 * i + 2u32));
            i += 1u32;
        }

        morton_codes[idx] = code;
    }

    /// k-NN with a distance-based early rejection.
    ///
    /// `distances`/`indices` are `[num_queries, k]`. `k` must be <= 32 (the
    /// register-array capacity compiled into the kernel).
    ///
    /// BUG(original): the early-rejection test was dead code with an
    /// arithmetic slip. It computed
    ///
    /// ```text
    /// let dx = (qx - px).abs(); ...
    /// let min_dist = dx * dx + dy * dy + dz * dz;
    /// if min_dist > worst { continue; }
    /// ```
    ///
    /// which is a squared distance and so was at least in the right units —
    /// but the branch body was `continue`, a statement CubeCL does not have,
    /// and more importantly the *name* is a lie: nothing about it stops early.
    /// The full `num_points` loop body always ran; only the insertion was
    /// skipped. Worse, `for shift in (j + 1..k).rev()` is a Rust-1.26+ reversed
    /// range syntax that has no CubeCL equivalent at all, so the insertion-sort
    /// shift could not have expanded.
    ///
    /// The rejection test is kept (it does save the insertion work) and
    /// expressed as a guarded block instead of `continue`. Everything else is
    /// the same insertion sort as
    /// [`crate::gpu_kernels::cubecl_backend::knn`].
    #[cube(launch_unchecked)]
    pub fn knn_early_stop_kernel(
        points: &Tensor<f32>,
        queries: &Tensor<f32>,
        distances: &mut Tensor<f32>,
        indices: &mut Tensor<u32>,
        k: usize,
        num_points: usize,
        num_queries: usize,
        max_search_radius: f32,
    ) {
        let q_idx = ABSOLUTE_POS;
        if q_idx >= num_queries {
            terminate!();
        }

        let qx = queries[q_idx * 3];
        let qy = queries[q_idx * 3 + 1];
        let qz = queries[q_idx * 3 + 2];

        let mut local_dists = Array::new(32usize);
        let mut local_indices = Array::new(32usize);
        let mut t = 0usize;
        while t < 32usize {
            local_dists[t] = 1e30f32;
            local_indices[t] = 0u32;
            t += 1usize;
        }

        let enabled = max_search_radius > 0.0;
        let mut radius_sq = 1e30f32;
        if enabled {
            radius_sq = max_search_radius * max_search_radius;
        }

        for p_idx in 0..num_points {
            let px = points[p_idx * 3];
            let py = points[p_idx * 3 + 1];
            let pz = points[p_idx * 3 + 2];

            let dx = qx - px;
            let dy = qy - py;
            let dz = qz - pz;
            let dist = dx * dx + dy * dy + dz * dz;

            // Reject anything outside the search radius outright.
            if dist > radius_sq {
                // Outside the radius: skip this point entirely.
            } else {
                let mut j = 0usize;
                while j < k {
                    if dist < local_dists[j] {
                        // Shift the sorted tail right by one.
                        let mut shift = k - 1;
                        while shift > j {
                            local_dists[shift] = local_dists[shift - 1];
                            local_indices[shift] = local_indices[shift - 1];
                            shift -= 1;
                        }
                        local_dists[j] = dist;
                        local_indices[j] = p_idx as u32;
                        break;
                    }
                    j += 1;
                }
            }
        }

        let mut j = 0usize;
        while j < k {
            distances[q_idx * k + j] = local_dists[j];
            indices[q_idx * k + j] = local_indices[j];
            j += 1;
        }
    }
}

/// Pairwise squared distance between all point pairs.
///
/// * `points` — `[num_points, 3]`
///
/// BUG(original): the original "tiled" kernel collapsed to
/// `output[i / block_size, i % block_size]`, writing almost nothing and
/// overwriting the rest. See [`pointcloud_opt::pairwise_distance_kernel`].
pub fn pairwise_squared_distance<R: Runtime>(
    ctx: &OptimizedContext<R>,
    points: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    if points.shape.len() != 2 || points.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "points must be [n, 3], got {:?}",
            points.shape
        )));
    }
    let num_points = points.shape[0];
    let total = num_points * num_points;
    if (total as u64) > (u32::MAX as u64) {
        return Err(crate::Error::InvalidInput(
            "n^2 exceeds the addressable size of a 1D launch".into(),
        ));
    }

    let output = TensorHandle::empty(client, vec![num_points, num_points], f32_storage());

    unsafe {
        pointcloud_opt::pairwise_distance_kernel::launch_unchecked(
            client,
            cubes_for(total),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points),
        )
    }
    .map_err(|e| launch_err("pairwise_distance", e))?;

    Ok(output)
}

/// Morton codes for a point cloud, for spatial hashing / BVH construction.
///
/// * `points` — `[num_points, 3]`
/// * `num_bits` — 1..=10; the code is `3 * num_bits` bits wide
/// * returns `[num_points]` u32
pub fn morton_codes<R: Runtime>(
    ctx: &OptimizedContext<R>,
    points: &TensorHandle<R>,
    min_bound: f32,
    max_bound: f32,
    num_bits: u32,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    if points.shape.len() != 2 || points.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "points must be [n, 3], got {:?}",
            points.shape
        )));
    }
    if num_bits == 0 || num_bits > 10 {
        return Err(crate::Error::InvalidInput(format!(
            "num_bits must be in 1..=10, got {num_bits}"
        )));
    }
    if !(max_bound > min_bound) {
        return Err(crate::Error::InvalidInput(format!(
            "max_bound ({max_bound}) must exceed min_bound ({min_bound})"
        )));
    }
    let num_points = points.shape[0];

    let codes = TensorHandle::empty(client, vec![num_points], u32_storage());

    unsafe {
        pointcloud_opt::morton_code_kernel::launch_unchecked(
            client,
            cubes_for(num_points),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            codes.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points),
            ScalarArg::new(min_bound),
            ScalarArg::new(max_bound),
            ScalarArg::new(num_bits),
        )
    }
    .map_err(|e| launch_err("morton_code", e))?;

    Ok(codes)
}

/// k-NN with a distance-based early rejection.
///
/// * `points` — `[num_points, 3]`
/// * `queries` — `[num_queries, 3]`
/// * `k` — 1..=32
/// * `max_search_radius` — points further than this are skipped entirely;
///   `<= 0` disables the rejection
///
/// BUG(original): see [`pointcloud_opt::knn_early_stop_kernel`] — the
/// "early stop" never stopped anything and the insertion sort could not have
/// expanded.
pub fn knn<R: Runtime>(
    ctx: &OptimizedContext<R>,
    points: &TensorHandle<R>,
    queries: &TensorHandle<R>,
    k: usize,
    max_search_radius: f32,
) -> crate::Result<(TensorHandle<R>, TensorHandle<R>)> {
    let client = ctx.client();
    if k == 0 || k > 32 {
        return Err(crate::Error::InvalidInput(
            "knn requires 1 <= k <= 32".into(),
        ));
    }
    if points.shape.len() != 2 || points.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "points must be [n, 3], got {:?}",
            points.shape
        )));
    }
    if queries.shape.len() != 2 || queries.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "queries must be [n, 3], got {:?}",
            queries.shape
        )));
    }
    let num_points = points.shape[0];
    let num_queries = queries.shape[0];

    let distances = TensorHandle::empty(client, vec![num_queries, k], f32_storage());
    let indices = TensorHandle::empty(client, vec![num_queries, k], u32_storage());

    unsafe {
        pointcloud_opt::knn_early_stop_kernel::launch_unchecked(
            client,
            cubes_for(num_queries),
            CubeDim::new_1d(CUBE_DIM),
            points.as_ref().as_tensor_arg(1),
            queries.as_ref().as_tensor_arg(1),
            distances.as_ref().as_tensor_arg(1),
            indices.as_ref().as_tensor_arg(1),
            ScalarArg::new(k),
            ScalarArg::new(num_points),
            ScalarArg::new(num_queries),
            ScalarArg::new(max_search_radius),
        )
    }
    .map_err(|e| launch_err("knn", e))?;

    Ok((distances, indices))
}

// ============================================================================
// ICP with analytical Jacobians
// ============================================================================

mod icp_opt {
    use super::*;

    /// Point-to-plane ICP residual with the analytic point-to-plane Jacobian.
    ///
    /// One unit per source point; writes `residuals[i]`. The caller applies
    /// `transform` before calling this, so `source` is expected to already be
    /// in the target frame.
    ///
    /// BUG(original): the residual used the wrong correspondence. The kernel
    /// searched the target cloud for the nearest point and remembered that
    /// point's normal, then computed the residual against
    /// `target[[idx, 0..2]]` — i.e. the source point at the *same index* as the
    /// source being processed, not the point it actually found. Since source
    /// and target are different clouds that index is meaningless, so the
    /// residual was `n^T (p_i - q_i)` with an unrelated `q_i` — in general not
    /// the distance to the surface at all. The search result is now carried
    /// forward: the nearest target index is tracked and used.
    #[cube(launch_unchecked)]
    pub fn point_to_plane_icp_kernel(
        source: &Tensor<f32>,
        target: &Tensor<f32>,
        target_normals: &Tensor<f32>,
        residuals: &mut Tensor<f32>,
        num_points: usize,
        num_target: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points {
            terminate!();
        }

        let tx = source[idx * 3];
        let ty = source[idx * 3 + 1];
        let tz = source[idx * 3 + 2];

        // Nearest target point and its normal.
        let mut min_dist = 1e10f32;
        let mut closest = 0u32;
        let mut nx = 0.0f32;
        let mut ny = 0.0f32;
        let mut nz = 1.0f32;

        for t_idx in 0..num_target {
            let dx = tx - target[t_idx * 3];
            let dy = ty - target[t_idx * 3 + 1];
            let dz = tz - target[t_idx * 3 + 2];
            let dist = dx * dx + dy * dy + dz * dz;

            if dist < min_dist {
                min_dist = dist;
                closest = t_idx as u32;
                nx = target_normals[t_idx * 3];
                ny = target_normals[t_idx * 3 + 1];
                nz = target_normals[t_idx * 3 + 2];
            }
        }

        let qx = target[closest as usize * 3];
        let qy = target[closest as usize * 3 + 1];
        let qz = target[closest as usize * 3 + 2];

        // Point-to-plane residual: r = n^T (p - q).
        residuals[idx] = (tx - qx) * nx + (ty - qy) * ny + (tz - qz) * nz;
    }

    /// The analytic point-to-plane Jacobian row for one correspondence.
    ///
    /// Writes six values: `jacobian[6 * i + 0..=5]`.
    ///
    /// For a point-to-plane residual `r = n^T (R p + t - q)` the Jacobian with
    /// respect to the left-multiplied twist `(w, v)` is
    /// `J = [n, p x n]` — which is exactly what the original computed and got
    /// right.
    ///
    /// BUG(original): the *output* was dropped. The kernel computed `j0..j5`
    /// into locals and then reached a comment — "In practice, would use atomic
    /// operations for parallel safety" — and fell off the end. No accumulation
    /// statement existed, so the caller's `jt_j` and `jt_r` were never written
    /// at all. That comment was not honest: it described an intention, and the
    /// 6x6 normal equations were silently absent.
    ///
    /// The honest port exposes the per-point Jacobian, which is what the
    /// original code actually computed and which is well-defined without
    /// atomics. Reducing those rows into `jt_j`/`jt_r` on the host is exact and
    /// single-threaded; doing it on the device would need either a float
    /// `fetch_add` the runtime may not have or a two-pass tree reduction, and
    /// is left as future work rather than shipped as a race.
    #[cube(launch_unchecked)]
    pub fn jacobian_row_kernel(
        source: &Tensor<f32>,
        target: &Tensor<f32>,
        target_normals: &Tensor<f32>,
        jacobian: &mut Tensor<f32>,
        num_points: usize,
        num_target: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= num_points {
            terminate!();
        }

        let tx = source[idx * 3];
        let ty = source[idx * 3 + 1];
        let tz = source[idx * 3 + 2];

        let mut min_dist = 1e10f32;
        let mut nx = 0.0f32;
        let mut ny = 0.0f32;
        let mut nz = 1.0f32;

        for t_idx in 0..num_target {
            let dx = tx - target[t_idx * 3];
            let dy = ty - target[t_idx * 3 + 1];
            let dz = tz - target[t_idx * 3 + 2];
            let dist = dx * dx + dy * dy + dz * dz;

            if dist < min_dist {
                min_dist = dist;
                nx = target_normals[t_idx * 3];
                ny = target_normals[t_idx * 3 + 1];
                nz = target_normals[t_idx * 3 + 2];
            }
        }

        // J = [n, p x n]  (p is the *transformed* source point).
        jacobian[idx * 6] = nx;
        jacobian[idx * 6 + 1] = ny;
        jacobian[idx * 6 + 2] = nz;
        jacobian[idx * 6 + 3] = ty * nz - tz * ny;
        jacobian[idx * 6 + 4] = tz * nx - tx * nz;
        jacobian[idx * 6 + 5] = tx * ny - ty * nx;
    }
}

/// Point-to-plane ICP residuals.
///
/// * `source` — `[num_points, 3]`, already transformed into the target frame
/// * `target` — `[num_target, 3]`
/// * `target_normals` — `[num_target, 3]`
///
/// BUG(original): the original paired the residual against `target[idx]` rather
/// than the target point it had actually found. See
/// [`icp_opt::point_to_plane_icp_kernel`].
pub fn point_to_plane_icp<R: Runtime>(
    ctx: &OptimizedContext<R>,
    source: &TensorHandle<R>,
    target: &TensorHandle<R>,
    target_normals: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let num_points = expect_points(source, "source")?;
    let num_target = expect_points(target, "target")?;
    if target_normals.shape != target.shape {
        return Err(crate::Error::InvalidInput(format!(
            "target_normals {:?} must match target {:?}",
            target_normals.shape, target.shape
        )));
    }
    if num_target == 0 {
        return Err(crate::Error::InvalidInput(
            "target must be non-empty".into(),
        ));
    }

    let residuals = TensorHandle::empty(client, vec![num_points], f32_storage());

    unsafe {
        icp_opt::point_to_plane_icp_kernel::launch_unchecked(
            client,
            cubes_for(num_points),
            CubeDim::new_1d(CUBE_DIM),
            source.as_ref().as_tensor_arg(1),
            target.as_ref().as_tensor_arg(1),
            target_normals.as_ref().as_tensor_arg(1),
            residuals.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points),
            ScalarArg::new(num_target),
        )
    }
    .map_err(|e| launch_err("point_to_plane_icp", e))?;

    Ok(residuals)
}

/// Per-correspondence point-to-plane Jacobian rows, `[num_points, 6]`.
///
/// BUG(original): the original kernel computed these six values and then
/// **discarded them** — the accumulation into `jt_j`/`jt_r` existed only as the
/// comment "In practice, would use atomic operations for parallel safety", so
/// the normal equations the caller allocated were never written. That comment
/// was dishonest: it described work that was not done. This returns the rows
/// themselves (no atomics needed) and documents that the reduction to `jt_j` /
/// `jt_r` belongs on the host.
pub fn icp_jacobian_rows<R: Runtime>(
    ctx: &OptimizedContext<R>,
    source: &TensorHandle<R>,
    target: &TensorHandle<R>,
    target_normals: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    let client = ctx.client();
    let num_points = expect_points(source, "source")?;
    let num_target = expect_points(target, "target")?;
    if target_normals.shape != target.shape {
        return Err(crate::Error::InvalidInput(format!(
            "target_normals {:?} must match target {:?}",
            target_normals.shape, target.shape
        )));
    }
    if num_target == 0 {
        return Err(crate::Error::InvalidInput(
            "target must be non-empty".into(),
        ));
    }

    let jacobian = TensorHandle::empty(client, vec![num_points, 6], f32_storage());

    unsafe {
        icp_opt::jacobian_row_kernel::launch_unchecked(
            client,
            cubes_for(num_points),
            CubeDim::new_1d(CUBE_DIM),
            source.as_ref().as_tensor_arg(1),
            target.as_ref().as_tensor_arg(1),
            target_normals.as_ref().as_tensor_arg(1),
            jacobian.as_ref().as_tensor_arg(1),
            ScalarArg::new(num_points),
            ScalarArg::new(num_target),
        )
    }
    .map_err(|e| launch_err("jacobian_row", e))?;

    Ok(jacobian)
}

/// Reduce `[num_points, 6]` Jacobian rows into the 6x6 `JTJ` and 6-vector `JTr`
/// of the point-to-plane normal equations, on the host.
///
/// Exists so the caller that used to get two silently-zero tensors gets the
/// real thing. The device-side reduction that the original comment promised is
/// not attempted — see [`icp_opt::jacobian_row_kernel`].
pub fn accumulate_normal_equations<R: Runtime>(
    client: &ComputeClient<R>,
    jacobian: &TensorHandle<R>,
    residuals: &TensorHandle<R>,
) -> crate::Result<([f32; 36], [f32; 6])> {
    let j = proto::tensor_to_slice(client, jacobian)?;
    let r = proto::tensor_to_slice(client, residuals)?;
    let n = r.len();
    if j.len() != n * 6 {
        return Err(crate::Error::InvalidInput(format!(
            "jacobian has {} values, expected {} for {n} points",
            j.len(),
            n * 6
        )));
    }

    let mut jtj = [0.0f32; 36];
    let mut jtr = [0.0f32; 6];
    for i in 0..n {
        for a in 0..6 {
            jtr[a] += j[i * 6 + a] * r[i];
            for b in 0..6 {
                jtj[a * 6 + b] += j[i * 6 + a] * j[i * 6 + b];
            }
        }
    }
    Ok((jtj, jtr))
}

fn expect_points<R: Runtime>(t: &TensorHandle<R>, what: &str) -> crate::Result<usize> {
    if t.shape.len() != 2 || t.shape[1] != 3 {
        return Err(crate::Error::InvalidInput(format!(
            "{what} must be [n, 3], got {:?}",
            t.shape
        )));
    }
    Ok(t.shape[0])
}

// ============================================================================
// Half-precision (FP16) and tensor cores
// ============================================================================

/// Why there is no fp16 path here.
///
/// The original file had an `fp16_opt` module with `Tensor<half>` kernels and
/// `half::from_f32`. **None of it could have worked**, and it is deleted rather
/// than ported:
///
/// * `half` is a dev-dependency of this crate, so even if it were in scope it
///   would not be available to the library;
/// * `Tensor<T>` requires `T: CubeType`, and `half::f16` does not implement
///   it — CubeCL's `Float` implementations are for its own `FloatExpand`
///   types and `f16` is not among them;
/// * `half::from_f32` is a conversion into a host-side `f16`, meaningless
///   inside a `#[cube]` kernel;
/// * `layer_norm_fp16_kernel` had *no body at all* beyond a comment and an
///   unused `idx`.
///
/// On top of that the "tensor cores" claim was never implemented:
/// `matmul_fp16` contained `let idx = global_idx();` followed by `// ...`,
/// which stores an index and does nothing with it.
///
/// So: [`select_precision`] still exists because it is a real, testable pure
/// function, but it reports a preference no kernel honours. Wiring fp16 up
/// properly means adding a CubeCL `Float` implementation for f16 and the
/// corresponding storage type — a feature, not a port.
pub const fn fp16_is_not_implemented() -> &'static str {
    "fp16 kernels were never implemented and have been removed; see the module docs"
}

// ============================================================================
// Launch configuration helpers
// ============================================================================

/// `(cube_count, cube_dim_x, cube_dim_y)` for an element-wise launch.
///
/// The device is not queried: CubeCL 0.9 exposes no shared-memory size or
/// max-invocation query on `DeviceProperties` that this needs, so the
/// host-side constant `proto::CUBE_DIM` is used, which is what every kernel in
/// this crate actually launches with.
///
/// BUG(original): the original returned `(grid_size, block_size, 1)`, i.e. cube
/// count, cube dimension and a hard-coded `1` for a dimension nobody set — a
/// triple that mixed a count and two sizes. The intent (query the device, pick
/// a good cube size) was never implemented; the returned triple now actually
/// describes the launch `proto::cubes_for` / `CubeDim::new_1d` produce.
pub fn calculate_launch_config(num_elements: usize) -> (u32, u32, u32) {
    let block_size = proto::CUBE_DIM;
    let grid_size = (num_elements as u32).div_ceil(block_size).max(1);
    (grid_size, block_size, 1)
}

/// Whether this device advertises atomic f32 add.
///
/// Used to decide whether a device-side normal-equation accumulation is even
/// possible. Requires a concrete runtime to answer, so this is the
/// runtime-generic form.
pub fn has_float_atomic_add_on<R: Runtime>(client: &ComputeClient<R>) -> bool {
    has_float_atomic_add(client)
}

/// Operations the [`select_precision`] heuristic considers fp16-friendly.
pub const FP16_BENEFICIAL: [&str; 6] = ["relu", "sigmoid", "tanh", "conv2d", "matmul", "attention"];

/// Precision label for an operation.
///
/// Pure host-side string selection; see [`fp16_is_not_implemented`] for why
/// nothing downstream acts on the answer.
pub fn select_precision(use_fp16: bool, operation: &str) -> &'static str {
    if use_fp16 && FP16_BENEFICIAL.iter().any(|op| operation.contains(op)) {
        "fp16"
    } else {
        "fp32"
    }
}
