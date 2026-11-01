//! Shared CubeCL plumbing for the ported `cubecl_*` kernel modules.
//!
//! This module holds the pieces that more than one of the ported files needs:
//! element-wise kernels, a reference matmul, and a host-side reduction.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

/// Storage type for `f32` tensors.
#[inline]
pub fn f32_storage() -> ElemType {
    ElemType::Float(cubecl::ir::FloatKind::F32)
}

/// Storage type for `u32` tensors.
#[inline]
pub fn u32_storage() -> ElemType {
    ElemType::UInt(cubecl::ir::UIntKind::U32)
}

/// Units per cube for element-wise launches.
pub const CUBE_DIM: u32 = 64;

/// WebGPU's guaranteed `maxComputeWorkgroupsPerDimension`.
pub const MAX_CUBES_PER_DIM: u32 = 65535;

/// Cubes covering `n` working units of a kernel indexed by `ABSOLUTE_POS`.
///
/// Up to [`MAX_CUBES_PER_DIM`] cubes this is a 1-D count. Beyond it the
/// count spreads over y as `(65535, ceil(cubes / 65535), 1)`: a 1-D count of
/// more than 65535 is a wgpu validation error (it poisoned the device for any
/// launch over 4,194,240 units). `ABSOLUTE_POS` is linear over the whole grid,
/// so the kernels need no change; the surplus cubes of the last row fall past
/// `n` and every kernel launched with this bounds-checks `ABSOLUTE_POS`.
#[inline]
pub fn cubes_for(n: usize) -> CubeCount {
    let cubes = (n.div_ceil(CUBE_DIM as usize)).max(1);
    let cubes = u32::try_from(cubes).expect("cube count fits in u32");
    if cubes <= MAX_CUBES_PER_DIM {
        CubeCount::new_1d(cubes)
    } else {
        CubeCount::new_2d(MAX_CUBES_PER_DIM, cubes.div_ceil(MAX_CUBES_PER_DIM))
    }
}

/// Upload `data` to a new contiguous `f32` tensor.
pub fn tensor_from_slice(
    client: &Client,
    data: &[f32],
    shape: Vec<usize>,
) -> crate::Result<TensorHandle> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(crate::Error::InvalidInput(format!(
            "tensor_from_slice: shape {shape:?} implies {expected} elements, got {}",
            data.len()
        )));
    }
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let handle = client.create_from_slice(&bytes);
    Ok(TensorHandle::new_contiguous(shape, handle, f32_storage()))
}

/// Read a tensor's bytes back to the host.
///
/// CubeCL 0.11 launches are fire-and-forget: a kernel that fails to compile
/// or launch records the error on the stream, and the next synchronising call
/// returns it. This read is that call, so it is where launch failures surface.
fn read_bytes(client: &Client, tensor: &TensorHandle) -> crate::Result<Vec<u8>> {
    let mut bytes = client
        .read_tensor(vec![tensor.clone().into_copy_descriptor()])
        .map_err(|e| crate::Error::RuntimeError(format!("cubecl readback failed: {e:?}")))?;
    Ok(bytes.remove(0).to_vec())
}

/// Download a `f32` tensor to the host.
pub fn tensor_to_slice(client: &Client, tensor: &TensorHandle) -> crate::Result<Vec<f32>> {
    let bytes = read_bytes(client, tensor)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Download a `u32` tensor to the host.
pub fn tensor_to_slice_u32(client: &Client, tensor: &TensorHandle) -> crate::Result<Vec<u32>> {
    let bytes = read_bytes(client, tensor)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ============================================================================
// Element-wise operations
// ============================================================================

/// Which element-wise binary operation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

/// Which element-wise unary operation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Relu,
    Sigmoid,
    Tanh,
}

#[cube(launch_unchecked)]
fn bin_kernel(lhs: &Tensor<f32>, rhs: &Tensor<f32>, output: &mut Tensor<f32>, n: usize, op: u32) {
    let i = ABSOLUTE_POS;
    if i >= n {
        terminate!();
    }
    let a = lhs[i];
    let b = rhs[i];
    // 0 = add, 1 = sub, 2 = mul
    if op == 0 {
        output[i] = a + b;
    } else if op == 1 {
        output[i] = a - b;
    } else {
        output[i] = a * b;
    }
}

#[cube(launch_unchecked)]
fn un_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>, n: usize, op: u32) {
    let i = ABSOLUTE_POS;
    if i >= n {
        terminate!();
    }
    let x = input[i];
    // 0 = relu, 1 = sigmoid, 2 = tanh
    if op == 0 {
        output[i] = max(x, 0.0f32);
    } else if op == 1 {
        output[i] = 1.0f32 / (1.0f32 + (-x).exp());
    } else {
        let e2x = (2.0f32 * x).exp();
        output[i] = (e2x - 1.0f32) / (e2x + 1.0f32);
    }
}

/// Run a binary element-wise op, allocating the output.
pub fn binary_elemwise(
    client: &Client,
    lhs: &TensorHandle,
    rhs: &TensorHandle,
    op: BinOp,
) -> crate::Result<TensorHandle> {
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::InvalidInput(format!(
            "shape mismatch: {:?} vs {:?}",
            lhs.shape(),
            rhs.shape()
        )));
    }
    let n: usize = lhs.shape().iter().product();
    let output = TensorHandle::empty(client, lhs.shape().clone(), f32_storage());

    let op_code = match op {
        BinOp::Add => 0u32,
        BinOp::Sub => 1u32,
        BinOp::Mul => 2u32,
    };

    unsafe {
        bin_kernel::launch_unchecked(
            client,
            cubes_for(n),
            CubeDim::new_1d(CUBE_DIM),
            lhs.clone().into_arg(),
            rhs.clone().into_arg(),
            output.clone().into_arg(),
            n,
            op_code,
        )
    };

    Ok(output)
}

/// Run a unary element-wise op, allocating the output.
pub fn unary_elemwise(
    client: &Client,
    input: &TensorHandle,
    op: UnOp,
) -> crate::Result<TensorHandle> {
    let n: usize = input.shape().iter().product();
    let output = TensorHandle::empty(client, input.shape().clone(), f32_storage());

    let op_code = match op {
        UnOp::Relu => 0u32,
        UnOp::Sigmoid => 1u32,
        UnOp::Tanh => 2u32,
    };

    unsafe {
        un_kernel::launch_unchecked(
            client,
            cubes_for(n),
            CubeDim::new_1d(CUBE_DIM),
            input.clone().into_arg(),
            output.clone().into_arg(),
            n,
            op_code,
        )
    };

    Ok(output)
}

// ============================================================================
// Reference matmul
// ============================================================================

/// `C[i, j] = sum_k A[i, k] * B[k, j]`, one unit per output element.
#[cube(launch_unchecked)]
fn matmul_kernel(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, m: u32, n: u32, k: u32) {
    let idx = ABSOLUTE_POS;
    let mi = m as usize;
    let ni = n as usize;
    let ki = k as usize;
    if idx >= mi * ni {
        terminate!();
    }
    let i = idx / ni;
    let j = idx % ni;

    let mut acc = 0.0f32;
    for p in 0..ki {
        acc += a[i * ki + p] * b[p * ni + j];
    }
    c[idx] = acc;
}

/// Row-major `f32` matmul on the device.
///
/// This is the naive O(M*N*K) kernel, not a tiled/tensor-core GEMM. It exists
/// so `cubecl_backend::matmul` has a real implementation; it is a correctness
/// reference, not a performance path.
pub fn launch_matmul(
    client: &Client,
    a: &TensorHandle,
    b: &TensorHandle,
) -> crate::Result<TensorHandle> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(crate::Error::InvalidInput(
            "matmul expects rank-2 tensors".into(),
        ));
    }
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k != k2 {
        return Err(crate::Error::InvalidInput(format!(
            "matmul inner dimensions disagree: {k} vs {k2}"
        )));
    }

    let c = TensorHandle::empty(client, vec![m, n], f32_storage());

    unsafe {
        matmul_kernel::launch_unchecked(
            client,
            cubes_for(m * n),
            CubeDim::new_1d(CUBE_DIM),
            a.clone().into_arg(),
            b.clone().into_arg(),
            c.clone().into_arg(),
            m as u32,
            n as u32,
            k as u32,
        )
    };

    Ok(c)
}

// ============================================================================
// Host-side reduction
// ============================================================================

/// Which reduction [`host_reduce`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedOp {
    Sum,
    Max,
    Min,
    Mean,
}

/// Reduce `input` along `axis` on the host, returning a rank-`n-1` tensor.
///
/// Correct but slow: this reads the tensor back and reduces on the CPU.
/// CubeCL 0.9 has no built-in reduction, and a hand-written two-pass parallel
/// reduction is easy to get subtly wrong, so this is deliberately the simple
/// version.
///
/// BUG(port, found 2026-10-01 — not an original defect): `Max` and `Min` were
/// **wrong for `axis > 0`**. The initialiser test was
///
/// ```text
/// if o == 0 && i == 0 { v } else { out[dst].max(v) }
/// ```
///
/// which seeds only output element `dst == 0` (`o == 0, i == 0`) from the data.
/// Every other output element starts at the `vec![0.0]` allocation and therefore
/// returns `0.0` whenever its true maximum is negative, and `Min` is the mirror
/// image (returns `0.0` whenever its true minimum is positive). `Sum` and `Mean`
/// were never affected because `0.0` is their identity.
///
/// The seed is now decided by *this output element's own* first contributing
/// input, `i == 0`, which is correct for every axis and for every shape. Found by
/// `crates/hal/tests/cubecl_kernels_test.rs`, which is checked against a broken
/// `host_reduce` on purpose.
pub fn host_reduce(
    client: &Client,
    input: &TensorHandle,
    axis: usize,
    op: RedOp,
) -> crate::Result<TensorHandle> {
    let shape = input.shape().clone();
    if axis >= shape.len() {
        return Err(crate::Error::InvalidInput(format!(
            "axis {axis} out of range for rank {}",
            shape.len()
        )));
    }

    let data = tensor_to_slice(client, input)?;

    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    let out_shape: Vec<usize> = shape[..axis]
        .iter()
        .chain(shape[axis + 1..].iter())
        .copied()
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);

    let mut out = vec![0.0f32; out_len];

    for o in 0..outer {
        for i in 0..axis_len {
            for n in 0..inner {
                let src = (o * axis_len + i) * inner + n;
                let dst = o * inner + n;
                let v = data[src];
                out[dst] = match op {
                    RedOp::Sum => out[dst] + v,
                    // `i == 0` is the first input that reaches *this* output
                    // element, so it is the right seed for every axis.
                    RedOp::Max => {
                        if i == 0 {
                            v
                        } else {
                            out[dst].max(v)
                        }
                    }
                    RedOp::Min => {
                        if i == 0 {
                            v
                        } else {
                            out[dst].min(v)
                        }
                    }
                    RedOp::Mean => out[dst] + v,
                };
            }
        }
    }

    if matches!(op, RedOp::Mean) {
        let d = axis_len as f32;
        for v in out.iter_mut() {
            *v /= d;
        }
    }

    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    let handle = client.create_from_slice(&bytes);
    Ok(TensorHandle::new_contiguous(
        out_shape,
        handle,
        f32_storage(),
    ))
}
