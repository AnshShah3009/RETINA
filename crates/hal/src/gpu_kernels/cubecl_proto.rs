//! Shared CubeCL 0.9 plumbing for the ported `cubecl_*` kernel modules.
//!
//! This module holds the pieces that more than one of the ported files needs:
//! element-wise kernels, a reference matmul, and a host-side reduction.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

/// Storage type for `f32` tensors.
#[inline]
pub fn f32_storage() -> StorageType {
    StorageType::from(cubecl::ir::FloatKind::F32)
}

/// Storage type for `u32` tensors.
#[inline]
pub fn u32_storage() -> StorageType {
    StorageType::from(cubecl::ir::UIntKind::U32)
}

/// Units per cube for element-wise launches.
pub const CUBE_DIM: u32 = 64;

/// Number of cubes needed to cover `n` working units in 1D.
#[inline]
pub fn cubes_for(n: usize) -> CubeCount {
    CubeCount::new_1d((n as u32).div_ceil(CUBE_DIM).max(1))
}

/// Upload `data` to a new contiguous `f32` tensor.
pub fn tensor_from_slice<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    shape: Vec<usize>,
) -> crate::Result<TensorHandle<R>> {
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

/// Download a `f32` tensor to the host.
pub fn tensor_to_slice<R: Runtime>(
    client: &ComputeClient<R>,
    tensor: &TensorHandle<R>,
) -> crate::Result<Vec<f32>> {
    let bytes = client.read_one_tensor(tensor.as_copy_descriptor());
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Download a `u32` tensor to the host.
pub fn tensor_to_slice_u32<R: Runtime>(
    client: &ComputeClient<R>,
    tensor: &TensorHandle<R>,
) -> crate::Result<Vec<u32>> {
    let bytes = client.read_one_tensor(tensor.as_copy_descriptor());
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
pub fn binary_elemwise<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    op: BinOp,
) -> crate::Result<TensorHandle<R>> {
    if lhs.shape != rhs.shape {
        return Err(crate::Error::InvalidInput(format!(
            "shape mismatch: {:?} vs {:?}",
            lhs.shape, rhs.shape
        )));
    }
    let n: usize = lhs.shape.iter().product();
    let output = TensorHandle::empty(client, lhs.shape.clone(), f32_storage());

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
            lhs.as_ref().as_tensor_arg(1),
            rhs.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(n),
            ScalarArg::new(op_code),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("elemwise launch failed: {e:?}")))?;

    Ok(output)
}

/// Run a unary element-wise op, allocating the output.
pub fn unary_elemwise<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    op: UnOp,
) -> crate::Result<TensorHandle<R>> {
    let n: usize = input.shape.iter().product();
    let output = TensorHandle::empty(client, input.shape.clone(), f32_storage());

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
            input.as_ref().as_tensor_arg(1),
            output.as_ref().as_tensor_arg(1),
            ScalarArg::new(n),
            ScalarArg::new(op_code),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("unary launch failed: {e:?}")))?;

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
pub fn launch_matmul<R: Runtime>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
    b: &TensorHandle<R>,
) -> crate::Result<TensorHandle<R>> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(crate::Error::InvalidInput(
            "matmul expects rank-2 tensors".into(),
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
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
            a.as_ref().as_tensor_arg(1),
            b.as_ref().as_tensor_arg(1),
            c.as_ref().as_tensor_arg(1),
            ScalarArg::new(m as u32),
            ScalarArg::new(n as u32),
            ScalarArg::new(k as u32),
        )
    }
    .map_err(|e| crate::Error::RuntimeError(format!("matmul launch failed: {e:?}")))?;

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
pub fn host_reduce<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    axis: usize,
    op: RedOp,
) -> crate::Result<TensorHandle<R>> {
    let shape = input.shape.clone();
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
                    RedOp::Max => {
                        if o == 0 && i == 0 {
                            v
                        } else {
                            out[dst].max(v)
                        }
                    }
                    RedOp::Min => {
                        if o == 0 && i == 0 {
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
