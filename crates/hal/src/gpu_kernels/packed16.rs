//! Host side of the `*_packed16.wgsl` shaders, which run f16 and bf16 tensors
//! on any device.
//!
//! A 16-bit tensor's buffer is bound as `array<u32>`, two values per word. The
//! shaders widen each value to f32 with integer operations (exact for both
//! formats, and needing neither `SHADER_F16` nor a bf16 type, which WGSL does
//! not have) and narrow results with round-to-nearest-even, as `half` does.
//! One shader serves both formats; `Packed16::code` selects the widening.
//!
//! Native `enable f16` shaders are not used for these kernels even when the
//! device has `SHADER_F16`: WGSL leaves the rounding of an f32 to f16
//! conversion to the implementation and lets the compiler contract a multiply
//! and add, so they could not reproduce the CPU backend's f16 results, and
//! one path everywhere is one behaviour to test.
//!
//! Subnormals: a bf16 value below 2^-126 widens to an f32 subnormal, and a
//! product, sum or difference of small bf16 values can land there too. A
//! device may flush f32 subnormals to zero in arithmetic and comparisons (the
//! RTX 5070 Ti on Vulkan does). Every f16 value and intermediate is a normal
//! f32, so f16 is unaffected. For bf16:
//!
//! * `threshold_bf16.wgsl` and `fast_nms_packed16.wgsl` compare bit patterns.
//! * `fast_packed16.wgsl` and `color_cvt_packed16.wgsl` do their f32
//!   operations on bits: `mul_bits` (the exact significand product rounded at
//!   the result's precision) and `add_bits` (one float add where no operand
//!   or result can be subnormal, otherwise the same add scaled by 2^64), with
//!   comparisons through an order key. They stay bit-identical to the CPU on
//!   subnormal inputs.
//! * `bilateral_packed16.wgsl` is not: on an image whose intensities are
//!   below about 2^-126 its weighted sums flush, exactly as the f32 shader's
//!   do, so it still equals the GPU f32 path narrowed but not the CPU there.
//!   Its contract is that f32 path (the weights come from the device's `exp`,
//!   so it is not CPU-exact anyway), and reproducing the CPU would mean
//!   rewriting the f32 bilateral's accumulation in the scaled domain, where
//!   the range kernel `exp(-d^2 / (2 sigma_color^2))` does not scale with the
//!   image unless sigma_color does.

use crate::Result;

/// The 16-bit float format of a tensor bound as packed u32 words.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Packed16 {
    F16,
    Bf16,
}

impl Packed16 {
    /// The shaders' `params.format`.
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }
}

/// Size of a buffer holding `len` 16-bit values: whole u32 words, because a
/// storage binding must be a multiple of 4 bytes and an odd count is not.
/// Never zero, which a binding also rejects.
pub(crate) fn buffer_bytes(len: usize) -> u64 {
    (len.div_ceil(2).max(1) * 4) as u64
}

/// The 16 bits of a 2-byte value, in the low half of a u32.
pub(crate) fn bits_of<T: bytemuck::Pod>(v: T) -> Result<u32> {
    let b = bytemuck::bytes_of(&v);
    match b {
        [lo, hi] => Ok(u16::from_le_bytes([*lo, *hi]) as u32),
        _ => Err(crate::Error::InvalidInput(format!(
            "expected a 16-bit value, got {} bytes",
            b.len()
        ))),
    }
}

/// Round `v` to `T` and widen it back: the f32 value the shaders see for a
/// constant the CPU backend computes as `T::from_f32(v)`.
pub(crate) fn rounded<T: cv_core::float::Float>(v: f32) -> f32 {
    T::from_f32(v).to_f32()
}
