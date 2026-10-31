//! Test helper utilities for cv-hal tests.
//!
//! This module is included by exactly one test binary,
//! `crates/hal/tests/perf_tests.rs` (see `mod helpers;` there), so anything
//! not called from that file is dead weight that the compiler flags with
//! "never used".
//!
//! ## History — why this file shrank to one struct
//!
//! At commit `b709d3c` ("GPU kernel improvements ... and test infrastructure")
//! this module was written as a general-purpose toolbox: 16 free functions,
//! 5 test patterns, an `mspf`-style RNG with four accessors, and tensor
//! builders for both `f32` and `u8`. Only `SimpleRng::new`/`next_f32` were ever
//! called, and only from `perf_tests.rs`. Every other item sat unused since the
//! day it was added.
//!
//! They were measured against the tests that actually exist, not assumed dead:
//!
//! * `create_f32_tensor` / `create_u8_tensor` are byte-for-byte the same as
//!   `create_test_tensor` in `perf_tests.rs:33` and `math_correctness_tests.rs:41`
//!   (`Tensor::from_vec(data.to_vec(), TensorShape::new(c, h, w)).unwrap()`),
//!   which is the construction `Tensor::from_vec` already offers directly. The
//!   helpers also hand-built the struct literal field by field, which is why they
//!   break whenever `Tensor` gains a field.
//! * `compute_variance` is character-identical to the private
//!   `compute_variance` at `math_correctness_tests.rs:291`, which is the copy
//!   that is actually called.
//! * `get_gpu_context` is a second GPU-acquisition path next to
//!   `perf_tests.rs:22`'s `try_gpu_context`, which is the one in use. The
//!   deleted version was also the *wrong* pattern for this repo: it returns
//!   `Option` but its callers in the original design were expected to unwrap,
//!   and the `Ok(_)` arm hides the adapter error. The surviving local helper
//!   prints the reason on the `None` path.
//! * `gaussian_kernel` builds a **2-D** kernel; `cv_hal::cpu::gaussian_kernel_1d`
//!   (the one the production blur actually uses) builds the 1-D form and is
//!   covered by three tests in `cpu_math_tests.rs:65-120`. The 2-D variant was
//!   never called by anything, including the blur.
//! * `patterns::*`, `sequential_f32_tensor`, `random_f32_tensor`,
//!   `random_u8_tensor`, `constant_f32_tensor`, `constant_u8_tensor`,
//!   `compute_mse`, `compute_psnr`, `tensors_close`, `compute_std` and the
//!   `next_f64`/`next_u8`/`next_u32` RNG accessors had no caller and no
//!   equivalent in any test file. `git log -S` over every branch finds no commit
//!   that ever referenced them by name.
//!
//! The decision for each was therefore *delete*, not *write a test*: the
//! remaining CPU-parity coverage is added where it tests real kernels, in
//! `crates/hal/tests/cpu_reference_parity.rs`, rather than by resurrecting a
//! generic toolbox whose only proposed use would be re-testing itself.

/// Simple pseudo-random number generator for tests.
///
/// Only `next_f32` is used by `perf_tests.rs`; the other accessors were removed
/// with the rest of the dead surface.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Note the name says `f32` and the value is produced in `f32`: the high 32
    /// bits of the LCG state are shifted down and divided by `u32::MAX >> 9`, so
    /// the result is exactly representable in `f32` and lands in `[0, 4)`.
    /// Callers in `perf_tests.rs` scale it (`* 255.0`, `* 10.0`), which is where
    /// the intended range comes from.
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f32) / (u32::MAX >> 9) as f32
    }
}
