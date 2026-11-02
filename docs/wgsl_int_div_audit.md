# WGSL integer division audit

Every integer `/` and `%` in the workspace's compute shaders, with the reason
its divisor can never be zero. `crates/hal/tests/wgsl_int_div_audit.rs`
re-derives the list from naga's IR and fails when it and the divisor table
below disagree. When it fails it prints the row to add.

## Why this exists

WGSL defines `x / 0` as `x`, `x % 0` as `0`, and `INT_MIN / -1` as `INT_MIN`.
SPIR-V and MSL leave those cases undefined. To close the gap, naga wraps every
integer division and modulo in a helper that swaps a bad divisor for 1. wgpu 30
can remove that helper from a shader module (gfx-rs/wgpu#9443, "Ability to
disable integer division safety checks on Vulkan and Metal"):

```rust
let checks = wgpu::ShaderRuntimeChecks {
    int_div_checks: false,
    ..wgpu::ShaderRuntimeChecks::checked()
};
// SAFETY: every divisor in `desc` is nonzero, and no signed division is
// INT_MIN / -1.
let module = unsafe { device.create_shader_module_trusted(desc, checks) };
```

On Vulkan the flag reaches naga as `spv::Options::emit_int_div_checks`
(`wgpu-hal` `vulkan/device.rs`). On Metal it becomes
`msl::Options::emit_int_div_checks`. Signed `%` is lowered to
`a - b * (a / b)` with or without the flag, because `OpSRem` on a negative
operand is poison without `VK_KHR_maintenance8`. So turning the checks off
removes only the divisor guard. The truncated semantics stay the same.

You can turn the checks off only after proving every divisor nonzero. This
document is that proof. The measurement below shows the change gains nothing
on this GPU, so the checks are **still on**. Every compute shader is created
with `Device::create_shader_module`.

## Scope and method

The corpus is every `.wgsl` file under `crates/hal/shaders`, `crates/hal/src`,
`crates/calib3d/src` and `crates/features/src`, plus every Rust raw string in
`crates/hal/src` that contains `@compute`. That comes to 108 files and 9 inline
shaders. `crates/viewer` is not included because it holds render shaders only.
The fused-kernel WGSL generated in `crates/runtime/src/pipeline/fusion.rs` is
not included either: it is stored as a string and never compiled.

Each source is parsed and validated by naga, the same front end wgpu uses.
Every `Expression::Binary { Divide | Modulo }` whose result is a signed or
unsigned integer (scalar or vector) is recorded. The text of its right operand
names the divisor. This works on the IR, not the text, so float division,
comments, and divisions folded at compile time never appear. Compound
assignment (`a %= b`) and named constants are seen through.

### Classes

* **A**: nonzero constant. The test checks the value: never 0, and never -1
  for signed operations.
* **B**: runtime value the host guarantees nonzero. The row cites the host code.
* **S**: runtime value the shader itself proves nonzero, by a dominating early
  return or by arithmetic (`2r + 1` is odd).
* **C**: could be zero. **None remain.** The rows that were C before this audit
  are marked *guard added*.

A row's key is (source, operator, divisor text), and the row records how many
times that key occurs. Inline shaders are keyed `file.rs#binding`, where
`binding` is the `let` or `fn` that holds the raw string.

### `ComputeContext::dispatch` passes raw uniforms

`ComputeContext::dispatch(name, buffers, uniforms, workgroups)` runs a named
built-in kernel (`GpuContext::get_kernel_source`) with uniform bytes supplied by
the caller. The host has no typed parameters on that path, so it cannot
validate them. Any class-B argument therefore holds only for the typed entry
points. That is why every guard added here sits in the shader, not the host:
an S divisor stays nonzero whatever bytes the uniform buffer holds. The one
remaining B source, `spatial.rs`, is not reachable through `dispatch`.

## Summary

92 integer divisions or modulos, spread over 34 of the 117 sources that naga
accepts (70 table rows):

| class | sites | rows | before this audit |
| --- | ---: | ---: | --- |
| A, nonzero constant | 57 | 37 | 55 |
| B, host-guaranteed | 2 | 2 | 4 (`feature_collection_f32.wgsl` moved to S) |
| S, shader-proven | 33 | 31 | 8 |
| C, needs a guard | 0 | 0 | 20 |

### Guards added

Each guard is in the shader, so it also covers the raw-uniform `dispatch` path.

* **`remap.wgsl`, `undistort.wgsl` `map_coord`**. `if (n <= 0) { return -1; }`
  at the top. An empty source axis with a non-empty destination used to reach
  `coord % n` with `n == 0` in Wrap mode, and `2 * n` in Reflect mode. Neither
  host (`gpu_kernels/remap.rs`, `gpu_kernels/undistort.rs`) rejects an empty
  source. With `n >= 1`, the divisors `n`, `2n` (even, never 0 for
  `n < 2^31`), and `2n - 2` (only when `n > 1`) are never 0 or -1. An empty
  source now samples the border, the same result Constant mode always gave.
* **`convolve_2d.wgsl`, `gaussian_blur_separable.wgsl` `get_input_index`**.
  `if (w <= 0 || h <= 0) { return -1; }`. `main` already returns for a zero
  width or height. But `w = i32(params.width)` is negative from 2^31 up, and
  at exactly 2^31 `2 * w` wraps to 0. Only a raw `dispatch` uniform could do
  that, because a typed call needs an image row of 2^31 f32 (8 GiB) in one
  storage binding. The guard
  makes every divisor (`w`, `h`, `2w`, `2h`, and `2n - 2` behind the existing
  `n > 1` / `n == 1` branches) at least 1.
* **`feature_collection.wgsl`, `feature_collection_f32.wgsl`
  `collect_points`**. The `% width` and `/ width` moved inside the existing
  `idx < width * height` check, which is false when width is 0. The f32 copy
  was already safe because `fast::extract_keypoints` returns early when
  `w * h == 0`. The packed-u8 copy has no host. No Rust file includes it.

## Divisor table

| source | op | divisor | count | class | why nonzero |
| --- | --- | --- | ---: | --- | --- |
| `crates/calib3d/src/stereo_matching/shaders/stereo.wgsl` | `/` | `2u` | 2 | A | literal |
| `crates/hal/shaders/bilateral.wgsl` | `%` | `4u` | 1 | A | literal (packed u8 lane) |
| `crates/hal/shaders/bilateral.wgsl` | `/` | `4u` | 2 | A | literal (packed u8 word) |
| `crates/hal/shaders/canny.wgsl` | `%` | `params.width` | 2 | S | `sobel` returns unless `i < width * height`. `nms` returns unless `word * 4 < n` and calls `nms_state(i)` only for `i < n = width * height`. Width 0 makes the product 0, so neither path reaches the division. |
| `crates/hal/shaders/canny.wgsl` | `/` | `params.width` | 2 | S | as above |
| `crates/hal/shaders/cast.wgsl` | `%` | `4u` | 1 | A | literal |
| `crates/hal/shaders/cast.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/color_cvt.wgsl` | `%` | `4u` | 1 | A | literal |
| `crates/hal/shaders/color_cvt.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `2 * h - 2` | 1 | S | Reflect101 branch runs only for `h != 1`. With `h >= 1` (guard added), `2h - 2 >= 2` and even. |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `2 * h` | 1 | S | Reflect branch, `h >= 1` (guard added), even, so never 0 or -1 |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `2 * w - 2` | 1 | S | Reflect101 branch runs only for `w != 1`. With `w >= 1` (guard added), `2w - 2 >= 2`. |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `2 * w` | 1 | S | Reflect branch, `w >= 1` (guard added) |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `i32(params.height)` | 1 | S | Wrap branch, `h >= 1` (guard added) |
| `crates/hal/shaders/convolve_2d.wgsl` | `%` | `i32(params.width)` | 1 | S | Wrap branch, `w >= 1` (guard added) |
| `crates/hal/shaders/convolve_2d.wgsl` | `/` | `2` | 2 | A | literal (kernel centre) |
| `crates/hal/shaders/farneback_poly_expand.wgsl` | `/` | `2` | 1 | A | literal |
| `crates/hal/shaders/farneback_update_flow.wgsl` | `/` | `2` | 1 | A | literal |
| `crates/hal/shaders/fast.wgsl` | `%` | `4u` | 1 | A | literal |
| `crates/hal/shaders/fast.wgsl` | `/` | `16u` | 1 | A | literal |
| `crates/hal/shaders/fast.wgsl` | `/` | `4u` | 2 | A | literal |
| `crates/hal/shaders/fast_bf16.wgsl` | `%` | `2u` | 3 | A | literal |
| `crates/hal/shaders/fast_bf16.wgsl` | `/` | `2u` | 3 | A | literal |
| `crates/hal/shaders/fast_nms.wgsl` | `%` | `4u` | 1 | A | literal |
| `crates/hal/shaders/fast_nms.wgsl` | `/` | `4u` | 2 | A | literal |
| `crates/hal/shaders/fast_nms_bf16.wgsl` | `%` | `2u` | 2 | A | literal |
| `crates/hal/shaders/fast_nms_bf16.wgsl` | `/` | `2u` | 2 | A | literal |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `2 * h - 2` | 1 | S | Reflect101 branch runs only for `h > 1`, so `2h - 2 >= 2` (guard added for `h >= 1`) |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `2 * h` | 1 | S | Reflect branch runs only for `h > 1` |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `2 * w - 2` | 1 | S | Reflect101 branch runs only for `w > 1` |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `2 * w` | 1 | S | Reflect branch runs only for `w > 1` |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `i32(params.height)` | 1 | S | Wrap branch, `h >= 1` (guard added) |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `%` | `i32(params.width)` | 1 | S | Wrap branch, `w >= 1` (guard added) |
| `crates/hal/shaders/gaussian_blur_separable.wgsl` | `/` | `2` | 1 | A | literal (kernel centre) |
| `crates/hal/shaders/hough.wgsl` | `%` | `params.width` | 1 | S | `vote` returns unless `i < width * height`, which is false for width 0 |
| `crates/hal/shaders/hough.wgsl` | `/` | `params.width` | 1 | S | as above |
| `crates/hal/shaders/lbvh_build.wgsl` | `/` | `2` | 3 | A | literal |
| `crates/hal/shaders/matching.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/morphology.wgsl` | `%` | `params.width` | 1 | S | `main` returns when `width == 0` or `height == 0` before any division (guard added when the shader was rewritten) |
| `crates/hal/shaders/morphology.wgsl` | `/` | `2u` | 2 | A | literal (kernel anchor) |
| `crates/hal/shaders/morphology.wgsl` | `/` | `max` | 1 | S | `idx / max(plane_size, 1u)`: at least 1 by construction |
| `crates/hal/shaders/morphology.wgsl` | `/` | `params.width` | 1 | S | as `% params.width` |
| `crates/hal/shaders/remap.wgsl` | `%` | `2 * n - 2` | 1 | S | Reflect101 branch runs only for `n != 1`. With `n >= 1` (guard added), `2n - 2 >= 2`. |
| `crates/hal/shaders/remap.wgsl` | `%` | `2 * n` | 1 | S | Reflect branch, `n >= 1` (guard added), even |
| `crates/hal/shaders/remap.wgsl` | `%` | `i32(len)` | 1 | S | Wrap branch, `n >= 1` (guard added) |
| `crates/hal/shaders/remap.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/resize.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/sift_descriptor.wgsl` | `%` | `8` | 2 | A | literal (signed, not -1) |
| `crates/hal/shaders/sift_extrema.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/sift_orientation.wgsl` | `%` | `36u` | 3 | A | literal |
| `crates/hal/shaders/sobel.wgsl` | `%` | `4u` | 1 | A | literal |
| `crates/hal/shaders/sobel.wgsl` | `/` | `4u` | 3 | A | literal |
| `crates/hal/shaders/stereo_match.wgsl` | `/` | `2u` | 1 | A | literal |
| `crates/hal/shaders/threshold_bf16.wgsl` | `%` | `2u` | 1 | A | literal |
| `crates/hal/shaders/threshold_bf16.wgsl` | `/` | `2u` | 1 | A | literal |
| `crates/hal/shaders/undistort.wgsl` | `%` | `2 * n - 2` | 1 | S | as `remap.wgsl`, the same `map_coord` (guard added) |
| `crates/hal/shaders/undistort.wgsl` | `%` | `2 * n` | 1 | S | as `remap.wgsl` (guard added) |
| `crates/hal/shaders/undistort.wgsl` | `%` | `i32(len)` | 1 | S | as `remap.wgsl` (guard added) |
| `crates/hal/shaders/undistort.wgsl` | `/` | `4u` | 1 | A | literal |
| `crates/hal/shaders/vector_ops.wgsl` | `/` | `2u` | 1 | A | literal |
| `crates/hal/src/gpu_kernels/feature_collection.wgsl` | `%` | `params_2.width` | 1 | S | inside `total_idx < width * height`, which is false for width 0 (guard added: the division used to come before the check) |
| `crates/hal/src/gpu_kernels/feature_collection.wgsl` | `/` | `params_2.width` | 1 | S | as above |
| `crates/hal/src/gpu_kernels/feature_collection_f32.wgsl` | `%` | `params_2.width` | 1 | S | inside `gid < width * height` (moved there by this audit). The host also returns early in `fast::extract_keypoints` when `w * h == 0`. |
| `crates/hal/src/gpu_kernels/feature_collection_f32.wgsl` | `/` | `params_2.width` | 1 | S | as above |
| `crates/hal/src/gpu_kernels/orientation.wgsl` | `%` | `u32(2 * r + 1)` | 1 | S | `2r + 1` is odd for every i32 `r`, even when it wraps, so it is never 0. The operation is unsigned, so INT_MIN / -1 cannot occur. |
| `crates/hal/src/gpu_kernels/orientation.wgsl` | `/` | `u32(2 * r + 1)` | 1 | S | as above |
| `crates/hal/src/gpu_kernels/pointcloud_normals_fast_gpu.wgsl` | `/` | `256u` | 1 | A | literal |
| `crates/hal/src/gpu_kernels/pointcloud_normals_morton.wgsl` | `/` | `2u` | 2 | A | literal |
| `crates/hal/src/gpu_kernels/spatial.rs#build_shader` | `%` | `params.num_buckets` | 1 | B | `spatial_hash_correspondences` is the only writer of `HashParams` and sets `num_buckets = 8192u32`. The shader is inline and has no `dispatch` entry. |
| `crates/hal/src/gpu_kernels/spatial.rs#correspondence_shader` | `%` | `params.num_buckets` | 1 | B | as above (`CorrParams`) |

## Sources naga rejects

**None.** Every source in the corpus parses and validates, and the test fails
if one stops: a shader naga rejects can never become a pipeline, and the
scanner cannot see its divisions. Six used to be rejected and were pinned
here. Each was fixed or deleted:

| source | error | outcome |
| --- | --- | --- |
| `crates/hal/shaders/morphology.wgsl` | `var val = if (...) { 255u } else { 0u };` is not WGSL | fixed, tested |
| `crates/hal/shaders/threshold_f16.wgsl` | `bitcast<f16>(u32)` changes the operand size | fixed, tested |
| `crates/hal/src/gpu_kernels/icp.rs#colored_icp_kernel` | passes a runtime-sized array by value | deleted |
| `crates/hal/src/gpu_kernels/icp.rs#compute_color_gradients_kernel` | same | deleted |
| `crates/hal/src/gpu_kernels/icp.rs#generalized_icp_kernel` | same | deleted |
| `crates/hal/src/gpu_kernels/icp.rs#gicp_accumulate_kernel` | `atomicAdd` on a non-atomic `f32` | deleted |

`morphology.wgsl` is the shader behind `GpuContext::morphology`, which
therefore could never create its pipeline, and no test reached it. The shader
was rewritten to match the CPU backend (replicate border, packed mask read on
the device, CHW planes, dense output words), and
`crates/hal/tests/gpu_morphology.rs` compares the two byte for byte. Its
divisions are in the table above.

`threshold_f16.wgsl` had no caller: `gpu_kernels::threshold` returned
`NotSupported` for f16, and the device never requested `SHADER_F16`. The
device now requests the feature when the adapter offers it, the uniform
carries the threshold and max value as f32 (an exact widening), and
`threshold` runs this shader when `GpuContext::supports_f16` and
`threshold_f16_packed.wgsl` otherwise. Neither divides.

The four ICP strings were public `&'static str` accessors with no host: no
code created a pipeline from them, and `perf_tests` only timed returning the
string. They were deleted, with the `ColoredICPConfig` and
`GeneralizedICPConfig` structs that configured nothing, rather than fixed,
because making them validate would not have made them usable. None computed
its algorithm:

* `colored_icp_kernel` wrote 21 Gauss-Newton terms per point and then
  overwrote the first six with `J * r`. It computed the photometric Jacobian
  and residual and never used them, so it was not colored ICP.
* `compute_color_gradients_kernel` indexed the point cloud as an image
  (`idx % width`, with `width` an unguarded uniform) and ignored the normals.
  Colored ICP needs the colour gradient in each point's tangent plane, fitted
  from its neighbours.
* `generalized_icp_kernel` formed `C_t + C_t` where GICP needs
  `C_t + R C_s Rᵀ`, never inverted it, and emitted products that are not a
  normal-equation system.
* `gicp_accumulate_kernel` needs float atomics, which WGSL does not have. It
  also zeroed its workgroup memory only in global thread 0 and wrote the
  result only from it, so at most one workgroup could ever count.

Colored ICP (`registration_colored_icp`) and point-to-plane ICP keep their
CPU implementations, and point-to-plane keeps its working GPU path
(`icp_correspondences`, `icp_accumulate`, `dense_step`). A GPU colored ICP or
GICP would be new work, not a repair of these strings.

## Signed modulo of negative operands

The border-folding code (`convolve_2d`, `gaussian_blur_separable`, `remap`,
`undistort`) applies `%` to negative coordinates. It assumes truncated
semantics, where the result takes the dividend's sign, and adds the period
when the result is negative. That is WGSL's definition. On this machine (RTX
5070 Ti, Vulkan, wgpu 30.0.1, naga 30.0.1), `a % b` and `a / b` for
`a in -20..=20` and nine divisors including -3 match Rust's
`wrapping_rem` / `wrapping_div` exactly, with the checks on and with them off.
`signed_division_of_negatives_is_truncated` in the audit test pins this.

Commit 9c14f40 dropped a mathematically correct `%` fold from
`pyramid_down.wgsl` after it gave wrong border taps. That matches the
`OpSRem` poison that naga now avoids by always lowering signed `%` itself
(see above). `pyramid_down.wgsl` no longer divides at all.

## Measurement: checks on vs off

`crates/hal/tests/int_div_checks_timing.rs` (ignored, needs a GPU) builds each
kernel twice from the same WGSL, once with the default checks and once with
`int_div_checks: false`. It binds identical buffers and times 10 back-to-back
dispatches per sample with GPU timestamp queries. It alternates which side
runs first and reports the median per-dispatch time. The two variants
really do differ: naga's SPIR-V for `gaussian_blur_separable.wgsl` shrinks
from 2301 to 2237 words without the checks, `canny.wgsl` from 2470 to 2375,
and `remap.wgsl` from 2899 to 2850.

```text
cargo test --release -p cv-hal --test int_div_checks_timing -- --ignored --nocapture
```

RTX 5070 Ti, Vulkan, release build, 81 samples per side. Five back-to-back
runs, taken with the GPU otherwise idle. Median microseconds per dispatch in
run 1, then the delta for each run. Delta is unchecked vs checked, so negative
means faster without the checks:

| kernel | checked | unchecked | run 1 | run 2 | run 3 | run 4 | run 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic: `%` and `/` by a runtime value, 64 iterations | 5534 | 5593 | +1.1% | +1.5% | +0.5% | +0.1% | +0.6% |
| gaussian 4K, k31, horizontal, Reflect101 (`%` per tap) | 2711 | 2725 | +0.5% | +0.1% | +0.0% | +0.0% | +0.6% |
| gaussian 4K, k31, vertical, Wrap (`%` per tap) | 2259 | 2264 | +0.2% | +0.8% | +6.9% | +0.6% | -0.5% |
| gaussian 4K, k31, horizontal, Replicate (control, no `%` runs) | 1036 | 1039 | +0.3% | -0.6% | +28.1% | +6.3% | -0.9% |
| convolve_2d 4K, 7x7, Reflect | 3168 | 3093 | -2.4% | +0.3% | +0.0% | -0.1% | +0.2% |
| convolve_2d 4K, 7x7, Wrap | 4036 | 4033 | -0.1% | -0.0% | +0.0% | -4.3% | +0.1% |
| remap 4K, u8, bicubic, Wrap | 4183 | 4282 | +2.4% | +0.2% | -0.0% | -0.0% | +0.2% |
| canny `sobel` 4K (`%`, `/` by width per pixel) | 195.3 | 195.0 | -0.1% | -0.2% | -0.1% | -0.1% | -0.1% |
| canny `nms` 4K | 176.7 | 177.2 | +0.3% | +0.3% | +0.2% | +0.3% | +0.6% |
| hough `vote` 1080p, 180 angles | 4539 | 4537 | -0.0% | +0.0% | -0.0% | +0.0% | +0.1% |
| resize u8 4K to 1080p (literal `/ 4u` only) | 18.94 | 18.95 | +0.0% | +0.0% | -0.2% | +0.1% | -3.1% |
| pyramid_down 4K (control, no integer division) | 227.7 | 227.5 | -0.1% | +0.4% | -0.1% | +0.2% | -0.0% |
| ORB orientation, r = 15, 20k keypoints | 193.9 | 195.7 | +0.9% | +0.0% | -0.0% | -0.9% | +0.0% |

No kernel shows a gain larger than its own spread. The per-side interquartile
range runs from 0.1% (hough) to 38%, and most kernels sit at 3-15%. The
deltas change sign from run to run. The largest swings are in the
Replicate control, which never executes a runtime division: +28.1% in run 3
and +6.3% in run 4. That swing is the noise floor. The one consistent sign
goes the wrong way. The synthetic kernel does nothing but divide by a
runtime value, yet it is 0.1-1.5% *slower* without the checks in all five
runs, never faster. NVIDIA's compiler turns the guard into one
compare and select next to a division that already costs many instructions.
In the real kernels, memory traffic hides even that.

**Decision:** the checks stay on. Turning them off would put an `unsafe`
contract on `GpuContext::create_compute_pipeline`, which also compiles
caller-supplied WGSL and backs the raw-uniform `dispatch` path, and it would
buy nothing measurable. The audit and the in-shader guards stay. They hold
whatever the setting, and they are what makes flipping it sound if a future
backend (Metal, a mobile Vulkan driver) shows a real cost. Re-run the harness
there first.
