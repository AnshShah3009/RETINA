# Findings — crates/slam + crates/3d audit (in progress)

## DEFECT 1 — tracker selects the device independently of the input tensor's storage (FIXED)

- **File**: `crates/slam/src/tracking.rs`, `process_frame` (device resolved at
  old lines 45-48), new `device_for_u8_tensor` + `cpu_device` helpers.
- **Root cause**: `ResourceGroup::device()` -> `get_device_by_id` ->
  `GpuContext::global()`, and `GLOBAL_CONTEXT` is a process-wide `OnceLock`
  (`crates/hal/src/gpu/mod.rs:13`). Any other code in the process that
  initialises it (`cv_runtime::registry()` does, `crates/runtime/src/device_registry.rs:234`)
  makes it succeed for everyone. So a `CpuStorage` image reached
  `ComputeDevice::Gpu`, and `GpuContext::match_descriptors` rejects `CpuStorage`
  (`crates/hal/src/gpu/compute_context_impl.rs:1733`).
- **Measurement (this machine HAS a GPU — AMD Radeon 890M / NVIDIA)**:
  - `get_device()` before any init: `Cpu(CpuBackend { device_id: DeviceId(0) })`
  - `GpuContext::global()` ok? false
  - after `GpuContext::init_global()`: `get_device()` -> `Gpu(...)`
  - `process_frame` -> `Err("Invalid input: GpuContext requires GpuStorage tensors")`
  - order dependence printed by the regression test pre-fix:
    `before global init: Err(Tracking failed)` /
    `after  global init: Err(Invalid input: GpuContext requires GpuStorage tensors)`
- **Fix**: `device_for_u8_tensor::<S>` selects the device from the input
  tensor's storage: GpuStorage -> group device; CpuStorage -> CPU device;
  anything else -> explicit error. Makes the tracker independent of
  `GLOBAL_CONTEXT` and therefore order-independent.
- **Why CPU rather than `to_gpu`** (the maintainer decision): reasoning is in
  the function's doc comment. Summary: (1) `convert_to_cpu` downloads the frame
  immediately afterwards anyway, so an upload is only reordered, not avoided,
  and is paid every frame; (2) the hot kernel can never use the GPU regardless —
  both descriptor tensors are built with `Tensor::from_vec` on the host (query
  from detector output, map from the cached CPU buffer), so the GPU matcher would
  need *both* uploaded per frame; (3) uploading would change behaviour for every
  caller who passes CPU tensors (extra transfers, allocator pressure, GPU-detector
  rounding) and introduce frame failures unrelated to tracking.
- **Alternative that would be needed for GPU acceleration**: explicit opt-in
  upload of image + both descriptor tensors, fallback to the CPU matcher when
  either upload fails, and a measurement that GPU matching beats CPU matching at
  realistic map sizes. A deliberate perf change, not a default.
- **Regression test**: `crates/slam/tests/tracking_device_selection.rs` (4 tests).
  - **Verified it catches the bug**: `git show HEAD:...tracking.rs > ...` -> 4/4
    FAIL; restored -> 4/4 pass.
  - **Mutants caught**:
    - A: "upload to GPU instead" (CPU branch returns group device) -> 4/4 FAIL
    - B: GPU/CpuStorage downcasts transposed -> 4/4 FAIL
- **Control in every test**: each asserts `get_device()` really is a `Gpu` before
  relying on that, so it cannot pass vacuously on a CPU-only machine. GPU tests
  skip cleanly with a printed note and an early return (no `.expect`).

## Reported but OUT OF TERRITORY — needs the cv-features owner

`Orb::extract` (`crates/features/src/orb.rs:803-818`) uses
`.par_iter().filter_map(|kp| compute_orb_descriptor(...))`, so keypoints whose
patch falls outside the image are dropped from `Descriptors` while `KeyPoints`
keeps them. The two arrays are therefore NOT index-parallel.

**Measured here (320x240 checkerboard, block 16, 500 features):**
`keypoints = 500, descriptors = 437`, and **238 of the 437 descriptors are
attached to the wrong keypoint** (compared via the `Descriptor::keypoint` each
descriptor carries).

`crates/slam/src/tracking.rs` `process_frame` relies on exactly that parallelism:
it builds the query tensor from `frame.descriptors` (length = descriptor count)
but indexes image points as `frame.keypoints.keypoints[m.query_idx]`. So every
descriptor past the first dropped one feeds PnP the pixel of a different feature.
`m.query_idx` is an index into the descriptor array; indexing the keypoint array
with it is only correct if the arrays are parallel.

There is already an ignored test recording this
(`crates/slam/tests/tracking_descriptor_alignment.rs`, `#[ignore]`, "only 1
landmark tracks from a coherent map; cause undiagnosed"). I measured the
mechanism, but the fix belongs in `crates/features`, which I must not edit.
Note `Descriptor::keypoint` carries the keypoint the descriptor was computed
from, so indexing keypoints via the descriptor is correct under either contract.
