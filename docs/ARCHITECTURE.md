# Architecture

How the workspace is put together: which crate owns what, how the CPU and GPU
backends fit behind one API, and the conventions every crate shares. Read this
before adding a crate or a new operation.

## Crate layers

RETINA is one repository and one Cargo workspace; there are no git submodules.
The crates change together (a HAL change touches the kernel, the CPU reference,
the parity tests and the domain wrapper at once), so one workspace keeps every
change atomic and tested as a whole, as wgpu, Bevy and tokio do. The reasoning
is in the [workspace-structure decision record](superpowers/specs/2026-11-03-workspace-structure-design.md).

Arrows point from a crate to what it depends on (normal dependencies; tests may
use anything). A crate depends only on crates in **lower** layers, so the graph
has no cycles and low-level crates stay light. `tests/crate_layers.rs` enforces
this: it fails on any dependency on the same or a higher layer, and on any crate
without a layer.

```text
layer 0  cv-core            types, tensors, point clouds, meshes, camera models, errors
         cv-geometry2d      2-D primitives and predicates
         cv-plot            plotting

layer 1  cv-hal        ->  core                       CPU / GPU / MLX / CubeCL backends
         cv-io         ->  core                       file formats and datasets
         cv-math       ->  core                       math primitives, special functions, k-d tree
         cv-eval       ->  core                       trajectory, reconstruction, retrieval metrics
         cv-photo      ->  core                       computational photography
         cv-rendering  ->  core                       Gaussian splatting, mesh rendering
         cv-videoio    ->  core                       video capture (FFmpeg, V4L2)
         cv-viewer     ->  core                       egui/wgpu 3-D viewer

layer 2  cv-optimize   ->  core, hal, math            factor graphs, sparse solvers, ISAM2, LM
         cv-pointcloud ->  core, math                 point-cloud algorithms (normals, FPFH, filters)
         cv-signal     ->  math                       signal processing (SciPy-style)
         cv-video      ->  core, hal                  background subtraction, optical flow, tracking
         cv-distributed -> hal                        cross-process shared memory / VRAM coordination

layer 3  cv-runtime    ->  core, distributed, hal     scheduler, resource groups, device registry

layer 4  cv-imgproc    ->  core, hal, runtime         image processing, cornerSubPix
         cv-3d         ->  core, hal, math, pointcloud, runtime   meshing, TSDF, GPU point clouds

layer 5  cv-calib3d    ->  core, hal, imgproc, math, runtime      calibration, PnP, stereo, RANSAC/USAC
         cv-features   ->  core, hal, imgproc, runtime            detectors, descriptors, matching
         cv-registration -> 3d, core, hal, pointcloud             ICP, global registration
         cv-dnn        ->  core, imgproc, runtime                 DNN pre/post-processing

layer 6  cv-objdetect  ->  calib3d, core, hal, imgproc            ArUco, ChArUco, AprilTag, HOG, cascades
         cv-cli        ->  core, eval, io                         cv-bench evaluation CLI

layer 7  cv-sfm        ->  calib3d, core, features, hal, optimize, runtime   structure from motion
         cv-slam       ->  calib3d, core, features, hal, optimize, runtime, video   SLAM
         cv-localization -> calib3d, core, eval, features, io     retrieval + PnP localization
         cv-examples   (examples)

layer 8  cv-python     (PyO3 bindings)
         rust-cv-native (root: facade re-exporting every library crate, workspace tests and benches)
```

**Adding a crate:** give it the lowest layer its dependencies allow, add it to
`LAYERS` in `tests/crate_layers.rs` and to the diagram above, inherit the
`[workspace.package]` fields, and re-export it from the root facade
(`src/lib.rs`, checked by `tests/facade_reexports.rs`).

Regenerate the dependency lists with
`cargo metadata --format-version 1 --no-deps` if crates change.

### Workspace conventions

- **One version.** Every crate inherits `version`, `edition`, `rust-version`
  (1.95), `license` (MIT) and `repository` from
  `[workspace.package]` and is released in lockstep.
- **One dependency table.** Crates name external and internal dependencies with
  `{ workspace = true }`; versions live in the root `[workspace.dependencies]`.
- **Supply chain.** `deny.toml` (checked in CI by `cargo deny`) allows permissive
  licences only, crates.io as the only source, and fails on RustSec advisories.
- **Names on crates.io.** `cv-core` and `retina` are taken there, so the first
  release publishes the crates as `retina-*` with a `retina-cv` facade.

## The hardware abstraction layer (`cv-hal`)

Operations that have both CPU and GPU implementations are methods of the
`ComputeContext` trait (`crates/hal/src/context.rs`). Backends implement it:

| Backend | Where | Notes |
| --- | --- | --- |
| CPU | `crates/hal/src/cpu/` | The reference implementation. Rayon-parallel, SIMD via `wide`. |
| GPU (wgpu) | `crates/hal/src/gpu/`, `src/gpu_kernels/`, `shaders/*.wgsl` | Vulkan, Metal, DX12. Kernels are tested against the CPU, many bit-for-bit. |
| MLX | `crates/hal/src/mlx.rs` | Apple MLX; most operations fall back to the CPU. |
| CubeCL | `crates/hal/src/gpu_kernels/cubecl_*.rs`, feature `cubecl` | Kernels written in Rust with CubeCL, run on wgpu. |

`ComputeDevice` (`crates/hal/src/compute.rs`) is an enum over the backends, so
callers can hold "some device" and dispatch through it. `cv-runtime` adds the
scheduling layer on top: `ResourceGroup`s, a device registry, and selection of
which GPU to use (`DeviceSelector`).

**Rules for HAL operations:**

- The CPU path is the specification. A GPU path must return the same result (the
  tests state the tolerance; many are exact) and the same errors.
- Empty input never panics on any backend
  (`crates/hal/tests/gpu_empty_tensors.rs` checks every entry point).
- Multichannel input is either processed plane by plane (where OpenCV filters each
  channel independently) or rejected with `InvalidInput` (where the operation is
  single-channel by definition) — the same on every backend
  (`crates/hal/tests/multichannel_planes.rs`).

## Tensors and images

- `Tensor<T, S>` (`cv-core`) pairs storage `S` (`CpuStorage`, `GpuStorage`) with a
  `TensorShape` and `DataType`.
- **Layout is planar CHW**: `TensorShape::new(channels, height, width)`, element
  `(c, y, x)` at index `c*H*W + y*W + x`. `Tensor::from_image_rgb` splits RGB into
  planes. Every HAL op follows this.
- Element types: `f32` everywhere; `f64`; `u8` for images; `f16`/`bf16` behind the
  `half-precision` feature (GPU kernels pack two 16-bit values per 32-bit word).
- Many `cv-imgproc` and `cv-features` functions take `image::GrayImage` /
  `RgbImage` directly; `_ctx` variants take a `ComputeDevice` or `ResourceGroup` to
  choose the backend.

## Feature flags

| Crate | Feature | What it enables |
| --- | --- | --- |
| `cv-core`, `cv-hal` | `full-precision` (default), `half-precision`, `all-precision` | Which float types tensors and kernels support |
| `cv-hal` | `cubecl` | CubeCL kernels and their tests |
| `cv-hal` | `mlx` | Apple MLX backend (macOS) |
| `cv-hal` | `cpu`, `simd` | Backend selection in CI builds |
| `cv-io` | `las`, `gltf` | LAS/LAZ point clouds, glTF meshes |
| `cv-videoio` | `ffmpeg`, `v4l2`, `native` | Video capture backends |
| `cv-features`, `cv-calib3d`, `cv-objdetect` | `gpu` | GPU-accelerated paths |
| `cv-localization` | `synthetic` | The synthetic-scene example |

## Testing layers

| Layer | Where | What it guards |
| --- | --- | --- |
| Unit tests | `#[cfg(test)]` in each module | Local invariants |
| Integration tests | `crates/*/tests/` | Public behaviour, regressions (one file per defect class is common) |
| Reference parity | `*_opencv_parity.rs`, `*_scipy.rs`, `parity/` | Agreement with OpenCV / Open3D / SciPy, reference values stored in the tests |
| CPU/GPU parity | `tests/gpu_parity_tests.rs`, `crates/hal/tests/gpu_*.rs` | GPU kernels match the CPU; shader bookkeeping |
| Doctests | doc comments | Examples in the docs compile and run |
| Examples | `crates/*/examples/`, `tests/example_smoke.rs` | Examples build and run to completion |
| Benchmarks | `crates/*/benches/`, `benches/` | Performance; must keep building |

GPU tests run when an adapter is present and print a skip message otherwise. CI's
macOS runner has a Metal GPU; the Linux runner has none.

## Where things go

| You are adding... | Put it in |
| --- | --- |
| A tensor type, error variant, camera model, geometry container (point cloud, mesh) | `cv-core` |
| An op with CPU and GPU implementations | `cv-hal` (`ComputeContext`), then a friendly wrapper in the domain crate |
| An image filter or transform | `cv-imgproc` |
| A detector, descriptor or matcher | `cv-features` |
| A marker, board or object detector (ArUco, HOG, cascades) | `cv-objdetect` |
| A robust estimator (RANSAC, USAC) | `cv-calib3d` |
| Calibration, PnP, epipolar geometry, stereo | `cv-calib3d` |
| Point-cloud or mesh processing | `cv-3d` (or `cv-pointcloud` for lightweight, dependency-free algorithms) |
| Registration (ICP, FPFH, global) | `cv-registration` |
| A file format or dataset loader | `cv-io` |
| An optimiser or factor | `cv-optimize` |
| A metric | `cv-eval` |
| A WGSL shader | `crates/hal/shaders/` (see the GPU recipe in CONTRIBUTING.md) |

## Further reading

- [PARITY.md](PARITY.md) — what matches which reference, and the known differences
- [bug-log.md](bug-log.md) — every defect fixed and how it was found
- [performance.md](performance.md) — measured timings and the sweeps behind defaults
- [evaluation.md](evaluation.md) — metrics, datasets and reproducibility rules
- [wgsl_int_div_audit.md](wgsl_int_div_audit.md) — the shader integer-division audit
