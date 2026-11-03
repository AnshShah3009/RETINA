# Changelog

All notable changes to RETINA are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break
APIs; breaking changes are marked **Breaking**).

Contributors: add a line under **Unreleased** for user-visible changes. Defects
also get a row in [docs/bug-log.md](docs/bug-log.md).

## [Unreleased]

### Added

- Open-source project files: the MIT license, third-party notices, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `GOVERNANCE.md`, issue forms, a pull
  request template, CODEOWNERS, Dependabot, `docs/ARCHITECTURE.md`, a docs index,
  and `scripts/check.sh` to run CI's checks locally. README examples are doctested.
- `cv-features`: OpenCV's `detectMarkers` pipeline in `ArucoDetector`
  (`DetectorParameters`, corner refinement, rejected candidates); AprilTag
  16h5/25h9/36h10/36h11 dictionaries; the four `CharucoParameters` options.
- `cv-imgproc`: `find_contours_with` (all retrieval modes, SIMPLE approximation,
  hierarchy); `cv-calib3d`: `corner_subpix_with` (window, zero zone, criteria).
- `cv-hal`: f16/bf16 GPU paths for threshold, FAST, bilateral, colour conversion
  and resize; GPU morphology; `GaussianPyramidPlan` with CubeCL graph replay.
- Benchmarks: `crates/hal/benches/gpu_ops.rs`, `crates/features/benches/features_cpu.rs`.
- CI runs on `research/cubecl`, tests CubeCL and f16/bf16 kernels on the macOS
  Metal runner, triggers on shader and fixture changes, and fails on any warning.

### Changed

- **Breaking:** workspace restructure
  ([decision record](docs/superpowers/specs/2026-11-03-workspace-structure-design.md)):
  - New crate `cv-objdetect` holds ArUco, ChArUco, AprilTag (`markers`), HOG and
    Haar cascades, moved from `cv-features` (OpenCV's `objdetect` module).
    `cv-features` no longer depends on `cv-calib3d`.
  - RANSAC and USAC move from `cv_features::{ransac, usac}` to
    `cv_calib3d::{ransac, usac}`.
  - `corner_subpix`, `corner_subpix_with` and `SubpixCriteria` move to
    `cv-imgproc`; `cv-calib3d` still re-exports them.
  - `TriangleMesh` moves to `cv_core::mesh` (re-exported by `cv_3d::mesh` and
    `cv_io::mesh`); `cv-io` no longer depends on `cv-3d` or wgpu.
  - `cv-scientific` is removed; use `cv-math`, `cv-geometry2d`, `cv-signal` and
    `cv-pointcloud` directly.
- RETINA is licensed under MIT (`license = "MIT"` in every crate). OpenCV- and
  AprilTag-derived content keeps its notices in `THIRD_PARTY_NOTICES.md`.
- One implementation each for statistical outlier removal, voxel downsampling
  and kNN normal estimation (f64 kernels in `cv-pointcloud`; `cv-3d` and the
  `cv-pointcloud` `PointCloud` functions delegate), and for the gray bilateral
  filter (`cv_imgproc::bilateral_filter_gray`; `cv_photo::bilateral_filter`
  delegates, and `cv-photo` now depends on `cv-imgproc`).
- `cv_3d::gpu::voxel_downsample` uses the global voxel grid and returns sorted,
  deterministic output (it anchored the grid at the first point and returned
  hash-map order).
- Every crate shares the workspace version (0.2.0), edition, MSRV 1.95, licence
  and repository through `[workspace.package]`. `tests/crate_layers.rs` enforces
  the crate layer order, and CI runs `cargo deny` (advisories, licences, sources).
- **Breaking:** `approx_poly_dp` returns `Result` and rejects negative, NaN or
  huge epsilon, as OpenCV does.
- **Breaking:** `ArucoDetector`'s tuning fields are replaced by `params:
  DetectorParameters`; `get_marker` returns OpenCV's bit grid (1 = white);
  `markers::ArucoDictionary` is now the `aruco` dictionary enum.
- **Breaking:** `CharucoParameters::min_markers` defaults to 2 (OpenCV's default).
- **Breaking:** multichannel tensors are planar CHW in `resize` (CPU and GPU), and
  GPU dense optical flow returns planar `(u, v)` planes; the CubeCL Lucas-Kanade
  flow returns `[2, h, w]` with OpenCV's sign.
- **Breaking:** single-channel-only operations return `InvalidInput` for
  multichannel input on every backend; resizing an empty source to a non-empty
  size returns `InvalidInput`.
- GPU Farneback and TV-L1 are ports of OpenCV's algorithms (`Tvl1Config` gained
  `num_scales`, `scale_step`, `median_filtering`; defaults are OpenCV's).
- Dependencies: wgpu 30, CubeCL 0.11, egui 0.37, nalgebra 0.35, rand 0.10,
  rstar 0.14, las 0.11, pollster 1.0, criterion 0.8.

### Fixed

- `cv-3d` statistical outlier removal and kNN normal estimation panicked
  ("comparison function does not correctly implement a total order") on a NaN
  point; non-finite points are now left out of neighbourhoods and statistics
  (outlier removal drops them, normals give them (0, 0, 1)).
- `cv_pointcloud::voxel_down_sample` treated an infinite voxel size as valid and
  merged every point into one voxel; it now returns the cloud unchanged, as for
  other invalid sizes.

- Hundreds of defects (420 `fix` commits since 0.2.0), each with a test that failed before the fix; see
  [docs/bug-log.md](docs/bug-log.md). Highlights: SIFT scale space, ChArUco
  corners one square off, `solve_pnp_dlt` on planar targets, stereo rectification
  with fx != fy, chessboard corner order, FPFH vs Open3D, ball pivoting, polygon
  buffer, Kalman prediction, glTF winding, GPU LBVH race on Metal, GPU Lanczos
  mapping, GPU bilateral NaN at zero sigma, empty-input panics on CPU and GPU.

### Performance

- TV-L1 optical flow −47% (fused primal/dual dispatch); ORB detect+compute 4x and
  FAST suppression 7x at 1080p; brute-force matching 3–4x; k-NN matching 3–6x;
  CubeCL conv2d autotuning −25..−44%; GPU uploads −27..−49% (wgpu 30).

## [0.2.0] - 2026-03-08

### Added

- **Full Type Genericity Integration**: `rust-cv-native` now fully supports parameterizing memory and compute operations over abstract `Float` types.
- Core modules, HAL cpu/gpu backends, image processing, and feature detection natively execute logic under varying precision formats depending on requirements rather than being hardcoded to single precision.
- **Half Precision Hardware Acceleration**: Added `bf16` and `f16` capabilities via parsing of the `half` crate wrapper directly inside `cv_core`. Enabling this is controlled through the `half-precision` feature in `Cargo.toml`.

### Changed

- Converted CPU fallback kernels and GPU shader dispatches to explicitly use `TensorCast` when jumping bounds between image data (`u8`) and representation logic (`Float`).
- `cv-features`, `cv-stereo`, and `cv-calib3d` fully modernized to drop nalgebra bindings from internal definitions, relying on `nalgebra_adapters` wrapper implementation to prevent bound collisions with `f16`/`bf16`.
- Bytemuck `Pod` conversions generalized explicitly across GPU descriptors for 16-bit payload mappings.

### Deprecated

- Using primitive types as algorithm inputs defaults without traits or context generics.

### Removed

- Removed unused and redundant CPU inference tests and unoptimized trait implementations from the `cv-3d` pipeline.

## [0.1.0] - 2026-02-15

- Initial Native Computer Vision library release in rust, implementing HAL bounds, core image formats, and ORB/AKAZE detection systems natively across GPU / CPU.
