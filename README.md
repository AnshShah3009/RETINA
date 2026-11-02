# RETINA

[![Build and Test](https://github.com/AnshShah3009/RETINA/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/AnshShah3009/RETINA/actions/workflows/build.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.95+](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org)

**A native Rust computer-vision workspace with CPU and GPU backends, built to
match OpenCV, Open3D, SciPy and Matplotlib — and to prove it.**

RETINA covers image processing, features, calibration, 3-D point clouds and
meshes, registration, SfM/SLAM, localization, signal processing, plotting and
I/O, in about 30 crates. The default build has no C or C++ dependencies. GPU
kernels run through [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12) and are tested
against the CPU implementation, many of them bit for bit.

What sets it apart is how results are checked:

- **Reference parity.** Functions that mirror OpenCV, Open3D, SciPy or Matplotlib
  are tested against values those libraries produced, stored in the tests so CI
  needs no Python. Known differences are written down in
  [docs/PARITY.md](docs/PARITY.md).
- **Every fix is proven.** Each bug fix lands with a test that failed before it;
  [docs/bug-log.md](docs/bug-log.md) records the defect, how it was found and the
  evidence.
- **Measured claims only.** Every number in the docs comes with the command that
  measured it ([docs/evaluation.md](docs/evaluation.md)).
- **Over 3,000 tests**, zero compiler warnings enforced in CI, and
  `#![forbid(unsafe_code)]` in 27 of 30 crates.

> RETINA is pre-1.0. APIs still change between minor versions; the
> [changelog](CHANGELOG.md) lists breaking changes.

## Quick start

Add the crates you need (they are not on crates.io yet, so use git):

```toml
[dependencies]
cv-imgproc = { git = "https://github.com/AnshShah3009/RETINA" }
cv-features = { git = "https://github.com/AnshShah3009/RETINA" }
image = "0.25"
```

Detect and match ORB features between two images:

```rust,no_run
use cv_features::matcher::{MatchType, Matcher};
use cv_features::orb_detect_and_compute;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = image::open("left.png")?.to_luma8();
    let b = image::open("right.png")?.to_luma8();

    let a = cv_imgproc::gaussian_blur(&a, 1.0);
    let b = cv_imgproc::gaussian_blur(&b, 1.0);

    let (_keypoints_a, descriptors_a) = orb_detect_and_compute(&a, 1000);
    let (_keypoints_b, descriptors_b) = orb_detect_and_compute(&b, 1000);

    let matches = Matcher::new(MatchType::BruteForceHamming)
        .with_cross_check()
        .match_descriptors(&descriptors_a, &descriptors_b);
    println!("{} matches", matches.matches.len());
    Ok(())
}
```

Run the examples in the repository:

```bash
cargo run -p cv-examples --example plot_demo          # line, scatter and 3-D plots
cargo run -p cv-localization --example synthetic_localization --features synthetic
```

### Python

Bindings are built with [maturin](https://www.maturin.rs) from the repository
root:

```bash
pip install maturin
maturin develop --release
python -c "import cv_native"
```

## Crates

| Area | Crate | What it provides |
| --- | --- | --- |
| Foundations | `cv-core` | Tensors (planar CHW), camera models, keypoints, errors |
| | `cv-hal` | CPU, GPU (wgpu), MLX and CubeCL backends behind `ComputeContext` |
| | `cv-runtime` | Scheduling, resource groups, device selection |
| | `cv-math`, `cv-geometry2d`, `cv-scientific` | Math primitives, 2-D geometry, scientific utilities |
| Images | `cv-imgproc` | Filters, morphology, colour, thresholds, histograms, warps, contours, template matching |
| | `cv-features` | FAST, ORB, SIFT, AKAZE, BRIEF, HOG, Harris, GFTT, ArUco/ChArUco/AprilTag, matchers, RANSAC/USAC |
| | `cv-photo` | Bilateral filtering, HDR, denoising, inpainting |
| | `cv-video` | Background subtraction, optical flow, Kalman filtering, tracking |
| | `cv-dnn` | ONNX inference via `tract` with OpenCV-style pre-processing |
| Geometry | `cv-calib3d` | Calibration, PnP, epipolar geometry, stereo rectification, chessboards |
| | `cv-3d` | Point clouds, normals, TSDF, meshing (Poisson, ball pivoting, alpha shapes) |
| | `cv-pointcloud` | Lightweight point-cloud algorithms |
| | `cv-registration` | ICP variants, FPFH, global registration |
| | `cv-sfm`, `cv-slam` | Structure from motion, bundle adjustment, SLAM with ISAM2 |
| | `cv-localization` | Retrieval + 2-D/3-D matching + PnP localization |
| | `cv-optimize` | Factor graphs, sparse solvers, ISAM2, Levenberg–Marquardt, `curve_fit` |
| Signals | `cv-signal` | Filters, FFT and resampling, SciPy-style |
| I/O and tools | `cv-io` | PLY, PCD, OBJ, STL, LAS (feature), glTF (feature), datasets (EuRoC, TUM, KITTI, COLMAP) |
| | `cv-videoio` | Video capture (FFmpeg, V4L2) |
| | `cv-plot`, `cv-viewer`, `cv-rendering` | Plotting, a 3-D viewer, Gaussian splatting |
| | `cv-eval`, `cv-cli` | Trajectory, reconstruction and retrieval metrics; the `cv-bench` CLI |
| | `cv-distributed` | Cross-process shared-memory and VRAM coordination |
| | `cv-python` | Python bindings (PyO3) |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) shows how they depend on each other,
how the backends fit together, and where new code belongs.

### Feature flags

| Feature | Crate | Enables |
| --- | --- | --- |
| `half-precision` | `cv-core`, `cv-hal` | `f16` / `bf16` tensors and GPU kernels |
| `cubecl` | `cv-hal` | CubeCL kernels |
| `mlx` | `cv-hal` | Apple MLX backend |
| `las`, `gltf` | `cv-io` | LAS/LAZ and glTF formats |
| `ffmpeg` | `cv-videoio` | FFmpeg capture (needs FFmpeg libraries) |

## GPU support

The GPU backend uses wgpu and runs wherever wgpu does: Vulkan (Linux, Windows),
Metal (macOS) and DX12 (Windows). GPU tests run when an adapter is present and
skip cleanly when it is not; CI runs them on a macOS (Metal) runner. Kernels are
held to the CPU implementation, and integer division in every shader is audited
for zero divisors ([docs/wgsl_int_div_audit.md](docs/wgsl_int_div_audit.md)).

## Visual localization

`cv-localization` answers "given a query image, where was it taken?" A `Database`
of landmarks and images is queried by retrieval (vocabulary, bag-of-words inverted
index, LSH), ratio-test matched, lifted to 2-D/3-D correspondences and solved with
PnP + RANSAC. It returns a pose with its inlier count, candidates and reprojection
RMSE — or `None` rather than a confident wrong pose.

```rust,ignore
let mut db = Database::new(Some(vocabulary));
db.add_image(database_image_from_colmap(&image, &id_map));
db.add_landmark(Landmark { position, descriptors });
db.build();

let localizer = Localizer::new(&db, LocalizerConfig::default());
if let Some(result) = localizer.localize(&query_keypoints, &query_descriptors, &intrinsics) {
    println!("{:?} from {} inliers", result.pose, result.inliers);
}
```

It is verified on a deterministic synthetic scene; no real-dataset localization
result is published because none has been measured. See
[docs/localization.md](docs/localization.md).

## Evaluation and benchmarks

The workspace ships the measurement layer used for its own claims:

- **`cv-eval`** — ATE (SE(3)/Sim(3) alignment), RPE, camera-centre RMSE,
  registration rate, reprojection RMSE, Chamfer distance, F-score, recall@k, mAP.
- **`cv-io::datasets`** — EuRoC/ASL, TUM RGB-D, KITTI odometry, COLMAP text models.
- **`cv-bench`** — a CLI over both:

```bash
cargo run -p cv-cli --bin cv-bench -- trajectory \
  --estimate estimate.txt --ground-truth groundtruth.txt --format tum --align sim3
```

Measured results — timings, mapping accuracy on TUM RGB-D and ETH3D, and the
comparison with visloc-rs — are in [docs/performance.md](docs/performance.md) and
[docs/VS_REFERENCE_COMPARISON.md](docs/VS_REFERENCE_COMPARISON.md), each with the
command that produced it.

## Building and testing

Requirements: Rust **1.95+** (stable). Optional: a GPU, FFmpeg (for
`cv-videoio --features ffmpeg`), Python 3.10+ (bindings, and regenerating
reference values).

```bash
cargo build --workspace
scripts/check.sh          # what CI checks: fmt, no warnings, clippy, tests
scripts/check.sh full     # plus doctests, feature builds, examples, benches
```

Use `cargo test --workspace --no-fail-fast`: plain `cargo test` stops at the first
failing test binary.

## Contributing

Contributions are welcome — bug reports, parity discrepancies, fixes,
optimisations, docs and reviews. Start with [CONTRIBUTING.md](CONTRIBUTING.md): it
covers setup, what a pull request needs, and step-by-step recipes for fixing a
bug, matching a reference library, speeding something up, and adding GPU kernels.
Issue forms ask for what a maintainer needs to reproduce a problem.

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md) — report vulnerabilities privately
- [Governance](GOVERNANCE.md) — maintainers, decisions, releases
- [Documentation index](docs/README.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
