# Retina

[![Build and Test](https://github.com/AnshShah3009/rust-cv-native/actions/workflows/build.yml/badge.svg)](https://github.com/AnshShah3009/rust-cv-native/actions/workflows/build.yml)

A comprehensive native Rust computer vision library with full in-house implementations (no external C/C++ dependencies), GPU acceleration using wgpu/WebGPU, and Python bindings with PyO3.

## Features

- **Core** - Basic types, camera models, frame conventions, tensors, robust estimation, error handling
- **Imgproc** - Image processing (filters, morphology, color conversion, thresholding, histogram, template matching)
- **Features** - Feature detection and descriptors (ORB, Harris, FAST, BRIEF, HOG, SIFT, GFTT)
- **Stereo** - Stereo vision, disparity matching, depth estimation, triangulation
- **Calib3d** - Camera calibration, pose estimation, chessboard detection, distortion correction
- **3D** - Point clouds, ICP registration, triangulation, mesh reconstruction (Poisson, BPA, Alpha Shapes)
- **SFM** - Structure from Motion, triangulation, bundle adjustment
- **SLAM** - ISAM2 incremental optimization, keyframe management, Kalman filtering
- **Registration** - ICP, global registration, SE(3) transforms, robust matching
- **Rendering** - Gaussian splatting, mesh processing, visualization
- **Plot** - 2D/3D visualization (cv-plot)
- **Video** - MOG2 background subtraction, Kalman filtering, optical flow, tracking
- **Videoio** - Video capture (FFmpeg-next), platform-specific backends
- **Optimize** - Factor graphs, sparse solvers, ISAM2, nonlinear optimization
- **DNN** - Deep neural network inference (ORT integration)
- **ObjDetect** - Object detection utilities
- **IO** - File I/O for various formats
- **Point-Cloud** - Point cloud processing
- **Scientific** - Scientific computing utilities
- **Runtime** - Async runtime utilities
- **Viewer** - 3D visualization

## Architecture

```
rust-cv-native/
├── crates/core/         # Core types, camera models, frame conventions, tensors
├── crates/hal/          # Hardware abstraction layer (CPU/GPU)
├── crates/math/         # Math primitives and special functions
├── crates/geometry2d/   # 2D geometry primitives
├── crates/imgproc/      # Image processing
├── crates/features/     # Feature detection and matching
├── crates/photo/        # Computational photography
├── crates/video/        # Video processing (MOG2, optical flow)
├── crates/videoio/      # Video I/O (FFmpeg backend)
├── crates/calib3d/      # Camera calibration and pose estimation
├── crates/3d/           # Point clouds, triangulation, mesh reconstruction
├── crates/registration/ # ICP, global registration
├── crates/pointcloud/   # Point cloud processing
├── crates/sfm/          # Structure from Motion
├── crates/slam/         # SLAM with ISAM2
├── crates/optimize/     # Optimization (ISAM2, sparse solvers)
├── crates/rendering/    # 3D rendering (Gaussian splatting)
├── crates/plot/         # Plotting and visualization
├── crates/viewer/       # 3D visualization
├── crates/signal_proc/  # Signal processing
├── crates/scientific/   # Scientific computing
├── crates/dnn/          # Deep neural network inference
├── crates/io/           # File I/O
├── crates/runtime/      # Async runtime and orchestration
├── crates/distributed/  # Cross-process shared-memory / VRAM coordination
├── crates/eval/         # Trajectory, reconstruction and retrieval metrics
├── crates/localization/ # Visual localization pipeline (retrieval → PnP)
├── crates/cli/          # cv-bench command-line evaluation tool
├── crates/python/       # Python bindings (PyO3)
└── crates/examples/     # Usage examples
```

## Visual localization

`cv-localization` turns the primitives into the question users actually ask —
given a query image, where was it taken? A `Database` of landmarks and
database images is queried by retrieval (`cv-features` vocabulary, BoW inverted
index and LSH), ratio-test matched, lifted to 2D-3D correspondences and solved
with PnP + RANSAC, returning a pose with its inlier count, candidate list and
reprojection RMSE. It returns `None` rather than a confident wrong pose.

```bash
cargo run -p cv-localization --example synthetic_localization --features synthetic
```

```rust
let mut db = Database::new(Some(vocabulary));
db.add_image(database_image_from_colmap(&image, &id_map));
db.add_landmark(Landmark { position, descriptors });
db.build();

let localizer = Localizer::new(&db, LocalizerConfig::default());
if let Some(result) = localizer.localize(&query_keypoints, &query_descriptors, &intrinsics) {
    println!("{:?} from {} inliers", result.pose, result.inliers);
}
```

The pipeline is verified on a deterministic synthetic scene (clean queries
localize to machine precision, noise degrades gracefully, non-overlapping
queries return `None`). No real-dataset result is published, because none has
been measured — see [docs/localization.md](docs/localization.md).

## Evaluation and benchmarking

The library ships the measurement layer for localization/SfM/SLAM work, so
results can be produced and re-checked rather than quoted:

- **`cv-eval`** — trajectory metrics (ATE with SE(3)/Sim(3) alignment, RPE,
  camera-centre RMSE, path length), reconstruction metrics (registration rate,
  reprojection RMSE, RMSE/extent, Chamfer distance, F-score) and retrieval
  metrics (recall@k, precision@k, mAP).
- **`cv-io::datasets`** — loaders for EuRoC/ASL, TUM RGB-D, KITTI odometry and
  COLMAP text models, including TUM's timestamp association.
- **`cv-bench`** — a CLI over both.

```bash
# trajectory error against ground truth, with alignment
cargo run -p cv-cli --bin cv-bench -- trajectory \
  --estimate estimate.txt --ground-truth groundtruth.txt --format tum --align sim3

# reconstruction quality from a COLMAP text model
cargo run -p cv-cli --bin cv-bench -- model \
  --images sparse/0/images.txt --points3d sparse/0/points3D.txt

# place-recognition quality
cargo run -p cv-cli --bin cv-bench -- retrieval --predictions ranked.txt --ground-truth relevant.txt --k 10
```

This repository publishes **no benchmark numbers it has not measured, with the
command that produced them**. See [docs/evaluation.md](docs/evaluation.md) for the
metric definitions, dataset formats and the reproducibility rules.

## Installation

### Rust

```bash
cargo build --workspace
```

### Python

```bash
cd python
pip install maturin
maturin develop
```

## Testing

**1,400+ tests** across all crates, including:
- cv-core: geometry, robust estimation, tensor operations, error handling
- cv-features: Harris, FAST, BRIEF, GFTT, HOG, ORB
- cv-stereo: stereo matching, triangulation
- cv-calib3d: camera calibration, pose estimation
- cv-sfm: triangulation, bundle adjustment
- cv-3d: mesh reconstruction (Poisson, BPA, Alpha Shapes), ICP
- cv-optimize: ISAM2, factor graphs
- cv-registration: SE(3) transforms, robust matching
- cv-video: background subtraction, optical flow
- cv-imgproc: filters, morphology, color conversion
- cv-eval: trajectory/reconstruction/retrieval metrics against analytic cases
- cv-io: dataset loaders, including malformed-input handling

GPU tests execute when a graphics adapter is available and skip cleanly when it
is not — on a machine with a GPU they are the only coverage the shaders get, since
hosted CI runners have no adapter.

```bash
cargo test --workspace
```

## Requirements

- Rust 1.70+
- Python 3.10+ (for Python bindings)
- FFmpeg (for video I/O)

## License

MIT
