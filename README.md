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
├── core/          # Core types, camera models, frame conventions, robust estimation
├── hal/           # Hardware abstraction layer (CPU/GPU)
├── imgproc/       # Image processing
├── features/      # Feature detection and matching
├── stereo/        # Stereo vision
├── calib3d/       # Camera calibration
├── registration/  # ICP, global registration
├── 3d/            # Point clouds, triangulation, mesh reconstruction
├── sfm/           # Structure from Motion
├── slam/          # SLAM with ISAM2
├── optimize/      # Optimization (ISAM2, rendering/     # Gaussian splatting
├── plot sparse solvers)
├──/          # Visualization
├── video/         # Video processing (MOG2, optical flow)
├── videoio/       # Video I/O (FFmpeg backend)
├── dnn/           # Deep neural networks
├── objdetect/     # Object detection
├── io/            # File I/O
├── point-cloud/   # Point cloud processing
├── scientific/    # Scientific computing
├── runtime/       # Async runtime
├── viewer/        # 3D visualization
├── python/        # Python bindings (PyO3)
└── examples/      # Usage examples
```

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

**300+ tests** across all crates including:
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

```bash
cargo test --workspace
```

## Requirements

- Rust 1.70+
- Python 3.10+ (for Python bindings)
- FFmpeg (for video I/O)

## License

MIT
