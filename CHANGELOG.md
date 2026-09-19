# Changelog

All notable changes to `rust-cv-native` will be documented in this file.

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
