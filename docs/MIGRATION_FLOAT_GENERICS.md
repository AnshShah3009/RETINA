# Migration Guide: Float Type Genericity

This document outlines the changes and new capabilities introduced by the full type genericity refactor, which adds support for `half-precision` variants alongside the standard `f32`/`f64`.

## For Library Users

### Using Different Precisions

Instead of hardcoded `f32` types for tensors and math representations, all algorithms in `rust-cv-native` now accept any type that implements the `Float` trait.

```rust
// f32 (default, fast)
let tensor = CpuTensor::<f32>::create(shape, 0.0)?;

// f64 (high precision)
let tensor = CpuTensor::<f64>::create(shape, 0.0)?;

// bf16 (speed with reasonable accuracy for AI/ML inferences)
let tensor = CpuTensor::<half::bf16>::create(shape, half::bf16::ZERO)?;

// f16 (maximum raw compute speed on GPU where supported)
let tensor = CpuTensor::<half::f16>::create(shape, half::f16::ZERO)?;
```

### Type Conversion Between Tensors

If an algorithm runs on a lower precision to save compute cost, or if your HAL uses `bf16` specifically, you can easily convert data back to `f32` using `convert_precision`:

```rust
let f32_result: Tensor<f32, _> = compute_fast::<f32>(&input)?;
let f64_result: Tensor<f64, _> = f32_result.convert_precision::<f64>(DataType::F64)?;
```

### When to Use Which Precision

- **f32 (Single Precision):** The default for almost everything. Use this when you need standard operations. The GPU backend defaults to this.
- **f64 (Double Precision):** Required when running iterative solvers like bundle adjustment, Levenberg-Marquardt, or solving PnP/Stereo geometry where rounding errors compound.
- **f16 (Half Precision):** The best memory footprint, giving exactly 2x capacity on GPU bounds. Prone to overflow issues beyond `65504`. Good for images and normalized outputs.
- **bf16 (Brain Float):** Retains the standard float dynamic exponent range but sacrifices fractional bits. Perfect for gradient accumulation or algorithms that expect large variations in exponent scales.

## Breaking Changes from 0.1.X

- `fast_detect` and `extract_keypoints` no longer assume `f32`.
- `Float` is completely uncoupled from `nalgebra`. To convert between the two, use `rust-cv-native::core::nalgebra_adapters::*`.
- Algorithms operating over `Features` and `Descriptors` are entirely type-driven. Calling `compute_brief` on u8 will no longer compile without explicit cast mappings.
