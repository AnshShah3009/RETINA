# Full Type Genericity Implementation Plan: bf16/fp16/fp32/fp64 Support

> **Status**: Design Phase
> **Target Duration**: 14-24 days (distributed workload)
> **Complexity**: HIGH - Architectural refactoring
> **For**: Agent-driven implementation

---

## Executive Summary

This plan converts `rust-cv-native` from hardcoded f32/f64 math to fully generic float types supporting:
- **bf16** (Brain Float 16)
- **fp16** (IEEE Half Precision)
- **fp32** (Single Precision) - current default
- **fp64** (Double Precision) - current high-precision

**Key Constraint**: Nalgebra doesn't support bf16/fp16, so we'll create a generic math abstraction layer that wraps nalgebra for f32/f64 and provides custom implementations for bf16/fp16.

---

## Phase 1: Core Trait System Design (Days 1-2)

### Goal
Create a unified Float trait that abstracts all floating-point operations across all precision levels.

### 1.1 Create Core Float Trait Module

**File**: `core/src/float.rs` (NEW)

```rust
/// Unified floating-point trait supporting bf16, fp16, fp32, fp64
pub trait Float:
    Copy + Clone + Debug + Default +
    PartialOrd + PartialEq +
    Add<Output = Self> + Sub<Output = Self> +
    Mul<Output = Self> + Div<Output = Self> +
    Neg<Output = Self> +
    Send + Sync + 'static
{
    // Conversion
    fn from_f32(val: f32) -> Self;
    fn from_f64(val: f64) -> Self;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;

    // Constants
    fn zero() -> Self;
    fn one() -> Self;
    fn pi() -> Self;
    fn two_pi() -> Self;
    fn epsilon() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;

    // Math operations
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn rsqrt(self) -> Self;
    fn recip(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn powf(self, exp: Self) -> Self;
    fn powi(self, exp: i32) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;

    // Trigonometry
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;

    // Predicates
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_normal(self) -> bool;
    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;

    // Comparison with tolerance
    fn approx_eq(self, other: Self, tolerance: Self) -> bool {
        (self - other).abs() <= tolerance
    }
}
```

### 1.2 Implement Float Trait for Builtin Types

**File**: `core/src/float.rs` (continuation)

Create implementations for:
- `impl Float for f32` (delegate to std methods)
- `impl Float for f64` (delegate to std methods)
- `impl Float for half::bf16` (wrap half crate)
- `impl Float for half::f16` (wrap half crate)

**Key Implementation**:
```rust
#[cfg(feature = "half")]
use half::{bf16, f16};

impl Float for f32 {
    #[inline]
    fn from_f32(val: f32) -> Self { val }
    #[inline]
    fn to_f32(self) -> f32 { self }
    #[inline]
    fn sqrt(self) -> Self { f32::sqrt(self) }
    // ... delegate all methods to f32
}

impl Float for f64 {
    #[inline]
    fn from_f32(val: f32) -> Self { val as f64 }
    #[inline]
    fn to_f32(self) -> f32 { self as f32 }
    #[inline]
    fn sqrt(self) -> Self { f64::sqrt(self) }
    // ... delegate all methods to f64
}

#[cfg(feature = "half")]
impl Float for f16 {
    #[inline]
    fn from_f32(val: f32) -> Self { f16::from_f32(val) }
    #[inline]
    fn to_f32(self) -> f32 { self.to_f32() }
    #[inline]
    fn sqrt(self) -> Self {
        let expanded = self.to_f32();
        f16::from_f32(expanded.sqrt())
    }
    // ... implement all methods via f32 conversion
}

#[cfg(feature = "half")]
impl Float for bf16 {
    #[inline]
    fn from_f32(val: f32) -> Self { bf16::from_f32(val) }
    #[inline]
    fn to_f32(self) -> f32 { self.to_f32() }
    #[inline]
    fn sqrt(self) -> Self {
        let expanded = self.to_f32();
        bf16::from_f32(expanded.sqrt())
    }
    // ... implement all methods via f32 conversion
}
```

### 1.3 Update Core Cargo.toml

**File**: `core/Cargo.toml`

Add:
```toml
[dependencies]
half = { version = "2.4", features = ["num-traits", "serde"], optional = true }

[features]
default = ["full-precision"]
full-precision = []
half-precision = ["half"]
all-precision = ["half"]
```

---

## Phase 2: Math Trait Abstractions (Days 2-3)

### Goal
Create traits for vector/matrix operations that work with any Float type.

### 2.1 Create Generic Vector Trait

**File**: `core/src/vector.rs` (NEW)

```rust
use crate::float::Float;

/// Generic N-dimensional vector trait
pub trait Vector<T: Float> {
    fn len(&self) -> usize;
    fn dot(&self, other: &Self) -> T;
    fn norm(&self) -> T {
        self.dot(self).sqrt()
    }
    fn normalize(&self) -> Self;
    fn cross_3d(&self, other: &Self) -> Self; // For 3D vectors
}

/// 3D Point trait
pub trait Point3D<T: Float> {
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
    fn distance_to(&self, other: &Self) -> T;
    fn transform(&self, matrix: &Matrix4<T>) -> Self;
}

/// 3x3 Matrix trait
pub trait Matrix3x3<T: Float> {
    fn identity() -> Self;
    fn mul_vector(&self, v: &[T; 3]) -> [T; 3];
    fn mul_matrix(&self, other: &Self) -> Self;
    fn determinant(&self) -> T;
    fn inverse(&self) -> Option<Self>;
}

/// 4x4 Matrix trait
pub trait Matrix4x4<T: Float> {
    fn identity() -> Self;
    fn mul_vector(&self, v: &[T; 4]) -> [T; 4];
    fn mul_matrix(&self, other: &Self) -> Self;
    fn inverse(&self) -> Option<Self>;
}
```

### 2.2 Wrapper Structs for Nalgebra Interop

**File**: `core/src/nalgebra_wrapper.rs` (NEW)

Create thin wrapper structs that allow nalgebra types to implement our generic traits:

```rust
use crate::float::Float;
use nalgebra::{Point3 as NA_Point3, Vector3 as NA_Vector3, Matrix3 as NA_Matrix3};

/// Wraps nalgebra types, converts Float <-> f32/f64 as needed
pub struct Vector3Wrapper<T: Float> {
    data: [T; 3],
}

pub struct Point3Wrapper<T: Float> {
    data: [T; 3],
}

pub struct Matrix3Wrapper<T: Float> {
    data: [[T; 3]; 3],
}

pub struct Matrix4Wrapper<T: Float> {
    data: [[T; 4]; 4],
}

// Implementations delegate to nalgebra for f32/f64, custom math for bf16/f16
```

---

## Phase 3: Tensor System Enhancement (Days 3-4)

### Goal
Make Tensor system fully generic over Float types.

### 3.1 Update Tensor Trait

**File**: `core/src/tensor.rs` (MODIFY)

```rust
use crate::float::Float;

// Add Float bound requirement
pub trait StorageFactory<T: 'static>: Storage<T> + Sized {
    fn create(shape: TensorShape, default: T) -> Result<Self>;
    fn create_zeros(shape: TensorShape) -> Result<Self>;
}

// For float tensors specifically
pub trait StorageFactory<T: Float>: Storage<T> + Sized {
    fn create(shape: TensorShape, default: T) -> Result<Self>;
    fn create_zeros(shape: TensorShape) -> Result<Self>;
}

// Update DataType enum to include bf16/f16
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    U8,
    U16,
    U32,
    I32,
    F16,    // NEW
    BF16,   // NEW
    F32,
    F64,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::F16 => 2,      // NEW
            DataType::BF16 => 2,     // NEW
            DataType::F64 => 8,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F16 | DataType::BF16 | DataType::F32 | DataType::F64)
    }
}
```

### 3.2 Add Type Information Methods

**File**: `core/src/tensor.rs` (MODIFY)

```rust
impl<T: Clone + Copy + fmt::Debug + 'static> CpuTensor<T> {
    /// Convert tensor to different float precision
    pub fn convert_precision<U: Clone + Copy + Float>(
        &self,
        target_dtype: DataType,
    ) -> Result<CpuTensor<U>>
    where
        T: Float,
    {
        // Convert T -> U
    }
}
```

---

## Phase 4: HAL Module Generification (Days 4-8)

### Goal
Make HAL CPU and GPU backends generic over Float types.

### 4.1 HAL CPU Backend

**Files to modify**:
- `hal/src/cpu/mod.rs` - Main CPU compute
- `hal/src/cpu/image_processing.rs` - Image ops
- `hal/src/cpu/features.rs` - Feature detection

**Strategy**:
1. Replace all `fn(...Storage<f32>...)` with `fn<T: Float>(...Storage<T>...)`
2. Use our Vector/Matrix wrapper traits instead of nalgebra directly
3. For each function, convert parameters from T -> f32/f64 if needed

**Example Transformation**:

```rust
// BEFORE (hardcoded f32)
fn convolve_2d<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
    input: &Tensor<f32, S>,
    kernel: &[f32],
) -> Result<Tensor<f32, CpuStorage<f32>>>

// AFTER (generic)
fn convolve_2d<T: Float, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
    input: &Tensor<T, S>,
    kernel: &[T],
) -> Result<Tensor<T, CpuStorage<T>>>
where
    T::Float
{
    // Implementation works with T directly
}
```

### 4.2 Specific HAL Functions to Generify

**List of 23 f32-specific functions**:
1. `gaussian_kernel_1d` - kernel generation
2. `convolve_2d` - convolution
3. `match_template` - template matching
4. `stereo_match` - stereo matching
5. `triangulate_points` - triangulation
6. `nms` - non-maximum suppression
7. `nms_boxes` - box NMS
8. `nms_rotated_boxes` - rotated box NMS
9. `pointcloud_transform` - point cloud transform
10. `pointcloud_normals` - normal estimation
11. `tsdf_integrate` - TSDF integration
12. `tsdf_raycast` - TSDF raycasting
13. `tsdf_extract_mesh` - mesh extraction
14. `optical_flow_lk` - Lucas-Kanade flow
15. `dense_icp_step` - ICP iteration
16. `sift_extrema` - SIFT detection
17. `compute_sift_descriptors` - SIFT descriptors
18. `icp_correspondences` - ICP matching
19. `icp_accumulate` - ICP accumulation
20. `akaze_diffusion` - AKAZE diffusion
21. `resize_image` - image resizing
22. `pyramid_down` - pyramid downsampling
23. `warp_affine` - affine warping

**Work Breakdown**:
- Batch 1 (Days 4-5): Image processing (1-5, 21-23)
- Batch 2 (Days 5-6): Feature detection (6-20)
- Batch 3 (Days 6-7): 3D operations (9-13)
- Batch 4 (Days 7-8): Remaining + validation

### 4.3 HAL GPU Backend

**Files to modify**:
- `hal/src/gpu.rs` - GPU context
- `hal/src/gpu_kernels/mod.rs` - Kernel dispatch
- `hal/src/storage.rs` - GPU storage

**Shader Variants Strategy**:
- Create shader variants for each precision level
- Use conditional compilation in shader code
- Example: `#if PRECISION == 16` blocks

**New Shaders to Create**:
- `*_f16.wgsl` - fp16 variants
- `*_bf16.wgsl` - bf16 variants
- Existing `*.wgsl` - f32/f64 (rename to `*_f32.wgsl`)

**GPU Function Updates**:
```rust
// Before: hardcoded f32 only
pub fn convolve_2d_gpu(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    kernel: &[f32],
) -> Result<Tensor<f32, GpuStorage<f32>>>

// After: generic with shader selection
pub fn convolve_2d_gpu<T: Float>(
    ctx: &GpuContext,
    input: &Tensor<T, GpuStorage<T>>,
    kernel: &[T],
) -> Result<Tensor<T, GpuStorage<T>>>
where
    T: Float
{
    let shader_name = match T::precision() {
        Precision::F16 => "convolve_2d_f16.wgsl",
        Precision::BF16 => "convolve_2d_bf16.wgsl",
        Precision::F32 => "convolve_2d_f32.wgsl",
        Precision::F64 => "convolve_2d_f64.wgsl",
    };
    // ... rest of implementation
}
```

---

## Phase 5: Module-by-Module Generification (Days 8-18)

### 5.1 Calib3d Module (Days 8-10)

**Files**:
- `calib3d/src/calibration.rs`
- `calib3d/src/project.rs`
- `calib3d/src/distortion.rs`
- `calib3d/src/essential_fundamental.rs`
- `calib3d/src/triangulation.rs`
- `calib3d/src/pnp.rs`

**Changes**:
- All functions accept `<T: Float>` parameter
- Replace `f64` calibration results with generic `T`
- Update return types: `CameraIntrinsics<T>`, `Distortion<T>`, etc.

**Priority Order**:
1. Core geometric functions (project, triangulate)
2. Distortion models
3. Pose estimation
4. Calibration workflows

### 5.2 Features Module (Days 10-11)

**Files**:
- `features/src/brief.rs`
- `features/src/orb.rs`
- `features/src/sift.rs`
- `features/src/akaze.rs`

**Changes**:
- Descriptor extraction generic over T
- Feature detection generic over T
- Matching algorithms generic (if they use float computations)

### 5.3 Stereo Module (Days 11-12)

**Files**:
- `stereo/src/rectification.rs`
- `stereo/src/matching.rs`
- `stereo/src/disparity.rs`

**Changes**:
- Stereo matching generic
- Disparity computation generic
- Rectification generic

### 5.4 3D Module (Days 12-14)

**Files**:
- `3d/src/tsdf/mod.rs`
- `3d/src/mesh/mod.rs`
- `3d/src/odometry/mod.rs`
- `3d/src/raycasting/mod.rs`
- `3d/src/reconstruction.rs`

**Changes**:
- TSDF integration generic
- Mesh operations generic
- Odometry algorithms generic
- Raycasting generic

### 5.5 Optimize Module (Days 14-15)

**Files**:
- `optimize/src/gpu_solver.rs`
- `optimize/src/bundle_adjustment.rs`

**Changes**:
- Sparse matrix operations generic
- Solver generic (CG, LSQR, etc.)
- Optimization algorithms generic

### 5.6 Scientific Module (Days 15-16)

**Files**:
- `scientific/src/*.rs` - Various scientific functions

**Changes**:
- Filtering operations generic
- Statistical operations generic
- Numerical methods generic

### 5.7 Additional Modules (Days 16-18)

**Files**:
- `point-cloud/src/*.rs`
- `registration/src/*.rs`
- `rendering/src/*.rs`
- `video/src/*.rs`
- `videoio/src/*.rs`
- `slam/src/*.rs`
- `sfm/src/*.rs`

---

## Phase 6: Nalgebra Integration Layer (Days 18-20)

### Goal
Ensure nalgebra types work seamlessly with our Float trait.

### 6.1 Create Nalgebra Adapters

**File**: `core/src/nalgebra_adapters.rs` (NEW)

```rust
use crate::float::Float;
use nalgebra::{Point3, Vector3, Matrix3, Matrix4};

/// Allow nalgebra types to work with our Float trait
/// For f32/f64: direct delegation
/// For bf16/f16: convert via f32

pub fn na_point3_to_array<T: Float>(p: Point3<f32>) -> [T; 3] {
    [T::from_f32(p.x), T::from_f32(p.y), T::from_f32(p.z)]
}

pub fn array_to_na_point3(arr: [f32; 3]) -> Point3<f32> {
    Point3::new(arr[0], arr[1], arr[2])
}

// Similar for Vector3, Matrix3, Matrix4
```

### 6.2 Update All Nalgebra Dependencies

**Pattern**:
- Keep nalgebra for f32/f64 operations
- Use conversion functions for bf16/f16
- Add feature flags for precision variants

---

## Phase 7: Testing & Validation (Days 20-23)

### 7.1 Unit Tests

**Strategy**:
- Add generic tests that run on all Float types
- Use parameterized tests

**Files to Create**:
- `core/tests/float_trait_tests.rs`
- `hal/tests/cpu_precision_tests.rs`
- `hal/tests/gpu_precision_tests.rs`
- `calib3d/tests/precision_tests.rs`

**Example Test**:
```rust
#[parameterized(type: f32, f64, f16, bf16)]
fn test_matrix_multiply(type: T) {
    let a = create_matrix::<T>(3, 3);
    let b = create_matrix::<T>(3, 3);
    let result = multiply::<T>(a, b);
    assert!(result.is_valid());
}
```

### 7.2 Accuracy Benchmarks

**Comparisons**:
- f32 vs f16 accuracy on FAST detection
- f32 vs bf16 accuracy on calibration
- f32 vs f64 for high-precision algorithms

**Files to Create**:
- `benches/precision_accuracy.rs`
- `tests/precision_regression_tests.rs`

### 7.3 Integration Tests

**Coverage**:
1. End-to-end FAST detection (all types)
2. End-to-end calibration workflow (all types)
3. End-to-end 3D reconstruction (all types)
4. GPU kernels with all precisions

### 7.4 Regression Testing

**Multi-GPU Tests**:
- Update `tests/multi_gpu_tests.rs` to test all precision levels
- Verify GPU/CPU parity for each type

---

## Phase 8: Documentation & Finalization (Days 23-24)

### 8.1 Type-Generic Documentation

**Updates**:
- Update all function docstrings to show generic signatures
- Add precision guidance: when to use bf16 vs f32
- Add conversion examples

### 8.2 Migration Guide

**File**: `docs/MIGRATION_FLOAT_GENERICS.md`

```markdown
# Migration Guide: Float Type Genericity

## For Library Users

### Using Different Precisions

```rust
// f32 (default, fast)
let tensor = CpuTensor::<f32>::create(...)?;

// f64 (high precision)
let tensor = CpuTensor::<f64>::create(...)?;

// bf16 (speed with reasonable accuracy)
let tensor = CpuTensor::<bf16>::create(...)?;

// f16 (maximum speed)
let tensor = CpuTensor::<f16>::create(...)?;
```

### Conversion Between Types

```rust
let f32_result = compute_fast::<f32>(input)?;
let f64_result = f32_result.convert_precision::<f64>(DataType::F64)?;
```
```

### 8.3 API Documentation

**Additions**:
- Floating-point traits documentation
- Precision trade-off guide
- GPU shader variants documentation

### 8.4 Example Programs

**Create**:
- `examples/multi_precision_calibration.rs`
- `examples/fast_detection_all_types.rs`
- `examples/precision_accuracy_comparison.rs`

---

## Phase 9: Feature Flags & Release (Days 23-24)

### 9.1 Update Cargo.toml

**Root `Cargo.toml`**:
```toml
[workspace]

[workspace.dependencies]
# ... existing deps

[features]
default = ["full-precision"]
full-precision = []
half-precision = ["core/half-precision"]
all-precisions = ["core/all-precisions"]
```

**Per-crate features**:
```toml
[features]
default = []
f16-support = ["half"]
bf16-support = ["half"]
full-support = ["half"]
```

### 9.2 Version Bump

- Bump to `0.2.0` (minor version - new feature)
- Document breaking changes (if any)
- Update CHANGELOG.md

### 9.3 Final Validation Checklist

- [ ] All 174 files compile
- [ ] All 500+ functions have generics
- [ ] All precision tests pass
- [ ] GPU shaders compile for all types
- [ ] Backward compatibility maintained
- [ ] Documentation complete
- [ ] Examples run successfully
- [ ] No regressions in existing tests

---

## Dependencies & Tools

### New Dependencies to Add

```toml
# core/Cargo.toml
half = { version = "2.4", features = ["num-traits", "serde"] }

# Optional deps
num-traits = "0.2"
```

### Build/Test Tools

- `cargo test --all-features` - Run all tests
- `cargo check --all` - Verify compilation
- `cargo bench --bench precision_accuracy` - Run benchmarks

---

## Risk Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Maintain backward compatibility by keeping f32/f64 as defaults

### Risk 2: GPU Shader Complexity
**Mitigation**: Test shader compilation for each type during CI

### Risk 3: Performance Regressions
**Mitigation**: Create benchmarks, compare baseline vs. new code

### Risk 4: Nalgebra Integration Issues
**Mitigation**: Create comprehensive adapter layer, test thoroughly

### Risk 5: Lost Precision in Conversions
**Mitigation**: Document precision loss, provide tolerance parameters

---

## Success Criteria

- ✅ All 500+ functions accept generic `T: Float` parameter
- ✅ bf16, f16, f32, f64 all supported and tested
- ✅ GPU shaders support all precision levels
- ✅ No performance regressions on f32 baseline
- ✅ All tests pass (unit, integration, regression)
- ✅ Documentation complete with examples
- ✅ FAST detection, calibration, 3D ops tested in all types
- ✅ Version 0.2.0 released with changelog

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1. Trait System | Days 1-2 | Float trait, impls for bf16/f16/f32/f64 |
| 2. Math Abstractions | Days 2-3 | Vector/Matrix traits, wrappers |
| 3. Tensor Enhancement | Days 3-4 | Generic Tensor, DataType enums |
| 4. HAL Generification | Days 4-8 | CPU/GPU backends generic |
| 5. Module Refactoring | Days 8-18 | All 500+ functions generified |
| 6. Nalgebra Integration | Days 18-20 | Adapters, seamless interop |
| 7. Testing & Validation | Days 20-23 | Comprehensive test suite |
| 8. Documentation | Days 23-24 | Migration guides, examples |
| 9. Release | Days 23-24 | v0.2.0 with all features |

**Total: 14-24 days** (depends on parallelization)

---

## Implementation Notes for Agent

1. **Start with Phase 1-2**: Get the trait system right before touching code
2. **Parallel Work**: Phases 5.1-5.7 can be parallelized across agents
3. **Test Early**: Run tests after each phase to catch issues
4. **Document as You Go**: Don't leave documentation until the end
5. **GPU Shaders**: Create shader templates early to unblock GPU work
6. **Nalgebra**: Keep interaction minimal - use adapters, not direct deps

---

## Files Checklist

### New Files to Create
- [ ] `core/src/float.rs` - Float trait
- [ ] `core/src/vector.rs` - Vector traits
- [ ] `core/src/nalgebra_adapters.rs` - Nalgebra integration
- [ ] `core/src/nalgebra_wrapper.rs` - Wrapper structs
- [ ] `hal/shaders/*_f16.wgsl` - f16 variants
- [ ] `hal/shaders/*_bf16.wgsl` - bf16 variants
- [ ] Tests for all modules
- [ ] Documentation files

### Files to Modify
- [ ] `core/Cargo.toml` - Add dependencies
- [ ] `core/src/lib.rs` - Export new modules
- [ ] `core/src/tensor.rs` - Generic updates
- [ ] `hal/src/cpu/mod.rs` - 500+ function signatures
- [ ] `hal/src/gpu.rs` - GPU dispatch logic
- [ ] All module files in calib3d/, features/, stereo/, 3d/, etc.
- [ ] All Cargo.toml files - Add feature flags

---

**Ready for Agent Implementation** ✅
