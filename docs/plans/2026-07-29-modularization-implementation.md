# RETINA Modularization — Implementation Summary

**Date:** 2026-07-29
**Base:** Phase 1-8 of the modularization plan
**Workspace:** 26 crates (was 22)

---

## Overview

Modularized the RETINA codebase by splitting monolithic files into focused modules and breaking multi-concern crates into dedicated sub-crates. All work maintains full backward compatibility via re-exports and meta-crates.

---

## Crate Structure (Post-Modularization)

```
crates/
├── core/                     # cv-core (unchanged crate, internal splits)
│   └── src/
│       ├── geometry/         # NEW: camera.rs, distortion.rs, pose.rs, mod.rs
│       ├── tensor/           # NEW: types.rs, ops.rs, mod.rs
│       └── ...
├── hal/                      # cv-hal (internal splits)
│   └── src/
│       ├── cpu/              # REFACTORED
│       │   ├── mod.rs        # 161L (was 5074L — struct + ComputeBackend impl + include!)
│       │   ├── compute_context_impl.rs  # NEW: 4793L — single ComputeContext trait impl
│       │   ├── border.rs     # NEW: border coordinate helpers
│       │   └── utils.rs      # NEW: pixel/descriptor utility helpers
│       └── gpu/              # CONVERTED from gpu.rs to directory module
│           ├── mod.rs        # 522L (was 3084L)
│           └── compute_context_impl.rs  # NEW: 2565L
├── math/                     # NEW: cv-math (stats, special, linalg, sparse, interpolate, integrate, spatial, geometry, jit)
├── geometry2d/               # NEW: cv-geometry2d (2D computational geometry, zero external deps)
├── signal_proc/              # NEW: cv-signal (FFT, signal processing, filters, windows, wavelets)
├── pointcloud/               # NEW: cv-pointcloud (normals, downsampling, RANSAC, FPFH)
├── scientific/               # CONVERTED to meta-crate re-exporting the 4 sub-crates above
├── video/src/tracking/       # CONVERTED from file to directory module
│   ├── template.rs, meanshift.rs, kcf.rs, mosse.rs, multi.rs, mod.rs
├── optimize/src/             # SPLIT general.rs into per-solver files
│   ├── nelder_mead.rs, bfgs.rs, lbfgsb.rs, levenberg_marquardt.rs, brent.rs, newton.rs
├── imgproc/src/
│   ├── contours.rs           # 1093L → extracted moments to moments.rs
│   └── moments.rs            # NEW
├── calib3d/src/
│   ├── essential.rs          # NEW: EssentialSolver from multiview.rs
│   ├── fundamental.rs        # NEW: FundamentalSolver from multiview.rs
│   ├── homography.rs         # NEW: HomographySolver from multiview.rs
│   ├── multiview.rs          # 5L (was 1231L) — re-export shim
│   ├── lib.rs                # 56L (was 1236L) — tests extracted to tests/lib_tests.rs
│   └── tests/lib_tests.rs    # NEW: 28 integration tests
├── registration/src/registration/
│   ├── global/               # CONVERTED to directory module
│   │   ├── fpfh.rs, ransac.rs, mod.rs
├── 3d/src/
│   ├── mesh/reconstruction/  # CONVERTED to directory module
│   │   ├── poisson.rs, ball_pivoting.rs, alpha_shapes.rs, marching_cubes.rs, delaunay.rs, mod.rs
│   ├── gpu/                  # SPLIT into per-domain files
│   │   ├── tsdf.rs, raycasting.rs, mesh.rs, registration.rs (+ existing point_cloud.rs)
│   ├── spatial/              # SPLIT
│   │   ├── kd_tree.rs, octree.rs, voxel_grid.rs (+ existing bvh.rs, hash_grid.rs)
│   └── filters/              # CONVERTED to directory module
│       ├── voxel_filter.rs, statistical_filter.rs, radius_filter.rs, bounding_box.rs,
│       │   normal_estimation.rs, transform.rs, uniform_filter.rs, crop_filter.rs, mod.rs
```

---

## Key Design Decisions

### 1. Why `include!()` not `pub mod` for trait impls?
Rust does not allow splitting `impl Trait for Type` across multiple files. The `include!()` macro at module level inlines the entire impl block from a separate file, achieving file-level separation while respecting the language constraint.

### 2. Why backward-compatible meta-crates?
`cv-scientific` is now a thin re-export layer over `cv-math`, `cv-geometry2d`, `cv-signal`, and `cv-pointcloud`. All existing `use cv_scientific::*` paths continue to work. New code should depend directly on the sub-crates.

### 3. What was NOT split?
- `cv-runtime` crate split → deferred. Orchestrator has hard imports from both distributed/ and observe/ modules. Requires type-level refactoring (NodeId extraction, trait-mediated dispatch).
- `cv-hal/gpu_kernels/mod.rs` → already well-structured with 41 individual kernel files. Only mod.rs helpers remain.
- `cv-io/pcd.rs` → large but coherent PCD format I/O.
- `cv-features/src/markers.rs` + `aruco.rs` → intentional complementary APIs (image-based vs tensor-based).

### 4. Test migration
- `cv-calib3d/src/lib.rs`: 28 inline tests moved to `tests/lib_tests.rs`
- All other test suites left in-place (can be migrated later).

---

## File Size Improvements

| File | Before | After |
|------|--------|-------|
| `cpu/mod.rs` | 5074L | 161L |
| `gpu.rs` → `gpu/mod.rs` | 3084L | 522L |
| `geometry.rs` → `geometry/mod.rs` | 1661L | distributed |
| `tensor.rs` → `tensor/mod.rs` | 1272L | distributed |
| `tracking.rs` → `tracking/mod.rs` | 1331L | 554L |
| `general.rs` | 1260L | 105L |
| `reconstruction.rs` → `reconstruction/mod.rs` | 1497L | distributed |
| `multiview.rs` | 1231L | 5L |
| `contours.rs` | 1093L | ~800L |
| `global.rs` | 1121L | ~200L |
| `lib.rs` (calib3d) | 1236L | 56L |

**Largest remaining file:** `gpu/compute_context_impl.rs` at 2565L (trait impl block — cannot be split without language changes).
