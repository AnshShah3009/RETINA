# Canonical Algorithm Homes Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One implementation per point-cloud filter and per CPU bilateral filter, with the other entry points delegating to it.

**Architecture:** Canonical f64 kernels live in `cv-pointcloud` (L2), which `cv-3d` (L4) already depends on; `cv-3d` re-exports or wraps them. The canonical gray bilateral lives in `cv-imgproc`; `cv-photo` moves to layer 5 and delegates. k-d trees stay as they are (the two trees have different, documented contracts).

**Tech Stack:** Rust 2024, nalgebra, `cv_math::spatial::KDTree` (f64).

**Spec:** `docs/superpowers/specs/2026-11-03-workspace-structure-design.md` (section "Duplicated algorithms"), refined by the duplicate survey summarised under each task.

## Global Constraints

- Every Open3D/OpenCV parity test keeps passing unchanged: `crates/3d/tests/statistical_outlier_open3d.rs` (364 / 410 / 340 / 484), `crates/3d/tests/normals_knn_open3d.rs` (1e-9), `crates/pointcloud/tests/normals_knn_open3d.rs` (1e-4), `crates/3d/tests/knn_filters_kdtree.rs`, `crates/photo/tests/bilateral_opencv_parity.rs` and `crates/imgproc/tests/bilateral_opencv_parity.rs` (exact).
- Public signatures of existing functions do not change. Behaviour changes only where a task says so, each with a test that failed before.
- Zero warnings, `cargo fmt --all`, `tests/crate_layers.rs` passes.

## Review Focus

- Non-finite points (NaN/inf) in every point-cloud entry point: skipped from neighbourhoods and statistics; never poison other points' results.
- Invalid parameters (voxel size 0, negative, NaN; k = 0): each public function keeps its current documented return (empty vs clone).
- Colours/normals slices shorter than points: no panic.
- f32 → f64 conversion at wrapper boundaries: boundary points in the outlier test must not flip (the Open3D counts are the check).
- Output order: deterministic (sorted voxel keys) everywhere after the change.

---

### Task 1: Statistical outlier removal in `cv-pointcloud`

**Files:** Create `crates/pointcloud/src/filters.rs`; modify `crates/pointcloud/src/{lib.rs,point_cloud.rs}`, `crates/3d/src/filters/statistical_filter.rs`; test `crates/3d/tests/statistical_outlier_non_finite.rs`.

**Interfaces — Produces:** `cv_pointcloud::filters::statistical_outlier_removal(points: &[Point3<f64>], nb_neighbors: usize, std_ratio: f64) -> (Vec<Point3<f64>>, Vec<usize>)`.

- [ ] Failing test `a_nan_point_does_not_change_which_points_survive`: a 30-point cloud with one far outlier; inserting one NaN point must give the same kept set (indices remapped) from `cv_3d::filters::statistical_outlier_removal`, and the NaN index is not kept. Today the NaN makes the threshold NaN.
- [ ] Move the f64 algorithm from `crates/3d/src/filters/statistical_filter.rs` (cv_math KD-tree, neighbourhood includes the point, sample std, keep if `md < thr || md == 0`), adding: non-finite points are left out of the tree and the statistics and are never kept.
- [ ] `cv_3d::filters::statistical_outlier_removal` re-exports it. `cv_pointcloud::remove_statistical_outliers` converts to f64, calls it (keeping its `k == 0` → clone guard), and gathers colours/normals by the kept indices.
- [ ] Verify all tests listed in Global Constraints plus `crates/pointcloud/tests/degenerate_inputs.rs`, root `tests/scientific_*`.
- [ ] Commit `refactor(pointcloud): one statistical outlier filter, which ignores non-finite points` (body notes the NaN fix).

### Task 2: Voxel downsampling in `cv-pointcloud`

**Files:** `crates/pointcloud/src/filters.rs`; `crates/3d/src/filters/{mod.rs,voxel_filter.rs}`, `crates/3d/src/gpu/mod.rs:~912`; test `crates/3d/tests/voxel_downsample_deterministic.rs`.

**Interfaces — Produces:** `cv_pointcloud::filters::{voxel_downsample, VoxelDownsampleResult}` with the current `cv_3d::filters` signature (f64 points, optional normals and colours, f64 size).

- [ ] Failing test `gpu_voxel_downsample_matches_the_filter_and_is_ordered`: `cv_3d::gpu::voxel_downsample(&pts_f32, s)` equals `cv_3d::filters::voxel_downsample` on the same points (as f64, then cast) in the same order, for a cloud whose first point is not on the global grid; and returns empty for NaN / 0 size.
- [ ] Move `voxel_downsample` and `VoxelDownsampleResult` unchanged; `cv_3d::filters` re-exports both.
- [ ] `cv_pointcloud::voxel_down_sample` becomes a wrapper that keeps its clone-on-invalid guard.
- [ ] `cv_3d::gpu::voxel_downsample` (both entry points) delegates (behaviour change: global grid instead of the `points[0]` anchor, sorted output). `VoxelGrid::downsample` is unchanged (its origin anchor is its contract).
- [ ] Verify `crates/3d/tests/{voxel_non_finite,length_checks,voxel_grid_bounds}.rs`, `crates/pointcloud/tests/{degenerate_inputs,voxel_down_sample_nan}.rs`.
- [ ] Commit `refactor(pointcloud): one voxel downsampler on the global grid`.

### Task 3: kNN normal estimation in `cv-pointcloud`

**Files:** `crates/pointcloud/src/filters.rs` (or `normals.rs`); `crates/3d/src/filters/normal_estimation.rs`, `crates/pointcloud/src/point_cloud.rs:~194`; test `crates/3d/tests/normals_knn_non_finite.rs`.

**Interfaces — Produces:** `cv_pointcloud::normals::estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>>` — eigenvector sign as computed, no orientation.

- [ ] Failing test `a_nan_point_gets_the_default_normal_and_leaves_neighbours_alone`: `cv_3d::filters::estimate_normals_knn` on a plane plus one NaN point gives (0,0,1) for the NaN point and the same normals for the others as without it.
- [ ] Kernel: neighbourhood k points including self, `m = k.min(n_finite)`, `m < 3` or non-finite point → (0,0,1), cv_math KD-tree and eigensolver (as `cv-3d` does today, so the 1e-9 test holds).
- [ ] `cv_3d::filters::estimate_normals_knn` = kernel + flip to z ≥ 0 (mesh reconstruction relies on it). `cv_pointcloud::estimate_normals` = f32→f64 wrapper, no flip.
- [ ] The fast/GPU normal family in `cv_3d::normals` and registration's radius PCA stay separate (different neighbourhood definitions).
- [ ] Verify both `normals_knn_open3d.rs`, `knn_filters_kdtree.rs`, `crates/pointcloud/tests/{planar_normals,degenerate_inputs}.rs`, `crates/3d` mesh reconstruction tests.
- [ ] Commit `refactor(pointcloud): one kNN normal estimator`.

### Task 4: Gray bilateral filter in `cv-imgproc`

**Files:** `crates/imgproc/src/bilateral.rs`, `crates/photo/{Cargo.toml,src/bilateral.rs}`, `tests/crate_layers.rs` (cv-photo → layer 5), `docs/ARCHITECTURE.md`.

**Interfaces — Produces:** `cv_imgproc::bilateral_filter_gray(src: &GrayImage, d: i32, sigma_color: f32, sigma_space: f32) -> GrayImage` — OpenCV `bilateralFilter` on CV_8UC1 with absolute `sigma_color`.

- [ ] Move the algorithm from `crates/photo/src/bilateral.rs` onto imgproc's shared helpers (`disc_taps`, `reflect_101`, `gauss_color_coeff`), absolute `sigma_color` (no /255 round trip), OpenCV rounding.
- [ ] `cv_photo::bilateral_filter` delegates; `cv-imgproc` becomes a normal dependency of `cv-photo`; `cv-photo` moves to layer 5 in `LAYERS` and the diagram.
- [ ] Verify `crates/photo/tests/{bilateral_opencv_parity,bilateral_rounding}.rs`, `crates/imgproc/tests/{bilateral_opencv_parity,zero_sized_images}.rs` pass unchanged.
- [ ] Commit `refactor(imgproc): one OpenCV-exact gray bilateral filter; cv-photo delegates`.

### Task 5: Docs and follow-ups

- [ ] ARCHITECTURE "where things go": point-cloud filters → `cv-pointcloud`; CHANGELOG lines for the NaN fix and the deterministic GPU voxel output.
- [ ] CONTRIBUTING starter item: the HAL CPU bilateral (`crates/hal/src/cpu/compute_context_impl.rs` ~3826) uses a square window and `ceil` radius, not OpenCV's disc and `cvRound`; CPU and GPU must change together.
- [ ] Full gate, push, sync branches.
