# Normal Estimation & Memory Architecture Optimization Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize normal estimation performance by implementing zero-copy unified memory paths and refining the kNN spatial indexing algorithms.

**Architecture:** 
1. **Unified Memory Awareness**: Enhance `GpuStorage` and `CpuStorage` to detect if they share physical memory (iGPU/UMA). If so, `sync_to_device` and `sync_to_host` become no-ops or simple cache flushes.
2. **Normal Estimation Optimization**:
    - **CPU**: Implement a truly fast KD-Tree or Octree-based kNN instead of the current voxel-hash approach for better spatial locality.
    - **GPU**: Eliminate the CPU-side Morton sort in `compute_normals` and implement a full-GPU radix sort + sliding window kNN to keep data on-device.
3. **Hybrid Path**: Refine the CPU-kNN phase to use a lock-free work-stealing grid to maximize CPU utilization before shipping covariances to the GPU.

**Tech Stack:** wgpu, WGSL, rayon, nalgebra

---

### Task 1: Unified Memory Detection & Zero-Copy Tensors

**Files:**
- Modify: `crates/hal/src/gpu.rs`
- Modify: `crates/runtime/src/memory.rs`

**Step 1: Implement UMA detection**
Add a method to `GpuContext` to check if the adapter is integrated and supports unified memory.

**Step 2: Optimize `sync_to_device` in `UnifiedBuffer`**
If `is_unified_memory()` is true, avoid the `write_buffer` call and instead use mapped memory or rely on the shared physical address space.

---

### Task 2: GPU Normal Estimation - Full Pipeline

**Files:**
- Create: `crates/hal/shaders/pointcloud_knn.wgsl`
- Modify: `crates/hal/src/gpu_kernels/pointcloud.rs`

**Step 1: Implement GPU Radix Sort**
Move the Morton sort from CPU to GPU using a compute shader. This avoids the 500k-point round-trip.

**Step 2: Implement GPU Sliding Window kNN**
Compute the k-nearest neighbors directly on the sorted GPU buffer.

**Step 3: Update `compute_normals`**
Chain the sort, kNN, and PCA kernels into a single command buffer submission.

---

### Task 3: CPU Normal Estimation - Spatial Indexing Refinement

**Files:**
- Modify: `crates/3d/src/gpu/mod.rs` (the `compute_normals_cpu` function)

**Step 1: Replace voxel-hash with a compact Grid-Link-List**
This avoids `hashbrown` overhead and improves cache hits during the 27-neighbor-voxel search.

**Step 2: Parallelize Grid Building**
Use `rayon` to build the spatial index in parallel.

---

### Task 4: Comprehensive Benchmarking & Profiling

**Files:**
- Modify: `crates/3d/tests/perf_profiling.rs`

**Step 1: Compare all 4 modes**
Add specific blocks for `Cpu`, `Gpu`, `Hybrid`, and `Approx` with 500k points.

**Step 2: Verify zero-copy wins**
Measure the time taken for `sync_to_device` specifically on the user's Intel Meteor Lake iGPU.
