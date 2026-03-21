# GPU LBVH-Based Normal Estimation Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a high-performance GPU Linear Bounding Volume Hierarchy (LBVH) to accelerate k-NN search for normal estimation, replicating the efficiency seen in NVIDIA Kaolin/Warp.

**Architecture:** 
1. **LBVH Construction**: Build a radix-tree on the GPU using Morton codes. Each internal node represents a range of points.
2. **LBVH Traversal**: Replace the $O(W)$ sliding window with a $O(\log N)$ hierarchical traversal in the k-NN compute shader.
3. **Optimized PCA**: Refine the batch PCA eigensolver for better numerical stability on the GPU.

**Tech Stack:** wgpu, WGSL, nalgebra

---

### Task 1: GPU LBVH Radix Tree Construction

**Files:**
- Create: `crates/hal/shaders/lbvh_build.wgsl`
- Create: `crates/hal/src/gpu_kernels/lbvh.rs`
- Modify: `crates/hal/src/gpu_kernels/mod.rs`

**Step 1: Implement Radix Tree Construction Shader**
Implement the algorithm by Karras (2012) to build a binary radix tree from sorted Morton codes in parallel.

**Step 2: Implement Rust Wrapper**
Create `lbvh.rs` to manage the lifecycle of the LBVH nodes buffer and dispatch the build kernel.

---

### Task 2: LBVH-Accelerated k-NN Shader

**Files:**
- Modify: `crates/hal/shaders/pointcloud_knn.wgsl`
- Modify: `crates/hal/src/gpu_kernels/pointcloud.rs`

**Step 1: Implement Stack-Based Traversal in WGSL**
Refactor the k-NN shader to traverse the radix tree. Use a small fixed-size stack in thread-local memory.

**Step 2: Update Pipeline Integration**
Update `compute_normals` in `pointcloud.rs` to insert the LBVH build step between Sort and k-NN.

---

### Task 3: Numerical Stability & Eigensolver Refinement

**Files:**
- Modify: `crates/hal/src/gpu_kernels/pointcloud_normals_batch_pca.wgsl`

**Step 1: Implement Jacobi Iterations (Optional/Fallback)**
Add a Jacobi-based eigensolver as a more robust alternative to the Cardano method for near-degenerate cases, inspired by Warp's `svd.h`.

---

### Task 4: Verification & Benchmarking

**Files:**
- Modify: `crates/3d/tests/perf_profiling.rs`

**Step 1: Run Large-Scale Benchmark**
Test with 1,000,000+ points to quantify the $O(\log N)$ scaling advantage of LBVH over the sliding window.

**Step 2: Accuracy Check**
Verify that LBVH-based k-NN finds the same neighbors as the brute-force/sliding window methods.
