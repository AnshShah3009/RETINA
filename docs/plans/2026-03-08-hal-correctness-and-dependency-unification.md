# HAL Correctness and Dependency Unification Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Unify workspace dependencies to the latest versions and fix remaining HAL correctness issues (FAST parity) and missing GPU dispatches.

**Architecture:** 
1. Use Cargo's `[workspace.dependencies]` in the root `Cargo.toml` to manage all shared libraries.
2. Align CPU and GPU implementations for FAST feature detection by synchronizing NMS and scoring logic.
3. Bridge existing GPU kernels in `cv-hal` to the high-level `3d` and `optimize` crates.

**Tech Stack:** Rust, WGPU (HAL), Nalgebra, Rayon, Tokio.

---

### Task 1: Audit and Unify Dependencies in Root Cargo.toml

**Files:**
- Modify: `rust-cv-native/Cargo.toml`

**Step 1: Audit all sub-crate dependencies**
Identify all unique dependencies and their versions across the workspace.

**Step 2: Update root Cargo.toml**
Add `[workspace.dependencies]` with the latest stable versions of:
- `tokio`, `wgpu`, `nalgebra`, `rayon`, `serde`, `bytemuck`, `thiserror`, `image`, `ndarray`, `rand`, `pollster`, `rstar`, `wide`.

**Step 3: Verify workspace compiles**
Run: `cargo check`
Expected: Compilation succeeds (though sub-crates aren't using workspace deps yet).

---

### Task 2: Refactor Sub-crates to use Workspace Dependencies

**Files:**
- Modify: `rust-cv-native/*/Cargo.toml` (all 15+ crates)

**Step 1: Update each sub-crate Cargo.toml**
Replace version strings with `workspace = true`.
Example: `tokio = { workspace = true, features = ["full"] }`

**Step 2: Run workspace-wide check**
Run: `cargo check --workspace`
Expected: SUCCESS

**Step 3: Commit**
```bash
git add .
git commit -m "refactor: unify workspace dependencies using [workspace.dependencies]"
```

---

### Task 3: Fix FAST Feature Detection Parity

**Files:**
- Modify: `rust-cv-native/hal/src/cpu/mod.rs` (FAST logic)
- Modify: `rust-cv-native/hal/shaders/fast.wgsl`
- Modify: `rust-cv-native/hal/shaders/fast_nms.wgsl`

**Step 1: Align Scoring Logic**
Ensure both CPU and GPU use the exact same SAD (Sum of Absolute Differences) calculation for corner scores.

**Step 2: Align NMS Tie-breaking**
Ensure both implementations handle identical scores in a 3x3 neighborhood the same way (e.g., coordinate-based preference).

**Step 3: Run Parity Test**
Run: `cargo test -p rust-cv-native --test multi_gpu_tests -- --nocapture`
Expected: FAST parity check passes.

---

### Task 4: Connect GPU Dispatches in 3D Crate

**Files:**
- Modify: `rust-cv-native/3d/src/tsdf/mod.rs`
- Modify: `rust-cv-native/3d/src/odometry/mod.rs`
- Modify: `rust-cv-native/3d/src/raycasting/mod.rs`

**Step 1: Implement TSDF GPU Dispatch**
Call `hal::tsdf_integrate` instead of the CPU loop when a GPU context is available.

**Step 2: Implement Odometry GPU Dispatch**
Call `hal::compute_odometry`.

**Step 3: Implement Raycasting GPU Dispatch**
Call `hal::raycast_mesh`.

**Step 4: Verify with functional tests**
Run: `cargo test -p cv-3d`
Expected: SUCCESS

---

### Task 5: Implement GPU Solver in Optimize Crate

**Files:**
- Modify: `rust-cv-native/optimize/src/gpu_solver.rs`

**Step 1: Implement GPU Dispatch**
Implement the `solve` method to use `GpuContext` for matrix operations if available.

**Step 2: Verify**
Run: `cargo test -p cv-optimize`
Expected: SUCCESS

---

### Task 6: Robust Error Handling (Replace panic!)

**Files:**
- Modify: `rust-cv-native/runtime/src/observe/events.rs`
- Modify: `rust-cv-native/registration/src/registration/mod_test.rs`

**Step 1: Replace panic! with Result**
Refactor code to return `crate::Error` instead of panicking on invalid event types or ICP failures.

**Step 2: Verify**
Run: `cargo test --workspace`
Expected: SUCCESS
