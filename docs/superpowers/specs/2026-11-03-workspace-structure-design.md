# Workspace structure: decision record

Status: accepted (2026-11-03). Maintainer delegated the decision.

## Question

Should RETINA be split into separately maintained repositories (git
submodules), and how should the ~30 crates be layered so the structure stays
maintainable as it grows?

## Decision summary

1. **One repository, one Cargo workspace. No git submodules.**
2. **Strict layers**, enforced by a test, with every crate depending only on
   lower layers.
3. **Fix the four layering violations** found in the survey.
4. **One canonical home per algorithm**; other crates re-export or delegate.
5. **Shared workspace metadata** (`[workspace.package]`, lockstep version,
   MSRV, licence) and supply-chain checks (`cargo-deny`).
6. **Publish names use a `retina-` prefix** when the crates first go to
   crates.io (`cv-core` and `retina` are taken there); the facade crate is
   `retina-cv`. The rename is deferred to the first release so it happens
   once, together with versioning.
7. **The three long-lived branches stay structurally identical.** Structure
   changes land on `master` and merge forward to `develop` and
   `research/cubecl` like every other change.

## Why no submodules

Large Rust projects keep tightly coupled crates in one workspace: wgpu
(wgpu, wgpu-core, wgpu-hal, wgpu-types, and naga, which was *merged in* from
its own repository in 2023), Bevy (~50 `bevy_*` crates), tokio, rust-analyzer,
nalgebra. The reasons apply here:

- RETINA's crates change together. A HAL change touches `cv-hal`, its WGSL
  shaders, the CPU reference, the parity tests and the domain wrapper in one
  commit. Submodules turn that into several PRs in several repos plus pointer
  bumps, and CI can no longer test the combination atomically.
- One `Cargo.lock`, one CI matrix, one issue tracker, one review queue.
- Submodules are a poor fit for contributors (`--recursive` clones, detached
  HEADs, pointer conflicts) — the opposite of the contribution-friendly goal.
- Cargo already gives the isolation submodules would: each crate builds,
  tests and publishes on its own.

A crate leaves the repository only if it gains an independent release cadence
and outside users who do not use the rest of RETINA. `cv-plot` (no internal
dependencies) is the only current candidate; revisit it if that happens.

## Layers

```text
L0 foundation   cv-core, cv-geometry2d, cv-plot
L1 backends     cv-hal, cv-io, cv-math, cv-eval, cv-photo, cv-rendering, cv-videoio, cv-viewer
L2 numerics     cv-optimize, cv-pointcloud, cv-signal, cv-video, cv-distributed
L3 runtime      cv-runtime
L4 domain       cv-imgproc, cv-3d
L5 domain       cv-calib3d, cv-features, cv-registration, cv-dnn
L6 domain       cv-objdetect, cv-cli
L7 pipelines    cv-sfm, cv-slam, cv-localization, cv-examples
L8 surface      cv-python, rust-cv-native (root: facade, workspace tests)
```

The table lives in one place — `tests/crate_layers.rs` — and the test fails
when a crate depends on a crate in the same or a higher layer (same-layer
edges are listed explicitly where intended). `docs/ARCHITECTURE.md` mirrors it.

## Layering violations and fixes

| Violation | Cause | Fix |
| --- | --- | --- |
| `cv-features → cv-calib3d` | ArUco/ChArUco/AprilTag and HOG/Haar live in `cv-features` and need PnP, undistortion and DLT; `ransac`/`usac` use calib3d's DLT solvers | New crate **`cv-objdetect`** (OpenCV's `objdetect` module, which holds ArUco/ChArUco, HOG and cascades since 4.7) takes `aruco`, `aruco_tables`, `charuco`, `markers`, `hog`, `haar_cascade`. `ransac` and `usac` move to **`cv-calib3d`** (OpenCV's USAC lives in calib3d). `cv-features` then depends on no geometry crate. |
| `corner_subpix` in `cv-calib3d` | Misplaced | Moves to **`cv-imgproc`** (OpenCV `cornerSubPix` is in imgproc); `cv-calib3d` re-exports it. |
| `cv-io` (L1) `→ cv-3d` (L4) | File formats belong low in the graph, but `cv-io` needs only the `TriangleMesh` type, but pulls in hal, runtime and wgpu | `TriangleMesh` moves to **`cv-core::mesh`** beside `PointCloud`; `cv_3d::mesh` re-exports it. |
| `cv-3d → cv-scientific` | `cv-scientific` is a 60-line re-export shim over math/geometry2d/signal/pointcloud | **Remove `cv-scientific`**; users depend on the real crates. Its behavioural tests move to the crates they test. |

`cv-runtime → cv-distributed` is a legitimate downward edge and stays.

## Duplicated algorithms (canonical homes)

| Algorithm | Copies | Canonical home |
| --- | --- | --- |
| Voxel downsampling | `cv-3d` filters, `cv-3d` gpu CPU fallback, `cv-pointcloud` | `cv-pointcloud` (CPU), `cv-hal` (GPU) |
| Statistical outlier removal | `cv-3d`, `cv-pointcloud` | `cv-pointcloud` |
| Normal estimation | `cv-3d`, `cv-pointcloud`, `cv-registration` | `cv-pointcloud` |
| k-d tree | `cv-3d::spatial::KDTree`, `cv-math::spatial::KDTree` | `cv-math` |
| Bilateral filter | `cv-imgproc`, `cv-photo` | `cv-imgproc` (OpenCV puts it in imgproc); `cv-photo` re-exports |

HAL GPU kernels with the same names as domain functions (`gaussian_blur`,
`match_template`) are backend implementations, not duplicates.

Each merge keeps the behaviour of the copy that matches the reference library
(the parity tests decide), and lands separately with its tests.

## Workspace metadata

- `[workspace.package]`: `version = "0.2.0"` (lockstep: every crate shares the
  workspace version, as Bevy and wgpu do), `edition = "2024"`,
  `rust-version = "1.95"`, `license = "MIT OR Apache-2.0"`, `repository`,
  `homepage`, `authors`. Every crate inherits them.
- Every internal dependency goes through `[workspace.dependencies]`.
- `publish = false` on `cv-examples` and `cv-python`.
- `deny.toml` + a CI job running `cargo deny check` (licences, bans,
  sources, advisories).

## Out of scope

- The `retina-*` rename (decision 6) — done in the release that first
  publishes to crates.io.
- Moving `cv-plot` out of the repository.
- Splitting `cv-hal` by backend (`wgpu-hal`-style); revisit if backends gain
  separate maintainers.
