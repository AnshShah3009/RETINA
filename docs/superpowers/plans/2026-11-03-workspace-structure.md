# Workspace Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a strict crate layering, remove the four layering violations, and give the workspace shared metadata and supply-chain checks.

**Architecture:** Monorepo, no submodules. A root integration test owns the layer table and fails on any upward or undeclared same-layer edge. Code moves keep the old paths working through re-exports where the dependency direction allows it.

**Tech Stack:** Cargo workspace (edition 2024, resolver 3), `cargo metadata`, `serde_json`, cargo-deny.

**Spec:** `docs/superpowers/specs/2026-11-03-workspace-structure-design.md`

## Global Constraints

- Rust 1.95, edition 2024, `cargo fmt --all` (max_width 100).
- Zero warnings: `cargo check --workspace --all-targets`, and with `-p cv-hal --features half-precision` and `--features cubecl`.
- `cargo test --workspace --no-fail-fast` passes (3,097+ tests) after every task.
- Lands on `master`, then `develop` (merge) and `research/cubecl` (fast-forward).
- Behaviour unchanged: moves only. Tests move with their code.

## Review Focus

- Python bindings (`crates/python`) import moved items: `cargo check -p cv-python` must pass after every move.
- Examples and doctests referencing old paths (`cv_features::aruco`, `cv_scientific::...`): `cargo test --doc --workspace` and `cargo build --examples --workspace`.
- `cv-features` `gpu` feature: `markers/gpu.rs` moves; `cargo check -p cv-objdetect --features gpu` must pass.
- Root `src/lib.rs` re-exports (`rust-cv-native`) of removed crates.
- Same-name collisions after moves (`FundamentalEstimator` exists in calib3d already): no glob re-export of the moved `ransac` module at the calib3d root.

---

### Task 1: Workspace metadata

**Files:** `Cargo.toml`, `crates/*/Cargo.toml`

- [ ] Add `[workspace.package]` with `version = "0.2.0"`, `edition = "2024"`, `rust-version = "1.95"`, `license = "MIT OR Apache-2.0"`, `repository = "https://github.com/AnshShah3009/RETINA"`, `homepage` (same), `authors = ["Ansh Shah and the RETINA contributors"]`.
- [ ] Each crate: `version.workspace = true`, `edition.workspace = true`, `rust-version.workspace = true`, `license.workspace = true`, `repository.workspace = true`; root package too.
- [ ] Internal deps written as `path = "../x"` switch to `{ workspace = true }`; internal workspace deps get `version = "0.2.0"` so they are publishable.
- [ ] `publish = false` in `crates/examples` and `crates/python`.
- [ ] Verify: `cargo metadata --no-deps --format-version 1 | jq '[.packages[] | select(.license != "MIT OR Apache-2.0")] | length'` → `0`; `cargo check --workspace --all-targets` clean.
- [ ] Commit `build: share package metadata across the workspace`.

### Task 2: Layer test

**Files:** Create `tests/crate_layers.rs`.

- [ ] Test `crate_dependencies_respect_layers`: run `cargo metadata --format-version 1 --no-deps` (via `env!("CARGO")`, manifest dir = workspace root), collect each workspace package's normal + build dependencies on other workspace packages (dev-dependencies excluded), and assert for every edge `layer(dep) < layer(pkg)`. Layer table = the spec's table (with `cv-objdetect` listed). Failure message lists every offending edge `a (Ln) -> b (Lm)`.
- [ ] Test `every_workspace_crate_has_a_layer`: every package in metadata appears in the table, and every table entry exists (so adding a crate forces a layer decision).
- [ ] Run: fails listing `cv-features (L5) -> cv-calib3d (L5)`, `cv-3d (L4) -> cv-scientific`, and the missing `cv-objdetect`. Keep a temporary `KNOWN_VIOLATIONS` list so the test passes; Tasks 3–7 delete entries from it; Task 7 deletes the list.
- [ ] Commit `test: enforce the crate layer order`.

### Task 3: Remove `cv-scientific`

**Files:** delete `crates/scientific`; modify `Cargo.toml`, `crates/3d/{Cargo.toml,src/normals.rs,src/gpu/mod.rs,src/spatial/kd_tree.rs,tests/statistical_outlier_open3d.rs}`, `crates/python/src/{scientific.rs,optimize.rs,three_d.rs}`, `crates/examples/src/scientific.rs`, `src/lib.rs`.

- [ ] Map each `cv_scientific::m` to its source per `crates/scientific/src/lib.rs` (fft/signal → `cv_signal`, geometry2d → `cv_geometry2d::geometry2d`, point_cloud → `cv_pointcloud::point_cloud`, the rest → `cv_math`).
- [ ] Move `crates/scientific/tests/{functional_tests,degenerate_inputs,perf_profiling}.rs` to the crate each test exercises (split per crate if mixed); delete `reexport_contract.rs`.
- [ ] Verify: `cargo test --workspace --no-fail-fast` same pass count minus `reexport_contract`; `grep -rn cv_scientific crates src tests` → only CHANGELOG.
- [ ] Commit `refactor!: remove the cv-scientific re-export crate`.

### Task 4: `TriangleMesh` into `cv-core`

**Files:** Create `crates/core/src/mesh.rs` (from `crates/3d/src/mesh/mod.rs`); `crates/3d/src/mesh/mod.rs` becomes `pub use cv_core::mesh::TriangleMesh;` plus its non-type items; `crates/io/{Cargo.toml,src/mesh.rs}`.

- [ ] Add `rand_distr` to `cv-core` deps (the mesh sampler needs `WeightedIndex`).
- [ ] Move the struct, its impls, `Default` and the module's unit tests.
- [ ] `cv-io`: `pub use cv_core::mesh::TriangleMesh;`, drop `cv-3d` from `[dependencies]` (keep as dev-dep only if a test needs it).
- [ ] Verify: `cargo tree -p cv-io -e normal | grep -c wgpu` → `0`; `cargo test -p cv-io -p cv-3d -p cv-core --features cv-io/las,cv-io/gltf`.
- [ ] Commit `refactor: move TriangleMesh into cv-core so cv-io no longer needs cv-3d`.

### Task 5: `corner_subpix` into `cv-imgproc`

**Files:** Create `crates/imgproc/src/corner_subpix.rs` from `crates/calib3d/src/pattern.rs:~100-260` (`SubpixCriteria`, `corner_subpix`, `corner_subpix_with` and private helpers); `crates/calib3d/src/pattern.rs` re-exports them; move `crates/calib3d/tests/corner_subpix_opencv_parity.rs` to `crates/imgproc/tests/`.

- [ ] Verify: `cargo test -p cv-imgproc --test corner_subpix_opencv_parity` passes; `cargo test -p cv-calib3d` passes.
- [ ] Commit `refactor: move corner_subpix to cv-imgproc, where OpenCV keeps cornerSubPix`.

### Task 6: `ransac` and `usac` into `cv-calib3d`

**Files:** move `crates/features/src/{ransac,usac}.rs` → `crates/calib3d/src/`; `crates/calib3d/src/lib.rs`: `pub mod ransac; pub mod usac; pub use usac::*;` and root re-exports of the non-colliding ransac items (`HomographyEstimator` collides? check; `RansacConfig`, `RansacMatcher`, `MatchPair`, `filter_matches_by_inliers`, `estimate_homography`, `estimate_fundamental`). `crate::dlt` paths replace `cv_calib3d::dlt`.

- [ ] Move their unit tests with them; fix tests in `crates/features/tests` that import them.
- [ ] Verify: `grep -n cv_calib3d crates/features/src/{ransac,usac}.rs` → none (files gone); `cargo test -p cv-calib3d`.
- [ ] Commit `refactor!: move RANSAC and USAC estimators to cv-calib3d`.

### Task 7: New crate `cv-objdetect`

**Files:** Create `crates/objdetect/{Cargo.toml,src/lib.rs}`; move `crates/features/src/{aruco.rs,aruco/,aruco_tables.rs,charuco.rs,markers.rs,markers/,hog.rs,haar_cascade/}` and `crates/features/tests/{aruco_pose,markers_opencv_parity}.rs` (+ their fixture data); register in root `[workspace]` members and `[workspace.dependencies]`.

- [ ] `cv-objdetect` deps: core, hal, imgproc, features (only if a moved file uses it), calib3d, runtime as needed, plus the external deps the moved files use; feature `gpu` mirrors `cv-features`'s `gpu` for `markers/gpu.rs`.
- [ ] `lib.rs` keeps the same module names and root re-exports the moved files had in `cv-features` (including the `charuco::CharucoBoard` disambiguation).
- [ ] Remove `cv-calib3d` from `cv-features` `[dependencies]`.
- [ ] Update `cv-python`, examples and root `src/lib.rs` imports.
- [ ] Delete `KNOWN_VIOLATIONS` from `tests/crate_layers.rs`.
- [ ] Verify: `cargo test --test crate_layers`; `cargo check -p cv-objdetect --features gpu --all-targets`; full workspace tests.
- [ ] Commit `refactor!: split ArUco, ChArUco, AprilTag, HOG and cascades into cv-objdetect`.

### Task 8: cargo-deny

**Files:** Create `deny.toml`; modify `.github/workflows/build.yml`, `scripts/check.sh` (full mode), `CONTRIBUTING.md` CI table.

- [ ] `deny.toml`: `[licenses] allow` = the licences `cargo deny list` reports for the current tree (no copyleft beyond what is present today, each exception named per crate); `[bans] multiple-versions = "warn"`; `[sources] unknown-registry = "deny"`, `unknown-git = "deny"`; `[advisories]` default (`yanked = "deny"`), ignores only with a reason comment.
- [ ] CI: job `deny` using `EmbarkStudios/cargo-deny-action@v2`.
- [ ] Verify: `cargo deny check` exits 0 locally.
- [ ] Commit `ci: check licences, sources and advisories with cargo-deny`.

### Task 9: Docs

**Files:** `docs/ARCHITECTURE.md` (layers, where things go), `README.md` (crates table), `CHANGELOG.md` (breaking: `cv-scientific` removed; objdetect split; ransac/usac and corner_subpix moved), `CONTRIBUTING.md` (adding a crate = add a layer entry), `GOVERNANCE.md` (monorepo policy, link to the spec).

- [ ] `cargo test --doc -p rust-cv-native` (README doctest) passes.
- [ ] Commit `docs: describe the layered workspace and the moves`.

### Task 10: Gate and branch sync

- [ ] `~/.cache/retina-gate/full.sh` → fmt 0, warnings 0/0/0, `GATE_DONE` with 0 failed.
- [ ] Push `master`; merge into `develop`; fast-forward `research/cubecl`; CI green on all three.

## Phase 2 (separate plan): canonical homes

One task per row of the spec's duplication table, each keeping the variant the parity tests pin, deleting the others behind re-exports, and moving their tests. Written after Phase 1 lands.
