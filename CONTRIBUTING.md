# Contributing to RETINA

Thanks for helping. RETINA is a pure-Rust computer-vision workspace that aims
for numerical parity with OpenCV, Open3D, SciPy and Matplotlib, with CPU and GPU
backends behind one API. This guide covers how to report problems, how to set up,
what a change needs before it can merge, and the conventions the code follows.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Contents

- [Ways to contribute](#ways-to-contribute)
- [The project's contract](#the-projects-contract)
- [Setting up](#setting-up)
- [Making a change](#making-a-change)
- [What every pull request needs](#what-every-pull-request-needs)
- [Recipes](#recipes)
  - [Fixing a bug](#fixing-a-bug)
  - [Matching a reference library (parity)](#matching-a-reference-library-parity)
  - [Making something faster](#making-something-faster)
  - [Adding or changing a GPU kernel](#adding-or-changing-a-gpu-kernel)
  - [Adding a new operation](#adding-a-new-operation)
- [Conventions](#conventions)
- [Commit messages](#commit-messages)
- [Branches and releases](#branches-and-releases)
- [Issue labels](#issue-labels)
- [Good places to start](#good-places-to-start)
- [License](#license)

## Ways to contribute

| You have... | Do this |
| --- | --- |
| A wrong result, crash or hang | Open a **Bug report** issue with a minimal reproduction |
| A result that differs from OpenCV / Open3D / SciPy | Open a **Parity discrepancy** issue with both outputs and the reference version |
| Something slow | Open a **Performance** issue with the size, hardware and timing |
| An idea or missing function | Open a **Feature request**; for anything large, agree on the design in the issue first |
| Time to review | Review open pull requests: run them, check the evidence, ask questions |
| Docs that confused you | Fix them; documentation PRs are welcome and need no issue |

Issue forms live in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) and ask
for exactly what a maintainer needs to reproduce the problem.

## The project's contract

These rules are why the test suite is trusted. Pull requests that break them are
sent back, however good the code is.

1. **Claims are measured.** No accuracy, parity or speed claim goes in docs,
   comments or commit messages unless it was measured, with the command or test
   that measured it. See [docs/evaluation.md](docs/evaluation.md).
2. **Bugs are proven before they are fixed.** A fix arrives with a test that
   fails on the old code and passes on the new one.
3. **Reference parity is explicit.** Where a function mirrors an OpenCV, Open3D,
   SciPy or Matplotlib function, its tests compare against values produced by
   that library (version stated), stored in the test so CI needs no Python. Any
   remaining difference is documented in [docs/PARITY.md](docs/PARITY.md).
4. **Tolerances are not loosened to make a test pass.** If an existing
   expectation is wrong, show why with evidence in the PR.
5. **Optimisations do not change results.** A speed-up must produce identical
   output (bit-identical where the existing contract is bit-exact) and show
   before/after benchmarks.
6. **No warnings.** CI fails on any compiler warning.
7. **No new `unsafe`.** 27 of the 30 crates have `#![forbid(unsafe_code)]`. The
   three exceptions (`cv-hal` for GPU buffer casts, `cv-distributed` for shared
   memory, `cv-examples`) need a written justification for any new `unsafe`.

## Setting up

**Required:**

- Rust **1.95 or newer**, stable channel (`rustup update stable`). The workspace
  uses edition 2024 and some dependencies need 1.95.
- A C toolchain and `pkg-config` (pulled in by some dependencies).

**Optional:**

- [`cargo-nextest`](https://nexte.st) — what CI uses to run tests
  (`cargo install cargo-nextest --locked`).
- A GPU with Vulkan, Metal or DX12 drivers. GPU tests run when an adapter is
  present and skip cleanly when it is not.
- Python 3.10+ with `opencv-python`, `numpy`, `scipy` and `open3d` — only to
  regenerate reference values for parity tests (see below).
- FFmpeg development libraries — only for `cv-videoio --features ffmpeg`.
- [`maturin`](https://www.maturin.rs) — only for the Python bindings.

**Build and test:**

```bash
git clone https://github.com/AnshShah3009/RETINA.git
cd RETINA
cargo build --workspace
scripts/check.sh          # what CI checks: fmt, warnings, clippy, tests
scripts/check.sh full     # also doctests, feature builds, examples, benches
```

**Python bindings** (from the repository root, where `pyproject.toml` lives):

```bash
pip install maturin
maturin develop --release
python -c "import cv_native"
```

A first full build compiles a lot (wgpu, nalgebra, the GPU stack) and the
`target/` directory grows to tens of gigabytes; keep disk space in mind.

## Making a change

1. **Find or open an issue** for anything beyond a small fix, so the approach can
   be agreed before you invest time. Comment on the issue to say you are on it.
2. **Fork and branch** from `master`. Name the branch after the change:
   `fix/<area>-<what>`, `feat/<area>-<what>`, `perf/<area>-<what>`, `docs/<what>`.
3. **Keep the change focused.** One bug, one feature or one optimisation per pull
   request. Unrelated clean-ups go in their own PR.
4. **Run `scripts/check.sh`** (and `full` if you touched features, GPU code or
   benches) before pushing.
5. **Open a pull request against `master`.** The template asks for the evidence
   described below. Draft PRs are welcome for early feedback.

## What every pull request needs

- `cargo fmt --all -- --check` passes.
- Zero compiler warnings from `cargo check --workspace --all-targets` (and with
  `-p cv-hal --features half-precision` / `--features cubecl` if you touched
  `cv-hal`).
- `cargo test --workspace --no-fail-fast` passes. Use `--no-fail-fast`: plain
  `cargo test` stops at the first failing test binary and hides later failures.
- New behaviour has tests; bug fixes have a test that failed before the fix.
- Public items have doc comments, and documentation matches behaviour.
- [CHANGELOG.md](CHANGELOG.md) has a line under **Unreleased** for user-visible
  changes, and [docs/bug-log.md](docs/bug-log.md) has a row for each defect fixed.

**What CI runs** ([`.github/workflows/build.yml`](.github/workflows/build.yml)),
on every push and pull request to `master`, `develop` and `research/cubecl`
that touches Rust code, manifests, shaders or test fixtures:

| Job | Checks |
| --- | --- |
| Linux | fmt, build, half-precision and CubeCL feature checks, LAS feature tests, no warnings, examples, clippy, nextest, doctests |
| Supply chain | `cargo deny check` against [`deny.toml`](deny.toml): RustSec advisories, permissive licences only, crates.io as the only source |
| macOS | CPU and MLX backend builds and tests, plus the CubeCL and f16/bf16 GPU kernel tests on the runner's **Metal GPU** |

The Linux runner has no GPU, so the macOS job is where GPU code is tested in CI.
Metal fuses multiply-adds and flushes subnormals, so arithmetic that is exact on
your desktop GPU can differ there by an ulp. Write GPU code and tests with that in
mind (see the GPU recipe).

## Recipes

### Fixing a bug

1. Reproduce it in a test that fails. Put it in the crate's `tests/` directory
   (integration test) or in the module's `#[cfg(test)]` block.
2. Name the test after the behaviour that must hold, e.g.
   `warp_samples_an_edge_coordinate_off_by_an_ulp`, not `test_bug_42`.
3. In the test's doc comment, say what was wrong and how it showed up — future
   readers need to know why the test exists.
4. Fix the code. Run the failing test, then the whole workspace.
5. In the PR and commit message: the symptom, the root cause, the fix, and the
   before/after evidence (the failing assertion message is ideal).
6. Add a row to [docs/bug-log.md](docs/bug-log.md).

### Matching a reference library (parity)

When a function mirrors a reference (`cv2.GaussianBlur`, Open3D's
`remove_statistical_outlier`, `scipy.signal.decimate`, …):

1. Generate reference values with the reference library on deterministic inputs
   (seeded generators or closed-form images). Include awkward cases: borders,
   odd sizes, ties, empty input, NaN/inf where the reference defines behaviour.
2. **Store the reference values in the test** (inline arrays or hex, or a small
   fixture next to the test) so CI does not need Python. Commit the script that
   produced them next to the fixture, with the library version in a comment.
   Existing examples:
   - `crates/imgproc/tests/clahe_opencv_parity.rs` — cv2 output stored as hex
   - `crates/features/tests/fixtures/markers/generate.py` — fixture generator
   - `crates/hal/tests/fixtures/generate_fixtures.py` — `.npy` fixtures
   - `parity/` — the cross-library harness (`python3 parity/parity_filters.py`)
3. Compare exactly where the computation allows it; otherwise state the tolerance
   and why (for example "cv2 computes in fixed point, ±1 LSB").
4. If you deliberately keep a difference, document it in
   [docs/PARITY.md](docs/PARITY.md).

### Making something faster

1. Find the hot path with evidence: a criterion benchmark (`crates/*/benches/`,
   root `benches/`) or a profiler. Add a focused benchmark if none covers it.
2. Measure before and after with the same build settings, twice, on a quiet
   machine. Report medians and the spread; ignore changes inside the noise.
3. Prove results are unchanged: existing tests plus, if needed, a test comparing
   the new path against the old one (bit-for-bit where the contract is exact).
   Parallel reductions must not change summation order where results depend on it.
4. Put the numbers in the commit message and the PR.

Benchmarks: `cargo bench -p <crate>` (or `--bench <name>`); `cargo bench
--workspace --no-run` must keep building.

### Adding or changing a GPU kernel

GPU code lives in `crates/hal`: WGSL shaders in `crates/hal/shaders/`, host code in
`crates/hal/src/gpu_kernels/`, entry points in `crates/hal/src/gpu/`.

- **Match the CPU backend.** Each GPU op has a CPU counterpart, and tests compare
  them (`tests/gpu_parity_tests.rs`, `crates/hal/tests/gpu_*.rs`). Many kernels
  are bit-exact with the CPU; keep them that way.
- **Integer division.** Every integer `/` and `%` must be listed in the audit
  (`crates/hal/tests/wgsl_int_div_audit.rs`, documented in
  [docs/wgsl_int_div_audit.md](docs/wgsl_int_div_audit.md)) with a reason its
  divisor cannot be zero. The test fails on unlisted ones and on any shader
  `naga` rejects, and prints the row to add.
- **Bookkeeping tests.** Adding or removing a `.wgsl` file updates the count in
  `crates/hal/tests/shader_bindings_limits.rs` and the element-type table in
  `tests/gpu_parity_tests.rs`.
- **Limits.** Stay within `wgpu::Limits::downlevel_defaults()` — notably four
  storage buffers per shader stage. Dispatch large 1-D ranges as 2-D grids so no
  axis exceeds 65,535 workgroups.
- **Portability.** WGSL atomics are relaxed and there is no cross-workgroup fence.
  Metal fuses multiply-adds and may flush subnormals. Avoid results that hinge on
  the last ulp (bounds tests on computed coordinates, exact float equality), or
  make them robust as the existing kernels do.
- **Empty input** must return the same result as the CPU, never panic (covered by
  `crates/hal/tests/gpu_empty_tensors.rs`).
- Run `cargo test -p cv-hal --no-fail-fast`, `cargo test --test gpu_parity_tests`,
  and the `half-precision` / `cubecl` feature tests if relevant.

### Adding a new operation

1. Open a feature request first if the API is new; say which reference function
   it mirrors and how it will be tested.
2. Put it in the crate that owns the domain (see
   [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). A new crate needs an agreed
   issue first, a layer in `tests/crate_layers.rs`, and the
   `[workspace.package]` fields; a crate may depend only on lower layers. Follow the reference library's
   semantics and parameter conventions (units, border defaults, coordinate
   conventions), and say in the doc comment which function it mirrors.
3. Validate inputs and return `Result` errors for bad input; public functions do
   not panic on user data.
4. Add parity tests as above, plus degenerate inputs (empty, single element,
   NaN/inf, mismatched lengths).
5. If it belongs in the HAL (`ComputeContext`), implement the CPU path first; a
   GPU path is optional and must match it.

## Conventions

- **Formatting:** `rustfmt` with the repository's [`rustfmt.toml`](rustfmt.toml)
  (Google's Rust style guide, 100 columns). Always run `cargo fmt --all`.
- **Edition 2024**, stable Rust. No nightly features.
- **Errors:** return `Result` with the crate's error type
  (`cv_core::Error` variants such as `InvalidInput`, `NotSupported`). Reserve
  `panic!`/`unwrap()` for internal invariants and tests.
- **Images and tensors:** multichannel tensors are **planar CHW**
  (`TensorShape::new(channels, height, width)`; element `c*H*W + y*W + x`).
- **Numerics:** follow the reference library's conventions (OpenCV degrees vs
  radians, pixel-centre conventions, border defaults) and say so in docs.
- **Determinism:** results must not depend on thread count or scheduling.
  RANSAC-style randomness takes a seed or documents that it is unseeded.
- **Documentation:** every public item has a doc comment saying what it does,
  its units, its failure modes, and which reference function it mirrors.
- **Dependencies:** prefer what the workspace already uses (see the root
  `Cargo.toml` `[workspace.dependencies]`). New dependencies need a reason in the
  PR; licences must be permissive and compatible with MIT (`deny.toml` lists them).

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org) with a narrative body:

```text
fix(hal): warp treats a coordinate an ulp outside the edge as on it

macOS CI failed per_plane_operations_process_every_channel: the GPU warp
of a 3-channel image wrote 0 at one pixel where the CPU sampled the source.
<what was wrong, why, and where>

<the fix, and why it is the right one>

<evidence: the failing assertion before, the passing result after,
benchmark numbers for perf changes>
```

- Types: `fix`, `feat`, `perf`, `refactor`, `test`, `docs`, `ci`, `build`,
  `chore`, `bench`.
- Scope: the crate without the `cv-` prefix (`hal`, `imgproc`, `features`, …).
- Subject: imperative or descriptive, lower case after the colon, no full stop,
  about 72 characters at most.
- The body explains *why* and gives the evidence; the diff already shows *what*.

## Branches and releases

| Branch | Purpose |
| --- | --- |
| `master` | Default branch. All pull requests target it. |
| `develop` | Mirrors `master`; maintainers merge `master` into it after each change lands. |
| `research/cubecl` | CubeCL experiments; kept in sync with `master` by maintainers. |

Contributors only need `master`; maintainers propagate merged changes to the
other branches and CI runs on all three. Versions follow
[Semantic Versioning](https://semver.org); while the version is `0.x`, minor
releases may contain breaking changes, which [CHANGELOG.md](CHANGELOG.md) calls
out. See [GOVERNANCE.md](GOVERNANCE.md) for how releases are cut.

## Issue labels

| Label | Meaning |
| --- | --- |
| `bug` | Wrong result, crash, hang or panic |
| `parity` | Differs from OpenCV / Open3D / SciPy / Matplotlib |
| `performance` | Slower than it should be |
| `enhancement` | New feature or API |
| `docs` | Documentation |
| `gpu` | Involves `cv-hal` GPU code or shaders |
| `good first issue` | Small, well-defined, with pointers to the code |
| `help wanted` | Maintainers would welcome a contributor taking it |
| `needs-triage` | New; a maintainer has not looked yet (set by the issue forms) |
| `needs-repro` | Waiting for a reproduction |
| `dependencies` | Dependency updates (set by Dependabot) |
| `crate:<name>` | The crate affected, e.g. `crate:imgproc` |

## Good places to start

Known open items, each small enough for a first PR and with a clear way to
verify it:

- **GPU parity:** the GPU Gaussian blur and Sobel paths in `cv-hal` ignore the
  border-mode argument (the CPU paths now default to OpenCV's reflect-101).
- **Colour:** `rgb_to_gray_simd` in `cv-hal` still uses float weights; OpenCV and
  `cv-imgproc` use 15-bit fixed point (`(9798 R + 19235 G + 3735 B + 2^14) >> 15`).
- **Features:** ORB suppresses FAST corners with its own 5-pixel rule rather than
  OpenCV's 3x3 non-maximum suppression (`fast_detect_opencv` already implements
  OpenCV's); USAC's local refinement keeps a refit model even when it scores worse.
- **Joint bilateral filter:** uses an L2 colour distance; OpenCV's colour
  bilateral uses L1.
- **Performance:** GPU `fast::extract_keypoints` waits on the GPU twice (count,
  then keypoints); readback staging buffers are allocated per call.
- **Docs:** examples for crates that have none (`crates/*/examples/`).

Check the issue tracker for `good first issue` and comment before starting, in
case someone is already working on it.

## License

RETINA is licensed under the [MIT license](LICENSE). By submitting a
contribution you agree that it is licensed under the MIT license too, without
any additional terms or conditions.

If your change copies or closely ports code or data from another project, keep
that project's licence compatible with MIT (permissive: MIT, BSD, Apache-2.0,
ISC, Zlib), say in the doc comment which file it follows, and add an entry to
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Copyleft (GPL, LGPL, AGPL)
code cannot be accepted.
