# GAP REPORT — Rust CV workspace at /home/Phoenix/RUST/RETINA

status: **complete**. Source left unmodified — this file is the only write.

Survey, read-only on source. Method note: counts below are **measured** with the
`grep -rc` command given in the brief; individual stub claims are verified by reading the body
unless marked `UNVERIFIED`.

Targets claimed by the workspace README/root crate: OpenCV, Open3D, Matplotlib, SciPy, VSLoc-RS.

---

## (a) Coverage table — public items vs `#[test]` functions, per crate

Method: `grep -rc "pub fn \|pub struct \|pub enum \|pub trait " <crate>/src` and
`grep -rc "#\[test\]" <crate>` summed per crate. **Measured**, not sampled.
`items/test` = public items per test; higher is worse. Worst first.

| crate | pub items | tests | items/test | note |
|---|---:|---:|---:|---|
| python | 96 | 0 | ∞ | no tests at all; PyO3 binding crate |
| examples | 1 | 0 | ∞ | harness crate, low risk |
| scientific | 0 | 12 | 0.0 | no `pub fn`/`pub struct` in `src` per grep — see note |
| localization | 54 | 13 | 4.2 | VSLoc-RS target; thinnest test coverage of real crates |
| video | 77 | 43 | 1.8 | Matplotlib target sits in `plot`, OpenCV video |
| plot | 90 | 26 | 3.5 | Matplotlib target |
| eval | 29 | 36 | 0.8 | |
| cli | 10 | 18 | 0.6 | |
| io | 56 | 322 | 0.2 | test-heavy |
| slam | 25 | 27 | 0.9 | Open3D/SLAM target |
| rendering | 66 | 45 | 1.5 | |
| 3d | 172 | 149 | 1.2 | Open3D target |
| pointcloud | 10 | 18 | 0.6 | Open3D target — very small surface |
| videoio | 16 | 16 | 1.0 | |
| registration | 26 | 59 | 0.4 | |
| sfm | 43 | 41 | 1.0 | |
| dnn | 8 | 11 | 0.7 | |
| distributed | 22 | 22 | 1.0 | |
| viewer | 22 | 18 | 1.2 | |
| geometry2d | 37 | 47 | 0.8 | |
| signal_proc | 42 | 36 | 1.2 | SciPy target |
| photo | 15 | 48 | 0.3 | |
| calib3d | 113 | 140 | 0.8 | |
| features | 189 | 150 | 1.3 | |
| imgproc | 190 | 169 | 1.1 | OpenCV target |
| 3d/optimize | 114 | 86 | 1.3 | SciPy target |
| hal | 259 | 233 | 1.1 | |
| core | 297 | 204 | 1.5 | |
| runtime | 244 | 139 | 1.8 | |
| math | 133 | 90 | 1.5 | SciPy target |

> Discrepancy vs the numbers in the task brief: the brief predicted `python` 80 / 0, `hal`
> 224 / 118, `imgproc` 190 / 166. Measured values here are higher for `hal` (259/233) and
> `python` (96/0). The difference is real and consistent with counting `pub trait` and
> `pub enum` as well; `imgproc` matches at 190/166→169. Treat the table above as the measured
> one. **Measured.**

---

## (b) Stubs, `todo!()` / `unimplemented!()`, and hardcoded-default returns

Workspace-wide count of `todo!(` + `unimplemented!(` in `crates/` + `src/`: **1** (`unimplemented!`,
`usac.rs:335`). **Measured.** `unreachable!` appears 2x, both as genuine "this arm cannot happen"
guards (`registration/mod.rs:441`, `runtime/observe/events.rs:162`) — read, not gaps.

The repo has clearly already been through one hardening round (see below): several former silent
stubs now return `Err`/`None`/empty **loudly** and say so in a doc comment. Those are listed in
(b2) as "closed", because a reader needs to know they were found and fixed rather than re-report them.

### (b1) Open stubs — verified by reading the body

| # | location | what it does | severity |
|---|---|---|---|
| 1 | `crates/photo/src/stitcher.rs:23-24` | `Stitcher::stitch()` does `if images.is_empty() { Ok(GrayImage::new(0,0)) } // Return first image as placeholder for Phase 4` then `Ok(images[0].clone())`. **Returns the input image, unmodified, as the "panorama".** `Ok()`, no warning, no log. `Stitcher` is public (`photo/src/lib.rs:52,60`) and `stitch` is its only method. | **HIGH — silent** |
| 2 | `crates/hal/src/gpu_kernels/pointcloud.rs:29-39` | `normals_cpu_analytic(points, k)` → `Vec::new()`. Doc admits "Not implemented". But its only caller `compute_normals_morton_gpu_or_cpu` (`:11-20`) uses it as the **silent CPU fallback**: if `GpuContext::global()` fails or the GPU kernel errs, the function returns an **empty Vec** with no error and no log — a point cloud of N points yields 0 normals and the caller cannot tell. | **HIGH — silent fallback** |
| 3 | `crates/hal/src/gpu_kernels/mod.rs:1238-1248` | `batch_nearest_neighbors` is documented "on GPU" but the body says `// Simplified: just return k nearest using brute force on CPU`. Takes `gpu: &GpuContext` and reads the buffer back to host (`read_buffer` + `pollster::block_on`) then loops on CPU. GPU in the name and signature, CPU in the body, no record returned. | **HIGH — silent downgrade** |
| 4 | `crates/hal/src/gpu_kernels/mod.rs:1619-1624` | `simplify_mesh(_gpu: &GpuContext, ...)` — the GPU parameter is `_gpu` (unused); body is uniform voxel-grid decimation on CPU. Doc calls it "Simplify mesh on GPU". Other GPU kernels in this file do use `gpu`; this one only borrows the name. | MED — silent downgrade |
| 5 | `crates/3d/src/gpu/mod.rs:722-732` | `voxel_to_point_normal_transfer` → `Vec::new()`. Doc states the earlier version returned a zero vector per point and this now returns empty "rather than fabricated normals" — so it is a **deliberate, documented removal**, but it is still a public API in the `gpu` module that produces nothing. | MED — documented |
| 6 | `crates/3d/src/odometry/mod.rs:445-456` | `compute_hybrid` → `None`; doc explains it replaces a former fabricated identity-transform + `fitness: 0.0`. Consequence: `OdometryMethod::Hybrid` cannot succeed. Public enum variant resolves to guaranteed failure. | MED |
| 7 | `crates/features/src/usac.rs:327-339` | `scorers::score_msac` body is `unimplemented!(...)`. Reachable: `score_magsac` (`:342-366`) calls `score_msac` inside its loop over `sigma_max`, so **`score_magsac` panics too**; both are `pub`. This is the one true `unimplemented!` in the workspace. | **HIGH — panic** |
| 8 | `crates/photo/src/hdr.rs:132` | Response-curve recovery is "a simplified Debevec approach"; caller-visible as `tonemap_*`/HDR merge. | LOW-MED |

### (b2) Formerly-silent stubs that now fail loudly (closed — re-reported only so they are not re-opened)

- `crates/calib3d/src/essential.rs:35` — Nistér 5-point solver returns `Err("... is not implemented")`;
  test at `:72-75` asserts it must not return matrices. **Capability still absent**, but honestly.
- `crates/hal/src/gpu/compute_context_impl.rs:473,548,563` — GPU `detect_objects`,
  `triangulate_points`, `find_chessboard_corners` return `Err(NotSupported("...use CPU backend"))`.
- `crates/optimize/src/sparse.rs:109,263` — MLX SpMV / transpose-SpMV return `Err("not implemented yet")`.
- `crates/3d/src/gpu/mod.rs:728` and `hal/.../pointcloud.rs:27` — see (b1) #2, #5.
- `crates/3d/src/odometry/mod.rs:456` — `compute_intensity` returns `None` for the same reason.

---

## (c) Documented-but-absent behaviour

Each entry: the doc/builder promises X, the body does Y. Verified by reading the body.
The Plot3D `.grid(bool)` precedent cited in the brief is **closed** — `Figure::grid` is
read by the exporter at `crates/plot/src/export.rs:72` and `Figure::legend` at `:216`.
Its sibling on the same struct is **still open**:

### (c1) `cv-plot` — `subplot()` accepts an index and discards it; no panel layout exists
- `crates/plot/src/chart.rs:173` — `pub fn subplot(&mut self, rows: usize, cols: usize, _index: usize) -> &mut Self`.
  Doc: "Add a new subplot". Body: `while self.subplots.len() < rows * cols { push(SubPlot::default()) }` — `_index`
  is **unused**, and there is no "current subplot" field on `Figure` (`chart.rs:84-91`: title, width, height,
  subplots, legend, grid).
- `chart.rs:124,137,148` — `add_series`/`scatter`/`bar` all append to `self.subplots.last_mut()`. So after
  `fig.subplot(2, 2, 0)`, series land in subplot **3**, not 0. `subplot(2,2,0)` and `subplot(2,2,3)` are
  indistinguishable in output.
- `crates/plot/src/export.rs:118-124` — `to_svg` iterates subplots but draws every series into the **single**
  plot area (`margin_left` + global `min_x..max_x` bounds computed across *all* subplots at `:52-65`).
  `grep -n "panel|offset_x|\brow\b|\bcol\b" export.rs` returns **nothing**: there is no row/col layout anywhere.
  Net: `rows`/`cols` have no visual effect either beyond padding a Vec. A 2x2 figure renders as overlaid series
  in one axes.

### (c2) `cv-video` — two `MOG2::new` parameters are inert
- `crates/video/src/mog2.rs:96,117` — `detect_shadows` is stored as `_detect_shadows` (underscore = deliberately
  unread) and the doc says "**Currently has no effect**". Only "reads" are in tests (`:566,575`).
- `mog2.rs:37,55,114` — `var_threshold` doc'd as "Variance threshold for detecting shadows … **Not currently used
  but reserved for future implementation**"; it is stored and forwarded into `Mog2Params` (`:254`) but shadow
  detection does not exist, so both parameters of the public constructor are decoration. `MOG2::new` doc also
  admits "Invalid parameters are silently accepted".
- This is an **OpenCV-named** API (`createBackgroundSubtractorMOG2`) whose shadow-detection contract is absent
  with no error and no flag on the output.

### (c3) `cv-hal` — `compute_distance_field` is neither signed nor a surface distance
- `crates/hal/src/gpu_kernels/mod.rs:1787-1788` — doc: "Compute distance field on GPU - **signed** distance to
  mesh surface".
- Body: initialises `vec![f32::MAX; total_voxels]` (`:1800`) and keeps `min_dist = min_dist.min(d)` over the
  **three vertices** of each face (`:1867` comment: "distance to triangle (simplified: distance to closest
  vertex)"). No inside/outside test and no negation anywhere in the function — the value is never negative,
  so it is an **unsigned** field; unfilled voxels stay at `f32::MAX`, and a point may sit below the true surface.
  Both `signed` and `surface` in the doc are unearned.

### (c4) `cv-hal` — `build_kdtree` returns Morton codes, not a tree
- `crates/hal/src/gpu_kernels/mod.rs:1172-1175` — doc: "Build KDTree on GPU (parallel construction) - simplified
  version / Returns sorted Morton codes as a basic spatial index".
- `GpuKDTree` (`:1436-1441`) has only `nodes_buffer`, `points_buffer`, `num_points`, `device` — no node
  structure, no split planes, no tree. Nearest-neighbour queries against it are answered by the CPU brute-force
  loop in `batch_nearest_neighbors` (see (d1)), so the "index" is not used as an index.
- **Bug found while verifying this** (`:1231-1235`): the two public buffers are populated **swapped against their
  names** — `nodes_buffer: points_buf, points_buffer: morton_buf`. `batch_nearest_neighbors` (`:1248`) reads
  `kdtree.nodes_buffer` into `points_data`, which works only because of the swap. Any other consumer of the
  `pub` fields gets points where it expects Morton codes and vice versa.

### Concurrent modification notice (affects this section's line numbers)

While this survey was being written, **another writer changed the workspace under me** at 06:04:
`crates/viewer/src/native_viewer/render.rs`, `crates/viewer/src/native_viewer/point_cloud.wgsl`
(both now show as `M` in `git status`) and a new untracked `crates/viewer/tests/sprite_pixels.rs`.
I did not make these changes — my only write is this file.

Consequences for a reader:
- The `render.rs` line numbers in (c5) (`:60`, `:78`, `:148`) and in the (f) Open3D bullet were read
  from the **pre-change** revision; the diff inserts a ~35-line `fn vertex_layout()` at `render.rs:50`,
  so those references have shifted. The *substance* of (c5) is unaffected: `viewer/src/lib.rs:3-13`
  is **not** in the modified set, so the stale "does not render anything" doc is still present and
  still wrong, and `native_viewer.rs:114-117` (also unmodified) still builds the renderer.
- The concurrent change is itself in the category this report hunts: the diff replaces
  `step_mode: VertexStepMode::Vertex` with `Instance` and documents that with `Vertex`, "every
  instance read the first six entries of the buffer, so each sprite was assembled from points 0..6
  (six different points per quad), and a cloud of fewer than six points was not a drawable
  configuration at all". That is a silent-wrongness bug being **fixed** by someone else, not one I
  found; recorded here so it is not double-counted or re-investigated.

### (c5) `cv-viewer` — the crate doc says it renders nothing; **that doc is stale and wrong**

**Correction of my own first pass.** `crates/viewer/src/lib.rs:3-13` states "**It does not render
anything.** `native_viewer` opens an egui window that reports each cloud's point and normal count;
there is no 3D viewport, no wgpu pipeline, and no use of `CreationContext::wgpu_render_state` …
nothing visual took its place." Every clause of that is contradicted by the file next door:
- `viewer/src/native_viewer.rs:114-117` — `renderer: cc.wgpu_render_state.as_ref().and_then(PointCloudRenderer::new).map(Arc::new)`
  — i.e. exactly the `CreationContext::wgpu_render_state` use the doc says does not exist.
- `viewer/src/native_viewer.rs:89` — `renderer: Option<Arc<PointCloudRenderer>>`; `:145-210` upload/recolour
  clouds; `:577-597` clear buffers; `:634` draw; `:370` reports `point_count`.
- `viewer/src/native_viewer/render.rs:59-150` — a real `wgpu` pipeline + `point_cloud.wgsl` shader, vertex
  attribute layouts, alpha-blended `ColorTargetState`.
- `viewer/src/native_viewer.rs:68-70` — "Every frame this window sets a camera on that renderer".
- `viewer/src/native_viewer.rs:831` — `pub use render::PointCloudRenderer;`.
- Corroborated by dependencies: `viewer/Cargo.toml` lists `wgpu`, `egui-wgpu`, `epaint`, `image`, and
  `pollster` (dev), and the committed `Cargo.lock` was missing `image`/`pollster` for this package
  (see §Build hygiene).

So this is a **documentation gap in the opposite direction to the Plot3D one**: the code has a working
3D viewport and the doc claims it does not, which discourages use of an existing capability and
contradicts `README.md:32` ("Viewer - 3D visualization") — here the README is the accurate document.
Genuine gaps that remain in this area are narrower: rendering is wgpu-only (`:112-113` — on the glow
path the renderer is `None` and the window says so, which is honest), and the whole crate is not
re-exported by the root crate (§Unreachable API).
- Compiler-measured dead field: `viewer/src/native_viewer/render.rs:60` `format` is never read. This is
  *benign* — `new()` binds a local `let format = state.target_format` at `:78` which is the value used in
  the `ColorTargetState` at `:148`; the field is a dead copy of it.

### (c6) `cv-optimize` — `Isam2` config is accepted and ignored
- `crates/optimize/src/isam2.rs:911-917` — "Legacy Isam2 interface for backward compatibility with Python
  bindings". `#[allow(dead_code)] optimize_on_update: bool` — the attribute is compiler-proof that the field is
  **never read**; `with_config(optimize_on_update, _batch_optimize)` takes a second argument `_batch_optimize`
  that is likewise discarded. `Isam2::new()` sets `true` for a flag nothing consults.

### (c7) `cv-3d` — two "different" normal algorithms ignore their distinguishing parameters
- `crates/3d/src/gpu/mod.rs:718-721` — `voxel_based_normals_simple(points, _voxel_size)` → body is
  `compute_normals(points, 30)`: `_voxel_size` unused, k hardcoded to 30.
- `crates/3d/src/gpu/mod.rs:734-741` — `approximate_normals_simple(points, k, _epsilon)` → `compute_normals(points, k)`:
  `_epsilon` unused. The two public entry points collapse to the same implementation with no way for a caller to
  set the parameter they passed.
- `crates/3d/src/gpu/mod.rs:34` — `// TODO: Actual GPU implementation in hal`; the whole `3d::gpu` module is a
  wrapper around CPU code (see also (b1) #5).

### (c8) Other dead-field/dead-item evidence (compiler-measured unless noted)
- `crates/video/src/optical_flow.rs:625` — `PolyCoeffs` carries `F`, "computed by full 6x6 solve, reserved for
  future use"; never read.
- `crates/hal/src/gpu_sparse.rs:12` — `#[allow(dead_code)] pub struct GpuSparseMatrix` (whole GPU CSR type dead).
- `crates/3d/src/raycasting/mod.rs:241` — `RayHitInfo { distance, point, normal, barycentric }` dead: no caller
  receives hit details.
- `crates/features/src/akaze.rs:534` — `to_cpu_f32` dead (AKAZE never converts GPU→CPU).
- `crates/imgproc/src/edges.rs:31` — `kernel_from_1d` dead (Scharr is hand-built).
- `crates/calib3d/src/essential_fundamental.rs:313` — `sample_unique_indices` dead.
- `crates/registration/src/registration/global/ransac.rs:541` — `random_sample` dead.
- `crates/plot/src/three_d.rs:731` — `bounding_box_all` (union bbox over several clouds) dead → multi-cloud 3D
  plots are not jointly scaled. **Compiler-measured**, meaning not read: not a functional bug by itself.
- `crates/features/src/orb.rs:17-24` — three spec ORB params (`first_level`, `edge_threshold`, `wta_k`)
  deliberately absent; the comment states they were removed as dead fields rather than silently kept.

---

## (d) Silent capability downgrades

Distinguished from (b) by *no error and no flag returned* — the caller cannot detect the downgrade.

| # | location | downgrade | detectable by caller? |
|---|---|---|---|
| d1 | `crates/hal/src/gpu_kernels/mod.rs:1238-1248` | `batch_nearest_neighbors` — "on GPU" documented, body is `// Simplified: just return k nearest using brute force on CPU`, including a blocking `read_buffer(...)` + `pollster::block_on` device→host copy. Accepts `gpu: &GpuContext`, ignores it. | **No** |
| d2 | `crates/hal/src/gpu_kernels/mod.rs:1619-1624` | `simplify_mesh(_gpu: &GpuContext, …)` — GPU handle accepted and unused; CPU voxel-grid decimation. | **No** |
| d3 | `crates/hal/src/gpu_kernels/pointcloud.rs:11-20,29-39` | `compute_normals_morton_gpu_or_cpu` — if `GpuContext::global()` fails *or* the GPU kernel errors, it silently returns `normals_cpu_analytic(...)` = `Vec::new()`. N points in, 0 normals out, `Ok`-shaped result, no log. The name promises a CPU fallback; the fallback is empty. | **No** |
| d4 | `crates/runtime/src/orchestrator.rs:395-414` | `best_runner()` / `try_best_runner()` fall back to `RuntimeRunner::Sync(reg.default_cpu()…)` with no indication. The `_gpu_wait` siblings (`:421-444`) *do* return `(runner, is_gpu)` — so the information exists but the primary entry point drops it. `3d/src/gpu/mod.rs:749` (GPU stereo matching) calls `best_runner()` and cannot tell whether it got a GPU. | **No** (the API to know exists elsewhere) |
| d5 | `crates/runtime/src/orchestrator.rs:798` | `// Any GPU backend matches a GPU request for now` — device selection matches *any* GPU to a GPU-typed request, so e.g. a backend-specific request is satisfied by a different GPU backend. | **No** |
| d6 | `crates/features/src/usac.rs:342-366` | `score_magsac(model, points, threshold, sigma_max)` — `threshold` is **unused** (compiler: `warning: unused variable: 'threshold'`, `usac.rs:345`). The loop sets its own `t = s as f64` from `sigma_max`; a caller's calibration is discarded. It also panics via (b1) #7. | **No** |
| d7 | `crates/hal/src/gpu/compute_context_impl.rs:607` | Morphology GPU path proceeds on an assumption in a comment ("Let's assume for now morphology on GPU is f32-compatible or we use casts") and has no dtype check before the `GpuStorage<u8>` downcast; failure surfaces only as a late `"Failed to downcast GPU result"`. | Partially (late error) |
| d8 | `crates/hal/src/gpu_kernels/undistort.rs:140` | `_ => 1, // Default to bilinear for now` — an unrecognised interpolation mode silently becomes bilinear instead of erroring. | **No** |
| d9 | `crates/photo/src/stitcher.rs:23-24` | `Stitcher::stitch` returns `images[0]` (see (b1) #1) — a stitched panorama that is not stitched. | **No** |
| d10 | `crates/optimize/src/sparse.rs:109,263` / `hal/.../compute_context_impl.rs:473,548,563` | The *opposite* pattern, for contrast: MLX SpMV and three GPU ops return `Err(NotSupported("…use CPU backend"))` rather than downgrading silently. **These are correct**; listed so the two patterns are not confused. | Yes — loud |

---

## (f) Capability notes against the five targets

Only claims that can be pointed at in the source. Status: **absent** / **partial** /
**present-but-unverified**. "Present" is not asserted anywhere — I did not run the test suite
(see §Survey coverage), so nothing below is claimed to be correct, only to exist and to be
non-trivial.

### OpenCV → `imgproc`, `video`, `photo`, `features`, `calib3d`, `dnn`, `videoio`
- **Present (read):** imgproc filters/morphology/colour/threshold/histogram/contours/moments/template
  matching; features ORB, FAST, Harris, GFTT, BRIEF, HOG, SIFT, AKAZE, `haar_cascade`,
  `aruco_tables`, `charuco`, `markers`; calib3d PnP (`pnp.rs`), DLT, calibration, undistort,
  chessboard; photo inpaint (Telea + NS), denoise, HDR/tonemap; video optical flow, tracking,
  MOG2; videoio FFmpeg + v4l; dnn ONNX via `tract`.
- **Absent — `calib3d`:** Nistér 5-point essential-matrix solver. `essential.rs:35` returns
  `Err("Nistér 5-point solver is not implemented…")`. `find_essential` exists
  (`essential_fundamental.rs`) but not the 5-point path; therefore `recover_pose`-style workflows
  from a 5-point essential matrix are unavailable. **Loud.**
- **Absent — `cv-dnn` is a thin shell.** `crates/dnn/src/` is 2 files; the entire public surface is
  `image_to_blob`, `blob_to_image`, and `DnnNet::{load, input_shape, forward, preprocess, input_chw}`
  (`lib.rs:70-319`). No layers, no ONNX graph inspection, no detection/segmentation heads. The
  module-level example (`lib.rs:21`) is ````ignore` — never compiled or executed — and
  `forward`'s example (`:161`) is `no_run`, so the ONNX path has **no doctest coverage**; the only
  unit tests are for the blob converter (`blob.rs:53-120`) and one in `lib.rs:372`.
- **Partial — `imgproc::segmentation::grab_cut`** (`segmentation.rs:243`): K-means colour prototypes +
  4-connected smoothness, *not* GMM + graph cut, and the doc says so. Same signature and constants
  (`GC_BGD`…`GC_PR_FGD`, `GrabCutMode`) as OpenCV, so a caller migrating code gets a different
  algorithm with an identical name.
- **Partial — contours** (`contours.rs:457`): "simplified Suzuki-Abe"; hierarchy output is not the
  OpenCV contract.
- **Partial — `video::mog2`**: shadow detection absent, `detect_shadows`/`var_threshold` inert (see (c2)).
- **Partial — photo stitching**: `photo::stitcher::Stitcher::stitch` is a stub returning image 0 (see (b1) #1)
  while a real two-image pipeline exists at `imgproc/src/stitching.rs:336 stitch_pair`.
  Two stitching APIs, one of which lies.
- **Unreachable:** `videoio`, `photo`, `dnn` are not re-exported by the root crate (see §Unreachable API).

### Open3D → `3d`, `registration`, `pointcloud`, `rendering`, `viewer`
- **Present (read):** TSDF volume with real marching cubes (`3d/src/tsdf/mod.rs:323` uses
  `MC_EDGE_TABLE` at `:861` inside `marching_cubes_cell`), mesh reconstruction entry points for
  Poisson / BPA / Alpha Shapes (`3d/src/mesh/reconstruction/{poisson,ball_pivoting,alpha_shapes}.rs`),
  ICP + global registration (`registration`), `voxel_down_sample`, `estimate_normals`,
  `orient_normals`, `remove_statistical_outliers`, `remove_radius_outliers`, `segment_plane`,
  `cluster_dbscan`, **`compute_fpfh_feature`** (`pointcloud/src/point_cloud.rs:766`).
  *(Correction: an early grep for `fn fpfh` returned ABSENT — wrong, and I would have reported a false
  gap. The function is named `compute_fpfh_feature` and is real, with tests at `:1071` and
  `tests/degenerate_inputs.rs:197`. Recorded because the brief warns about exactly this failure mode.)*
- **Partial — Poisson**: `mesh/reconstruction/poisson.rs:3,51` — "simplified … regular grid … octree is"
  (the real Poisson uses an adaptive octree).
- **Stub — Delaunay**: `mesh/reconstruction/delaunay.rs:1` — file header literally says
  "Delaunay-based reconstruction (**placeholder**)". **UNVERIFIED** how deep the placeholder goes;
  I read the header only.
- **Present (read) — viewer rendering, contrary to its own doc**: a real wgpu viewport with camera,
  point upload and a WGSL pipeline (`viewer/src/native_viewer.rs:114-117`, `render.rs:59-150`);
  the crate's header doc denying this is stale (see (c5)). wgpu-only: `None` on the glow path.
- **Absent — `3d::gpu` normal transfer / hybrid odometry**: (b1) #5, #6.
- **Partial — GPU mesh ops**: `hal::gpu_kernels::simplify_mesh` and `batch_nearest_neighbors` are CPU
  bodies under GPU names (see (d1), (d2)); `compute_distance_field` is unsigned (see (c3)).
- **Small surface for a target:** `pointcloud` has 10 public items and 18 tests — the Open3D
  replacement's dedicated point-cloud crate is an order of magnitude smaller than `3d`.

### Matplotlib → `plot`
- **Present (read):** line/scatter/bar series (`chart.rs`), SVG + HTML export (`export.rs:9 to_svg`
  → `:301 save_svg`, `:249 to_html` → `:310 save_html`),
  colour/`Color` API and style (`style.rs`), 3D scatter/point-cloud plot (`three_d.rs`) producing
  the `plot_3d.html`/`.svg` artifacts at the repo root.
- **Absent — PNG export.** `export.rs:325-329` `save_png` returns
  `Err(PlotError::Export("PNG export is not supported. Use save_svg() or save_html() instead."))`.
  Loud and honest, but it means the commonest Matplotlib call shape (`savefig("x.png")`) has no path.
- **Broken — subplot layout** (see (c1)): `subplot(rows, cols, index)` ignores `index`, series always
  go to the last subplot, and the exporter has no panel layout at all. Any figure with more than one
  panel is mis-rendered without an error.
- **Absent — plotting types:** no `imshow`/heatmap, no `errorbar`, no `contourf`, no statistics
  box/violin plots (greps for those names return nothing in `crates/plot`). `hist` exists only as
  `imgproc`'s histogram, not as a plot type.
- **Present-but-unverified:** nothing in `crates/plot` is exercised by a test count above 26 for 90
  public items; `plot` is the worst-tested advertised module apart from `python`/`localization`.

### SciPy → `math`, `signal_proc`, `pointcloud`, `geometry2d`, `optimize`, facade `scientific`
- **Present (read, by module name in `crates/math/src/` and `scientific/src/lib.rs:12-40`):**
  `stats`, `special`, `linalg`, `sparse`, `interpolate`, `integrate`, `spatial`, `geometry`, `jit`;
  `signal_proc` (FFT, filters, windows, spectral, wavelets); `geometry2d` (computational geometry).
  133 public items in `math`, 42 in `signal_proc` — the largest single-crate surface among the
  targets after `core`/`hal`.
- **Partial — `scientific` is a façade with no code of its own**: `scientific/src/lib.rs` is pure
  `pub use` re-exports of `cv-math`/`cv-geometry2d`/`cv-signal`/`cv-pointcloud`; its 0-item count in
  the coverage table is an artefact of counting `pub fn|struct|enum|trait` only, **not** an empty
  crate. It is also not re-exported by the root crate.
- **Partial — sparse solvers**: `optimize/src/sparse.rs` has a `CgSolver` whose earlier behaviour
  "substituted" a zero step for a real iterate (see the assertion text at `sparse.rs:490-497`); that
  is fixed, but a test written to pin the fix — `cg_iteration_cap_is_not_reported_as_a_solution`
  (`sparse.rs:501`) — **has no `#[test]` attribute and never runs**. The behaviour it was written to
  pin is therefore unpinned. Same defect at `features/src/orb.rs:1611`. **Compiler-measured**
  ("function … is never used").
- **Absent:** no `scipy.optimize`-equivalent top-level minimizer API was located; `optimize` is
  factor-graph/ISAM2/sparse-linear oriented. **UNVERIFIED** — I did not enumerate `optimize`'s full
  surface (114 items) for a `minimize`/`least_squares` analogue; a targeted search there is the right
  next step.
- `hal`'s MLX sparse path is explicitly unimplemented (`sparse.rs:109,263`), loud.

### VSLoc-RS → `localization`
- **Present (read):** full composition is real — retrieval `Database` with optional BoW
  (`cv_features::retrieval::BowDatabase`) or descriptor-match ranking, 2D–3D lifting through
  landmarks, ratio-test matching, `cv_calib3d::pnp::solve_pnp_ransac` + refinement, and
  `evaluate_localization` aggregating translation/rotation error + success rate
  (`localization/src/lib.rs:1-40`). `synthetic`/`benchmark` are feature-gated, so the
  synthetic-scene and benchmark surface is **not built by default** (both are no-ops without the
  feature — the crate's own runnable demo is behind it).
- **Weakest-tested real crate:** 54 items / 13 tests = 4.2 items per test — worse than any other
  non-empty crate except `python` (∞) and `examples` (∞). Of those items, a large share are
  getters and feature-gated synthetic helpers, so the *effective* localization surface
  (`Database`, `Localizer`, `evaluate_localization`) is small — but it is the target with the least
  test pressure per unit of claimed capability.
- **Partial:** pose refinement and PnP correctness are inherited from `calib3d`, which is the crate
  carrying the absent 5-point solver; localization does not itself depends on that path
  (it uses PnP), so the target is not blocked by it. **Present-but-unverified** end-to-end; I did not
  execute the `synthetic_localization` example.

---

## Unreachable / un-integrated public API

- `src/lib.rs` re-exports **11** of the workspace's **30** member crates (`d3`, `calib3d`, `core`,
  `features`, `hal`, `imgproc`, `optimize`, `runtime`, `sfm`, `slam`, `video`). **Measured** by
  reading `src/lib.rs` against the `members` list in the root `Cargo.toml`.
  Not reachable via `rust_cv_native::`: `videoio`, `photo`, `io`, `dnn`, `rendering`, `registration`,
  `plot`, `viewer`, `math`, `geometry2d`, `signal_proc`, `pointcloud`, `distributed`, `eval`, `cli`,
  `localization`, `scientific`, `examples`. A user of the root crate cannot reach the Matplotlib
  replacement (`plot`), the Open3D registration crate, the SciPy surface (`math`), or the VSLoc-RS
  target (`localization`) without a second `Cargo.toml` entry. The root `Cargo.toml` only declares
  `scientific` and `registration` as **dev**-dependencies.
- `crates/hal/src/gpu_sparse.rs` — `GpuSparseMatrix` carries `#[allow(dead_code)]`: the GPU CSR type
  is never used (see (c8)).
- `crates/viewer` — advertised in the README and **does** render (see (c5)), but is reachable only by
  a direct dependency, and its own crate doc wrongly tells users it cannot.

## Build hygiene

- **The committed `Cargo.lock` did not match the manifests.** `cargo check` rewrote it, adding exactly
  two edges to the `cv-viewer` package: `image` (a real `[dependencies]` entry) and `pollster` (a
  `[dev-dependencies]` entry). Both were already in `crates/viewer/Cargo.toml`; the lockfile had not
  been regenerated. Anyone building the workspace gets a dirty `Cargo.lock` as their first action.
  This is the only working-tree change produced **by this survey**, and it is a side effect of
  `cargo check`, not an edit: at the time of checking, `git status --short -- '*.rs'` = 0 lines.
  I did not revert it, because that would require `git checkout`, which the brief forbids.
  (Other files have since changed — see the concurrent-writer note below; those are not mine.)
- Workspace builds clean: `cargo check --workspace --all-targets` exits 0 in ~29 s with warnings only
  (the warnings are the evidence used in §c and §d).
- **Concurrent writer active.** The statement "no `.rs` file is modified" was true when I checked it
  (`git status --short -- '*.rs'` = 0) and is no longer true: at 06:04 another writer modified
  `crates/viewer/src/native_viewer/render.rs` and `point_cloud.wgsl` and added
  `crates/viewer/tests/`. I made none of those changes. See the notice in (c). Any line reference in
  this report pointing into `crates/viewer/src/native_viewer/render.rs` should be re-resolved against
  the current revision.

## Method, and what this survey did NOT cover

**Measured:** the coverage table (§a); the 1 `unimplemented!` / 0 `todo!` counts; the 57-item
admission grep and its per-crate distribution; all `cargo check --workspace --all-targets`
warnings quoted here (dead fields, unused params, never-used functions); the root re-export count.

**Read-only (no execution):** every stub/doc-body claim has a `file:line` and was read, but the
**test suite was not run** — so nothing here asserts that non-stub code is *correct*, only that it is
not a stub. Where a body was not opened I have marked it `UNVERIFIED` (delaunay's depth,
`optimize`'s minimizer surface, `plot`'s 3D DOM output).

**Not covered at all** (visible gaps in this survey):
1. `crates/core` (297 items) and `crates/hal` (259 items) internals — touched only where admissions/warnings
   pointed; the HAL CPU compute context is ~4000 lines and I read ~60 of them.
2. `crates/runtime` (244 items) beyond device selection.
3. `crates/io` (56 items / 322 tests), `crates/distributed`, `crates/eval`, `crates/cli`,
   `crates/rendering` (Gaussian splatting), `crates/signal_proc`, `crates/math`, `crates/geometry2d`.
4. Numerical correctness of anything: no golden-file or parity comparison against OpenCV/Open3D was
   performed, and the repo's own `tests/gpu_parity_tests.rs` was not executed.
5. The 8 test-only "never used" functions (e.g. `hal/tests/helpers/mod.rs`, `sfm`'s
   `compute_residuals_for_param`) are dead test helpers; I checked two of them in detail and both
   were benign, so I have not enumerated the rest.
6. `benches/` and `crates/examples` beyond the one admission comment.
7. Two false positives I chased and am **not** reporting as gaps, recorded so they are not
   re-investigated: `sfm/mapper.rs:691` (`loop_closures` unused binding — the value *is* reported via
   `report.loop_closures`), and `sfm/bundle_adjustment.rs:411` (`compute_residuals_for_param_local`
   returns a "dummy second for signature parity", but the caller passes *different* `p_plus`/`p_minus`
   states at `:387-389`, so the finite-difference Jacobian is not identically zero).


Workspace total: **57** matches for `TODO|FIXME|XXX|HACK|for now|simplified|not implemented|placeholder`
under `crates/*/src`. **Measured.** Distribution: hal 16, 3d 7, runtime 6, imgproc 4, geometry2d 4,
features 4, calib3d 4, photo 3, registration 2, optimize 2, video/slam/sfm/math/examples 1 each;
**viewer, videoio, signal_proc, scientific, rendering, python, pointcloud, plot, localization, io,
eval, dnn, distributed, core, cli: 0.** (Note: 4 of the geometry2d hits are a local variable named
`simplified` in a `simplify()` test, not an admission — real count 53.)

**hal (16)** — the densest cluster, all in the GPU layer:
- `gpu/compute_context_impl.rs:473,548,563` GPU detect_objects / triangulate_points / find_chessboard_corners → `NotSupported` (see b2).
- `gpu/compute_context_impl.rs:607` "Let's assume for now morphology on GPU is f32-compatible or we use casts" — dtype assumption inside the GPU morphology path.
- `gpu/compute_context_impl.rs:1471` "GPU resize: Nearest is not implemented" (loud).
- `gpu/compute_context_impl.rs:1700` "GPU subtract only supports f32 for now" (loud).
- `gpu_kernels/mod.rs:1017` "max vertices = all voxels (simplified)" — GPU marching-cubes output sizing.
- `gpu_kernels/mod.rs:1172` `build_kdtree` "simplified version / Returns sorted Morton codes" — see below.
- `gpu_kernels/mod.rs:1238` batch NN "simplified brute force" → CPU (see b1 #3).
- `gpu_kernels/mod.rs:1619` `simplify_mesh` "on GPU - simplified" (see b1 #4).
- `gpu_kernels/mod.rs:1808` distance field "simplified brute-force".
- `gpu_kernels/mod.rs:1867` "distance to triangle (simplified: distance to closest vertex)".
- `gpu_kernels/pointcloud.rs:27` normals CPU fallback not implemented (see b1 #2).
- `gpu_kernels/undistort.rs:140` `_ => 1, // Default to bilinear for now`.
- `gpu/mod.rs:578` "Create a simplified compute pipeline".
- `cpu/compute_context_impl.rs:3993` "Subtraction not implemented for this type".
- (2 remaining are the `Not implemented ...` doc headers for the same sites above.)

**3d (7)**: `odometry/mod.rs:456`, `gpu/mod.rs:34` "TODO: Actual GPU implementation in hal" (the
`gpu` module is a thin wrapper), `gpu/mod.rs:728`, `mesh/reconstruction/mod.rs:24` "PCA (simplified)",
`mesh/reconstruction/delaunay.rs:1` "Delaunay-based reconstruction (**placeholder**)" — the file
header itself, `mesh/reconstruction/poisson.rs:3,51` "simplified Poisson ... regular grid ... octree is".

**runtime (6)**: `orchestrator.rs:798` "Any GPU backend matches a GPU request for now" (device
selection is by loose match), `memory.rs:290,398,446` transfer `not implemented for {:?}`,
`memory_manager.rs:88` "Return to global pool for now", `pipeline/fusion.rs:711` "Custom fused kernel
(placeholder)".

**features (4)**: `usac.rs:316` (see b1 #7), `haar_cascade/mod.rs:168` min_neighbors grouping
"simplified", `orb.rs:17` — three spec ORB params (`first_level`, `edge_threshold`, `wta_k`)
deliberately absent and previously dead fields.

**imgproc (4)**: `segmentation.rs:215,239` GrabCut is K-means + 4-connected smoothness, explicitly
**not** GMM + graph cut; `contours.rs:457,488` "simplified Suzuki-Abe" border following.

**calib3d (4)**: `essential.rs:35,72,75` Nistér 5-point absent; `project.rs:305` Jacobian by
**numerical differentiation** "for now".

**photo (3)**: `stitcher.rs:24` (see b1 #1), `hdr.rs:132` simplified Debevec, `inpaint.rs:260`
"simplified Navier-Stokes-like equation".

**registration (2)**: `registration/colored.rs:99` and `registration/mod.rs:644` "Compute jacobian
(simplified)" — both in ICP Jacobian assembly.

**optimize (2)**: `sparse.rs:109,263` MLX SpMV.

**Singletons**: `video/mog2.rs:96` `detect_shadows` "(false for now)"; `slam/tracking.rs:144`
a comment conceding a placeholder "even if physics is wrong"; `sfm/mapper.rs:3221` placeholder
keypoint index; `math/geometry.rs:291` (inside a test); `examples/src/bin/orb_matching.rs:69`
dummy mask in an example binary.

---

---

## Follow-up: the facade gap, verified and left for a deliberate decision

The root crate `rust-cv-native` re-exports **11 of 30** crates, and the list is
alphabetical up to `video`:

```
3d calib3d core features hal imgproc optimize runtime sfm slam video
```

so it reads as a snapshot taken when those were the crates that existed, not as a
selection. Evidence that it is drift rather than curation: **none** of the omitted
crates sets `publish = false`, and the four named competitive targets are among the
omissions — `plot` (Matplotlib), `math` (SciPy), `registration` (Open3D) and
`localization` (VSLoc-RS). A user of the facade cannot reach four of the five things
the project says it replaces.

**Not closed here, deliberately.** The re-export lines alone do not compile — the
omitted crates are not declared as dependencies of `rust-cv-native`, so completing the
facade means adding **16 dependencies** to the root `Cargo.toml`. That changes the
facade's build graph and feature unification for every downstream user, which is an
architectural decision rather than a fix, and one that deserves a build-and-test pass
on its own rather than being tacked onto an audit.

**If someone closes it:** add the dependencies, re-export all 26 library crates, and
exclude `cli` (a binary) and `python` (a PyO3 cdylib) — neither is a library API
surface. Then confirm `cargo build -p rust-cv-native` and the full suite, since 16 new
edges into one crate is exactly where a feature-unification problem would surface.
