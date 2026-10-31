# GAP REPORT — Rust CV workspace at /home/Phoenix/RUST/RETINA

status: **complete** (survey), **living** (closures). The survey itself was read-only; entries
have since been marked **CLOSED** in the same commit as each fix.

Survey, read-only on source. Method note: counts below are **measured** with the
`grep -rc` command given in the brief; individual stub claims are verified by reading the body
unless marked `UNVERIFIED`.

Targets claimed by the workspace README/root crate: OpenCV, Open3D, Matplotlib, SciPy, VSLoc-RS.

---

> **Currency.** This survey was taken at a point in time; entries are marked
> **CLOSED** where the defect has since been fixed. The coverage table reflects the
> moment of measurement and is **not** refreshed as tests are added — re-measure
> before planning work from it. Fixes since the survey, marked inline below:
>
> - `photo::Stitcher::stitch` fabricating a panorama — **closed**, see (b1) #1.
> - `rendering`'s projected covariance being in the world frame while its Jacobian
>   was camera-space, and opacity round-tripping complemented — **fixed**.
> - `hal`'s `batch_nearest_neighbors` panicking on a NaN coordinate — **fixed**; the
>   GPU-in-name / CPU-in-the-body downgrade in (d1) remains open.
> - `imgproc`'s normalised Gaussian not preserving a constant (kernel summing to
>   `0.99999964`, so `128` blurred to `127`) — **fixed**. It was not in this survey;
>   found later by the numerical parity harness.
> - `video`'s `MOG2::new` inert parameters, `photo`'s inpaint and HDR simplifications,
>   `slam`'s GPU-for-CPU device selection — **fixed**.
> - `optimize`'s CG using an **absolute** residual threshold, so accuracy depended on
>   the units of the problem: at a scale where the absolute rule burned 50 iterations
>   and landed 0.7 from the truth, a relative rule converged in four. Same class as
>   `qr_solve` and `cond`. **fixed**.
> - `3d`'s splat footprint omitting a factor of `aspect`, sizing each splat on a
>   different projection than the one placing it — **fixed**, and now pinned by
>   `footprint_is_the_placement_jacobian`.
>
> Still open and worth reading before planning: the facade re-exports 11 of 30
> crates. `Interpolation::Linear` aliases when downscaling — the same as OpenCV's
> `INTER_LINEAR` (use `INTER_AREA`-style filtering for decimation), so it is a parity
> property, not a defect. Since closed and kept here only so they are not re-opened:
> `erf` (now machine precision, `5f17b5b`), `score_msac`'s `unimplemented!()` (b1 #7 —
> the workspace now has **zero** `todo!`/`unimplemented!`), and the Nistér 5-point
> solver (`ca4ca75`).

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
| 1 | `crates/photo/src/stitcher.rs` | **CLOSED.** Was: `stitch()` returned `images[0].clone()` — the input unmodified, as the "panorama", with `Ok()` and no warning. Now a real estimator: it finds the horizontal offset that maximises overlap agreement and cross-fades the seam. Verified numerically — two ramps 24 apart, aligned with a one-column overlap, give `33 = 31 + ⅔·(34−31)` at the seam, which only a blend produces. | **was HIGH — now fixed** |
| 2 | `crates/hal/src/gpu_kernels/pointcloud.rs:29-39` | **CLOSED.** Was: `normals_cpu_analytic(points, k)` → `Vec::new()`, reached as the silent CPU fallback of `compute_normals_morton_gpu_or_cpu` - N points in, 0 normals out, no error, no log. The `Vec::new()` was itself a fix for an earlier zero-vector-per-point version (fabricated data), so the absence was at least visible - but a caller mapping normals[i] to points[i] got nothing. Now implemented for real: brute-force kNN + PCA smallest-eigenvector, the same math as the GPU pipeline, with the eigenvector's arbitrary sign pinned deterministically. Tested (`tests/normals_cpu.rs`): one normal per point, plane normal on a planar cloud, radial normals on a Fibonacci sphere, finite degenerate handling. | **was HIGH silent fallback — now fixed** |
| 3 | `crates/hal/src/gpu_kernels/mod.rs:1238-1248` | `batch_nearest_neighbors` is documented "on GPU" but the body says `// Simplified: just return k nearest using brute force on CPU`. Takes `gpu: &GpuContext` and reads the buffer back to host (`read_buffer` + `pollster::block_on`) then loops on CPU. GPU in the name and signature, CPU in the body, no record returned. | **HIGH — silent downgrade** |
| 4 | `crates/hal/src/gpu_kernels/mod.rs:1619-1624` | **CLOSED.** Was: `simplify_mesh(_gpu: &GpuContext, ...)` documented "Simplify mesh on GPU", GPU handle unused, body a CPU voxel-grid decimation. There is no GPU kernel and zero callers, so the fix is honesty rather than machinery: the unused handle is **gone** from the signature and the doc says CPU, with the shape of a future GPU implementation noted where it would go. What runs is what is claimed. | **was MED silent downgrade — now fixed** |
| 5 | `crates/3d/src/gpu/mod.rs:722-732` | **CLOSED.** Was: `voxel_to_point_normal_transfer` → `Vec::new()` ("rather than fabricated normals", after an earlier zero-per-point version) - a documented but empty public API. Now implemented with a **pinned convention** (nothing else defined one): voxels in lexicographic key order, `normals[j]` for the j-th occupied voxel, each point copying its voxel's normal. Pairing against `voxel_downsample`'s output would not work - its `HashMap` iteration order is not reproducible - which is stated in the doc. Tested. | **was MED documented-empty — now fixed** |
| 6 | `crates/3d/src/odometry/mod.rs:445-456` | **CLOSED.** Was: `compute_hybrid` → `None` (and `compute_intensity` beside it), so `OdometryMethod::Hybrid`/`Intensity` were public variants that could not succeed - after previously returning a fabricated identity transform. Both are implemented now: photometric Gauss-Newton (`Intensity`) and one system carrying geometric + photometric rows (`Hybrid`), with Levenberg-Marquardt damping and accept/reject. Getting them to recover ground truth exposed **four real bugs** along the way, each now fixed and tested (`tests/odometry_synthetic_recovery.rs`, rendered scenes with known poses): the pyramid dispatcher passed full-resolution colours against scaled frames (wrong pixels at every level but the finest); it fabricated `Some(identity)` when every scale failed - the exact sin the entry was closed for, reintroduced at the dispatch level; the photometric omega block had a spurious sign (`-P' x grad_p` where `a^T[b]x = (a x b)^T` gives `+`); and the convergence check treated "no step accepted" as convergence. | **was MED guaranteed-failure variant — now fixed** |
| 7 | `crates/features/src/usac.rs:327-339` | **CLOSED.** Was: `scorers::score_msac` body `unimplemented!(...)`, reachable through `score_magsac`'s loop, and - the gap report missed this - through `score_ransac` too, which delegated to `score_msac` while its docs called it "the usable scorer". The real architecture takes scorer *closures* (`Fn(&M, &[Point2]) -> (usize, f64, Vec<bool>)`), so the fix is the model-distance metric the old docs wished for, in that same closure style: all three scorers take `residual: impl Fn(&M, &Point2<f64>) -> f64` and work. `score_ransac` now scores what its name says (inlier count, score = outlier count for the minimizer) instead of delegating to MSAC; `score_magsac`'s never-read `threshold` parameter is gone. Tested: each ranks a true line model above a wrong one (`scorers_rank_the_true_model_first`), replacing the old `#[should_panic]` test that had pinned the panic as the contract. | **was HIGH panic — now fixed** |
| 8 | `crates/photo/src/hdr.rs:132` | **CLOSED.** Was: response-curve recovery described as a "simplified Debevec approach" — the normal-equations system was assembled but only solved with a 200-iteration Gauss-Seidel sweep on the *squared* matrix (`A^T A x = A^T b`), the comment promising "Cholesky" never wired up. That never converges on the ill-conditioned Debevec system: on a 2:1 radiance-ratio scene the recovered ratio was ~1.70 (true 2.0), and a uniform scene's recovered value sat at `1/dt_mid` instead of the scene radiance — i.e. the *scale* was right only by accident of the arbitrary `g(125)=0` anchor, not from convergence. The anchor offset is inherent to Debevec (no radiometric reference exists), so absolute radiance is unrecoverable by design and the HDR is relative — but the iterative solve left even the *relative* shape wrong. Fix: `cholesky_solve` (dense Cholesky–Banachiewicz on the SPD normal matrix) replaces Gauss-Seidel; the matrix is ~280×280 so a direct solve is microseconds and exact. Recovery now matches a reference Gaussian-elimination solve to 4 dp: 2:1 ratio scene → 1.926 (true 2.0), uniform scene → 10.0006 = `1/dt_mid`. Pinned by `debevec_recovers_radiance_ratios` (2:1 scene, ratio within 10%) and `debevec_uniform_scene_hits_anchored_scale` (uniform value within 0.5% of `1/dt_mid`). Absolute radiance is intentionally *not* asserted — there is no reference to set it. | LOW-MED — fixed |

### (b2) Formerly-silent stubs that now fail loudly (closed — re-reported only so they are not re-opened)

- `crates/calib3d/src/essential.rs` — Nistér 5-point solver used to return `Err("... is not implemented")`.
  **CLOSED — implemented** (`ca4ca75`); PnP RANSAC also moved to P3P minimal samples (`4d6fcd2`).
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

### (c1) `cv-plot` — `subplot()` accepts an index and discards it; no panel layout exists — **CLOSED** (panel layout + `subplot_grid` implemented and tested; see `subplot_layout` tests)
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

### (c2) `cv-video` — two `MOG2::new` parameters are inert — **CLOSED (honest docs)** (detect_shadows/var_threshold documented as inert; `_detect_shadows` unread; no shadow detection added)
- `crates/video/src/mog2.rs:96,117` — `detect_shadows` is stored as `_detect_shadows` (underscore = deliberately
  unread) and the doc says "**Currently has no effect**". Only "reads" are in tests (`:566,575`).
- `mog2.rs:37,55,114` — `var_threshold` doc'd as "Variance threshold for detecting shadows … **Not currently used
  but reserved for future implementation**"; it is stored and forwarded into `Mog2Params` (`:254`) but shadow
  detection does not exist, so both parameters of the public constructor are decoration. `MOG2::new` doc also
  admits "Invalid parameters are silently accepted".
- This is an **OpenCV-named** API (`createBackgroundSubtractorMOG2`) whose shadow-detection contract is absent
  with no error and no flag on the output.

### (c3) `cv-hal` — `compute_distance_field` is neither signed nor a surface distance — **CLOSED**
- Was: doc claimed "on GPU" / "**signed** distance to mesh surface" while the body ignored the GPU
  handle and returned unsigned distance to the closest *vertex* of each face (never negative, not a
  surface distance). Now: unused handle gone; each voxel is the signed Euclidean distance to the
  closest point on any triangle (Ericson), sign from the winning face's winding normal; voxels with
  no face in the narrow band stay at `+∞`. Pinned by
  `distance_field_is_signed_surface_distance_not_closest_vertex`.

### (c4) `cv-hal` — `build_kdtree` returns Morton codes, not a tree — **CLOSED (honest docs)**
- `crates/hal/src/gpu_kernels/mod.rs:1172-1175` — doc: "Build KDTree on GPU (parallel construction) - simplified
  version / Returns sorted Morton codes as a basic spatial index".
- `GpuKDTree` (`:1436-1441`) has only `points_buffer`, `morton_buffer`, `num_points`, `device` — no node
  structure, no split planes, no tree. Nearest-neighbour queries against it are brute force (now on the GPU,
  see (d1)), so the "index" is not used as an index. The fields are named for what they hold and say so.
- **The buffer-name swap is fixed.** Was (`:1231-1235`): the two public buffers were populated **swapped
  against their names** — `nodes_buffer: points_buf, points_buffer: morton_buf` — and
  `batch_nearest_neighbors` read `nodes_buffer` as points, working only because of the double error. The
  fields are now `points_buffer` (the cloud) and `morton_buffer` (the codes), each documented.

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

### (c5) `cv-viewer` — the crate doc said it renders nothing; **CLOSED (doc corrected)**

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

### (c6) `cv-optimize` — `Isam2` config is accepted and ignored — **CLOSED (ignored params documented; field removed)**
- `crates/optimize/src/isam2.rs:911-917` — "Legacy Isam2 interface for backward compatibility with Python
  bindings". `#[allow(dead_code)] optimize_on_update: bool` — the attribute is compiler-proof that the field is
  **never read**; `with_config(optimize_on_update, _batch_optimize)` takes a second argument `_batch_optimize`
  that is likewise discarded. `Isam2::new()` sets `true` for a flag nothing consults.

### (c7) `cv-3d` — two "different" normal algorithms ignore their distinguishing parameters — **CLOSED (params documented as currently ignored)**
- `crates/3d/src/gpu/mod.rs:718-721` — `voxel_based_normals_simple(points, _voxel_size)` → body is
  `compute_normals(points, 30)`: `_voxel_size` unused, k hardcoded to 30.
- `crates/3d/src/gpu/mod.rs:734-741` — `approximate_normals_simple(points, k, _epsilon)` → `compute_normals(points, k)`:
  `_epsilon` unused. The two public entry points collapse to the same implementation with no way for a caller to
  set the parameter they passed.
- `crates/3d/src/gpu/mod.rs:34` — `// TODO: Actual GPU implementation in hal`; the whole `3d::gpu` module is a
  wrapper around CPU code (see also (b1) #5).

### (c8) Other dead-field/dead-item evidence (compiler-measured unless noted)
- `crates/video/src/optical_flow.rs:625` — **CLOSED — removed.** `PolyCoeffs` no longer carries `F`; the fitted constant term is produced by the unchanged 6×6 solve but discarded rather than stored unused. `PolyCoeffs` is now the 5 gradient coefficients the flow consumes (A..E).
- `crates/3d/src/raycasting/mod.rs:241` — **CLOSED — removed.** `RayHitInfo { distance, point, normal, barycentric }` is deleted; the only field read by callers was `distance`, so `ray_triangle_intersection` now returns `Option<f32>`. The dead `point`/`normal`/`barycentric` math is gone with it.
- `crates/features/src/akaze.rs:534` — **CLOSED — removed.** `to_cpu_f32` (AKAZE never converts GPU→CPU).
- `crates/imgproc/src/edges.rs:31` — **CLOSED — removed.** `kernel_from_1d` (Scharr is hand-built).
- `crates/calib3d/src/essential_fundamental.rs:313` — **CLOSED — removed.** `sample_unique_indices`.
- `crates/registration/src/registration/global/ransac.rs:541` — **CLOSED — removed.** `random_sample`.
- `crates/plot/src/three_d.rs:731` — **CLOSED — removed.** free `bounding_box_all` over a slice of point clouds (multi-cloud 3D plots are not jointly scaled). The `Plot3D::bounding_box` *method* is still used and is unchanged.
- `crates/hal/src/gpu_sparse.rs:12` — **Retained (intentional).** `#[allow(dead_code)] pub struct GpuSparseMatrix` is public API in `cv-hal` with its own unit test (`test_triplets_to_csr`); deleting public surface is out of scope for a dead-code pass. Still honestly `#[allow(dead_code)]`-marked, so it is not silently dead — noted, not fixed here.
- `crates/features/src/orb.rs:17-24` — three spec ORB params (`first_level`, `edge_threshold`, `wta_k`) deliberately absent; the comment states they were removed as dead fields rather than silently kept.

---

## (d) Silent capability downgrades

Distinguished from (b) by *no error and no flag returned* — the caller cannot detect the downgrade.

| # | location | downgrade | detectable by caller? |
|---|---|---|---|
| d1 | `crates/hal/src/gpu_kernels/mod.rs:1238-1248` | **CLOSED** — was: "on GPU" documented, body `// Simplified: just return k nearest using brute force on CPU`, with a blocking device→host copy of the whole cloud. Now a real WGSL kernel (`knn_bruteforce.wgsl`, one invocation per query, sorted top-k) with only the results read back; `test_knn_vs_cpu` passes through it. `k` is capped at 128 (the kernel's list size) and larger k is **refused loudly** with a pointer to `batch_nearest_neighbors_cpu`, which is the old loop under its honest name; `batch_radius_search` (k up to 1024) calls it explicitly. | n/a — fixed |
| d2 | `crates/hal/src/gpu_kernels/mod.rs:1619-1624` | **CLOSED** — `simplify_mesh` no longer accepts the GPU handle it ignored; see (b1) #4. | n/a — fixed |
| d3 | `crates/hal/src/gpu_kernels/pointcloud.rs:11-20,29-39` | **CLOSED** — was: the CPU fallback `normals_cpu_analytic` = `Vec::new()`, N points in, 0 normals out. Now a real implementation (brute-force kNN + PCA); see (b1) #2. | n/a — fixed |
| d4 | `crates/runtime/src/orchestrator.rs:395-414` | **CLOSED** — was: `best_runner()` / `try_best_runner()` returned a bare `RuntimeRunner`, dropping the `is_gpu` signal their `_gpu_wait` siblings return, so callers could not tell (or forgot to check `runner.device()`). Both now return `(RuntimeRunner, bool)` like the siblings; a caller that ignores the flag discards a visible `.1`. The ~20 call sites route through `.map(\|(r, _)\| r)` where the flag is genuinely irrelevant. | n/a — fixed |
| d5 | `crates/runtime/src/orchestrator.rs` | **CLOSED** — was: Default-path device selection treated any GPU as a match for any GPU request (`// Any GPU backend matches a GPU request for now`), so a Vulkan request could silently land on WebGPU. Default / Latency / Throughput / PowerSave now exact-match like `Require`; `Prefer` stays loose across GPUs by design. Pinned by `default_path_exact_matches_gpu_backend` and `default_request_does_not_silently_pick_a_different_gpu`. | n/a — fixed |
| d6 | `crates/features/src/usac.rs:342-366` | **CLOSED** — was: `score_magsac`'s `threshold` unused (caller's calibration discarded) and the whole path panicking via (b1) #7. The signature now takes the residual metric and drops `threshold` entirely (the sigma sweep *is* the threshold range); see (b1) #7. | n/a — fixed |
| d7 | `crates/hal/src/gpu/compute_context_impl.rs:607` | **CLOSED** — was: a comment "Let's assume for now morphology on GPU is f32-compatible or we use casts" over an unchecked `GpuStorage<u8>` downcast with a late `"Failed to downcast GPU result"`. The assumption was moot - the trait method is `morphology<S: Storage<u8>>`, so the dtype is fixed by the type system - but the comment read like an open question and the errors explained nothing. The comment now states the actual invariant (the first downcast pins `S`; the second cannot fail for a caller that got past it), and both errors say what is wrong and what to do. | n/a — fixed |
| d8 | `crates/hal/src/gpu_kernels/undistort.rs:140` | **CLOSED** — was: `_ => 1, // Default to bilinear for now`, an unrecognised interpolation silently becoming bilinear (and the calib3d wrapper above it doing the same). Both now **decline** with `NotSupported`, which routes `undistort_image_ex_ctx` to its CPU `remap_ex_ctx` fallback where cubic and lanczos are honoured. The GPU result's u8 conversion also stops truncating (`round_ties_even`, the quantisation contract the imgproc fix set). | n/a — fixed |
| d9 | `crates/photo/src/stitcher.rs` | **CLOSED** — was returning `images[0]`; see (b1) #1. | n/a |
| d10 | `crates/optimize/src/sparse.rs:109,263` / `hal/.../compute_context_impl.rs:473,548,563` | The *opposite* pattern, for contrast: MLX SpMV and three GPU ops return `Err(NotSupported("…use CPU backend"))` rather than downgrading silently. **These are correct**; listed so the two patterns are not confused. | Yes — loud |

---

## (g) Second review, 2026-10-31 — findings (16 open, was 91)

Fresh per-crate review. Every entry is **CONFIRMED** (a scratch test showed the wrong output)
or **ARGUED** (a precise line-level input → output argument). IDs: `g-h` hal, `g-f`
features/imgproc, `g-i` io/runtime/distributed/video/photo/videoio, `g-m` slam/rendering/
plot/cli/dnn/python. Line numbers are as of `dae9a8f`.

### (g1) hal (0 open, was 12)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-h1 | `gpu_kernels/{threshold,color,icp,subtract,pointcloud_transform}.rs` | **CLOSED.** `dispatch_1d` helper splits oversized grids into 2-D (65535 × n) with `num_workgroups`-based indexing in the shaders; 4K threshold/cvt_color/subtract/transform/ICP pass (`tests/gpu_dispatch_limits.rs`, 13 tests on real hardware). Was: 1-D dispatches exceed wgpu's 65535-group limit: `threshold` / `cvt_color` on a 3840x2160 frame panic with `dispatch group size dimension ([129600, 1, 1]) must be less or equal to 65535` | was HIGH — fixed | CONFIRMED |
| g-h2 | `cpu/border.rs` | **CLOSED.** Was: `Reflect` and `Reflect101` swapped vs OpenCV (cb25ce4 swapped them and pinned the swap in tests); `cv-imgproc`'s own mapping was right. Now the same periodic folding as imgproc; test tables are `cv2.copyMakeBorder` output, including coordinates more than one width outside | was HIGH — fixed | CONFIRMED vs `cv2.copyMakeBorder` |
| g-h3 | `cpu/compute_context_impl.rs` `akaze_contrast_k` | **CLOSED.** Zero gradients excluded on CPU and in the GPU shader (OpenCV's `compute_k_percentile`), 0.03 fallback when none remain. Pinned by `tests/second_review_cpu_fixes.rs`. Was: 70th-percentile gradient over all pixels is 0 on mostly flat images; diffusion then divides `0/0` → 4032/4096 pixels NaN | was HIGH — fixed | CONFIRMED |
| g-h4 | `cpu/compute_context_impl.rs` `tsdf_extract_mesh` | **CLOSED.** Corners at voxel centres and unobserved cells skipped on CPU and in the GPU marching-cubes shaders: wall at z=0.4 meshes to 726 vertices all at 0.400 (was 3036 incl. a phantom at 0.5). Pinned by `cv-hal/tests/tsdf_extract_mesh_surface.rs`. Was: vertex positions use `i*vs` while integration uses `(i+0.5)*vs` (half-voxel shift), and zero-weight corners are meshed → phantom back surface (wall at z=0.4 gives vertices at 0.375 and 0.471) | was HIGH — fixed | CONFIRMED |
| g-h5 | `cpu/compute_context_impl.rs` `hough_lines` | **CLOSED.** rayon `fold`/`reduce`: one accumulator per task, not per row. Was: one full accumulator allocated per image row (~3.4 GB at 1080p) | was HIGH — fixed | ARGUED |
| g-h6 | `cpu/compute_context_impl.rs` `hough_lines` | **CLOSED.** OpenCV `HoughLinesStandard` binning + `findLocalMaximums` 4-neighbour test, sorted by votes; matches `cv2.HoughLines` exactly on the pinned case (`tests/second_review_cpu_fixes.rs`). Was: no peak suppression (GPU path has 3x3 NMS): one line returns 3 | was MED — fixed | CONFIRMED |
| g-h7 | `cpu/compute_context_impl.rs` `optical_flow_lk` | **CLOSED.** Bouguet pyramidal LK on CPU and GPU (template at the feature's level coordinates, next frame bilinear at x + g + v, g = 2(g + v)); 2.3 px shift error 0.0042 px and stable over iterations (was oscillating 0.22–0.52). GPU kernel had the same defect, worse. Pinned by `tests/lk_subpixel_bouguet.rs`. Was: sub-pixel part applied twice; 2.3 px shift oscillates 2.15…2.76 and does not converge | was MED — fixed | CONFIRMED |
| g-h8 | `cpu/compute_context_impl.rs` `convolve_2d`, `morphology`, `warp`, `nms`, `stereo` | **CLOSED.** Tensors are planar CHW (`core/tensor/types.rs`), so the defect was planes 1.. being folded onto plane 0 by an `h`-relative border map; `convolve_2d`, `morphology` and `warp` now process each plane (warp keeps the channel count), `nms`/`stereo_match` reject c ≠ 1. Pinned by `multi_plane_tensors_are_processed_per_plane`. Was: interleaved multi-channel input treated as `c*h` rows → scrambled output | was MED — fixed | CONFIRMED |
| g-h9 | `cpu/compute_context_impl.rs` `stereo_match` | **CLOSED.** Best disparity starts at the NO_MATCH sentinel. Same test file. Was: pixels with no testable disparity report 0 (valid) instead of the no-match marker | was MED — fixed | CONFIRMED |
| g-h10 | `cpu/compute_context_impl.rs` `match_descriptors` | **CLOSED.** Queries with no candidate are skipped. Same test file. Was: empty train set + ratio ≥ 1 fabricates a match to index 0 | was LOW — fixed | CONFIRMED |
| g-h11 | `cpu/compute_context_impl.rs` SIFT descriptor | **CLOSED.** CPU renormalises after the 0.2 clamp (as the GPU shader already did); both backends round when quantising. Step-edge descriptor norm 204 → 512 (`tests/second_review_cpu_fixes.rs`). Was: not renormalised after the 0.2 clamp | was LOW — fixed | ARGUED |
| g-h12 | AKAZE Scharr derivatives (CPU + WGSL) | **CLOSED.** `[3, 10, 3]/32` taps on CPU and in the WGSL shader: unit gain (was 0.3125). Same test file. Was: taps `[1,3,1]/32` give gain 0.3125 on a unit ramp; should be `[3,10,3]/32` | was LOW — fixed | ARGUED |

### (g2) features / imgproc (0 open, was 12)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-f1 | `features/src/brief.rs` | **CLOSED.** Angle converted with `to_radians`; only exactly `-1.0` means unset. Pinned by `tests/brief_angle_degrees.rs`. Was: `kp.angle` (degrees everywhere else) fed to `cos`/`sin` as radians; negative angles treated as "unset" | was HIGH — fixed | CONFIRMED: hamming(0°,360°)=124 |
| g-f2 | `imgproc/src/geometry.rs` `get_pixel_lanczos_with_border` | **CLOSED.** Taps read at offset `k - 3` to match their weights; pinned by `tests/lanczos_tap_alignment.rs` (identity remap and warp return the input). Was: tap index off by one: Lanczos warp/remap shifted 1 px | was HIGH — fixed | CONFIRMED |
| g-f3 | `imgproc/src/histogram.rs` `clahe` | **CLOSED.** OpenCV's LUT (`cdf·255/area`), clip-limit residual spreading and tile interpolation; 5 cases bit-exact vs cv2 (`tests/clahe_opencv_parity.rs`). Was: equalizeHist-style LUT instead of OpenCV's `cdf*255/area`; constant 100 → 255 | was HIGH — fixed | CONFIRMED |
| g-f4 | `imgproc/src/hough.rs` | **CLOSED.** Takes an edge map, OpenCV binning, `> threshold`; HoughLinesP ported incl. its RNG; bit-exact incl. order (`tests/hough_opencv_parity.rs`). Was: runs Canny on its input (edges of an edge map), rho binned with floor → one line returns two at 48.58/50.58 | was HIGH — fixed | CONFIRMED |
| g-f5 | `imgproc/src/edges.rs` Canny NMS | **CLOSED.** No implicit blur, L1 default, OpenCV NMS; 7 images bit-exact vs `cv2.Canny`; `CannyOptions` for L2/blur (`tests/canny_opencv_parity.rs`). Was: `>=` on both sides keeps 2-px edges; L2 default and implicit σ=1 blur differ from OpenCV | was MED — fixed | CONFIRMED |
| g-f6 | `imgproc/src/threshold.rs` `adaptive_threshold` | **CLOSED.** Rounded mean, BORDER_REPLICATE, OpenCV's integer delta rule; 56/56 cases exact (`tests/adaptive_threshold_opencv_parity.rs`). Was: truncated mean, shrinking-window border, fractional `c` comparison | was MED — fixed | CONFIRMED |
| g-f7 | `features/src/usac.rs` `score_magsac` | **CLOSED.** Marginalised MSAC over 10 sigmas in `(0, σ_max]`; works below 1 and ranks by fit. Pinned by `magsac_scores_below_unit_sigma`. Was: `1..=sigma_max as usize` never runs for `sigma_max < 1` → no model | was MED — fixed | CONFIRMED |
| g-f8 | `features/src/usac.rs` adaptive iterations | **CLOSED.** `max_iterations = 0` now adapts N from the best model's inlier ratio (cap 10 000); nonzero stays a fixed budget, as documented. Pinned by `adaptive_iterations_track_the_inlier_ratio`. Was: count fixed at 50% inlier assumption, never updated from the best model | was MED — fixed | ARGUED |
| g-f9 | `imgproc/src/resize.rs` `pyr_down`/`pyr_up` | **CLOSED.** 5-tap Gaussian + Reflect101, `(w+1)/2` sizes; 8 cases exact (`tests/pyramid_opencv_parity.rs`). Was: bilinear resize to `w/2` instead of 5-tap Gaussian + `(w+1)/2`; 1x10 → black 1x1 | was MED — fixed | CONFIRMED |
| g-f10 | `features/src/matcher.rs` | **CLOSED.** `MatchType::BruteForceL2` (OpenCV `NORM_L2` on bytes); `BruteForce` stays the documented Hamming alias for the ORB pipelines. Cross-check against an empty train set no longer panics. Pinned by `tests/matcher_l2.rs`. Was: only Hamming exists; SIFT float-histogram bytes matched by Hamming | was MED — fixed | ARGUED |
| g-f11 | `features/src/orb.rs` rBRIEF | **CLOSED.** Rotated offsets rounded (OpenCV's `cvRound`); pinned by `steered_sampling_is_symmetric_under_half_turns` (fails at 30° with truncation). Was: rotated taps floored, OpenCV rounds | was LOW — fixed | ARGUED |
| g-f12 | `imgproc` `par_chunks_mut(width)` sites | **CLOSED.** 64 function/shape pairs no longer panic on 0×N / N×0 (`tests/zero_sized_images.rs`). Was: zero-width image panics `chunk_size must not be zero` | was LOW — fixed | CONFIRMED |

### (g3) io / runtime / distributed / video / photo / videoio (6 open, was 12)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-i1 | `runtime/src/pipeline/` | CPU-only pipeline cannot run two chained ops (`TransientBufferPool only supports GPU`); GPU↔CPU op hand-off never copies data | HIGH | CONFIRMED (CPU) / ARGUED (GPU) |
| g-i2 | `video/src/optical_flow.rs` `track_point` | **CLOSED.** Template and gradients sampled bilinearly at the fractional feature position, OpenCV-style f32 pyramid, pyramids built once per call: static sub-pixel points stay put, 3-level (9.6, −4.3) shift recovered to 0.004 px (was 0.81 px off). Was: template sampled at `floor(lx)`: a static sub-pixel point (60.5, 60.5) tracks to (60, 60) | was HIGH — fixed | CONFIRMED |
| g-i3 | `io/src/pcd.rs` | **CLOSED.** Reader accepts up to LZF's worst-case growth (`n + n/32 + 16`). Pinned by `tests/second_review_io_parsing.rs`. Was: reader rejects `compressed > uncompressed`, so the crate's own LZF writer output on random data is unreadable | was MED-HIGH — fixed | CONFIRMED |
| g-i4 | `io/src/pcd.rs` ASCII | **CLOSED.** Column offsets are cumulative COUNTs. Same test file. Was: column index ignores `COUNT > 1` | was MED — fixed | CONFIRMED |
| g-i5 | `io/src/pcd.rs` ASCII rgb | **CLOSED.** `TYPE U`/`I` packed colour parsed as an integer token. Same test file. Was: `TYPE U` packed rgb parsed as f32 then `to_bits()` → pure red reads `[0.498,0,0]` | was MED — fixed | CONFIRMED |
| g-i6 | `io/src/obj.rs` | **CLOSED.** Negative indices resolved against vertices read so far; the robustness test that pinned the rejection now pins acceptance. Was: negative (relative) face indices rejected | was MED — fixed | CONFIRMED |
| g-i7 | `distributed/.../shared_memory.rs` `Drop` | refcount decrement + unlink without the `flock` `new` takes → orphaned region, overcommit | MED | ARGUED |
| g-i8 | `distributed/.../shared_memory.rs` `acquire_slot` | slot adopted by PID alone (PID namespaces, PID reuse) | MED | ARGUED |
| g-i9 | `distributed/.../shared_memory.rs` | atomics written through `as_ptr()` of a shared slice (Stacked Borrows UB); reapers' non-saturating `fetch_sub` | LOW | ARGUED |
| g-i10 | `runtime/src/pipeline/allocator.rs` | `trim` uses `shrink_to` (frees nothing); re-`allocate` leaks accounting | LOW | ARGUED |
| g-i11 | `videoio/src/backends/ffmpeg.rs` `grab` | one frame per packet; multi-frame packets → EAGAIN → stream dies — **BLOCKED locally**: `cargo check -p cv-videoio --features ffmpeg` fails building `ffmpeg-sys-next 7.1.3` against the system FFmpeg 8 (libavcodec 63), so a fix cannot be compiled or tested here; left open rather than edited blind | LOW | ARGUED |
| g-i12 | `photo/src/bilateral.rs` | **CLOSED.** Rounded and clamped; pinned by `tests/bilateral_rounding.rs` (uniform images now fixed points; step-edge mean 100.18 → within 0.1 of 100.5). Was: output truncated not rounded | was LOW — fixed | ARGUED |

Also: (c2) says `MOG2`'s `var_threshold` is inert — it is not; `mog2_update` uses it as the match threshold. Doc fix only.

### (g4) slam / rendering / plot / cli / dnn / python (0 open, was 11)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-m1 | `slam/src/tracking.rs` | **CLOSED.** RGB-D bootstrap from depth, monocular two-view (5-point) initialisation, visible tracking loss (`Err` + `Tracker::state`), descriptor/keypoint pairing fixed. Pinned by `tests/slam_bootstrap.rs` (RGB-D ≤ 1.25 cm / 0.13° over 5 frames). Was: map is never populated, so `process_frame` / `process_frame_with_depth` / Python `PySlam` always fail | was HIGH — fixed | CONFIRMED |
| g-m2 | `rendering/.../rasterize.rs` `backward` | **CLOSED.** Real backward (colour, opacity, 2D mean, position, scale, quaternion) matching finite differences; Adam optimizer; background composited; `train_with_report` reports `Stalled` on zero gradient. Toy fit loss 1.49e-2 → 5.6e-12. Was: returns all-zero gradients; trainer never fits | was HIGH — fixed | CONFIRMED |
| g-m3 | `rendering/.../optimize.rs` `densify` | **CLOSED.** Gradient-driven clone/split (accumulated NDC mean gradient vs `grad_threshold`), split scale ÷1.6, stats reset: 3 Gaussians stay 3 without gradient (were 2038 after 10 calls). Was: every small Gaussian doubled each call (exponential growth); `grad_threshold` unread | was HIGH — fixed | CONFIRMED |
| g-m4 | `rendering/.../io.rs` | **CLOSED.** Columns read by property name; ascii and binary (LE/BE) PLY; 3DGS storage conventions (sigmoid opacity, exp scale, SH-DC colour, rot w-first); writer emits the INRIA layout. Pinned by `tests/ply_inria_format.rs`. Was: Gaussian PLY columns read by position, header ignored; binary → `Ok(empty)`; `rot_0` read as x not w | was HIGH — fixed | CONFIRMED |
| g-m5 | `cli/src/main.rs` EuRoC | **CLOSED.** Paired by timestamp via `tum::associate` and `--max-dt`; pinned by `euroc_pairs_by_timestamp_across_sampling_rates` (ATE 0.500087 → ~0). Was: GT/estimate paired by index not timestamp: identical 200 Hz vs 20 Hz trajectories give ATE 0.5 m | was HIGH — fixed | CONFIRMED |
| g-m6 | `dnn/src/lib.rs` preprocess | **CLOSED.** Luma plane repeated per channel (planar CHW); pinned by `test_preprocess_is_planar_chw`. Was: HWC data labelled NCHW → RGB models see a garbled image | was HIGH — fixed | ARGUED |
| g-m7 | `slam/src/tracking.rs` `refine_pose_with_icp` | **CLOSED.** ICP increment composition factored into `apply_icp_increment` (translation τ only); zero update now leaves the pose unchanged (was moved by 0.188). Was: relative translation applied twice | was MED — fixed | ARGUED |
| g-m8 | `rendering/.../optimize.rs` `reset_opacity` | **CLOSED.** Opacities capped at the target (3DGS §5.2); the unit test that pinned raising now pins capping. Was: raises low opacities instead of capping at 0.01 | was MED — fixed | CONFIRMED |
| g-m9 | `plot/src/export.rs` `format_tick_value` | **CLOSED.** Decimals from the tick step (fewest that write it exactly). Pinned by `tick_label_tests` in `export.rs`. Was: decimals from value, not step → duplicate labels `0,0,0.001,…` | was MED — fixed | CONFIRMED |
| g-m10 | `plot/src/export.rs` bars | **CLOSED.** Matplotlib `bar`: 0.8 data units wide, from y = 0, bar extents in the autoscale with y = 0 sticky; pinned against matplotlib's limits in `tests/svg_geometry.rs`. Was: drawn from axis min, not 0 | was MED — fixed | CONFIRMED |
| g-m11 | `plot/src/three_d.rs` `camera_position` | **CLOSED.** Eye along `project`'s view axis `(sin a cos e, sin e, cos a cos e)`; pinned by `camera_convention_tests`. Was: HTML camera convention disagrees with SVG `project` (~22° at defaults) | was LOW-MED — fixed | CONFIRMED |

Minor (LOW): python `pose.matrix()` flattened column-major (numpy gets the transpose) — **CLOSED** (`1c888fd`);
python `sparse_solve_cg` checks `ncols` not `nrows`; `slam` returns `Ok(last pose)` when PnP fails — **CLOSED** with g-m1;
`SphericalHarmonics::eval` mixes channels (unused).

### (g5) calib3d / sfm (0 open, was 7)

Partial pass: essential, pnp, essential_fundamental, fundamental/homography wrappers,
`cv_core::robust::Ransac`. Triangulation, distortion, calibration and `crates/sfm` are covered by (g8).

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-c1 | `calib3d/src/essential.rs` (ca4ca75) | **CLOSED.** Three causes: basis buffer layout, A's column order vs Nistér's monomial order (`NISTER_COLUMN_ORDER`), and a squared-condition null-space basis (now SVD of padded Q). True E found 200/200 (was 0/200); worst \|det E\| 5.3e-8. Pinned by `tests/essential_five_point_ground_truth.rs`. Was: Nistér 5-point: true E among solutions in 0/200 synthetic scenes; returned matrices fit the 5 points but violate `2EEᵀE − tr(EEᵀ)E = 0` and `det E = 0`. Null-space buffer laid out 9x4 row-major where OpenCV's `getCoeffMat` expects 4x9 basis-major — and fixing that alone is not enough | was HIGH — fixed | CONFIRMED |
| g-c2 | `calib3d/src/pnp.rs` P3P Kabsch (4d6fcd2) | **CLOSED.** Row 2 of Vᵀ is flipped now; pinned by `tests/p3p_random_ground_truth.rs` (1000 random exact problems, was 520/1000 before the fix). Was: reflection fix negates column 2 of Vᵀ instead of row 2 → true pose found in 500/1000 exact problems | was HIGH — fixed | CONFIRMED |
| g-c3 | `calib3d/src/pnp.rs` `sample_unique_indices` | **CLOSED.** Sampler uses the LCG's high bits (`state >> 33`). Was: `state % n` uses an LCG's low bits: for n = 2^m only n distinct samples ever drawn | was MED — fixed | CONFIRMED |
| g-c4 | `calib3d/src/pnp.rs` P3P roots | **CLOSED.** Candidates with distance ratio `p <= 0` are skipped; the same test asserts no candidate puts a point behind the camera. Was: distance ratio `p` not checked positive → behind-camera poses accepted; scoring ignores `z <= 0` | was MED — fixed | ARGUED |
| g-c5 | `calib3d/src/pnp.rs` PnP RANSAC | **CLOSED.** Refinement from ≥ 4 inliers (kept only if no worse); `solve_pnp_refine` accepts 4 points; behind-camera points score ∞ in PnP RANSAC; model hoisted out of the loop. Pinned by `tests/pnp_refine_four_points.rs`. Was: final refinement only with ≥ 6 inliers; 4–5 return the raw P3P pose | was MED — fixed | ARGUED |
| g-c6 | `core/src/robust.rs` adaptive iterations | **CLOSED.** All-inlier best model ends the search; sampling shuffles only k indices. Pinned by `ransac_stops_at_once_when_every_point_is_an_inlier`. Was: all-inlier case never lowers the iteration bound | was LOW — fixed | ARGUED |
| g-c7 | `calib3d` `find_essential_mat_ransac` | **CLOSED.** RANSAC minimal sample is Nistér 5-point + 1 disambiguating point (OpenCV's choice), 8-point kept for the consensus refit, confidence 0.999 (OpenCV default). On a forward-motion, plane-dominated scene with 20% outliers the 8-point sample left < 5 inliers; now on par with `cv2.findEssentialMat` (worst 6.9–10.2° vs cv2 7.76°). Pinned by `tests/essential_ransac_forward_motion.rs`. Was: 8-point essential RANSAC recovered a translation direction ~55° wrong on the SLAM bootstrap scene (rendered wall/floor/boxes), where a 5-point RANSAC on `EssentialSolver::estimate_5point` succeeds; found while fixing g-m1. Re-measure after `229914c` (consensus refit) | was MED — fixed | CONFIRMED (by the g-m1 fix's author) |

### (g6) 3d / registration / pointcloud (0 open, was 12)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-d1 | `3d/src/tsdf/mod.rs` `integrate_ctx` / `extract_mesh` | **CLOSED.** KinectFusion sign (positive = free), per-voxel projective integration, zero-weight cells skipped, seam cells meshed from neighbour blocks: all vertices on the plane (was 8153/12921 > 3 cm off). Pinned by `cv-3d/tests/tsdf_surface_accuracy.rs`. Was: free space negative while untouched voxels are +1 (phantom surface at the band edge), z-depth used as ray range, block-seam cells never meshed: 8153/12921 vertices > 3 cm off a z=1 plane | was HIGH — fixed | CONFIRMED |
| g-d2 | `registration/.../colored.rs` | **CLOSED.** Reimplemented per Park/Zhou/Koltun 2017 (target colour gradients, √λ/√(1−λ) weighting, Jacobians at the transformed point); pose error ~6e-8. Pinned by `tests/colored_icp_converges.rs`. Was: unsigned residuals, Jacobian at the untransformed point, ad-hoc photometric term: 0.01 offset not recovered (point-to-plane recovers it) | was HIGH — fixed | CONFIRMED |
| g-d3 | `registration/.../gnc.rs` | **CLOSED.** Schedule from a saved reference loss, loss restored on return: μ stays in [μ0, 100μ0], repeated calls identical. Pinned by `tests/gnc_schedule_anneals.rs`. Was: schedule read from the already-updated loss → μ grows to 2.7e12; loss state leaks across calls | was HIGH — fixed | CONFIRMED |
| g-d4 | `registration/mod.rs` `get_information_matrix_from_point_clouds` | **CLOSED.** Open3D's Σ GᵀG with G = [-[q]× │ I] and a `max_correspondence_distance` parameter. Pinned by `tests/information_matrix_open3d.rs`. Was: not Open3D's `Σ GᵀG`, `G = [-[q]x │ I]`: zero rotation information at perfect alignment, origin-dependent | was HIGH — fixed | CONFIRMED |
| g-d5 | `3d/src/raycasting/mod.rs` `point_inside_mesh_bvh` | **CLOSED.** Majority vote of three generic-direction rays; pinned by `tests/point_inside_mesh_cube.rs` (cube centre was outside). Was: shared-edge hits counted twice: cube centre reported outside | was MED — fixed | CONFIRMED |
| g-d6 | `3d/src/gpu/mod.rs` `compute_normals_cpu` | **CLOSED.** Shell result used only when provably exact (k-th distance ≤ voxel size); otherwise exact k-NN from the KD-tree. Pinned by `tests/normals_sparse_region.rs`. Was: 27-voxel neighbourhood with a global voxel size; sparse regions silently get `(0,0,1)` | was MED — fixed | CONFIRMED |
| g-d7 | `registration/mod.rs` point-to-plane ICP | **CLOSED.** Metrics measured after the update (point-to-point rmse) and Open3D's 1e-6 relative stop. Pinned by `tests/icp_reports_open3d_metrics.rs`. Was: rmse/fitness measured before the update but reported with it; no relative convergence | was MED — fixed | CONFIRMED |
| g-d8 | `pointcloud/src/point_cloud.rs` `voxel_down_sample` | **CLOSED.** Non-finite points skipped before bucketing. Pinned by `cv-pointcloud/tests/voxel_down_sample_nan.rs`. Was: NaN point joins voxel (0,0,0) and poisons a real point's centroid | was MED — fixed | CONFIRMED |
| g-d9 | `3d/src/spatial/kd_tree.rs` `search_radius` | **CLOSED.** Radius prune is inclusive (`<=`) like the accept test. Pinned by `cv-3d/tests/kd_tree_radius_boundary.rs`. Was: strict prune vs inclusive accept: 2 of 3 points at exactly r returned | was LOW — fixed | CONFIRMED |
| g-d10 | `3d/src/spatial/octree.rs` `insert` | **CLOSED.** `insert` returns `false` for out-of-bounds points. Was: out-of-bounds points dropped silently | was LOW — fixed | CONFIRMED |
| g-d11 | `registration/.../global/ransac.rs` | **CLOSED.** Nearest-feature matching without a metric gate (Open3D); `ransac_n` sets the sample size. Pinned by `tests/global_ransac_feature_gate.rs` (3 mm noise: error 2.83 → < 0.15). Was: one distance gates FPFH feature space and metric space; `ransac_n` ignored | was LOW — fixed | ARGUED |
| g-d12 | `pointcloud/src/point_cloud.rs` `cluster_dbscan` | **CLOSED.** Each point queued at most once; noise points reached become border points directly. Pinned by `tests/dbscan_queue.rs`. Was: visited neighbours re-queued → O(N²) seed memory | was LOW — fixed | ARGUED |

### (g7) core / math / optimize / geometry2d / signal_proc / scientific (7 open, was 12)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-n1 | `geometry2d` `in_circumcircle` / `delaunay_triangulation` | absolute EPS on a length⁴ determinant: same 40 points scaled 1e-5 → 0 triangles | HIGH | CONFIRMED |
| g-n2 | `geometry2d` `convex_hull` / `is_valid` / `perpendicular_distance` | absolute EPS on length² quantities: hull loses vertices at 1e-3, degenerate at 1e-5 | MED | CONFIRMED |
| g-n3 | `math/src/linalg.rs` `null_space` | **CLOSED.** Wide matrices zero-padded to square before the SVD. Pinned by `tests/eig_and_null_space.rs`. Was: thin SVD: `null_space([[1,0,0]])` is 3x0 (should be 3x2) | was HIGH — fixed | CONFIRMED |
| g-n4 | `math/src/linalg.rs` `eig` | **CLOSED.** Eigenvectors are null vectors of `A − λI` (complex SVD), packed like LAPACK `dgeev`; repeated eigenvalues get independent vectors. Same test file. Was: returns Schur vectors as eigenvectors: `[[1,1],[0,2]]` residual 1.0 | was HIGH — fixed | CONFIRMED |
| g-n5 | `optimize/src/brent.rs` `brentq` | **CLOSED.** Port of SciPy's `brentq.c`: x-tolerance only, exact-zero stop; 8 evaluations on the cubic (SciPy 8). Pinned in `tests/regression_tests.rs`. Was: `tol` also gates \|f\| (returns 2.0 for root 1.5); stale `fc` → 81 evals vs SciPy's 8 | was HIGH — fixed | CONFIRMED |
| g-n6 | `optimize/src/lbfgsb.rs` | direction ignores the active set; failed line search accepted: f = 1.0135 vs SciPy 1.0 | HIGH | CONFIRMED |
| g-n7 | `math/src/sparse.rs` `gmres_solve` / `cg_solve` | lucky breakdown skips Givens; absolute thresholds; `Ok` when not converged | MED | CONFIRMED |
| g-n8 | `signal_proc` `resample` | **CLOSED.** Even-length downsample folds `X[num/2]` into the Nyquist bin; matches SciPy to 1e-8 in the same test file. Was: even-length Nyquist bin halved, not folded | was MED — fixed | CONFIRMED vs SciPy |
| g-n9 | `signal_proc` `welch` / `stft` / `spectrogram` | no constant detrend, symmetric Hann: PSD[0] 249.9 vs SciPy 0.175 | MED | CONFIRMED vs SciPy |
| g-n10 | `signal_proc` `find_peaks` | **CLOSED.** SciPy's `_local_maxima_1d` plateau scan (midpoint, rounded down). Pinned by `tests/scipy_find_peaks_resample.rs`. Was: plateau peaks never found | was MED — fixed | CONFIRMED vs SciPy |
| g-n11 | `math/src/stats.rs` normal cdf/ppf, p-values | lower-tail cancellation (`cdf(-10)` = 0), unrefined `ppf`, `1 - cdf` p-values | MED | CONFIRMED vs SciPy |
| g-n12 | `math/src/integrate.rs` | `quad` is a fixed trapezoid with error 0 and evaluates endpoints; `ode_solve` overshoots `t_end` and stops silently at 10k steps | MED | ARGUED |

### (g8) sfm / triangulation / distortion / calibration (0 open, was 9)

Second calib3d pass, covering what (g5) did not reach. `calibrate_camera_planar` matches
`cv2.calibrateCamera` to every printed digit (rms 0.237948) and the BA analytic Jacobian
matches finite differences (~1e-7, including θ≈π) — checked, not defects.

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-s1 | `core/src/geometry/distortion.rs` `remove` | **CLOSED.** Damped Newton on the full 2-D map (analytic Jacobian) after the radial bracket; f32 delegates to it; `undistort_points` errors instead of returning the distorted point. Matches `cv2.undistortPointsIter` to 1e-6 px (`tests/undistort_strong_tangential.rs`). Was: undistortion fixed-point diverges with tangential terms; `remove` then returns the *distorted* point: 166.6 px error at r = 0.8 (`cv2.undistortPointsIter` exact) | was HIGH — fixed | CONFIRMED vs cv2 |
| g-s2 | `sfm/src/mapper.rs` `triangulate_track` | **CLOSED.** Cheirality in camera A's frame; pinned by `an_orbit_registers_views_facing_away_from_the_seed` (22/36 before, ≥ 34/36 after). Was: cheirality tests world z, not depth in camera A: 36-view orbit registers 19/36 | was HIGH — fixed | CONFIRMED |
| g-s3 | `calib3d/src/project.rs` rotation Jacobian | **CLOSED.** rvec via unit quaternion (stable at π) and central differences rebuilt from rvec; matches an independent FD to 1e-3 px/rad at θ = 1, π−1e-3, π−1e-7 (`tests/project_jacobian_near_pi.rs`). Was: `rotation_matrix_to_rodrigues` divides by sin θ: max \|J − FD\| = 2.1e8 at θ = π | was HIGH — fixed | CONFIRMED |
| g-s4 | `calib3d/src/stereo.rs` `stereo_rectify_matrices` | **CLOSED.** OpenCV's Bouguet (`cvStereoRectify`): half-split relative rotation, baseline turned onto its own dominant axis with its sign, sign carried in P2/Q. R1, R2, P2 match `cv2.stereoRectify` to 1e-9 on side-by-side, reversed, vertical and rotated rigs (`tests/stereo_rectify_bouguet.rs`). Was: always rotates the baseline onto +x: right-camera-on-left rectifies 180° turned, vertical stereo 90° (cv2 gives R1 = I) | was MED — fixed | CONFIRMED vs cv2 |
| g-s5 | `sfm/src/mapper.rs` `sample_unique_indices` | **CLOSED.** High LCG bits, as g-c3. Was: same LCG low-bit sampler as g-c3: n distinct samples for n = 2^m | was MED — fixed | CONFIRMED |
| g-s6 | `sfm/src/bundle_adjustment.rs` `bundle_adjust_sequential` | **CLOSED.** Cost and residuals re-measured after `remove_outliers`. (Argued fix; no dedicated test — the step-acceptance comparison is now between like quantities.) Was: cost not recomputed after `remove_outliers` → any next step accepted | was MED — fixed | ARGUED |
| g-s7 | `sfm/src/bundle_adjustment.rs` `compute_point_reprojection_error_index` | **CLOSED.** Behind-camera observation scores `INFINITY`; pinned by `landmark_behind_its_camera_is_pruned`. Was: behind-camera observations skipped (error 0) → never pruned | was MED — fixed | ARGUED |
| g-s8 | `calib3d/src/calibration.rs` | **CLOSED.** Removed. Was: leftover debug `eprintln!` on every calibration / every 10 LM iterations | was LOW — fixed | CONFIRMED |
| g-s9 | `calib3d/src/distortion.rs` `fisheye_init_undistort_rectify_map` | **CLOSED.** Zero-size guard copied from the pinhole map; pinned in `tests/undistort_map_singular_intrinsics.rs`. Was: zero-size image panics (`chunk_size must not be zero`) | was LOW — fixed | CONFIRMED |

### (g9) follow-ups found while fixing (g1)–(g8) (3 open, was 4)

| id | location | defect | severity | evidence |
|---|---|---|---|---|
| g-x1 | `hal/src/cpu/compute_context_impl.rs` `hough_lines` | **CLOSED.** rho reported as `(r - (num_rho-1)/2)·rho` with integer division, matching cv2 4.13 on an even bin count (`hough_lines_rho_with_an_even_bin_count_matches_opencv`); `>= threshold` kept and documented, as the HAL GPU path uses it. Was: when `numrho` is even, cv2 4.13 still reports rho on whole multiples of the resolution; the formula from `6b527e5` is half a bin off there, and it keeps `>= threshold` where OpenCV uses `>`. imgproc's `hough_lines` matches 4.13 exactly; the two should share one implementation | was LOW — fixed | ARGUED (found during g-f4) |
| g-x2 | `hal/shaders/canny.wgsl`, `hough.wgsl` | GPU Canny and Hough still run the pre-g-f4/g-f5 algorithms, so `*_ctx` on a GPU runner disagrees with cv2 and with the CPU path (documented on the `_ctx` functions) | MED | ARGUED |
| g-x3 | `imgproc` Gaussian kernels | OpenCV's 8-bit Gaussian uses fixed small-kernel tables (and `cvRound(6σ+1)\|1` sizing); applied for `adaptive_threshold` only - other Gaussian paths may differ by ±1 LSB | LOW | ARGUED |
| g-x4 | `rendering` `SphericalHarmonics::eval` | degree ≥ 1 is not the standard SH basis (channels mixed, constants missing); rendering uses DC only, so loaded `f_rest` view-dependence is wrong | LOW | ARGUED (noted in g4, still open) |

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
- **Partial — GPU mesh ops**: `hal::gpu_kernels::simplify_mesh` is honest CPU (see (d2));
  `batch_nearest_neighbors` is now a real GPU kernel (see (d1)); `compute_distance_field` is a real
  signed surface field on CPU (see (c3)).
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
- `gpu_kernels/mod.rs` distance field closest-vertex stub — **closed** (see c3).
- `gpu_kernels/pointcloud.rs:27` normals CPU fallback not implemented (see b1 #2).
- `gpu_kernels/undistort.rs:140` `_ => 1, // Default to bilinear for now`.
- `gpu/mod.rs:578` "Create a simplified compute pipeline".
- `cpu/compute_context_impl.rs:3993` "Subtraction not implemented for this type".
- (2 remaining are the `Not implemented ...` doc headers for the same sites above.)

**3d (7)**: `odometry/mod.rs:456`, `gpu/mod.rs:34` "TODO: Actual GPU implementation in hal" (the
`gpu` module is a thin wrapper), `gpu/mod.rs:728`, `mesh/reconstruction/mod.rs:24` "PCA (simplified)",
`mesh/reconstruction/delaunay.rs:1` "Delaunay-based reconstruction (**placeholder**)" — the file
header itself, `mesh/reconstruction/poisson.rs:3,51` "simplified Poisson ... regular grid ... octree is".

**runtime (5, was 6)**: orchestrator GPU-backend matching is **closed** (see d5); `memory.rs:290,398,446` transfer `not implemented for {:?}`,
`memory_manager.rs:88` "Return to global pool for now", `pipeline/fusion.rs:711` "Custom fused kernel
(placeholder)".

**features (4)**: `usac.rs:316` (see b1 #7), `haar_cascade/mod.rs:168` min_neighbors grouping
"simplified", `orb.rs:17` — three spec ORB params (`first_level`, `edge_threshold`, `wta_k`)
deliberately absent and previously dead fields.

**imgproc (4)**: `segmentation.rs:215,239` GrabCut is K-means + 4-connected smoothness, explicitly
**not** GMM + graph cut; `contours.rs:457,488` "simplified Suzuki-Abe" border following.

**calib3d (4)**: `essential.rs:35,72,75` Nistér 5-point absent; `project.rs:305` Jacobian by
**numerical differentiation** "for now".

**photo (1, was 3)**: `stitcher.rs` is **closed** (see b1 #1); `hdr.rs` Debevec is **closed** (Cholesky solve replaces Gauss-Seidel; see b1 #8); `inpaint.rs:260`
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
