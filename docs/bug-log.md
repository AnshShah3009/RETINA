# Bug and defect log

Every defect found and fixed in this repository, with the evidence that found it
and the measurement that confirmed the fix. Grouped by where the defect lived.
Nothing here is theoretical: each entry was reproduced on real data or a real
test, and the "found by" column says which.

## Rendering and image processing

| Defect | Found by | Fix |
| --- | ---: | --- |
| Marching-cubes edge table row 33 dropped edge 9, emitting a bogus vertex with a zero normal for cube index `0x21` | reading the table against `tri_table` | Corrected `0x139` → `0x339`; also row 222 |
| Canny non-maximum suppression compared the wrong diagonals, thinning along edges instead of across them, so edges differed from OpenCV | CPU/GPU parity review | Swapped the direction arms on both the CPU path and `canny.wgsl` |
| Canny hysteresis wrote 0/1 on the CPU and 0/255 on the GPU — a 255× difference for the same edge | parity test | CPU now writes 255 |
| `resize` ignored its `Interpolation` argument and always returned bilinear, so `Nearest`/`Cubic`/`Lanczos` were silently wrong | code review of a reported wrong result | Dispatches per variant; `Cubic` and `Lanczos-3` implemented rather than faked |
| `warp_perspective` treated the caller's matrix as dst→src while `warp_affine` inverted it first, so an OpenCV-style forward homography came out inverted | convention mismatch found while unifying | Both now take a forward (src→dst) homography |
| `sharpen` added `1 + amount` at the kernel centre, giving DC gain `1 + amount` and brightening every image | DC-gain analysis | Centre term is now `1 + 4·amount`, unit gain |
| `rfft` on empty input sliced `spectrum[..1]` on an empty `Vec` | audit | Returns empty |
| Sobel/Harris/GFTT: `sobel_kernels_1d`, `gaussian_kernel_1d`, `pixel_at`, `compute_sobel_gradients` existed verbatim in three places | dedup audit | Hoisted to `cv-features::gradients` |
| Cubic and Lanczos kernels duplicated verbatim inside `cv-imgproc` | dedup audit | One private `kernels` module |

## GPU / HAL

| Defect | Found by | Fix |
| --- | ---: | --- |
| `*%` is not a WGSL operator — 9 sites, so the spatial-hash shaders failed to parse and the GPU spatial ICP path had never run | running `perf_tests` on a real GPU | Replaced with `*`, which already wraps |
| The spatial-hash correspondence shader read bucket entries through a binding pointing at the counts buffer, so every lookup returned 0 and all correspondences silently mapped to target 0 | reading the bind group against the shader | Correct buffer; counts and buckets merged into one table to stay inside the 4-storage-buffer limit |
| `voxel_grid_downsample` sized its output for `vec3<f32>` at 12 bytes/element while WGSL strides 16, so writes past ~¾ capacity went out of bounds | layout arithmetic | Buffer sized for the stride; element count matches the tensor shape |
| `tsdf_gpu::extract_surface` dispatched 1-D while indexing 3-D, so only the `y=0,z=0` row was ever written | dispatch arithmetic | `(vol_x.div_ceil(64), vol_y, vol_z)` |
| `pointcloud.rs` normals dispatched `num_points` workgroups for a `@workgroup_size(256)` shader — 256× the work | dispatch arithmetic | `div_ceil(256)` |
| Device creation requested 256 MiB limits unconditionally, failing on adapters advertising less | limits review | `min()` against `adapter.limits()` |
| GPU `nms_boxes` suppressed in input order without sorting by score, so it disagreed with the CPU path and returned a different, incorrect set | parity test | Sorts by score first |
| TV-L1 warp bound the packed flow buffer to both `input_flow_x` and `input_flow_y`, so u and v were the same component; the same aliasing affected the divergence pass | reading the bindings | Indexed by plane offset |
| `template_matching`, `sparse` and `subtract` underflowed or panicked on mismatched/empty input | audit | Return errors |
| Non-`f32` inputs hit `downcast_ref::<GpuStorage<f32>>().unwrap()` and panicked | audit | `DataType` guards |
| `Mog2Params` implemented `bytemuck::Pod` for every `T: Float`, but the layout has padding at `T = f64`, so safe code could read uninitialised bytes | unsafe review | Concrete `f32`-only impl plus a size assertion |
| `ThresholdParams` had the same over-broad bound (`u8` would be unsound) and ran `bytes_of` before its dtype check | unsafe review | Bound tightened, guard moved first |
| Device limits were chosen by wgpu, so a multi-GPU machine's device was not controllable or attributable | reproducibility audit | `DeviceSelector` (Default/Discrete/Integrated/name/index); a non-matching name errors rather than falling back |
| `pointer_safety_tests.rs` transmuted `f32`→`f32` and never touched the code it was named after, so it could not catch a removed guard | code review | Rewritten to drive the real entry points and assert `NotSupported` for non-`f32` |
| ORB's `detect_ctx` fed a `[0,1]`-normalised tensor to FAST with a 0..255 threshold, so no corner could ever score | audit | Thresholds now in the same units as the data |
| `orb_detect_and_compute` returned the full detection list beside a shorter descriptor list — a 31x31 patch reaching past the frame is dropped by `extract`, so on a 160x120 test frame 216 keypoints came back against 154 descriptors, and every index past the first drop referred to a different keypoint | audit, after the `wta_k` investigation turned up three dead ORB fields | Returns the descriptors' own keypoints, so the two are index-parallel by contract; two tests pin it, one of which fails on the old code with exactly this 216-vs-154 mismatch |
| Point-to-plane ICP reported the caller's initial transform with `fitness: 1.0` and `inlier_rmse: 0.0` whenever the target cloud had no normals: the accumulation body was skipped for every correspondence, `ata` stayed a zero matrix, the update was silently skipped — and a target without normals is the ordinary case, since `PointCloud` makes the field optional | review of the registration crate | counts the correspondences that contributed geometry and returns `None` below three; RMSE divided by that count rather than the number offered |
| Colored ICP's photometric Jacobian ignored all three arguments and returned a constant, so its `J Jᵀ` block was rank 1 and the normal equations singular for any `lambda < 1` — the solver never moved while the function reported success | same review | derived from the point's coordinates scaled by the luma difference, so the block is full rank |
| A singular `ata` in colored ICP was handled by `if let Some(..) = ata.try_inverse()`, so the update was silently skipped and the initial transform was returned with `fitness: 1.0` | same | returns `None`; the step is solved by QR least squares, since a Gauss-Newton step is `A⁻¹b` only when well-conditioned and `try_inverse` rejects merely ill-conditioned matrices |
| Colored ICP normalised a possibly-zero difference vector to get a fallback normal, giving NaN that propagated through the whole Jacobian; a zero difference is what an already-aligned pair produces | same | `try_normalize` with a zero fallback |
| `f_score` returned `NaN` for `precision = inf` (`inf / inf`), which passed its `denominator <= 0.0` guard and contradicted the module's claim that it is total | same review | non-finite inputs yield 0.0 |
| `Orb` declared `wta_k`, `edge_threshold` and `first_level` as fields with no setter and no reader, so the struct read as supported configuration that did nothing | same audit | `first_level` removed; a comment states plainly which spec behaviour is not implemented, with the measurement behind it |
| ORB's pyramid scale was computed, stored in `kp.size`, and then never read — the descriptor sampled a fixed 31x31 pattern regardless of the level, so a corner found at level 0 and at level 5 produced bit-identical descriptions and the pyramid bought nothing | audit of `kp.size` after the dead-field sweep | `Orb::with_scale_aware_descriptor` scales the sampling to each keypoint's own level. Opt-in, since on corrected measurement it helps only ETH3D electro (3/5 -> 4/5) and costs courtyard a view |
| GPU `warp` dispatched `dst_w.div_ceil(4).div_ceil(16)` while the shader indexes one pixel per invocation with 16-wide workgroups — a quarter of the destination width was never written, and for a 64-wide destination only the first 16 columns were | review of the HAL GPU dispatch arithmetic | `div_ceil(16)` |
| GPU `warp` clamped output to [0, 255]; the CPU backend does not, so an f32 warp of any tensor outside that range returned different data per backend | same review | clamp removed |
| GPU `optical_flow_lk` read its initial guess from a buffer created with `create_buffer` and never seeded, sampling gradients at uninitialised coordinates and in practice returning the input points unchanged | same review | buffer seeded from the initial points, matching the CPU and the shader's own comment |
| GPU `tsdf_raycast` returned 0.0 outside the volume where the CPU returns 1.0 — 0.0 is the iso-surface, so a ray leaving the volume reported a surface hit on the boundary, putting a shell of phantom depth around any partially-observed volume | same review | returns 1.0, matching the CPU |
| `fast_detect` guarded rows with `y >= h - 3` on a `usize`, which wraps below 3 so the guard never fired and the loop read past the end of the source; any frame under 7px triggered it | same review | returns an all-zero response map, covered by a new test |
| `compute_hybrid`, `voxel_to_point_normal_transfer` and `normals_cpu_analytic` returned fabricated results — identity pose with `fitness: 0.0`, zero vectors per point, and zero normals from the *only* CPU fallback — where a caller had no way to tell absence from data | same review | all three report absence now |
| The Lowe ratio test kept the most ambiguous matches: `0.0/0.0` is NaN and `NaN > threshold` is false, so a zero second-best distance — exactly the duplicate-descriptor case the test exists to reject — was kept and flowed into track building and PnP | review of the feature pipeline | a zero second-best distance is rejected explicitly |
| A `CV_SFM_DIAG_SPAN` environment variable left over from my own diagnostics changed seed ranking, contradicting the module's documented determinism guarantee | the same review | removed; the tree was audited for the pattern and the remaining env vars are legitimate configuration |
| GPU stereo matching never ran: `stereo_match.wgsl` declared `fn get_pixel(data: ptr<storage, ...>)` and WGSL forbids passing a storage-space pointer into a function, so every dispatch was a hard validation failure | parity harness covering operations `multi_gpu_tests.rs` omits | reads the bound arrays through a module-scope index helper |
| The separable Gaussian blur had no channel concept, indexing `y * width + x` — a 3-channel image was blurred as one long row, every output pixel averaging all three planes: 26% mean relative error against the CPU, over 100% on some pixels | same | carries `channels` in the uniform, dispatches `width * channels`, indexes `(y * width + x) * channels + ch` |
| …and then still wrote only the first plane: the invocation index was mapped as `x = gid % total, ch = gid / total`, which is backwards for a channels-innermost `(c, h, w)` layout, so `ch` was 0 for every invocation | printing the first row of each plane — channel 0 exact, channels 1 and 2 all zero | `x = gid / c, ch = gid % c`; all three channels now match the CPU, worst difference 0.375 against 1,695,618 |
| GPU ICP never ran, for four independent reasons: the shader used `new` as a local (a WGSL reserved word) and passed a storage-space pointer into a function (forbidden); six storage bindings against a device limit of four, so the pipeline could not be created; points read at a stride of 3 instead of the 4 the caller's `(1, N, 4)` packing uses; and `icp_correspondences` used `shape.height` as the point count, correct only for a `(3, N, 1)` layout | parity harness plus a new `every_shader_compiles` test | helpers inlined, accumulators merged into one buffer, point arrays interleaved, stride and count corrected. Verified on hardware: CPU and GPU both recover a known 2cm offset as -0.02000 |
| The ctx ICP loop uploaded the source once and never re-transformed it, so the correspondence set was a fixed point and the "ICP" was one Gauss-Newton step repeated | same | the running pose is applied before each search; the existing parity test takes the CPU early-return and never reached it |
| GPU SpMV never ran: `spmv.wgsl` declared a `SparseMatrix` struct with runtime-sized array fields, which WGSL rejects ("Field 'row_ptr' can't be dynamically-sized"), and the struct was never used; the shader then needed five storage bindings against a limit of four | divergence probe over operations the parity suite omits | dead struct removed; `row_ptr` and `col_indices` concatenated into one buffer, which is natural for CSR, with lengths in a uniform |
| CPU `icp_accumulate` indexed the point arrays with correspondence indices and no bounds check, panicking with "index out of bounds: the len is 12 but the index is 12" when the correspondences came from a different cloud | same | a bad correspondence costs an inlier rather than the whole call |
| Fast global registration had no ratio test, so a target set whose FPFH histograms were all alike gave every source point the same nearest target and the correspondence set collapsed to `(i, 0)` — physically impossible, and returned as `Ok` with `fitness = 0.625, rmse = 0` | same | Lowe ratio test drops ambiguous matches; the sibling RANSAC entry point already validated, this one did not |
| `FastGlobalRegistrationOption::maximum_iterations` was read by nothing, so tightening it had no effect; `tuple_scale` was equally dead | same | the first is removed, the second is now the ratio test, and every remaining field is documented and read |
| The LBVH build declared five storage bindings against a device limit of four, so no pipeline in the file could be created. Because wgpu derives the layout per entry point, `compute_aabbs` pushed the two tree entry points over the limit even though neither touches three of the four buffers it needs | divergence probe | split into `lbvh_build.wgsl` (tree, two buffers) and `lbvh_aabb.wgsl` (AABB, four); each pipeline also needs its own bind group, since wgpu keys those to the exact pipeline |
| `read_las`, the PLY reader and two PCD paths each reserved output memory using a count read from the file's own header, so a few bytes could request tens of gigabytes — `element vertex 40000000000` is about 300 GB, and one PCD header cost three separate allocations | IO audit | clamped to 200M, well above any real cloud and low enough to keep the reservation under ~2.4 GB; the vectors still grow for genuinely large files |
| The CPU `warp` guarded its bilinear sample with `sx < w - 1`, excluding the last valid column, so the final column of every destination row was never written and kept the zero it was initialised with — while the GPU sampled it correctly. 89 of 1961 pixels on a 53-wide identity warp, and the CPU was the wrong one | table-driven CPU/GPU parity sweep | bound is inclusive, and `x1`/`y1` are clamped since `x0 + 1` would otherwise index past the row |
| `hough.wgsl` and `hough_circles.wgsl` read their input with the packed-u8 idiom while the host binds a float tensor, so almost every pixel looked black and was skipped before voting | parity sweep | both shaders read `array<f32>` directly; this is the third instance of that class (after Canny and `match_template`) |
| The half-extent in that fix was computed as `round(patch_size * level_scale)` — 31 for the base case rather than 15 — silently tightening the border test by sixteen pixels and dropping a ring of keypoints **with the feature flag off**. TUM fr1_desk fell 23/40 -> 8/40 on a clean tree, deterministically, and read as a property of the data | re-running the full sweep after the descriptor work, and noticing a number had moved with no command change | Corrected to integer division of the diameter; pinned by a regression test asserting the default path is unaffected |

## Math and geometry

| Defect | Found by | Fix |
| --- | ---: | --- |
| `bessel_i0` used a leading constant of ~0.05 where `1/√(2π)` is required — I₀(5) was ~3× low | comparing against reference values | Correct polynomial |
| `bessel_k0` had a doubled log term and a spurious factor of `x` | reference values | Both corrected |
| `expi` treated Ei as odd; `Ei(-1)` returned −1.895 instead of −0.219 | reference values | Uses `Ei(-x) = −E1(x)` |
| `gamma` returned `+∞` for negative non-integers | reference values | Reflection formula |
| `eigh` sorted with `partial_cmp().unwrap()`, panicking on NaN eigenvalues | audit | `total_cmp` |
| `lu_decompose` recovered the permutation by row-matching with a fixed 1e-10 tolerance, so large-magnitude matrices silently got identity pivots | audit | Reads nalgebra's LU permutation |
| `Tensor::slice` validated range ends but not starts, so a reversed range panicked | audit | Validates both |
| `polygon_iou` assumed counter-clockwise input, giving wrong or zero IoU for clockwise polygons | audit | Winding normalised |
| `estimate_5point` stored degree-3 monomials while its consumer read the degree-2 basis, referenced an unbound `z`, and had an unreachable fallback — it returned wrong matrices | audit | Returns an explicit error rather than a plausible-looking wrong result; no callers |
| `recover_pose_from_essential` scored candidates from `i32::MIN`, so a decomposition with *zero* points in front of both cameras could win | measuring registration failures | Requires positive depth for at least half the points |
| Homography DLT written four times and Hartley normalisation five times, one copy skipping normalisation entirely | dedup audit | One `cv-calib3d::dlt`; the others delegate |
| Three 8-point fundamental solvers | dedup audit | One implementation |
| `geometry2d::polygon_intersection` used Sutherland–Hodgman, only correct for a convex clip ring, silently wrong for concave input | dedup audit | `geo` boolean ops |

## Mapping and reconstruction

| Defect | Found by | Fix |
| --- | ---: | --- |
| The sparse BA Jacobian was built by central differences — for every parameter of every observation it cloned the whole parameter vector and recomputed every residual (~33,000 full rebuilds for one Jacobian); 14 views of TUM never finished and a 60-view run hit 6.4 GB | timing a 14-view run that exceeded 5 minutes | Analytic Jacobian, O(observations) |
| The analytic derivative initially used the textbook `expm([w]ₓ)`, which is **not** what nalgebra computes (it uses the quaternion form `I + 2aK + 2K²`); caught by a test comparing against finite differences | the test I wrote for exactly this | Correct closed form, pinned by a permanent test |
| The sequential solver then materialised `JᵀJ` densely and Cholesky-factored it — O(parameters³) on an empty system | profiling after the Jacobian fix | Sparse normal equations solved with CG |
| Model selection compared the essential score (Sampson error in *normalised* coordinates) against the homography score (transfer error in *pixels*) — different measures and units, so the comparison was arbitrary. Pair (5,6) on ETH3D courtyard has **931 essential inliers**, more than almost any accepted pair, and was rejected for scoring 0.056 against 0.42 — splitting the reconstruction in two disconnected halves | listing which pairs were rejected and noticing the best-supported one was not among them | Compares **support**: degeneracy needs the homography to explain substantially more matches *and* the essential model to be too weak to seed |
| The track graph was frozen at seed time, so a view registered later brought none of its own tracks and the map could never grow past the seed's pair graph | measuring the map against the features it was built from | Registering a view extends the tracks of every verified pair incident on it |
| The map's descriptor table held one copy per observation, so a landmark seen in five views contributed five near-identical candidates and the ratio test discarded everything else (a 7,000-descriptor table of duplicates gave 37 matches where distinct landmarks gave 485–712) | matching counts with identical table sizes | One descriptor per landmark |
| Local BA took every landmark visible to its cameras, so the "local" problem grew with the whole map — 400 points at the start of a 60-view run, 841 by the end, 79 ms → 174 ms | timing the local solve | Capped and ranked, with the sweep recorded |
| The F-matrix RANSAC threshold was fixed at 1.5 px regardless of resolution; at 6208×4134, views reached PnP with 46–48 correspondences and 2–7 inliers | ETH3D per-view failure reasons | Exposed as `--f-threshold`; 4 px raised verified pairs 19 → 24 of 39 |
| The COLMAP loader applied the first camera's intrinsics to every view, on scenes spanning up to six DSLRs with different focal lengths | ETH3D electro run | Restricts to the largest single-camera group and reports the drop |
| Registration could exclude every pair on a shallow planar capture (13 of 13 on courtyard), leaving no map at all | inspecting the planar counts | Falls back to the best pair when nothing else qualifies |
| `pointer_safety_tests` transmuted `f32` to `f32` and asserted `size_of`, so it could not catch a removed type guard | code review | Rewritten to drive real entry points |

## I/O

| Defect | Found by | Fix |
| --- | ---: | --- |
| PCD: `SIZE`/`COUNT` shorter than `FIELDS` caused an out-of-bounds index; stride and field offsets were derived from differently-clamped vectors, so `rgb` could be read out of bounds | fuzz-shaped inputs, audit | Header validated; both derived from the same clamped vectors |
| PCD: `stride * count` read from the file with no cap — overflow or a multi-GB allocation | audit | `checked_mul` plus a sanity cap |
| LAS: GPS time silently discarded, because the writer picked point format 0/2, neither of which carries it | round-trip test | Format chosen from the data present |
| `Color::hex` panicked on `"#fff"` or any short string | audit | Length checked, 3-digit form handled |
| `ndarray_to_points` indexed columns 0..3 without checking, so a `(N,2)` array panicked instead of raising | audit | Validated |
| `Mog2::new(_, 0, ..)` produced `alpha = inf`, poisoning the model with NaN | audit | History clamped to ≥1 |
| Mean-shift centroid cast to `u32` after drifting negative, wrapping to ~4.29e9 | audit | Clamped |

## Data structures

| Defect | Found by | Fix |
| --- | ---: | --- |
| `k_nearest_neighbors(_, 0)` called `peek().unwrap()` on an empty heap | audit | Returns empty |
| Ball-pivoting circumcenter used invalid barycentric weights — not equidistant from the triangle, so pivots were wrong | checking the output against a right triangle | Correct weights, with an equidistance test |
| Higher-order-SH Gaussian PLY panicked (3-coefficient `coeffs` indexed at `3..`) | audit | Resized before copy |
| `SphericalHarmonics::new` allocated `(degree+1)²` while `dc()` read 3 and `eval()` read 12 | audit | Allocates `3(degree+1)²` |
| `GaussianCloud::remove` wrote an out-of-range index into `active_indices` | audit | Moved index registered |
| Raycasting indexed `vertices[0]` on an empty mesh; mesh-to-mesh distance divided by zero | audit | Early returns |
| TSDF `estimate_normal` normalised a zero gradient, producing NaN | audit | Zero guard |
| `HashGrid::radius_search` only inspected a 3×3×3 stencil, incomplete when `radius > cell_size`, and double-counted colliding buckets | audit | Stencil from `ceil(radius/cell_size)`, deduplicated |
| `downsample_depth` underflowed on a zero-sized frame | audit | `saturating_sub`, early return |
| `build_voxel_grid` filled all-black due to a broken `#[cfg(not(test))]` guard | code review | Removed the attribute |

## Performance

| Defect | Found by | Fix |
| --- | ---: | --- |
| `fast_detect` scanned every pixel of every pyramid level serially — 40 ms/frame, with no rayon in the file at all | measuring the pipeline before optimising anything | Parallel over rows, direct slice indexing; 26 ms |
| ORB descriptor extraction looped over keypoints serially | same measurement | Parallel, order-preserving |
| Local BA ran every ten registrations globally, so early error was never repaired locally — 18.1° rotation error at 150 views | measuring the 150-view run | Local BA after each registration: 18.1° → 3.8° |
| `SparseLMSolver` sent the whole Jacobian to the GPU as f32 twice per CG iteration for a system where the CPU is faster and more precise | the 122× BA speedup | Native f64 path below a size threshold |
| The incremental registration loop could spin forever: a view was retried when its correspondence count *changed*, and retriangulation could move that count back to a value already seen | tracing the loop with progress output | Bounded attempts and idle passes |

## Cross-cutting

- **26 crate roots now carry `#![forbid(unsafe_code)]`**, up from 3. All `unsafe` is confined to `cv-hal` (GPU reinterpretation) and `cv-distributed` (mmap, futex, flock). `unsafe impl Send/Sync` for the shared-memory coordinator was **removed** — `memmap2` already provides both, so the hand-written impls were no-ops whose comment stated the wrong reason.
- **`is_process_alive` cast a `u32` pid to `i32`**; a large value became negative and `kill(-n, 0)` probes a process *group* rather than failing.
- **`shutter`/futex wake measurement measured the wrong interval** — the waiter's whole elapsed time minus the main thread's, so it included thread spawn and coordinator creation. It reported 20–40 ms against a 20 ms bound on macOS CI. Now synchronises on the waiter parking and compares the best of five trials: **17.6 µs**.

## Parsers, and a window that drew nothing

- **`f64::from_str` accepts `inf`, `nan`, `1e400`.** These are valid IEEE-754 spellings, not syntax errors, so every dataset reader that only checked the parse accepted them: one malformed column produced a pose or timestamp of `inf` that flowed silently into every metric computed from the trajectory. EuRoC, TUM, KITTI and COLMAP now refuse such a file, naming the line and the offending text. The same class was fixed in PLY, and in STL, OBJ and PCD vertex coordinates.
- **A PLY `element face` block was read as further vertices.** The reader had no element model - it read `num_vertices` rows and stopped, which is only correct when the vertex element happens to be last. A face row's first three integers are shaped exactly like coordinates, so `3 0 0 1` became a vertex with no error anywhere.
- **`element vertex 40000000000` in a 90-byte PLY reserved ~300 GB.** Clamping the count to 200M still reserved 2.4 GB, because the clamp bounded the *claim* rather than the data. A vertex row is at least 6 bytes, so the reservation is now bounded by what could exist: measured growth for that file drops from 2.4 GB to 2 KB.
- **A singular KITTI rotation was accepted as a pose.** A zero row, a truncated file or a header read at the wrong offset has no inverse and cannot be a rotation, but every downstream consumer treats it as one. Now rejected on determinant.
- **Two tests proved nothing.** `ply_vertex_count_is_bounded_before_allocation` reimplemented a 200M clamp locally and asserted its own constants; `ply_hostile_vertex_count` measured process-wide `VmSize`, which also counts the test binary's own allocations and so could not detect the fix it was written for. Both now assert the property they were named for.
- **The viewer painted its background *after* the render callback**, putting an opaque rectangle over the point clouds. Black window, no error.
- **`look_at` built the camera basis 180° out** (`z = eye - target`, then `up × z`), so the eye mapped to its own position and the cloud was drawn behind the camera. It compiled and drew *something*. Verified numerically before the fix: eye → origin, target → (0,0,−5).
- **Wheel zoom used `zoom_delta`**, which is a pinch factor and is exactly `1.0` for an ordinary notch - so the wheel did nothing, and scrolling down drove the eye through the object. Now read from `raw_scroll_delta.y`.
- **The orbit was dead**: `Response::drag_delta()` is only non-zero when the widget captured the pointer, and a rect inside a `CentralPanel` never does.
- **The demo opened empty**, which is indistinguishable from a broken renderer.
- **Two smoke tests waited on processes that never exit.** `demo` is a GUI app; `orbdiag` sweeps the full dataset. Both used `.output()` or a completion deadline and could only pass if the example happened to finish first. `orbdiag`'s own comment said it "is expected to run past any reasonable timeout", and then the test waited for it. Suite time 71s → 8s.

## The viewer shader, found by reading pixels back

- **The matrix multiply was transposed.** The host's `look_at` builds a row-vector convention matrix with the translation in the last *row*; WGSL's `mat4x4` is column-major, so `view[i]` is column `i` and `view[i][j]` is row `j`, column `i`. The shader summed `view[r][c] * world[c]`, computing M-transpose — its own comment said "row-major multiply, so the matrix is indexed view[col][row]" and then indexed it the other way. Verified: the old form gives eye-space `(0.47, 0.60, −0.16)` where the intended answer is `(0, 0, −2.5)`.
- **The pipeline had never drawn a pixel.** A readback test that renders into an offscreen texture and counts lit pixels is what found this; every value-level test passed throughout.
- **A depth term I added was not the fix, and I claimed it was.** With `near = 0.01, far = 1000` this scene lands at z_ndc 0.994–0.997 and rasterises nothing, while a constant 0.5 renders everything. That value is *inside* the [0, 1] range, the constants are finite in f32, and rescaling near/far to the scene does not help — so the cause is **not identified**. The shader uses a conservative mid-range depth with that recorded at the call site, rather than a derivation I cannot support. There is no depth attachment, so depth only needs to stay in range.
- **The first version of that test asserted "more than zero lit pixels" and was useless** — the transposed shader renders 39 pixels and would have passed. The threshold is 10,000, measured: 75,854 correct, 39 transposed, 0 with the depth term wrong. Verified by reintroducing the bug.

## A hang, which is worse than a wrong answer

- **`solve_dlt_homography` never returned on NaN input.** The design matrix
  carries NaN into LAPACK's bidiagonalisation, whose convergence test is a
  *comparison* — and every comparison against NaN is false, so it never
  converges. Found by a scratch probe: a single NaN observation fed to
  `calibrate_camera_planar` hung, and the workspace test suite timed out after
  360 s waiting on it. `solve_dlt_fundamental` had the identical exposure and is
  now guarded the same way.
- Verified: against the unfixed code the new tests time out; with the guard they
  pass in under a second.
- **The exposure was still live, and two sites were reachable from the public
  API.** The remaining 24 `.svd(true, true)` call sites were re-measured rather
  than assumed: a 3×3 with one NaN returns (with NaN singular values), a 4×4 with
  one NaN still had to be killed at 200 s. `solve_pnp_dlt` and
  `triangulate_points` each hung on a *single* NaN correspondence among twelve —
  a malformed detector output, which is exactly how this is reached in practice.
  Both, plus `solve_pnp_ransac` and `undistort_points`, now reject non-finite
  input at the boundary where caller data enters, rather than at each SVD site: a
  guard that has to be remembered in eight places is a guard that will be missed in
  the ninth. `cv_math::linalg::try_svd` adds the guarded primitive for new code.
  Six tests; against the unfixed code the suite **hangs** rather than failing,
  which is the honest signal for this class.
- **`undistort_points` returned `Ok([NaN, 10.0])`** for a NaN pixel — not a hang,
  but a NaN coordinate handed back as a successful result, which then flows into
  every downstream metric computed from it.
- **The survey continued past those, and found two more.** `find_essential_mat`
  hung on one NaN correspondence among ten, the same mechanism, now guarded in
  the shared 8-point solver so the RANSAC wrappers are covered too.
  `registration_gnc` did not hang — it was quieter. A NaN coordinate makes that
  point's residual NaN, `exp(-NaN)` is 0, so every robust loss gave it weight 0
  and it contributed nothing. Measured, clouds differing only in that coordinate:
  `finite -> fitness 1.000000`, `one NaN -> fitness 0.983333`, both `Some`, both
  with a correct identity transform. The answer was right, but the caller was
  told a malformed cloud registered cleanly, and the vanished point (1 of 60) was
  visible only by comparing 0.983333 against 59/60 — an accident of `exp`, not a
  contract, and a loss whose weight does not saturate would instead poison the
  covariance.
- **`RobustLossType` was a public parameter of `registration_gnc` that no caller
  could name**, since it was not re-exported from the module. Now exported.
- **`cv_sfm::triangulate_point_dlt` hung too** — same 4×4 system, same
  `.svd(true, true)`, and confirmed by measurement rather than by inference:
  finite input returned `[1.032, 0.780, 2.000]` at the depth the geometry implies,
  and one NaN pixel had to be killed at 120 s. The two triangulations are
  near-identical implementations in different crates; only the calib3d one had been
  guarded.
- **`Trajectory::ate` returned `rmse = NaN`** for a trajectory holding one
  non-finite coordinate, with every other statistic NaN too. It returns a struct of
  plain `f64` with no error channel, so a malformed input and a genuinely poor
  registration were indistinguishable — and since `NaN > x` is always false, any
  `if ate.rmse > threshold` quietly took the other branch. Not reachable from a
  dataset file (the readers reject `inf`/`nan`/`1e400` at parse time), but
  reachable from a trajectory assembled in memory. `AteResult` now carries
  `is_valid`.
- **The survey is complete.** All 24 `.svd(true, true)` sites were examined; the
  remaining ones (`aruco::solve_pnp_dlt`, `eval::umeyama`, the registration
  covariance builders) are private helpers reached only through callers that
  already validate, and neither hung nor returned a wrong value under the probes
  used.

## One idiom, three more sites

The LAS `f64::MIN` defect turned out to be an idiom rather than a one-off, so the
whole tree was searched for it. Fourteen sites seed a running `max` with `f32::MIN`
or `f64::MIN`. Ten already guard the empty case and are correct. Three were not:

- **`Aabb::empty()` had no way to express "empty"** — it seeded
  `min = f32::MAX, max = f32::MIN`, so an Aabb that never received a point reported
  `min = 3.403e38 > max = -3.403e38` and nothing distinguished it from a real box.
  Three measured consequences: `from_points(&[])` returned exactly that;
  `longest_axis()` computed `max - min = -6.8e38`, which **overflows to -inf** in
  f32, so both comparisons went false and it returned axis 2 for *every* input,
  letting a degenerate face (repeated vertex index) silently pick the split axis of
  a subtree; and `intersect_ray` returned `true` for an empty box from **every**
  origin and direction, because the overflow made `tmin`/`tmax` non-finite and the
  final comparison was decided by NaN ordering. Emptiness is now an explicit field.
- **`write_glb` wrote the sentinel pair into the file.** An empty mesh produced a
  876-byte GLB whose POSITION accessor read
  `"min": [3.4028235e38, ...], "max": [-3.4028235e38, ...]`. glTF requires both
  when they are written, so an empty mesh now records a degenerate box at the
  origin — valid, and the file drops to 644 bytes.
- **`bounding_box.rs` projected onto its own axes with the same seed**, guarded on
  a non-empty `points` slice, so it is correct as written.

## Six float comparisons that could panic

Searched for `partial_cmp().unwrap()` — the panic the log records for `eigh` on
NaN eigenvalues. Six sites. **Two of the four reachable ones were checked and
found *not* to panic**, which is the reason to measure rather than assume:

Rust's `f32::min`/`max` return the **non-NaN** operand, so
`f32::MAX.min(f32::NAN) == f32::MAX`. A sentinel accumulation therefore
*swallows* a NaN coordinate — `compute_adaptive_voxel_size` does exactly this,
and a probe with one NaN among 500 points did not panic. My first instinct was
wrong, and the comment committed alongside the fix says so rather than claiming
a crash that does not happen.

- **`convex_hull` panicked on one NaN coordinate.** Here the NaN went straight
  into a comparison, not through a `min`/`max`: `finite -> 7 vertices`,
  `one NaN x -> PANIC`, `a NaN point -> PANIC`. Swapping in `total_cmp` stops the
  panic but returns a *wrong* hull — a NaN has no position, the cross-product
  test compares against NaN, and every such comparison is false, which gave nine
  vertices from eight points where finite input gives seven. Non-finite points
  are now dropped, so the hull is computed over the points that have a position.
- **The STRtree sorts** compared the same way and are now `total_cmp`.
- `compute_adaptive_voxel_size`'s span sort is hardened to `total_cmp`. This is
  defence in depth, not a fix: its inputs are provably finite today, but a span
  computed as `n / (sx*sy*sz)` overflows for a degenerate cloud, and an unwrap on
  a comparison that could see NaN is a panic waiting for the input that produces
  it.

- **`find_essential_mat_ransac_handles_outliers` is flaky.** It runs 600 RANSAC
  samples with no seed, so the sampled hypothesis varies run to run; the
  recovered translation direction is near-degenerate on that synthetic scene and
  the assertion fails intermittently. Pre-existing, and left alone here.

## OPEN, measured, not fixed: `gaussian_blur` at `sigma = 0` returns an all-zero image

Found while judging a `ctx`-variant test whose expectation turned out to be
mis-framed. The observation under it is real. Measured on a constant 123 image:

```text
sigma 0:   non-ctx -> min 0     ctx -> min 0
sigma 0.5: non-ctx -> min 122   ctx -> min 122
sigma 1:   non-ctx -> min 123   ctx -> min 123
sigma 3:   non-ctx -> min 123   ctx -> min 123
```

Two things follow, and only the first is a defect.

**`sigma = 0` destroys the image.** A zero-sigma Gaussian is a delta, so the
correct output is the input unchanged; instead the whole image becomes `0`. Both
paths agree, so this is a property of the shared kernel construction rather than of
the `_ctx` entry points — which is why the test that surfaced it, asserting of a
`ctx` variant "the blur broke a constant image" while its non-`ctx` reference
behaved identically, was the wrong framing.

**`sigma = 0.5` off by one is not a defect.** `123 -> 122` is rounding in a
near-delta kernel, and it is stable across both paths.

Left unfixed because the fix belongs in the shared kernel builder and changes
behaviour for every caller of both entry points — a decision about whether
`sigma = 0` means "identity" or "invalid". The latter would want an error channel
that `gaussian_blur(&GrayImage, f32) -> GrayImage` does not have.

## Sweeping the whole workspace for the fabrication pattern

Grepped every `unwrap_or`/`unwrap_or_else` whose fallback constructs a matrix or
vector, then triaged each by hand. The pattern is worth sweeping because it has
already produced five real defects — but *most* hits are correct, and saying which
is the point of this section.

### Fixed from this sweep

- **`hal` GPU undistort** — `Matrix3::identity()` for `Inv(NewK)` and `Inv(R)`.
  Measured `try_inverse() -> None`, `det = 0`, fallback exactly the identity, so the
  intrinsics normalisation vanished and the call returned `Ok`.
- **`3d` TSDF integration** — `Matrix4::identity()` for a singular `extrinsics`, so
  every world voxel was treated as already in camera space. Measured: a world voxel
  passed through unchanged.
- **`video` Kalman update** — `SMatrix::zeros()` for a singular innovation
  covariance, which means *ignore the measurement*. Measured `state moved
  0.000e0`. See its own section above.

### CHECKED AND CLEAN — deliberately not changed

- **`registration::colored`** — `unwrap_or(Vector3::zeros())` for a missing surface
  normal. A zero normal contributes nothing to the point-to-plane Jacobian, which
  is the correct amount when there is no normal, and the alternative (`normalize`
  on a zero difference vector) is NaN. A zero *normal* is a meaningful value; a
  zero *inverse matrix* is not. Already carries the reasoning.
- **`3d` TSDF gradient normal** — `try_normalize(1e-8).unwrap_or(Vector3::zeros())`
  for a locally flat region. Zero is the honest "no gradient here" and the caller
  can test for it. Documented.
- **`sfm::mapper` homography fallback** —
  `unwrap_or((Matrix3::identity(), vec![false; n]))`. This *looks* like the
  fabrication pattern and is not: the all-false mask makes `h_inliers == 0`, and
  the planar decision requires `h_inliers > f_inliers`, which zero cannot satisfy.
  So the fallback can only ever degrade to "not planar" — the safe direction. The
  identity `homography_score` is recorded in the diagnostic but does not drive the
  decision. A fabricated value that can only make a model *less* likely to be
  chosen is not the same hazard as one that is chosen.
- **`io::las_to_point_cloud`** — `unwrap_or_else(|_| PointCloud::new(points))`
  drops colours when `with_colors` rejects a length mismatch. Lower severity than
  the others for a specific reason: the degraded result has `colors == None`, which
  the caller *can* detect after asking for colours. It is a detectable loss, not a
  well-formed lie. Left as-is; noted because the LAS reader's own consistency
  checks are `debug_assert_eq!` and so compiled out in release.

### The distinction that separates these

A fallback is only in this bug class when the substituted value is **well-formed
and indistinguishable from a real answer** — an identity or zero matrix, which
downstream treats as data. If the degraded result is *detectable* by the caller
(`Option::None`, an out-of-band flag, a strictly-worse model score), it is a
design choice rather than a lie, and changing it is not automatically an
improvement.

## CORRECTED: the LM/CG mismatch, and how my first account of it was wrong

An earlier entry here claimed `SparseLMSolver::minimize` in `cv-optimize` discards
`CgSolver`'s non-convergence report. **That was wrong in two ways**, and both
corrections matter more than the original claim.

**Wrong #1 — the wrong function.** `SparseLMSolver::minimize` calls
`solve_lm_step(&j, &r, lambda)?` and `solve_lm_step` propagates its errors with
`?` too. `cv-optimize` is clean; the `unwrap_or_else(|_| DVector::zeros(n_params))`
lives only in `sfm`'s `bundle_adjust_sequential`, which duplicates the LM loop
instead of calling `SparseLMSolver`. I had read the two loops and conflated them.

**Wrong #2 — the wrong mechanism.** I framed it as "the linear system is never
solved". Measured, by running the case with the error printed:

```text
conjugate gradient did not converge in 200 iterations (tolerance 1.000e-10):
||A x - b|| = 1.301e-7
```

That is not a broken system. It is a *solvable* one that CG brought to `1.3e-7`
against a strict `1e-10` tolerance. The step was almost certainly usable — LM
validates steps by whether the cost decreases, not by whether the linear solve was
exact.

**The real shape is an interface problem, not a local patch.**
`CgSolver::solve` returns `Result<DVector<f64>, String>`, so an unconverged solve
*cannot* hand back its approximate iterate. That leaves the caller two options:
propagate `Err` and abandon an optimisation that was working, or fabricate — and it
fabricates a zero step, which is also exactly what a rejected step looks like.

**I tried the local patch and it was wrong.** Changing
`bundle_adjust_sequential` to return `false` on a failed solve turned a
*legitimate* refinement into a reported failure — `map_ba`'s
`local_ba_writes_back_only_the_selected_subset` caught it, because its config uses
`ba_max_iterations: 1` and the single solve lands at `1.3e-7`. An existing test
caught a bad fix, which is the point of having them. Reverted in full; `sfm` is
back to 0 failures.

The fix needs an interface that can carry a partial solution — `Ok(iterate,
converged: bool)`, or a tolerant-enough default and a documented contract — which
is a decision for `cv-optimize` rather than a patch at the call site. Left
unfixed deliberately, with the reasoning here.

## Sweeping the whole workspace for the fabrication pattern

Grepped every `unwrap_or`/`unwrap_or_else` whose fallback constructs a matrix or
vector, then triaged each by hand. The pattern is worth sweeping because it has
already produced five real defects — but *most* hits are correct, and saying which
is the point of this section.

### Fixed from this sweep

- **`hal` GPU undistort** — `Matrix3::identity()` for `Inv(NewK)` and `Inv(R)`.
  Measured `try_inverse() -> None`, `det = 0`, fallback exactly the identity, so the
  intrinsics normalisation vanished and the call returned `Ok`.
- **`3d` TSDF integration** — `Matrix4::identity()` for a singular `extrinsics`, so
  every world voxel was treated as already in camera space. Measured: a world voxel
  passed through unchanged.
- **`video` Kalman update** — `SMatrix::zeros()` for a singular innovation
  covariance, which means *ignore the measurement*. Measured `state moved
  0.000e0`. See its own section above.

### CHECKED AND CLEAN — deliberately not changed

- **`registration::colored`** — `unwrap_or(Vector3::zeros())` for a missing surface
  normal. A zero normal contributes nothing to the point-to-plane Jacobian, which
  is the correct amount when there is no normal, and the alternative (`normalize`
  on a zero difference vector) is NaN. A zero *normal* is a meaningful value; a
  zero *inverse matrix* is not. Already carries the reasoning.
- **`3d` TSDF gradient normal** — `try_normalize(1e-8).unwrap_or(Vector3::zeros())`
  for a locally flat region. Zero is the honest "no gradient here" and the caller
  can test for it. Documented.
- **`sfm::mapper` homography fallback** —
  `unwrap_or((Matrix3::identity(), vec![false; n]))`. This *looks* like the
  fabrication pattern and is not: the all-false mask makes `h_inliers == 0`, and
  the planar decision requires `h_inliers > f_inliers`, which zero cannot satisfy.
  So the fallback can only ever degrade to "not planar" — the safe direction. The
  identity `homography_score` is recorded in the diagnostic but does not drive the
  decision. A fabricated value that can only make a model *less* likely to be
  chosen is not the same hazard as one that is chosen.
- **`io::las_to_point_cloud`** — `unwrap_or_else(|_| PointCloud::new(points))`
  drops colours when `with_colors` rejects a length mismatch. Lower severity than
  the others for a specific reason: the degraded result has `colors == None`, which
  the caller *can* detect after asking for colours. It is a detectable loss, not a
  well-formed lie. Left as-is; noted because the LAS reader's own consistency
  checks are `debug_assert_eq!` and so compiled out in release.

### The distinction that separates these

A fallback is only in this bug class when the substituted value is **well-formed
and indistinguishable from a real answer** — an identity or zero matrix, which
downstream treats as data. If the degraded result is *detectable* by the caller
(`Option::None`, an out-of-band flag, a strictly-worse model score), it is a
design choice rather than a lie, and changing it is not automatically an
improvement.

## CONFIRMED, NOT YET FIXED: the LM step discards the solver's own failure report

Two halves of an intent mismatch, both readable in the code but in *different*
crates, which is why neither side looks wrong on its own.

`CgSolver::solve` was deliberately written to report non-convergence, and its
comment says why, naming this exact caller:

> The iteration cap is not convergence: an unconverged `x` is a wrong answer for
> `A x = b`, and the caller (a Levenberg-Marquardt step, for instance) has no way
> to see it. Report it.

The Levenberg-Marquardt step does not see it:

```rust
let delta = cg.solve(&device, &lhs, &neg_jtr)
    .unwrap_or_else(|_| DVector::zeros(n_params));
```

So the report is thrown away and replaced by a **zero step**. Because `delta` is
zero, `next_params == current_params` and therefore `next_err == current_err`
*exactly*, so the strict `next_err < current_err` test classifies it as a
**rejected step** — the same branch a legitimate LM rejection takes. The loop then
increases damping and retries, up to 12 times, and exits with

```rust
if rejections >= 12 || !lambda.is_finite() {
    break; // stalled: keep best parameters
}
```

`stalled` is the wrong word for "the linear system never solved", and the code has
no way to tell the two apart. The observable outcome is that bundle adjustment
runs, makes no progress, and returns un-optimised parameters as though it had
converged — the caller cannot see it, because `bundle_adjust` returns `()`.

There is a second, subtler consequence. `bundle_adjust` documents an explicit
fallback for solver failure:

```rust
if bundle_adjust_ctx(state, config, &group) { return; }
// Ctx path unavailable (no compute device / solver failure):
// fall through to the sequential CPU implementation.
```

`bundle_adjust_ctx` maps `Err` to `false` for exactly that reason. But an inner
solve failure never reaches `Err`, because the LM step absorbed it — so the
fallback that exists for this case **never fires**, and the caller keeps the
un-optimised result instead of retrying on the CPU.

Both `CgSolver::solve` and `SparseLMSolver::minimize` live in `cv-optimize`, and
`bundle_adjust_sequential` in `sfm` repeats the same `unwrap_or_else(zeros)`.
Confirmed by reading both sides; **not yet fixed**, because the two files were
owned by a concurrent agent. The fix needs a deliberate decision, because the
sequential path has no channel: `bundle_adjust` returns `()`.

Worth noting as *checked and clean*: a zero `delta` cannot fake convergence. The
`delta.norm() < convergence_threshold` break lives inside the `next_err <
current_err` branch, and a zero delta makes that test false, so it is unreachable
from a failed solve.

## Two claims I made that measurement disproved

Recorded because the corrections are the substance, and because both errors
shipped inside a commit before being caught.

- **"`U·Wᵀ·Vᵀ` is not a rotation at all."** Wrong. `Wᵀ ≠ ±W`, but the two
  permutations involved sit symmetrically in `U Wᵀ Vᵀ = (U P)(W)(Pᵀ Vᵀ)` and
  cancel, leaving a proper rotation — measured `max ||R Rᵀ − I|| = 1.6e-15`,
  `min det = 1.0`.
- **"It duplicates `U·W·Vᵀ`, so nothing was mis-recovered."** Also wrong, and this
  one mattered. The two are complementary — 203/400 and 197/400 recoveries of the
  true rotation — and together they cover essentially every input, which is why
  the original worked. A fix built on my second claim replaced `Wᵀ` with
  `diag(1,-1,-1)`, which recovers the pose in **0/400** cases, and cost three
  valid scenes in six: "no valid pose candidate" while the true pose scored 12/12
  on cheirality when substituted by hand.

A third, found by an agent and initially disbelieved by me: **within one
`a` entry's candidate window the `|difference|` values form a V, not a sorted
sequence.** With `a = [1.0]` and `b = [0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]` they are
`[1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]` — descending then ascending. So a k-way
merge over per-`a` windows, treating each as a sorted run, cannot work, however
careful the merge. Three implementations were discarded on that reasoning before
the fourth succeeded.

## A test that pinned the bug it was named for

`tum_associate_finite_timestamps_with_infinite_tolerance_never_match` asserted
`matches.is_empty()` and its own failure message began `KNOWN BUG: ... returned 50
matches; the documented 'every pair within max_dt' rule does not hold`. Under a
name that read as a specification rather than a bug report. Inverted.

Writing its replacement took three attempts, and **the first two controls were
wrong**: `max_dt = 1.0` does not shrink the result (consecutive integer stamps,
every adjacent pair qualifies) and neither does `0.5` (each `a[i]`/`b[i]` pair is
at difference 0, inside any positive tolerance). No positive tolerance can
exclude a zero-difference pair, so a shrinking control has to come from the
negative side. Both dead ends are in the test.

## Two tests that failed CI while passing locally

Both were GPU-only and used `expect("no GPU adapter")`. They pass here — this
machine has a GPU — and failed on every runner without one, and nextest cancels a
binary's remaining tests on first failure, so one took three more down with it.
The convention elsewhere in the crate (`perf_tests.rs`, `gpu_parity_tests.rs`,
`gpu_feature_detection.rs`) is an early return; these now match it.

Worth stating plainly: a test that cannot run in CI reporting *failure* there is
how people learn to ignore red.

## Turning the probes into tests, and what that immediately found

`tests/gpu_parity_tests.rs` carried eleven `probe_*` tests that computed a CPU/GPU
difference and **printed** it. Not one asserted. The file header calls this
"visible rather than silent", but visibility needs a human reading output — so all
eleven passed unconditionally, and the bug log's credit to this file as the
evidence that found the multichannel-blur, stereo-compile and LBVH defects was
misplaced: those were found by *reading printed numbers by hand*.

Eight are now assertions. **The first thing that did was find a live divergence:**
the CPU stereo matcher left border pixels at `0.0` while the GPU shader writes
`-1.0`. Nothing downstream broke — `disparity_to_depth` filters `d < 0.5`, which
catches both — but a caller reading the raw map cannot tell a border from a real
zero-disparity match.

Two more needed judgement rather than mechanical conversion, and both are the kind
of thing that goes wrong quietly:

- **`pyramid_down` is a genuine algorithmic divergence.** The CPU subsamples at
  even offsets after a blur; the GPU runs a full bilinear `resize`. Different
  filters. On uniform random noise they differ by ~55%, which measures the noise
  rather than either implementation, so the probe now uses a smooth ramp where the
  number means something — and it still differs, so it is still asserted.
- **`GrayToRgb` is simply unimplemented on the GPU.** My first attempt turned that
  into a panic, which would have made a known-missing feature look like a
  regression. It now asserts only that the failure is `Not supported`.

A static test also replaces a check that could never have worked: `every_shader_compiles`
calls `create_shader_module`, which performs **no layout derivation**, and covered
only `crates/hal/shaders/*` while every offending file was under `src/gpu_kernels/`.
The new `hal/tests/shader_bindings_limits.rs` parses the WGSL, builds the call
graph, and asserts no entry point *reaches* more than 4 storage buffers — which is
what the device limit applies to. Verified by re-injecting three reachable bindings
into a fixed shader: the test fails, and passes again once reverted.

Also replaced `test_device_not_found_error`, which bound `default_cpu()` to `_` and
asserted nothing.

## What a parallel audit found, and what it got wrong

Eight agents swept the tree by crate. Every finding below was re-verified by hand
before acting, because **the audit's confident claims were sometimes false** —
and the two most confident ones were the false ones.

**Disproved on measurement** (both would have been "fixes" of a non-bug):

- *"A near-black PLY vertex round-trips to white: only 0 and 1 survive, every
  interior value is shifted by 255x."* False. Measured the real round trip: black,
  white, red, green, blue, orange and mid-grey all survive. `255 > 1.0` divides
  correctly, and byte 0 divides to 0. But the claim was **directionally right and
  one step short** — byte `1` does read as `1.0`, because `1/255` is a legitimate
  near-black *and* is what a normalised `1.0` looks like. That single byte was a
  real 255x inversion, now fixed by scaling on the *declared* property type.
- *"The 3-D spherical-flip test is wrong; 298/300 points survive on a closed
  sphere."* The follow-up agent traced this to the wrong file. The spherical-flip
  HPR implementation is **correct** (measured: 0 of 40, 80, 120 and 200
  back-hemisphere points survive at four sample sizes); the "almost everything
  survives" result came from a different, depth-buffer-based file.

**Real, and worse than reported:**

- **The PLY colour bug is only the bottom 1/255 of the range.** The primary-colour
  cases pass, which is exactly why a hand-written test misses it.
- **`spherical_jn` had the wrong sign parity for even `n`.** `j_n` is an
  (n+1)-order function, so `j_n(-x) = (-1)^n j_n(x)`, but every branch negated
  unconditionally. Right for odd `n`, wrong for even: `j_2(-1)/j_2(1) = -1`
  where it must be `+1`.
- **`log_gamma` returned NaN for every negative argument in (-1,0), (-3,-2), …**
  `pi/sin(pi x)` is negative there and `ln()` of a negative is NaN. The existing
  test covered `gamma(-1.5)` only — the one negative non-integer that happens to
  land on an interval where `sin(-1.5pi) = +1`.
- **`qr_solve` and `cond` used absolute thresholds** against quantities that scale
  with the input, so a well-conditioned system scaled by 1e-14 was declared
  rank-deficient.
- **The 3-D visibility depth-buffer marks occluded points visible.** Measured on a
  300-point closed sphere at 512²: 300/300 visible, including every back-hemisphere
  point. A projected point is a sub-pixel dot with no area, so occlusion depends
  only on sampling density vs. buffer resolution — and *raising* the resolution
  makes it worse, which is the tell. The existing test asserts only
  `visible_count < points.len()`, which 247/300 satisfies.
- **The 8-point essential solver was wrong at its own minimum sample.** nalgebra's
  thin SVD returns `min(m, n)` right singular vectors, so with exactly 8
  correspondences the null vector is not among them and row `nrows()-1` picked an
  arbitrary vector. Measured: worst Sampson residual 21.7 px at n = 8 against
  ~1e-12 px at n = 9, 12 and 16 — where the true `E` gives ~4.5e-14 px. Because
  `EssentialEstimator::min_sample_size()` is 8 and RANSAC samples exactly that
  many, *every* minimal sample in RANSAC was drawn from a broken solver. `dlt.rs`
  had already documented and fixed this for the homography and fundamental solvers.

**A note on how the two 3-D claims interacted:** the first agent ran out of turns
mid-claim and reported the headline number without isolating it; the second had to
re-derive which file it came from before the real defect could be separated from
the spurious one. Neither number was trustworthy on its own.

## A zero matrix is not a neutral fallback

`Gaussian::inverse_covariance` and `ProjectedGaussian::inv_cov_2d` both returned
`try_inverse().unwrap_or(Matrix3::zeros())`. The rasterizer reads the 2-D inverse
to evaluate `mahalanobis = a·dx² + 2b·dx·dy + c·dy²`, so every coefficient being 0
made `mahalanobis` 0 at **every** pixel of the splat's tile — and
`alpha = exp(-0.5·0)·opacity` is full opacity, not zero. Measured 0.8 at pixel
offsets of (0,0), (10,10), (100,100) and (1000,1000) alike. A splat with no
defined footprint painted its **entire tile** — the opposite of vanishing, and
something a render gives no hint of.

This is the `compute_hybrid` shape the log records — a fabricated value where a
caller cannot tell absence from data — but worse than the ones already fixed,
because the fabricated value here is read as a *valid conic* and yields a
confidently wrong image. Both now return `Option`, and the rasterizer skips the
splat.

`Gaussian::new` clamps scale to `>= 1e-4`, so this only arises from a splat built
outside that constructor: a deserialised file, or a field assigned directly.

## The identity is not a neutral fallback either

`CameraIntrinsics::inverse_matrix` returned `try_inverse().unwrap_or(Matrix3::identity())`.
The intrinsic matrix has determinant `fx · fy`, so it is singular exactly when a
focal length is zero — and the fields are public, so that value is constructible.
The identity is a *valid* matrix meaning "this camera has no intrinsics", so
every point is used as its own pixel coordinate. Measured with `fx = 0`: a pixel
`(123, 456, 1)` maps to `(123, 456, 1)` unchanged.

Eight call sites fed that into calibration, essential-matrix, PnP,
triangulation and structure-from-motion solves. `solve_pnp_dlt` with `fx = fy = 0`
returned a **confident pose** — a rotation and a translation — for an image whose
camera had no focal length at all. Now `try_inverse_matrix` returns `Option`, and
the four call sites that have an error channel use it; the ones that do not
(`fundamental_from_essential`, and the private `normalize_with_intrinsics` behind
`find_essential_mat`) keep the old behaviour, which the test file records rather
than leaves implicit.

## The same fallback, twice more, and the sibling that was already right

`init_undistort_rectify_map` and the stereo rectifier both did
`new_intrinsics.matrix().try_inverse().unwrap_or(Matrix3::identity())`, the same
defect as `inverse_matrix`, one call layer up. Measured on an 8×6 frame:

```text
valid:   map_x row 0 = [0,1,2,3,4,5,6,7]   map_y row 0 = [0; 8]
zero fx: map_x row 0 = [4,4,4,4,4,4,4,4]   map_y row 0 = [3; 8]
```

Every destination collapsed onto the principal point, so undistorting with that
map reproduces a single pixel across the entire frame. And all 48 destinations
still mapped *inside* the source image, so the result passed any bounds check —
the failure is invisible to every validation the function already had.

**`fisheye_init_undistort_rectify_map` was already correct** — same
responsibility, same call shape, and it had always returned an error for a
non-invertible matrix. That is the strongest available evidence that the pinhole
version was an oversight rather than a deliberate fallback, so it now matches.

The fisheye entry points were public in the `distortion` module but not
re-exported from the crate root, so no caller outside the crate could name them —
including the reference implementation. Now exported.

## A volume larger than its data, padded into looking valid

`tsdf_gpu::raycast_volume` fetched each element with `.get(i).unwrap_or(..)`, so a
`vol_dims` larger than the supplied arrays silently padded them. Two constants made
that padding look correct rather than absent:

- a missing TSDF value became `0.0`, and **0.0 is the iso-surface**, so unobserved
  space reported a surface hit;
- a missing weight became `1.0`, i.e. one observation.

The weight turned out never to be read at all — `tsdf_raycast.wgsl` declares the
buffer as interleaved `(sdf, weight)` but every mention is either the `sdf` half or a
comment naming the layout. So the padding was invisible even in shape.

Both arrays are now length-checked against the volume they describe, and the voxel
count uses `saturating_mul`: `(vol_x * vol_y * vol_z) as usize` wrapped to a small
number on overflowing dimensions, and the loop would then have iterated over *that*
instead, allocating a buffer of the wrong size.

## ICP, FPFH and rays

- **Point-to-plane ICP reported `fitness: 1.0` for a registration that never
  happened**, three ways: a singular `A` skipped the solve silently; too few
  correspondences fell through to the tail and reported `rmse = f32::MAX`; and
  the returned transform was the *live* iterate while the metrics tracked the
  best-*fitness* one — which freezes at iteration 0, overstating the error by 5-6
  orders of magnitude.
- **FPFH panicked on a zero radius** (`p.x / 0.0` → `i32::MAX` → `vx + dx`
  overflow) and **fabricated all-zero descriptors** for a cloud too sparse to
  estimate normals: 6 features, 0 with any non-zero bin, reported `Ok`.
- **`Ray::new` normalised unconditionally**, so a zero direction gave NaN that
  propagated silently through every later transform.
- **`VoxelGrid` indexed a point slice with unvalidated indices** — the same class
  as `icp_accumulate`, and it panicked on a stale index.
- **ORB's GPU path never ran non-max suppression**, and truncated to the
  candidate budget before it could.

## Counted

| | |
| --- | ---: |
| Defects fixed | **121+** |
| Commits | 460+ |
| Tests | 1,824 (from 1,267) |
| Duplicate implementations removed | 12 |
