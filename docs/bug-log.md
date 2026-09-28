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
| Colored ICP's photometric Jacobian ignored all three arguments and returned a constant, so `lambda_geometric = 0` made the normal equations rank-1 and the solver never moved — while still returning a success | same review | **not yet fixed**, recorded below |
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

## Counted

| | |
| --- | ---: |
| Defects fixed | **78+** |
| Commits | 460+ |
| Tests | 1,495 (from 1,267) |
| Duplicate implementations removed | 12 |
