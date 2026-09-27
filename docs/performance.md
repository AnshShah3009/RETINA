# Performance notes

Measured numbers, the machine they were taken on, and what they rule out. All
figures are from `/home/Phoenix/RUST/RETINA` on this workstation (24 cores, RTX
5070 Ti + AMD Radeon 890M, wgpu 28 over Vulkan), on TUM RGB-D 640x480 frames.

Reproduce with the commands in each section.

## ORB and the localization pipeline

Before any optimisation, per frame on a 640x480 TUM image with 1000 keypoints:

| stage | before | after | note |
| --- | ---: | ---: | --- |
| image decode | 3.5 ms | 2.1 ms | library `image` |
| Gaussian blur (ORB pre-pass) | 4.1 ms | 3.8 ms | already internally parallel |
| ORB detection (8 pyramid levels) | 40.4 ms | 26.4 ms | FAST scan parallelised per row |
| ORB descriptor extraction | 45.9 ms | 44.0 ms | already parallel per keypoint |
| **localization query, end to end** | **~1650 ms** | **441 ms** | 3.7x |

The two hot loops were single-threaded. `fast_detect` scanned every pixel of
every pyramid level serially — no rayon anywhere in the file — while
descriptor extraction was equally serial. Both are now parallel and
order-preserving, so output is identical to the serial scan; the raster order
matters because the response sort and keypoint budget depend on it.

After the change, `fast_detect` alone is **0.79 ms/frame** for a single level.

## Why ORB is not going on the GPU

Measured rather than assumed:

```
CPU fast_detect :    0.79 ms/frame
  upload alone   :    0.08 ms/frame   (irreducible: u8 image to the device)
```

A GPU FAST kernel therefore has to finish in under 0.08 ms to beat the
parallel CPU path — about 10x faster than the *original serial* implementation
was. On 640x480 images with ~1,500 candidates per level, plus a non-maximum
suppression and a response sort that are inherently sequential in the current
design, that is not reachable. GPU ORB only becomes worth revisiting at much
larger image sizes or many images per batch, and even then the NMS/sort stages
need a GPU formulation first.

`crates/features/src/orb.rs::detect_ctx` exists but is dead code; it is not
wired into the pipeline and should not be until the above changes.

## Bundle adjustment

| | before | after |
| --- | ---: | ---: |
| one BA solve (9 cameras, 1,002 points) | 35,000 ms | 286 ms |

Two causes, both fixed: the sparse Jacobian was built by central differences
(O(observations x parameters) full state rebuilds), and the sequential solver
then materialised `J^T J` densely and Cholesky-factored it. The Jacobian is now
analytic and the normal equations stay sparse.

## Local bundle adjustment

The local window picks cameras correctly but originally took every landmark
visible to any of them, so the "local" problem grew with the map: 400 points at
the start of a 60-view run, 841 by the end, 79 ms to 174 ms per call.

Capping the landmark set (ranked by how many selected cameras observe each,
capped, deterministic tie-break) was chosen from a sweep on the 150-view
`fr1_desk` run:

| cap | registered | rotation | centre | wall |
| ---: | ---: | ---: | ---: | ---: |
| none | 63.3% | 3.85° | 5.8 cm | 212 s |
| 600 | 63.3% | 6.47° | 6.5 cm | 150 s |
| 300 | 63.3% | 9.35° | 11.5 cm | 116 s |
| **800 (default)** | **63.3%** | **3.85°** | **5.8 cm** | **195 s** |

300 and 600 are faster but cost real accuracy, so 800 keeps the uncapped
rotation error while still cutting the run.

## What is still serial

* The 8 ORB pyramid levels are still processed one after another; parallelising
  them would nest inside rayon's own pool and needs a decision about
  oversubscription, not a `par_iter`.
* Local BA reuses the global solver for a small window — correct, but the next
  real speedup is a genuinely local solve.
* `cv-localization` and `cv-sfm` use no GPU at all. The 38 kernels in
  `crates/hal/src/gpu_kernels` are exercised by the HAL tests (and by the
  two-GPU parity test), not by these pipelines. PnP RANSAC and descriptor
  matching are the candidates worth profiling on the GPU next, not image
  filtering.

## ETH3D

The mapper reads COLMAP text models, so it runs on ETH3D without a conversion
step. Courtyard, 6208x4134 DSLR, 12 consecutive views from one camera:

```bash
cargo run --release -p cv-sfm --example tum_sfm -- \
    --dir datasets/courtyard --frames 12 --stride 1 --window 3 --features 8000
```

| | |
| --- | ---: |
| registered | **83.3%** (10/12) |
| 3D points / mean track | 6,838 / 2.57 |
| reprojection RMSE | 0.50 px |
| camera-centre RMSE | 0.3 cm |
| rotation RMSE (pose-aware) | 0.07° |
| wall time | 41 s |

The feature budget dominates at this image size:

| `--features` | registered |
| ---: | ---: |
| 2,000 | 25.0% |
| 5,000 | 58.3% |
| 8,000 | **83.3%** |
| 10,000 | 83.3% |

TUM-sized defaults leave a 25-megapixel image far too sparsely covered. The
default is deliberately left alone — the right feature count is scene-dependent,
and quietly changing it would hide the effect — but this is the first thing to
raise on any high-resolution sequence.

This scene is also the degenerate regime: 18 of 21 verified pairs are classified
planar (a shallow ring around a largely flat courtyard), so the
homography-versus-essential selection is excluding most of the pair graph from
seeding, as intended.

## ETH3D at scale, and where this stands against visloc-rs

The 12-view courtyard number above is the best case, not the representative one.
Scaling up, on the same machine with the same code:

| scene | views used | registered | centre RMSE | rotation | wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| courtyard, 12 views | 12 | 83.3% | 0.3 cm | 0.07° | 41 s |
| courtyard, all one camera | 23 | 26.1% | 0.1 cm | 0.02° | 242 s |
| electro, 12 views | 12 | 16.7% | ~0 cm | 0.01° | — |

**Registration falls off sharply with view count, and the cause is consistent
across both scenes:** 16 of 23 courtyard views fail with "no 3D point visible in
this view" — the map never grows past the seed neighbourhood. Electro is worse:
only 1 of 30 pairs passes fundamental verification, because it is a largely
planar scene observed from a shallow ring, so the homography-versus-essential
selection correctly refuses most of the pair graph and there is little left to
seed from.

The poses that *are* recovered stay accurate (0.1-0.3 cm, 0.02-0.07 degrees), so
this is a coverage failure, not a precision failure. The mapper finds a small
correct map and cannot extend it.

### Against visloc-rs

Their published result is 99.88% camera registration (9,996/10,008) across ETH3D
low-resolution many-view scenes, and 3.50 cm camera-centre RMSE on Electro.
Ours, on their dataset family, is 26.1% and 16.7% on the DSLR-resolution scenes
we have downloaded, with 0.1-0.3 cm RMSE on what it does register.

**That is not parity and it is not close.** Our accuracy is better; their
coverage is roughly four times better. The honest reading: this mapper can build
a small, geometrically accurate reconstruction from a well-conditioned sequence,
and cannot yet extend one across a long or weakly-constrained capture. The
missing capability is incremental map growth — the next concrete piece of work is
widening the pair graph and seeding from multiple pairs, rather than the single
best pair the mapper currently commits to.

Neither number is a like-for-like comparison: they publish low-resolution
many-view aggregates across many scenes, we have measured two DSLR-resolution
scenes. But on the only axis that matters for a mapper — how much of a sequence
it can reconstruct — we are behind, and the gap is not a tuning difference.

## Why courtyard stops registering, in detail

The failure is not a matching bug, and the ablation that proves it is worth
recording.

The example selects views sorted by **file name**, not spatially, so the 15 views
are scattered across the whole courtyard. Measuring where each unregistered view
sits relative to the reconstructed cluster:

| view | distance to nearest registered camera |
| ---: | ---: |
| 7 | 5.20 m |
| 8 | 5.45 m |
| 9 | 5.79 m |
| 10 | 6.15 m |
| 11 | 6.86 m |
| 12 | 7.28 m |
| 13 | 6.43 m |
| 14 | 12.53 m |

The registered cameras sit around ground-truth centres x = +3..+6 m, y = +1,
z = +3.5; view 7 is at (-2.21, -0.32, -5.99) and view 14 at (+10.39, +4.57,
+7.02). The map covers one region of the courtyard and the remaining views
genuinely have little or no overlap with it.

The ablation settles what the mapper does with those views. For view 7, map
matching yields 90 correspondences over 90 *distinct* landmarks (so no duplicate
crowding), every one of them in front of the camera at a plausible 1.7-10 m
depth — and PnP cannot find any consistent pose at all. For view 8, PnP returns a
pose that is wrong by 176 degrees in rotation and 8 m in translation. The matches
are coincidental rather than correct: a landmark seen from 5 m away in a cluttered
courtyard produces a few plausible-looking descriptor coincidences, and PnP
correctly refuses to build a pose from them.

So the 30.4% is not the mapper failing at its own task. The task is "extend this
map across the scene", and the remaining views are somewhere else. A production
pipeline would either capture overlapping views or detect the gap; our harness
does neither, and the number should be read as a property of the selection, not
of the algorithm. The honest comparison to visloc-rs's 99.88% — which registers
near-every camera in a scene chosen for many-view overlap — still does not hold,
and closing it needs a mapper that can bridge gaps this one declines to.

## Contiguous view selection

Selecting views by file name spans the scene, so the unregistered ones have no
overlap with the map and registration is not being asked a question it can
answer. `--contiguous` keeps the longest run of views whose camera centres are
within `--contiguous-radius` metres:

```bash
cargo run --release -p cv-sfm --example tum_sfm -- \
    --dir datasets/courtyard --stride 1 --contiguous --contiguous-radius 2.0 \
    --window 3 --features 8000 --f-threshold 8.0
```

| selection | registered |
| --- | ---: |
| name-ordered (`--stride 1`) | 7 / 12 — 58.3% |
| contiguous (`--stride 1 --contiguous`) | 7 / 7 — **100.0%** |

The difference is entirely the selection. The mapper reconstructs an overlapping
capture essentially completely, with sub-millimetre camera-centre error; what it
does not do is bridge a multi-metre gap, which is what the remaining views
demanded. Bridging those gaps — loop closure, or a retrieval-guided pair graph
that crosses empty space — is the genuine remaining work, and this measurement
says so precisely instead of leaving it mixed in with a selection artefact.

## Why 58% and not 100%: the capture has a hole in it

Following the audit trail to the bottom. The mapper's view list for courtyard
camera 1, sorted by file name, is:

```
view  0..6  = DSC_0286 .. DSC_0292   (consecutive)
view  7..11 = DSC_0302 .. DSC_0306   (consecutive)
```

There is a **ten-frame gap in the capture** between views 6 and 7 — the rig
skipped those frames, so every consecutive pair straddling that break is a pair
of images ten frames apart.

The matcher output for the mapper's exact view list, at the mapper's own
settings (ratio 0.75 plus cross-check):

| pair | frames | matches |
| --- | --- | ---: |
| (0,1) … (5,6) | consecutive | 890, 954, 1118, 1029, 967, 1078 |
| **(6,7)** | **DSC_0292 → DSC_0302** | **97** |
| (7,8) … (10,11) | consecutive | 506, 698, 626, 200 |

That single row explains everything else in this investigation. The
reconstruction covers DSC_0286–0292 and cannot cross a ten-frame hole; views 7+
are on the other side of it. With the ground-truth pose of view 7, **0 of 5,139
map landmarks project inside its image** (they land above the top edge, median
y = −5973 for a height of 4135), while the control — a registered view — gives
**76.7%** of landmarks within 4 px of a real keypoint. The map geometry is
sound.

So the mapper is behaving correctly and the 58% was never a matching,
descriptor, ratio, pair-window or model-selection problem; all of those were
measured and cleared on the way. It is a capture with a hole in it, handed to a
mapper that does not bridge holes. Bridging needs either a pair window wide
enough to propose across the gap — the pairs exist, they fail verification
because the images are ten frames apart — or loop closure. visloc-rs's many-view
ETH3D pipeline has both; we have neither, and that is the real distance to
their 99.88%.

## Long contiguous runs: the number that matters

The courtyard result is bounded by a hole in the capture, so it understates the
mapper. On a trajectory that actually moves through the scene - TUM RGB-D
fr1_xyz, contiguous selection, 640x480 - registration is essentially complete:

| views | registered | points | centre RMSE | rotation |
| ---: | ---: | ---: | ---: | ---: |
| 20 | **100.0%** (20/20) | 3,944 | 1.65 cm | 0.70 deg |
| 30 | **100.0%** (30/30) | - | 2.63 cm | - |
| 45 | **97.8%** (44/45) | 6,438 | 4.61 cm | 1.62 deg |

That is the same order as visloc-rs's 99.88%, on a different dataset and with
far fewer images (tens of views here, thousands in their ETH3D runs). The honest
claim is that incremental registration is close to saturated on a well-captured
trajectory, and that the courtyard figure was a property of its capture rather
than of the algorithm.

What is still missing relative to their pipeline, in order:

1. **Loop closure.** On courtyard it would not have helped - the far side never
   returns (nearest earlier view is 2.07 m, 9.29 m across the hole), so there is
   no loop to close. A sequence that genuinely revisits is the case to test next.
2. **Gap crossing.** A 9.3 m baseline on a cluttered courtyard yields 97
   descriptor matches where adjacent frames yield ~1000. Proposing and verifying
   such pairs is a different problem from ordinary pair selection, and nothing
   here attempts it.
3. **Many-view evaluation.** Every number above is tens of views. Their claims are
   on 1,000-10,000 image scenes. Until we run one of those, "par or better" is
   an untested claim rather than a measured one.

## Loop closure

A sequential mapper only ever matches a view to its temporal neighbours, so a
trajectory that returns to somewhere it has already been leaves the second visit
as a separate island. `close_loops` reconnects them: after a view registers, every
earlier view beyond `--loop-min-gap` is matched and verified with the same bar a
temporal pair must clear, and a verified pair is fused into the track graph.

Fusing is gated on consistency. Once both endpoints have poses, each landmark the
pair shares is projected into the other view and compared with the observation
the loop claims; a loop is accepted when the median residual is inside
`--loop-max-px` and at least half the observations agree. A loop that only agrees
after discarding most of its matches is the wrong epipolar hypothesis wearing a
confident inlier count.

### It does not pay for itself, and defaults off

Measured on TUM fr1_xyz, 45 views, `--window 3 --features 1200`:

| | registered | closures | centre RMSE | pose-aware rotation |
| --- | --- | ---: | ---: | ---: |
| loop closure off | **44/45 (97.8%)** | 0 | 0.0461 m | **1.62°** |
| gate at 1.0 px | 44/45 (97.8%) | 1 | 0.0890 m | — |
| gate at 2.5 px | 39/45 (86.7%) | 5 | 0.0489 m | — |
| gate at 4.0 px | 41/45 (91.1%) | 18 | 0.0673 m | 2.39° |
| gate at 12.0 px | 38/45 (84.4%) | 22 | **0.0286 m** | — |
| no gate | 42/45 (93.3%) | 25 | 0.0339 m | 2.30° |

Centre error falls monotonically as more loops fuse — 4.6 cm down to 2.9 cm — and
the map grows from 6,438 to 11,060 points, so the *geometry* the loops add is
genuinely good. But registration falls with it, and no configuration recovered the
44/45 baseline.

The cause is ordering, not the loops. Fusion extends tracks and moves landmarks,
which invalidates poses that PnP had already estimated; nothing re-registers them,
so the loss compounds along the sequence. Fusing is correct; fusing *and then
re-optimising globally* is what is missing. Until that exists, enabling it trades
real accuracy for a registration count that is only a proxy.

On TUM fr1_desk, which revisits more aggressively, 0 loops fire at all and both
settings give 9/40 — that sequence stalls earlier for an unrelated reason.

`--loop-closure` enables it; `--loop-max-px` and `--loop-min-gap` tune it.

## Pair window, and why the seed decides the run

The mapper pairs each view with the next `--window` views in time. A window of 3
is too tight for a fast-moving handheld camera, and it was the single biggest
quality lever found:

| window | fr1_xyz (45 views) | fr1_desk (40 views) |
| --- | --- | --- |
| 3 | 44/45 — 97.8% | 9/40 — 22.5% |
| 5 | **45/45 — 100%** | **25/40 — 62.5%** |
| 8 | — | 25/40 — 62.5% |

The default is now 5.

### The feature-count sweep is not monotone, and that is the real finding

At window 5 on fr1_desk, registration by feature count:

| features | 800 | 1200 | 1500 | 2000 | 3000 |
| --- | --- | --- | --- | --- | --- |
| registered | 12/40 | 25/40 | 9/40 | 17/40 | 28/40 |

A tuning curve that goes 12 -> 25 -> 9 -> 17 -> 28 is not a tuning problem. The
cause is the seed: changing the feature count changes which pair is ranked best,
and the chosen seeds differ completely — `(14, 16)` at 1200 features, `(24, 26)`
at 1500 and 3000. One seed leads somewhere and another strands the run, and the
result is decided before any view registers.

The seed metric cannot see this. `seed_from_pair` counts the points that
triangulate correctly *from that pair alone* — positive depth in both cameras,
parallax above threshold, reprojection under `max_reproj_px`. It is a purely
local measure, so a pair that is locally excellent and sits at a dead end of the
sequence outranks a pair that would have carried the whole trajectory. Raising
`--seed-hypotheses` from 8 to 32 changes nothing (25/40 either way), because the
surplus hypotheses are ranked below the same local winner, not alongside a better
one.

What is needed is a seed metric with lookahead — a pair that also has verified
neighbours on both sides is the one that can start a chain. Until then, fr1_desk's
62.5% is a floor imposed by initialization, not by PnP, matching the 15
"PnP RANSAC failed" and 14 "too few inliers" in the breakdown.
