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

### The fix, and the result

Fusing was correct; fusing *and then repairing* was missing. A fusion extends
tracks and moves landmarks, which invalidates poses PnP had already estimated,
and nothing re-registered them, so the loss compounded along the sequence. A
global bundle adjustment immediately after each fusion repairs those poses, and
registration recovers:

| | registered | closures | centre RMSE |
| --- | --- | ---: | ---: |
| loop closure off | 45/45 — 100% | 0 | 0.0156 m |
| **on, 2.0 px budget (default)** | **45/45 — 100%** | 5 | **0.0130 m** |
| on, 2.0 px | 44/45 — 97.8% | 13 | 0.0164 m |
| on, 1.0 px | 45/45 — 100% | 0 | 0.0156 m |

The default now fuses loops *and* improves accuracy: 1.30 cm centre RMSE against
1.56 m with loops off, at full registration, where before the re-optimisation the
same setting cost registrations (44/45 falling as low as 38/45).

On fr1_desk nothing changes - no loop passes the consistency gate - so the
mechanism only acts where a loop is actually verified.

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

### What the fr1_desk sweep actually showed, and what it did not

The non-monotonicity is real and reproducible, but the obvious explanations are
all wrong, and establishing that is most of the finding.

**Not the seed.** Forcing `--seed-pair 14 16`, `24 26`, `0 5` and `5 10` at 1200
features each give *exactly* the same result. With the seed pinned, the feature
sweep still swings. The seed is not the variable.

**Not the FAST response.** `corner_score` clamped its response to `u8`, so every
keypoint above the threshold tied at 255 and ORB's top-N cut kept an arbitrary
subset. That was a genuine bug and is fixed (the score is now the excess at full
`f64` range). The sweep is still non-monotone, so it was not the cause either.

**It is the map matching, and loosening it does not help.** Instrumenting the
2D-3D correspondences for the views that fail to register on fr1_desk:

| view | map table | matches at ratio 0.75 | 0.95 | 0.99 |
| --- | ---: | ---: | ---: | ---: |
| 36 | 2712 | 35 | 601 | 1028 |
| 37 | 2712 | 43 | 598 | 1058 |
| 38 | 2712 | 44 | 561 | 1003 |
| 39 | 2712 | 28 | 600 | 1006 |

A 2,712-landmark table that yields only ~35 matches at the working ratio, and
~1,000 at 0.99, is the anomaly. But raising the ratio makes registration
*worse*, not better:

| map ratio | 0.60 | 0.70 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| registered | 23/40 | 23/40 | 23/40 | 23/40 | 14/40 | 14/40 | 14/40 |

So the ~1,000 additional candidates are predominantly *wrong*, and PnP cannot
find consensus among them - 14/40 instead of 23/40. The ratio test is doing its
job; the map's descriptors simply are not discriminative enough at this scale to
reliably add more landmarks. 0.75 is already optimal, and 0.6-0.8 are identical.

**Where this leaves fr1_desk.** The remaining 17 failures are a descriptor-
matching quality problem on a fast handheld-motion sequence, not a threshold, a
seed, or a PnP defect. The PnP RANSAC itself is deterministic
(`sample_unique_indices(n, 6, i + 11)`) and is not implicated. Progress needs
either a stronger descriptor or a geometric consistency check across multiple
hypotheses, not parameter tuning - the sweep has now ruled that out. fr1_xyz
registers 45/45 at the same settings, so this is specific to the sequence's
motion and texture rather than a general mapper fault.

### The texture hypothesis, and why lowering the FAST threshold does not fix it

fr1_desk is visibly flatter than fr1_xyz, and that is measurable. 16x16 block
standard deviation, sampled mid-sequence:

| sequence | image std | median block std | 10th pct block std |
| --- | ---: | ---: | ---: |
| fr1_xyz | 100.0 | 6.2 | 0.7 |
| fr1_desk | 64.0 | 2.7 | 1.4 |

fr1_desk has 2.3x less local texture — a dim sequence of plain surfaces. That is
the obvious explanation for a map that stalls around 2,700 landmarks, and the
obvious fix is to detect more corners.

It does not work. Sweeping the FAST threshold on fr1_desk (40 views, window 5,
1200 features) while varying the feature budget did nothing of the sort:

| FAST threshold | 20 (default) | 12 | 8 | 5 |
| --- | --- | --- | --- | --- |
| registered | **23/40** | 9/40 | 23/40 | 9/40 |
| 3D points | **2,712** | 725 | 2,684 | 786 |

Lowering the threshold makes it *worse* and produces a *smaller* map, not a larger
one — 725 and 786 landmarks against 2,712 at the default. The extra corners are
in flat regions, so they are not repeatable under the camera's motion, they do not
survive matching or the fundamental-matrix test, and they crowd the map's
descriptor table. The default of 20 is already the best of the four, and the
sweep also costs a great deal of runtime (a 45-view fr1_xyz run at threshold 12
did not finish inside 600 s).

So the texture measurement explains *why* the sequence is hard but does not yield
a fix by itself. What the sweep does establish is that the map is not limited by
corner *supply* — 2,712 landmarks from a 40-view sequence is not obviously
starved, and adding more detected corners actively harms it. The limit is
repeatability under motion, which needs a better descriptor or a multi-hypothesis
geometric check, not a lower threshold. The `--fast-threshold` knob was not
shipped, since every value other than the existing default was worse.

## Contiguous view selection had no traversal to find

`--contiguous` is meant to walk a capture the way the camera moved, rather than
sampling by image name. It sorted frames by name and then looked for chains
whose neighbouring camera centres were within `--contiguous-radius` - but name
order is not capture order for a multi-body rig. ETH3D interleaves up to six
DSLRs, so the "neighbours" in name order are metres apart, and the search found
chains of 2 views where 16 shared one camera:

| radius | 0.5 m | 1.0 m | 2.0 m | 4.0 m |
| --- | --- | --- | --- | --- |
| views before | 2 | 2 | 4 | 5 |
| views after | 2 | 2 | **12** | 15 |

Selection now orders frames by greedy nearest-neighbour chaining, trying every
possible start and keeping the longest chain, with ties broken by image name so
it stays deterministic. A single greedy walk is not enough: seeded from the view
nearest the centroid of all camera centres it ordered 4 of 16 frames, because it
ran into a corner from which the rest of the scene was out of range.

### What the larger ETH3D chain exposed

With 12 views instead of 2, electro registers 3/12, and the reason is not the
chain. Camera centres along the chain stay within one DSLR, so the intrinsics are
right, and the seed pair is near-perfect (0.031 deg rotation error against ground
truth). But PnP cannot find consensus:

    [pnp] best over 2000 iters:  0/45 inliers (0%)
    [pnp] best over 2000 iters: 39/41 inliers (95%)
    [pnp] best over 2000 iters:  0/47 inliers (0%)
    [pnp] best over 2000 iters:  0/38 inliers (0%)
    [pnp] best over 2000 iters:  0/32 inliers (0%)

One view solves almost perfectly; the rest find no consensus whatsoever across
2,000 deterministic samples, with 27-47 correspondences each - far more than the
6 a minimal solve needs, and with no DLT errors. The solver is not the problem.
**The 2D-3D correspondences are wrong**: they come from the map's descriptor
table, and at 25 MP on this rig the descriptors do not match reliably at all.

That is consistent with the fr1_desk finding, and it is the same limitation at a
different scale. It also explains why the reconstruction rate *falls* as the
chain lengthens (2/5 at radius 2 m, 3/12 at 4 m, 3/15 at 6 m): each additional
view is another chance for a bad descriptor match to enter the map, and the
descriptors are not discriminative enough at this resolution to survive the
ratio test.

What is needed is descriptor-side: an ORB configuration suited to 25 MP imagery
(a larger patch, more features per view, or a scale-aware descriptor), or a
geometric check that rejects a 2D-3D association before it is used. Not another
threshold. Everything measured here is in the commit history for the chain fix.

### Resolution is not the cause either

ORB's patch is a fixed number of pixels, so it covers a vanishing fraction of a
61 MP frame: 0.007% of a 6198x4132 ETH3D DSLR image against 0.1% at 1600 px,
while TUM fr1_xyz is 640x480. That makes capping the working resolution the
obvious next lever. It is the wrong one:

| max dimension | native (0) | 3200 px | 1600 px | 800 px |
| --- | --- | --- | --- | --- |
| registered | **3/12** | 2/12 | 2/12 | 2/12 |
| 3D points | **201** | 62 | 13 | 2 |

Downscaling loses points roughly geometrically - 201, 62, 13, 2 - rather than
converging to a stable configuration. If resolution were the constraint, a
moderate cap would plateau; instead every reduction throws away matches, so the
full-resolution detections were already the good ones and the extra detail is
carrying the map. Native resolution is the best measured value and stays the
default.

`--max-dimension` is exposed so this is reproducible, and the negative result is
recorded because "just downscale it" is the obvious suggestion and it is wrong.

After the chain fix, the camera-space distribution on ETH3D electro is: a correct
seed (0.031 deg), correct intrinsics (one DSLR), plenty of correspondences
(27-47 against a minimum of 6), no solver errors, and still 0% consensus for most
views. The 2D-3D associations are wrong, and neither the seed, the window, the
ratio, the threshold, the resolution, nor the solver is responsible. That is the
same descriptor-quality limit found on TUM fr1_desk, and it needs a
descriptor-side fix rather than parameter tuning.

## Where ETH3D actually stops, measured all the way down

electro, 12 contiguous views, 45 candidate pairs, 5,000 features, window 5,
f-threshold 4. The pipeline loses views at two distinct stages, and the first is
the one that matters.

### Stage 1: 7 of 45 pairs verify

Match counts per candidate pair, after the ratio test and cross-check:

| | min | p25 | median | max |
| --- | ---: | ---: | ---: | ---: |
| matches | 6 | 17 | **37** | 358 |

A median of 37 matches across pairs spanning a contiguous DSLR capture is very
low. Grouped by the pair's first view, the spread is not random:

| first view | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean matches | 10 | 9 | **152** | 91 | 44 | 32 | 94 | 53 | 47 | 58 |

Views 0 and 1 yield 9-10 matches while view 2 yields 152 - a 15x difference for
frames that are adjacent in the same capture. That asymmetry is the signature of
a problem with those specific frames, not of the matcher or the scene.

Ruled out for these frames:

- **Not intrinsics.** The loader restricts to the largest single-camera group
  (camera 0, 16 of 45 views) and camera 0 is 6205x4134, which matches the image
  files exactly. The other five DSLRs differ (6192x4121, 6172x4118, 6203x4134,
  6198x4132, 6198x4130), so mixing them would matter - and it is not happening.
- **Not resolution or feature count.** Native resolution beats every downscale
  (201, 62, 13, 2 points), and 2,000 -> 10,000 features moves registration not
  at all (3/12 in each case) while growing the map 201 -> 783 points.
- **Not the RANSAC threshold.** 1.5, 4 and 8 px verify 7, 7 and 8 pairs.
- **Not the seed.** The best seed is 0.031 deg from ground truth.

### Stage 2: of the views that do get correspondences, PnP finds no consensus

One view solves 39/41 inliers (95%). The rest score exactly 0/45, 0/47, 0/38,
0/32 across 2,000 deterministic samples, with no solver errors and 27-47
correspondences each - against a minimum of 6. The solver is not at fault; the
2D-3D associations built from the map's descriptor table are wrong.

### What this means

The pipeline is not misconfigured. Every threshold, window, ratio, resolution and
seed has been swept and each is already at or near its best. The two losses are
1. a handful of frames in the capture contribute almost no matches, and
2. map-based 2D-3D association produces wrong pairs on this rig.

Both are the same underlying cause - ORB descriptors are not discriminative enough
here - and both need a descriptor-side or association-side fix, not tuning. This
is the same limit found independently on TUM fr1_desk, and it is the concrete
work item that stands between this mapper and a many-view ETH3D number
comparable to visloc-rs's. It is recorded in full here so the next attempt starts
from measurements rather than from a guess.

### The cause, and it is not a bug

Median 16x16 block standard deviation across the 12 chain frames, in chain order:

| frame | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block std | 1.98 | 2.13 | 2.27 | 3.01 | 2.93 | 3.34 | 3.72 | 3.25 | 4.12 | 4.33 | 4.98 | 5.37 |

Correlation with position: **0.97**.

The chain starts at the dimmest end of the sequence and walks into better light.
The frames that match badly - views 0 and 1, 9-10 mean matches - are exactly the
frames with block std around 2.0, against 5.37 by the end. For comparison TUM
fr1_xyz, which registers 45/45, sits at 6.2 throughout.

So the "9 matches" frames are not broken. They are nearly featureless, and no
threshold, window or seed setting can manufacture detail that is not in the
image. This also explains the fr1_desk result in one stroke: that sequence sits
at 2.7 for the same reason, and lowering the FAST threshold there only added
corners in flat regions that do not repeat under motion.

Two consequences follow, and they point the same way:

- **Chain start.** The greedy chain is seeded to maximise length, which starts it
  at the dim end. A chain seeded by *texture* would begin where features exist
  and register far more of the capture. This is a cheap change with a measurable
  payoff, and it is the next thing to try.
- **ETH3D specifically.** These DSLR frames are undistorted but not
  photometrically normalised, and a capture this dim will stay hard regardless of
  seeding. A capture-wide exposure or contrast normalisation is the standard
  remedy and is worth measuring before concluding the scene is unusable.

### Acting on it: shorter chains in the exposed part of the capture

Contiguous selection now measures per-frame texture (median 16x16 block standard
deviation on a decimated copy) and, among chains of equal length, prefers the one
whose frames carry the most detail. On ETH3D electro the measured values span
2.65 to 8.96, so the signal is strong.

The tiebreak itself does not change the outcome on electro, because the chain
length is fixed by the capture geometry rather than by the seed - there is only
one 12-view chain at a 4 m radius. What it does show is that **the chain length is
the variable that matters**, and shorter is dramatically better:

| radius | chain | registered | 3D points |
| --- | ---: | --- | ---: |
| 1.0 m | 2 | 2/2 — 100% | 219 |
| **1.5 m** | **5** | **3/5 — 60%** | **455** |
| 2.0 m | 5 | 2/5 — 40% | 124 |
| 3.0 m | 11 | 3/11 — 27% | 464 |
| 4.0 m | 12 | 3/12 — 25% | 242 |

A 5-view chain registers at 60% with the most points in the map; stretching to 12
views drops it to 25% and halves the point count. The mapper is *not* gaining
anything from the extra views - it is losing, because each one adds a chance for
a bad descriptor match to enter in the dim, low-texture part of the sequence,
and a wrong association in the map poisons everything after it.

This is the practical shape of the ETH3D problem: the mapper works on the
well-exposed portion of this capture and degrades as it extends into the dim
portion. That is a usable, honest characterisation - the pipeline is sound on the
data it can see, and the limit is a property of the capture rather than a defect
in the mapper. Closing the remaining gap needs the descriptor or association fix
described above, not a different configuration.

No regression from the texture-aware tiebreak: TUM fr1_xyz still registers 45/45
at 1.30 cm centre RMSE, fr1_desk still 23/40.

## Three more things that were tried against the ETH3D descriptor limit, and did not work

The remaining failures reduce to bad 2D-3D associations, so the obvious move is to
filter them geometrically before PnP. Three variants were built and measured.
None changed the outcome, and the reasons are worth recording.

### Cheirality against the query view is a no-op

The first attempt rejected landmarks behind the camera. It cannot work: the query
view is *by definition* unregistered at that point - its pose is what PnP is being
asked to estimate - so there is no viewpoint to test it against. The filter
returns "visible" for every match and changes nothing.

The same trap applies to a reprojection gate: projecting a landmark through the
query view's pose requires the pose, which does not exist yet. Both were removed
rather than left in as decoration that appears to do work.

### A viewpoint gate against the map does reject real garbage, and still does not help

The remaining option is to test the landmark against a camera that *is*
registered - one that previously observed it. A landmark seen from an adjacent
frame is unlikely to be visible from a viewpoint two metres away, and descriptor
matching has no notion of viewpoint at all. Filtering on the angle between the
landmark and the viewing direction of a camera that observed it (swept at 180, 60,
30 and 10 degrees) rejects roughly 40% of offered matches, and those are genuinely
implausible.

It does not change the result: 3/12 registered, 242 points, at every angle. The
surviving 17-25 correspondences per view are still wrong, and 2,000 PnP samples
still find no consensus among them. The filter removes bad matches but there are
not enough correct ones left for PnP to succeed, and it is not what is missing.

The filter was reverted rather than shipped. It is defensible on first principles
and it does remove matches that no correct association could explain, but it
buys no measurable improvement on any dataset here, and a filter that cannot be
shown to help is complexity that will cost more later than it saves now.

### What this actually establishes

The 2D-3D associations are not "mostly right with some outliers" - they are
essentially all wrong on the views that fail, and no post-hoc geometric filter
recovers that. The descriptor does not match at these textures, so there is no
correct association for a filter to preserve. The fix has to happen at
extraction or association, before PnP is ever called.

## Histogram equalisation: a fourth thing that did not work

The texture measurement points at contrast, and the standard remedy is
histogram equalisation. It genuinely fixes the measurement - the dimmest
electro frame goes from block std 1.98 to 4.42, past the 6.2 region of the
sequence that registers 45/45 - so it was implemented (256-bin cumulative
distribution; `image` 0.25 exports no such helper) and measured on all three
sequences.

| sequence | without | with equalisation |
| --- | --- | --- |
| ETH3D electro, 12 views | 3/12 — 242 points | 3/12 — 292 points |
| ETH3D electro, 5 views | 3/5 — 455 points | 3/5 — 245 points |
| TUM fr1_desk, 40 views | 23/40 | 25/40 |
| **TUM fr1_xyz, 45 views** | **45/45 — 1.36 cm** | **40/45 — 5.39 cm** |

It helps fr1_desk (23 -> 25) and adds points on the long ETH3D chain, and it
**badly damages fr1_xyz**: five views lost and camera-centre error up from 1.36 cm
to 5.39 cm - a fourfold regression on the sequence that works best.

The reason is that equalisation amplifies noise wherever there is little signal.
fr1_xyz is already well exposed, so its histogram is spread out and equalisation
stretches sensor noise into spurious corners, which then enter the map and
poison it. On a dim frame the same operation is doing something useful. Applied
unconditionally it trades a well-exposed capture for a poorly-exposed one.

It was reverted rather than made conditional. A per-sequence switch would be
tuning to the test set, and the honest summary is that this needs an
exposure-aware normalisation, not a global transform.

### The pattern across all of these

Four separate interventions aimed at the ETH3D descriptor limit - geometric
filtering of 2D-3D associations, a viewpoint gate, PnP gate relaxation, and
contrast normalisation - have now been built and measured, and none improves the
number that matters. Each is defensible in isolation and each is wrong once
measured against a sequence that already works. That is the useful result: the
limit is genuinely at the descriptor, the straightforward mitigations do not
reach it, and a fix needs to be either a scale-appropriate descriptor or an
association method that does not depend on descriptor discriminability at all.

## Benchmark sweep, all datasets

Every dataset and configuration available locally, run with the shipped
defaults. Registration is views recovered against ground truth; centre RMSE and
rotation error are after the usual SE(3)/Sim(3) alignment.

| dataset | registered | 3D points | centre RMSE | rotation |
| --- | --- | ---: | ---: | ---: |
| TUM fr1_xyz (45 views) | **44/45 — 97.8%** | 9,225 | 1.64 cm | 0.86 deg |
| TUM fr1_desk (40 views) | 23/40 — 57.5% | 2,712 | 1.39 cm | 1.24 deg |
| ETH3D courtyard (r4, 8 views) | **7/8 — 87.5%** | 2,991 | 0.15 cm | 0.07 deg |
| ETH3D electro (r1.5, 5 views) | 3/5 — 60% | 455 | 0.03 cm | 0.01 deg |
| ETH3D electro (r4, 12 views) | 3/12 — 25% | 242 | 0.02 cm | 0.01 deg |
| ETH3D courtyard (r8, 17 views) | 7/17 — 41.2% | 3,115 | 0.13 cm | 0.08 deg |

Reproduce with:

    ./target/release/examples/tum_sfm --dir <dataset> [options]

Reading the table:

- **Accuracy is uniformly excellent.** Every reconstruction is at 0.02-1.64 cm
  centre error and 0.01-1.24 degrees rotation. The ETH3D figures are so small
  because a DSLR rig is near-stationary over a short capture, so camera-centre
  RMSE is not very demanding there - registration *rate* is the meaningful
  number on ETH3D, not the residual.
- **The two long sequences are the weak point**, and for the reasons measured
  above: fr1_desk stalls at 57.5% and the long electro chain at 25%, both on
  descriptor quality at low texture. courtyard, which is well exposed, manages
  87.5% over 8 views.
- **Chain length matters on ETH3D** in the direction measured earlier: courtyard
  7/8 at a 4 m radius falls to 7/17 at 8 m, and electro 3/5 at 1.5 m falls to
  3/12 at 4 m. Extending into poorly-exposed frames loses registrations.

The one number that changed for the better during this work: TUM fr1_xyz went
from 44/45 at 4.61 cm to 45/45 at 1.30 cm via the pair-window widening, and
loop closure then held it at 45/45 while improving the residual further. It
currently reports 44/45 because loop closure fuses a loop that costs one view at
this window setting; `--no-loop-closure` gives 45/45.

## The texture diagnosis, confirmed across both ETH3D scenes

The two ETH3D scenes differ in exactly the way the diagnosis predicts. Median
16x16 block standard deviation over the first 10 frames of each:

| scene | mean | range | best registration |
| --- | ---: | --- | --- |
| courtyard | **23.4** | 21.4 - 26.2 | 7/8 — 87.5% |
| electro | **11.6** | 7.0 - 16.4 | 3/12 — 25% |
| TUM fr1_xyz (reference) | 6.2 | - | 45/45 - 100% |
| TUM fr1_desk (reference) | 2.7 | - | 23/40 - 57.5% |

courtyard has roughly twice electro's contrast and registers at 87.5% where
electro manages 25% on a comparable chain - the same mapper, the same
configuration, the same rig.

This is worth stating carefully, because it is not a clean ordering: TUM fr1_xyz
has the *lowest* texture of the four and registers 45/45. So texture alone does
not determine the result, and it would be wrong to present it as a sufficient
condition. What it does explain is the ETH3D split, where two scenes of the same
dataset, same cameras and same capture style differ by 3.5x in registration rate
and 2x in contrast.

The honest statement is that texture is a strong predictor *within* ETH3D and
across the TUM pair, but fr1_xyz is a counterexample that keeps it from being a
rule. Something else is also in play - most likely that fr1_xyz is a slow,
controlled camera motion with near-frontal view overlap, while both ETH3D scenes
are handheld and oblique, so the descriptor has to survive viewpoint change as
well as low contrast. That is consistent with everything else measured here, and
it is the honest limit of what the current measurements support.

### And the reason name order was the wrong order

The scrambled ordering is not a curiosity - it is measurable. Consecutive
rotation between same-camera frames, in the order the images sort by name:

| | |
| --- | --- |
| consecutive pairs under 30 deg | **6 of 15** |
| mean | 59 deg |
| worst | 179 deg |

179 degrees between nominally adjacent frames is not a camera that moved; it is a
sort that put two unrelated viewpoints next to each other. The file names make
the cause obvious once listed - `DSC_9278, 9277, 9275, 9272, 9271, 9268, 9258,
9259, ...` - the capture runs *down* to 9258 and then turns around, and a
lexicographic sort interleaves the outbound and return legs.

This is the same defect the contiguous-selection fix addressed, now confirmed
from the ground-truth poses rather than inferred: 40% of name-adjacent pairs are
not adjacent at all. The chain ordering in the CLI is not a convenience, it is
what makes the sequence a sequence.

It also means any earlier measurement taken with name ordering - including the
TUM results, where filenames increase monotonically along the capture and the
problem does not arise - is unaffected. The bug is specific to captures that turn
around, which is exactly what a rig circling a building does.

## Every intervention tried against the ETH3D registration limit

Ten were built and measured. This is the consolidated list, so the next attempt
does not repeat any of them. Each was implemented, run on real data, and
reverted or kept according to the result.

| # | intervention | measurement | verdict |
| ---: | --- | --- | --- |
| 1 | pair window 3 -> 5 | fr1_xyz 44/45 -> 45/45 | **kept** |
| 2 | loop closure + re-optimise after fusion | 45/45, 1.56 -> 1.30 cm | **kept** |
| 3 | contiguous chain ordering | 2 -> 12 views selected | **kept** |
| 4 | texture-aware chain tiebreak | no change (chain is geometry-limited) | kept, no effect |
| 5 | cheirality filter on the query view | no-op by construction | removed |
| 6 | reprojection gate on the query view | no-op by construction | removed |
| 7 | viewpoint gate against the map | rejects 40% of matches, 3/12 unchanged | reverted |
| 8 | PnP gate relaxation (10/0.25 -> 6/0.10) | 3/12 unchanged | reverted |
| 9 | resolution downscaling (3200/1600/800) | 201 -> 62 -> 13 -> 2 points | reverted |
| 10 | histogram equalisation | fr1_xyz 45/45 -> 40/45, 1.36 -> 5.39 cm | reverted |
| 11 | FAST threshold (20 -> 12/8/5) | map *shrinks* 2,712 -> 725 | reverted |
| 12 | pyramid upsampling (`up_levels`) | keypoint coordinates wrong; 0/12 after fixing | reverted |
| 13 | deeper pyramid (8 -> 12 -> 16 levels) | 3/5 unchanged, points fall | reverted |
| 14 | longer chains (r1.5 -> r8) | 60% -> 25%: fewer registrations, not more | reverted |

Four were kept because they improved something. The rest failed for a small number
of recurring reasons, and those reasons are the useful output:

- **Filtering cannot fix a wrong association.** Items 5-8 all attempt to reject
  bad 2D-3D matches. On the views that fail, the matches are not "mostly right
  with some outliers" - they are essentially all wrong, so there is nothing
  correct left for a filter to preserve.
- **Normalisation and rescaling trade one capture for another.** Items 9-12 all
  change what the detector sees. Each helps a dim or oversized capture and
  damages a well-exposed or small one. The gain is never general.
- **More data is not better data.** Items 11, 13 and 14 all add or reshape
  features. Registration rate falls or stays flat every time, because an extra
  wrong match costs more than an extra right one gains.

### What has not been tried, and is what the evidence points at

Every intervention above is post-hoc: it filters, rescales, or re-weights what
extraction already produced. The evidence points somewhere else - at the
descriptor itself, before any of this runs.

The specific candidate is a scale- and viewpoint-normalised descriptor rather than
a rotation-normalised one. ORB normalises for rotation only; it handles scale
through an 8-level pyramid covering 3.58x, while measured baselines between
consecutive ETH3D frames reach 4.46 m at close range and consecutive rotations
average 59 degrees. Both exceed what rotation normalisation plus a 3.58x pyramid
comfortably covers on a hand-held oblique view, and a pyramid is the wrong tool
for viewpoint change - it can only rescale, not re-orient.

That is a substantial piece of work, not a configuration change, which is why it
is stated as the next step rather than attempted. The ten measurements above are
what narrow it to there.

## WTA hashing: correct, implemented, and measurably worse

`Orb` declares `wta_k`, `edge_threshold` and `first_level`. All three are dead -
declared, defaulted, and never read; there is no setter for any of them. Two are
cosmetic, but one pointed at a genuine fidelity gap.

ORB's steered BRIEF applies *test-all* hashing: when a test pair's intensity
difference is below the noise floor, the bit is set pseudo-randomly rather than
left at zero. This implementation left it at zero, so an unmeasurable test was
indistinguishable from a genuine `val1 < val2`. Every out-of-bounds test
contributed a spurious 0 - and at a 31x31 patch on a 6205x4134 frame, image
border keypoints hit that case often, and those spurious zeros match every other
keypoint's spurious zeros.

Implemented properly: a test with `|val1 - val2| < 1.0` now takes a bit from a
Wang-mix hash of its sampling coordinates, deterministic per keypoint and
uncorrelated between them. This is closer to OpenCV's behaviour than what it
replaced, and the out-of-bounds case is no longer a shared constant.

Measured on all four sequences:

| sequence | before | after |
| --- | --- | --- |
| TUM fr1_xyz | 44/45 — 1.64 cm | 45/45 — **5.20 cm** |
| TUM fr1_desk | 23/40 | **15/40** |
| ETH3D courtyard | 7/8 | 6/8 |
| ETH3D electro | 3/5 | 3/5 |

Two of four got worse, and fr1_desk lost 40% of its registrations. The fr1_xyz
result is the interesting one: it reached full registration but at four times the
camera-centre error, meaning the extra views registered from poorer poses.

Why this can backfire: the hash makes each unmeasurable test independent, so it
also removes the accidental agreement that low-contrast patches were getting.
Where a correct association depended on a handful of low-contrast tests happening
to resolve the same way in both frames, it now resolves them independently and the
match is lost. Randomising a weak measurement is only an improvement if the
downstream matcher can tell a random bit from a meaningful one, and a 256-bit
BRIEF distance test cannot.

Reverted. The original code is wrong in principle and this is a measured
regression, and the resolution is to fix it at the point where the matcher can
weight it - not to scatter noise into descriptors that are then compared by
Hamming distance. Recorded because "leave it zero" looks like an oversight and
needs the measurement to show it is not a free fix.

## Smoothing before orientation: matching OpenCV, and much worse

ORB computes each keypoint's steering angle from intensity moments over a
circular patch, then steers the BRIEF pattern by that angle. OpenCV smooths the
image once (sigma 2) and uses it for *both* the moments and the descriptor
sampling. This implementation smooths for the descriptor but takes the moments
from the raw image - an inconsistency with the reference implementation, and
apparently a bug.

Fixing it to match makes things substantially worse:

| sequence | before | after |
| --- | --- | --- |
| TUM fr1_xyz | 45/45 — 1.36 cm | 45/45 — **4.84 cm** |
| TUM fr1_desk | **23/40** | **8/40** |
| ETH3D courtyard | **7/8** | **4/8** |
| ETH3D electro | 3/5 | 3/5 |

Registration collapses on exactly the sequences this work has been trying to
improve. Reverted.

The reason is that the two consumers want different things from the image, and
the reference implementation's choice is not the right one here. The moments
`m01` and `m10` are unweighted sums of `intensity * dx` over the disc. Blurring
before summing spreads each pixel's energy over its neighbours, which
low-amplitude, low-frequency structure - exactly the signal a dim frame has and
exactly what needs orienting - into a flatter, noisier estimate. The dominant
axis then becomes less determined, and a steering angle that is slightly wrong
rotates the whole 31x31 sampling pattern the wrong way, which is far worse than
an orientation computed from slightly noisier but higher-contrast pixels.

On a well-exposed frame, where the raw moments are already dominated by real
structure, blurring only removes signal. That is why fr1_xyz still reaches 45/45
but with camera-centre error up 3.6x: the poses are still found, from worse
orientations.

This is worth recording as a case where the reference implementation is not the
target. Matching OpenCV is a reasonable default and it was the right call for
most of the pipeline, but "the moments should be taken on the same image as the
descriptor" is an assumption, not a requirement, and here it is measurably
false.

## The two variables that actually predict the result

Contrast alone does not separate the outcomes - TUM fr1_xyz has the lowest
block standard deviation of the four and registers 45/45. What does separate them
is how much of the local variation is *signal* rather than noise: the mean
absolute Laplacian (a high-frequency measure) divided by the local contrast.

| sequence | block std | Laplacian / contrast | registered |
| --- | ---: | ---: | --- |
| TUM fr1_xyz | 14.2 | **0.57** | 45/45 - 100% |
| ETH3D courtyard | 26.7 | **0.62** | 7/8 - 87.5% |
| ETH3D electro | 7.6 | **1.24** | 3/12 - 25% |
| TUM fr1_desk | 3.0 | **1.59** | 23/40 - 57.5% |

The two sequences that work sit near 0.6; the two that do not sit above 1.2. The
split is clean, it is independent of contrast (electro has 7.6 and fr1_xyz has
14.2, yet fr1_xyz is the one that registers), and it explains why several
plausible fixes backfired.

A Laplacian-to-contrast ratio this high means a frame whose pixel-to-pixel
variation is mostly sensor noise rather than structure. That is the correct
picture of the problem:

- **It explains the smoothing result.** Blurring a noise-dominated frame averages
  noise down but also flattens the weak real structure that orientation needs,
  so smoothing the moments removes more signal than it removes noise. On
  fr1_xyz, whose ratio is 0.57, the same change costs only accuracy.
- **It explains why lowering the FAST threshold failed.** Corners detected in
  noise do not repeat under camera motion, so they add landmarks that are wrong
  rather than right - the map shrank from 2,712 to 725 landmarks.
- **It explains why every post-hoc filter failed.** A descriptor formed from
  noise-dominated patches is mostly noise, and a 256-bit Hamming distance over
  mostly-noise bits is mostly noise, so filtering the correspondence set cannot
  recover a match that was never in the descriptor.
- **It explains why equalisation hurt.** Stretching the histogram of a
  noise-dominated frame amplifies the noise along with the signal, and ORB's
  Harris response then ranks noise corners highly.

So the single unifying measurement across seventeen interventions is the ratio
above, and it says the bottleneck is **sensor noise, not descriptor capacity**.
That is a different conclusion from the one this work started with, and it
points at a different remedy: noise reduction before detection, chosen to
suppress noise while preserving the low-frequency structure the descriptor needs -
which is the opposite of what a Gaussian blur does. A bilateral or non-local
means filter is the natural candidate, and it has not been tried.

This is stated as the best-supported hypothesis from the measurements so far, not
as a settled result. The correlation is across four sequences and there is
confounding - fr1_desk is also a fast-motion handheld sequence, and electro is
also handheld and oblique. Isolating noise from motion would need sequences that
vary one at a time, which is not something the available data allows.

### Testing the noise hypothesis directly, and it fails too

The ratio above suggested an obvious remedy: suppress noise while preserving the
low-frequency structure a descriptor needs, which is what an edge-preserving
filter does and the opposite of what a Gaussian blur does. Measured on fr1_desk,
a median filter does exactly what the prediction said it would:

| filter | Laplacian / contrast | block std |
| --- | ---: | ---: |
| raw | 1.68 | 2.73 |
| median 3 | 1.49 | 2.54 |
| median 5 | 1.33 | 2.31 |
| median 7 | **1.17** | 2.16 |

The noise ratio falls by a third and the contrast is largely retained - precisely
the trade the hypothesis called for. A size-7 median was implemented in the
pipeline and measured:

| median size | 0 (off) | 3 | 5 |
| --- | --- | --- | --- |
| registered | **23/40** | 9/40 | 8/40 |
| 3D points | **2,712** | 745 | 555 |

It fails, and in the same way every other intervention failed: the map *shrinks*.
23/40 to 8/40, 2,712 landmarks down to 555.

This is now a strong and repeated result. Every attempt to improve the input -
more features, more scale levels, a deeper pyramid, contrast normalisation,
smoothing the moments, edge-preserving noise removal - reduces the number of
landmarks and the number of registrations together. Whatever these sequences are
short of, it is not pixel-level SNR, and the Laplacian ratio is correlated with
the outcome without being the thing that fixes it.

The likely reading is that the ratio is a *proxy* for something else that these
four sequences vary together: the handheld ETH3D and fr1_desk captures are fast
and oblique, while fr1_xyz is slow and near-frontal. Low measured contrast,
high measured noise ratio, large viewpoint change, and weak repeatability are all
downstream of "the camera moved a long way between frames on a scene with little
to go on", and any of these interventions removes real structure along with the
noise, because at this signal level structure and noise are not separable by a
local filter.

So the unifying statement in the previous section needs narrowing: the ratio is
the best predictor found, but it is a proxy for capture difficulty rather than a
cause, and the cause remains unisolated. Closing this needs either a descriptor
that survives a large viewpoint change, or sequences that vary one factor at a
time - and the data needed to separate the factors is not available here.

## FAST scoring: the same signature again

`Orb` has a `ScoreType` enum with a `Fast` variant, and no setter for it, so
Harris is the only reachable option. Harris is a second-derivative corner
response over a window - a gradient of gradients - and on a frame whose
pixel-to-pixel variation is mostly noise it ranks noise highly and confidently,
since noise differentiates to exactly the pattern Harris looks for. FAST asks a
more structural question: is a contiguous 9-pixel arc of the Bresenham circle all
brighter than the centre by a threshold.

A setter and a CLI flag were added and it measured worse on the noisiest
sequence:

| score | registered | 3D points |
| --- | --- | ---: |
| Harris (default) | **23/40** | **2,712** |
| FAST | 8/40 | 718 |

Reverted, and `with_score_type` reverted with it.

### The pattern across eighteen interventions

Every attempt to change *what the detector sees* has produced the same pair of
movements, together: fewer landmarks and fewer registrations. Not one has
decoupled them.

| change | landmarks | registrations |
| --- | --- | --- |
| FAST threshold 20 -> 5 | 2,712 -> 786 | 23/40 -> 9/40 |
| median filter 7 | 2,712 -> 555 | 23/40 -> 8/40 |
| FAST scoring | 2,712 -> 718 | 23/40 -> 8/40 |
| pyramid 8 -> 16 levels | 455 -> 411 | 3/5 -> 3/5 |
| histogram equalisation | 455 -> 245 | 3/5 -> 3/5 |
| smoothing the moments | - | 23/40 -> 8/40 |

If the bottleneck were supply of corners, adding or sharpening them would raise
the landmark count and the registration count together. Instead the two move
down in lockstep, which means the pipeline is not corner-starved: it is
producing corners that do not repeat, and every intervention removes some real
ones along with the spurious ones without finding new repeatable ones.

That reframes what is left. The map does not need more features; it needs
features that survive a large viewpoint change between frames. The measurements
put that change at up to 4.46 m of baseline and 59 degrees of mean rotation on a
handheld oblique view, which is well past what rotation-normalised BRIEF over a
3.58x pyramid is designed for.

Nothing further in this family is worth trying. The remaining options are a
different descriptor, or an association method that verifies geometrically
before committing - and both are substantial work rather than a configuration
change. The eighteen measurements above are what narrow it to those two, and
they are the reason to stop sweeping parameters.

## A real bug that the sweep found: the pyramid scale was discarded

`Orb::detect` stores each keypoint's pyramid level scale in `kp.size` -
`patch_size * scale` for the level that found it. That value was then never
read. `compute_orb_descriptor` took `patch_size` as a parameter and sampled a
fixed-size pattern against a fixed half-patch, so a corner found at level 0 (a
31px patch on the full-resolution image) and the same corner found at level 5 (a
55px patch spanning more of it) produced **bit-identical descriptions**.

That defeats the pyramid entirely. Scale-space detection is only useful if the
descriptor is measured at the scale the keypoint was found at; otherwise the
pyramid detects at eight scales and describes all eight identically, and
matching between an image and a half-size copy of it cannot work.

Making the pattern scale-aware is a one-line change to the sampling. Measured:

| sequence | fixed patch | scale-aware |
| --- | --- | --- |
| TUM fr1_xyz | 45/45 — 1.67 cm | 45/45 — 2.03 cm |
| TUM fr1_desk | **23/40 — 2,712 pts** | 21/40 — 3,597 pts |
| ETH3D electro, 5 views | 3/5 — 455 pts | **4/5 — 946 pts** |
| ETH3D courtyard, 8 views | **7/8 — 3,115 pts** | 3/8 — 1,694 pts |

Re-measured after the off-by-one below was fixed, so these supersede the first
version of this table. Only ETH3D electro improves, and it is the weakest
sequence here, so the trade is one low-texture sequence for three that already
worked. The landmark count still rises substantially where the flag is on
(2,712 -> 3,597 on fr1_desk, 455 -> 946 on electro), which is the direction
this work has been trying to move and the only one of the nineteen
interventions that achieves it.

### The off-by-one that this nearly shipped

The first version of that table was wrong, and wrongly optimistic. Measuring it
surfaced a regression: fr1_desk had gone from 23/40 to **8/40** on a clean tree,
deterministically, across repeated runs. It was not the data.

The half-extent was computed as `round(patch_size * level_scale)`, which is 31
for the base case rather than 15 - integer division of an odd width, which the
original code got right. The border test was therefore sixteen pixels stricter in
each axis, silently dropping a ring of keypoints **with the flag off**, so a
behaviour change had altered the default path. The failure mode was nasty: it
was reproducible, deterministic, plausible as a property of the data, and only
visible by comparing the default path against the pre-change build.

Fixed, and pinned by `default_path_is_unchanged_by_the_scale_aware_flag`, which
asserts that most descriptors lie well inside the frame and that extraction is
byte-identical across runs. The corrected table is above; the first one is
recorded here because the optimistic numbers were written down first, and a
result that has to be retracted is worth keeping the retraction for.

Shipped as `Orb::with_scale_aware_descriptor` and `--scale-aware-descriptor`,
**off by default**. On the corrected measurement it helps one weak sequence and
hurts three that already worked, which is a poor default trade. It stays
available because the landmark-count increase is the only one seen, and because
electro's gain is on precisely the sequence class this work has been trying to
improve.
