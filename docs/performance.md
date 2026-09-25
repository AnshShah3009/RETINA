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
