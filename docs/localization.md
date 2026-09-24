# Visual localization

`cv-localization` answers the question a localization system is judged on: given
a query image, where was it taken? It sits on the existing primitives — ORB/SIFT
detectors, binary matching, PnP + RANSAC, the retrieval index in `cv-features` —
and adds the database, the retrieval-to-pose flow, and the evaluation hook.

## The flow

```
query keypoints + descriptors
        │
        ├─ retrieve candidates        vocabulary + BoW inverted index, or
        │                             descriptor-match ranking, or LSH
        ├─ match to the best candidate ratio test (default 0.75)
        ├─ lift to 2D–3D               via the candidate's landmark indices
        ├─ solve                       PnP + RANSAC, then refine on inliers
        └─ LocalizationResult          pose, inliers, matches, candidates, reprojection RMSE
```

`Localizer::localize` returns `None` rather than a confident wrong pose: too few
matches or a failed solve is a failure, not a guess.

## Building a database

A `Database` is a set of `Landmark`s (a 3D point plus the descriptors that
observe it) and `DatabaseImage`s (pose, keypoints, descriptors, and the landmark
each keypoint sees). Because a COLMAP text model already carries poses,
observations and tracks, `database_image_from_colmap` maps one into the other —
descriptors are the caller's job, since COLMAP does not store them.

```rust
let mut db = Database::new(Some(vocabulary));
for image in &colmap_images {
    db.add_image(database_image_from_colmap(image, &point_id_to_landmark));
}
for (id, point) in landmarks {
    db.add_landmark(Landmark { position: point.position, descriptors: point.descriptors });
}
db.build();                       // BoW inverted index + LSH index
```

## Retrieval

`cv_features::retrieval` provides the three components such systems use:

- `Vocabulary::train` — k-medians in Hamming space (per-bit majority
  re-estimation), deterministic for a given seed.
- `BowVector` + `BowDatabase` — TF-IDF weights and an inverted index that scores
  only images sharing a word.
- `LshIndex` — multi-table bit-sampling LSH over binary descriptors.

Everything is deterministic: the same seed and inputs give the same ranking, so a
localization result can be reproduced.

## Evaluating

`evaluate_localization` summarises a batch of queries (success rate, mean and
median translation and rotation error), and `cv-eval::retrieval` scores the
place-recognition stage on its own (recall@k, precision@k, mAP).

```bash
cargo run -p cv-localization --example synthetic_localization --features synthetic
```

```
scene: 200 landmarks, 8 views, extent 51.55
localized: inliers=75 matches=75 candidates=[4, 5, 3, 2, 1, 6, 7, 0] rmse=0.000px
translation error 3.245e-16, rotation error 0.000e0 deg, success rate 1.00
```

## Measured results

Real-data measurements on the TUM RGB-D sequences below. Single runs on one
machine, reported with the exact command so they can be reproduced or
contradicted. Dataset root: `/home/Phoenix/RUST/datasets`.

Sequence sizes: `rgbd_dataset_freiburg1_desk` 613 RGB frames,
`rgbd_dataset_freiburg1_xyz` 798 RGB frames, both with ground truth.

```bash
cargo build --release -p cv-localization --features synthetic --example tum_benchmark

# same sequence: database from the first half, queries from the second half
./target/release/examples/tum_benchmark \
    --db-dir  datasets/rgbd_dataset_freiburg1_desk \
    --db-frames 120 --query-frames 60 --stride 3 --features 1200

# cross sequence: database from one traverse, queries from another
./target/release/examples/tum_benchmark \
    --db-dir  datasets/rgbd_dataset_freiburg1_desk \
    --query-dir datasets/rgbd_dataset_freiburg1_xyz \
    --db-frames 120 --query-frames 60 --stride 3 --features 1200
```

### Track-merged map (current default, `--match-window 5`)

| Metric | same sequence | cross sequence |
| --- | ---: | ---: |
| landmarks | 11,458 | 11,315 |
| observations per landmark (mean / median) | 2.75 / 2.00 | 2.52 / 2.00 |
| **hit rate @1** | **75.0%** | **93.3%** |
| hit rate @5 / @10 | 98.3% / 100% | 100% / 100% |
| localization success | 21.7% (13/60) | 21.7% (13/60) |
| translation error (median) | 0.044 m | 0.036 m |
| rotation error (median) | 1.69° | 2.28° |
| mean inliers | 23.4 | 19.3 |

### Before track merging (per-pair triangulation, kept for comparison)

| Metric | same sequence | cross sequence |
| --- | ---: | ---: |
| landmarks | 25,849 | 23,277 |
| observations per landmark | 1.52 | 1.56 |
| hit rate @1 | 75.0% | 93.3% |
| localization success | 25.0% (15/60) | 21.7% (13/60) |
| translation error (median) | 0.066 m | 0.054 m |
| rotation error (median) | 2.63° | 3.04° |

The old map looked bigger only because every frame pair produced its own
triangulation of the same physical point. Merging matches into tracks (union-find
over `(frame, keypoint)` links inside a temporal window, triangulated from the
widest-baseline pair and verified by reprojection in **every** observing view)
roughly halves the pose error on both configurations at a smaller landmark count,
because each landmark now carries the descriptors of all the views that saw it.

### Match-window sweep (cross sequence)

Wider windows build longer tracks: better geometry per landmark, fewer landmarks
overall. There is no dominant setting, so the trade-off is published rather than
hidden behind a default.

| `--match-window` | landmarks | obs/landmark | success | translation (median) | rotation (median) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 13,168 | 2.38 | 15.0% | 0.038 m | 1.46° |
| **5 (default)** | **11,315** | **2.52** | **21.7%** | **0.036 m** | **2.28°** |
| 8 | 10,570 | 2.62 | 16.7% | 0.032 m | 1.46° |

Same-sequence success, for contrast, is higher at window 2 (28.3%) than at
window 5 (21.7%). Window 5 is the default because the cross-sequence case — a
different traverse of the same room — is the more representative task, and it is
the setting that keeps success at the pre-merge level while gaining the accuracy.

The hit radius for a correct retrieval is 1.0 m, and hit rate uses the
localization convention: a query counts when *any* of its top-k candidates is
within that radius. The same runs report `recall@1 = 1.4-1.6%` under the
information-retrieval convention (fraction of *all* frames within the radius
retrieved); on a database this dense dozens of frames qualify per query, so that
number says nothing about retrieval quality — both are printed side by side in
the report for exactly that reason.

**How to read this.** Retrieval is the part that works: across a different
traverse of the same room, the correct place is ranked first 93% of the time.
Localization success (21.7%) is the bottleneck, and it is a coverage problem,
not a matching problem: the poses that are recovered are accurate to 3.6-4.4 cm
and 1.7-2.3°, and mean inliers on successes are 19-23 against the 12 required.
The map is built from a single traversal with a temporal matching window, so a
query view often has too few correspondences to reach the inlier floor. Building
the map from more viewpoints — or estimating the database poses instead of
taking them from ground truth, which is what a real system must do — is the next
piece of work. These numbers are the baseline it has to beat.

## Mapping results

`cv-sfm` builds a map from images with no ground-truth poses, and is scored with
`cargo run --release -p cv-sfm --example tum_sfm -- --dir <dataset> --frames N
--stride S --window W`. These are the measured numbers that established the
performance work and the remaining gaps.

| sequence | views | stride | window | registered | points | centre RMSE | rotation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fr1_xyz | 14 | 10 | 3 | **100.0%** | 1,784 | 2.0 cm | 1.1° |
| fr1_xyz | 60 | 5 | 3 | **73.3%** | 5,658 | 5.4 cm | 9.2° |
| fr1_desk | 40 | 5 | 3 | 22.5% | — | — | — |
| fr1_desk | 40 | 2 | 3 | **100.0%** | 3,257 | 8.5 cm | 6.5° |
| fr1_desk | 150 | 2 | 3 | 43.3% | 6,456 | 13.0 cm | 18.1° |

(The xyz rows predate the bundle-adjustment fix; the desk rows are after it.)

**Frame spacing dominates the registration rate.** The same mapper, the same
code, the same sequence: at stride 5 it registers 22.5% of 40 views, at stride 2
it registers 100%. `fr1_desk` is a slow trajectory, so stride 5 samples frames
that no longer overlap enough to seed or to support PnP — 35 of 60 views failed
with "no 3D point visible in this view", the signature of a map that cannot grow
past the seed neighbourhood. Widening the pair window (3 → 6 → 10) barely helps
(15.0% → 16.7% → 16.7%): the limiting factor is inter-frame motion, not the pair
graph. A production pipeline must choose the stride from the sequence, not fix it.

**Two failures remain, and they are different problems.**

1. *Coverage* — solved by adequate frame spacing, as above.
2. *Drift* — at 150 views the map is still incomplete (43.3%) and the poses
   have drifted badly (18.1° rotation). There is no local bundle adjustment: BA
   runs globally every ten registrations, so an error introduced early is never
   repaired locally and the whole reconstruction bends. This is the next piece of
   work, and it is what separates this mapper from the mature implementations.

For reference, the mapper currently uses: 1.5 px / 500-iteration F-matrix RANSAC
for pair verification, eight-point essential-matrix seeding from the best of 8
hypotheses, 1° minimum parallax and 4 px reprojection for triangulation, and
registration gated at ≥10 PnP inliers and ≥0.25 inlier ratio (the ratio is load
-bearing — removing it registered one more view and cost an order of magnitude in
accuracy). It has no homography-versus-essential model selection, no local BA, and
no observation filtering.

## What is proven, and what is not
Proven by the test suite, on a deterministic synthetic scene:

- a clean query localizes to machine precision (translation error ~3e-16 of a
  51.55-unit scene extent) with 75 inliers, and the correct database view ranks
  first among candidates;
- Gaussian pixel noise degrades the result gracefully — sigma = 1 px gives
  0.019% of extent and 0.08 degrees, sigma = 2 px gives 0.057% and 0.16 degrees;
- a query with no overlap returns `None` instead of a wrong pose;
- repeated queries are bit-identical.

Not proven here, and therefore not claimed: any result on a real dataset. There
is no ETH3D, EuRoC, TUM, KITTI or OpenLORIS number in this repository, because
none has been measured. When one is, it must come with the command, the dataset
revision, the hardware and the raw output — see
[docs/evaluation.md](evaluation.md).

## Status against the reference implementations

Relative to `visloc-rs` (the reference this layer was built to be measured
against), what exists here is the localization capability and its measurement
layer. What does not exist yet: an incremental SfM mapper that builds the map
from images, a VI-SLAM fusion stack, and published real-data comparisons. Those
are the next pieces, and the evaluation tooling is already in place to judge them.
