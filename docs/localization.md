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

First real-data measurements, run on the TUM RGB-D sequences below. These are
single runs on one machine, reported with the exact command so they can be
reproduced or contradicted. Dataset root: `/home/Phoenix/RUST/datasets`.

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

| Metric | same sequence | cross sequence |
| --- | ---: | ---: |
| database frames / landmarks | 102 / 25,849 | 120 / 23,277 |
| observations per landmark | 1.52 | 1.56 |
| **hit rate @1** | **75.0%** | **93.3%** |
| hit rate @5 | 98.3% | 100% |
| hit rate @10 | 100% | 100% |
| localization success rate | 25.0% (15/60) | 21.7% (13/60) |
| translation error (median) | 0.066 m | 0.054 m |
| rotation error (median) | 2.63° | 3.04° |
| mean inliers | 24.6 | 18.2 |
| mean query time | 1651 ms | 1879 ms |

The hit-radius for a correct retrieval is 1.0 m, and hit rate is the
localization convention: a query counts when *any* of its top-k candidates is
within that radius. The same runs report `recall@1 = 1.4-1.6%` under the
information-retrieval convention (fraction of *all* frames within the radius
that were retrieved); on a database this dense dozens of frames qualify per
query, so that number says nothing about retrieval quality — both are printed
side by side in the report for exactly that reason.

**How to read this.** Retrieval is the part that works: across a different
traverse of the same room, the correct place is ranked first 93% of the time.
Localization success (21-25%) is the bottleneck, and the cause is visible in the
table: the map has only ~1.5 observations per landmark, because the benchmark
harness triangulates each consecutive database pair and does not merge tracks.
A landmark seen by one or two keyframes supports few 2D-3D correspondences, so
PnP often lacks the 12 inliers it requires. The pose is accurate when it
succeeds (5-7 cm), which is consistent with that diagnosis: the localizer is not
wrong, it is starved. Track merging and a denser reconstruction are the next
piece of work, and these numbers are the baseline it has to beat.

Earlier sparse configuration for comparison (31 database frames, 2,308
landmarks, same sequence): 3.2% success, 0.044 m median translation error —
map density, not the matching stage, moves this number.

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
