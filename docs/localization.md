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
