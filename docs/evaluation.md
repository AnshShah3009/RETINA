# Evaluation and benchmarking

`rust-cv-native` ships the measurement layer for visual localization, SfM and SLAM
work: dataset loaders, standard error metrics, and a command-line tool that ties
them together. This exists so results can be produced, checked and re-run instead
of quoted.

- `cv-eval` — trajectory, reconstruction and retrieval metrics.
- `cv-io::datasets` — loaders for EuRoC/ASL, TUM RGB-D, KITTI odometry and COLMAP text models.
- `cv-bench` — CLI over both.

## Honesty policy

This repository publishes **no benchmark numbers it has not measured on this
machine, with the command that produced them**. There are no comparison tables
against COLMAP, ORB-SLAM3 or any other system here, because none have been run in
this repository yet. When results are added, they must include the exact command,
the dataset revision, the hardware and the raw output. Any number without those is
not a result.

## Metrics

### Trajectory (`cv_eval::trajectory`)

| Function | Meaning |
| --- | --- |
| `Trajectory::ate(&gt, Alignment::{None,Se3,Sim3})` | absolute trajectory error over camera centres, after Umeyama alignment; returns rmse/mean/median/max/std plus the estimated similarity transform |
| `Trajectory::rpe(&gt, delta_frames)` | relative pose error (translation and rotation) between poses `delta_frames` apart |
| `Trajectory::camera_center_rmse(&gt)` | RMSE of camera centres with no alignment |
| `Trajectory::path_length()` | accumulated camera-centre path length |

`Alignment::None` reports raw error; `Se3` estimates rotation and translation only
(rigid, the usual ATE convention); `Sim3` also estimates scale (use it when the
estimate comes from monocular SfM, which is only defined up to scale).

### Reconstruction (`cv_eval::reconstruction`)

`registration_rate`, `reprojection_rmse`, `rmse_over_extent`, `chamfer_distance`
(symmetric nearest-neighbour), `f_score`.

### Retrieval (`cv_eval::retrieval`)

`recall_at_k`, `precision_at_k`, `mean_average_precision` for ranked candidate
lists against ground-truth relevant sets — the metrics used to report visual
localization place recognition.

## Dataset loaders (`cv_io::datasets`)

| Module | Format |
| --- | --- |
| `euroc` | `cam0/data.csv`, `state_groundtruth_estimate0/data.csv` (nanosecond timestamps, position, quaternion, velocity, biases) |
| `tum` | `rgb.txt` / `depth.txt` index files, `groundtruth.txt`, plus `associate()` implementing TUM's nearest-timestamp one-to-one pairing |
| `kitti` | `poses.txt` row-major 3x4 `[R\|t]`, `times.txt` |
| `colmap` | text model: `cameras.txt`, `images.txt`, `points3D.txt` |

Every loader returns a descriptive `Result` — malformed, truncated or
non-numeric input is an error, never a panic, because these files come from
outside the process.

## CLI

```bash
cargo run -p cv-cli --bin cv-bench -- help
```

### Trajectory evaluation

```bash
cv-bench trajectory \
  --estimate  estimate.txt \
  --ground-truth groundtruth.txt \
  --format tum --align se3 --rpe-delta 1 --max-dt 0.02
```

Poses are paired by timestamp for `tum` (`--max-dt` is the largest allowed gap,
default 0.02 s); `kitti` and `euroc` are index-aligned, truncating to the shorter
list with a warning. Output: pose counts, path length of both trajectories,
camera-centre RMSE, ATE statistics for the chosen alignment, and RPE
translation/rotation RMSE.

For a monocular estimate against a metric ground truth, `--align sim3` recovers
the scale that the estimate is missing; the reported `scale` is that estimate.

### Reconstruction evaluation

```bash
cv-bench model \
  --images    sparse/0/images.txt \
  --points3d  sparse/0/points3D.txt \
  --cameras   sparse/0/cameras.txt
```

Reports camera and image counts, registration rate (images with at least one 2D
observation), total and per-image observations, and — with `--points3d` — point
count, mean track length and COLMAP's mean reprojection error. Each optional file
is genuinely optional.

### Retrieval evaluation

```bash
cv-bench retrieval --predictions ranked.txt --ground-truth relevant.txt --k 10
```

One query per line in both files; ids are comma- or space-separated and ranked
best-first. Prints recall@k, precision@k and mAP.

## Reproducing a measurement

```bash
cargo test --workspace          # the metrics themselves are tested against analytic cases
cargo run -p cv-cli --bin cv-bench -- trajectory --estimate a.txt --ground-truth b.txt --format tum
```

The metrics are unit-tested against closed-form cases: an SE(3)-transformed
trajectory must give ATE ≈ 0, a scaled trajectory must be recovered by Sim(3),
RPE of a rigidly transformed trajectory must be ≈ 0, and the retrieval metrics are
checked against hand-computed rankings.

## Where this is going

The pieces here are the prerequisites for the end-to-end pipelines (localization,
tracking, mapping, SLAM), not a substitute for them. A pipeline crate can be
evaluated with these tools the moment it exists; until then, treat the metric
layer as the contract that any future pipeline has to satisfy.
