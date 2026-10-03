# Capability-gap comparison: RETINA vs. visloc-rs

**Purpose.** `GAPS.md` records what this workspace *claims*; `docs/bug-log.md` records
what has been *fixed*. Neither compares the workspace against a real, running
implementation of the same problem in the same language. This document does.

**Reference.** `/home/Phoenix/refs/visloc-rs` — "Structure from Motion,
visual-inertial SLAM, 3D Gaussian Splatting and map-based localization — in pure Rust."
COLMAP and GTSAM are present under `/home/Phoenix/refs/` and are used only where noted.

**Verdicts used (only these three):**

| Verdict | Meaning |
|---|---|
| **PRESENT and comparable** | the workspace has it and it looks equivalent in scope |
| **ABSENT** | the reference has it, the workspace does not |
| **WEAKER** | the workspace has it but with reduced scope; the row says exactly how |

**Conventions.** `WS:` = a `file:line` in this workspace. `REF:` = a `file:line` in
visloc-rs. Both are workspace-relative for `WS:` and absolute for `REF:`. A row
without a workspace pointer is a claim; a row without a reference pointer is a guess.

---

## Scale context (measured, not estimated)

| Measure | RETINA (`crates/`, `src/`) | visloc-rs (`crates/`, `pipelines/`, `src/`) |
|---|---|---|
| Rust source lines | **190,528 lines** in 542 `.rs` files (`crates/` + `src/`) | **57,084** lines in 136 files (`crates/`) + **3,889** in `src/` + **254,844** in `pipelines/` |
| `#[test]` / `#[tokio::test]` count | **2,419** | **535** |
| Test files | 152 | 14 (all in `crates/*/tests/`) |

visloc-rs's depth is not in `crates/` — it is in `pipelines/` (254k lines:
`basalt/src/vio/aom.rs` alone is 21,486 lines, `slam/src/incremental_sfm.rs` 18,506,
`slam/src/bundle.rs` 17,047). RETINA is broader in surface area (image processing,
plotting, plotting-to-SVG, video I/O, signal processing, a HAL/GPU backend, a Python
binding layer) but far shallower in the SLAM/SfM core. This comparison is therefore
area-by-area, not a verdict on either project.

---
## 1. Place recognition / loop closure

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Bag-of-words vocabulary + TF-IDF + inverted-file retrieval | PRESENT and comparable | WS: `crates/features/src/retrieval/vocabulary.rs:15` (`Vocabulary::train`), `crates/features/src/retrieval/bow.rs:14` (`BowVector`), `crates/features/src/retrieval/database.rs:86` (`BowDatabase::query`) | REF: `/home/Phoenix/refs/visloc-rs/crates/vision/src/vocab_tree/index.rs:1` (`InvertedFile`/`InvertedIndex`, TF-IDF + burstiness), `/home/Phoenix/refs/visloc-rs/crates/vision/src/vocab_tree/hkm.rs:62` | Both are flat-k-means-over-byte-descriptors + IDF-weighted inverted index. Comparable in kind. |
| Hierarchical / recursive vocabulary (HKM) | ABSENT | WS: no `hkm`/`hierarchical`/`branching_factor` anywhere in `crates/features/src/retrieval/` (grepped `hkm`, `hierarchical_kmeans`, `branching_factor` — 0 hits) | REF: `crates/vision/src/vocab_tree/hkm.rs:396` lines, `HierarchicalVocabulary::build` at `hkm.rs:62` | visloc-rs's HKM recurses (branching factor × depth) reusing the same deterministic k-means++ at each node. |
| Hamming-embedding projection for retrieval scoring | ABSENT | WS: `crates/features/src/retrieval/lsh.rs:17` is random-projection **LSH over `u64` bytes** for descriptor→candidates, not a Hamming embedding for TF-IDF scoring; no `embedding_dim`/median-threshold code in `crates/features/` (grep `embedding_dim`, `median` in `crates/features/src/retrieval/` → 0) | REF: `crates/vision/src/vocab_tree/index.rs:11-23` (`ComputeHammingEmbedding`, per-word **median** threshold, `kMinEntries = 5`), `index.rs:35` (`HammingWeightLut`, `exp(-h²/σ²)`, σ=16) | Different feature: LSH approximates nearest neighbours; a Hamming embedding re-weights TF-IDF scoring. Not interchangeable. |
| VLAD global descriptor | ABSENT | WS: grep `-i vlad` over `crates/` → **0 hits**. No VLAD anywhere. | REF: `crates/vision/src/place_recognition/mod.rs:192` (`pub fn vlad`), doc at `place_recognition/mod.rs:14-20` | visloc-rs also offers `mean_pool` (`:244`) and `cosine_similarity` (`:275`). |
| Mean-pooling global descriptor | ABSENT | WS: no global-image-descriptor aggregator; `crates/features/src/retrieval/bow.rs` is a histogram, not a pooled descriptor. | REF: `crates/vision/src/place_recognition/mod.rs:244` | |
| Cosine-similarity retrieval over global descriptors | ABSENT | WS: `crates/features/src/retrieval/bow.rs:98` `l1_distance`/`l2_distance` only; no cosine-similarity retrieval entry point. | REF: `crates/vision/src/place_recognition/mod.rs:275`, `retrieve_mutual` at `:306` | |
| Mutual-nearest-neighbour retrieval (query↔db) | ABSENT | WS: grep `mutual` in `crates/features/` → 0 hits. | REF: `crates/vision/src/place_recognition/mod.rs:306` `pub fn retrieve_mutual` | |
| Candidate-pair generation from retrieval (dedup + symmetric) | PRESENT and comparable | WS: `crates/sfm/src/mapper.rs:153` `PairSelection::SequentialWithRetrieval{window, vocab_size, neighbours, seed}` | REF: `crates/vision/src/vocab_tree/pair_generator.rs:70` `generate_pairs`, options at `:43` | Both turn top-N retrieval into a deduplicated symmetric pair stream. |
| Retrieval→geometric-verification→pose bridge composition | PRESENT and comparable | WS: `crates/sfm/src/mapper.rs:558` `map_views` (retrieval pairs → F-RANSAC → E/PnP registration), `crates/features/src/retrieval/database.rs:86` | REF: `crates/vision/src/place_recognition/mod.rs:411` `propose_bridges`, and `:536` `propose_metric_bridges` | |
| LSH / binary-hash descriptor index | PRESENT and comparable | WS: `crates/features/src/retrieval/lsh.rs:17`, `insert` `:80`, `query` `:95`, `candidate_count` `:130` | REF: `crates/vision/src/features/sift.rs` binary-hamming path — see also `crates/sift-gpu/src/matcher.rs` | visloc-rs's equivalent is GPU SIFT binary descriptors; the workspace's is a generic CPU LSH. Different substrate, same role. |

## 2. Feature detection and description

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| SIFT detector + descriptor | PRESENT and comparable | WS: `crates/features/src/sift.rs:28` (`Sift`), `build_scale_space` `:93`, `detect_and_compute` `:331`, `sift_detect_ctx` `:823` | REF: `crates/vision/src/features/sift.rs:1002` `extract_sift`, `:1759` `describe_sift_keypoints`, `:2985` `extract_sift_features` (4,381 lines) | visloc-rs has VLFeat orientation (`:31` `SiftDetector`, `:1747` `diagnose_sift_vlfeat_detector`) and 46 in-file tests vs the workspace's SIFT test count of 0 in-file. |
| ORB | PRESENT and comparable | WS: `crates/features/src/orb.rs:107` (`Orb`), `detect_and_compute` `:538`, `orb_detect_and_compute_scale_aware` `:1002`; GPU kernel `crates/hal/src/gpu_kernels/fast.rs` | REF: no ORB — grep `-i orb` in `crates/vision/src` → 0. The reference's binary fast corner is `crates/vision/src/features/mod.rs:227` `CornerFeatureExtractor` (Harris-family corner + `describe_at` `:286`) | Workspace ahead: ORB is the workspace's binary/FAST/hierarchy detector and its SIFT alternative for fast loop closure. |
| AKAZE | PRESENT and comparable | WS: `crates/features/src/akaze.rs:97` `Akaze::new`, `detect_and_compute_ctx` `:159` | REF: no AKAZE — grep `-i akaze` in `crates/` → 0. Nearest reference detector is the SIFT-only path `crates/vision/src/features/sift.rs:2985` | Workspace ahead. |
| BRIEF | PRESENT and comparable | WS: `crates/features/src/brief.rs:22`, `extract_brief` `:143` | REF: no BRIEF — grep `-i brief` in `crates/` → 0. The reference's binary descriptors come from GPU SIFT `crates/sift-gpu/src/extractor.rs` | Workspace ahead. |
| FAST corner | PRESENT and comparable | WS: `crates/features/src/fast.rs:8` `fast_detect`, `corner_score` `:158`, NMS `:279` | REF: `crates/vision/src/features/mod.rs:227` `CornerFeatureExtractor` (config `:210` `CornerFeatureConfig`) | Both have a fast corner detector; the workspace's also has the ORB descriptor stack built on it. |
| GFTT | PRESENT and comparable | WS: `crates/features/src/gftt.rs:17` `gftt_detect`, `:27` `gftt_detect_with_params` | REF: no GFTT — grep `-i gftt` in `crates/` → 0; nearest is the corner extractor `crates/vision/src/features/mod.rs:227` | Workspace ahead (a specific GFTT detector is not in the reference). |
| LBD line descriptor | PRESENT and comparable | WS: `crates/features/src/lbd.rs:33` `Lbd::new`, `compute` `:46`; line matcher `crates/features/src/line_matcher.rs` | REF: no line features — grep `-i '\blbd\b\|line descriptor'` in `crates/vision/src` → 0. Reference is keypoint-only (`crates/vision/src/features/mod.rs:26` `FeatureSet`) | Workspace ahead: the reference has no line-feature path at all. |
| Learned features (SuperPoint) | ABSENT | WS: the ONNX **runtime** exists — `crates/dnn/src/lib.rs:70` `DnnNet`, `:166` `load`, `:224` `forward` on `tract-onnx` (`Cargo.toml:144` `tract-onnx = "0.21.7"`) — but no SuperPoint (or any learned keypoint) front-end: grep `superpoint` over `crates/`, `src/` → **0**; `crates/features/` has no ONNX path | REF: `crates/vision/src/features/superpoint_onnx.rs:803` lines (8 in-file tests) | The gap is the *front-end*, not the runtime. A SuperPoint extractor would slot into the existing `DnnNet`. |
| LightGlue matcher | ABSENT | WS: grep `lightglue`, `glue` over `crates/` → **0**. Only classical matching exists (`crates/features/src/matcher.rs:16`) | REF: `crates/vision/src/features/lightglue_onnx.rs:477` lines (11 in-file tests) + `crates/vision/tests/lightglue_onnx_parity.rs` (7 tests) | |
| Learned global descriptor (ONNX) | ABSENT | WS: `DnnNet` (`crates/dnn/src/lib.rs:70`) is a generic runner with no global-descriptor head; grep `global_descriptor` over `crates/` → **0** | REF: `crates/vision/src/features/global_descriptor_onnx.rs:301` lines; consumed by `src/global_descriptor_store.rs:440` lines | |
| GPU SIFT (CUDA extractor + matcher crate) | ABSENT | WS: `crates/hal/src/gpu_kernels/sift.rs:18` `sift_extrema` is one WGSL kernel for DoG extrema, not a full GPU SIFT extractor+matcher pipeline. No `sift-gpu` equivalent crate. | REF: `crates/sift-gpu/src/extractor.rs:757` lines, `crates/sift-gpu/src/matcher.rs:369` lines, `crates/sift-gpu/src/descriptor_matcher.rs`, `crates/sift-gpu/src/lib.rs`; benches in `crates/sift-gpu/examples/` | **Caveat, flagged as unsure:** I did not read `sift-gpu`'s back-end to confirm whether it is CUDA (wgsl/compute-shader) rather than CPU. The structural gap — a dedicated GPU SIFT crate with its own extractor and descriptor matcher — is certain either way. |
| HOG-like deep feature adapter | ABSENT | WS: grep `hoglike\|hog_like\|DeepFeatureSet\|MultiScale` over `crates/` → **0** | REF: `crates/vision/src/features/deep.rs:190` `HogLikeFeatureExtractor`, `:505` `MultiScaleDeepExtractor`, `:404` `CornerDeepAdapter`, 14 in-file tests | |
| Feature-extractor trait / pluggable backend | WEAKER | WS: `crates/features/src/lib.rs` exposes concrete structs; no `FeatureExtractor` trait found (grep `trait FeatureExtractor` → 0) | REF: `crates/vision/src/features/mod.rs:18` `pub trait FeatureExtractor`, plus `FnFeatureExtractor` (`:403`) and `ProvidedFeatureExtractor` (`:383`) | The workspace has more detectors but no trait seam to swap them; visloc-rs has fewer detectors behind a trait. |

## 3. Feature matching

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Brute-force descriptor matching | PRESENT and comparable | WS: `crates/features/src/matcher.rs:16` `Matcher`, `match_descriptors` `:45`/`fn :155`, `knn_match` `:170` | REF: `crates/vision/src/matching/mod.rs:68` `BruteForceMatcher`, `l2_distance` `:79` | |
| Lowe ratio test | PRESENT and comparable | WS: `crates/features/src/matcher.rs:204` `filter_matches_by_ratio_test`, config `:39` `with_ratio_test` | REF: `crates/vision/src/matching/mod.rs:99` `match_descriptors_cross_checked` (ratio + cross-check in one) | |
| Mutual / cross-check filtering | WEAKER | WS: `Matcher::with_cross_check` `crates/features/src/matcher.rs:33` — a boolean on the matcher; grep `mutual` → 0, no standalone mutual-NN filter | REF: `crates/vision/src/matching/mutual_softmax.rs:362` lines, 8 in-file tests | visloc-rs has mutual-NN **plus a softmax correspondence-confidence stage** as a separate module with its own tests. |
| Approximate NN index (IVF / FLANN-style) | PRESENT and comparable | WS: `crates/features/src/flann.rs` (whole file), `crates/features/src/retrieval/lsh.rs:17` | REF: `crates/vision/src/matching/ivf.rs:81` `IvfMatcher`, `IvfConfig` `:56`, `train` `:123`, 5 in-file tests | Both have a trained ANN matcher over descriptor space. |
| GPU descriptor matching | WEAKER | WS: `crates/hal/src/gpu_kernels/matching.rs` (matching kernels), `crates/features/src/markers/gpu.rs:97` Hamming-distance marker matching only | REF: `crates/vision/src/matching/gpu.rs:427` lines + `crates/sift-gpu/src/matcher.rs:369` lines + `crates/sift-gpu/src/descriptor_matcher.rs` | The workspace's GPU matching kernel set is generic; visloc-rs has a dedicated CUDA SIFT matcher crate with a GPU descriptor matcher. |
| Matcher trait (pluggable) | PRESENT and comparable | WS: `crates/features/src/matcher.rs:16` `Matcher` trait + impls | REF: `crates/vision/src/matching/mod.rs:28` `pub trait Matcher`, `CrossCheckMatcher` `:461` | |
| Weighted / confidence-aware matching | ABSENT | WS: no `estimate_with_weights`-style weighting entry point in `crates/features/src/matcher.rs` | REF: `crates/vision/src/ransac/mod.rs:42` `estimate_with_weights` (PROSAC-style), `crates/vision/src/pnp/generalized.rs` weight path | The weighting exists in visloc-rs's RANSAC layer rather than the matcher. |

## 4. Two-view geometry (essential, fundamental, homography)

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Fundamental matrix — 8-point + Hartley normalisation | PRESENT and comparable | WS: `crates/calib3d/src/essential_fundamental.rs:50` `find_fundamental_mat`, `:284` `estimate_fundamental_8_point`, `:298` `sampson_error` | REF: `crates/vision/src/two_view/fundamental.rs:38` `estimate_fundamental_dlt`, `:91` `fundamental_squared_sampson_error` | |
| Essential matrix — 8-point | PRESENT and comparable | WS: `crates/calib3d/src/essential_fundamental.rs:32` `find_essential_mat`, `:204` `estimate_essential_8_point`, `:264` `enforce_essential_constraints` (singular-value projection) | REF: `crates/vision/src/two_view/mod.rs:145` `EightPointEssentialMatrixEstimator` | |
| Essential matrix — **Nistér 5-point** (minimal) | **ABSENT** | WS: `crates/calib3d/src/essential.rs:14` `EssentialSolver::estimate_5point` returns `Err("Nistér 5-point solver is not implemented…")` at `:34-39`. Its 2 in-file tests (`:47`, `:62`) assert it *errors*. | REF: `crates/vision/src/two_view/five_point.rs:1257` lines; `FivePointEssentialMatrixEstimator` at `crates/vision/src/two_view/mod.rs:234`; shared root-recovery reused from `pnp/gp3p.rs` (`two_view/five_point.rs:62`, `two_view/mod.rs:199`) | **Verified ABSENT** by reading the body — it is an explicit `AlgorithmError` stub, not a hidden fallback. |
| Fundamental-matrix RANSAC | PRESENT and comparable | WS: `crates/calib3d/src/essential_fundamental.rs:125` `find_fundamental_mat_ransac`, estimator `:78` `FundamentalEstimator` (min sample 8, `:83`) | REF: `crates/vision/src/two_view/fundamental.rs:149` `fundamental_ransac`, `FundamentalRansacConfig` `:120`, `FundamentalReport` `:138` | |
| Essential-matrix RANSAC | PRESENT and comparable | WS: `crates/calib3d/src/essential_fundamental.rs:96` `find_essential_mat_ransac`, threshold converted px→normalised via `f = 0.5*(fx+fy)` (`:104`) | REF: `crates/vision/src/two_view/mod.rs:326` `EssentialRansac`, `:306` `EssentialRansacConfig`, `:344` `estimate`, `:361` `estimate_with_weights` | |
| Homography DLT | PRESENT and comparable | WS: `crates/calib3d/src/homography.rs:12` `HomographySolver::estimate` → `crates/calib3d/src/dlt.rs` `solve_dlt_homography`; 3 in-file tests incl. exact 4-point recovery (`homography.rs:57`) | REF: `crates/vision/src/two_view/homography.rs:60` `estimate_homography_dlt`, `:33` `homography_squared_error` | |
| Homography RANSAC | PRESENT and comparable | WS: `crates/imgproc/src/stitching.rs:176` `find_homography_ransac` (in the stitching module, not calib3d) | REF: `crates/vision/src/two_view/homography.rs:130` `homography_ransac`, config `:104`, report `:122` | Same algorithm, different crate home on each side. |
| **Homography decomposition → multiple (R,t)** | **ABSENT** | WS: grep `decompose_homography`, `decomposeHomography`, `homography_decompose` over `crates/` → **0 hits**. `crates/calib3d/src/homography.rs` is 77 lines and ends at `estimate`. | REF: `crates/vision/src/two_view/homography.rs:270` `decompose_homography_matrix` (547-line module), `:230` `HomographyMotion`, `:504` `pose_from_homography_matrix` | A planar-scene pose path the workspace simply does not have. |
| **Two-view geometry classifier** (H/E/F + calibrated-vs-uncalibrated + multi-model support counting) | **ABSENT** | WS: no `classify`/`ConfigurationType` analogue; `crates/calib3d/src/` returns one matrix per solver with no model-count comparison | REF: `crates/vision/src/two_view/colmap_verification.rs:45` `ConfigurationType`, `:239` `TwoViewGeometryVerifier::classify`, options `:84` `TwoViewGeometryOptions`, report `:189`; 7 in-file tests | This is COLMAP's `EstimateTwoViewGeometry` port — the single most load-bearing verifier in either system. |
| Triangulation-angle computation for a two-view pair | ABSENT | WS: grep `triangulation_angle` in `crates/` → 0 | REF: `crates/vision/src/two_view/colmap_verification.rs:611` `two_view_triangulation_angle`, `:623` `two_view_pose_and_triangulation_angle` | |
| **Correspondence graph / view graph with transitive tracks** | **ABSENT** | WS: no `CorrespondenceGraph`; `crates/sfm/src/mapper.rs` tracks in its own private structures | REF: `crates/vision/src/two_view/correspondence_graph.rs:274` `CorrespondenceGraph` (1,160 lines), `add_two_view_geometry` `:392`, `extract_transitive_correspondences` `:535`, `finalize` `:589`; 14 in-file tests | |
| Relative-pose recovery with cheirality check + scale | PRESENT and comparable | WS: `crates/calib3d/src/triangulation.rs:105` `recover_pose_from_essential`, `:18` `triangulate_points` | REF: `crates/vision/src/two_view/mod.rs:504` `CheiralityOptions`, `:555` `RelativePoseRecovery`, `:581` `RelativePoseEstimator`, `:615` `estimate_with_scale`, `:708`/`728` `recover_relative_pose(_with_options)` | visloc-rs additionally reports `chirality_margin` (`:570`) as a diagnostic. |
| Bridge-candidate generation (connected components) | ABSENT | WS: 0 (`connected_components` in `crates/calib3d/` → 0; `crates/3d/src/spatial/` is unrelated geometry) | REF: `crates/vision/src/two_view/rescue.rs:93` `connected_components`, `:142` `generate_bridge_candidates`, 10 in-file tests | |

## 5. Pose from n points (PnP)

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| DLT PnP (linear, 6-point) | PRESENT and comparable | WS: `crates/calib3d/src/pnp.rs:55` `solve_pnp_dlt` | REF: `crates/vision/src/pnp/mod.rs:56` `DltPnP` | |
| EPnP | PRESENT and comparable | WS: `crates/calib3d/src/pnp.rs:935` `PnpSolver::estimate_epnp`, degenerate-rank guard `:1076`, ratio guard `:1085` | REF: no `EPnP` symbol anywhere in `crates/` or `pipelines/` (grep `epnp\|EPnP` → only unrelated `repnp_free_*` config keys in `pipelines/slam/src/global_sfm.rs:3138-3217`) | **Workspace ahead on EPnP.** visloc-rs relies on P3P/G3P + GN refinement instead. |
| P3P (Grunert minimal) | PRESENT and comparable | WS: `crates/calib3d/src/pnp.rs:727` `PnpSolver::estimate_p3p`, quartic exposed for testing at `:704-705`, 3 in-file tests (`:1287`…) | REF: `crates/vision/src/pnp/p3p.rs:35` `P3PGrunert`, 427 lines, 4 in-file tests (`:329`, `:359`, `:389`, `:402`) | Both are Grunert/Haralick quartic + absolute orientation. |
| PnP + RANSAC | PRESENT and comparable | WS: `crates/calib3d/src/pnp.rs:400` `solve_pnp_ransac` — **minimal sample k = 6** (`:415`), floor of 64 iterations (`:417`) | REF: `crates/vision/src/ransac/mod.rs:82` `PnPRansac<P, R>` generic over estimator + refiner | The workspace's 6-point sample is not the reference's minimal solver; visloc-rs can be given `P3PGrunert`. |
| Gauss-Newton pose refinement | PRESENT and comparable | WS: `crates/calib3d/src/pnp.rs:505` `solve_pnp_refine`, `:529` `solve_pnp_refine_ctx` | REF: `crates/vision/src/pnp/mod.rs:211` `GaussNewtonPoseRefiner` behind `trait PoseRefiner` (`:46`) | |
| **Generalized / non-central absolute pose (multi-camera rig)** | **ABSENT** | WS: grep `gp3p\|g3p\|GP3P\|gr6p\|generalized_pnp\|generalized camera` over `crates/` → **0 hits** | REF: `crates/vision/src/pnp/generalized.rs:1142` lines: `RigSensor` (`:31`), `GeneralizedCameraRig` (`:39`), `GeneralizedDltPoseEstimator` (`:133`); `crates/vision/src/pnp/gp3p.rs:1396` lines faithful COLMAP/PoseLib gP3P port; `crates/vision/src/pnp/gr6p.rs:2072` + `gr6p_data.rs:977` (g6P); 12 + 5 in-file tests | Three separate solvers the workspace has none of. |
| Pose refinement with pose prior / weights | ABSENT | WS: `solve_pnp_refine` takes no prior | REF: `crates/vision/src/ransac/mod.rs:58` `estimate_with_pose_prior_and_weights` | |

## 6. Camera calibration and camera models

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Pinhole intrinsics (K) | PRESENT and comparable | WS: `crates/core/src/geometry/camera.rs:125` `CameraIntrinsics`, `matrix` `:158`, `project` `:196`, `unproject` `:207`; f32 twin `:229` | REF: `crates/core/src/types/camera.rs:83` `Camera`, `intrinsics()` `:200`, `project` `:269`, `unit_ray_from_pixel` `:343` | |
| **Camera-model enum covering COLMAP's full set** | WEAKER | WS: no `CameraModel` enum in the workspace. Two fixed models: `crates/core/src/geometry/camera.rs:22` `PinholeModel` / `:74` `PinholeModelF32`, with distortion carried by `crates/core/src/geometry/distortion.rs:6` `Distortion` (k1,k2,p1,p2,k3) and `:433` `FisheyeDistortion` (k1..k4). No `FOV`, no `DoubleSphere`, no `SimpleRadial`/`Radial` variants. | REF: `crates/core/src/types/camera.rs:15` `CameraModel` with **11 variants**: `Pinhole`, `SimplePinhole`, `SimpleRadial`, `Radial`, `OpenCv`, `FullOpenCv`, `OpenCvFisheye`, `SimpleRadialFisheye`, `RadialFisheye`, `Fov`, `DoubleSphere`, `Unknown`; COLMAP name mapping both directions (`:37` / `:55`) | **How reduced:** the workspace has *2* models (pinhole-family + Kannala-Brandt fisheye); the reference has 11 with a COLMAP name round-trip. A `RADIAL`- or `FOV`-intrinsics COLMAP model cannot be expressed. |
| Distortion: Brown-Conrady k1..k3 + p1,p2 | PRESENT and comparable | WS: `crates/core/src/geometry/distortion.rs:15` `Distortion`, `apply` `:30`, `remove_checked` `:70`; f32 `:184` | REF: `crates/vision/src/distortion.rs:34` `RadialTangential`, `from_euroc_coefficients` `:59`, `distort_normalized` `:82`, `undistort_normalized` `:101` | |
| Distortion: **rational polynomial (k4,k5,k6)** | ABSENT | WS: `Distortion::new` takes exactly 5 params (`crates/core/src/geometry/distortion.rs:15`) — no k4/k5/k6 slot exists | REF: `crates/core/src/types/camera.rs:21-22` `FullOpenCv` `[fx,fy,cx,cy,k1..k6,p1,p2]` | |
| Distortion: thin-prism / division / per-axis tangential | ABSENT | WS: `crates/core/src/geometry/distortion.rs` has no `s1..s4` (thin prism) or `taux/tauy` | REF: not present either — **both absent** | Recorded for completeness; not a gap. |
| Fisheye / Kannala-Brandt | PRESENT and comparable | WS: `crates/core/src/geometry/distortion.rs:433` `FisheyeDistortion`, `apply` `:454`, `remove` `:471`; f32 `:523` | REF: `crates/vision/src/distortion.rs:59` `from_euroc_coefficients`; `crates/core/src/types/camera.rs:24` `OpenCvFisheye` | |
| Fisheye image undistortion + rectify maps | PRESENT and comparable | WS: `crates/calib3d/src/distortion.rs:66` `init_undistort_rectify_map`, `:141` `fisheye_init_undistort_rectify_map`, `:196` `undistort_image_ex`, `:356` `fisheye_undistort_image` | REF: `crates/vision/src/distortion.rs:134` `undistort_pixel` (point-level; visloc-rs's `io` crate is where image IO lives) | |
| Stereo rectification (pinhole + fisheye) | PRESENT and comparable | WS: `crates/calib3d/src/stereo.rs`, `rectification` in `crates/calib3d/src/stereo_matching/rectification.rs`; fisheye stereo path `crates/calib3d/src/stereo.rs:388` | REF: `crates/io/src/calibration.rs:24` `KittiProjection`, `stereo_baseline_from` `:65`, `to_pinhole_camera` `:76`; no rectification module in `crates/vision` (grep `rectif` in `crates/vision` → doc references only) | **Workspace ahead here.** |
| KITTI projection calibration parsing | ABSENT | WS: `crates/io/src/datasets/kitti.rs` is 223 lines with only `read_poses` (`:37`) and `read_times` (`:121`) — no calibration parsing | REF: `crates/io/src/calibration.rs:109` `read_kitti_calibration_txt`, `:126` `parse_kitti_calibration_txt`, `:171` `kitti_projection_to_pinhole_camera`, 9 in-file tests | |
| Zhang / planar calibration from chessboard | PRESENT and comparable | WS: `crates/calib3d/src/calibration.rs:76` `calibrate_camera_planar`, `:94` `_with_options`, plus `crates/calib3d/tests/calibrate_planar_hang.rs` | REF: visloc-rs has **no calibration-from-observations routine** (no `chessboard`/`zhang`/`calibrate` symbol in `crates/`); it consumes calibrations from datasets | **Workspace ahead.** |
| Chessboard corner detection | PRESENT and comparable | WS: `crates/calib3d/src/chessboard.rs:42` `find_chessboard_corners_robust`; also `crates/features/src/charuco.rs`, `crates/features/src/aruco.rs`, `crates/features/src/pattern.rs` | REF: no counterpart — the reference's marker-free front-end starts from extracted features (`crates/vision/src/features/mod.rs:18` `FeatureExtractor`); the nearest reference is its query-feature loader `crates/io/src/query_features.rs:21` | Workspace ahead: CharUco/AprilTag/pattern detection has no equivalent on the reference side. |
| Calibration refinement (iterative LM) | PRESENT and comparable | WS: `crates/calib3d/src/calibration.rs:356`/`376` `refine_camera_calibration_iterative(_with_options)` | REF: no counterpart — the reference refines *poses*, not intrinsics (`crates/vision/src/pnp/mod.rs:211` `GaussNewtonPoseRefiner` refines pose only) | Workspace ahead. |
| Point-to-plane triangulation of a multi-view track | PRESENT and comparable | WS: `crates/calib3d/src/triangulation.rs:18` `triangulate_points`, `:389` `Triangulator` | REF: `crates/vision/src/stereo.rs:29` `triangulate_stereo_pixel`; multi-view via `pipelines/slam/src/colmap_incremental/incremental_triangulator.rs:1633` lines | |

## 7. Dense stereo / structured light / RGB-D

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Dense rectified-stereo disparity (block matching + LR check + uniqueness) | PRESENT and comparable | WS: `crates/calib3d/src/stereo_matching/block_matching.rs:14` `BlockMatcher`, `:416` `stereo_block_match`, metrics `with_metric` `:131` | REF: `crates/vision/src/dense_stereo.rs:133` `dense_disparity_map`, config `:20` `DenseStereoConfig` with `uniqueness_ratio` `:35` and `lr_consistency_px` `:38`, 6 in-file tests in `stereo.rs` | Both do SAD block matching on a rectified row with a left-right check. |
| Semi-global (SGM) matching | PRESENT and comparable | WS: `crates/calib3d/src/stereo_matching/sgm.rs:19` `SgmMatcher`, `:377` `stereo_sgm`, `with_penalties` `:145` | REF: none — grep `sgm\|semi.global` over `crates/` → **0** | **Workspace ahead**: visloc-rs's dense stereo is block-matching only. |
| Dense stereo → back-projected metric point cloud | PRESENT and comparable | WS: `crates/calib3d/src/stereo_matching/depth.rs`; `crates/calib3d/src/stereo_matching/gpu.rs` | REF: `crates/vision/src/dense_stereo.rs:218` `dense_stereo_points`, `DenseStereoPoint` `:54` | |
| Stereo depth gating / adaptive gate for VO | ABSENT | WS: grep `depth_gate\|StereoDepthGate` over `crates/` → **0** | REF: `crates/vision/src/stereo_vo.rs:56` `StereoDepthGate`, `adaptive()` `:66`, config `:79`, diagnostics `:141`, 37 in-file tests | |
| Sparse stereo triangulation of feature matches | PRESENT and comparable | WS: `crates/calib3d/src/stereo.rs`; `crates/sfm/src/triangulation.rs:18` | REF: `crates/vision/src/stereo.rs:29` `triangulate_stereo_pixel`; `crates/vision/src/stereo_vo.rs:218` `triangulate_stereo_features`, `:315` `triangulate_stereo_feature_matches` | |
| Stereo VO front-end (end-to-end) | ABSENT | WS: no stereo-VO driver; `crates/slam/src/tracking.rs:40` `process_frame` is monocular LK-based | REF: `crates/vision/src/stereo_vo.rs:4531` lines, `StereoVoFrontend` `:2004`, config `:1692`; `crates/vision/src/stereo_bootstrap.rs:771` lines `bootstrap_stereo_landmarks` | visloc-rs's largest single `vision` file. |
| **TSDF fusion (KinectFusion-style)** | PRESENT and comparable | WS: `crates/3d/src/tsdf/mod.rs:73` `TsdfVolume::new`, `:89` `integrate`, `:121` `integrate_ctx`, `:325` `extract_mesh`, `:380` `extract_point_cloud`; GPU TSDF `crates/hal/src/gpu_kernels/tsdf.rs`; 17 in-file/integration tests incl. `crates/3d/tests/tsdf_mesh_correctness.rs` | REF: `crates/gsplat-train/src/mesh.rs:47` `Tsdf`, `:76` `new`, `:100` `integrate`, `:198` `extract`, `:346` `keep_supported`, `:378` `remove_small_components` | Both have TSDF. The workspace's additionally has a GPU path; visloc-rs's is CPU inside the gsplat trainer. |
| **RGB-D odometry** | PRESENT and comparable | WS: `crates/3d/src/odometry/mod.rs:30` `compute_rgbd_odometry`, `:56` `compute_rgbd_odometry_ctx`; tests `crates/3d/tests/odometry_degenerate_frames.rs` | REF: none as a per-frame primitive — visloc-rs's RGB-D path is via `crates/vision/src/stereo_vo.rs:2004` `StereoVoFrontend` | |
| Depth-image → point cloud / unprojection | PRESENT and comparable | WS: `crates/core/src/geometry/camera.rs:207` `CameraIntrinsics::unproject`; `crates/pointcloud/src/point_cloud.rs` | REF: `crates/core/src/types/camera.rs:343` `unit_ray_from_pixel` + `crates/vision/src/dense_stereo.rs:218` | |
| **Projector-pattern / structured-light phase/shift decoding** | ABSENT (both) | WS: grep `structured_light\|StructuredLight\|projector\|phase_shift` over `crates/` → **0** | REF: grep same over `crates/` → **0** | Neither has it. Recorded so it is not mistaken for a gap. |

## 8. Optical flow

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Pyramidal Lucas-Kanade point tracking | PRESENT and comparable | WS: `crates/video/src/optical_flow.rs:16` `LucasKanade`, `track_point` `:53`, `track_points` `:262`, `track_keypoints` `:275`, `calc_optical_flow_lk` `:837`; pyramid levels `:44`; tests `crates/video/tests/lk_pyramid_units.rs` | REF: `crates/vision/src/optical_flow/tracker.rs:126` `OpticalFlowTracker`, `:80` `track_points_between`, `:153` `process`; pyramid `crates/vision/src/optical_flow/pyramid.rs` `build_pyramid` re-exported at `mod.rs:19` | Both are pyramidal LK over a feature set. |
| LK single-pass variant | PRESENT and comparable | WS: `crates/video/src/optical_flow.rs:210` `track_point_single_pass` | REF: `crates/vision/src/optical_flow/tracker.rs:80` `track_points_between` (implicitly pyramidal); `crates/vision/src/optical_flow/fast.rs` for the fast path | |
| Dense flow (Farnebäck) | PRESENT and comparable | WS: `crates/video/src/optical_flow.rs:371` `Farneback`, `compute` `:469`, `calc_optical_flow_farneback` `:847` | REF: no Farnebäck — grep `-i farneback\|dense_flow` over `crates/` → **0**; the reference is sparse LK + LSSD (`crates/vision/src/optical_flow/lssd.rs`) | **Workspace ahead**: visloc-rs has no dense flow field. |
| LSSD / patch-based tracking | ABSENT | WS: grep `lssd` over `crates/` → **0** | REF: `crates/vision/src/optical_flow/lssd.rs:13` `LssdPatch`, `:25` `from_image`, `:113` `track_step` | |
| Basalt-config VIO flow front-end (`BasaltOpticalFlowConfig`) | ABSENT | WS: no Basalt port; grep `basalt` → **0** | REF: `crates/vision/src/optical_flow/config.rs`, re-exported `crates/vision/src/optical_flow/mod.rs:14` `BasaltOpticalFlowConfig`, `:15` `BasaltVioConfig`; `pipelines/basalt/` (69k+ lines) | |
| DPVO (deep patch VO) patch/correlation stack | ABSENT | WS: grep `dpvo\|patch_vo\|correlation_volume` over `crates/` → **0** | REF: `crates/vision/src/dpvo/correlation.rs:753` lines, `softagg.rs:414`, `patchify.rs`, `npz.rs:707`, `onnx_session.rs:558`; `crates/dpvo-cuda-runtime/src/lib.rs:646`; `pipelines/slam/src/dpvo_vo.rs:7276` lines | |
| GPU optical-flow kernels | PRESENT and comparable | WS: `crates/hal/src/gpu_kernels/optical_flow.rs` (WGSL LK/flow) | REF: `crates/dpvo-cuda-runtime/src/lib.rs` (CUDA correlation volume for DPVO) | Different algorithms, both GPU. |

## 9. Bundle adjustment

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| BA problem representation (cameras + landmarks + observations) | PRESENT and comparable | WS: `crates/sfm/src/bundle_adjustment.rs:9` `Landmark`, `:30` `SfMState`, `add_landmark` `:51`, `to_parameters` `:105` / `from_parameters` `:131` | REF: `pipelines/slam/src/bundle.rs:1266` `BundleAdjustment`; `pipelines/slam/src/colmap_incremental/bundle_adjustment.rs:151` `BundleAdjustmentConfig`, `:286` `BundleAdjustmentOptions` | |
| Sparse vs dense | **WEAKER** | WS: **sparse Hessian + CG**, not dense. `crates/sfm/src/bundle_adjustment.rs:211` `numerical_jacobian_sparse` → `SparseMatrix`; `:594` `CgSolver` (200 iters, tol 1e-10); the dense path was measured at 35 s and replaced (comment at `:583-588`) | REF: **sparse with an explicit Schur complement** — `pipelines/slam/src/bundle.rs:1` "Bundle adjustment with Schur-complement landmark elimination"; `solve_step`'s per-landmark Schur reduction (`bundle.rs:37`); `pipelines/slam/src/block_cholesky.rs:2582` lines block-Cholesky; plus a **matrix-free PCG GPU path** in `crates/ba-gpu/src/solver.rs:552` `optimize` | **How reduced:** the workspace solves `JᵀJ + λD` with CG on the normal equations. visloc-rs eliminates the block-diagonal landmark Hessian analytically (Schur), then Cholesky-solves the camera block, and can run the whole LM loop on GPU with PCG. The workspace's numerics are correct but do not scale to large problems the same way. |
| Schur complement / marginalisation path | ABSENT | WS: grep `-i schur` over `crates/`, `src/` → only `crates/math/src/linalg.rs:343-352`, which uses nalgebra's `schur()` for **complex eigendecomposition**, unrelated to BA. grep `-i marginali` → **0**. | REF: `pipelines/slam/src/bundle.rs:1` and `:37` (Schur landmark elimination); `pipelines/slam/src/marginalization.rs:194` lines; `pipelines/slam/src/marginalization_sqrt.rs:532` lines (sqrt-marginalisation); `pipelines/slam/src/vi_sqrt_window.rs` | **This is the single largest algorithmic gap in the comparison.** See "most consequential" below. |
| Robust kernel | PRESENT and comparable | WS: `crates/sfm/src/bundle_adjustment.rs:536` `robust_kernel: bool` — a **boolean**, not a kernel choice. Forces the sequential path (`:556`: `if !config.use_sparsity || config.robust_kernel` → sequential). Applied every 5th iteration as a plain threshold (`:668` `iteration % 5 == 0`), plus `remove_outliers` `:481`. The generic 6-kernel library exists at `crates/core/src/robust_loss.rs:9` `RobustLoss` (GemanMcClure/Welsch/Huber/TLS/Cauchy/Tukey with `cost` `:29` and influence `:66`) but BA does not use it. | REF: `pipelines/slam/src/pose_graph.rs:87` `pub enum RobustKernel { None, Huber{delta}, Cauchy{c} }`, `cost()` `:101`, influence weight `:118`; threaded through to GPU at `crates/ba-gpu/src/solver.rs:558-562`, where `check()` (`:212`) **rejects** anything other than `None`/`Huber` (`:218-221`) | **Verdict note:** the kernel *menu* is comparable (Huber + Cauchy both exist on each side). **How reduced:** the BA wiring is a bool + a periodic hard threshold, so BA's effective robustification is Cauchy-free, non-selectable, and mutually exclusive with the sparse path — whereas visloc-rs selects the kernel and can run it in the GPU path. |
| Robust kernel: GNC-TLS global registration | PRESENT and comparable | WS: `crates/registration/src/registration/gnc.rs:13` `GNCOptimizer`, `new_geman_mcclure` `:29`, `new_tls` `:43`, `new_welsch` `:55`, `solve_registration` `:69`, `registration_gnc` `:356`; tests `crates/registration/tests/gnc_convergence_and_inlier_gate.rs` | REF: `pipelines/slam/src/gnc.rs` | Comparable. |
| Local BA windowing | ABSENT | WS: no `Local`/`Global` scope — `bundle_adjust` (`crates/sfm/src/bundle_adjustment.rs:551`) always solves the whole problem | REF: `crates/ba-gpu/src/solver.rs:26` `BaScope::Local` vs `Global`; `pipelines/slam/src/local_submap.rs`, `covisibility_ba.rs:1790` lines, `stereo_vo_ba.rs:2733` lines | |
| Covisibility-graph BA | ABSENT | WS: 0 (`covisibil` grep → 0) | REF: `pipelines/slam/src/covisibility_ba.rs:1790` lines | |
| BA Jacobian — analytic vs numeric | WEAKER | WS: `crates/sfm/src/bundle_adjustment.rs:182` `numerical_jacobian` — **finite differences** | REF: analytic Jacobians (e.g. `pipelines/slam/src/finite_difference.rs` is a *verification* module for the analytic path, not the production Jacobian) | The workspace computes Jacobians by finite differences; that costs accuracy and speed. |
| BA test coverage | WEAKER | WS: 20 in-file tests in `crates/sfm/src/bundle_adjustment.rs` | REF: **58** tests in `pipelines/slam/tests/bundle_adjustment.rs` | 58 vs 20 on the same named capability. |

## 10. Pose-graph / SLAM backends

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Pose-graph optimisation (nodes, edges, information matrix, optimise) | PRESENT and comparable | WS: `crates/slam/src/pose_graph.rs:4` `PoseGraph`, `add_node` `:32`, `add_edge` `:36`, `set_fixed` `:51`, `optimize` `:55` | REF: `pipelines/slam/src/pose_graph.rs` (2,638 lines) | |
| Pose-graph robust kernel | PRESENT and comparable | WS: `crates/core/src/robust_loss.rs:9` `RobustLoss` (6 kernels) — used by the *optimiser* (`crates/optimize/`) rather than by `crates/slam/src/pose_graph.rs` | REF: `pipelines/slam/src/pose_graph.rs:87` `RobustKernel` (None/Huber/Cauchy), applied per edge | Comparable capability; different wiring (below). |
| **Incremental pose graph / pose-graph surgery** | ABSENT | WS: no `incremental_pose_graph` analogue; `crates/slam/src/pose_graph.rs` is a single 460-line file | REF: `pipelines/slam/src/incremental_pose_graph.rs` | |
| **SIM3 pose graph** (scale-correcting) | ABSENT | WS: SIM(3) exists only as a **trajectory-evaluation alignment** (`crates/eval/src/trajectory.rs:37` `Alignment::Sim3`, `:304` `umeyama(..., true)`) and a CLI flag (`crates/cli/src/args.rs:57`). There is no SIM(3) pose-graph optimiser — grep `sim3` over `crates/` hits only `crates/cli/`, `crates/eval/`. | REF: `pipelines/slam/src/sim3_pose_graph.rs`, `pipelines/slam/src/dpvo_sim3_backend.rs:1599` lines | Evaluation is not estimation; the scale-correcting *graph* is the missing part. |
| Factor-graph / iSAM2-style incremental smoothing | PRESENT and comparable | WS: `crates/optimize/src/factor_graph.rs`, `crates/optimize/src/isam2.rs:66` `Isam2Solver` (`update` `:91`, `estimate_pose3` `:203`), `crates/optimize/src/factors.rs` | REF: `pipelines/slam/src/sparse_factor_graph.rs`; incremental BA `pipelines/slam/src/online_slam_vi_ba.rs:4052` lines | Comparable *shape*; the reference's is 4k+ lines against the workspace's ~1k. |
| **Ordered view graph** | ABSENT | WS: grep `view_graph\|ordered_view` over `crates/` → **0** | REF: `pipelines/slam/src/ordered_view_graph.rs` | |
| Loop gating / geometric consistency matrix (PCM) | ABSENT | WS: has a *single* consensus gate (`crates/sfm/src/mapper.rs:2190`, median residual at `:2248`); the covisibility graph is a doc-level concept only (`crates/sfm/src/mapper.rs:15`, `:162`, `:484` — no `covisibil` type exists). grep `pcm`, `loop_gating` over `crates/` → **0** | REF: `pipelines/slam/src/pcm.rs`; `pipelines/slam/src/loop_gating.rs`; cross-session `consistent_session_bridges` (see `crates/vision/src/place_recognition/mod.rs:9-10`) | |
| Sim(3) submap / multi-session map merge | ABSENT | WS: grep `submap\|atlas` over `crates/slam/`, `crates/sfm/` → **0** | REF: `pipelines/slam/src/map_atlas.rs:2462` lines, `local_submap.rs`, `submap_alignment.rs`, `submap_overlap.rs`, `submap_partition.rs` | |
| Online SLAM driver (push frame → track → map → loop) | WEAKER | WS: `crates/slam/src/lib.rs:34` `Slam::process_image` — a 50-line facade over `crates/slam/src/tracking.rs:40` `process_frame` (513-line tracking, 175-line mapping). 31 in-crate tests. | REF: `pipelines/slam/src/online_slam.rs:7707` lines, `OnlineSlam` `:2775`, `process_frame` `:3010`, `scan_appearance_loops` `:2941`, `reset_sequence_state` `:5620`; **134** tests in `pipelines/slam/tests/online_slam.rs` | **How reduced:** 50 lines vs 7,707; 31 tests vs 134. The workspace's `Slam` has no loop-closure entry point, no session reset, no metric-scale path. |
| Feature VO | WEAKER | WS: `crates/slam/src/tracking.rs` uses LK (`crates/video/src/optical_flow.rs:16`) + ORB matching | REF: `pipelines/tracking/src/tracker.rs:2595` lines, `pipelines/tracking/src/trajectory.rs:2715` lines; `crates/vision/src/stereo_vo.rs:2004` `StereoVoFrontend` | |
| Calibration-time triangulation / global SfM | WEAKER | WS: `crates/sfm/src/mapper.rs:558` `map_views` (incremental, sequential+retrieval pairs), `crates/sfm/examples/tum_sfm.rs`; 41 in-crate tests | REF: `pipelines/slam/src/incremental_sfm.rs:18506` lines, `global_sfm.rs:5831` lines, `hierarchical_sfm.rs:2799` lines | |

## 11. Loop-closure integration

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Loop detection (retrieval → candidate pairs) | PRESENT and comparable | WS: `crates/sfm/src/mapper.rs:153` `SequentialWithRetrieval`, `:207` `loop_closure: bool`, `:210` `loop_closure_min_gap: 10` (default `:337`) | REF: `pipelines/slam/src/loop_closure.rs`, `pipelines/slam/src/vo_loop_closure.rs:1944` lines, `pipelines/slam/src/hierarchical_loop_closure.rs`, `pipelines/slam/src/dpvo_loop_closure.rs` | |
| Loop verification (essential / PnP / hybrid) | PRESENT and comparable | WS: `crates/sfm/src/mapper.rs:2190-2248` — loop accepted on median residual ≤ `loop_closure_max_px` (default 2.0 px, `:343`) with a GNC gate (`:213`) | REF: `pipelines/slam/src/loop_closure.rs:157` `EssentialMatrixLoopClosureVerifier`, `:314` `PnPLoopClosureVerifier`, `:760` `HybridLoopClosureVerifier`, `:727` `verify_loop_closure_candidates_hybrid` | |
| Loop closure → constraint → pose-graph edge | PRESENT and comparable | WS: `crates/sfm/src/mapper.rs:889` (`loop_closures += 1`) then feeds the pose graph | REF: `pipelines/slam/src/loop_closure.rs:551` `LoopClosureConstraint`, `:594` `to_pairwise_pose_factor`, `:609` `loop_closure_constraints_from_candidates`, `:624` `pairwise_pose_factors_from_loop_closures` | |
| 2D-3D correspondence construction for a loop candidate | ABSENT | WS: no `correspondences_for_loop_candidate` analogue; the mapper derives them internally from tracks | REF: `pipelines/slam/src/loop_closure.rs:471` `correspondences_2d3d_for_loop_candidate`, `:510` `correspondences_for_loop_candidate` | |
| Pairwise all-pairs loop scan | ABSENT | WS: no scanner; pairs come only from the mapper's `PairSelection` | REF: `pipelines/slam/src/loop_closure.rs:883` `scan_pairwise_loop_closures`, config `:848` | |

## 12. IMU / visual-inertial fusion

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| IMU preintegration (residuals + Jacobians) | ABSENT | WS: grep `preintegration` over `crates/`, `src/` → **0 hits** | REF: `pipelines/basalt/src/imu/preintegration.rs`; `pipelines/basalt/src/imu/factors.rs:208` `preintegration_residual_and_jacobian`, `:1515` `preintegration_residual`, `:235` `preintegration_jacobian_f32` (3,606 lines) | |
| IMU factor (whitened, 9-DOF) | ABSENT | WS: 0 | REF: `pipelines/basalt/src/imu/factors.rs:58` `WhitenedImuFactor`, `:95` `whitened_preintegration_factor`, `:1548` `sqrt_information`, `:154` `ImuFactorError` | |
| IMU bias random-walk factor | ABSENT | WS: 0 — grep `imu_bias` → **0** | REF: `pipelines/basalt/src/imu/factors.rs:73` `BiasRandomWalkNoise`, `:88` `WhitenedBiasWalkFactor`, `:1394` `whitened_bias_random_walk_factor` | |
| IMU bias initialisation / scale-gyro | ABSENT | WS: 0 | REF: `pipelines/basalt/src/initialization.rs`; `pipelines/slam/src/vi_initializer.rs` | |
| VI frontend (visual + inertial, Basalt port) | ABSENT | WS: grep `visual_inertial`, `vins`, `basalt` → **0 hits each** | REF: `pipelines/basalt/src/vio/aom.rs:21486` lines, `window.rs:14564`, `estimator.rs:7228`; `pipelines/basalt/` totals >50k lines | visloc-rs's largest subsystem by far. |
| IMU measurement ingestion into the SLAM driver | ABSENT | WS: `Slam::process_image` (`crates/slam/src/lib.rs:34`) takes only an image | REF: `pipelines/slam/src/online_slam.rs:2846` `push_imu_measurement`, `:2872` `push_vi_initialization_measurement`, `:2898` `take_pending_imu_factor` | |
| KITTI-OxTS IMU/pose log parsing | ABSENT | WS: `crates/io/src/datasets/kitti.rs` (223 lines) has `read_poses`/`read_times` only — no OxTS | REF: `crates/io/src/kitti_imu.rs:99` `read_kitti_oxts_dir`, `:148` `parse_kitti_oxts_sample`, `:214` `parse_kitti_oxts_timestamps_txt`; 10 tests in `crates/io/tests/kitti_imu.rs` | |
| EuRoC IMU CSV + sensor YAML parsing | ABSENT | WS: `crates/io/src/datasets/euroc.rs` (392 lines) has `read_groundtruth` `:127` and `read_image_index` `:197` only — no IMU, no sensor YAML | REF: `crates/io/src/euroc.rs:127` `read_euroc_imu_csv`, `:289` `read_euroc_camera_sensor_yaml`, `:338` `read_euroc_imu_sensor_yaml`, `:381` `read_euroc_dataset_dir` | |
| Kalman filter (single-sensor smoothing) | PRESENT and comparable | WS: `crates/video/src/kalman.rs`; re-exported `crates/slam/src/lib.rs:7-10` (`KalmanFilter`, `ExtendedKalmanFilter`, `DynamicKalmanFilter`); test `crates/video/tests/kalman_singular_innovation.rs` | REF: `pipelines/basalt/src/stream.rs` uses a fixed-size state stream, not an EKF surface | Workspace ahead on the generic filter API; it is not a substitute for preintegration. |

## 13. 3D Gaussian splatting

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| Gaussian primitives (position, scale, rotation, opacity, SH) | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/types.rs:168` `Gaussian`, `covariance` `:263`, `SphericalHarmonics` `:57` (`eval` `:101`), `GaussianCloud` `:433` | REF: `crates/gsplat-core/src/gaussian.rs`, `crates/gsplat-core/src/splat.rs` (6 in-file tests), `crates/gsplat-core/src/sh.rs` (5 tests) | |
| Forward rasterisation (tiled, tile bounds, alpha blending) | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/rasterize.rs:217` `GaussianRasterizer`, `:238` `project_gaussians`, `:254` `compute_tile_bounds`, `:344` `rasterize`; cull test `crates/rendering/tests/tile_cull_matches_alpha_cutoff.rs` | REF: `crates/gsplat-render/src/renderer.rs:473` `Renderer`, `:871` `render`, `:1109` `render_depth`; `renderer_geo.rs`, `packing.rs`, `uniforms.rs`; 13 in-file tests in `gsplat-render/src/lib.rs` | Both tile-based. |
| CPU forward render | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/rasterize.rs:344` (CPU path is the only path) | REF: `crates/gsplat-core/src/cpu_render.rs:83` `render`, `:441` `render_f64` | |
| **GPU forward render** | ABSENT | WS: grep `wgpu\|ComputeDevice\|ResourceGroup` in `crates/rendering/src/gaussian_splatting/` → **0 hits**. `crates/rendering/src/lib.rs:9` exposes only `gaussian_splatting`; there is no gsplat kernel under `crates/hal/src/gpu_kernels/` (the 41 kernel files there include `sparse.rs`, `radix_sort.rs` but nothing Gaussian). | REF: `crates/gsplat-render/src/gpu.rs`, `renderer.rs:54` `GpuScene::upload`, `:91` `Renderer::from_gpu_scene`; `crates/gsplat-render/src/sort.rs:79` `RadixSorter` over `wgpu::Device` (`:91`), `renderer_backward.rs:389` lines | **Structural, not a wording issue:** the workspace's splatting is CPU-only; visloc-rs's is a wgpu pipeline with a radix sort and a GPU backward pass. |
| Backward pass / analytic gradients | WEAKER | WS: `crates/rendering/src/gaussian_splatting/rasterize.rs:515` `backward` returns a `GaussianGradient` (`:540`); `crates/rendering/src/gaussian_splatting/differentiable.rs` is a **3-line re-export shim** | REF: `crates/gsplat-core/src/backward.rs:837` lines, `:561` `render_backward_screen`, `:587` `render_backward`; `crates/gsplat-render/src/renderer_backward.rs:389` lines | **How reduced:** the workspace has one `backward` entry point; the reference has separate screen-space and scene-space backward passes plus a GPU backward. |
| Densification | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/optimize.rs:8` `DensificationConfig`, `:181` `GaussianOptimizer::densify`, `:35` `PruningConfig`, `:246` `prune`, `:54` `OpacityResetConfig`, `:253` `reset_opacity` | REF: `crates/gsplat-train/src/densify.rs:23` `DensifyConfig`, `:140` `densify`, `:110` `DensifyReport`, `:219` `reset_opacity`, `:303` `BrushRefineConfig` | |
| Training loop | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/optimize.rs:266` `GaussianTrainer`, `:279` `train_step`, `:293` `train` | REF: `crates/gsplat-train/src/trainer.rs:240` `Trainer`, `:483` `new`, `:796` `step`, `:756` `take_profile`, `:776` `take_densify_report` (1,533 lines) | |
| PLY I/O for splats | PRESENT and comparable | WS: `crates/rendering/src/gaussian_splatting/io.rs:9` `read_ply_gaussian_cloud`, `:113` `write_ply_gaussian_cloud`, `:188` `to_ply_string`; round-trip test `crates/rendering/tests/ply_opacity_roundtrip.rs` | REF: `crates/gsplat-core/src/ply.rs:335` lines | |
| Load a COLMAP reconstruction as a splat scene | ABSENT | WS: no `load_colmap_scene`; `crates/io/src/datasets/colmap.rs` reads COLMAP text into its own types (`:117`, `:262`, `:369`) with no splat conversion | REF: `crates/gsplat-core/src/colmap_scene.rs:55` `load_colmap_scene`, `:69` `scene_from_visual_map`, `:138` `camera_view_from`, `:173` `pinhole_intrinsics`; writer `crates/io/src/colmap/mod.rs:525` `write_colmap_reconstruction_for_3dgs` | |
| Train-from-dataset driver (COLMAP / EuRoC / photos) | ABSENT | WS: `crates/examples/src/bin/gaussian_splatting_basic.rs` and `crates/examples/src/bin/gaussian_splatting_basic.rs` are demo binaries, not a dataset trainer; no `gsplat-train` equivalent | REF: `crates/gsplat-train/src/dataset.rs`, `:euroc.rs:1229` lines, `:photos.rs:594` lines; examples `gsplat_euroc.rs`, `gsplat_photos.rs`, `gsplat_train.rs`, `gsplat_eval.rs` | |
| Splat metrics (PSNR/SSIM/LPIPS) | ABSENT | WS: no PSNR/SSIM in `crates/rendering/` (grep `psnr\|ssim\|lpips` → 0) | REF: `crates/gsplat-train/src/metrics.rs` | |

## 14. Mesh extraction

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| TSDF → mesh extraction | PRESENT and comparable | WS: `crates/3d/src/tsdf/mod.rs:325` `extract_mesh`, `:380` `extract_point_cloud`; GPU `crates/hal/src/gpu_kernels/marching_cubes.rs` + `crates/3d/src/gpu/mesh.rs`; tests `crates/3d/tests/tsdf_mesh_correctness.rs`, `crates/hal/tests/mesh_gpu_tests.rs` | REF: `crates/gsplat-train/src/mesh.rs:198` `Mesh::extract` | |
| Mesh post-processing (support filter, small-component removal) | PRESENT and comparable | WS: `crates/3d/src/mesh/processing.rs`; `crates/3d/src/mesh/mod.rs:90` `compute_vertex_normals`, `:160` `sample_points` | REF: `crates/gsplat-train/src/mesh.rs:346` `keep_supported`, `:378` `remove_small_components`, `:416` `write_ply` | |
| **Poisson surface reconstruction** | PRESENT and comparable | WS: `crates/3d/src/mesh/reconstruction/poisson.rs:25` `poisson_reconstruction` | REF: no Poisson — grep `-i poisson` over `crates/` → **0** | **Workspace ahead.** |
| **Ball-pivoting surface reconstruction** | PRESENT and comparable | WS: `crates/3d/src/mesh/reconstruction/ball_pivoting.rs` | REF: no ball pivoting — grep `-i 'ball_pivot\|pivoting'` over `crates/` → **0** | **Workspace ahead.** |
| Alpha shapes | PRESENT and comparable | WS: `crates/3d/src/mesh/reconstruction/alpha_shapes.rs` | REF: no alpha shapes — grep `-i alpha_shape` over `crates/` → **0** | **Workspace ahead.** |
| Mesh extraction from splats (depth render → TSDF → mesh) | ABSENT | WS: no `extract_from_splat`; splats and meshes are separate crates with no bridge (`crates/rendering` vs `crates/3d`) | REF: `crates/gsplat-train/src/mesh.rs:494` `extract_from_splat`, `:40` `DepthFrame`, `:455` `MeshOptions`, `:483` `rig_scale` | |
| Mesh file I/O (OBJ/STL/PLY/PCD/glTF) | PRESENT and comparable | WS: `crates/io/src/obj.rs`, `crates/io/src/stl.rs`, `crates/io/src/ply.rs`, `crates/io/src/pcd.rs`, `crates/io/src/gltf_io.rs`, `crates/io/src/mesh.rs`; 10 robustness test files | REF: `crates/gsplat-core/src/ply.rs:335` lines (PLY only, for splats); `crates/gsplat-train/src/mesh.rs:416` `write_ply` | **Workspace ahead** on formats (6 vs 1). |

## 15. Global point-cloud registration (workspace-only area)

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| FPFH features | PRESENT and comparable | WS: `crates/registration/src/registration/global/fpfh.rs:32` `compute_fpfh_features` — full Rusu pipeline (kNN-PCA normals → voxel index → SPFH → FPFH), 33-bin; degenerate-input test `crates/registration/tests/fpfh_degenerate.rs` | REF: no FPFH — grep `fpfh\|compute_fpfh_feature\|point_feature_histogram` over `crates/` and `pipelines/` → **0** | **Workspace ahead.** (Recorded here because an earlier audit wrongly reported FPFH absent by grepping `fn fpfh` instead of `compute_fpfh_features`; that function exists at `fpfh.rs:32`.) |
| FPFH-based RANSAC global registration | PRESENT and comparable | WS: `crates/registration/src/registration/global/ransac.rs:65` `registration_ransac_based_on_feature_matching`; test `crates/registration/tests/ransac_evaluate_reports_rmse.rs` | REF: 0 | Workspace ahead. |
| Fast global registration (FGR) | PRESENT and comparable | WS: `crates/registration/src/registration/global/ransac.rs:159` `registration_fgr_based_on_feature_matching` | REF: 0 | Workspace ahead. |
| ICP / multi-scale ICP | PRESENT and comparable | WS: `crates/registration/src/registration/mod.rs:128` `registration_icp_point_to_plane`, `:585` `registration_multi_scale_icp`, `:684` `get_information_matrix_from_point_clouds`, `:747` `evaluate_registration`; GPU `crates/hal/src/gpu_kernels/icp.rs` | REF: 0 — visloc-rs is a *pose-estimation* project, not a point-cloud-alignment project | Workspace ahead. |
| Coloured ICP | PRESENT and comparable | WS: `crates/registration/src/registration/colored.rs` | REF: 0 | Workspace ahead. |


| Splat metrics (PSNR/SSIM/LPIPS) | ABSENT | WS: no metrics module for splats. The only PSNR in the tree is an ad-hoc MSE→dB computation inside a denoising **test** (`crates/photo/src/denoise.rs:998`); grep `psnr\|ssim\|lpips` over `crates/` hits only `crates/photo/src/denoise.rs` and `crates/hal/tests/helpers/mod.rs`. No SSIM implementation, no LPIPS. | REF: `crates/gsplat-train/src/metrics.rs:10` `psnr`, `:126` `ssim`, `:78` `ssim_with_grad`; consumed in `crates/gsplat-train/src/loss.rs` and `crates/gsplat-train/src/trainer.rs` | |

### Correction to the row above (read before citing)

The train-from-dataset row's workspace pointer should be
`crates/examples/src/gaussian_splatting_basic.rs` (not `.../bin/...`). It is a demo
binary that constructs a synthetic cloud and rasterises it; there is no dataset
loader feeding the trainer. That is the whole of the claim.

## 16. Dataset and file I/O

| Capability | Verdict | Workspace pointer | visloc-rs pointer | Notes |
|---|---|---|---|---|
| **COLMAP sparse model — text** | PRESENT and comparable | WS: `crates/io/src/datasets/colmap.rs:117` `read_cameras_text`, `:262` `read_images_text`, `:369` `read_points3d_text`, and writers `:453`/`:486`/`:523` | REF: `crates/io/src/colmap/mod.rs:148` `read_colmap_text_model`, `:791` `format_cameras_txt`, `:811` `format_images_txt`, `:863` `format_points3d_txt`, `:889`/`:921`/`:942` parsers | |
| **COLMAP sparse model — binary** | ABSENT | WS: `grep -ci binary` in `crates/io/src/datasets/colmap.rs` → **0**. All six public functions are `_text`. | REF: `crates/io/src/colmap/mod.rs:173` `read_colmap_binary_model`, `:1012` `parse_cameras_bin`, `:1050` `parse_images_bin`, `:1107` `parse_points3d_bin`; provider `:75` `from_binary_model_dir`, `:93` `_validated` | A COLMAP model produced by any real COLMAP run is often binary; the workspace cannot read it. |
| COLMAP 3DGS-export writing (cameras/points for splat training) | ABSENT | WS: `crates/io/src/datasets/colmap.rs` writes its own schema only; no `_for_3dgs` variant | REF: `crates/io/src/colmap/mod.rs:231` `write_colmap_text_model_for_3dgs`, `:372` `_binary_model_for_3dgs`, `:525` `write_colmap_reconstruction_for_3dgs`, `:647` `_with_cameras` | |
| COLMAP map validation for localisation | ABSENT | WS: no `validate_map`; no map-validation entry point in `crates/io/src/datasets/colmap.rs` | REF: `crates/io/src/colmap/mod.rs:126` `validate_map`, `:130` `validate_for_localization` | |
| **KITTI — poses + timestamps** | PRESENT and comparable | WS: `crates/io/src/datasets/kitti.rs:37` `read_poses`, `:121` `read_times` | REF: `crates/io/src/kitti.rs:35` `read_kitti_image_sequence_dir`, `:45` `_with_timestamp_file` | |
| KITTI — calibration | ABSENT | WS: 0 (see §6) | REF: `crates/io/src/calibration.rs:109`, `:126`, `:171` | |
| KITTI — OXTS / IMU | ABSENT | WS: 0 | REF: `crates/io/src/kitti_imu.rs:99`, `:148`, `:214` | |
| **EuRoC — ground truth + image index** | PRESENT and comparable | WS: `crates/io/src/datasets/euroc.rs:127` `read_groundtruth`, `:197` `read_image_index`, `:82` `read_csv` | REF: `crates/io/src/euroc.rs:186` `read_euroc_ground_truth_csv`, `:74` `read_euroc_image_manifest`, `:381` `read_euroc_dataset_dir` | |
| EuRoC — IMU + sensor calibration | ABSENT | WS: 0 | REF: `crates/io/src/euroc.rs:127`, `:289`, `:338` | |
| **TUM trajectory format** | PRESENT and comparable | WS: `crates/io/src/datasets/tum.rs:80` `read_index`, `:127` `read_groundtruth`, `:709` `associate`; tests `crates/io/tests/tum_associate_scaling.rs`, `crates/io/tests/dataset_pose_validation.rs` | REF: parsed in the tracking pipeline, not `io`: `pipelines/tracking/src/trajectory.rs:1054` `from_tum_poses_str` (`:1072`-`:1074` per-field parse), `:74` `to_tum_pose_record`; `crates/io/src/` has no TUM module (grep `-i '\btum\b'` over `crates/` → **0**) | Comparable in scope — a TUM **dataset** reader (image directory + RGB-D + association into the SLAM driver) exists only on the workspace side; the reference parses only the trajectory text. |
| GNSS / GPS sensor log | ABSENT | WS: 0 | REF: `crates/io/src/sensors.rs:20` `read_gnss_measurements_txt`, `:26` `parse_gnss_measurements_txt` | |
| Generic image sequence reading + timestamps | ABSENT | WS: no dataset module reads image sequences; `crates/videoio/` is video (ffmpeg/v4l2/gif/png-sequence) | REF: `crates/io/src/images.rs:220` `read_common_image_sequence`, `:230` `_with_timestamps`, `:247` `read_timestamp_nanoseconds_txt`, `:308` `_dir`; PGM `:115`, PNG `:194` | |
| Descriptor / feature text formats (query features, landmark descriptors, two-view matches) | ABSENT | WS: 0 — no reader for these; `crates/features/src/descriptor.rs` is in-memory only | REF: `crates/io/src/query_features.rs:21`, `crates/io/src/descriptors.rs:19`, `crates/io/src/two_view_matches.rs:70`/`76` | |
| External deep-feature IO | ABSENT | WS: 0 | REF: `crates/io/src/external_deep.rs` | |
| PLY / OBJ / STL / PCD / LAS / glTF mesh-cloud IO | PRESENT and comparable | WS: `crates/io/src/ply.rs`, `obj.rs`, `stl.rs`, `pcd.rs`, `las_io.rs`, `gltf_io.rs`; 10 test files in `crates/io/tests/` | REF: not in the `io` crate — `crates/gsplat-core/src/ply.rs:335` only | Workspace ahead: 6 formats vs 1. |


---

## Verdict tally

| Verdict | Rows |
|---|---|
| PRESENT and comparable | **79** |
| ABSENT | **57** |
| WEAKER | **10** |
| **Total rows** | **146** |

(The raw table has 178 `|`-leading lines; 32 of those are headers, the verdict legend,
and the scale-context table.)

## What visloc-rs has that the workspace lacks entirely

Roughly ordered by how hard it would be to add. "Hard" here means the work is not a
single self-contained module — it needs new theory, new dependencies, or a new
subsystem.

**Tier 1 — a subsystem, not a module (weeks-to-months of work each)**

1. **IMU preintegration + visual-inertial fusion.** `pipelines/basalt/src/imu/` plus
   `pipelines/basalt/src/vio/` (>50k lines; `aom.rs` alone is 21,486 lines). The
   workspace has zero IMU code — `grep preintegration`, `imu_bias`, `visual_inertial`,
   `basalt` all return 0. This is not a gap in an existing feature; it is an absent
   field of study for the workspace. `crates/video/src/kalman.rs` is a generic filter
   and does not substitute.
2. **Two-view geometry *verification* (COLMAP's `EstimateTwoViewGeometry`).**
   `crates/vision/src/two_view/colmap_verification.rs` (1,032 lines) classifies a pair
   into H/E/F × calibrated/uncalibrated, counts models, and reports a triangulation
   angle. The workspace has three independent solvers and no classifier. Every
   downstream decision in both SfM systems — is this a loop closure? is this a seed? is
   this pair planar? — depends on it.
3. **Nistér 5-point.** `crates/vision/src/two_view/five_point.rs` (1,257 lines). The
   workspace's `EssentialSolver::estimate_5point` (`crates/calib3d/src/essential.rs:14`)
   is an explicit `AlgorithmError` stub. Without it the workspace's essential RANSAC can
   only use an 8-point sample — larger, less robust inlier sets per draw, and no
   degenerate-config handling.
4. **Visual-inertial / deep-VO front-ends:** DPVO (`crates/vision/src/dpvo/`, ~2,400
   lines + a 646-line CUDA runtime), Basalt (`pipelines/basalt/`), and the stereo-VO
   driver `crates/vision/src/stereo_vo.rs` (4,531 lines — visloc-rs's largest `vision`
   file).
5. **Schur/marginalisation path in BA.** `pipelines/slam/src/bundle.rs` (17,047 lines),
   `marginalization.rs`, `marginalization_sqrt.rs`, `block_cholesky.rs` (2,582 lines),
   and the matrix-free PCG GPU solver `crates/ba-gpu/src/solver.rs`. The workspace's
   BA is CG on `JᵀJ + λD` with a finite-difference Jacobian.

**Tier 2 — a well-scoped module (days to a couple of weeks)**

6. **Generalized / non-central absolute pose:** `pnp/generalized.rs` (1,142 lines),
   `pnp/gp3p.rs` (1,396), `pnp/gr6p.rs` (2,072) + `gr6p_data.rs` (977). Zero hits for
   `gp3p|g3p|gr6p|generalized` in the workspace.
7. **VLAD + mean-pool global descriptors + mutual-NN retrieval.**
   `place_recognition/mod.rs:192`, `:244`, `:275`, `:306`. Zero hits for `vlad` in the
   workspace.
8. **Correspondences in the scene graph:** `two_view/correspondence_graph.rs`
   (1,160 lines, 14 tests) with transitive-track extraction.
9. **ONNX learned front-ends:** SuperPoint (803 lines), LightGlue (477 lines),
   ONNX global descriptor (301 lines) — plus their parity test suites. The workspace
   has the *runtime* (`crates/dnn/src/lib.rs:70` `DnnNet` on `tract-onnx`), so these are
   integration work, not new dependencies.
10. **COLMAP binary model I/O:** `colmap/mod.rs:173`, `:1012`, `:1050`, `:1107`.
    `grep -ci binary` in the workspace's `colmap.rs` returns 0.
11. **Homography decomposition → (R,t):** `two_view/homography.rs:270`, `:504`.
12. **KITTI calibration + OXTS/IMU and EuRoC IMU/sensor-YAML readers:**
    `io/src/calibration.rs`, `io/src/kitti_imu.rs`, `io/src/euroc.rs:127`/`:289`/`:338`.
13. **SLAM-atlas machinery:** submaps (`map_atlas.rs`, 2,462 lines; `local_submap.rs`;
    `submap_alignment.rs`), SIM(3) pose graphs, incremental pose-graph surgery, ordered
    view graph, PCM loop gating.
14. **Hamming-embedding retrieval scoring** (`vocab_tree/index.rs`) and hierarchical
    k-means (`vocab_tree/hkm.rs`).

## Where the workspace is ahead

This is a real answer, not a courtesy. On several axes the workspace is not merely
comparable but broader.

1. **Camera calibration, end to end.** The workspace does Zhang/planar calibration
   (`crates/calib3d/src/calibration.rs:76`, `:94`), chessboard corner detection
   (`crates/calib3d/src/chessboard.rs:42`), CharUco/AprilTag/pattern detection
   (`crates/features/src/charuco.rs`, `aruco.rs`, `pattern.rs`), iterative LM
   refinement (`:356`, `:376`), and stereo rectification including fisheye
   (`crates/calib3d/src/distortion.rs:141`, `crates/calib3d/src/stereo.rs:388`).
   visloc-rs has **none** of this — `grep -i 'zhang|chessboard|calibrate_camera'` over
   its `crates/` returns 0. It *consumes* calibrations from datasets; it never derives one.
2. **Point-cloud registration — an entire discipline the reference does not have.**
   FPFH (`crates/registration/src/registration/global/fpfh.rs:32`), FPFH-RANSAC
   (`global/ransac.rs:65`), FGR (`:159`), point-to-plane ICP
   (`registration/mod.rs:128`), multi-scale ICP (`:585`), coloured ICP
   (`registration/colored.rs`), GNC-TLS/Geman-McClure/Welsch
   (`registration/gnc.rs:29`-`:55`), GPU ICP (`crates/hal/src/gpu_kernels/icp.rs`).
   Zero FPFH/ICP hits anywhere in visloc-rs's 254k + 57k lines.
3. **RGB-D / TSDF and mesh reconstruction.** TSDF with GPU integration
   (`crates/3d/src/tsdf/mod.rs:89`, `:121`, `crates/hal/src/gpu_kernels/tsdf.rs`),
   RGB-D odometry (`crates/3d/src/odometry/mod.rs:30`), and four surface-reconstruction
   algorithms — Poisson (`mesh/reconstruction/poisson.rs:25`), ball pivoting, alpha
   shapes, marching cubes. visloc-rs has exactly one TSDF (inside its gsplat trainer,
   CPU, `crates/gsplat-train/src/mesh.rs:47`) and no Poisson/ball-pivot/alpha-shapes.
4. **Feature zoo breadth.** ORB, AKAZE, BRIEF, GFTT, LBD, FAST, Harris, HOG, ArUco,
   CharUco — versus visloc-rs's SIFT + a Harris-family corner extractor. If you need a
   specific detector, the workspace has it and the reference does not.
5. **Dense optical flow.** `Farneback` (`crates/video/src/optical_flow.rs:371`, `:469`,
   `:847`). visloc-rs has no dense flow field at all — `grep -i 'farneback|dense_flow'`
   → 0.
6. **Semi-global stereo matching.** `crates/calib3d/src/stereo_matching/sgm.rs:19`,
   `:377`. visloc-rs's dense stereo is block-matching only (`dense_stereo.rs:133`).
7. **EPnP.** `crates/calib3d/src/pnp.rs:935` `estimate_epnp` with rank/ratio degeneracy
   guards (`:1076`, `:1085`) and its own minimal-sample regression test
   (`crates/calib3d/tests/epnp_minimal_sample.rs`). visloc-rs has no EPnP — `grep epnp`
   over its whole tree returns only unrelated `repnp_free_*` config keys.
8. **Mesh/cloud file formats.** Six (PLY, OBJ, STL, PCD, LAS, glTF) versus one (PLY).
9. **A GPU HAL + runtime architecture.** `crates/hal/` with ~41 WGSL kernel files, a
   CPU SIMD fallback (`crates/hal/src/cpu/simd.rs`), a resource-group runtime
   (`crates/runtime/src/orchestrator.rs`, `pipeline/`), and a Python binding layer
   (`crates/python/`). visloc-rs binds its GPU code directly to `wgpu` with no
   abstraction layer.
10. **Plotting, signal processing, and video I/O** — `crates/plot/` (SVG output),
    `crates/signal_proc/` (FFT/DWT/filtfilt), `crates/videoio/` (ffmpeg/v4l2/gif).
    Out of scope for the reference, so no verdict is meaningful, but it is surface area
    the reference does not attempt.

**Test-density caveat, in the workspace's favour.** The workspace has **2,419** tests
against the reference's 535. Within the compared areas, though, the reference is
*denser*: `pipelines/slam/tests/bundle_adjustment.rs` has 58 tests against the
workspace's 20 in BA; `pipelines/slam/tests/online_slam.rs` has 134 against 31 in
`crates/slam/`; `crates/vision/src/features/sift.rs` has 46 in-file tests against 3 in
`crates/features/src/sift.rs`. The workspace's test count is concentrated in
`crates/hal`, `crates/io`, `crates/core`, `crates/imgproc`, `crates/3d` — the parts the
reference does not have.

## The three most consequential differences

For someone deciding which project to build on, in order of consequence:

**1. The reference has a complete visual SLAM system; the workspace has a set of
algorithms.** `crates/slam/` in the workspace totals **1,271 lines** and its entry point
is a 50-line facade (`crates/slam/src/lib.rs:34`). The reference's
`pipelines/slam/src/online_slam.rs` alone is **7,707 lines** with **134** tests, and
`incremental_sfm.rs` is **18,506**. Concretely: the workspace's `Slam` cannot accept an
IMU measurement, cannot run a loop closure through the driver, cannot reset a session,
and has no metric-scale backend. The reference can do all four. Everything else in this
document is a feature gap; this is an architectural one.

**2. Bundle adjustment numerics.** The workspace solves the normal equations with
conjugate gradients over a finite-difference Jacobian
(`crates/sfm/src/bundle_adjustment.rs:211`, `:594`). The reference eliminates landmarks
analytically via the Schur complement, Cholesky-solves the camera block
(`pipelines/slam/src/bundle.rs:1`, `block_cholesky.rs`), and can run the LM loop on GPU
with matrix-free PCG (`crates/ba-gpu/src/solver.rs:552`). The workspace's own code
records the difference it can measure — a 35-second dense factorisation replaced by a
millisecond CG solve at `:583-588` — but CG-on-`JᵀJ` degrades as the problem grows,
where Schur stays tractable. Any user who reaches a problem larger than a few hundred
cameras hits this ceiling, and it is the hardest gap on this list to close: it is a
rewrite of the solver core, not an addition.

**3. The workspace's robust-estimation front end is 8-point, not 5-point, and has no
pair classifier.** visloc-rs runs Nistér's 5-point
(`crates/vision/src/two_view/five_point.rs`, 1,257 lines) and then COLMAP's
`TwoViewGeometryVerifier::classify` (`two_view/colmap_verification.rs:239`). The
workspace's `EssentialSolver::estimate_5point`
(`crates/calib3d/src/essential.rs:14`) returns `Err` by design, and
`crates/calib3d/src/` has no classifier. Together these mean the workspace cannot ask
the two questions the whole pipeline turns on: *how many models does this pair support,
and which one?* and *is this planar?* The mapper fudges the second with a score-margin
heuristic (`crates/sfm/src/mapper.rs:236` `planar_score_margin`) rather than a model
comparison. Both of these are self-contained, well-understood ports — they are the
highest value-per-line items in the ABSENT list.

## Rows I am unsure about

Flagged rather than presented as fact:

1. **GPU SIFT (section 2).** I confirmed the workspace has no `sift-gpu`-equivalent
   crate and that `crates/hal/src/gpu_kernels/sift.rs` is a single kernel. I did **not**
   read visloc-rs's `crates/sift-gpu/src/` back-end to confirm whether it is CUDA,
   wgpu, or compute-shader. The *structural* gap is certain; the "CUDA" wording in the
   row is not.
2. **Feature-extractor trait (section 2).** I verified `grep 'trait FeatureExtractor'`
   returns 0 in the workspace, but I did not read all 12 modules of `crates/features/src/`
   to rule out a differently-named pluggable seam.
3. **BA "WEAKER" verdict (section 9).** The Schur/PCG/finite-difference claims are
   directly evidenced. But "reduced scope" for BA also touches whether the workspace's
   *output* is less accurate in practice; I did not run either system, so I am judging
   architecture, not measured accuracy.
4. **Loop-closure rows (section 11).** The workspace's mapper loop closure
   (`crates/sfm/src/mapper.rs:207`, `:889`, `:2190`) is real and counted as PRESENT. I did
   not verify end-to-end that it is exercised by a passing test, only that it exists and
   that `crates/slam/tests/integration_tests.rs` is present.
5. **TUM (section 16).** The verdict rests on a scope judgement — the workspace has a TUM
   *dataset* reader (`crates/io/src/datasets/tum.rs`) and the reference has a TUM
   *trajectory-text* parser in a different crate (`pipelines/tracking/src/trajectory.rs:1054`).
   Both are "present"; calling them comparable is defensible but arguable.
6. **Row counts.** 146 scored rows. The split 79/57/10 is mechanical, but the *placement*
   of borderline capabilities (SGM vs block matching, TSDF placement, dense flow) is a
   judgement.

## How to re-verify any ABSENT row

Each ABSENT row names the grep that establishes it. The pattern used was: search the
workspace for **the concept under several plausible names** — the reference's symbol, a
generic English term, and the common OpenCV/algorithm spelling — not just one string. The
FPFH row in section 15 exists precisely because an earlier audit did that check wrongly
and reached the opposite conclusion; `compute_fpfh_features` is at
`crates/registration/src/registration/global/fpfh.rs:32` and FPFH is a **workspace**
strength, not a gap.
