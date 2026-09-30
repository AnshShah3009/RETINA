//! Aggregated robustness suite for the `cv-io` file-format parsers.
//!
//! This is a *shim* target: cargo only auto-discovers integration-test targets
//! that are direct children of `tests/` (it does not walk subdirectories), so
//! the real suites live in `tests/robustness/` and are included as modules from
//! here. Each is a separate module, so a failure names the file it came from.
//!
//! Layout of the suites:
//!
//! | module                 | parser(s)                                   |
//! |------------------------|---------------------------------------------|
//! | `pcd_robustness`       | `crates/io/src/pcd.rs`                      |
//! | `ply_robustness`       | `crates/io/src/ply.rs`                      |
//! | `stl_robustness`       | `crates/io/src/stl.rs`                      |
//! | `obj_robustness`       | `crates/io/src/obj.rs`                      |
//! | `datasets_robustness`  | `crates/io/src/datasets/*.rs`               |
//! | `mutation_corpus`      | all of the above, deterministic mutations    |
//!
//! `crates/io/src/las_io.rs` is feature-gated behind `--features las` and is
//! covered by `format_roundtrip_tests.rs`.

#[path = "robustness/common/mod.rs"]
#[allow(dead_code)]
mod common;

#[path = "robustness/pcd_robustness.rs"]
mod pcd_robustness;

#[path = "robustness/ply_robustness.rs"]
mod ply_robustness;

#[path = "robustness/stl_robustness.rs"]
mod stl_robustness;

#[path = "robustness/obj_robustness.rs"]
mod obj_robustness;

#[path = "robustness/datasets_robustness.rs"]
mod datasets_robustness;

#[path = "robustness/mutation_corpus.rs"]
mod mutation_corpus;
