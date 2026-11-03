//! The crate layer order from `docs/ARCHITECTURE.md`, enforced.
//!
//! Every workspace crate sits in a layer and may depend (through normal or
//! build dependencies) only on crates in strictly lower layers. That keeps the
//! graph acyclic by construction and stops a low-level crate from quietly
//! pulling in a heavy one. Dev-dependencies are exempt: a test may use any
//! crate.
//!
//! Adding a crate means adding it to `LAYERS`; the second test fails until
//! that decision is made.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

/// Layer of each workspace crate. See `docs/ARCHITECTURE.md` for what each
/// layer is for.
const LAYERS: &[(&str, u8)] = &[
    // L0 foundation
    ("cv-core", 0),
    ("cv-geometry2d", 0),
    ("cv-plot", 0),
    // L1 backends, I/O and leaf utilities
    ("cv-hal", 1),
    ("cv-io", 1),
    ("cv-math", 1),
    ("cv-eval", 1),
    ("cv-photo", 1),
    ("cv-rendering", 1),
    ("cv-videoio", 1),
    ("cv-viewer", 1),
    // L2 numerics
    ("cv-optimize", 2),
    ("cv-pointcloud", 2),
    ("cv-signal", 2),
    ("cv-video", 2),
    ("cv-distributed", 2),
    // L3 runtime
    ("cv-runtime", 3),
    ("cv-scientific", 3),
    // L4 domain: images and 3-D data
    ("cv-imgproc", 4),
    ("cv-3d", 4),
    // L5 domain: geometry, features, registration
    ("cv-calib3d", 5),
    ("cv-features", 5),
    ("cv-registration", 5),
    ("cv-dnn", 5),
    // L6 domain built on features and geometry
    ("cv-cli", 6),
    // L7 pipelines
    ("cv-sfm", 7),
    ("cv-slam", 7),
    ("cv-localization", 7),
    ("cv-examples", 7),
    // L8 public surface
    ("cv-python", 8),
    ("rust-cv-native", 8),
];

/// Edges that break the layer order today and are being removed by the
/// workspace-structure plan (docs/superpowers/plans/2026-11-03-workspace-structure.md).
/// Nothing may be added here.
const KNOWN_VIOLATIONS: &[(&str, &str)] = &[("cv-features", "cv-calib3d"), ("cv-io", "cv-3d")];

/// Workspace crates and their normal + build dependencies on other workspace
/// crates.
fn workspace_graph() -> BTreeMap<String, BTreeSet<String>> {
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
    let packages = metadata["packages"].as_array().expect("packages array");
    let names: BTreeSet<String> = packages
        .iter()
        .map(|p| p["name"].as_str().expect("package name").to_owned())
        .collect();

    packages
        .iter()
        .map(|p| {
            let deps = p["dependencies"]
                .as_array()
                .expect("dependencies array")
                .iter()
                .filter(|d| d["kind"].as_str() != Some("dev"))
                .map(|d| d["name"].as_str().expect("dependency name").to_owned())
                .filter(|d| names.contains(d))
                .collect();
            (p["name"].as_str().unwrap().to_owned(), deps)
        })
        .collect()
}

fn layer(name: &str) -> Option<u8> {
    LAYERS.iter().find(|(n, _)| *n == name).map(|&(_, l)| l)
}

#[test]
fn every_workspace_crate_has_a_layer() {
    let graph = workspace_graph();
    let missing: Vec<_> = graph.keys().filter(|n| layer(n).is_none()).collect();
    assert!(
        missing.is_empty(),
        "crates without a layer in LAYERS: {missing:?}"
    );

    let stale: Vec<_> = LAYERS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !graph.contains_key(*n))
        .collect();
    assert!(
        stale.is_empty(),
        "LAYERS lists crates that no longer exist: {stale:?}"
    );
}

#[test]
fn crate_dependencies_respect_layers() {
    let graph = workspace_graph();
    let mut violations = Vec::new();
    let mut still_known = BTreeSet::new();
    for (krate, deps) in &graph {
        let Some(from) = layer(krate) else { continue };
        for dep in deps {
            let Some(to) = layer(dep) else { continue };
            if to < from {
                continue;
            }
            if KNOWN_VIOLATIONS.contains(&(krate.as_str(), dep.as_str())) {
                still_known.insert((krate.as_str(), dep.as_str()));
                continue;
            }
            violations.push(format!("{krate} (L{from}) -> {dep} (L{to})"));
        }
    }
    assert!(
        violations.is_empty(),
        "dependencies on the same or a higher layer:\n  {}",
        violations.join("\n  ")
    );
    let fixed: Vec<_> = KNOWN_VIOLATIONS
        .iter()
        .filter(|v| !still_known.contains(*v))
        .collect();
    assert!(
        fixed.is_empty(),
        "remove fixed entries from KNOWN_VIOLATIONS: {fixed:?}"
    );
}
