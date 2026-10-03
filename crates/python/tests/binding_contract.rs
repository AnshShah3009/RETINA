//! Regression tests for the Python binding layer (`cv-python`).
//!
//! ## Why these live in an integration test
//!
//! `crates/python` is a PyO3 **cdylib**. Its `Cargo.toml` declared
//! `crate-type = ["cdylib"]`, which produced no linkable Rust library for the
//! integration-test harness: a test file under `crates/python/tests/` that
//! referred to `cv_native` failed to compile with
//!
//! ```text
//! error[E0433]: cannot find module or crate `cv_native` in this scope
//! ```
//!
//! That is why the crate advertised 96 public Python items and 0 tests - there
//! was no mechanism to test it at all. `"rlib"` was added to `crate-type` so the
//! binding helpers can be exercised from a normal `cargo test`; `cargo build -p
//! cv-python` still produces the same `libcv_native.so` cdylib, so the shipped
//! artefact is unchanged.
//!
//! Every test below is a **pure-Rust** assertion against the helper the binding
//! delegates to.
//!
//! ## What is deliberately NOT here, and why
//!
//! `set_normals` now returns `PyResult<()>` so a length mismatch raises
//! `ValueError` in Python. Constructing that `PyErr` calls into CPython, so the
//! assertion on the *error* cannot live in this file - see
//! `set_normals_mismatch_is_measured_in_python` in the audit notes. The tests
//! here pin what *is* reachable without an interpreter: the shape/length
//! invariants, the round trips, and the fact that the validation runs at all
//! (a mismatched list is still rejected, and the cloud is left untouched).

use cv_native::core::PyPointCloud;

// ── PyPointCloud: normals must stay one-per-point ──────────────────────────

#[test]
fn set_normals_rejects_a_list_that_is_not_one_per_point() {
    let mut cloud = PyPointCloud::new(vec![(0.0, 0.0, 0.0); 5]);

    // Too few: the two arrays go out of step and every point past index 2 reads
    // the wrong normal (or past the end of the shorter list).
    assert!(
        cloud.set_normals(vec![(0.0, 0.0, 1.0); 3]).is_err(),
        "3 normals for 5 points must be rejected"
    );

    // Too many is the same defect in the other direction.
    assert!(
        cloud.set_normals(vec![(0.0, 0.0, 1.0); 9]).is_err(),
        "9 normals for 5 points must be rejected"
    );

    // Off by one in either direction is still off by one.
    assert!(cloud.set_normals(vec![(0.0, 0.0, 1.0); 4]).is_err());
    assert!(cloud.set_normals(vec![(0.0, 0.0, 1.0); 6]).is_err());
}

#[test]
fn set_normals_accepts_exactly_one_normal_per_point() {
    // Control: the well-formed case has to keep working, or the fix above would
    // just be a binding that refuses everything. This is the test that pins the
    // "was not rejected" half of the contract without needing a live
    // interpreter.
    let mut cloud = PyPointCloud::new(vec![(0.0, 0.0, 0.0); 5]);
    assert!(cloud.set_normals(vec![(1.0, 2.0, 3.0); 5]).is_ok());

    // An empty cloud takes an empty list, for the same reason.
    let mut empty = PyPointCloud::new(vec![]);
    assert!(empty.set_normals(vec![]).is_ok());
    assert_eq!(empty.num_points(), 0);
}

// ── the f32 round trip through the binding must not lose the values ────────

#[test]
fn point_cloud_flat_round_trip_preserves_every_coordinate() {
    // `to_numpy` flattens (x, y, z) triplets; `num_points` and the flat length
    // must agree, because a caller slicing by one but iterating by the other
    // reads past the end.
    let points = vec![(1.0f32, 2.0, 3.0), (-4.5, 0.25, 7.0), (0.0, 0.0, 0.0)];
    let cloud = PyPointCloud::new(points.clone());

    let flat = cloud.to_numpy();
    assert_eq!(
        flat.len(),
        cloud.num_points() * 3,
        "flat length vs num_points"
    );
    for (i, (x, y, z)) in points.iter().enumerate() {
        assert_eq!(&flat[3 * i..3 * i + 3], &[*x, *y, *z], "point {i}");
    }

    // Control: `points_to_list` reports the same points in the same order.
    assert_eq!(cloud.points_to_list(), points);
}

// ── PyTensor: the reported shape must match the data it hands back ─────────

#[test]
fn tensor_shape_and_data_length_agree() {
    for (c, h, w) in [
        (1usize, 1usize, 1usize),
        (2, 3, 4),
        (3, 224, 224),
        (1, 0, 5),
    ] {
        let t = cv_native::core::PyTensor::zeros((c, h, w));
        assert_eq!(t.shape(), (c, h, w));
        assert_eq!(
            t.to_numpy().len(),
            c * h * w,
            "({c},{h},{w}): to_numpy returned {} values",
            t.to_numpy().len()
        );
        assert!(t.to_numpy().iter().all(|v| *v == 0.0));

        let ones = cv_native::core::PyTensor::ones((c, h, w));
        assert_eq!(ones.to_numpy().len(), c * h * w);
        assert!(ones.to_numpy().iter().all(|v| *v == 1.0));
    }
}
