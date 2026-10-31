//! GLB files must never carry an inverted POSITION accessor.
//!
//! The accessor bounds were seeded with `min = f32::MAX` / `max = f32::MIN`.
//! `f32::MIN` is the most *negative* finite f32, not the smallest positive one,
//! so `max` could never rise above it. With no vertices to fold in, the
//! sentinels were written straight into the file. Measured: `write_glb` on an
//! empty mesh produced
//!
//! ```text
//! "min": [3.4028235e38, 3.4028235e38, 3.4028235e38]
//! "max": [-3.4028235e38, -3.4028235e38, -3.4028235e38]
//! ```
//!
//! - min greater than max on every axis, in a 876-byte file. No importer can
//! read that as an extent.
//!
//! glTF requires `min` and `max` to be present when written, so the fix records a
//! degenerate box at the origin, which is valid.

#![cfg(feature = "gltf")]

use cv_io::gltf_io::write_glb;
use nalgebra::Point3;

/// Parse the POSITION accessor's `min`/`max` out of a written GLB's JSON chunk.
fn position_bounds(bytes: &[u8]) -> (Vec<f64>, Vec<f64>) {
    let text = String::from_utf8_lossy(bytes);
    let i = text
        .find("\"min\"")
        .expect("the GLB should record a POSITION min");
    let tail = &text[i..];

    let read_vec = |key: &str| -> Vec<f64> {
        let at = tail.find(key).expect("accessor should carry both bounds");
        let after = &tail[at + key.len()..];
        let open = after.find('[').expect("a JSON array");
        let close = after[open..].find(']').expect("a JSON array");
        after[open + 1..open + close]
            .split(',')
            .map(|v| v.trim().parse::<f64>().expect("a number"))
            .collect()
    };

    (read_vec("\"min\""), read_vec("\"max\""))
}

fn write_temp(name: &str, verts: &[Point3<f32>], faces: &[[usize; 3]]) -> Vec<u8> {
    let dir = std::env::temp_dir().join("glb_bounds_tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(name);
    write_glb(&path, verts, faces, None).expect("write_glb should succeed");
    std::fs::read(&path).expect("the file should exist")
}

#[test]
fn an_empty_mesh_records_a_valid_degenerate_box() {
    let bytes = write_temp("empty.glb", &[], &[]);
    let (min, max) = position_bounds(&bytes);

    assert_eq!(min.len(), 3);
    assert_eq!(max.len(), 3);
    for i in 0..3 {
        assert!(
            min[i] <= max[i],
            "accessor axis {i} is inverted: min={} > max={}",
            min[i],
            max[i]
        );
    }
    assert!(
        min.iter().chain(max.iter()).all(|v| v.is_finite()),
        "an accessor bound must be finite: min={min:?} max={max:?}"
    );
}

/// The control: a single vertex records its own coordinates.
#[test]
fn a_single_vertex_records_its_own_bounds() {
    let bytes = write_temp("one.glb", &[Point3::new(1.0, 2.0, 3.0)], &[]);
    let (min, max) = position_bounds(&bytes);
    assert_eq!(min, vec![1.0, 2.0, 3.0]);
    assert_eq!(max, vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_real_mesh_records_its_true_extent() {
    let verts = [
        Point3::new(-4.0, 1.0, 0.5),
        Point3::new(9.0, -2.0, 3.0),
        Point3::new(0.0, 7.0, -6.0),
    ];
    let faces = [[0usize, 1, 2]];
    let bytes = write_temp("three.glb", &verts, &faces);
    let (min, max) = position_bounds(&bytes);
    assert_eq!(min, vec![-4.0, -2.0, -6.0], "min wrong");
    assert_eq!(max, vec![9.0, 7.0, 3.0], "max wrong");
}

/// All-negative coordinates, where a max seeded at 0.0 would also be wrong.
#[test]
fn an_all_negative_mesh_records_its_true_extent() {
    let verts = [
        Point3::new(-10.0, -20.0, -30.0),
        Point3::new(-1.0, -2.0, -3.0),
    ];
    let bytes = write_temp("negative.glb", &verts, &[[0usize, 1, 0]]);
    let (min, max) = position_bounds(&bytes);
    assert_eq!(min, vec![-10.0, -20.0, -30.0]);
    assert_eq!(max, vec![-1.0, -2.0, -3.0]);
}
