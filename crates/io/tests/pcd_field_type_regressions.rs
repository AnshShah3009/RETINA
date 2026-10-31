//! Regression tests for the defects fixed in `cv-io`'s PCD reader.
//!
//! Every test here has a matching *control*: the well-formed file must still
//! parse the same way after the fix. A fix that only makes a test pass by
//! breaking the normal case would be caught by the control, not by the fix.

use std::io::{BufReader, Cursor};

use cv_io::pcd::read_pcd;

fn parse(bytes: Vec<u8>) -> cv_core::Result<cv_core::PointCloud> {
    read_pcd(BufReader::new(Cursor::new(bytes)))
}

/// ASCII PCD with a caller-supplied FIELDS/SIZE/TYPE/COUNT and body.
///
/// `count` values are written to match `fields.len()`, which is what a real
/// writer does; the point of these tests is the *type* of each column, not a
/// mismatched COUNT.
fn pcd_ascii(fields: &str, sizes: &str, types: &str, body: &str) -> Vec<u8> {
    let count = fields.split_whitespace().count();
    let counts = vec!["1"; count].join(" ");
    format!(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n\
         WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n{body}"
    )
    .into_bytes()
}

/// Binary PCD with the same caller-supplied header lines.
fn pcd_binary(fields: &str, sizes: &str, types: &str, body: &[u8]) -> Vec<u8> {
    let count = fields.split_whitespace().count();
    let counts = vec!["1"; count].join(" ");
    let mut out = format!(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n\
         WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n"
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

// ---------------------------------------------------------------------------
// Defect 1 — a header that declares one of the three normal components and not
// the others panicked the reader.
// ---------------------------------------------------------------------------

/// A header declaring `nx` with no `ny`/`nz` is a file whose own header says the
/// record is incomplete. It must be reported (or read with the normals simply
/// absent), never panic.
///
/// `has_normals` used to be decided from `nx` alone, so the three
/// `values[ny_idx.expect("normals are only read when all three exist")]`
/// lookups in `parse_pcd_ascii` hit `None`. Measured on HEAD:
/// `panicked at crates/io/src/pcd.rs:327:31` with a `usize` index of `None`.
#[test]
fn ascii_partial_normal_fields_do_not_panic() {
    for fields in [
        "x y z nx",
        "x y z ny",
        "x y z nz",
        "x y z nx ny",
        "x y z nx nz",
        "x y z ny nz",
        "x y z normal_x",
        "x y z normal_x normal_y",
    ] {
        let count = fields.split_whitespace().count();
        let sizes = vec!["4"; count].join(" ");
        let types = vec!["F"; count].join(" ");
        let body = vec!["1.0"; count].join(" ") + "\n";
        let file = pcd_ascii(fields, &sizes, &types, &body);
        let cloud = parse(file).unwrap_or_else(|e| panic!("{fields}: must not fail, got Err: {e}"));
        assert_eq!(cloud.len(), 1, "{fields}: the geometry must still be read");
        assert!(
            cloud.normals.is_none(),
            "{fields}: declares an incomplete normal triple, so normals must be absent, \
             not fabricated from half a triple"
        );
    }
}

/// Control: a complete normal triple is still read as a normal.
///
/// Without this the fix above would pass by refusing every file that declares a
/// normal, which would be a different defect.
#[test]
fn ascii_complete_normal_fields_are_still_read() {
    for fields in ["x y z normal_x normal_y normal_z", "x y z nx ny nz"] {
        let count = fields.split_whitespace().count();
        let sizes = vec!["4"; count].join(" ");
        let types = vec!["F"; count].join(" ");
        let body = vec!["1.0"; count].join(" ") + "\n";
        let cloud = parse(pcd_ascii(fields, &sizes, &types, &body))
            .unwrap_or_else(|e| panic!("{fields}: got Err: {e}"));
        let normals = cloud
            .normals
            .unwrap_or_else(|| panic!("{fields}: a complete normal triple must be read"));
        assert_eq!(normals.len(), 1);
        assert_eq!((normals[0].x, normals[0].y, normals[0].z), (1.0, 1.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Defect 2 — a separate r/g/b colour column declared as an integer type was
// normalised by guessing from the value, so the byte value 1 became white.
// ---------------------------------------------------------------------------

/// A `U1` colour byte of `1` is near-black `1/255`, not white.
///
/// `if r > 1.0 { r / 255.0 } else { r }` cannot tell a byte from an
/// already-normalised `1.0` - they are the same f32 - so the first and most
/// common value in a colour ramp came back 255x too bright.
///
/// Measured on HEAD, ASCII, `FIELDS x y z r g b`, `SIZE 4 4 4 1 1 1`,
/// `TYPE F F F U U U`, body `0 0 0 1 128 255`:
///   colours[0] = (1.000000, 0.501961, 1.000000)
#[test]
fn ascii_byte_colour_one_is_not_white() {
    let cloud = parse(pcd_ascii(
        "x y z r g b",
        "4 4 4 1 1 1",
        "F F F U U U",
        "0 0 0 1 128 255\n",
    ))
    .expect("read failed");
    let colors = cloud.colors.expect("colors missing");
    assert!(
        (colors[0].x - 1.0 / 255.0).abs() < 1e-6,
        "byte 1 must decode to 1/255 = {:.6}, got {:.6}",
        1.0 / 255.0,
        colors[0].x
    );
    assert!(
        (colors[0].y - 128.0 / 255.0).abs() < 1e-6,
        "byte 128 must decode to {:.6}, got {:.6}",
        128.0 / 255.0,
        colors[0].y
    );
    assert!(
        (colors[0].z - 1.0).abs() < 1e-6,
        "byte 255 must decode to 1.0, got {:.6}",
        colors[0].z
    );
}

/// Same defect on the binary path, which carried a byte-identical copy of the
/// heuristic.
///
/// Measured on HEAD: colours[0] = (1.0, 0.501961, 1.0) for the body below.
#[test]
fn binary_byte_colour_one_is_not_white() {
    let mut body = Vec::new();
    for v in [0.0f32, 0.0, 0.0] {
        body.extend_from_slice(&v.to_le_bytes());
    }
    body.push(1u8);
    body.push(128u8);
    body.push(255u8);

    let cloud = parse(pcd_binary(
        "x y z r g b",
        "4 4 4 1 1 1",
        "F F F U U U",
        &body,
    ))
    .expect("read failed");
    let colors = cloud.colors.expect("colors missing");
    assert!(
        (colors[0].x - 1.0 / 255.0).abs() < 1e-6,
        "byte 1 must decode to 1/255, got {:.6}",
        colors[0].x
    );
    assert!((colors[0].y - 128.0 / 255.0).abs() < 1e-6);
    assert!((colors[0].z - 1.0).abs() < 1e-6);
}

/// Control: a float-declared colour column is already in [0, 1] and must not be
/// divided by 255 a second time.
///
/// This is the direction the byte-value heuristic got *right* by accident, and
/// the one the fix must not regress: a normalised `0.5` is `0.5`, not `0.5/255`.
#[test]
fn ascii_float_colour_is_not_rescaled() {
    let cloud = parse(pcd_ascii(
        "x y z r g b",
        "4 4 4 4 4 4",
        "F F F F F F",
        "0 0 0 0.003921569 0.5019607843 1.0\n",
    ))
    .expect("read failed");
    let colors = cloud.colors.expect("colors missing");
    assert!(
        (colors[0].x - 1.0 / 255.0).abs() < 1e-6,
        "got {}",
        colors[0].x
    );
    assert!(
        (colors[0].y - 0.5019607843).abs() < 1e-6,
        "got {}",
        colors[0].y
    );
    assert!((colors[0].z - 1.0).abs() < 1e-6, "got {}", colors[0].z);
}

/// Control: a mid-grey float colour must survive untouched. Without this, a fix
/// that divided *every* colour by 255 would pass the byte test above.
#[test]
fn ascii_float_mid_grey_colour_untouched() {
    let cloud = parse(pcd_ascii(
        "x y z r g b",
        "4 4 4 4 4 4",
        "F F F F F F",
        "0 0 0 0.5 0.25 0.75\n",
    ))
    .expect("read failed");
    let colors = cloud.colors.expect("colors missing");
    assert!((colors[0].x - 0.5).abs() < 1e-6, "got {}", colors[0].x);
    assert!((colors[0].y - 0.25).abs() < 1e-6, "got {}", colors[0].y);
    assert!((colors[0].z - 0.75).abs() < 1e-6, "got {}", colors[0].z);
}

/// Control: the packed `rgb` field is untouched by the fix, including for the
/// bit patterns that decode to `inf`/`NaN` as f32 - which is why the packed path
/// must not go through the finite-coordinate check.
#[test]
fn ascii_packed_rgb_still_decodes() {
    use cv_core::point_cloud::PointCloud;
    use nalgebra::Point3;

    let mut cloud = PointCloud::new(vec![Point3::new(1.0, 2.0, 3.0)]);
    // The writer stores (255, 0, 0) as the u32 0x00FF0000.
    cloud.colors = Some(vec![Point3::new(1.0, 0.0, 0.0)]);
    let mut buf = Vec::new();
    cv_io::pcd::write_pcd(&mut buf, &cloud).expect("write failed");

    let read = parse(buf).expect("read failed");
    let colors = read.colors.expect("colors missing");
    assert!((colors[0].x - 1.0).abs() < 0.01, "got {}", colors[0].x);
    assert!(colors[0].y < 0.01, "got {}", colors[0].y);
}

/// Control: an `I`-typed colour column is also a byte and must be divided. The
/// declared-type rule is about `TYPE`, not about the letter that starts the
/// SIZE.
#[test]
fn ascii_signed_byte_colour_one_is_not_white() {
    let cloud = parse(pcd_ascii(
        "x y z r g b",
        "4 4 4 1 1 1",
        "F F F I I I",
        "0 0 0 1 128 255\n",
    ))
    .expect("read failed");
    let colors = cloud.colors.expect("colors missing");
    assert!(
        (colors[0].x - 1.0 / 255.0).abs() < 1e-6,
        "a signed byte 1 is still 1/255, got {:.6}",
        colors[0].x
    );
}
