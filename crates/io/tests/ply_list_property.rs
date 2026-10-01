//! PLY `property list` must be reported, not mis-parsed.
//!
//! A `property list <count_type> <value_type> <name>` occupies a *variable*
//! number of body columns: one for the count, then that many values. This reader
//! has no element/property model - it pushed every property's last token as a
//! fixed column name - so a vertex declared
//!
//! ```text
//! element vertex 1
//! property float x
//! property float y
//! property float z
//! property list uchar int vertex_indices
//! property uchar red
//! property uchar green
//! property uchar blue
//! ```
//!
//! with body row `1 2 3 1 42 255 0 0` read the `42` (the first list value) as
//! green and the `255` as blue. Measured: colours `[0.165, 1.0, 0.0]` where
//! `[1.0, 0.0, 0.0]` was correct - a red vertex read as green.
//!
//! This is the same class of defect as the `element face` bug: the reader has no
//! element model, so a body field lands in the wrong property and nothing
//! notices.
//!
//! Supporting lists properly needs a real element/property model. Until then the
//! file is refused, because plausible-looking wrong geometry is worse than an
//! error.

use std::io::BufReader;

/// A vertex element with a list property before the colours.
fn with_list_property(body: &str) -> String {
    format!(
        "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property list uchar int vertex_indices
property uchar red
property uchar green
property uchar blue
end_header
{body}"
    )
}

#[test]
fn a_list_property_is_reported_rather_than_mis_parsed() {
    let dir = tempdir();
    let path = write(
        &dir,
        "list.ply",
        &with_list_property("1 2 3 1 42 255 0 0\n"),
    );

    let err = cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .expect_err("a list property must be reported, not mis-parsed");

    let msg = err.to_string();
    assert!(
        msg.contains("list"),
        "the error should name the problem, got: {msg}"
    );
    assert!(
        msg.contains("vertex_indices"),
        "the error should name the offending property, got: {msg}"
    );
}

/// The list may come after the properties it would corrupt, in which case the
/// scalar fields are already correct - but the row width still differs from what
/// the reader expects, so it is refused for the same reason.
#[test]
fn a_trailing_list_property_is_also_reported() {
    let text = "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property list uchar int face_indices
end_header
1 2 3 255 0 0 1 7
";
    let dir = tempdir();
    let path = write(&dir, "trailing.ply", text);

    let err = cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .expect_err("a trailing list property changes the row width too");
    assert!(err.to_string().contains("list"), "got: {err}");
}

/// The ordinary case must still work, or the guard is refusing everything.
#[test]
fn a_plain_vertex_element_still_parses() {
    let text = "ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
1 2 3 255 0 0
4 5 6 0 255 0
";
    let dir = tempdir();
    let path = write(&dir, "plain.ply", text);

    let cloud = cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .expect("a plain PLY parses");
    assert_eq!(cloud.points.len(), 2);
    assert_eq!(
        (cloud.points[0].x, cloud.points[0].y, cloud.points[0].z),
        (1.0, 2.0, 3.0)
    );
    assert_eq!(
        (cloud.points[1].x, cloud.points[1].y, cloud.points[1].z),
        (4.0, 5.0, 6.0)
    );

    let colors = cloud.colors.as_ref().expect("colours were declared");
    let c = colors[0];
    let near = |a: f32, b: u8| (a - b as f32 / 255.0).abs() < 1e-3;
    assert!(
        near(c.x, 255) && near(c.y, 0) && near(c.z, 0),
        "red vertex read as {c:?}"
    );
}

// --- tiny helpers, so the test file needs no dev-dependencies ---

fn tempdir() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = format!(
        "{}/cv_ply_list_{}_{}",
        std::env::temp_dir().display(),
        std::process::id(),
        n
    );
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write(dir: &str, name: &str, contents: &str) -> String {
    let path = format!("{dir}/{name}");
    std::fs::write(&path, contents).expect("write the fixture");
    path
}
