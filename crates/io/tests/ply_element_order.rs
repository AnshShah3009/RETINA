//! PLY permits elements in any order; the reader must honour that.
//!
//! `read_ply` records only whether the header declared an element *after* the
//! vertices (`vertex_block_ends`), never how many body rows belong to earlier
//! elements. The body loop is a plain `for _ in 0..num_vertices`, so with
//!
//! ```text
//! element face 2
//! property list uchar int vertex_indices
//! element vertex 2
//! property float x
//! property float y
//! property float z
//! end_header
//! 3 0 0 1
//! 3 0 1 2
//! 1 2 3
//! 4 5 6
//! ```
//!
//! the two face rows are consumed as coordinates and the real vertex block is
//! never read. Measured: `Ok`, 2 points, `[[3,0,0],[3,0,1]]` — total data loss
//! reported as success.
//!
//! Three further consequences of the missing element model are pinned here:
//!
//! * `props` accumulated across element boundaries, so a face property declared
//!   before the vertices could be picked up as a vertex column, shifting every
//!   index after it.
//! * Two `element vertex` blocks unioned their properties, giving a width no
//!   real row has and a spurious "Not enough values".
//! * A vertex element without `x` was rejected as "missing x property" because
//!   the search spanned the whole header, including later elements.
//!
//! This reader still refuses `property list` inside a *vertex* element — a list
//! occupies a variable number of body columns and cannot be a fixed index — so
//! the face rows are skipped as whole rows, which is exactly what the format
//! requires.

use std::io::BufReader;

/// A process-unique suffix; see the note in `ply_packed_rgb_finite.rs`.
fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let seq = N.fetch_add(1, Ordering::Relaxed);
    std::process::id() as u64 * 1_000_000 + seq
}

fn read(name: &str, text: &str) -> cv_core::PointCloud {
    let dir = std::env::temp_dir().join("cv_ply_element_order_tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{}_{}.ply", name, unique_id()));
    std::fs::write(&path, text).expect("write the fixture");

    cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .unwrap_or_else(|e| panic!("{name}: read_ply failed: {e}\nthe file was:\n{text}"))
}

fn coords(cloud: &cv_core::PointCloud, i: usize) -> (f32, f32, f32) {
    let p = cloud.points[i];
    (p.x, p.y, p.z)
}

/// The reported case: `element face` declared *before* `element vertex`.
///
/// The face rows look exactly like coordinates (`3 0 0 1`), so before the fix
/// they were accepted silently and the true vertices were never read.
#[test]
fn a_face_element_before_the_vertices_is_skipped() {
    let text = "ply
format ascii 1.0
element face 2
property list uchar int vertex_indices
element vertex 2
property float x
property float y
property float z
end_header
3 0 0 1
3 0 1 2
1 2 3
4 5 6
";
    let cloud = read("face_first", text);
    assert_eq!(
        cloud.len(),
        2,
        "the two face rows must not be consumed as vertices"
    );
    assert_eq!(coords(&cloud, 0), (1.0, 2.0, 3.0), "first vertex");
    assert_eq!(coords(&cloud, 1), (4.0, 5.0, 6.0), "second vertex");
}

/// The same file with the elements in the conventional order, which already
/// worked. It must keep working — the fix is about *order*, not about faces.
#[test]
fn control_a_face_element_after_the_vertices_is_skipped() {
    let text = "ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
1 2 3
4 5 6
3 0 0 1
3 0 1 2
";
    let cloud = read("face_last", text);
    assert_eq!(cloud.len(), 2);
    assert_eq!(coords(&cloud, 0), (1.0, 2.0, 3.0));
    assert_eq!(coords(&cloud, 1), (4.0, 5.0, 6.0));
}

/// CONTROL: no face element at all, and the plain vertices survive unchanged.
/// A fix that simply started skipping rows would show up here.
#[test]
fn control_a_plain_vertex_element_reads() {
    let text = "ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
1 2 3 255 0 0
4 5 6 0 255 0
7 8 9 0 0 255
";
    let cloud = read("plain", text);
    assert_eq!(cloud.len(), 3);
    assert_eq!(coords(&cloud, 2), (7.0, 8.0, 9.0));
    let colors = cloud.colors.as_ref().expect("colours were declared");
    assert!(
        (colors[2].z - 1.0).abs() < 1e-6,
        "the blue vertex read as {:?}",
        colors[2]
    );
}

/// A `face` property declared before the vertices must not become a vertex
/// column. `vertex_block_ends` alone did not stop `props` accumulating across
/// the element boundary, so `nx` from a face element could win the name search
/// and shift the colour indices.
#[test]
fn a_face_property_before_the_vertices_is_not_a_vertex_column() {
    let text = "ply
format ascii 1.0
element face 1
property float nx
element vertex 1
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0
1 2 3 255 0 0
";
    let cloud = read("face_nx_first", text);
    assert_eq!(cloud.len(), 1);
    assert_eq!(coords(&cloud, 0), (1.0, 2.0, 3.0));
    let colors = cloud.colors.as_ref().expect("colours were declared");
    let c = colors[0];
    let near = |a: f32, b: u8| (a - b as f32 / 255.0).abs() < 1e-6;
    assert!(
        near(c.x, 255) && near(c.y, 0) && near(c.z, 0),
        "red vertex read as {c:?} — the face property shifted the colour columns"
    );
}

/// Two `element vertex` blocks must not union their properties. Before the fix
/// `props` accumulated, so the second block's `nx/ny/nz` were found even though
/// the first block declared no normals, the row width stopped matching either
/// block's rows, and the whole file was refused.
///
/// Both blocks declare the same properties here, which is the only case a
/// reader with a single column model can serve. With *differing* properties the
/// file is genuinely ambiguous for such a reader, and that is deliberately left
/// out of this test rather than pinned as a behaviour.
#[test]
fn two_vertex_elements_do_not_union_their_properties() {
    let text = "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
element vertex 1
property float x
property float y
property float z
end_header
1 2 3
4 5 6
";
    let cloud = read("two_vertices", text);
    assert_eq!(
        cloud.len(),
        2,
        "both vertex blocks contribute their own points"
    );
    assert_eq!(coords(&cloud, 0), (1.0, 2.0, 3.0));
    assert_eq!(coords(&cloud, 1), (4.0, 5.0, 6.0));
}

/// A vertex element whose properties are the same as a *later* element's. The
/// name search must be taken against the vertex element, not the last one seen,
/// or the last element's column order is used to read the vertices.
#[test]
fn a_later_x_property_does_not_redefine_the_vertex_columns() {
    let text = "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
element camera 1
property float x
property float y
property float z
end_header
1 2 3
9 9 9
";
    let cloud = read("later_x", text);
    assert_eq!(cloud.len(), 1);
    assert_eq!(coords(&cloud, 0), (1.0, 2.0, 3.0));
}

/// CONTROL for the element model: a `property list` inside the *vertex* element
/// must still be refused. The fix skips non-vertex rows as whole rows, and this
/// proves it did not start silently parsing a variable-width vertex row.
#[test]
fn control_a_list_property_in_the_vertex_element_is_still_refused() {
    let text = "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property list uchar int vertex_indices
end_header
1 2 3 1 42
";
    let dir = std::env::temp_dir().join("cv_ply_element_order_tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("vertex_list_{}.ply", unique_id()));
    std::fs::write(&path, text).expect("write the fixture");

    let err = cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .expect_err("a list property in the vertex element must still be refused");
    assert!(
        err.to_string().contains("list"),
        "the error should name the problem, got: {err}"
    );
}
