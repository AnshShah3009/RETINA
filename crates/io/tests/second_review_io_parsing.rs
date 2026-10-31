//! Readers rejecting or misreading valid files (GAPS g-i3..g-i6).

use cv_core::PointCloud;
use cv_io::{read_pcd, write_pcd_binary_compressed, ObjMesh};
use nalgebra::Point3;
use std::io::Cursor;

fn ascii_pcd(fields: &str, size: &str, ty: &str, count: &str, row: &str) -> String {
    format!(
        "VERSION .7\nFIELDS {fields}\nSIZE {size}\nTYPE {ty}\nCOUNT {count}\nWIDTH 1\nHEIGHT 1\n\
         VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n{row}\n"
    )
}

/// g-i3: LZF grows incompressible data by one control byte per 32-byte
/// literal run, so the crate's own writer produces `compressed > uncompressed`
/// on random points - and the reader refused it.
#[test]
fn binary_compressed_round_trips_incompressible_points() {
    let mut s: u32 = 99;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    let points: Vec<Point3<f32>> = (0..1000)
        .map(|_| Point3::new(rnd(), rnd(), rnd()))
        .collect();
    let cloud = PointCloud {
        points: points.clone(),
        colors: None,
        normals: None,
    };
    let mut buf = Vec::new();
    write_pcd_binary_compressed(&mut buf, &cloud).unwrap();
    let back = read_pcd(Cursor::new(buf)).expect("reads its own compressed output");
    assert_eq!(back.points, points);
}

/// g-i4: a field with `COUNT 2` occupies two columns.
#[test]
fn ascii_respects_count() {
    let pcd = ascii_pcd("h x y z", "4 4 4 4", "F F F F", "2 1 1 1", "9 9 1 2 3");
    let c = read_pcd(Cursor::new(pcd)).unwrap();
    assert_eq!(c.points, vec![Point3::new(1.0, 2.0, 3.0)]);
}

/// g-i5: packed rgb declared `TYPE U` is an integer token, not float bits.
#[test]
fn ascii_packed_rgb_declared_unsigned() {
    let pcd = ascii_pcd(
        "x y z rgb",
        "4 4 4 4",
        "F F F U",
        "1 1 1 1",
        "0 0 0 16711680",
    );
    let c = read_pcd(Cursor::new(pcd)).unwrap();
    assert_eq!(c.colors.unwrap(), vec![Point3::new(1.0, 0.0, 0.0)]);
    // rgba above 2^24: the low bits must survive (f32 would round them).
    let pcd = ascii_pcd(
        "x y z rgba",
        "4 4 4 4",
        "F F F U",
        "1 1 1 1",
        "0 0 0 4278190081",
    );
    let c = read_pcd(Cursor::new(pcd)).unwrap();
    assert_eq!(c.colors.unwrap(), vec![Point3::new(0.0, 0.0, 1.0 / 255.0)]);
}

/// g-i6: negative OBJ indices are relative to the vertices read so far.
#[test]
fn obj_relative_face_indices() {
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 0 0 1\nf 1 -1 -2\n";
    let m = ObjMesh::read(Cursor::new(obj)).unwrap();
    assert_eq!(m.faces, vec![vec![0, 1, 2], vec![0, 3, 2]]);
    assert!(ObjMesh::read(Cursor::new("v 0 0 0\nf -1 -2 -1\n")).is_err());
}
