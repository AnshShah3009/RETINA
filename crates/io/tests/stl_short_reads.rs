//! `read_stl` reads the same mesh however the reader splits its bytes.
//!
//! `Read::read` may return fewer bytes than asked for (pipes, sockets,
//! decompressors, `BufReader` with a small buffer). The 80-byte header was
//! taken with a single `read`, so a reader that delivered one byte at a time
//! left 79 header bytes unread: a binary file then took header bytes as its
//! triangle count, and an ASCII file whose `facet` keyword fell outside the
//! one buffered byte was sent down the binary path; all three cases below
//! failed with `IoError("failed to fill whole buffer")`.

use cv_io::TriangleMesh;
use cv_io::stl::{read_stl, write_stl_ascii, write_stl_binary};
use nalgebra::Point3;
use std::io::{BufReader, Cursor, Read};

/// A reader that returns at most one byte per `read` call.
struct OneByteAtATime<R>(R);

impl<R: Read> Read for OneByteAtATime<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min(1);
        self.0.read(&mut buf[..n])
    }
}

fn trickle(bytes: Vec<u8>) -> BufReader<OneByteAtATime<Cursor<Vec<u8>>>> {
    BufReader::with_capacity(1, OneByteAtATime(Cursor::new(bytes)))
}

fn tetrahedron() -> TriangleMesh {
    TriangleMesh::with_vertices_and_faces(
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ],
        vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    )
}

fn assert_same(a: &TriangleMesh, b: &TriangleMesh, what: &str) {
    assert_eq!(a.vertices, b.vertices, "{what}: vertices");
    assert_eq!(a.faces, b.faces, "{what}: faces");
}

#[test]
fn binary_stl_survives_one_byte_reads() {
    let mut bytes = Vec::new();
    write_stl_binary(&mut bytes, &tetrahedron()).unwrap();
    let whole = read_stl(Cursor::new(bytes.clone())).unwrap();
    let trickled = read_stl(trickle(bytes)).expect("one-byte reads");
    assert_same(&trickled, &whole, "binary");
    assert_eq!(trickled.faces.len(), 4);
}

#[test]
fn ascii_stl_survives_one_byte_reads() {
    let mut bytes = Vec::new();
    write_stl_ascii(&mut bytes, &tetrahedron()).unwrap();
    // A long solid name pushes the first `facet` past the 80-byte header, so
    // detection has to look beyond it.
    let text = String::from_utf8(bytes).unwrap();
    let text = text.replacen("solid", &format!("solid {}", "n".repeat(100)), 1);
    let bytes = text.into_bytes();
    let whole = read_stl(Cursor::new(bytes.clone())).unwrap();
    let trickled = read_stl(trickle(bytes)).expect("one-byte reads");
    assert_same(&trickled, &whole, "ascii");
    assert_eq!(trickled.faces.len(), 4);
}

#[test]
fn a_tiny_ascii_file_is_still_read() {
    // Shorter than the 80-byte header: reading it must not demand 80 bytes.
    let bytes = b"solid t\nendsolid t\n".to_vec();
    let mesh = read_stl(trickle(bytes.clone())).expect("tiny ASCII STL, one-byte reads");
    assert!(mesh.faces.is_empty());
    let mesh = read_stl(Cursor::new(bytes)).expect("tiny ASCII STL");
    assert!(mesh.faces.is_empty());
}
