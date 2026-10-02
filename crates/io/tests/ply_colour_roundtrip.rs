//! PLY colour must round-trip, including at the dark end of the range.
//!
//! `write_ply` quantises to `uchar` (`(c * 255.0) as u8`) and declares
//! `property uchar red|green|blue`. `read_ply` normalised with a *threshold*
//! heuristic:
//!
//! ```text
//! let norm = |v: f32| if v > 1.0 { v / 255.0 } else { v };
//! ```
//!
//! That cannot tell a byte from an already-normalised float. The legitimate byte
//! value `1` is a near-black `1/255`, but it is also what a normalised `1.0`
//! looks like — and `1.0 > 1.0` is false, so it was passed through unchanged.
//! Measured through the real round trip: a vertex coloured `1/255` came back
//! `(1.0, 1.0, 1.0)`. Near-black became white, a factor of 255.
//!
//! Byte `0` is also ambiguous but harmless, since `0/255 == 0`. The two cases
//! most likely to mislead a reader are the ones pinned below.

use cv_core::PointCloud;
use nalgebra::Point3;

fn roundtrip(colour: [f32; 3]) -> [f32; 3] {
    // A distinct path per call: nextest runs tests concurrently in one process,
    // so a shared filename has the tests overwriting each other's file.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join("ply_colour_tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("rt_{n}.ply"));

    let points: Vec<Point3<f32>> = (0..3).map(|i| Point3::new(i as f32, 0.0, 0.0)).collect();
    let colors: Vec<Point3<f32>> = (0..3)
        .map(|_| Point3::new(colour[0], colour[1], colour[2]))
        .collect();
    let cloud = PointCloud {
        points,
        colors: Some(colors),
        normals: None,
    };

    {
        let mut f = std::fs::File::create(&path).expect("create");
        cv_io::ply::write_ply(&mut f, &cloud).expect("write_ply");
    }
    let f = std::fs::File::open(&path).expect("open");
    let back = cv_io::ply::read_ply(std::io::BufReader::new(f)).expect("read_ply");

    let c = back.colors.as_ref().expect("colours should round-trip");
    assert_eq!(c.len(), 3, "every vertex should keep its colour");
    [c[0].x, c[0].y, c[0].z]
}

fn assert_roundtrips(colour: [f32; 3], what: &str) {
    let got = roundtrip(colour);
    for (i, ch) in ["x", "y", "z"].iter().enumerate() {
        assert!(
            (got[i] - colour[i]).abs() < 0.01,
            "{what}: channel {ch} wrote and read back as {:.6}, expected {:.6} \
             (error {:.6})",
            got[i],
            colour[i],
            (got[i] - colour[i]).abs()
        );
    }
}

#[test]
fn primary_colours_round_trip() {
    assert_roundtrips([0.0, 0.0, 0.0], "black");
    assert_roundtrips([1.0, 1.0, 1.0], "white");
    assert_roundtrips([1.0, 0.0, 0.0], "red");
    assert_roundtrips([0.0, 1.0, 0.0], "green");
    assert_roundtrips([0.0, 0.0, 1.0], "blue");
}

/// The case that was broken. `1/255` is byte `1`, which the threshold read as a
/// normalised `1.0` — pure white.
#[test]
fn the_darkest_byte_value_does_not_become_white() {
    let one_byte = 1.0 / 255.0;
    assert_roundtrips([one_byte; 3], "1/255");
    assert_roundtrips([one_byte, 0.0, 0.0], "1/255 red only");

    // Two and three are outside the broken range, and pin that the fix is a
    // proper division rather than a special case for byte 1.
    assert_roundtrips([2.0 / 255.0; 3], "2/255");
    assert_roundtrips([3.0 / 255.0; 3], "3/255");
}

#[test]
fn interior_colours_round_trip() {
    assert_roundtrips([0.5, 0.5, 0.5], "mid grey");
    assert_roundtrips([1.0, 0.5, 0.0], "orange");
    assert_roundtrips([0.25, 0.75, 0.1], "mixed");
}

/// The control that makes the fix's scope explicit: a `float`-declared colour is
/// already 0..1 and must NOT be divided by 255.
#[test]
fn a_float_declared_colour_is_not_rescaled() {
    let dir = std::env::temp_dir().join("ply_colour_tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("float_colour_u8.ply");

    let header = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    let body = "0 0 0 255 128 0\n";
    std::fs::write(&path, format!("{header}{body}")).expect("write");

    let f = std::fs::File::open(&path).expect("open");
    let cloud = cv_io::ply::read_ply(std::io::BufReader::new(f)).expect("read");
    let c = cloud.colors.as_ref().expect("colours present")[0];
    assert!((c.x - 1.0).abs() < 0.01, "red channel: {}", c.x);
    assert!((c.y - 128.0 / 255.0).abs() < 0.01, "green channel: {}", c.y);
    assert!((c.z - 0.0).abs() < 0.01, "blue channel: {}", c.z);

    // Now the same file with `float` colour properties: 1.0 must stay 1.0, not
    // become 1/255. This is the branch the old threshold heuristic got wrong in
    // the other direction, for a third-party writer.
    let path2 = dir.join("float_colour_f32.ply");
    let header2 = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float red\nproperty float green\nproperty float blue\nend_header\n";
    // The body has to hold *normalised* values here: a `float` colour property
    // is already 0..1, so writing 255 into it declares a colour far outside the
    // representable range rather than a byte.
    let body2 = "0 0 0 1.0 0.502 0.0\n";
    std::fs::write(&path2, format!("{header2}{body2}")).expect("write");
    let f2 = std::fs::File::open(&path2).expect("open");
    let cloud2 = cv_io::ply::read_ply(std::io::BufReader::new(f2)).expect("read");
    let c2 = cloud2.colors.as_ref().expect("colours present")[0];
    assert!(
        (c2.x - 1.0).abs() < 0.01,
        "a float-declared red of 1.0 must stay 1.0, got {} - dividing it by 255 \
         would darken every float-coloured cloud a third-party writer produced",
        c2.x
    );
    assert!(
        (c2.y - 0.502).abs() < 0.01,
        "a float-declared green of 0.502 must stay 0.502, got {}",
        c2.y
    );
}
