//! A packed PLY `rgb` column must not be treated as a number.
//!
//! `read_ply` parses every body token with `f32::from_str` and rejects any
//! token that is not finite:
//!
//! ```text
//! if !v.is_finite() { return Err(Error::ParseError(format!("Non-finite coordinate: {s}"))) }
//! ```
//!
//! The exponent field of the literal is the top bits of the colour: the f32
//! exponent occupies bits 23..=30, which is the low seven bits of `R` followed
//! by the high bit of `G`. Verified by printing `f32::from_bits` and re-parsing
//! the text (which is exactly what the file format does):
//!
//! ```text
//! 0x00ff0000 -> 0.0000...023418052    finite -> red,               read correctly before
//! 0xff000000 -> -1.7014118e38         finite -> (0,0,0),           read correctly before
//! 0xff00ff00 -> -1.7146522e38         finite -> green,             read correctly before
//! 0x7f800000 -> inf          (R=127,G=128)                      REFUSED before
//! 0xff800000 -> -inf         (R=255,G=128)                      REFUSED before
//! 0x7fff0000 -> NaN          (R=127,G=255)                      REFUSED before
//! ```
//!
//! So the refusals are the colours whose top byte is `0x7F` with `G >= 0x80`, or
//! whose high bit is set with `G >= 0x80` and the `0x7F` pattern — a quarter of
//! the saturated red/magenta corner of the cube, hot pink among them.
//!
//! Two corrections to the boundary as originally reported, both established by
//! running the numbers rather than reading an exponent off a diagram:
//!
//! * The threshold is `G`'s top bit (bit 23), not `R`'s. `R >= 0x80` on its own
//!   is harmless: `0xff000000`, pure red, is `-1.7014118e38` and always read
//!   without complaint. Refusal needs `G >= 0x80` as well.
//! * `0xffff0000` cannot be recovered through the *text* of an ASCII file. A
//!   print always writes the bare word `NaN`, which parses back as `0x7fc00000`
//!   — sign and payload gone before the reader sees a byte. Such a colour is now
//!   read rather than refused, but it decodes to nonsense. Only a binary PLY
//!   could carry it, and the format this reader supports is ASCII only. Pinned
//!   below as the documented limit of the fix.
//!
//! `crates/io/src/pcd.rs` gets this right for both its ASCII and binary paths:
//! it parses the tokens, runs the finiteness check on the three coordinates
//! only, and bit-reinterprets the colour afterwards.

use std::io::BufReader;

/// A process-unique suffix.
///
/// The suite runs tests concurrently in one process, and a temp file left
/// behind by an aborted earlier run is indistinguishable from one just written.
/// Both have bitten this repo before: a shared filename has had tests
/// overwriting each other's cloud, and a later run read the stale file and
/// reported a colour failure that no longer existed.
fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let seq = N.fetch_add(1, Ordering::Relaxed);
    std::process::id() as u64 * 1_000_000 + seq
}

/// A one-vertex PLY whose colour column is the packed `rgb` float.
fn packed_rgb_ply(packed: u32) -> String {
    let literal = f32::from_bits(packed);
    format!(
        "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property uchar rgb
end_header
1 2 3 {literal}
"
    )
}

/// Read the single vertex back, or fail with a message that says why.
fn read_one(name: &str, text: &str) -> cv_core::PointCloud {
    let dir = std::env::temp_dir().join("cv_ply_packed_rgb_tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{}_{}.ply", name, unique_id()));
    std::fs::write(&path, text).expect("write the fixture");

    cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
        .unwrap_or_else(|e| panic!("{name}: read_ply failed: {e}\nthe file was:\n{text}"))
}

/// Assert a packed colour decodes to the r/g/b bytes it was built from.
fn assert_packed(name: &str, packed: u32) {
    let cloud = read_one(name, &packed_rgb_ply(packed));
    assert_eq!(cloud.len(), 1, "{name}: one vertex in, one point out");
    let c = cloud
        .colors
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: packed rgb 0x{packed:08x} produced no colours"))
        .clone();
    let want = [
        ((packed >> 16) & 0xFF) as f32 / 255.0,
        ((packed >> 8) & 0xFF) as f32 / 255.0,
        (packed & 0xFF) as f32 / 255.0,
    ];
    assert!(
        (c[0].x - want[0]).abs() < 1e-6
            && (c[0].y - want[1]).abs() < 1e-6
            && (c[0].z - want[2]).abs() < 1e-6,
        "{name}: packed 0x{packed:08x} (literal {:?}) read as {:?}, wanted {:?}",
        f32::from_bits(packed),
        c[0],
        want
    );
    assert_eq!(
        (cloud.points[0].x, cloud.points[0].y, cloud.points[0].z),
        (1.0, 2.0, 3.0),
        "{name}: the coordinates must be unaffected"
    );
}

/// CONTROL: a packed colour whose bit pattern happens to be an ordinary finite
/// float. If this failed, the test would be passing for the wrong reason.
#[test]
fn control_packed_colour_with_a_finite_bit_pattern_reads() {
    assert_packed("control", 0x00_7F_00_00); // exponent 127: -1.7014118e38
}

/// Pure red. `R = 0xFF` sets bit 31, whose exponent contribution makes the
/// literal `-inf`: the colour is exactly the value the finiteness check threw
/// away.
#[test]
fn pure_red_is_read() {
    assert_packed("red", 0xFF00_0000);
}

/// The documented boundary pair. `0xff800000` is `-inf` and is an honest
/// infinity (the sign bit rides on the packed value), so it round-trips bit-exact
/// and decodes to a real colour: `R = 0xFF`, `G = 0x80`.
#[test]
fn the_inf_literal_is_still_a_colour() {
    assert_packed("inf_literal", 0xFF80_0000);
}

/// CONTROL for the ASCII channel: `NaN` does NOT round-trip. Rust's `f32` has
/// one NaN payload and a print always writes the bare word `NaN`, so the sign
/// bit is dropped before the reader ever sees the file. Verified by running the
/// round trip this reader performs:
///
/// ```text
/// 0x7fff0000 -> prints "NaN" -> parses as 0x7fc00000 -> colour (R=0xC0)
/// 0xffff0000 -> prints "NaN" -> parses as 0x7fc00000 -> colour (R=0xC0)
/// ```
///
/// Both are nonsense as colours - the encoded `R` is lost entirely - and neither
/// is something the finiteness check can fix, because the information is already
/// gone by the time a byte of text exists. Only a binary PLY could carry it. This
/// pins that limit so a future change to the column handling cannot quietly
/// start *claiming* to recover these, and confirms such a token is now read at
/// all rather than refused.
#[test]
fn control_a_nan_literal_loses_its_payload_in_ascii() {
    for &packed in &[0x7FFF_0000u32, 0xFFFF_0000] {
        let cloud = read_one(&format!("nan_{packed:08x}"), &packed_rgb_ply(packed));
        let c = cloud.colors.as_ref().expect("colours were declared")[0];
        // 0x7fc00000: R = 0xC0, G = 0, B = 0.
        let want = [0xC0u8 as f32 / 255.0, 0.0, 0.0];
        assert!(
            (c.x - want[0]).abs() < 1e-6 && c.y == 0.0 && c.z == 0.0,
            "packed 0x{packed:08x} decoded as {c:?}, expected the payload-losing read \
             to give {want:?}"
        );
    }
}

/// Bit 23 is the low bit of the f32 exponent, and it is the high bit of `G`:
/// `G >= 0x80` is the first value that can push the literal to a non-finite one.
/// Both sides are pinned so the boundary cannot move.
#[test]
fn the_g_0x80_boundary_reads_on_both_sides() {
    // Highest G below the threshold: finite, so it worked before the fix.
    assert_packed("g_7f", 0x00_7F_00_00);
    // First G at the threshold: `inf`, refused before the fix.
    assert_packed("g_80", 0x00_80_00_00);
}

/// The grid sweep from the defect report: saturated corner combinations that the
/// finiteness check used to refuse. The `inf` literals are bit-exact round trips
/// and decode to the colour they encode; the `NaN` ones are read but cannot
/// survive ASCII, so they are pinned by what the reader can actually recover.
#[test]
fn the_failing_corner_grid_reads() {
    // (R=127, G=128) and (R=255, G=128): `inf` / `-inf`, bit-exact.
    assert_packed("grid_7f800000", 0x7F80_0000);
    assert_packed("grid_ff800000", 0xFF80_0000);
    // (R=127, G=128, B=255) and (R=127, G=255): both print as `NaN`. Accepted
    // rather than refused, which is the defect; the payload is unrecoverable.
    for &packed in &[0x7F80_00FFu32, 0x7FFF_0000] {
        let cloud = read_one(&format!("grid_{packed:08x}"), &packed_rgb_ply(packed));
        assert_eq!(cloud.len(), 1, "grid_{packed:08x} must reach the reader");
        assert!(
            cloud.colors.is_some(),
            "grid_{packed:08x} was refused instead of read"
        );
    }
}

/// Green and blue packs, which are ordinary finite literals but whose channels
/// also live in the top bytes of the value. They must be unaffected.
#[test]
fn green_and_blue_packs_read() {
    assert_packed("green", 0xFF00_FF00);
    assert_packed("blue", 0xFF00_00FF);
    assert_packed("magenta", 0x00FF_00FF);
}

/// CONTROL for the guard itself: a genuinely non-finite *coordinate* must still
/// be rejected. The fix moves the check to x/y/z only, and this is what proves
/// it did not simply delete the check.
#[test]
fn control_non_finite_coordinates_are_still_rejected() {
    for (name, body) in [
        ("nan_x", "nan 2 3 255 0 0"),
        ("inf_y", "1 inf 3 255 0 0"),
        ("neg_inf_z", "1 2 -inf 255 0 0"),
    ] {
        let text = format!(
            "ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
{body}
"
        );
        let dir = std::env::temp_dir().join("cv_ply_packed_rgb_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(format!("{}_{}.ply", name, unique_id()));
        std::fs::write(&path, text).expect("write the fixture");

        let err = cv_io::ply::read_ply(BufReader::new(std::fs::File::open(&path).unwrap()))
            .err()
            .unwrap_or_else(|| panic!("{name}: a non-finite coordinate must be rejected"));
        assert!(
            err.to_string().to_lowercase().contains("non-finite"),
            "{name}: the error should still say why, got: {err}"
        );
    }
}
