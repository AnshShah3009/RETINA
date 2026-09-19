//! Tests for Morton code encoding correctness.

use cv_hal::gpu_kernels::morton_encode;

#[test]
fn test_morton_encode_zero() {
    assert_eq!(morton_encode(0, 0, 0), 0);
}

#[test]
fn test_morton_encode_single_bit() {
    // x=1 should have bits at position 0, 3, 6, ...
    let code = morton_encode(1, 0, 0);
    assert_eq!(code, 1); // Bit 0

    // y=1 should have bits at position 1, 4, 7, ...
    let code = morton_encode(0, 1, 0);
    assert_eq!(code, 2); // Bit 1

    // z=1 should have bits at position 2, 5, 8, ...
    let code = morton_encode(0, 0, 1);
    assert_eq!(code, 4); // Bit 2
}

#[test]
fn test_morton_encode_no_axis_collision() {
    // Each axis should occupy different bit positions
    let code = morton_encode(7, 0, 0); // Only x bits
    assert_eq!(code & 0x12492492, 0); // y bits should be 0
    assert_eq!(code & 0x24924924, 0); // z bits should be 0

    let code = morton_encode(0, 7, 0); // Only y bits
    assert_eq!(code & 0x09249249, 0); // x bits should be 0
    assert_eq!(code & 0x24924924, 0); // z bits should be 0

    let code = morton_encode(0, 0, 7); // Only z bits
    assert_eq!(code & 0x09249249, 0); // x bits should be 0
    assert_eq!(code & 0x12492492, 0); // y bits should be 0
}

#[test]
fn test_morton_encode_all_axes() {
    // Encode all three axes with same value
    let code = morton_encode(1, 1, 1);
    // x bit 0, y bit 1, z bit 2
    assert_eq!(code, 7); // Bits 0, 1, 2 set
}

#[test]
fn test_morton_encode_max_10bit() {
    // Max 10-bit value = 1023 = 0x3FF
    let x = morton_encode(1023, 0, 0);
    let y = morton_encode(0, 1023, 0);
    let z = morton_encode(0, 0, 1023);

    // Combined should have all bits set in their respective positions
    let combined = morton_encode(1023, 1023, 1023);
    assert_eq!(combined, x | y | z);
}

#[test]
fn test_morton_encode_spatial_ordering() {
    // Points that are close in 3D should have close Morton codes
    let code1 = morton_encode(0, 0, 0);
    let code2 = morton_encode(1, 0, 0);
    let code3 = morton_encode(2, 0, 0);

    // Adjacent x values should have increasing Morton codes
    assert!(code1 < code2);
    assert!(code2 < code3);
}

#[test]
fn test_morton_encode_preserves_bits() {
    // Each encoded axis should preserve the original value in its bit pattern
    for val in [0, 1, 5, 10, 100, 500, 1023] {
        let code = morton_encode(val, 0, 0);

        // Extract x bits (bits 0, 3, 6, 9, ...)
        let mut x_bits = 0u32;
        for i in 0..10 {
            if code & (1 << (i * 3)) != 0 {
                x_bits |= 1 << i;
            }
        }
        assert_eq!(x_bits, val as u32, "x value {} not preserved", val);
    }
}

#[test]
fn test_morton_encode_symmetry() {
    // Encoding x,y,z should be symmetric
    let code1 = morton_encode(3, 5, 7);
    let code2 = morton_encode(3, 5, 7);
    assert_eq!(code1, code2);
}

#[test]
fn test_morton_encode_different_coordinates() {
    // Different coordinates should produce different codes
    let codes = [
        morton_encode(0, 0, 0),
        morton_encode(1, 0, 0),
        morton_encode(0, 1, 0),
        morton_encode(0, 0, 1),
        morton_encode(1, 1, 0),
        morton_encode(1, 0, 1),
        morton_encode(0, 1, 1),
        morton_encode(1, 1, 1),
    ];

    // All codes should be unique
    for i in 0..codes.len() {
        for j in (i + 1)..codes.len() {
            assert_ne!(codes[i], codes[j], "Collision at ({:?})", (i, j));
        }
    }
}
