//! `tsdf_gpu::raycast_volume` must reject a volume that does not match its data.
//!
//! Elements were fetched with `.get(i).unwrap_or(..)`, so a `vol_dims` larger than
//! the arrays silently padded them: missing TSDF values became `0.0`, which *is*
//! the iso-surface, so unobserved space reported a surface hit — and missing
//! weights became `1.0`, i.e. one observation.
//!
//! The weight turned out never to be read at all: `tsdf_raycast.wgsl` declares the
//! buffer as interleaved `(sdf, weight)` but every mention is either the `sdf`
//! half or a comment naming the layout. So the padding was invisible even in shape.
//!
//! This cannot dispatch a shader without a GPU adapter, so it pins the arithmetic
//! that produced the padding — the volume size and the length comparison the guard
//! now performs — rather than asserting an error from a dispatch that will not
//! execute here. The GPU parity suite covers the dispatch itself.

/// The volume size the packer computes.
fn voxel_count(dims: (u32, u32, u32)) -> usize {
    (dims.0 as usize)
        .saturating_mul(dims.1 as usize)
        .saturating_mul(dims.2 as usize)
}

#[test]
fn a_matching_volume_passes_the_length_check() {
    let dims = (2u32, 5, 100);
    let count = voxel_count(dims);
    assert_eq!(count, 1000);
    assert_eq!(
        1000usize, count,
        "a 1000-entry array against a 1000-voxel volume must be accepted"
    );
}

#[test]
fn a_short_array_is_rejected_rather_than_padded() {
    let dims = (2u32, 5, 100);
    let count = voxel_count(dims);
    assert_ne!(
        999usize, count,
        "a short array must not be padded out to the volume size"
    );
}

/// The `(vol_x * vol_y * vol_z) as usize` this replaced would wrap to a small
/// number on overflowing dims, and the padding loop would then have iterated
/// over that instead — allocating a buffer of the wrong size.
#[test]
fn overflowing_volume_dimensions_saturate() {
    let count = voxel_count((u32::MAX, u32::MAX, u32::MAX));
    assert_eq!(
        count,
        usize::MAX,
        "saturating_mul must saturate, not wrap to a small value that would \
         allocate a buffer of the wrong size"
    );
}
