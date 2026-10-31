//! The SLAM descriptor cache must not serve stale bytes.
//!
//! `WorldMap::descriptor_bytes` used to validate its cache by *length alone*.
//! `WorldMap::points` is a public field of `Arc<RwLock<MapPoint>>` and
//! `MapPoint::descriptor` is public, so any in-place edit - loop closure,
//! descriptor re-estimation, map point refinement - changes a descriptor without
//! changing the length.
//!
//! The consequence is not a stale read but a **wrong pose**. The tracker binds
//! this buffer as the map's descriptor tensor and treats a match's `train_idx`
//! as an index into it, so a stale buffer means `train_idx` identifies a
//! different landmark than the one the match was computed against, and
//! `point_indices.push(m.train_idx)` feeds the wrong world point into
//! `solve_pnp_ransac`. Nothing reports it.

use cv_slam::mapping::MapExt;
use cv_slam::{MapPoint, WorldMap};
use nalgebra::Point3;
use std::sync::{Arc, RwLock};

fn descriptor(seed: u8) -> Vec<u8> {
    vec![seed; 32]
}

fn map_with(n: u64) -> WorldMap {
    let mut map = WorldMap::default();
    for i in 0..n {
        map.add_point(MapPoint::new(
            i,
            Point3::new(i as f32, 0.0, 0.0),
            descriptor(i as u8),
        ));
    }
    map
}

#[test]
fn a_fresh_cache_is_reused() {
    let mut map = map_with(4);
    let first = map.descriptor_bytes();
    let second = map.descriptor_bytes();
    assert!(
        Arc::ptr_eq(&first, &second),
        "an unchanged map should reuse the cached buffer, not rebuild it"
    );
}

/// The defect. One descriptor edited in place, length unchanged, and the cache
/// handed back the old bytes.
#[test]
fn an_in_place_descriptor_edit_invalidates_the_cache() {
    let mut map = map_with(4);
    let before = map.descriptor_bytes();
    let stale_byte = before[32];

    // Edit a descriptor through the public field, exactly as loop closure or
    // re-estimation would. The length does not change.
    {
        let mut p = map.points[1].write().unwrap();
        p.descriptor = vec![0xAB; 32];
    }

    let after = map.descriptor_bytes();
    assert!(
        !Arc::ptr_eq(&before, &after),
        "the cache was reused after a descriptor changed in place"
    );

    // And the bytes must actually be the new ones.
    let slice = after.clone();
    assert_eq!(
        slice[32], 0xAB,
        "byte 32 should be the edited descriptor's first byte, not the stale \
         {stale_byte:#04x}"
    );
    assert_eq!(after[33], 0xAB);
}

/// Short descriptors are zero-padded, so the padding must be part of the
/// comparison too: a stale tail would otherwise pass a length-only check.
#[test]
fn a_short_descriptor_change_is_noticed() {
    let mut map = WorldMap::default();
    map.add_point(MapPoint::new(0, Point3::origin(), vec![1, 2, 3]));
    let first = map.descriptor_bytes();
    assert_eq!(first.len(), 32, "descriptors are padded to 32 bytes");
    assert_eq!(&first[3..], &[0u8; 29][..], "the tail must be zeroed");

    {
        let mut p = map.points[0].write().unwrap();
        // Same length, different content in the part that was padding.
        p.descriptor = vec![9, 9, 9, 9];
    }
    let second = map.descriptor_bytes();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(&second[..4], &[9, 9, 9, 9]);
    assert_eq!(&second[4..], &[0u8; 28][..]);
}

/// Appending a point already invalidated the cache, and must keep doing so.
#[test]
fn appending_still_invalidates() {
    let mut map = map_with(2);
    let first = map.descriptor_bytes();
    map.add_point(MapPoint::new(
        99,
        Point3::new(5.0, 5.0, 5.0),
        descriptor(0x7F),
    ));
    let second = map.descriptor_bytes();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.len(), 3 * 32);
}

/// A descriptor of a different length for the same point must invalidate too -
/// `min(32)` hides that difference from the byte comparison unless the padding
/// is checked, which it is.
#[test]
fn a_grown_descriptor_is_noticed() {
    let mut map = WorldMap::default();
    map.add_point(MapPoint::new(0, Point3::origin(), vec![7; 8]));
    let first = map.descriptor_bytes();
    {
        let mut p = map.points[0].write().unwrap();
        p.descriptor = vec![7; 32];
    }
    let second = map.descriptor_bytes();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second[8], 7, "the newly covered bytes must be present");
}
