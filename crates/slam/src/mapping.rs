use crate::types::{MapPoint, WorldMap};
use cv_core::KeyPoint;
use cv_features::{Descriptor, Descriptors};
use std::sync::{Arc, RwLock};

pub trait MapExt {
    fn get_descriptors(&self) -> Descriptors;
    fn add_point(&mut self, point: MapPoint);
}

impl MapExt for WorldMap {
    fn get_descriptors(&self) -> Descriptors {
        let mut descs = Descriptors::new();
        for p_lock in &self.points {
            // Preserve one descriptor per map-point even when a poisoned lock
            // is encountered. Descriptor indices are used as map-point indices
            // by the tracker, so silently dropping a point corrupts matches.
            let p = p_lock
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            descs.push(Descriptor::new(p.descriptor.clone(), KeyPoint::default()));
        }
        descs
    }

    fn add_point(&mut self, point: MapPoint) {
        self.points.push(Arc::new(RwLock::new(point)));
        // The cache is now stale. The map only grows by appending, so comparing
        // the length is sufficient to detect that.
        self.descriptor_cache = None;
    }
}

impl WorldMap {
    /// Flattened descriptors for every map point, in `points` order.
    ///
    /// Returns a cached buffer when one is valid. The tracker binds this as the
    /// map's descriptor tensor, and used to rebuild it every frame - a
    /// `flat_map` clone of every descriptor in the map, O(total map points) per
    /// frame, for a map that only ever grows.
    pub fn descriptor_bytes(&mut self) -> Arc<Vec<u8>> {
        let n = self.points.len();
        if self.descriptor_cache_len == n {
            if let Some(cached) = &self.descriptor_cache {
                return Arc::clone(cached);
            }
        }
        let mut flat: Vec<u8> = Vec::with_capacity(n * 32);
        for p in &self.points {
            let d = p
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .descriptor
                .clone();
            let len = d.len().min(32);
            flat.extend_from_slice(&d[..len]);
            flat.resize(flat.len() + (32 - len), 0);
        }
        let arc = Arc::new(flat);
        self.descriptor_cache = Some(Arc::clone(&arc));
        self.descriptor_cache_len = n;
        arc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn descriptors_preserve_map_point_indices_when_lock_is_poisoned() {
        let mut map = WorldMap::new();
        map.add_point(MapPoint::new(0, Point3::origin(), vec![1; 32]));
        map.add_point(MapPoint::new(1, Point3::origin(), vec![2; 32]));

        let first = Arc::clone(&map.points[0]);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = first.write().expect("lock should be available");
            panic!("poison map-point lock");
        }));

        let descriptors = map.get_descriptors();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors.descriptors[0].data, vec![1; 32]);
        assert_eq!(descriptors.descriptors[1].data, vec![2; 32]);
    }
}

#[cfg(test)]
mod descriptor_cache_tests {
    use super::*;
    use crate::types::MapPoint;

    fn point(id: u64, byte: u8) -> MapPoint {
        MapPoint::new(id, nalgebra::Point3::new(0.0, 0.0, 0.0), vec![byte; 32])
    }

    /// The cache must return exactly what a fresh build would, and must notice a
    /// point being added.
    ///
    /// The tracker binds this buffer directly, so a stale cache would match
    /// against the wrong descriptors - the same silent-wrong-answer shape as the
    /// bugs this cache exists to fix.
    #[test]
    fn descriptor_bytes_matches_a_fresh_build_and_invalidates_on_add() {
        let mut map = WorldMap::new();
        map.add_point(point(0, 10));
        map.add_point(point(1, 20));

        let first = map.descriptor_bytes();
        assert_eq!(first.len(), 2 * 32, "two points of 32 bytes each");
        assert!(
            first[0] == 10 && first[32] == 20,
            "descriptors are in points order"
        );

        // A second call must return the same content.
        let second = map.descriptor_bytes();
        assert_eq!(*first, *second, "a warm cache must match a cold one");

        // Adding a point must invalidate it.
        map.add_point(point(2, 30));
        let third = map.descriptor_bytes();
        assert_eq!(third.len(), 3 * 32, "the cache grew stale after an add");
        assert!(
            third[64] == 30,
            "the new point's descriptor is missing from the rebuilt cache"
        );
    }
}
