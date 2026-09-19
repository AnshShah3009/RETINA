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
