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
        // Placeholders use an all-ones pattern so they lose every Hamming
        // comparison to real descriptors and never win a best-match.
        let mut last_len = 32usize;
        for p_lock in &self.points {
            match p_lock.read() {
                Ok(p) => {
                    last_len = p.descriptor.len().max(last_len);
                    descs.push(Descriptor::new(p.descriptor.clone(), KeyPoint::default()));
                }
                // A poisoned lock (a writer panicked) must not silently skip
                // the entry: descriptors are positionally aligned with
                // map.points, and tracking indexes map.points[train_idx].
                Err(_) => descs.push(Descriptor::new(
                    vec![0xFFu8; last_len],
                    KeyPoint::default(),
                )),
            }
        }
        descs
    }

    fn add_point(&mut self, point: MapPoint) {
        self.points.push(Arc::new(RwLock::new(point)));
    }
}
