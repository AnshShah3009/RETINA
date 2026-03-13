use cv_core::slam::{MapPoint, WorldMap};
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
            if let Ok(p) = p_lock.read() {
                descs.push(Descriptor::new(p.descriptor.clone(), KeyPoint::default()));
            }
        }
        descs
    }

    fn add_point(&mut self, point: MapPoint) {
        self.points.push(Arc::new(RwLock::new(point)));
    }
}
