use cv_core::CpuTensor;

use super::kcf::{KcfConfig, KcfTracker};
use super::mosse::MosseTracker;
use super::{BoundingBox, ObjectTracker, TrackerType};

/// Multi-object tracker that manages several independent single-object trackers.
///
/// Each tracked object is assigned a unique ID. Trackers that report `None`
/// (lost) are kept in the list so their IDs remain stable; use [`remove`](Self::remove)
/// to explicitly drop a tracker.
pub struct MultiObjectTracker {
    trackers: Vec<(usize, Box<dyn ObjectTracker>)>,
    next_id: usize,
}

impl MultiObjectTracker {
    /// Create an empty multi-object tracker.
    pub fn new() -> Self {
        Self {
            trackers: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a new object to track. Returns the assigned tracker ID.
    pub fn add(
        &mut self,
        frame: &CpuTensor<f32>,
        bbox: BoundingBox,
        tracker_type: TrackerType,
    ) -> usize {
        let mut tracker: Box<dyn ObjectTracker> = match tracker_type {
            TrackerType::Kcf => Box::new(KcfTracker::new(KcfConfig::default())),
            TrackerType::Mosse => Box::new(MosseTracker::new(0.125)),
        };
        tracker.init_tracker(frame, bbox);
        let id = self.next_id;
        self.next_id += 1;
        self.trackers.push((id, tracker));
        id
    }

    /// Update all trackers with a new frame.
    ///
    /// Returns a vector of `(id, Option<BoundingBox>)` for every active tracker.
    pub fn update(&mut self, frame: &CpuTensor<f32>) -> Vec<(usize, Option<BoundingBox>)> {
        self.trackers
            .iter_mut()
            .map(|(id, t)| (*id, t.update_tracker(frame)))
            .collect()
    }

    /// Remove a tracker by its ID.
    pub fn remove(&mut self, id: usize) {
        self.trackers.retain(|(tid, _)| *tid != id);
    }

    /// Number of active trackers.
    pub fn len(&self) -> usize {
        self.trackers.len()
    }

    /// Returns `true` if there are no active trackers.
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new()
    }
}
