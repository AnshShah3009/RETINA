use nalgebra::Point2;

#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub angle: f64,
    pub response: f64,
    pub octave: i32,
    pub class_id: i32,
}

impl KeyPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            size: 1.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }

    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self
    }

    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    pub fn with_response(mut self, response: f64) -> Self {
        self.response = response;
        self
    }

    pub fn with_octave(mut self, octave: i32) -> Self {
        self.octave = octave;
        self
    }

    pub fn pt(&self) -> Point2<f64> {
        Point2::new(self.x, self.y)
    }

    pub fn scaled_pt(&self, scale: f64) -> Point2<f64> {
        Point2::new(self.x * scale, self.y * scale)
    }
}

impl Default for KeyPoint {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KeyPointF32 {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
    pub padding: i32,
}

impl From<KeyPoint> for KeyPointF32 {
    fn from(kp: KeyPoint) -> Self {
        Self {
            x: kp.x as f32,
            y: kp.y as f32,
            size: kp.size as f32,
            angle: kp.angle as f32,
            response: kp.response as f32,
            octave: kp.octave,
            class_id: kp.class_id,
            padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeatureMatch {
    pub query_idx: i32,
    pub train_idx: i32,
    pub distance: f32,
    pub img_idx: i32,
}

impl FeatureMatch {
    pub fn new(query_idx: i32, train_idx: i32, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            distance,
            img_idx: 0,
        }
    }

    pub fn with_img_idx(mut self, img_idx: i32) -> Self {
        self.img_idx = img_idx;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Matches {
    pub matches: Vec<FeatureMatch>,
    pub mask: Option<Vec<u8>>,
}

impl Matches {
    pub fn new() -> Self {
        Self {
            matches: Vec::new(),
            mask: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            matches: Vec::with_capacity(capacity),
            mask: None,
        }
    }

    pub fn push(&mut self, m: FeatureMatch) {
        self.matches.push(m);
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn filter_by_distance(&mut self, max_distance: f32) {
        self.matches.retain(|m| m.distance <= max_distance);
    }

    pub fn apply_ratio_test(&self, ratio_threshold: f32) -> Vec<FeatureMatch> {
        self.matches
            .iter()
            .filter(|m| m.distance <= ratio_threshold)
            .copied()
            .collect()
    }
}

impl Default for Matches {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct KeyPoints {
    pub keypoints: Vec<KeyPoint>,
}

impl KeyPoints {
    pub fn new() -> Self {
        Self {
            keypoints: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keypoints: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, kp: KeyPoint) {
        self.keypoints.push(kp);
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &KeyPoint> {
        self.keypoints.iter()
    }
}

impl Default for KeyPoints {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;

    #[test]
    fn keypoint_new_default_fields() {
        let kp = KeyPoint::new(1.0, 2.0);
        assert_eq!(kp.x, 1.0);
        assert_eq!(kp.y, 2.0);
        assert_eq!(kp.size, 1.0);
        assert_eq!(kp.angle, -1.0);
        assert_eq!(kp.response, 0.0);
        assert_eq!(kp.octave, 0);
        assert_eq!(kp.class_id, -1);
    }

    #[test]
    fn keypoint_builder_chain() {
        let kp = KeyPoint::new(0.0, 0.0)
            .with_size(5.0)
            .with_angle(45.0)
            .with_response(0.8)
            .with_octave(2);
        assert_eq!(kp.size, 5.0);
        assert_eq!(kp.angle, 45.0);
        assert_eq!(kp.response, 0.8);
        assert_eq!(kp.octave, 2);
    }

    #[test]
    fn keypoint_pt_returns_point2() {
        let kp = KeyPoint::new(3.5, 4.5);
        let pt = kp.pt();
        assert_eq!(pt.x, 3.5);
        assert_eq!(pt.y, 4.5);
    }

    #[test]
    fn keypoint_scaled_pt_applies_scale() {
        let kp = KeyPoint::new(4.0, 8.0);
        let scaled = kp.scaled_pt(2.0);
        assert_eq!(scaled.x, 8.0);
        assert_eq!(scaled.y, 16.0);
    }

    #[test]
    fn keypoint_scaled_pt_zero_scale() {
        let kp = KeyPoint::new(4.0, 8.0);
        let scaled = kp.scaled_pt(0.0);
        assert_eq!(scaled.x, 0.0);
        assert_eq!(scaled.y, 0.0);
    }

    #[test]
    fn keypoint_default() {
        let kp = KeyPoint::default();
        assert_eq!(kp.x, 0.0);
        assert_eq!(kp.y, 0.0);
        assert_eq!(kp.size, 1.0);
        assert_eq!(kp.octave, 0);
    }

    #[test]
    fn keypoint_f32_conversion() {
        let kp = KeyPoint::new(1.5, 2.5)
            .with_size(3.0)
            .with_angle(45.0)
            .with_response(0.9)
            .with_octave(1);
        let kp32: KeyPointF32 = kp.into();
        assert!((kp32.x - 1.5_f32).abs() < 1e-6);
        assert!((kp32.y - 2.5_f32).abs() < 1e-6);
        assert!((kp32.size - 3.0_f32).abs() < 1e-6);
    }

    #[test]
    fn feature_match_new() {
        let m = FeatureMatch::new(0, 1, 0.5);
        assert_eq!(m.query_idx, 0);
        assert_eq!(m.train_idx, 1);
        assert!((m.distance - 0.5).abs() < 1e-6);
        assert_eq!(m.img_idx, 0);
    }

    #[test]
    fn feature_match_with_img_idx() {
        let m = FeatureMatch::new(0, 1, 0.5).with_img_idx(3);
        assert_eq!(m.img_idx, 3);
    }

    #[test]
    fn matches_new_empty() {
        let m = Matches::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn matches_with_capacity() {
        let m = Matches::with_capacity(10);
        assert!(m.is_empty());
        assert_eq!(m.matches.capacity(), 10);
    }

    #[test]
    fn matches_push_and_len() {
        let mut m = Matches::new();
        m.push(FeatureMatch::new(0, 0, 0.5));
        assert_eq!(m.len(), 1);
        m.push(FeatureMatch::new(1, 1, 0.7));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn matches_filter_by_distance() {
        let mut m = Matches::new();
        m.push(FeatureMatch::new(0, 0, 0.3));
        m.push(FeatureMatch::new(1, 1, 0.5));
        m.push(FeatureMatch::new(2, 2, 0.8));
        m.filter_by_distance(0.5);
        assert_eq!(m.len(), 2);
        assert!(m.matches.iter().all(|match_| match_.distance <= 0.5));
    }

    #[test]
    fn matches_apply_ratio_test() {
        let mut m = Matches::new();
        m.push(FeatureMatch::new(0, 0, 0.3));
        m.push(FeatureMatch::new(1, 1, 0.8));
        let filtered = m.apply_ratio_test(0.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].distance, 0.3);
    }

    #[test]
    fn keypoints_new_empty() {
        let kps = KeyPoints::new();
        assert!(kps.is_empty());
        assert_eq!(kps.len(), 0);
    }

    #[test]
    fn keypoints_with_capacity() {
        let kps = KeyPoints::with_capacity(5);
        assert!(kps.is_empty());
        assert_eq!(kps.keypoints.capacity(), 5);
    }

    #[test]
    fn keypoints_push_and_len() {
        let mut kps = KeyPoints::new();
        kps.push(KeyPoint::new(1.0, 2.0));
        assert_eq!(kps.len(), 1);
        assert!(!kps.is_empty());
    }

    #[test]
    fn keypoints_iter() {
        let mut kps = KeyPoints::new();
        kps.push(KeyPoint::new(1.0, 2.0));
        kps.push(KeyPoint::new(3.0, 4.0));
        let collected: Vec<_> = kps.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].x, 1.0);
    }

    #[test]
    fn keypoints_default() {
        let kps = KeyPoints::default();
        assert!(kps.is_empty());
    }
}
