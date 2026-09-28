use crate::KeyPoint;

#[derive(Debug, Clone)]
pub struct Descriptor {
    pub data: Vec<u8>,
    pub keypoint: KeyPoint,
}

impl Descriptor {
    pub fn new(data: Vec<u8>, keypoint: KeyPoint) -> Self {
        Self { data, keypoint }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Hamming distance between two descriptors.
    ///
    /// Descriptors of different lengths are not comparable, and `zip` would
    /// silently stop at the shorter one - so an 8-byte descriptor compared
    /// against a 32-byte one examines only the first 8 bytes and reports the
    /// distance between those, which can be 0 for descriptors that differ
    /// everywhere else. A caller comparing mixed-length descriptors would then
    /// get a confident "identical" match between things that are not the same
    /// kind of thing, and no ratio test can recover from it because the distance
    /// itself was wrong.
    ///
    /// A length mismatch therefore returns the maximum distance, which fails any
    /// ratio test and fails any threshold comparison. Every descriptor in the ORB
    /// pipeline is 32 bytes, so this cannot fire there; it is reachable if
    /// descriptors of different lengths are ever mixed, as BREIF's
    /// parameterised length allows.
    pub fn hamming_distance(&self, other: &Descriptor) -> u32 {
        if self.data.len() != other.data.len() {
            return u32::MAX;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct Descriptors {
    pub descriptors: Vec<Descriptor>,
}

impl Descriptors {
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            descriptors: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, desc: Descriptor) {
        self.descriptors.push(desc);
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Descriptor> {
        self.descriptors.iter()
    }
}

impl Default for Descriptors {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_new() {
        let kp = KeyPoint::new(1.0, 2.0);
        let d = Descriptor::new(vec![0b10101010u8, 0b11110000], kp);
        assert_eq!(d.data.len(), 2);
        assert_eq!(d.keypoint.x, 1.0);
    }

    #[test]
    fn descriptor_hamming_identical_is_zero() {
        let kp = KeyPoint::new(0.0, 0.0);
        let bytes = vec![0b10101010u8, 0b11110000, 0b00001111];
        let d = Descriptor::new(bytes.clone(), kp.clone());
        assert_eq!(d.hamming_distance(&d), 0);
    }

    #[test]
    fn descriptor_hamming_all_different_is_max() {
        let kp = KeyPoint::new(0.0, 0.0);
        let a = Descriptor::new(vec![0xFFu8; 4], kp.clone());
        let b = Descriptor::new(vec![0x00u8; 4], kp);
        assert_eq!(a.hamming_distance(&b), 32); // 4 bytes × 8 bits
    }

    #[test]
    fn descriptor_hamming_partial_overlap() {
        let kp = KeyPoint::new(0.0, 0.0);
        let a = Descriptor::new(vec![0b11110000u8], kp.clone());
        let b = Descriptor::new(vec![0b00001111u8], kp);
        assert_eq!(a.hamming_distance(&b), 8); // all 8 bits differ
    }

    #[test]
    fn descriptor_size() {
        let kp = KeyPoint::new(0.0, 0.0);
        let d = Descriptor::new(vec![0u8; 32], kp);
        assert_eq!(d.size(), 32);
    }

    #[test]
    fn descriptors_new_empty() {
        let ds = Descriptors::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn descriptors_with_capacity() {
        let ds = Descriptors::with_capacity(10);
        assert!(ds.is_empty());
        assert_eq!(ds.descriptors.capacity(), 10);
    }

    #[test]
    fn descriptors_push_and_len() {
        let mut ds = Descriptors::new();
        let kp = KeyPoint::new(0.0, 0.0);
        ds.push(Descriptor::new(vec![0u8; 8], kp));
        assert_eq!(ds.len(), 1);
        assert!(!ds.is_empty());
    }

    #[test]
    fn descriptors_iter() {
        let mut ds = Descriptors::new();
        let kp = KeyPoint::new(0.0, 0.0);
        ds.push(Descriptor::new(vec![0u8; 8], kp.clone()));
        ds.push(Descriptor::new(vec![1u8; 8], kp));
        let collected: Vec<_> = ds.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn descriptors_default() {
        let ds = Descriptors::default();
        assert!(ds.is_empty());
    }
}

#[cfg(test)]
mod hamming_length_tests {
    use super::*;
    use crate::keypoint::KeyPoint;

    fn desc(bytes: Vec<u8>) -> Descriptor {
        Descriptor::new(bytes, KeyPoint::new(0.0, 0.0))
    }

    /// Descriptors of different lengths must not compare as a short distance.
    ///
    /// `zip` stops at the shorter slice, so before this was guarded an 8-byte
    /// descriptor compared against a 32-byte one measured only the first 8
    /// bytes. Two descriptors differing in every one of their last 24 bytes came
    /// back at distance 0 - a confident "these are identical" for descriptors
    /// that are not even the same kind of thing, which no downstream ratio test
    /// can correct because the distance was already wrong.
    #[test]
    fn mismatched_lengths_are_maximally_distant() {
        let short = desc(vec![0x00; 8]);
        let long_same_prefix = desc({
            let mut v = vec![0x00u8; 32];
            v[31] = 0xFF; // differs only in the tail the old code never read
            v
        });

        assert_eq!(short.hamming_distance(&long_same_prefix), u32::MAX);
        assert_eq!(long_same_prefix.hamming_distance(&short), u32::MAX);
    }

    /// Equal lengths must be unaffected - the guard is not a shortcut.
    #[test]
    fn equal_lengths_still_measure_the_real_distance() {
        let a = desc(vec![0b1010_1010, 0xFF]);
        assert_eq!(a.hamming_distance(&a), 0);

        let mut b = a.data.clone();
        b[0] = 0b1010_0000; // two bits differ
        assert_eq!(a.hamming_distance(&desc(b)), 2);
    }
}
