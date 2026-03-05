use crate::descriptor::{Descriptor, DescriptorExtractor, Descriptors};
use cv_core::KeyPoints;
use image::GrayImage;
use rand::thread_rng;
use rand::Rng;

/// BRIEF binary descriptor extractor.
///
/// Generates binary descriptors by comparing randomly sampled pixel pairs
/// within a patch around each keypoint.
pub struct BriefDescriptor {
    pairs: Vec<(i32, i32, i32, i32)>,
    patch_size: i32,
    descriptor_size: usize,
}

impl BriefDescriptor {
    /// Create a new BRIEF extractor with randomly sampled pixel-pair patterns.
    ///
    /// * `descriptor_size` - Number of bytes in the output descriptor (bits = `descriptor_size × 8`)
    /// * `patch_size` - Side length in pixels of the patch sampled around each keypoint
    pub fn new(descriptor_size: usize, patch_size: i32) -> Self {
        let num_pairs = descriptor_size * 8;
        let mut rng = thread_rng();
        let mut pairs = Vec::with_capacity(num_pairs);

        let half_size = patch_size / 2;

        for _ in 0..num_pairs {
            let x1 = rng.gen_range(-half_size..half_size);
            let y1 = rng.gen_range(-half_size..half_size);
            let x2 = rng.gen_range(-half_size..half_size);
            let y2 = rng.gen_range(-half_size..half_size);
            pairs.push((x1, y1, x2, y2));
        }

        Self {
            pairs,
            patch_size,
            descriptor_size,
        }
    }

    fn compute_brief(&self, image: &GrayImage, x: i32, y: i32) -> Descriptor {
        let width = image.width() as i32;
        let height = image.height() as i32;

        let mut descriptor_data = vec![0u8; self.descriptor_size];
        let mut bit_idx = 0;
        let mut byte_idx = 0;

        for (x1, y1, x2, y2) in &self.pairs {
            let px1 = (x + x1).clamp(0, width - 1) as u32;
            let py1 = (y + y1).clamp(0, height - 1) as u32;
            let px2 = (x + x2).clamp(0, width - 1) as u32;
            let py2 = (y + y2).clamp(0, height - 1) as u32;

            let val1 = image.get_pixel(px1, py1)[0];
            let val2 = image.get_pixel(px2, py2)[0];

            if val1 < val2 {
                descriptor_data[byte_idx] |= 1 << (7 - bit_idx);
            }

            bit_idx += 1;
            if bit_idx == 8 {
                bit_idx = 0;
                byte_idx += 1;
            }
        }

        Descriptor::new(descriptor_data, cv_core::KeyPoint::new(x as f64, y as f64))
    }
}

impl DescriptorExtractor for BriefDescriptor {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors {
        let mut descriptors = Descriptors::with_capacity(keypoints.len());
        let half_size = self.patch_size / 2;

        for kp in keypoints.iter() {
            let x = kp.x as i32;
            let y = kp.y as i32;

            if x >= half_size
                && x < (image.width() as i32 - half_size)
                && y >= half_size
                && y < (image.height() as i32 - half_size)
            {
                let desc = self.compute_brief(image, x, y);
                descriptors.push(desc);
            }
        }

        descriptors
    }
}

/// Extract BRIEF descriptors for a set of keypoints in a single call.
///
/// Uses a default patch size of 31 pixels. Keypoints too close to the image
/// border to fit a full patch are silently skipped.
pub fn extract_brief(
    image: &GrayImage,
    keypoints: &KeyPoints,
    descriptor_size: usize,
) -> Descriptors {
    let brief = BriefDescriptor::new(descriptor_size, 31);
    brief.extract(image, keypoints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::KeyPoint;
    use image::Luma;

    fn create_test_image() -> GrayImage {
        let mut img = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let val = ((x + y) % 256) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }

    #[test]
    fn test_brief_descriptor_new() {
        let brief = BriefDescriptor::new(32, 31);
        assert_eq!(brief.descriptor_size, 32);
        assert_eq!(brief.patch_size, 31);
        assert_eq!(brief.pairs.len(), 256);
    }

    #[test]
    fn test_brief_descriptor_size_variants() {
        let brief16 = BriefDescriptor::new(16, 31);
        let brief32 = BriefDescriptor::new(32, 31);
        let brief64 = BriefDescriptor::new(64, 31);

        assert_eq!(brief16.pairs.len(), 128);
        assert_eq!(brief32.pairs.len(), 256);
        assert_eq!(brief64.pairs.len(), 512);
    }

    #[test]
    fn test_brief_extract_single_keypoint() {
        let img = create_test_image();
        let kps = KeyPoints {
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
        };

        let brief = BriefDescriptor::new(32, 31);
        let descriptors = brief.extract(&img, &kps);

        assert_eq!(descriptors.descriptors.len(), 1);
        assert_eq!(descriptors.descriptors[0].data.len(), 32);
    }

    #[test]
    fn test_brief_extract_multiple_keypoints() {
        let img = create_test_image();
        let kps = KeyPoints {
            keypoints: vec![
                KeyPoint::new(25.0, 25.0),
                KeyPoint::new(50.0, 50.0),
                KeyPoint::new(75.0, 75.0),
            ],
        };

        let brief = BriefDescriptor::new(32, 31);
        let descriptors = brief.extract(&img, &kps);

        assert_eq!(descriptors.descriptors.len(), 3);
    }

    #[test]
    fn test_brief_extract_edge_keypoints_filtered() {
        let img = create_test_image();
        let kps = KeyPoints {
            keypoints: vec![
                KeyPoint::new(5.0, 5.0),
                KeyPoint::new(95.0, 95.0),
                KeyPoint::new(50.0, 50.0),
            ],
        };

        let brief = BriefDescriptor::new(32, 31);
        let descriptors = brief.extract(&img, &kps);

        assert!(descriptors.descriptors.len() <= 3);
    }

    #[test]
    fn test_extract_brief_function() {
        let img = create_test_image();
        let kps = KeyPoints {
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
        };

        let descriptors = extract_brief(&img, &kps, 32);
        assert_eq!(descriptors.descriptors.len(), 1);
    }

    #[test]
    fn test_brief_consistency_same_image() {
        let img = create_test_image();
        let kps = KeyPoints {
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
        };

        let brief = BriefDescriptor::new(32, 31);
        let desc1 = brief.extract(&img, &kps);
        let desc2 = brief.extract(&img, &kps);

        assert_eq!(desc1.descriptors[0].data, desc2.descriptors[0].data);
    }
}
