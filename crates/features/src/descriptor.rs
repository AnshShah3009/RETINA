use cv_core::KeyPoints;
pub use cv_core::{Descriptor, Descriptors};
use image::GrayImage;

/// Trait for computing binary descriptors at a set of keypoints.
pub trait DescriptorExtractor {
    /// Extract descriptors for each keypoint in the image.
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors;
}
