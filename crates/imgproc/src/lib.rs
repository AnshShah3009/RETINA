pub mod bilateral;
pub mod color;
pub mod connected_components;
pub mod contours;
pub mod convolve;
pub mod distance_transform;
pub mod edges;
pub mod geometry;
pub mod histogram;
pub mod hough;
pub mod local_threshold;
pub mod morph;
pub mod morphology_advanced;
pub mod resize;
pub mod segmentation;
pub mod simd;
pub mod stitching;
pub mod template_matching;
pub mod threshold;

pub use bilateral::*;
pub use color::*;
pub use connected_components::*;
pub use contours::*;
pub use convolve::*;
pub use distance_transform::*;
pub use edges::*;
pub use geometry::*;
pub use histogram::*;
pub use local_threshold::*;
pub use morph::*;
pub use morphology_advanced::*;
pub use resize::*;
pub use template_matching::*;
pub use threshold::*;

pub type ImgprocError = cv_core::Error;
pub type Result<T> = cv_core::Result<T>;

pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(cv_core::Error::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}
