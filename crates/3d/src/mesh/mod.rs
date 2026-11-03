//! Triangle mesh processing and reconstruction.
//!
//! The `TriangleMesh` type itself is defined in `cv-core` and re-exported here.

pub use cv_core::mesh::TriangleMesh;

pub mod processing;
pub mod reconstruction;
