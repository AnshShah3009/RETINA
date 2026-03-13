//! Triangle Mesh Data Structure
//!
//! Re-exports TriangleMesh from cv-3d, which is the authoritative source.
//! This module preserves I/O-specific functionality while delegating
//! the core mesh type to cv-3d.

// Re-export TriangleMesh from cv-3d as the authoritative definition
pub use cv_3d::mesh::TriangleMesh;
