//! 3D File I/O Module
//!
//! Supports reading and writing point clouds and meshes in various formats:
//! - PLY (Polygon File Format)
//! - OBJ (Wavefront Object)
//! - STL (STereoLithography)
//! - PCD (Point Cloud Data - PCL format)

pub mod mesh;
pub mod obj;
pub mod pcd;
pub mod ply;
pub mod stl;

pub use obj::{read_obj, write_obj, ObjMesh};
pub use pcd::{read_pcd, write_pcd, PcdData};
pub use ply::{read_ply, write_ply};
pub use stl::{read_stl, write_stl, write_stl_ascii, write_stl_binary};
// TriangleMesh is now the authoritative definition from cv-3d, re-exported here for convenience
pub use mesh::TriangleMesh;

pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type IoError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type IoResult<T> = cv_core::Result<T>;
