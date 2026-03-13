//! CPU-based normal estimation operations
//!
//! Normal estimation operations include:
//! - Normal estimation using PCA on k-nearest neighbors
//! - Normal orientation consistency
//!
//! CPU normal estimation — re-exports from `cv-scientific`.
//!
//! For the public API prefer the top-level functions in `cv_point_cloud`
//! (e.g. [`cv_point_cloud::estimate_normals_cpu`]).
//!
//! | Function | Algorithm | Complexity |
//! |---|---|---|
//! | [`estimate_normals`] | R\*-tree kNN + analytic eigensolver | O(nk log n) |
//! | [`orient_normals`] | Neighbour-voting propagation | O(nk log n) |
//! | [`compute_normals_from_depth`] | Cross-product from structured depth | **O(n)** |

pub use cv_scientific::point_cloud::{
    compute_normals_from_depth, estimate_normals, orient_normals,
};
