//! GPU-accelerated normal estimation operations
//!
//! GPU-based normal estimation with configurable compute modes

//! GPU-accelerated normal estimation — re-exports from `cv-3d`.
//!
//! For the public API prefer the top-level functions in `cv_point_cloud`
//! (e.g. [`cv_point_cloud::estimate_normals_gpu`]).
//!
//! All exact methods use the Open3D analytic 3×3 eigensolver.
//! All GPU shaders are WGSL: portable across Metal, Vulkan, DX12, and WebGPU.
//!
//! | Function | Backend | ~Speed (40k pts) |
//! |---|---|---|
//! | [`compute_normals`] | Morton sort (CPU) + GPU PCA | 17 ms |
//! | [`compute_normals_cpu`] | Voxel-hash (CPU, all cores) | 19 ms |
//! | [`compute_normals_hybrid`] | CPU kNN + GPU batch PCA | 20 ms |
//! | [`compute_normals_approx_cross`] | 2-neighbour cross-product | 10 ms |
//! | [`compute_normals_approx_integral`] | Ring cross-product average | 12 ms |

pub use cv_3d::gpu::point_cloud::{
    compute_normals, compute_normals_approx_cross, compute_normals_approx_integral,
    compute_normals_cpu, compute_normals_ctx, compute_normals_hybrid, compute_normals_simple,
};
