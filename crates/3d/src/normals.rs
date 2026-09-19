//! Unified normal estimation convenience functions.
//!
//! One import, every normal estimation method available directly.
//!
//! # Quick start
//!
//! ```ignore
//! use cv_3d::normals::*;
//! use nalgebra::Point3;
//!
//! let points: Vec<Point3<f32>> = /* your cloud */;
//!
//! // Auto: picks GPU if available, otherwise fast CPU
//! let normals = estimate_normals_auto(&points, 15);
//!
//! // Or choose the method that fits your needs:
//! let normals = estimate_normals_cpu(&points, 15);
//! let normals = estimate_normals_gpu(&points, 15);
//!
//! // For RGBD / depth cameras (fastest — O(n)):
//! let normals = estimate_normals_from_depth(&depth, 640, 480, fx, fy, cx, cy);
//! ```
//!
//! # Method Comparison
//!
//! | Function | Quality | Speed (40k pts) | Best for |
//! |---|---|---|---|
//! | [`estimate_normals_auto`] | exact | ~17 ms | general use |
//! | [`estimate_normals_cpu`] | exact | ~19 ms | CPU-only / no GPU |
//! | [`estimate_normals_gpu`] | exact | ~17 ms | GPU available |
//! | [`estimate_normals_hybrid`] | exact | ~20 ms | large clouds + discrete GPU |
//! | [`estimate_normals_approx_integral`] | smooth approx | ~12 ms | rendering, visualisation |
//! | [`estimate_normals_from_depth`] | exact (structured) | **< 1 ms** | RGBD / depth cameras |

use nalgebra::{Point3, Vector3};

// ── Direct normal estimation functions ───────────────────────────────────────

/// Estimate normals, automatically selecting the fastest available backend.
///
/// Tries GPU (Morton sort + WebGPU PCA) first; falls back to
/// [`estimate_normals_cpu`] when no GPU context is available.
///
/// # Parameters
/// - `points`: Input point cloud (any coordinate system / units).
/// - `k`: Nearest-neighbour count. Typical: 10-15 (fast), 20-30 (high quality).
///
/// # Returns
/// One unit-length normal per input point.  Degenerate neighbourhoods produce `(0,0,1)`.
pub fn estimate_normals_auto(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals(points, k)
}

/// Estimate normals on CPU using voxel-hash kNN + analytic eigensolver.
///
/// Recommended for CPU-only systems or clouds <= 100k points where GPU
/// transfer overhead outweighs compute gains.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10-30).
pub fn estimate_normals_cpu(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals_cpu(points, k, 0.0)
}

/// Estimate normals using Morton-sort (CPU) + batch PCA on GPU (WebGPU).
///
/// On Apple Silicon, the WebGPU backend uses Metal automatically.
/// Falls back to [`estimate_normals_cpu`] if no GPU context is available.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10-30).
pub fn estimate_normals_gpu(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals(points, k)
}

/// Estimate normals on CPU using KDTree + analytic eigensolver.
///
/// This is the classic approach used by Open3D and most 3D libraries.
/// O(n log n) tree construction, O(n log k) queries.
///
/// For uniformly distributed point clouds, [`estimate_normals_cpu`] (voxel hash)
/// is typically faster. For non-uniform clouds, KDTree may find better neighbors.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10-30).
pub fn estimate_normals_cpu_kdtree(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals_cpu_kdtree(points, k)
}

/// Estimate normals using tiled GPU brute-force kNN + analytic PCA.
///
/// Each workgroup cooperatively loads tiles of 256 points into shared memory,
/// giving 256x better memory bandwidth than naive global scans.
///
/// Complexity: O(n²) but with ~256x lower bandwidth cost.
/// Works well for small-medium clouds (1K-100K points).
///
/// Falls back to CPU voxel-hash if GPU is unavailable.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10-30, capped at 16 on GPU).
pub fn estimate_normals_gpu_tiled(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals_gpu_tiled(points, k)
}

/// Estimate normals: CPU voxel-hash kNN + GPU batch eigenvectors.
///
/// Splits work by hardware affinity:
/// - **CPU** handles kNN (spatially-irregular memory access -- cache hierarchy wins).
/// - **GPU** handles eigenvectors (compute-bound, fully parallel -- GPU wins).
///
/// Best for large clouds (> 100k points) on a discrete GPU.
/// Falls back to CPU-only on systems without GPU.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10-30).
pub fn estimate_normals_hybrid(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals_hybrid(points, k)
}

/// Fast-approximate normals by averaging cross-products from a ring of neighbours.
///
/// Accumulates `cross(n_i, n_{i+1})` over all neighbours in the local voxel,
/// then normalises the sum.  Produces smoother normals than
/// standard PCA at similar speed (~2.5x faster than exact PCA).
pub fn estimate_normals_approx_integral(points: &[Point3<f32>]) -> Vec<Vector3<f32>> {
    crate::gpu::point_cloud::compute_normals_approx_integral(points, 0.0)
}

/// Estimate normals from a structured depth image (**O(n) -- no spatial index**).
///
/// For each pixel, back-projects it and its four axis-aligned neighbours to 3-D,
/// then computes `normal = normalize(cross(right-left, down-up))`.
///
/// This is the correct method for any RGBD / depth-camera pipeline -- it is
/// orders of magnitude faster than the k-NN PCA path for structured grids.
///
/// # Parameters
/// - `depth`: Row-major depth map, `H x W` elements (metric units).
/// - `width`, `height`: Image dimensions.
/// - `fx`, `fy`: Focal lengths in pixels.
/// - `cx`, `cy`: Principal point in pixels.
///
/// # Returns
/// Per-pixel normals in camera space, facing the viewer.
/// Border pixels and zero-depth pixels return `(0, 0, 1)`.
pub fn estimate_normals_from_depth(
    depth: &[f32],
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Vec<Vector3<f32>> {
    cv_scientific::point_cloud::compute_normals_from_depth(depth, width, height, fx, fy, cx, cy)
}

// ── Config-based API (for users who prefer a settings struct) ─────────────────

/// Which compute backend to use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ComputeMode {
    /// Auto-selects GPU if available, otherwise CPU. Recommended default.
    #[default]
    Auto,
    /// CPU-only: voxel-hash kNN + analytic eigensolver.
    Cpu,
    /// GPU: Morton sort + WebGPU PCA (Metal on Apple Silicon).
    Gpu,
    /// Hybrid: CPU kNN + GPU batch eigenvectors.
    Hybrid,
    /// Apple Silicon explicit: tries Metal via wgpu, falls back to CPU.
    /// Functionally equivalent to `Gpu` on Apple Silicon.
    Mlx,
    // Legacy aliases kept for source compatibility.
    #[doc(hidden)]
    CPU,
    #[doc(hidden)]
    GPU,
    #[doc(hidden)]
    Adaptive,
}

/// Configuration struct for the config-based `compute_normals` dispatcher.
///
/// For most use-cases, prefer the direct functions (`estimate_normals_cpu`,
/// `estimate_normals_gpu`, etc.) -- they are self-documenting and have zero overhead.
/// This struct is useful when you want to pass a configuration object around.
#[derive(Debug, Clone)]
pub struct NormalComputeConfig {
    /// Nearest-neighbour count.  `0` means library default (15).
    pub k: usize,
    /// Voxel size hint for spatial hash.  `0.0` means auto-estimated.
    pub voxel_size: f32,
    /// Which backend to use.
    pub mode: ComputeMode,
}

impl Default for NormalComputeConfig {
    fn default() -> Self {
        Self::auto()
    }
}

impl NormalComputeConfig {
    /// Auto-selects the fastest available path. Recommended starting point.
    pub fn auto() -> Self {
        Self {
            k: 15,
            voxel_size: 0.0,
            mode: ComputeMode::Auto,
        }
    }
    /// CPU-only path.
    pub fn cpu() -> Self {
        Self {
            k: 15,
            voxel_size: 0.0,
            mode: ComputeMode::Cpu,
        }
    }
    /// GPU path (Morton sort + WebGPU).
    pub fn gpu() -> Self {
        Self {
            k: 15,
            voxel_size: 0.0,
            mode: ComputeMode::Gpu,
        }
    }
    /// Hybrid CPU kNN + GPU batch eigenvectors.
    pub fn hybrid() -> Self {
        Self {
            k: 15,
            voxel_size: 0.0,
            mode: ComputeMode::Hybrid,
        }
    }
    /// Apple Silicon (Metal via wgpu).
    pub fn mlx() -> Self {
        Self {
            k: 15,
            voxel_size: 0.0,
            mode: ComputeMode::Mlx,
        }
    }
    /// Fast approximate: k=10, CPU cross-product method.
    pub fn fast() -> Self {
        Self {
            k: 10,
            voxel_size: 0.0,
            mode: ComputeMode::Cpu,
        }
    }
    /// High quality: k=30 neighbours, auto backend.
    pub fn high_quality() -> Self {
        Self {
            k: 30,
            voxel_size: 0.0,
            mode: ComputeMode::Auto,
        }
    }
}

/// Config-based normal estimation dispatcher.
///
/// For most cases, prefer the direct functions (`estimate_normals_cpu` etc.).
/// This function is provided for code that passes a `NormalComputeConfig` around.
pub fn compute_normals(points: &[Point3<f32>], cfg: &NormalComputeConfig) -> Vec<Vector3<f32>> {
    let k = if cfg.k == 0 { 15 } else { cfg.k };
    match cfg.mode {
        ComputeMode::Cpu | ComputeMode::CPU => {
            crate::gpu::point_cloud::compute_normals_cpu(points, k, cfg.voxel_size)
        }
        ComputeMode::Hybrid => estimate_normals_hybrid(points, k),
        ComputeMode::Gpu | ComputeMode::GPU | ComputeMode::Mlx => estimate_normals_gpu(points, k),
        ComputeMode::Auto | ComputeMode::Adaptive => estimate_normals_auto(points, k),
    }
}
