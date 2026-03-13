//! Point Cloud Operations — unified workspace crate.
//!
//! One import, every normal estimation method available directly.
//!
//! # Quick start
//!
//! ```ignore
//! use cv_point_cloud::*;
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
//! let normals = estimate_normals_approx_cross(&points);
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
//! | [`estimate_normals_approx_cross`] | fast approx | ~10 ms | real-time preview, ICP init |
//! | [`estimate_normals_approx_integral`] | smooth approx | ~12 ms | rendering, visualisation |
//! | [`estimate_normals_from_depth`] | exact (structured) | **< 1 ms** | RGBD / depth cameras |
//!
//! All exact methods use the **Open3D analytic 3×3 eigensolver**
//! (Geometric Tools `RobustEigenSymmetric3x3`): trigonometric eigenvalues +
//! best-cross-product eigenvector.  No iteration, always exact.
//!
//! GPU shaders are WGSL (WebGPU) — portable across Metal (Apple),
//! Vulkan (Linux/Android), DX12 (Windows), and WebGPU.

pub mod cpu;
pub mod filters;
pub mod gpu;
pub mod hidden_point_removal;

use nalgebra::{Point3, Vector3};

// ── Direct normal estimation functions ───────────────────────────────────────

/// Estimate normals, automatically selecting the fastest available backend.
///
/// Tries GPU (Morton sort + WebGPU PCA) first; falls back to
/// [`estimate_normals_cpu`] when no GPU context is available.
///
/// # Parameters
/// - `points`: Input point cloud (any coordinate system / units).
/// - `k`: Nearest-neighbour count. Typical: 10–15 (fast), 20–30 (high quality).
///
/// # Returns
/// One unit-length normal per input point.  Degenerate neighbourhoods → `(0,0,1)`.
///
/// # Example
/// ```ignore
/// let normals = estimate_normals_auto(&cloud, 15);
/// ```
pub fn estimate_normals_auto(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals(points, k)
}

/// Estimate normals on CPU using voxel-hash kNN + analytic eigensolver.
///
/// Recommended for CPU-only systems or clouds ≤ 100k points where GPU
/// transfer overhead outweighs compute gains.
///
/// Uses **aHash** (`hashbrown`) for O(1) voxel lookup and `select_nth_unstable`
/// (O(n) partial-select) instead of a full sort.  Rayon-parallel across all cores.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10–30).
///
/// # Performance
/// ~19 ms for 40k points on an 8-core CPU.
pub fn estimate_normals_cpu(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals_cpu(points, k, 0.0)
}

/// Estimate normals using Morton-sort (CPU) + batch PCA on GPU (WebGPU).
///
/// On Apple Silicon, the WebGPU backend uses Metal automatically.
/// Falls back to [`estimate_normals_cpu`] if no GPU context is available.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10–30).
///
/// # Performance
/// ~17 ms for 40k points on an Intel iGPU; faster on discrete GPU.
pub fn estimate_normals_gpu(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals(points, k)
}

/// Estimate normals: CPU voxel-hash kNN + GPU batch eigenvectors.
///
/// Splits work by hardware affinity:
/// - **CPU** handles kNN (spatially-irregular memory access — cache hierarchy wins).
/// - **GPU** handles eigenvectors (compute-bound, fully parallel — GPU wins).
///
/// Best for large clouds (> 100k points) on a discrete GPU.
/// Falls back to CPU-only on systems without GPU.
///
/// # Parameters
/// - `points`: Input point cloud.
/// - `k`: Nearest-neighbour count (typical: 10–30).
pub fn estimate_normals_hybrid(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals_hybrid(points, k)
}

/// Fast-approximate normals using the 2 nearest neighbours (cross-product method).
///
/// `normal = normalize(cross(v_nearest1, v_nearest2))`
///
/// No covariance matrix, no eigensolver — ~3× faster than exact PCA.
/// Suitable for real-time previews or as an initialiser for ICP.
///
/// # Accuracy
/// Good for smooth, uniformly sampled surfaces.  May produce noisy or flipped
/// normals near sharp edges or when the two nearest neighbours are nearly
/// co-linear with the query point.
pub fn estimate_normals_approx_cross(points: &[Point3<f32>]) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals_approx_cross(points, 0.0)
}

/// Fast-approximate normals by averaging cross-products from a ring of neighbours.
///
/// Accumulates `cross(n_i, n_{i+1})` over all neighbours in the local voxel,
/// then normalises the sum.  Produces smoother normals than
/// [`estimate_normals_approx_cross`] at similar speed (~2.5× faster than exact PCA).
///
/// # Use cases
/// Rendering, visualisation, and any application that values smoothness over
/// sharp-edge accuracy.
pub fn estimate_normals_approx_integral(points: &[Point3<f32>]) -> Vec<Vector3<f32>> {
    cv_3d::gpu::point_cloud::compute_normals_approx_integral(points, 0.0)
}

/// Estimate normals from a structured depth image (**O(n) — no spatial index**).
///
/// For each pixel, back-projects it and its four axis-aligned neighbours to 3-D,
/// then computes `normal = normalize(cross(right−left, down−up))`.
///
/// This is the correct method for any RGBD / depth-camera pipeline — it is
/// orders of magnitude faster than the k-NN PCA path for structured grids.
///
/// # Parameters
/// - `depth`: Row-major depth map, `H × W` elements (metric units).
/// - `width`, `height`: Image dimensions.
/// - `fx`, `fy`: Focal lengths in pixels.
/// - `cx`, `cy`: Principal point in pixels.
///
/// # Returns
/// Per-pixel normals in camera space, facing the viewer.
/// Border pixels and zero-depth pixels return `(0, 0, 1)`.
///
/// # Performance
/// ~0.5 ms for 320×240 (76k pixels) — ~134× faster than k-NN PCA.
///
/// # Example
/// ```ignore
/// // Typical RealSense / Oak-D intrinsics
/// let normals = estimate_normals_from_depth(
///     &depth, 640, 480,
///     615.0, 615.0,  // fx, fy
///     320.0, 240.0,  // cx, cy
/// );
/// ```
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
/// `estimate_normals_gpu`, etc.) — they are self-documenting and have zero overhead.
/// This struct is useful when you want to pass a configuration object around.
///
/// # Example
/// ```ignore
/// let cfg = NormalComputeConfig { k: 20, mode: ComputeMode::Gpu, ..Default::default() };
/// let normals = compute_normals(&points, &cfg);
/// ```
#[derive(Debug, Clone)]
pub struct NormalComputeConfig {
    /// Nearest-neighbour count.  `0` → library default (15).
    pub k: usize,
    /// Voxel size hint for spatial hash.  `0.0` → auto-estimated.
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
            cv_3d::gpu::point_cloud::compute_normals_cpu(points, k, cfg.voxel_size)
        }
        ComputeMode::Hybrid => estimate_normals_hybrid(points, k),
        ComputeMode::Gpu | ComputeMode::GPU | ComputeMode::Mlx => estimate_normals_gpu(points, k),
        ComputeMode::Auto | ComputeMode::Adaptive => estimate_normals_auto(points, k),
    }
}
