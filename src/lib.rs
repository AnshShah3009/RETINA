pub use cv_3d as d3;
pub use cv_calib3d as calib3d;
pub use cv_core as core;
pub use cv_features as features;
pub use cv_hal as hal;
pub use cv_imgproc as imgproc;
pub use cv_optimize as optimize;
pub use cv_runtime as runtime;
pub use cv_sfm as sfm;
pub use cv_slam as slam;
pub use cv_video as video;

/// Initialize the entire cv-native library.
///
/// This performs:
/// 1. Global thread pool initialization.
/// 2. Asynchronous GPU context discovery.
pub async fn init() -> Result<(), String> {
    cv_core::init_global_thread_pool(None)?;
    // GPU init is best-effort — library works on CPU-only systems
    cv_hal::gpu::GpuContext::init_global()
        .await
        .map_err(|e| format!("GPU initialization failed: {}", e))?;
    Ok(())
}

/// Initialize a single global Rayon thread pool for all CPU-parallel routines.
pub fn init_thread_pool(num_threads: Option<usize>) -> Result<(), String> {
    cv_core::init_global_thread_pool(num_threads)
}
