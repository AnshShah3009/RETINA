mod file;
mod shared_memory;

pub use file::FileCoordinator;
pub use shared_memory::{ShmCoordinator, SHM_TOTAL_SIZE};

use cv_hal::DeviceId;
use std::collections::HashMap;

/// Default shared memory size for the v2 layout (18 KB).
pub const SHM_SIZE: usize = SHM_TOTAL_SIZE;

/// Trait for publishing and reading per-device load across cooperating processes.
pub trait LoadCoordinator: Send + Sync {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()>;

    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>>;

    fn cleanup(&self);

    fn register(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn heartbeat(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// Reserve memory on a device for this process (no-op by default).
    fn reserve_device(
        &self,
        _device_idx: u8,
        _memory_mb: u32,
        _compute_pct: u32,
    ) -> std::io::Result<()> {
        Ok(())
    }

    /// Release a device reservation held by this process (no-op by default).
    fn release_device(&self, _device_idx: u8) -> std::io::Result<()> {
        Ok(())
    }

    /// Join an affinity group (no-op by default).
    fn join_group(&self, _group_id: u32) -> std::io::Result<()> {
        Ok(())
    }

    /// Get per-device memory usage: `(device_idx, used_mb, total_mb)` (empty by default).
    fn device_memory_usage(&self) -> Vec<(u8, u32, u32)> {
        vec![]
    }

    /// Find the best device for this process considering affinity group peers (None by default).
    fn best_device_for_group(&self, _needed_mb: u32) -> Option<u8> {
        None
    }

    /// Initialize a device's total memory capacity (no-op by default).
    /// Only the first writer wins (CAS inside ShmCoordinator).
    fn init_device(&self, _device_idx: u8, _total_memory_mb: u32) -> std::io::Result<()> {
        Ok(())
    }

    /// Block until a device has at least `needed_mb` free, or `timeout` expires.
    /// Returns `Ok(())` when memory is available, `Err(TimedOut)` on timeout.
    /// Default implementation returns immediately (no budget tracking).
    fn wait_for_device_memory(
        &self,
        _device_idx: u8,
        _needed_mb: u32,
        _timeout: std::time::Duration,
    ) -> std::io::Result<()> {
        Ok(())
    }
}

pub enum CoordinatorType {
    File { path: std::path::PathBuf },
    SharedMemory { name: String, size: usize },
}

pub fn create_coordinator(
    coord_type: CoordinatorType,
) -> std::io::Result<Box<dyn LoadCoordinator>> {
    match coord_type {
        CoordinatorType::File { path } => Ok(Box::new(FileCoordinator::new(path))),
        CoordinatorType::SharedMemory { name, size } => {
            Ok(Box::new(ShmCoordinator::new(&name, size)?))
        }
    }
}

/// Auto-detects an appropriate coordinator based on environment variables.
///
/// Checks `CV_RUNTIME_COORDINATOR` (for file-based coordination) and `CV_RUNTIME_SHM`
/// (for shared-memory coordination), in that order.
///
/// # Important
/// This function creates a new coordinator instance each time it's called.
/// To avoid creating multiple coordinators for the same resource, callers
/// **MUST** cache the result using `OnceLock` or similar synchronization primitive.
pub fn auto_detect_coordinator() -> Option<Box<dyn LoadCoordinator>> {
    if let Ok(path) = std::env::var("CV_RUNTIME_COORDINATOR") {
        let path_buf = std::path::PathBuf::from(path);
        Some(Box::new(FileCoordinator::new(path_buf)))
    } else if let Ok(name) = std::env::var("CV_RUNTIME_SHM") {
        ShmCoordinator::new(&name, SHM_SIZE)
            .ok()
            .map(|c| Box::new(c) as Box<dyn LoadCoordinator>)
    } else {
        None
    }
}
