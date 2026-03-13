//! GPU storage implementation using handle-based design.
//!
//! GpuStorage provides a lightweight handle-based interface for GPU-resident data.
//! The actual data lives in the GPU device runtime, while GpuStorage holds only
//! a handle, device_id, and shape metadata.

use cv_core::storage::{GpuStorageMarker, Storage};
use cv_core::BufferHandle;
use std::any::Any;
use std::fmt::Debug;

/// GPU-resident storage with lightweight handle-based design.
///
/// GpuStorage does not store data directly; instead it holds:
/// - A handle to identify the buffer in the GPU runtime
/// - A device_id to track which device owns this storage
/// - Shape metadata for multi-dimensional arrays
///
/// The actual data is managed by the GPU device runtime (e.g., wgpu).
#[derive(Debug, Clone)]
pub struct GpuStorage {
    /// Unique handle identifying this storage buffer
    handle: BufferHandle,
    /// Device ID where this storage resides
    device_id: u32,
    /// Multi-dimensional shape (empty for flat buffers)
    shape: Vec<usize>,
}

impl GpuStorage {
    /// Create a new GpuStorage with the given handle, device_id, and shape.
    pub fn new(handle: BufferHandle, device_id: u32, shape: Vec<usize>) -> Self {
        Self {
            handle,
            device_id,
            shape,
        }
    }

    /// Get the device ID where this storage resides.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Calculate the total number of elements based on shape.
    fn calculate_len(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

impl Storage<f32> for GpuStorage {
    fn handle(&self) -> BufferHandle {
        self.handle
    }

    fn capacity(&self) -> usize {
        self.calculate_len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        self.calculate_len()
    }

    fn data_type_name(&self) -> &'static str {
        "f32"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn device(&self) -> cv_core::storage::DeviceType {
        cv_core::storage::DeviceType::Cuda
    }
}

impl GpuStorageMarker for GpuStorage {
    type Element = f32;

    fn to_cpu(&self) -> std::result::Result<Vec<Self::Element>, String> {
        // In a real implementation, this would transfer data from GPU to CPU.
        // For now, return an error as we don't have GPU device access.
        Err("GPU to CPU transfer not implemented in this build".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_storage_creation() {
        let handle = BufferHandle(123);
        let device_id = 0;
        let shape = vec![2, 3, 4];

        let storage = GpuStorage::new(handle, device_id, shape.clone());

        assert_eq!(storage.handle(), handle);
        assert_eq!(storage.device_id(), device_id);
        assert_eq!(storage.shape(), &shape[..]);
    }

    #[test]
    fn test_gpu_storage_trait_impl() {
        let handle = BufferHandle(456);
        let device_id = 1;
        let shape = vec![10, 20];

        let storage = GpuStorage::new(handle, device_id, shape);

        // Verify Storage trait methods
        assert_eq!(storage.handle(), handle);
        assert_eq!(storage.capacity(), 200); // 10 * 20
        assert_eq!(storage.len(), 200);
        assert_eq!(storage.data_type_name(), "f32");
        assert!(!storage.is_empty());

        // Verify device type
        assert_eq!(storage.device(), cv_core::storage::DeviceType::Cuda);
    }

    #[test]
    fn test_gpu_storage_marker() {
        let handle = BufferHandle(789);
        let storage = GpuStorage::new(handle, 0, vec![100]);

        // Verify GpuStorageMarker trait
        let _result = storage.to_cpu();
        // Should return error since GPU device is not available in test
    }

    #[test]
    fn test_gpu_storage_multi_device() {
        let handle1 = BufferHandle(1);
        let handle2 = BufferHandle(2);

        let storage1 = GpuStorage::new(handle1, 0, vec![5, 5]);
        let storage2 = GpuStorage::new(handle2, 1, vec![5, 5]);

        // Different device IDs should be tracked
        assert_eq!(storage1.device_id(), 0);
        assert_eq!(storage2.device_id(), 1);

        // Same shape but different handles
        assert_ne!(storage1.handle(), storage2.handle());
        assert_eq!(storage1.len(), storage2.len());
    }
}
