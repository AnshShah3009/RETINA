use crate::device_registry::{registry, SubmissionIndex};
use crate::Result;
use cv_core::BufferHandle;
use cv_hal::DeviceId;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wgpu::BufferUsages;

/// Atomic counter for generating unique BufferHandle IDs for UnifiedBuffer.
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique BufferHandle ID.
#[inline]
fn generate_handle_id() -> u64 {
    NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst)
}

pub struct BufferSlice {
    pub offset: u64,
    pub size: u64,
}

pub struct UnifiedBuffer<T> {
    /// Unique handle for this buffer (for Storage trait integration)
    handle: BufferHandle,
    host_data: Arc<RwLock<Vec<T>>>,
    device_data: Option<wgpu::Buffer>,
    device_id: Option<DeviceId>,

    host_version: u64,
    device_version: Option<(DeviceId, u64)>,

    last_write_submission: SubmissionIndex,
    last_read_submission: SubmissionIndex,

    len: usize,

    slices: Vec<BufferSlice>,
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static + std::fmt::Debug> UnifiedBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            handle: BufferHandle(generate_handle_id()),
            host_data: Arc::new(RwLock::new(vec![T::default(); len])),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
            slices: Vec::new(),
        }
    }

    pub fn with_data(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            handle: BufferHandle(generate_handle_id()),
            host_data: Arc::new(RwLock::new(data)),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
            slices: Vec::new(),
        }
    }

    /// Get the unique handle for this buffer.
    ///
    /// The handle can be used for storage trait integration and tracking.
    pub fn get_handle(&self) -> BufferHandle {
        self.handle
    }

    /// Create a UnifiedBuffer from an existing CpuStorage.
    ///
    /// This enables integration with the cv-core Storage trait,
    /// allowing data to flow between different storage backends.
    ///
    /// # Arguments
    ///
    /// * `storage` - A CpuStorage<T> containing the initial data
    ///
    /// # Returns
    ///
    /// A new UnifiedBuffer with the data cloned from storage.
    pub fn from_storage(storage: cv_core::CpuStorage<T>) -> Self
    where
        T: Clone,
    {
        Self::with_data(storage.to_vec())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn host_view(&self) -> parking_lot::RwLockReadGuard<'_, Vec<T>> {
        self.host_data.read()
    }

    pub fn host_view_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Vec<T>> {
        self.host_data.write()
    }

    pub fn host_data_valid(&self) -> bool {
        if let Some((_, dv)) = self.device_version {
            self.host_version >= dv
        } else {
            true
        }
    }

    pub fn device_data_valid(&self) -> bool {
        self.device_version.is_some()
    }

    pub fn ensure_host_current(&self) -> Result<bool> {
        if let Some((_, dv)) = self.device_version {
            Ok(dv <= self.host_version)
        } else {
            Ok(true)
        }
    }

    pub fn mark_host_dirty(&mut self) {
        self.host_version += 1;
    }

    pub fn create_slice(&mut self, offset: usize, size: usize) -> Result<usize> {
        if offset + size > self.len {
            return Err(crate::Error::MemoryError(format!(
                "Slice [{}, {}) out of bounds for buffer of length {}",
                offset,
                offset + size,
                self.len
            )));
        }

        let slice_id = self.slices.len();
        self.slices.push(BufferSlice {
            offset: (offset * std::mem::size_of::<T>()) as u64,
            size: (size * std::mem::size_of::<T>()) as u64,
        });

        Ok(slice_id)
    }

    pub fn get_slice(&self, slice_id: usize) -> Option<&BufferSlice> {
        self.slices.get(slice_id)
    }

    pub fn sync_to_device(&mut self, target_id: DeviceId) -> Result<()> {
        if let Some((did, dv)) = self.device_version {
            if did == target_id && dv >= self.host_version {
                return Ok(());
            }
        }

        let reg = registry()?;
        let runtime = reg.get_device(target_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", target_id))
        })?;

        let host_guard = self.host_data.read();

        let size = (self.len * std::mem::size_of::<T>()) as u64;
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        let buffer = if let Some(ref b) = self.device_data {
            if b.size() >= size && self.device_id == Some(target_id) {
                b.clone()
            } else {
                if let Some(old_b) = self.device_data.take() {
                    if let Some(old_id) = self.device_id {
                        if let Some(old_runtime) = reg.get_device(old_id) {
                            old_runtime.memory().retire_buffer(
                                old_b,
                                self.last_write_submission.max(self.last_read_submission),
                            );
                        }
                    }
                }

                let device = match runtime.context() {
                    crate::device_registry::BackendContext::Gpu(ctx) => ctx.device(),
                    _ => {
                        return Err(crate::Error::NotSupported(
                            "sync_to_device only supports GPU".into(),
                        ))
                    }
                };

                runtime.memory().get_buffer(device, size, usages)
            }
        } else {
            let device = match runtime.context() {
                crate::device_registry::BackendContext::Gpu(ctx) => ctx.device(),
                _ => {
                    return Err(crate::Error::NotSupported(
                        "sync_to_device only supports GPU".into(),
                    ))
                }
            };

            runtime.memory().get_buffer(device, size, usages)
        };

        self.device_data = Some(buffer.clone());
        self.device_id = Some(target_id);

        match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                gpu_ctx
                    .queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(&host_guard));
            }
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "sync_to_device not implemented for {:?}",
                    runtime.backend()
                )))
            }
        }

        self.device_version = Some((target_id, self.host_version));
        Ok(())
    }

    pub fn sync_slice_to_device(&mut self, target_id: DeviceId, slice_id: usize) -> Result<()> {
        let (slice_offset, slice_size) = {
            let slice = self.slices.get(slice_id).ok_or_else(|| {
                crate::Error::MemoryError(format!("Invalid slice id {}", slice_id))
            })?;
            (slice.offset, slice.size)
        };

        if self.device_data.is_none() || self.device_id != Some(target_id) {
            self.sync_to_device(target_id)?;
        }

        let reg = registry()?;
        let runtime = reg.get_device(target_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", target_id))
        })?;

        let buffer = self
            .device_data
            .as_ref()
            .ok_or_else(|| crate::Error::MemoryError("Device buffer not allocated".into()))?;

        let host_guard = self.host_data.read();

        let elem_offset = (slice_offset as usize) / std::mem::size_of::<T>();
        let elem_count = (slice_size as usize) / std::mem::size_of::<T>();

        let slice_data = &host_guard[elem_offset..elem_offset + elem_count];

        match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                gpu_ctx
                    .queue
                    .write_buffer(buffer, slice_offset, bytemuck::cast_slice(slice_data));
            }
            _ => {
                return Err(crate::Error::NotSupported(
                    "sync_slice_to_device only supports GPU".into(),
                ))
            }
        }

        self.host_version += 1;
        self.device_version = Some((target_id, self.host_version));

        Ok(())
    }

    pub async fn sync_to_host(&mut self) -> Result<()> {
        let (did, dv) = self
            .device_version
            .ok_or_else(|| crate::Error::MemoryError("No device data to sync from".into()))?;

        if self.host_version >= dv {
            return Ok(());
        }

        let reg = registry()?;
        let runtime = reg
            .get_device(did)
            .ok_or_else(|| crate::Error::RuntimeError(format!("Device {:?} not found", did)))?;

        let buffer = self
            .device_data
            .as_ref()
            .ok_or_else(|| crate::Error::MemoryError("No device buffer to sync from".into()))?;

        let size = self.len * std::mem::size_of::<T>();

        let data: Vec<T> = match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                cv_hal::gpu_kernels::buffer_utils::read_buffer(
                    gpu_ctx.device.clone(),
                    &gpu_ctx.queue,
                    buffer,
                    0,
                    size,
                )
                .await?
            }
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "sync_to_host not implemented for {:?}",
                    runtime.backend()
                )))
            }
        };

        {
            let mut host_guard = self.host_data.write();
            *host_guard = data;
        }

        self.host_version = dv;
        Ok(())
    }

    pub fn sync_to_host_blocking(&mut self) -> Result<()> {
        let (did, dv) = self
            .device_version
            .ok_or_else(|| crate::Error::MemoryError("No device data to sync from".into()))?;

        if self.host_version >= dv {
            return Ok(());
        }

        let reg = registry()?;
        let runtime = reg
            .get_device(did)
            .ok_or_else(|| crate::Error::RuntimeError(format!("Device {:?} not found", did)))?;

        let buffer = self
            .device_data
            .as_ref()
            .ok_or_else(|| crate::Error::MemoryError("No device buffer to sync from".into()))?;

        let size = self.len * std::mem::size_of::<T>();

        let data: Vec<T> = match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                futures::executor::block_on(cv_hal::gpu_kernels::buffer_utils::read_buffer(
                    gpu_ctx.device.clone(),
                    &gpu_ctx.queue,
                    buffer,
                    0,
                    size,
                ))?
            }
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "sync_to_host not implemented for {:?}",
                    runtime.backend()
                )))
            }
        };

        {
            let mut host_guard = self.host_data.write();
            *host_guard = data;
        }

        self.host_version = dv;
        Ok(())
    }

    pub async fn map_async(&self) -> Result<Vec<T>> {
        if !self.host_data_valid() {
            return Err(crate::Error::MemoryError(
                "Host data is stale. Call sync_to_host() first.".into(),
            ));
        }

        let guard = self.host_data.read();
        Ok(guard.clone())
    }

    pub fn mark_device_write(&mut self, index: SubmissionIndex) {
        self.last_write_submission = index;
        if let Some((did, dv)) = self.device_version {
            self.device_version = Some((did, dv + 1));
        }
    }

    pub fn mark_device_read(&mut self, index: SubmissionIndex) {
        self.last_read_submission = index;
    }

    pub fn device_buffer(&self) -> Option<&wgpu::Buffer> {
        self.device_data.as_ref()
    }

    pub fn device_buffer_cloned(&self) -> Option<wgpu::Buffer> {
        self.device_data.clone()
    }

    pub fn device_id(&self) -> Option<DeviceId> {
        self.device_id
    }

    pub fn host_version(&self) -> u64 {
        self.host_version
    }

    pub fn device_version(&self) -> Option<(DeviceId, u64)> {
        self.device_version
    }

    pub fn into_host_data(self) -> Vec<T> {
        // Try to take ownership of the data without cloning if Arc is not shared.
        // Since the struct implements Drop, we need to use ManuallyDrop to prevent
        // double cleanup when we move the Arc out of self.
        use std::mem::ManuallyDrop;
        let this = ManuallyDrop::new(self);
        let arc = unsafe { std::ptr::read(&this.host_data) };
        match Arc::try_unwrap(arc) {
            Ok(lock) => lock.into_inner(),
            Err(arc) => arc.read().clone(),
        }
    }

    pub fn share(&self) -> Arc<RwLock<Vec<T>>> {
        self.host_data.clone()
    }
}

impl<T> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if let (Some(buffer), Some(id)) = (self.device_data.take(), self.device_id.take()) {
            if let Ok(reg) = registry() {
                if let Some(runtime) = reg.get_device(id) {
                    runtime.memory().retire_buffer(
                        buffer,
                        self.last_write_submission.max(self.last_read_submission),
                    );
                }
            }
        }
    }
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static + std::fmt::Debug> From<Vec<T>>
    for UnifiedBuffer<T>
{
    fn from(data: Vec<T>) -> Self {
        Self::with_data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_buffer_creation() {
        let buf: UnifiedBuffer<f32> = UnifiedBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.is_empty() == false);
    }

    #[test]
    fn test_unified_buffer_with_data() {
        let data = vec![1.0f32, 2.0, 3.0];
        let buf = UnifiedBuffer::with_data(data.clone());
        assert_eq!(buf.len(), 3);

        let view = buf.host_view();
        assert_eq!(*view, data);
    }

    #[test]
    fn test_host_view() {
        let buf: UnifiedBuffer<i32> = UnifiedBuffer::new(5);

        {
            let mut view = buf.host_view_mut();
            view[0] = 42;
        }

        let view = buf.host_view();
        assert_eq!(view[0], 42);
    }

    #[test]
    fn test_slice_creation() {
        let mut buf: UnifiedBuffer<f32> = UnifiedBuffer::new(100);

        let slice_id = buf.create_slice(10, 20).unwrap();
        assert_eq!(slice_id, 0);

        let slice = buf.get_slice(slice_id).unwrap();
        assert_eq!(slice.offset, 40);
        assert_eq!(slice.size, 80);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let mut buf: UnifiedBuffer<f32> = UnifiedBuffer::new(10);

        let result = buf.create_slice(5, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_mark_host_dirty() {
        let mut buf: UnifiedBuffer<f32> = UnifiedBuffer::new(10);
        let v1 = buf.host_version();

        buf.mark_host_dirty();
        assert!(buf.host_version() > v1);
    }

    // ===== Handle-Based Access Tests (Task 6) =====

    #[test]
    fn test_unified_buffer_get_handle() {
        let buf: UnifiedBuffer<f32> = UnifiedBuffer::new(10);
        let handle = buf.get_handle();

        // Handle should be non-zero (from our generation function)
        assert_ne!(handle.0, 0);
    }

    #[test]
    fn test_unified_buffer_handle_uniqueness() {
        let buf1: UnifiedBuffer<f32> = UnifiedBuffer::new(10);
        let buf2: UnifiedBuffer<f32> = UnifiedBuffer::new(10);

        let handle1 = buf1.get_handle();
        let handle2 = buf2.get_handle();

        // Each buffer should have a unique handle
        assert_ne!(handle1, handle2);
    }

    #[test]
    fn test_unified_buffer_from_storage() {
        let cpu_storage = cv_core::CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let buf = UnifiedBuffer::from_storage(cpu_storage);

        assert_eq!(buf.len(), 4);

        let view = buf.host_view();
        assert_eq!(*view, vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
