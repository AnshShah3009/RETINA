use super::BufferId;
use crate::device_registry::registry;
use crate::Result;
use cv_hal::DeviceId;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device};

pub struct BufferAlloc {
    pub buffer: Buffer,
    pub size: usize,
    pub data: Option<Vec<u8>>,
}

pub struct TransientBufferPool {
    device_id: DeviceId,
    buffers: DashMap<BufferId, Arc<RwLock<BufferAlloc>>>,
    free_lists: RwLock<HashMap<u64, Vec<Buffer>>>,
    total_allocated: RwLock<usize>,
}

impl TransientBufferPool {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            buffers: DashMap::new(),
            free_lists: RwLock::new(HashMap::new()),
            total_allocated: RwLock::new(0),
        }
    }

    fn size_bucket(size: usize) -> u64 {
        let size = size as u64;
        if size <= 1024 * 1024 {
            size.next_power_of_two().max(256)
        } else {
            ((size + 1024 * 1024 - 1) / (1024 * 1024)) * 1024 * 1024
        }
    }

    pub fn allocate(&self, id: BufferId, size: usize) -> Result<Arc<RwLock<BufferAlloc>>> {
        let reg = registry()?;
        let device_runtime = reg.get_device(self.device_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", self.device_id))
        })?;

        let device = match device_runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => gpu_ctx.device(),
            _ => {
                return Err(crate::Error::NotSupported(
                    "TransientBufferPool only supports GPU".into(),
                ))
            }
        };

        let buffer = self.get_or_create_buffer(device, size)?;

        let alloc = Arc::new(RwLock::new(BufferAlloc {
            buffer,
            size,
            data: None,
        }));

        self.buffers.insert(id, alloc.clone());
        *self.total_allocated.write() += size;

        Ok(alloc)
    }

    pub fn allocate_or_update(&self, id: BufferId, size: usize, data: &[u8]) -> Result<()> {
        if !self.buffers.contains_key(&id) {
            self.allocate(id, size)?;
        }

        if let Some(entry) = self.buffers.get(&id) {
            let mut alloc = entry.write();
            alloc.data = Some(data.to_vec());
        }

        Ok(())
    }

    fn get_or_create_buffer(&self, device: &Device, size: usize) -> Result<Buffer> {
        let bucket = Self::size_bucket(size);

        {
            let mut free_lists = self.free_lists.write();
            if let Some(pool) = free_lists.get_mut(&bucket) {
                if let Some(buffer) = pool.pop() {
                    return Ok(buffer);
                }
            }
        }

        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transient Buffer"),
            size: bucket,
            usage: usages,
            mapped_at_creation: false,
        }))
    }

    pub fn get_buffer(&self, id: BufferId) -> Option<Buffer> {
        self.buffers
            .get(&id)
            .map(|entry| entry.read().buffer.clone())
    }

    pub fn get_buffer_data(&self, id: BufferId) -> Option<Vec<u8>> {
        self.buffers
            .get(&id)
            .and_then(|entry| entry.read().data.clone())
    }

    pub fn get_buffer_alloc(&self, id: BufferId) -> Option<Arc<RwLock<BufferAlloc>>> {
        self.buffers.get(&id).map(|entry| entry.clone())
    }

    pub fn release(&self, id: BufferId) {
        if let Some((_, alloc)) = self.buffers.remove(&id) {
            let alloc_guard = alloc.read();
            let buffer = alloc_guard.buffer.clone();
            let size = alloc_guard.size;
            drop(alloc_guard);

            // Decrement total_allocated when releasing buffer
            {
                let mut total = self.total_allocated.write();
                *total = total.saturating_sub(size);
            }

            let bucket = Self::size_bucket(size);
            let mut free_lists = self.free_lists.write();
            let pool = free_lists.entry(bucket).or_insert_with(Vec::new);

            if pool.len() < 16 {
                pool.push(buffer);
            }
        }
    }

    pub fn release_all(&self) {
        let ids: Vec<BufferId> = self.buffers.iter().map(|entry| *entry.key()).collect();
        for id in ids {
            self.release(id);
        }
    }

    pub fn total_allocated(&self) -> usize {
        *self.total_allocated.read()
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn trim(&self) {
        let mut free_lists = self.free_lists.write();
        for (_, pool) in free_lists.iter_mut() {
            pool.shrink_to(pool.len() / 2);
        }
    }
}

impl Drop for TransientBufferPool {
    fn drop(&mut self) {
        self.release_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bucket() {
        assert_eq!(TransientBufferPool::size_bucket(100), 256);
        assert_eq!(TransientBufferPool::size_bucket(1024), 1024);
        assert_eq!(TransientBufferPool::size_bucket(1025), 2048);
        assert_eq!(
            TransientBufferPool::size_bucket(2 * 1024 * 1024),
            2 * 1024 * 1024
        );
        assert_eq!(
            TransientBufferPool::size_bucket(2 * 1024 * 1024 + 1),
            3 * 1024 * 1024
        );
    }
}
