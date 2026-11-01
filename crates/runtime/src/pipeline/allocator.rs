use super::BufferId;
use crate::Result;
use crate::device_registry::{BackendContext, registry};
use cv_core::Tensor;
use cv_hal::DeviceId;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::WgpuGpuStorage;
use dashmap::DashMap;
use parking_lot::Mutex;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use wgpu::{Buffer, BufferUsages, Device};

/// Where the current contents of a pipeline buffer live.
#[derive(Default)]
struct Residency {
    /// Host copy. `Some` when the host holds the latest bytes (a CPU op wrote
    /// them, or a readback cached them).
    host: Option<Vec<u8>>,
    /// The GPU buffer holds the latest bytes (a kernel wrote them, or an
    /// upload copied the host bytes there).
    device_valid: bool,
}

/// One pipeline buffer.
///
/// On a GPU device it is backed by a `wgpu::Buffer`; on any other device it is
/// host memory only. The pool tracks which side holds the latest bytes and
/// copies between them on demand (`read_host`, `sync_to_device`).
pub struct BufferAlloc {
    /// GPU backing; `None` on CPU (and other non-wgpu) devices.
    pub buffer: Option<Arc<Buffer>>,
    pub size: usize,
    state: Mutex<Residency>,
}

impl BufferAlloc {
    /// Whether any node has written this buffer yet.
    pub fn is_written(&self) -> bool {
        let st = self.state.lock();
        st.host.is_some() || st.device_valid
    }
}

pub struct TransientBufferPool {
    device_id: DeviceId,
    /// The device's wgpu context, or `None` for a non-GPU device. Resolved on
    /// first use.
    gpu: OnceLock<Option<Arc<GpuContext>>>,
    buffers: DashMap<BufferId, Arc<BufferAlloc>>,
    free_lists: RwLock<HashMap<u64, Vec<Buffer>>>,
    total_allocated: RwLock<usize>,
}

fn align4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

fn unwritten(id: BufferId) -> crate::Error {
    crate::Error::RuntimeError(format!(
        "Pipeline buffer {:?} is read before any node wrote it",
        id
    ))
}

impl TransientBufferPool {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            gpu: OnceLock::new(),
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
            size.div_ceil(1024 * 1024) * 1024 * 1024
        }
    }

    /// The device's GPU context, `None` if the device is not a wgpu GPU.
    fn gpu(&self) -> Result<Option<Arc<GpuContext>>> {
        if let Some(g) = self.gpu.get() {
            return Ok(g.clone());
        }
        let reg = registry()?;
        let device_runtime = reg.get_device(self.device_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", self.device_id))
        })?;
        let g = match device_runtime.context() {
            BackendContext::Gpu(ctx) => Some(ctx.clone()),
            _ => None,
        };
        Ok(self.gpu.get_or_init(|| g).clone())
    }

    /// Whether this pool backs buffers with GPU memory.
    pub fn is_gpu(&self) -> Result<bool> {
        Ok(self.gpu()?.is_some())
    }

    pub fn allocate(&self, id: BufferId, size: usize) -> Result<Arc<BufferAlloc>> {
        let gpu = self.gpu()?;

        // Re-allocating an id replaces its buffer: release the old one first so
        // it returns to the free list and stops counting toward
        // `total_allocated` (previously both leaked on every replacement).
        self.release(id);

        // A non-GPU device gets host-only buffers: CPU-only pipelines used to
        // fail here with "TransientBufferPool only supports GPU".
        let buffer = match &gpu {
            Some(ctx) => Some(self.get_or_create_buffer(ctx.device(), size)?),
            None => None,
        };

        let alloc = Arc::new(BufferAlloc {
            buffer,
            size,
            state: Mutex::new(Residency::default()),
        });

        self.buffers.insert(id, alloc.clone());
        *self.total_allocated.write() += size;

        Ok(alloc)
    }

    /// Return the allocation for `id`, allocating `size` bytes if absent.
    pub fn ensure_allocated(&self, id: BufferId, size: usize) -> Result<Arc<BufferAlloc>> {
        match self.get_buffer_alloc(id) {
            Some(a) => Ok(a),
            None => self.allocate(id, size),
        }
    }

    fn alloc_of(&self, id: BufferId) -> Result<Arc<BufferAlloc>> {
        self.get_buffer_alloc(id).ok_or_else(|| unwritten(id))
    }

    /// Store host bytes as the buffer's new contents (the GPU copy, if any,
    /// becomes stale until `sync_to_device`). Allocates if needed.
    pub fn allocate_or_update(&self, id: BufferId, size: usize, data: &[u8]) -> Result<()> {
        self.ensure_allocated(id, size)?;
        self.write_host(id, data.to_vec())
    }

    /// Store host bytes as the buffer's new contents.
    pub fn write_host(&self, id: BufferId, data: Vec<u8>) -> Result<()> {
        let alloc = self.get_buffer_alloc(id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Pipeline buffer {:?} is not allocated", id))
        })?;
        if data.len() > alloc.size {
            return Err(crate::Error::RuntimeError(format!(
                "Pipeline buffer {:?}: {} bytes written into a {}-byte buffer",
                id,
                data.len(),
                alloc.size
            )));
        }
        let mut st = alloc.state.lock();
        st.host = Some(data);
        st.device_valid = false;
        Ok(())
    }

    /// The buffer's current contents on the host, reading them back from the
    /// GPU if a kernel produced them. Errors if nothing has written it.
    pub fn read_host(&self, id: BufferId) -> Result<Vec<u8>> {
        let alloc = self.alloc_of(id)?;
        let mut st = alloc.state.lock();
        if let Some(h) = &st.host {
            return Ok(h.clone());
        }
        if !st.device_valid {
            return Err(unwritten(id));
        }
        let (Some(gpu), Some(buffer)) = (self.gpu()?, alloc.buffer.as_ref()) else {
            return Err(crate::Error::RuntimeError(format!(
                "Pipeline buffer {:?} marked GPU-resident on a non-GPU device",
                id
            )));
        };
        let bytes = Self::readback(&gpu, buffer, alloc.size)?;
        st.host = Some(bytes.clone());
        Ok(bytes)
    }

    /// Make the GPU buffer hold the latest contents, uploading the host copy
    /// if it is newer. With `require_written`, a buffer nobody has written is
    /// an error (kernel inputs); without it, an unwritten buffer is left as is
    /// (kernel outputs).
    pub fn sync_to_device(&self, id: BufferId, require_written: bool) -> Result<()> {
        let alloc = self.alloc_of(id)?;
        let (Some(gpu), Some(buffer)) = (self.gpu()?, alloc.buffer.as_ref()) else {
            return Err(crate::Error::NotSupported(
                "Pipeline buffers live on the host on a non-GPU device".into(),
            ));
        };
        let mut st = alloc.state.lock();
        if st.device_valid {
            return Ok(());
        }
        match &st.host {
            Some(host) => {
                // `write_buffer` needs a 4-byte multiple. Zero-pad to the
                // whole (aligned) buffer so a short CPU result never leaves a
                // stale tail from a previous use of a pooled buffer.
                let mut padded = vec![0u8; align4(alloc.size)];
                padded[..host.len()].copy_from_slice(host);
                gpu.queue.write_buffer(buffer, 0, &padded);
                st.device_valid = true;
                Ok(())
            }
            None if require_written => Err(unwritten(id)),
            None => Ok(()),
        }
    }

    /// Record that a kernel wrote the buffer on the GPU: the host copy, if
    /// any, is now stale.
    pub fn mark_device_written(&self, id: BufferId) -> Result<()> {
        let alloc = self.alloc_of(id)?;
        let mut st = alloc.state.lock();
        st.host = None;
        st.device_valid = true;
        Ok(())
    }

    /// Copy `size` bytes of `buffer` to the host, blocking until the GPU work
    /// that writes it has finished.
    fn readback(gpu: &GpuContext, buffer: &Buffer, size: usize) -> Result<Vec<u8>> {
        let aligned = align4(size) as u64;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Readback"),
            size: aligned,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pipeline Readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, aligned);
        let submission = gpu.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(120)),
            })
            .map_err(|e| crate::Error::RuntimeError(format!("GPU readback poll: {:?}", e)))?;
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|_| crate::Error::RuntimeError("GPU readback map timed out".into()))?
            .map_err(|e| crate::Error::RuntimeError(format!("GPU readback map: {:?}", e)))?;

        let mut bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        bytes.truncate(size);
        Ok(bytes)
    }

    fn get_or_create_buffer(&self, device: &Device, size: usize) -> Result<Arc<Buffer>> {
        let bucket = Self::size_bucket(size);

        {
            let mut free_lists = self.free_lists.write();
            if let Some(pool) = free_lists.get_mut(&bucket)
                && let Some(buffer) = pool.pop()
            {
                return Ok(Arc::new(buffer));
            }
        }

        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        Ok(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transient Buffer"),
            size: bucket,
            usage: usages,
            mapped_at_creation: false,
        })))
    }

    /// The GPU buffer behind `id`; `None` if unallocated or on a CPU device.
    pub fn get_buffer(&self, id: BufferId) -> Option<Arc<Buffer>> {
        self.buffers.get(&id).and_then(|entry| entry.buffer.clone())
    }

    /// The host copy of `id`, if the host currently holds its latest bytes.
    /// Use [`read_host`](Self::read_host) to also fetch GPU-resident data.
    pub fn get_buffer_data(&self, id: BufferId) -> Option<Vec<u8>> {
        self.buffers
            .get(&id)
            .and_then(|entry| entry.state.lock().host.clone())
    }

    pub fn get_buffer_alloc(&self, id: BufferId) -> Option<Arc<BufferAlloc>> {
        self.buffers.get(&id).map(|entry| entry.clone())
    }

    /// A tensor view of the GPU buffer behind `id`; `None` if unallocated or
    /// on a CPU device.
    pub fn create_tensor(&self, id: BufferId) -> Option<Tensor<u8, WgpuGpuStorage<u8>>> {
        let alloc = self.buffers.get(&id)?;
        let buffer = alloc.buffer.clone()?;
        let storage = WgpuGpuStorage::from_buffer(buffer, alloc.size);
        Some(Tensor {
            storage,
            shape: cv_core::TensorShape {
                channels: 1,
                height: alloc.size,
                width: 1,
            },
            dtype: cv_core::DataType::U8,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn release(&self, id: BufferId) {
        if let Some((_, alloc)) = self.buffers.remove(&id) {
            let size = alloc.size;

            // Decrement total_allocated when releasing buffer
            {
                let mut total = self.total_allocated.write();
                *total = total.saturating_sub(size);
            }

            // Only return to pool if this is the last reference. Clone the
            // buffer out first, then drop the `BufferAlloc` (which also holds a
            // strong reference): otherwise the refcount can never reach zero,
            // `Arc::try_unwrap` always fails, and the buffer is dropped instead
            // of being pooled.
            let Some(buffer) = alloc.buffer.clone() else {
                return;
            };
            drop(alloc);
            if let Ok(raw_buffer) = Arc::try_unwrap(buffer) {
                let bucket = Self::size_bucket(size);
                let mut free_lists = self.free_lists.write();
                let pool = free_lists.entry(bucket).or_default();

                if pool.len() < 16 {
                    pool.push(raw_buffer);
                }
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

    /// Number of released buffers parked in the free lists for reuse.
    pub fn free_buffer_count(&self) -> usize {
        self.free_lists.read().values().map(Vec::len).sum()
    }

    pub fn trim(&self) {
        let mut free_lists = self.free_lists.write();
        for pool in free_lists.values_mut() {
            // `shrink_to` only lowers capacity, never length, so it freed no
            // buffers at all. Truncating drops the excess `wgpu::Buffer`s.
            pool.truncate(pool.len() / 2);
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

    fn gpu_pool() -> Option<TransientBufferPool> {
        let gpu = registry().ok()?.default_gpu()?;
        Some(TransientBufferPool::new(gpu.id()))
    }

    #[test]
    fn reallocating_an_id_releases_the_old_buffer() {
        let Some(pool) = gpu_pool() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        pool.allocate(BufferId(1), 1000).unwrap();
        pool.allocate(BufferId(1), 3000).unwrap();
        assert_eq!(pool.buffer_count(), 1);
        assert_eq!(
            pool.total_allocated(),
            3000,
            "the replaced 1000-byte allocation is still counted"
        );
        // The replaced buffer went back to its free list rather than leaking.
        assert_eq!(pool.free_buffer_count(), 1);
        pool.release_all();
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn trim_drops_half_of_each_free_list() {
        let Some(pool) = gpu_pool() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        for i in 0..8 {
            pool.allocate(BufferId(i), 512).unwrap();
        }
        pool.release_all();
        assert_eq!(pool.free_buffer_count(), 8);
        pool.trim();
        assert_eq!(pool.free_buffer_count(), 4, "trim freed nothing");
        pool.trim();
        assert_eq!(pool.free_buffer_count(), 2);
    }
}
