use crate::device_registry::SubmissionIndex;
use cv_hal::DeviceId;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferUsages, Device};

/// A buffer that is no longer needed but may still be in use by the GPU.
pub struct RetiredBuffer {
    pub buffer: Buffer,
    pub safe_after: SubmissionIndex,
}
/// Manages memory for a specific device, including buffer pooling and deferred destruction.
pub struct MemoryManager {
    device_id: DeviceId,
    device: Option<Arc<Device>>,
    retirement_queue: Mutex<Vec<RetiredBuffer>>,
}

impl MemoryManager {
    pub fn new(device_id: DeviceId, device: Option<Arc<Device>>) -> Self {
        Self {
            device_id,
            device,
            retirement_queue: Mutex::new(Vec::new()),
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Enqueue a buffer for retirement.
    ///
    /// The buffer will only be returned to the pool after the GPU has finished
    /// executing all commands up to `safe_after`.
    pub fn retire_buffer(&self, buffer: Buffer, safe_after: SubmissionIndex) {
        if let Ok(mut queue) = self.retirement_queue.lock() {
            queue.push(RetiredBuffer { buffer, safe_after });
        }
    }

    /// Reclaim retired buffers that are now safe to reuse.
    pub fn collect_garbage(&self, last_completed: SubmissionIndex) {
        let mut queue = match self.retirement_queue.lock() {
            Ok(q) => q,
            Err(_) => return,
        };

        // If we don't have a device, we can't have GPU buffers to return to a pool.
        let device = match &self.device {
            Some(d) => d,
            None => {
                queue.clear();
                return;
            }
        };

        // We use a simple filter here. For better performance with large queues,
        // we could use a more efficient data structure.
        let mut i = 0;
        while i < queue.len() {
            if last_completed >= queue[i].safe_after {
                let retired = queue.swap_remove(i);

                // Return to global pool for now.
                let usages =
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
                cv_hal::gpu_kernels::buffer_utils::global_pool().return_buffer(
                    device,
                    retired.buffer,
                    usages,
                );
            } else {
                i += 1;
            }
        }
    }

    /// Get a buffer from the pool.
    pub fn get_buffer(&self, device: &wgpu::Device, size: u64, usage: BufferUsages) -> Buffer {
        cv_hal::gpu_kernels::buffer_utils::global_pool().get(device, size, usage)
    }
}
