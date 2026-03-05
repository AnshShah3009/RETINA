use crate::gpu::GpuContext;
use cv_core::storage::{DeviceType, Storage};
use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU storage using wgpu buffers (legacy wgpu-based implementation).
///
/// This storage holds data on the GPU. Accessing it as a slice on the CPU is not supported directly.
/// Use `to_cpu()` to download data (not part of Storage trait).
///
/// Note: This is a legacy implementation. For new code, use cv_hal::GpuStorage which uses
/// a handle-based design.
#[derive(Clone)]
pub struct WgpuGpuStorage<T> {
    pub buffer: Option<Arc<wgpu::Buffer>>,
    pub len: usize,
    pub usage: wgpu::BufferUsages,
    _phantom: PhantomData<T>,
}

impl<T> WgpuGpuStorage<T> {
    pub fn from_buffer(buffer: Arc<wgpu::Buffer>, len: usize) -> Self {
        Self::from_buffer_with_usage(
            buffer,
            len,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
    }

    pub fn from_buffer_with_usage(
        buffer: Arc<wgpu::Buffer>,
        len: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            buffer: Some(buffer),
            len,
            usage,
            _phantom: PhantomData,
        }
    }

    /// Access the underlying buffer. Returns None if the buffer has been dropped.
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer
            .as_ref()
            .expect("WgpuGpuStorage buffer accessed after drop")
    }

    /// Try to access the underlying buffer. Returns None if the buffer has been dropped.
    pub fn try_buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_deref()
    }
}

impl<T> Drop for WgpuGpuStorage<T> {
    fn drop(&mut self) {
        if let Some(arc_buf) = self.buffer.take() {
            if let Ok(buffer) = Arc::try_unwrap(arc_buf) {
                if let Ok(ctx) = GpuContext::global() {
                    ctx.return_buffer(buffer, self.usage);
                }
            }
        }
    }
}

impl<T: bytemuck::Pod + fmt::Debug + Any + 'static> Storage<T> for WgpuGpuStorage<T> {
    fn handle(&self) -> cv_core::BufferHandle {
        // For legacy wgpu storage, use a dummy handle based on buffer pointer
        if let Some(buf) = &self.buffer {
            cv_core::BufferHandle(buf.as_ref() as *const _ as u64)
        } else {
            cv_core::BufferHandle(0)
        }
    }

    fn capacity(&self) -> usize {
        self.len
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn len(&self) -> usize {
        self.len
    }

    fn data_type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn device(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn as_slice(&self) -> Option<&[T]> {
        None
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T: bytemuck::Pod + fmt::Debug + Any + 'static> cv_core::storage::StorageFactory<T>
    for WgpuGpuStorage<T>
{
    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String> {
        let ctx = GpuContext::global()
            .map_err(|e| format!("Failed to get global GPU context: {:?}", e))?;
        let len = data.len();
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("StorageFactory buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Ok(Self {
            buffer: Some(Arc::new(buffer)),
            len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            _phantom: PhantomData,
        })
    }

    fn new(size: usize, default_value: T) -> std::result::Result<Self, String> {
        let data = vec![default_value; size];
        Self::from_vec(data)
    }
}

impl<T> fmt::Debug for WgpuGpuStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WgpuGpuStorage(len={})", self.len)
    }
}

impl<T> PartialEq for WgpuGpuStorage<T> {
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality for buffer
        match (&self.buffer, &other.buffer) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) && self.len == other.len,
            (None, None) => self.len == other.len,
            _ => false,
        }
    }
}

impl<T> Eq for WgpuGpuStorage<T> {}

// Type alias for backward compatibility
pub type GpuStorage<T> = WgpuGpuStorage<T>;
