use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    DirectML,
    TensorRT,
    WebGPU,
    Mlx,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Vulkan => write!(f, "Vulkan"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::DirectML => write!(f, "DirectML"),
            BackendType::TensorRT => write!(f, "TensorRT"),
            BackendType::WebGPU => write!(f, "WebGPU"),
            BackendType::Mlx => write!(f, "MLX"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Compute,
    Simd,
    TensorCore,
    RayTracing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Compute,
    Transfer,
    Graphics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueId(pub u32);

/// Monotonically increasing submission index for tracking GPU operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SubmissionIndex(pub u64);

impl SubmissionIndex {
    /// Increment and return the new submission index.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

pub trait ComputeBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn name(&self) -> &str;
    fn device_id(&self) -> DeviceId;
    fn supports(&self, capability: Capability) -> bool;
    fn queue(&self, queue_type: QueueType) -> QueueId;
    fn preferred_queue(&self) -> QueueType;
}
