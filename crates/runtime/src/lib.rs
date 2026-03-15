pub mod device_registry;
pub mod distributed;
pub mod error;
pub mod executor;
pub mod memory;
pub mod memory_manager;
pub mod observe;
pub mod orchestrator;
pub mod pipeline;

pub use device_registry::{registry, DeviceRegistry, DeviceRuntime, SubmissionIndex};
pub use error::ErrorContext;
pub use memory::UnifiedBuffer;
pub use observe::{
    observability, DispatchBackend, DispatchReason, Metrics, ObservabilityLayer, RuntimeEvent,
};
pub use orchestrator::{
    best_runner, best_runner_gpu_wait, best_runner_gpu_wait_for, default_runner, record_fallback,
    scheduler, try_best_runner, try_default_runner, GroupPolicy, ResourceGroup, RuntimeRunner,
    TaskPriority, TaskScheduler, WorkloadHint,
};
pub use pipeline::{AsyncPipelineHandle, ExecutionEvent, PipelineResult};
pub use pipeline::{BufferAlloc, TransientBufferPool};
pub use pipeline::{BufferId, Pipeline, PipelineNode};
pub use pipeline::{ExecutionGraph, NodeDependency, NodeId};
pub use pipeline::{FusedKernel, FusionPattern, KernelFuser};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("HAL error: {0}")]
    HalError(#[from] cv_hal::Error),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),

    #[error("Initialization error: {0}")]
    InitError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! submit_to {
    ($group_name:expr, $f:block) => {
        match $crate::scheduler() {
            Ok(s) => s.submit($group_name, move || $f),
            Err(e) => Err(e),
        }
    };
}
