mod allocator;
mod fusion;
mod graph;
mod handle;

pub use allocator::{BufferAlloc, TransientBufferPool};
pub use fusion::{FusedKernel, FusionPattern, KernelFuser};
pub use graph::{ExecutionGraph, NodeDependency, NodeId};
pub use handle::{AsyncPipelineHandle, ExecutionEvent, PipelineResult};

use crate::device_registry::registry;
use crate::orchestrator::RuntimeRunner;
use crate::Result;
use cv_core::Tensor;
use cv_hal::context::ComputeContext;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

pub enum PipelineNode {
    Kernel {
        name: String,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        params: Vec<u8>,
    },
    CpuOp {
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        #[allow(clippy::type_complexity)]
        op: Arc<dyn Fn(&[&[u8]]) -> Vec<Vec<u8>> + Send + Sync>,
    },
    Barrier,
}

impl std::fmt::Debug for PipelineNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineNode::Kernel {
                name,
                inputs,
                outputs,
                params,
            } => f
                .debug_struct("Kernel")
                .field("name", name)
                .field("inputs", inputs)
                .field("outputs", outputs)
                .field("params", &format!("[{} bytes]", params.len()))
                .finish(),
            PipelineNode::CpuOp {
                inputs, outputs, ..
            } => f
                .debug_struct("CpuOp")
                .field("inputs", inputs)
                .field("outputs", outputs)
                .field("op", &"<function>")
                .finish(),
            PipelineNode::Barrier => write!(f, "Barrier"),
        }
    }
}

impl Clone for PipelineNode {
    fn clone(&self) -> Self {
        match self {
            PipelineNode::Kernel {
                name,
                inputs,
                outputs,
                params,
            } => PipelineNode::Kernel {
                name: name.clone(),
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                params: params.clone(),
            },
            PipelineNode::CpuOp {
                inputs,
                outputs,
                op,
            } => PipelineNode::CpuOp {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                op: op.clone(),
            },
            PipelineNode::Barrier => PipelineNode::Barrier,
        }
    }
}

pub struct Pipeline {
    nodes: Vec<PipelineNode>,
    buffers: HashMap<BufferId, usize>,
    preferred_device: Option<DeviceId>,
    execution_graph: Option<ExecutionGraph>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            buffers: HashMap::new(),
            preferred_device: None,
            execution_graph: None,
        }
    }

    pub fn with_device(mut self, device_id: DeviceId) -> Self {
        self.preferred_device = Some(device_id);
        self
    }

    pub fn register_buffer(&mut self, id: BufferId, size: usize) {
        self.buffers.insert(id, size);
    }

    pub fn add_kernel(
        mut self,
        name: &str,
        inputs: &[BufferId],
        outputs: &[BufferId],
        params: Vec<u8>,
    ) -> Self {
        self.nodes.push(PipelineNode::Kernel {
            name: name.to_string(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            params,
        });
        self
    }

    pub fn add_cpu_op<F>(mut self, inputs: &[BufferId], outputs: &[BufferId], op: F) -> Self
    where
        F: Fn(&[&[u8]]) -> Vec<Vec<u8>> + Send + Sync + 'static,
    {
        self.nodes.push(PipelineNode::CpuOp {
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            op: Arc::new(op),
        });
        self
    }

    pub fn add_barrier(mut self) -> Self {
        self.nodes.push(PipelineNode::Barrier);
        self
    }

    pub fn build(mut self) -> Result<Self> {
        self.execution_graph = Some(ExecutionGraph::build(&self.nodes)?);
        Ok(self)
    }

    pub fn execute(self, runner: &RuntimeRunner) -> Result<()> {
        let graph = self.execution_graph.as_ref().ok_or_else(|| {
            crate::Error::RuntimeError("Pipeline not built. Call .build() first.".into())
        })?;

        let device_id = self.preferred_device.unwrap_or_else(|| runner.device_id());
        let reg = registry()?;
        let device_runtime = reg.get_device(device_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", device_id))
        })?;

        let allocator = TransientBufferPool::new(device_id);

        for &node_id in &graph.topology_order {
            match &self.nodes[node_id.0] {
                PipelineNode::Kernel {
                    name,
                    inputs,
                    outputs,
                    params,
                } => {
                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        "Executing kernel: {} ({} inputs, {} outputs)",
                        name,
                        inputs.len(),
                        outputs.len()
                    );

                    let crate::device_registry::BackendContext::Gpu(gpu_ctx) =
                        device_runtime.context()
                    else {
                        return Err(crate::Error::NotSupported(
                            "Kernel execution only supported on GPU".into(),
                        ));
                    };

                    // Tensors stay alive through the dispatch call.
                    let tensors =
                        prepare_kernel_buffers(&allocator, &self.buffers, inputs, outputs)?;
                    let refs: Vec<&GpuTensor> = tensors.iter().collect();
                    let workgroups = kernel_workgroups(&self.buffers, outputs);
                    gpu_ctx.dispatch(name, &refs, params, workgroups)?;
                    finish_kernel(&allocator, outputs)?;

                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        "Dispatched kernel '{}' with {} buffers, workgroups {:?}",
                        name,
                        refs.len(),
                        workgroups
                    );
                }
                PipelineNode::CpuOp {
                    inputs,
                    outputs,
                    op,
                } => {
                    let input_data = gather_cpu_inputs(&allocator, &self.buffers, inputs)?;
                    let input_slices: Vec<&[u8]> =
                        input_data.iter().map(|v| v.as_slice()).collect();
                    let results = op(&input_slices);
                    store_cpu_outputs(&allocator, &self.buffers, outputs, results)?;
                }
                PipelineNode::Barrier => {
                    if let crate::device_registry::BackendContext::Gpu(gpu_ctx) =
                        device_runtime.context()
                    {
                        let _ = gpu_ctx.wait_idle();
                    }
                }
            }
        }

        Ok(())
    }

    pub fn execute_async(self, runner: &RuntimeRunner) -> AsyncPipelineHandle {
        let device_id = self.preferred_device.unwrap_or_else(|| runner.device_id());
        let graph = self.execution_graph.clone();
        let buffers = self.buffers.clone();
        let nodes = self.nodes.clone();

        handle::spawn_pipeline_execution(nodes, buffers, graph, device_id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

pub(crate) type GpuTensor = Tensor<u8, cv_hal::storage::WgpuGpuStorage<u8>>;

/// The registered size of `id`. An unregistered id is an error: silently
/// skipping it (the old `filter_map`) shifted every later CPU-op input and
/// every later kernel binding into the wrong slot.
fn registered_size(buffers: &HashMap<BufferId, usize>, id: BufferId) -> Result<usize> {
    buffers.get(&id).copied().ok_or_else(|| {
        crate::Error::RuntimeError(format!(
            "Pipeline buffer {:?} is used by a node but was never registered",
            id
        ))
    })
}

/// Get a kernel's buffers onto the GPU and return them as tensors in binding
/// order (inputs, then outputs).
///
/// Inputs must have been written by an earlier node; host-resident data (from
/// a CPU op) is uploaded first. Outputs are allocated, and uploaded too if a
/// CPU op left newer host data in them (read_write bindings).
pub(crate) fn prepare_kernel_buffers(
    pool: &TransientBufferPool,
    buffers: &HashMap<BufferId, usize>,
    inputs: &[BufferId],
    outputs: &[BufferId],
) -> Result<Vec<GpuTensor>> {
    for &id in inputs {
        pool.ensure_allocated(id, registered_size(buffers, id)?)?;
        pool.sync_to_device(id, true)?;
    }
    for &id in outputs {
        pool.ensure_allocated(id, registered_size(buffers, id)?)?;
        pool.sync_to_device(id, false)?;
    }
    inputs
        .iter()
        .chain(outputs)
        .map(|&id| {
            pool.create_tensor(id).ok_or_else(|| {
                crate::Error::RuntimeError(format!("Pipeline buffer {:?} has no GPU buffer", id))
            })
        })
        .collect()
}

/// One 1-D workgroup per 64 bytes of the largest output.
pub(crate) fn kernel_workgroups(
    buffers: &HashMap<BufferId, usize>,
    outputs: &[BufferId],
) -> (u32, u32, u32) {
    let max_size = outputs
        .iter()
        .filter_map(|id| buffers.get(id).copied())
        .max()
        .unwrap_or(1);
    (((max_size as f64 / 64.0).ceil() as u32).max(1), 1, 1)
}

/// After a dispatch: the kernel's outputs now live on the GPU.
pub(crate) fn finish_kernel(pool: &TransientBufferPool, outputs: &[BufferId]) -> Result<()> {
    outputs
        .iter()
        .try_for_each(|&id| pool.mark_device_written(id))
}

/// A CPU op's inputs, in order, on the host (read back from the GPU when a
/// kernel produced them). A missing or never-written input is an error.
pub(crate) fn gather_cpu_inputs(
    pool: &TransientBufferPool,
    buffers: &HashMap<BufferId, usize>,
    inputs: &[BufferId],
) -> Result<Vec<Vec<u8>>> {
    inputs
        .iter()
        .map(|&id| {
            registered_size(buffers, id)?;
            pool.read_host(id)
        })
        .collect()
}

/// Store a CPU op's results as the new host contents of its outputs.
pub(crate) fn store_cpu_outputs(
    pool: &TransientBufferPool,
    buffers: &HashMap<BufferId, usize>,
    outputs: &[BufferId],
    results: Vec<Vec<u8>>,
) -> Result<()> {
    if results.len() != outputs.len() {
        return Err(crate::Error::RuntimeError(format!(
            "CPU op declared {} outputs but returned {}",
            outputs.len(),
            results.len()
        )));
    }
    for (&id, data) in outputs.iter().zip(results) {
        pool.ensure_allocated(id, registered_size(buffers, id)?)?;
        pool.write_host(id, data)?;
    }
    Ok(())
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub trait PipelineBuilder {
    fn build_pipeline(&self) -> Result<Pipeline>;
}
