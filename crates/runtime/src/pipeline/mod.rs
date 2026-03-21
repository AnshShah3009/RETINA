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
                    name: _name,
                    inputs,
                    outputs,
                    params: _params,
                } => {
                    let output_sizes: Vec<_> = outputs
                        .iter()
                        .filter_map(|&id| self.buffers.get(&id).copied())
                        .collect();

                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        "Executing kernel: {} ({} inputs, {} outputs)",
                        _name,
                        inputs.len(),
                        output_sizes.len()
                    );

                    if !outputs.is_empty() {
                        for (i, &output_id) in outputs.iter().enumerate() {
                            if i < output_sizes.len() {
                                allocator.allocate_or_update(output_id, output_sizes[i], &[])?;
                            }
                        }
                    }

                    // Execute kernel via cv-hal dispatch
                    if let crate::device_registry::BackendContext::Gpu(gpu_ctx) =
                        device_runtime.context()
                    {
                        // Create tensors for all buffers (keep them alive for dispatch)
                        let mut all_tensors: Vec<Tensor<u8, cv_hal::storage::WgpuGpuStorage<u8>>> = Vec::new();

                        // Input tensors
                        for &input_id in inputs {
                            if let Some(tensor) = allocator.create_tensor(input_id) {
                                all_tensors.push(tensor);
                            }
                        }

                        // Output tensors
                        for &output_id in outputs {
                            if let Some(tensor) = allocator.create_tensor(output_id) {
                                all_tensors.push(tensor);
                            }
                        }

                        // Get references to tensors
                        let all_refs: Vec<&Tensor<u8, cv_hal::storage::WgpuGpuStorage<u8>>> =
                            all_tensors.iter().collect();

                        // Compute workgroup size based on largest buffer
                        let max_size = outputs
                            .iter()
                            .filter_map(|&id| allocator.get_buffer_alloc(id))
                            .map(|alloc| alloc.size)
                            .max()
                            .unwrap_or(1);

                        let workgroups = (
                            ((max_size as f64 / 64.0).ceil() as u32).max(1),
                            1,
                            1,
                        );

                        // Dispatch kernel - tensors stay alive through this call
                        gpu_ctx.dispatch(_name, &all_refs, _params, workgroups)?;

                        #[cfg(feature = "tracing")]
                        tracing::debug!(
                            "Dispatched kernel '{}' with {} buffers, workgroups {:?}",
                            _name,
                            all_refs.len(),
                            workgroups
                        );
                    } else {
                        return Err(crate::Error::NotSupported(
                            "Kernel execution only supported on GPU".into(),
                        ));
                    }
                }
                PipelineNode::CpuOp {
                    inputs,
                    outputs,
                    op,
                } => {
                    let input_data: Vec<Vec<u8>> = inputs
                        .iter()
                        .filter_map(|&id| allocator.get_buffer_data(id))
                        .collect();

                    let input_slices: Vec<&[u8]> =
                        input_data.iter().map(|v| v.as_slice()).collect();
                    let results = op(&input_slices);

                    for (i, &output_id) in outputs.iter().enumerate() {
                        if i < results.len() {
                            if let Some(size) = self.buffers.get(&output_id) {
                                allocator.allocate_or_update(output_id, *size, &results[i])?;
                            }
                        }
                    }
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

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub trait PipelineBuilder {
    fn build_pipeline(&self) -> Result<Pipeline>;
}
