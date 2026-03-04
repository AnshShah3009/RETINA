use super::{BufferId, ExecutionGraph, PipelineNode, TransientBufferPool};
use crate::Error;
use crate::Result;
use cv_hal::context::ComputeContext;
use cv_hal::DeviceId;
use cv_hal::SubmissionIndex;
use std::collections::HashMap;
use tokio::sync::{broadcast, oneshot};

#[derive(Debug)]
pub struct PipelineResult {
    pub device_id: DeviceId,
    pub submissions: Vec<SubmissionIndex>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    Started {
        node_count: usize,
    },
    NodeCompleted {
        node_index: usize,
        node_name: String,
    },
    Completed {
        success: bool,
        error_msg: Option<String>,
    },
    Error {
        error: String,
    },
}

pub struct AsyncPipelineHandle {
    result_receiver: Option<oneshot::Receiver<Result<PipelineResult>>>,
    event_sender: broadcast::Sender<ExecutionEvent>,
    device_id: DeviceId,
}

impl AsyncPipelineHandle {
    pub(crate) fn new(
        result_receiver: oneshot::Receiver<Result<PipelineResult>>,
        event_sender: broadcast::Sender<ExecutionEvent>,
        device_id: DeviceId,
    ) -> Self {
        Self {
            result_receiver: Some(result_receiver),
            event_sender,
            device_id,
        }
    }

    /// Waits for the pipeline execution to complete.
    ///
    /// # Deprecated
    /// Use [`wait_for_completion`] instead, which is more idiomatic for handles.
    #[deprecated(
        since = "0.1.0",
        note = "Use `wait_for_completion` instead, which is more idiomatic for handles"
    )]
    pub async fn await_result(mut self) -> Result<PipelineResult> {
        let rx = self
            .result_receiver
            .take()
            .ok_or_else(|| Error::RuntimeError("Result already consumed".into()))?;

        rx.await
            .map_err(|_| Error::RuntimeError("Pipeline execution channel closed".into()))?
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ExecutionEvent> {
        self.event_sender.subscribe()
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub async fn wait_for_completion(&mut self) -> Result<PipelineResult> {
        let rx = self
            .result_receiver
            .take()
            .ok_or_else(|| Error::RuntimeError("Result already consumed".into()))?;

        rx.await
            .map_err(|_| Error::RuntimeError("Pipeline execution channel closed".into()))?
    }
}

pub(crate) fn spawn_pipeline_execution(
    nodes: Vec<PipelineNode>,
    buffers: HashMap<BufferId, usize>,
    graph: Option<ExecutionGraph>,
    device_id: DeviceId,
) -> AsyncPipelineHandle {
    let (result_tx, result_rx) = oneshot::channel();
    let (event_tx, _) = broadcast::channel(16);
    let event_tx_clone = event_tx.clone();

    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let node_count = nodes.len();

        let _ = event_tx_clone.send(ExecutionEvent::Started { node_count });

        let result = execute_pipeline(nodes, buffers, graph, device_id, &event_tx_clone).await;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(submissions) => {
                let pipeline_result = PipelineResult {
                    device_id,
                    submissions,
                    execution_time_ms,
                };
                let _ = event_tx_clone.send(ExecutionEvent::Completed {
                    success: true,
                    error_msg: None,
                });
                let _ = result_tx.send(Ok(pipeline_result));
            }
            Err(e) => {
                let err_str = e.to_string();
                let _ = event_tx_clone.send(ExecutionEvent::Error {
                    error: err_str.clone(),
                });
                let _ = event_tx_clone.send(ExecutionEvent::Completed {
                    success: false,
                    error_msg: Some(err_str),
                });
                let _ = result_tx.send(Err(e));
            }
        }
    });

    AsyncPipelineHandle::new(result_rx, event_tx, device_id)
}

async fn execute_pipeline(
    nodes: Vec<PipelineNode>,
    buffers: HashMap<BufferId, usize>,
    graph: Option<ExecutionGraph>,
    device_id: DeviceId,
    event_tx: &broadcast::Sender<ExecutionEvent>,
) -> Result<Vec<SubmissionIndex>> {
    let graph = match graph {
        Some(g) => g,
        None => ExecutionGraph::build(&nodes)?,
    };

    let reg = crate::device_registry::registry()?;
    let device_runtime = reg
        .get_device(device_id)
        .ok_or_else(|| Error::RuntimeError(format!("Device {:?} not found", device_id)))?;

    let allocator = TransientBufferPool::new(device_id);
    let mut submissions = Vec::new();

    for &node_id in &graph.topology_order {
        let node = &nodes[node_id.0];

        match node {
            PipelineNode::Kernel {
                name,
                inputs,
                outputs,
                params,
            } => {
                for &input_id in inputs {
                    if allocator.get_buffer(input_id).is_none() {
                        if let Some(&size) = buffers.get(&input_id) {
                            allocator.allocate(input_id, size)?;
                        }
                    }
                }

                for &output_id in outputs {
                    if let Some(&size) = buffers.get(&output_id) {
                        allocator.allocate(output_id, size)?;
                    }
                }

                let submission = device_runtime.next_submission();
                submissions.push(submission);

                #[cfg(feature = "tracing")]
                tracing::debug!("Executing kernel '{}' on device {:?}", name, device_id);

                let _ = (name, inputs, outputs, params);

                let _ = event_tx.send(ExecutionEvent::NodeCompleted {
                    node_index: node_id.0,
                    node_name: name.clone(),
                });
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

                let input_slices: Vec<&[u8]> = input_data.iter().map(|v| v.as_slice()).collect();
                let results = op(&input_slices);

                for (i, &output_id) in outputs.iter().enumerate() {
                    if i < results.len() {
                        if let Some(&size) = buffers.get(&output_id) {
                            allocator.allocate_or_update(output_id, size, &results[i])?;
                        }
                    }
                }

                let _ = event_tx.send(ExecutionEvent::NodeCompleted {
                    node_index: node_id.0,
                    node_name: "cpu_op".to_string(),
                });
            }
            PipelineNode::Barrier => {
                if let crate::device_registry::BackendContext::Gpu(gpu_ctx) =
                    device_runtime.context()
                {
                    gpu_ctx.wait_idle()?;
                }

                let _ = event_tx.send(ExecutionEvent::NodeCompleted {
                    node_index: node_id.0,
                    node_name: "barrier".to_string(),
                });
            }
        }
    }

    allocator.release_all();
    Ok(submissions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_handle_creation() {
        let nodes = vec![PipelineNode::Barrier];
        let buffers = HashMap::new();
        let graph = ExecutionGraph::build(&nodes).unwrap();
        let device_id = cv_hal::DeviceId(0);

        let handle = spawn_pipeline_execution(nodes, buffers, Some(graph), device_id);

        assert_eq!(handle.device_id(), device_id);
    }
}
