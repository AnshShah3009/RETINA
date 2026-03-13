use crate::pipeline::NodeId;
use cv_hal::{DeviceId, SubmissionIndex};
use std::time::Instant;

/// Runtime events for observability and debugging
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// Device operation submitted
    SubmissionStarted {
        submission_id: SubmissionIndex,
        device_id: DeviceId,
        operation: String,
        timestamp: Instant,
    },
    /// Device operation completed
    SubmissionCompleted {
        submission_id: SubmissionIndex,
        device_id: DeviceId,
        duration_ms: u64,
        timestamp: Instant,
    },
    /// Pipeline node execution started
    PipelineNodeStarted {
        pipeline_id: u64,
        node_id: NodeId,
        node_name: String,
        timestamp: Instant,
    },
    /// Pipeline node execution completed
    PipelineNodeCompleted {
        pipeline_id: u64,
        node_id: NodeId,
        duration_ms: u64,
        timestamp: Instant,
    },
    /// Memory operation (allocation/release)
    MemoryEvent {
        kind: MemoryEventKind,
        size_bytes: usize,
        timestamp: Instant,
    },
    /// Device health change
    DeviceHealthChanged {
        device_id: DeviceId,
        state: DeviceHealth,
        reason: Option<String>,
        timestamp: Instant,
    },
}

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventKind {
    Allocated,
    Released,
    Synced,
}

/// Device health state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceHealth {
    Healthy,
    Degraded,
    Failed,
    Recovered,
}

impl RuntimeEvent {
    /// Get timestamp of event
    pub fn timestamp(&self) -> Instant {
        match self {
            RuntimeEvent::SubmissionStarted { timestamp, .. } => *timestamp,
            RuntimeEvent::SubmissionCompleted { timestamp, .. } => *timestamp,
            RuntimeEvent::PipelineNodeStarted { timestamp, .. } => *timestamp,
            RuntimeEvent::PipelineNodeCompleted { timestamp, .. } => *timestamp,
            RuntimeEvent::MemoryEvent { timestamp, .. } => *timestamp,
            RuntimeEvent::DeviceHealthChanged { timestamp, .. } => *timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submission_started_event() {
        let event = RuntimeEvent::SubmissionStarted {
            submission_id: SubmissionIndex(1),
            device_id: DeviceId(0),
            operation: "bilateral_filter".into(),
            timestamp: Instant::now(),
        };

        match event {
            RuntimeEvent::SubmissionStarted { operation, .. } => {
                assert_eq!(operation, "bilateral_filter");
            }
            _ => unreachable!("Wrong event type"),
        }
    }

    #[test]
    fn test_pipeline_node_event() {
        let event = RuntimeEvent::PipelineNodeStarted {
            pipeline_id: 42,
            node_id: NodeId(0),
            node_name: "kernel1".into(),
            timestamp: Instant::now(),
        };

        match event {
            RuntimeEvent::PipelineNodeStarted { pipeline_id, .. } => {
                assert_eq!(pipeline_id, 42);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_memory_event() {
        let event = RuntimeEvent::MemoryEvent {
            kind: MemoryEventKind::Allocated,
            size_bytes: 4096,
            timestamp: Instant::now(),
        };

        assert_eq!(event.timestamp().elapsed().as_secs(), 0);
    }

    #[test]
    fn test_device_health_event() {
        let event = RuntimeEvent::DeviceHealthChanged {
            device_id: DeviceId(0),
            state: DeviceHealth::Failed,
            reason: Some("CUDA out of memory".into()),
            timestamp: Instant::now(),
        };

        match event {
            RuntimeEvent::DeviceHealthChanged { state, .. } => {
                assert_eq!(state, DeviceHealth::Failed);
            }
            _ => panic!("Wrong event type"),
        }
    }
}
