use std::sync::Arc;
use wgpu::{CommandEncoder, Device, Queue};

/// A batch executor for GPU operations.
///
/// Instead of submitting work immediately, it collects commands and submits them
/// in a single batch to reduce driver overhead.
pub struct BatchExecutor {
    device: Arc<Device>,
    queue: Arc<Queue>,
    encoder: Option<CommandEncoder>,
    pending_ops: usize,
}

impl BatchExecutor {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batch Executor Encoder"),
        });

        Self {
            device,
            queue,
            encoder: Some(encoder),
            pending_ops: 0,
        }
    }

    /// Get the current command encoder to record a pass
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        // Re-create encoder if it was flushed/taken
        if self.encoder.is_none() {
            self.encoder = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Batch Executor Encoder (Refilled)"),
                },
            ));
        }
        self.encoder.as_mut().expect("BatchExecutor command encoder is missing - this should be impossible given self.encoder initialization")
    }

    /// Record an operation count (for statistics or thresholds)
    pub fn increment_pending(&mut self) {
        self.pending_ops += 1;
    }

    /// Submit all recorded commands
    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            if self.pending_ops > 0 {
                self.queue.submit(std::iter::once(encoder.finish()));
                self.pending_ops = 0;
            }
        }
    }
}

impl Drop for BatchExecutor {
    fn drop(&mut self) {
        self.flush();
    }
}
