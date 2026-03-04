use cv_hal::{DeviceId, SubmissionIndex};
use std::time::{Instant, Duration};
use crate::ErrorContext;

/// Tracks lifecycle of a single submission (GPU operation or pipeline node)
/// Automatically publishes completion event on drop
#[derive(Debug)]
pub struct SubmissionContext {
    id: SubmissionIndex,
    device_id: DeviceId,
    operation: String,
    created_at: Instant,
    completed_at: Option<Instant>,
    error: Option<ErrorContext>,
}

impl SubmissionContext {
    /// Create new submission context
    pub fn new(id: SubmissionIndex, device_id: DeviceId, operation: impl Into<String>) -> Self {
        Self {
            id,
            device_id,
            operation: operation.into(),
            created_at: Instant::now(),
            completed_at: None,
            error: None,
        }
    }

    /// Mark submission as completed
    pub fn mark_completed(&mut self) -> Duration {
        self.completed_at = Some(Instant::now());
        self.duration().unwrap_or(Duration::ZERO)
    }

    /// Mark submission with error
    pub fn mark_error(&mut self, error: ErrorContext) {
        self.error = Some(error);
        self.completed_at = Some(Instant::now());
    }

    /// Get duration if completed
    pub fn duration(&self) -> Option<Duration> {
        self.completed_at.map(|end| end.duration_since(self.created_at))
    }

    /// Check if completed
    pub fn is_completed(&self) -> bool {
        self.completed_at.is_some()
    }

    /// Check if has error
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get error if any
    pub fn error(&self) -> Option<&ErrorContext> {
        self.error.as_ref()
    }

    /// Get submission ID
    pub fn id(&self) -> SubmissionIndex {
        self.id
    }

    /// Get device ID
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Get operation name
    pub fn operation(&self) -> &str {
        &self.operation
    }
}

impl Drop for SubmissionContext {
    fn drop(&mut self) {
        // In future: auto-publish completion event if not already completed
        // For now, just a marker for proper cleanup
        if !self.is_completed() {
            // Submission was abandoned - could indicate resource leak
            eprintln!(
                "WARNING: Submission {:?} abandoned before completion ({})",
                self.id, self.operation
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submission_context_creation() {
        let ctx = SubmissionContext::new(
            SubmissionIndex(1),
            DeviceId(0),
            "test_kernel",
        );
        assert!(!ctx.is_completed());
        assert!(ctx.duration().is_none());
    }

    #[test]
    fn test_submission_mark_completed() {
        let mut ctx = SubmissionContext::new(
            SubmissionIndex(1),
            DeviceId(0),
            "test_kernel",
        );

        std::thread::sleep(Duration::from_millis(10));
        let duration = ctx.mark_completed();

        assert!(ctx.is_completed());
        assert!(duration.as_millis() >= 10);
        assert!(ctx.duration().is_some());
    }

    #[test]
    fn test_submission_error_marking() {
        let mut ctx = SubmissionContext::new(
            SubmissionIndex(1),
            DeviceId(0),
            "test_kernel",
        );

        let error = crate::ErrorContext::new(
            "Test error",
            crate::error::ErrorKind::DeviceExecution {
                device_id: DeviceId(0),
                kernel: "test".into(),
            },
        );

        ctx.mark_error(error);
        assert!(ctx.has_error());
        assert!(ctx.error().is_some());
    }
}
