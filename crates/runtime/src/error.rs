use cv_hal::DeviceId;
use std::fmt;

/// Structured error context with source chain and metadata
#[derive(Debug)]
pub struct ErrorContext {
    pub message: String,
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
    pub context: ErrorKind,
    pub retryable: bool,
}

/// Error categorization for better debugging and handling
#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// Device operation (GPU/CPU kernel)
    DeviceExecution { device_id: DeviceId, kernel: String },
    /// Memory allocation/sync
    Memory {
        required: usize,
        available: Option<usize>,
    },
    /// Pipeline execution
    Pipeline { node_name: String, stage: String },
    /// Distributed coordination
    Coordination { reason: String },
    /// Type/contract violation
    ContractViolation { expected: String, actual: String },
    /// User API misuse
    ApiMisuse { reason: String },
    /// Scheduler internal
    SchedulerInternal { reason: String },
}

impl ErrorContext {
    /// Create new error context
    pub fn new(message: impl Into<String>, kind: ErrorKind) -> Self {
        Self {
            message: message.into(),
            source: None,
            context: kind,
            retryable: false,
        }
    }

    /// Add source error (chain causality)
    pub fn with_source(mut self, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        self.source = Some(source);
        self
    }

    /// Mark error as retryable
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        self.retryable
    }

    /// Get error kind
    pub fn kind(&self) -> &ErrorKind {
        &self.context
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", self.message, self.context)?;
        if let Some(source) = &self.source {
            write!(f, " (caused by: {})", source)?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new(
            "Operation failed",
            ErrorKind::Memory {
                required: 1024,
                available: Some(512),
            },
        );
        assert_eq!(ctx.message, "Operation failed");
        assert!(!ctx.is_retryable());
    }

    #[test]
    fn test_error_context_retryable() {
        let ctx = ErrorContext::new(
            "Network timeout",
            ErrorKind::Coordination {
                reason: "Coordinator unreachable".into(),
            },
        )
        .retryable();
        assert!(ctx.is_retryable());
    }

    #[test]
    fn test_error_with_source() {
        let source_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ctx = ErrorContext::new(
            "GPU kernel failed",
            ErrorKind::DeviceExecution {
                device_id: DeviceId(0),
                kernel: "bilateral_filter".into(),
            },
        )
        .with_source(Box::new(source_err));
        assert!(ctx.source.is_some());
    }
}
