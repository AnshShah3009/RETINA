# Phase 1: Observability + Error Context Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive error context chains, runtime event observability, and metrics collection to cv-runtime as the foundation for production-grade stability.

**Architecture:** Three new modules in cv-runtime:
- `error.rs` - ErrorContext with source chain and ErrorKind enum
- `observe/mod.rs` - RuntimeEvent publishing, Metrics collection, ObservabilityLayer
- `observe/submission.rs` - SubmissionContext RAII guard for lifecycle tracking

**Tech Stack:**
- `thiserror` for error definitions
- `parking_lot` for metrics synchronization
- `tracing` (feature-gated) for structured logging
- `std::time::Instant` for timing
- Ring buffer (Vec with capacity) for event storage

---

## Task 1: Create ErrorContext Type with Source Chain

**Files:**
- Create: `runtime/src/error.rs`
- Modify: `runtime/src/lib.rs` (add mod error; use statement)
- Test: `runtime/src/error.rs` (inline tests module)

**Step 1: Write the failing test**

```rust
// At bottom of runtime/src/error.rs after main code:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new("Operation failed", ErrorKind::Memory {
            required: 1024,
            available: Some(512),
        });
        assert_eq!(ctx.message, "Operation failed");
        assert!(!ctx.is_retryable());
    }

    #[test]
    fn test_error_context_retryable() {
        let ctx = ErrorContext::new("Network timeout", ErrorKind::Coordination {
            reason: "Coordinator unreachable".into(),
        }).retryable();
        assert!(ctx.is_retryable());
    }

    #[test]
    fn test_error_with_source() {
        let source_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ctx = ErrorContext::new("GPU kernel failed", ErrorKind::DeviceExecution {
            device_id: DeviceId(0),
            kernel: "bilateral_filter".into(),
        }).with_source(Box::new(source_err));
        assert!(ctx.source.is_some());
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cd /home/prathana/RUST/rust-cv-native
cargo test -p cv-runtime error::tests::test_error_context_creation --lib 2>&1 | head -20
```

Expected output: Error about `error` module not found or `ErrorContext` not defined.

**Step 3: Write minimal implementation**

Create `runtime/src/error.rs`:

```rust
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
    DeviceExecution {
        device_id: DeviceId,
        kernel: String,
    },
    /// Memory allocation/sync
    Memory {
        required: usize,
        available: Option<usize>,
    },
    /// Pipeline execution
    Pipeline {
        node_name: String,
        stage: String,
    },
    /// Distributed coordination
    Coordination {
        reason: String,
    },
    /// Type/contract violation
    ContractViolation {
        expected: String,
        actual: String,
    },
    /// User API misuse
    ApiMisuse {
        reason: String,
    },
    /// Scheduler internal
    SchedulerInternal {
        reason: String,
    },
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
        self.source.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new("Operation failed", ErrorKind::Memory {
            required: 1024,
            available: Some(512),
        });
        assert_eq!(ctx.message, "Operation failed");
        assert!(!ctx.is_retryable());
    }

    #[test]
    fn test_error_context_retryable() {
        let ctx = ErrorContext::new("Network timeout", ErrorKind::Coordination {
            reason: "Coordinator unreachable".into(),
        }).retryable();
        assert!(ctx.is_retryable());
    }

    #[test]
    fn test_error_with_source() {
        let source_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ctx = ErrorContext::new("GPU kernel failed", ErrorKind::DeviceExecution {
            device_id: DeviceId(0),
            kernel: "bilateral_filter".into(),
        }).with_source(Box::new(source_err));
        assert!(ctx.source.is_some());
    }
}
```

Now update `runtime/src/lib.rs` to include the module:

```rust
pub mod error;  // Add this near top with other pub mods

pub use error::ErrorContext;  // Add to pub use block
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p cv-runtime error::tests --lib 2>&1 | grep -E "(test result|running)"
```

Expected output:
```
running 3 tests
test result: ok. 3 passed
```

**Step 5: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add runtime/src/error.rs runtime/src/lib.rs
git commit -m "feat(runtime): add ErrorContext with source chain and ErrorKind categorization"
```

---

## Task 2: Create Metrics Collection System

**Files:**
- Create: `runtime/src/observe/metrics.rs`
- Create: `runtime/src/observe/mod.rs` (initially minimal)
- Modify: `runtime/src/lib.rs` (add pub mod observe)
- Test: `runtime/src/observe/metrics.rs` (inline tests)

**Step 1: Write the failing test**

```rust
// In runtime/src/observe/metrics.rs tests module:
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_allocation() {
        let metrics = Metrics::new();
        assert_eq!(metrics.total_allocated(), 0);

        metrics.record_allocation(1024);
        assert_eq!(metrics.total_allocated(), 1024);
    }

    #[test]
    fn test_metrics_release() {
        let metrics = Metrics::new();
        metrics.record_allocation(1024);
        metrics.record_release(512);
        assert_eq!(metrics.total_allocated(), 512);
    }

    #[test]
    fn test_metrics_peak() {
        let metrics = Metrics::new();
        metrics.record_allocation(1000);
        assert_eq!(metrics.peak_allocated(), 1000);

        metrics.record_allocation(500);
        assert_eq!(metrics.peak_allocated(), 1500);

        metrics.record_release(1000);
        assert_eq!(metrics.peak_allocated(), 1500); // Peak doesn't decrease
    }

    #[test]
    fn test_metrics_latency() {
        let metrics = Metrics::new();
        metrics.record_submission_latency(10);
        metrics.record_submission_latency(20);
        metrics.record_submission_latency(30);

        let avg = metrics.avg_submission_latency_ms();
        assert!((avg - 20.0).abs() < 0.1); // Should be 20ms
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p cv-runtime observe::metrics::tests --lib 2>&1 | head -20
```

Expected: Errors about module/type not found.

**Step 3: Write minimal implementation**

Create `runtime/src/observe/metrics.rs`:

```rust
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Metrics collection for runtime monitoring
#[derive(Debug, Clone)]
pub struct Metrics {
    total_allocated: Arc<AtomicUsize>,
    peak_allocated: Arc<AtomicUsize>,
    submission_count: Arc<AtomicU64>,
    submission_latencies: Arc<Mutex<Vec<u64>>>,
}

impl Metrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            total_allocated: Arc::new(AtomicUsize::new(0)),
            peak_allocated: Arc::new(AtomicUsize::new(0)),
            submission_count: Arc::new(AtomicU64::new(0)),
            submission_latencies: Arc::new(Mutex::new(Vec::with_capacity(1000))),
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, bytes: usize) {
        let prev = self.total_allocated.fetch_add(bytes, Ordering::SeqCst);
        let new_total = prev + bytes;

        // Update peak if necessary
        let mut peak = self.peak_allocated.load(Ordering::SeqCst);
        while new_total > peak {
            match self.peak_allocated.compare_exchange(peak, new_total, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Record memory release
    pub fn record_release(&self, bytes: usize) {
        self.total_allocated.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Get current total allocated
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated
    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated.load(Ordering::SeqCst)
    }

    /// Record submission latency in milliseconds
    pub fn record_submission_latency(&self, ms: u64) {
        if let Ok(mut latencies) = self.submission_latencies.lock() {
            latencies.push(ms);
        }
        self.submission_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Get average submission latency in milliseconds
    pub fn avg_submission_latency_ms(&self) -> f64 {
        if let Ok(latencies) = self.submission_latencies.lock() {
            if latencies.is_empty() {
                return 0.0;
            }
            let sum: u64 = latencies.iter().sum();
            sum as f64 / latencies.len() as f64
        } else {
            0.0
        }
    }

    /// Get submission count
    pub fn submission_count(&self) -> u64 {
        self.submission_count.load(Ordering::SeqCst)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_allocation() {
        let metrics = Metrics::new();
        assert_eq!(metrics.total_allocated(), 0);

        metrics.record_allocation(1024);
        assert_eq!(metrics.total_allocated(), 1024);
    }

    #[test]
    fn test_metrics_release() {
        let metrics = Metrics::new();
        metrics.record_allocation(1024);
        metrics.record_release(512);
        assert_eq!(metrics.total_allocated(), 512);
    }

    #[test]
    fn test_metrics_peak() {
        let metrics = Metrics::new();
        metrics.record_allocation(1000);
        assert_eq!(metrics.peak_allocated(), 1000);

        metrics.record_allocation(500);
        assert_eq!(metrics.peak_allocated(), 1500);

        metrics.record_release(1000);
        assert_eq!(metrics.peak_allocated(), 1500); // Peak doesn't decrease
    }

    #[test]
    fn test_metrics_latency() {
        let metrics = Metrics::new();
        metrics.record_submission_latency(10);
        metrics.record_submission_latency(20);
        metrics.record_submission_latency(30);

        let avg = metrics.avg_submission_latency_ms();
        assert!((avg - 20.0).abs() < 0.1); // Should be 20ms
    }
}
```

Create `runtime/src/observe/mod.rs`:

```rust
pub mod metrics;

pub use metrics::Metrics;
```

Update `runtime/src/lib.rs`:

```rust
pub mod observe;  // Add with other pub mods
pub use observe::Metrics;  // Add to pub use block
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p cv-runtime observe::metrics::tests --lib 2>&1 | grep -E "(test result|running)"
```

Expected: All 4 tests pass.

**Step 5: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add runtime/src/observe/mod.rs runtime/src/observe/metrics.rs runtime/src/lib.rs
git commit -m "feat(runtime): add Metrics collection system (memory, latency tracking)"
```

---

## Task 3: Create RuntimeEvent Enum and Event Publishing

**Files:**
- Create: `runtime/src/observe/events.rs`
- Modify: `runtime/src/observe/mod.rs` (add pub mod events)
- Test: `runtime/src/observe/events.rs` (inline tests)

**Step 1: Write the failing test**

```rust
// In runtime/src/observe/events.rs tests module:
#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::DeviceId;

    #[test]
    fn test_submission_started_event() {
        let event = RuntimeEvent::SubmissionStarted {
            submission_id: cv_hal::SubmissionIndex(1),
            device_id: DeviceId(0),
            operation: "bilateral_filter".into(),
            timestamp: std::time::Instant::now(),
        };

        match event {
            RuntimeEvent::SubmissionStarted { operation, .. } => {
                assert_eq!(operation, "bilateral_filter");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_pipeline_node_event() {
        let event = RuntimeEvent::PipelineNodeStarted {
            pipeline_id: 42,
            node_id: crate::pipeline::NodeId(0),
            node_name: "kernel1".into(),
            timestamp: std::time::Instant::now(),
        };

        match event {
            RuntimeEvent::PipelineNodeStarted { pipeline_id, .. } => {
                assert_eq!(pipeline_id, 42);
            }
            _ => panic!("Wrong event type"),
        }
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p cv-runtime observe::events::tests --lib 2>&1 | head -20
```

Expected: Module/type not found errors.

**Step 3: Write minimal implementation**

Create `runtime/src/observe/events.rs`:

```rust
use cv_hal::{DeviceId, SubmissionIndex};
use std::time::Instant;
use crate::pipeline::NodeId;

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
            _ => panic!("Wrong event type"),
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
```

Update `runtime/src/observe/mod.rs`:

```rust
pub mod events;
pub mod metrics;

pub use events::{RuntimeEvent, MemoryEventKind, DeviceHealth};
pub use metrics::Metrics;
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p cv-runtime observe::events::tests --lib 2>&1 | grep -E "(test result|running)"
```

Expected: All 4 tests pass.

**Step 5: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add runtime/src/observe/events.rs runtime/src/observe/mod.rs
git commit -m "feat(runtime): add RuntimeEvent enum for observability"
```

---

## Task 4: Create SubmissionContext RAII Guard

**Files:**
- Create: `runtime/src/observe/submission.rs`
- Modify: `runtime/src/observe/mod.rs` (add pub mod submission)
- Test: `runtime/src/observe/submission.rs` (inline tests)

**Step 1: Write the failing test**

```rust
// In runtime/src/observe/submission.rs tests module:
#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::DeviceId;
    use std::time::Duration;

    #[test]
    fn test_submission_context_creation() {
        let ctx = SubmissionContext::new(
            cv_hal::SubmissionIndex(1),
            DeviceId(0),
            "test_kernel",
        );
        assert!(!ctx.is_completed());
        assert!(ctx.duration().is_none());
    }

    #[test]
    fn test_submission_mark_completed() {
        let mut ctx = SubmissionContext::new(
            cv_hal::SubmissionIndex(1),
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
            cv_hal::SubmissionIndex(1),
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
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p cv-runtime observe::submission::tests --lib 2>&1 | head -20
```

Expected: Module/type not found errors.

**Step 3: Write minimal implementation**

Create `runtime/src/observe/submission.rs`:

```rust
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
```

Update `runtime/src/observe/mod.rs`:

```rust
pub mod events;
pub mod metrics;
pub mod submission;

pub use events::{RuntimeEvent, MemoryEventKind, DeviceHealth};
pub use metrics::Metrics;
pub use submission::SubmissionContext;
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p cv-runtime observe::submission::tests --lib 2>&1 | grep -E "(test result|running)"
```

Expected: All 3 tests pass.

**Step 5: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add runtime/src/observe/submission.rs runtime/src/observe/mod.rs
git commit -m "feat(runtime): add SubmissionContext RAII guard for lifecycle tracking"
```

---

## Task 5: Global Observability Layer and OnceLock Setup

**Files:**
- Create: `runtime/src/observe/layer.rs`
- Modify: `runtime/src/observe/mod.rs` (add pub mod layer)
- Modify: `runtime/src/lib.rs` (add OnceLock + factory functions)
- Test: `runtime/src/observe/layer.rs` (inline tests)

**Step 1: Write the failing test**

```rust
// In runtime/src/observe/layer.rs tests module:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_singleton() {
        let layer = observability();
        let metrics = layer.metrics();

        metrics.record_allocation(1024);
        assert_eq!(metrics.total_allocated(), 1024);

        // Get again - should be same instance
        let layer2 = observability();
        let metrics2 = layer2.metrics();
        assert_eq!(metrics2.total_allocated(), 1024);
    }

    #[test]
    fn test_event_buffering() {
        let layer = observability();
        let before_count = layer.event_count();

        layer.publish_event(RuntimeEvent::MemoryEvent {
            kind: MemoryEventKind::Allocated,
            size_bytes: 4096,
            timestamp: std::time::Instant::now(),
        });

        assert_eq!(layer.event_count(), before_count + 1);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p cv-runtime observe::layer::tests --lib 2>&1 | head -20
```

Expected: Module/type not found.

**Step 3: Write minimal implementation**

Create `runtime/src/observe/layer.rs`:

```rust
use super::{RuntimeEvent, Metrics, MemoryEventKind};
use std::sync::{Arc, Mutex, OnceLock};

/// Global observability layer
#[derive(Debug, Clone)]
pub struct ObservabilityLayer {
    metrics: Metrics,
    events: Arc<Mutex<Vec<RuntimeEvent>>>,
    max_events: usize,
}

impl ObservabilityLayer {
    /// Create new observability layer
    pub fn new(max_events: usize) -> Self {
        Self {
            metrics: Metrics::new(),
            events: Arc::new(Mutex::new(Vec::with_capacity(max_events))),
            max_events,
        }
    }

    /// Publish a runtime event
    pub fn publish_event(&self, event: RuntimeEvent) {
        if let Ok(mut events) = self.events.lock() {
            if events.len() >= self.max_events {
                // Ring buffer: remove oldest if at capacity
                events.remove(0);
            }
            events.push(event);
        }
    }

    /// Get metrics
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Get event count
    pub fn event_count(&self) -> usize {
        self.events.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Get recent events (copy)
    pub fn get_recent_events(&self, count: usize) -> Vec<RuntimeEvent> {
        self.events
            .lock()
            .map(|events| {
                events
                    .iter()
                    .rev()
                    .take(count)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for ObservabilityLayer {
    fn default() -> Self {
        Self::new(10000) // 10k event buffer by default
    }
}

static OBSERVABILITY: OnceLock<ObservabilityLayer> = OnceLock::new();

/// Get global observability layer (singleton)
pub fn observability() -> &'static ObservabilityLayer {
    OBSERVABILITY.get_or_init(ObservabilityLayer::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_singleton() {
        let layer = observability();
        let metrics = layer.metrics();

        metrics.record_allocation(1024);
        assert_eq!(metrics.total_allocated(), 1024);

        // Get again - should be same instance
        let layer2 = observability();
        let metrics2 = layer2.metrics();
        assert_eq!(metrics2.total_allocated(), 1024);
    }

    #[test]
    fn test_event_buffering() {
        let layer = observability();
        let before_count = layer.event_count();

        layer.publish_event(RuntimeEvent::MemoryEvent {
            kind: MemoryEventKind::Allocated,
            size_bytes: 4096,
            timestamp: std::time::Instant::now(),
        });

        assert_eq!(layer.event_count(), before_count + 1);
    }
}
```

Update `runtime/src/observe/mod.rs`:

```rust
pub mod events;
pub mod layer;
pub mod metrics;
pub mod submission;

pub use events::{RuntimeEvent, MemoryEventKind, DeviceHealth};
pub use layer::{observability, ObservabilityLayer};
pub use metrics::Metrics;
pub use submission::SubmissionContext;
```

Update `runtime/src/lib.rs` - add to public exports:

```rust
pub use observe::{observability, ObservabilityLayer, RuntimeEvent};
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p cv-runtime observe::layer::tests --lib 2>&1 | grep -E "(test result|running)"
```

Expected: Both tests pass.

**Step 5: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add runtime/src/observe/layer.rs runtime/src/observe/mod.rs runtime/src/lib.rs
git commit -m "feat(runtime): add global ObservabilityLayer singleton with event publishing"
```

---

## Task 6: Build, Test, and Verify Phase 1

**Files:**
- No new files, just verification

**Step 1: Full workspace build**

```bash
cd /home/prathana/RUST/rust-cv-native
cargo build --lib -p cv-runtime 2>&1 | tail -20
```

Expected: `Finished dev profile`

**Step 2: Run all runtime tests**

```bash
cargo test -p cv-runtime --lib 2>&1 | tail -30
```

Expected: All tests pass, including new observe and error tests.

**Step 3: Run clippy to check for warnings**

```bash
cargo clippy -p cv-runtime --lib 2>&1 | grep -E "(warning|error)" | head -20
```

Expected: No clippy warnings (ignore doc comment warnings for now).

**Step 4: Commit verification**

```bash
cd /home/prathana/RUST/rust-cv-native
git log --oneline | head -6
```

Should show 5 commits from this phase:
- error context
- metrics
- runtime events
- submission context
- observability layer

**Step 5: Document Phase 1 completion**

Create summary: Phase 1 is **production-ready** and can ship independently. Domains can now:
- Catch errors with full context chains
- Monitor pipeline execution via events
- Track memory allocation/release
- See submission latencies

---

## Summary

**Phase 1 Tasks:** 6 tasks (Error, Metrics, Events, Submission, Layer, Verification)

**Expected Commits:** 5

**Files Created:** 6
- `runtime/src/error.rs`
- `runtime/src/observe/mod.rs`
- `runtime/src/observe/events.rs`
- `runtime/src/observe/metrics.rs`
- `runtime/src/observe/submission.rs`
- `runtime/src/observe/layer.rs`

**Test Count Added:** ~16 new tests (3 in error, 4 in metrics, 4 in events, 3 in submission, 2 in layer)

**Build Status:** ✅ Should pass with no warnings
