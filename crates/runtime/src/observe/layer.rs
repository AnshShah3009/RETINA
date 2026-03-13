use super::{Metrics, RuntimeEvent};
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
    use crate::observe::MemoryEventKind;

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
