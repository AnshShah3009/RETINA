use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
            match self.peak_allocated.compare_exchange(
                peak,
                new_total,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
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
