//! GPU-side timestamp profiling via wgpu's `TIMESTAMP_QUERY` feature.
//!
//! Provides accurate kernel execution timing by writing begin/end timestamps
//! directly on the GPU, bypassing CPU-side synchronization noise.
//!
//! # Usage
//!
//! ```ignore
//! let timer = GpuTimer::new(&ctx);
//! if let Some(ref timer) = timer {
//!     // Pass timer to compute pass creation
//!     let desc = timer.compute_pass_descriptor(0);
//!     let mut pass = encoder.begin_compute_pass(&desc);
//!     // ... dispatch ...
//!     drop(pass);
//!     // Resolve + read back
//!     timer.resolve(&mut encoder);
//! }
//! ctx.submit(encoder);
//! let durations = timer.unwrap().read_back(&ctx);
//! ```

use std::time::Duration;
use wgpu::{Device, Queue};

/// Maximum number of timestamp pairs (begin + end) per timer.
const MAX_QUERIES: u32 = 256;

/// GPU timestamp profiler using wgpu's TIMESTAMP_QUERY feature.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    period_ns: f32,
    next_index: u32,
}

impl GpuTimer {
    /// Create a new GPU timer. Returns `None` if the adapter doesn't support timestamps.
    pub fn new(ctx: &super::GpuContext) -> Option<Self> {
        if !ctx
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }

        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GpuTimer QuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_QUERIES * 2, // begin + end per pass
        });

        let resolve_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTimer Resolve"),
            size: (MAX_QUERIES as u64) * 2 * 8, // u64 per timestamp
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTimer Readback"),
            size: (MAX_QUERIES as u64) * 2 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            period_ns: ctx.queue.get_timestamp_period(),
            next_index: 0,
        })
    }

    /// Create a `ComputePassTimestampWrites` for the next timed pass.
    /// Returns the pass index (use with `duration()` later) and the timestamp writes struct.
    pub fn begin_pass(&mut self) -> (u32, wgpu::ComputePassTimestampWrites<'_>) {
        let pass_index = self.next_index;
        assert!(
            pass_index < MAX_QUERIES,
            "GpuTimer: exceeded max query count"
        );
        self.next_index += 1;

        let writes = wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_index * 2),
            end_of_pass_write_index: Some(pass_index * 2 + 1),
        };
        (pass_index, writes)
    }

    /// Resolve all recorded timestamps into the resolve buffer.
    /// Call this BEFORE submitting the command encoder.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.next_index * 2;
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            (count as u64) * 8,
        );
    }

    /// Read back timestamps from GPU and return durations for each timed pass.
    /// Must be called after the command buffer has been submitted and the device polled.
    pub fn read_durations(&self, device: &Device, _queue: &Queue) -> Vec<Duration> {
        let count = self.next_index;
        if count == 0 {
            return Vec::new();
        }

        let slice = self.readback_buffer.slice(..((count as u64) * 2 * 8));
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(120)),
        });
        if receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .ok()
            .and_then(|r| r.ok())
            .is_none()
        {
            return Vec::new();
        }

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data[..count as usize * 2 * 8]);

        let mut durations = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let begin = timestamps[i * 2];
            let end = timestamps[i * 2 + 1];
            let ticks = end.saturating_sub(begin);
            let ns = (ticks as f64) * (self.period_ns as f64);
            durations.push(Duration::from_nanos(ns as u64));
        }

        drop(data);
        self.readback_buffer.unmap();
        durations
    }

    /// Number of passes timed so far.
    pub fn pass_count(&self) -> u32 {
        self.next_index
    }

    /// Reset the timer for reuse (e.g., next frame).
    pub fn reset(&mut self) {
        self.next_index = 0;
    }
}

/// Check if the current GPU supports timestamp queries.
pub fn supports_timestamps(ctx: &super::GpuContext) -> bool {
    ctx.device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
}
