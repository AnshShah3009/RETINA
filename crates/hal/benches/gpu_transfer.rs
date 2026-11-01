//! Host <-> device transfer cost of the `buffer_utils` helpers.
//!
//! Every GPU kernel in this crate uploads through `buffer_utils::create_buffer`
//! and reads back through `buffer_utils::read_buffer`, so a host-side copy in
//! either is paid on every call. This measures both at the sizes the image and
//! point-cloud kernels move: a 640x480 f32 frame, a 1080p f32 frame and a
//! 4M-point cloud.
//!
//! Skips (prints and returns) when no GPU adapter is available.
//!
//!   cargo bench -p cv-hal --bench gpu_transfer

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::buffer_utils;
use std::time::Duration;
use wgpu::BufferUsages;

const SIZES: [(&str, usize); 3] = [
    ("640x480_f32", 640 * 480),
    ("1080p_f32", 1920 * 1080),
    ("4M_points_xyzw", 4 * 1024 * 1024 * 4),
];

fn bench_transfer(c: &mut Criterion) {
    let Ok(ctx) = GpuContext::new() else {
        eprintln!("skipping gpu_transfer: no GPU adapter");
        return;
    };

    let mut group = c.benchmark_group("gpu_transfer");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for (label, len) in SIZES {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let bytes = (len * 4) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("upload", label), &data, |b, data| {
            b.iter(|| {
                let buf = buffer_utils::create_buffer(
                    &ctx.device,
                    black_box(data),
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );
                // Submit nothing, but make sure the mapped-at-creation copy is
                // flushed to the device before the buffer is dropped.
                ctx.queue.submit(std::iter::empty());
                black_box(buf)
            });
        });

        let src = buffer_utils::create_buffer(
            &ctx.device,
            &data,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        group.bench_with_input(BenchmarkId::new("readback", label), &src, |b, src| {
            b.iter(|| {
                let out: Vec<f32> = pollster::block_on(buffer_utils::read_buffer(
                    ctx.device.clone(),
                    &ctx.queue,
                    black_box(src),
                    0,
                    len * 4,
                ))
                .expect("readback");
                black_box(out)
            });
        });
    }

    // Small uniform buffers: every dispatch builds one, so the per-call
    // overhead of the padding path matters more than its bandwidth.
    group.throughput(Throughput::Elements(1));
    let params = [1u32, 2, 3, 4, 5, 6, 7];
    group.bench_function("uniform_28_bytes", |b| {
        b.iter(|| {
            black_box(buffer_utils::create_buffer(
                &ctx.device,
                black_box(&params),
                BufferUsages::UNIFORM,
            ))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_transfer);
criterion_main!(benches);
