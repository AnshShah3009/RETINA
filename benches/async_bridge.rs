use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::GrayImage;
use tokio::runtime::Runtime;

fn benchmark_async_bridge(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (left, right) = (GrayImage::new(256, 256), GrayImage::new(256, 256));

    let mut group = c.benchmark_group("async_bridge");
    group.sample_size(10);

    // Baseline: Direct synchronous call
    group.bench_function("sync_direct", |b| {
        b.iter(|| {
            let _ = cv_calib3d::stereo_block_match(black_box(&left), black_box(&right), 11, 32);
        })
    });

    // Wrapped: Via spawn_blocking (simulates async_ops)
    group.bench_function("async_wrapper", |b| {
        b.to_async(&rt).iter(|| {
            let l = left.clone();
            let r = right.clone();
            async move {
                tokio::task::spawn_blocking(move || cv_calib3d::stereo_block_match(&l, &r, 11, 32))
                    .await
                    .unwrap()
            }
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_async_bridge);
criterion_main!(benches);
