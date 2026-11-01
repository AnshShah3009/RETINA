//! Device time of the CubeCL kernels, launch plus synchronisation.
//!
//! Each iteration launches one wrapper and waits for the stream to drain, so
//! the figure is what a caller blocking on the result pays: compilation is
//! excluded by a warm-up launch, allocation of the output is included because
//! every wrapper allocates one.
//!
//! Requires the `cubecl` feature and a GPU adapter:
//!
//!   cargo bench -p cv-hal --features cubecl --bench cubecl_kernels

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl_wgpu::WgpuDevice;
use cv_hal::gpu_kernels::cubecl_advanced as adv;
use cv_hal::gpu_kernels::cubecl_backend as be;
use cv_hal::gpu_kernels::cubecl_optimized as opt;
use cv_hal::gpu_kernels::cubecl_proto as proto;
use std::time::Duration;

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn up(c: &Client, v: &[f32], shape: Vec<usize>) -> TensorHandle {
    proto::tensor_from_slice(c, v, shape).expect("upload")
}

fn sync(c: &Client) {
    futures::executor::block_on(c.sync()).expect("cubecl stream error");
}

/// `(label, batch, c_in, c_out, h, w, k, stride, pad)`
const CONV_SHAPES: [(&str, usize, usize, usize, usize, usize, usize, usize, usize); 3] = [
    ("rgb_480p_3x3_c16", 1, 3, 16, 480, 640, 3, 1, 1),
    ("c16_240p_3x3_c32", 1, 16, 32, 240, 320, 3, 1, 1),
    ("gray_1080p_5x5_c8", 1, 1, 8, 1080, 1920, 5, 1, 2),
];

fn bench_conv2d(c: &mut Criterion) {
    let client = cubecl::Device::Wgpu(WgpuDevice::default()).client();
    let be_ctx = be::CubeCLContext::new(client.clone());
    let opt_ctx = opt::OptimizedContext::new(client.clone(), false);

    let mut group = c.benchmark_group("cubecl_conv2d");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for (label, b, ci, co, h, w, k, stride, pad) in CONV_SHAPES {
        let input = up(&client, &data(b * ci * h * w, 1), vec![b, ci, h, w]);
        let weights = up(&client, &data(co * ci * k * k, 2), vec![co, ci, k, k]);

        // Warm-up: compiles both kernels and runs the autotuner for this key.
        let _ = be::conv2d(&be_ctx, &input, &weights, stride, pad).unwrap();
        let _ = opt::conv2d_tiled(&opt_ctx, &input, &weights, stride, pad).unwrap();
        let _ = opt::conv2d(&opt_ctx, &input, &weights, stride, pad).unwrap();
        sync(&client);

        group.bench_with_input(BenchmarkId::new("naive", label), &(), |bch, _| {
            bch.iter(|| {
                let out = be::conv2d(&be_ctx, &input, &weights, stride, pad).unwrap();
                sync(&client);
                black_box(out)
            });
        });
        group.bench_with_input(BenchmarkId::new("tiled", label), &(), |bch, _| {
            bch.iter(|| {
                let out = opt::conv2d_tiled(&opt_ctx, &input, &weights, stride, pad).unwrap();
                sync(&client);
                black_box(out)
            });
        });
        // `opt::conv2d`, the public entry point: the tiled kernel before
        // autotuning, the tuner's pick after.
        group.bench_with_input(BenchmarkId::new("opt_conv2d", label), &(), |bch, _| {
            bch.iter(|| {
                let out = opt::conv2d(&opt_ctx, &input, &weights, stride, pad).unwrap();
                sync(&client);
                black_box(out)
            });
        });
    }
    group.finish();
}

fn bench_misc(c: &mut Criterion) {
    let client = cubecl::Device::Wgpu(WgpuDevice::default()).client();
    let be_ctx = be::CubeCLContext::new(client.clone());
    let adv_ctx = adv::AdvancedContext::new(client.clone());

    let mut group = c.benchmark_group("cubecl_misc");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let (h, w) = (1080, 1920);
    let a = up(&client, &data(h * w, 3), vec![h, w]);
    let bt = up(&client, &data(h * w, 4), vec![h, w]);
    let _ = be::add(&be_ctx, &a, &bt).unwrap();
    let _ = adv::gaussian_blur(&adv_ctx, &a).unwrap();
    sync(&client);
    group.bench_function("add_1080p", |bch| {
        bch.iter(|| {
            let out = be::add(&be_ctx, &a, &bt).unwrap();
            sync(&client);
            black_box(out)
        });
    });
    group.bench_function("gaussian_blur_1080p", |bch| {
        bch.iter(|| {
            let out = adv::gaussian_blur(&adv_ctx, &a).unwrap();
            sync(&client);
            black_box(out)
        });
    });

    let n = 512;
    let ma = up(&client, &data(n * n, 5), vec![n, n]);
    let mb = up(&client, &data(n * n, 6), vec![n, n]);
    let _ = be::matmul(&be_ctx, &ma, &mb).unwrap();
    sync(&client);
    group.bench_function("matmul_512", |bch| {
        bch.iter(|| {
            let out = be::matmul(&be_ctx, &ma, &mb).unwrap();
            sync(&client);
            black_box(out)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_conv2d, bench_misc);
criterion_main!(benches);
