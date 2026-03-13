use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cv_core::{float::Float, CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

#[cfg(feature = "half-precision")]
use half::{bf16, f16};

fn generate_test_image<T: Float>(width: usize, height: usize) -> CpuTensor<T> {
    let shape = TensorShape::new(1, height, width);
    let mut data = vec![T::ZERO; shape.len()];

    // Create a smooth gradient image
    for y in 0..height {
        for x in 0..width {
            let val = (x as f32 / width as f32) * 255.0;
            data[y * width + x] = T::from_f32(val);
        }
    }

    Tensor::from_vec(data, shape).unwrap()
}

fn bench_precision_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Precision Accuracy");
    let cpu = CpuBackend::new().unwrap();
    let width = 256;
    let height = 256;

    // Baseline F64
    let img_f64 = generate_test_image::<f64>(width, height);
    let thresh_f64 = 127.0;
    let max_f64 = 255.0;

    // F32
    let img_f32 = generate_test_image::<f32>(width, height);
    let thresh_f32 = 127.0_f32;
    let max_f32 = 255.0_f32;

    group.bench_function("threshold_f64", |b| {
        b.iter(|| {
            black_box(
                cpu.threshold(
                    &img_f64,
                    thresh_f64,
                    max_f64,
                    cv_hal::context::ThresholdType::Binary,
                )
                .unwrap(),
            )
        })
    });

    group.bench_function("threshold_f32", |b| {
        b.iter(|| {
            black_box(
                cpu.threshold(
                    &img_f32,
                    thresh_f32,
                    max_f32,
                    cv_hal::context::ThresholdType::Binary,
                )
                .unwrap(),
            )
        })
    });

    #[cfg(feature = "half-precision")]
    {
        let img_f16 = generate_test_image::<f16>(width, height);
        let thresh_f16 = f16::from_f32(127.0);
        let max_f16 = f16::from_f32(255.0);

        let img_bf16 = generate_test_image::<bf16>(width, height);
        let thresh_bf16 = bf16::from_f32(127.0);
        let max_bf16 = bf16::from_f32(255.0);

        group.bench_function("threshold_f16", |b| {
            b.iter(|| {
                black_box(
                    cpu.threshold(
                        &img_f16,
                        thresh_f16,
                        max_f16,
                        cv_hal::context::ThresholdType::Binary,
                    )
                    .unwrap(),
                )
            })
        });

        group.bench_function("threshold_bf16", |b| {
            b.iter(|| {
                black_box(
                    cpu.threshold(
                        &img_bf16,
                        thresh_bf16,
                        max_bf16,
                        cv_hal::context::ThresholdType::Binary,
                    )
                    .unwrap(),
                )
            })
        });

        // Calculate Accuracy Discrepancy (Not really a criterion benchmark, but good for validation inline)
        let res_f64 = cpu
            .threshold(
                &img_f64,
                thresh_f64,
                max_f64,
                cv_hal::context::ThresholdType::Binary,
            )
            .unwrap();
        let res_f16 = cpu
            .threshold(
                &img_f16,
                thresh_f16,
                max_f16,
                cv_hal::context::ThresholdType::Binary,
            )
            .unwrap();

        let slice_f64 = res_f64.storage.as_slice().unwrap();
        let slice_f16 = res_f16.storage.as_slice().unwrap();

        let mut total_error: f64 = 0.0;
        for i in 0..slice_f64.len() {
            let diff = (slice_f64[i] - slice_f16[i].to_f64()).abs();
            total_error += diff;
        }
        println!(
            "  => Average precision loss from F64 to F16 Thresholding: {}",
            total_error / slice_f64.len() as f64
        );
    }

    group.finish();
}

criterion_group!(benches, bench_precision_accuracy);
criterion_main!(benches);
