//! CPU feature pipeline benchmarks: ORB, FAST, SIFT and descriptor matching.
//!
//! Inputs are a deterministic synthetic scene (overlapping rectangles and
//! discs of random grey levels plus mild noise) at 640x480 and 1920x1080, so
//! the detectors find thousands of real corners.
//!
//! Run: `cargo bench -p cv-features --bench features_cpu`

use criterion::{Criterion, criterion_group, criterion_main};
use cv_core::TensorShape;
use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_features::{
    DescriptorExtractor, Descriptors, FlannMatcher, MatchType, Matcher, Orb, Sift, fast_detect,
    knn_match, non_max_suppression, orb_detect_and_compute,
};
use cv_hal::compute::ComputeDevice;
use cv_hal::cpu::CpuBackend;
use image::GrayImage;
use std::hint::black_box;
use std::time::Duration;

fn scene(w: u32, h: u32) -> GrayImage {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut img = GrayImage::from_pixel(w, h, image::Luma([128]));
    let shapes = (w * h / 1000) as usize;
    for i in 0..shapes {
        let cx = (next() % w as u64) as i64;
        let cy = (next() % h as u64) as i64;
        let r = (next() % 40 + 4) as i64;
        let v = (next() % 256) as u8;
        for y in (cy - r).max(0)..(cy + r).min(h as i64) {
            for x in (cx - r).max(0)..(cx + r).min(w as i64) {
                let inside = i % 2 == 0 || (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;
                if inside {
                    img.put_pixel(x as u32, y as u32, image::Luma([v]));
                }
            }
        }
    }
    for p in img.pixels_mut() {
        let n = (next() % 7) as i16 - 3;
        p[0] = (p[0] as i16 + n).clamp(0, 255) as u8;
    }
    img
}

fn frame_640() -> GrayImage {
    scene(640, 480)
}

fn frame_1080(_src: &GrayImage) -> GrayImage {
    scene(1920, 1080)
}

/// Same texture shifted by a few pixels, so matching has real correspondences.
fn shifted(src: &GrayImage, dx: u32, dy: u32) -> GrayImage {
    let (w, h) = src.dimensions();
    GrayImage::from_fn(w, h, |x, y| {
        *src.get_pixel((x + dx).min(w - 1), (y + dy).min(h - 1))
    })
}

fn bench_orb(c: &mut Criterion) {
    let small = frame_640();
    let big = frame_1080(&small);
    let mut g = c.benchmark_group("orb");
    g.sample_size(20).measurement_time(Duration::from_secs(6));
    g.bench_function("detect_and_compute_640x480_1000", |b| {
        b.iter(|| orb_detect_and_compute(black_box(&small), 1000))
    });
    g.bench_function("detect_and_compute_1080p_2000", |b| {
        b.iter(|| orb_detect_and_compute(black_box(&big), 2000))
    });
    // The stages of `orb_detect_and_compute` on the 1080p frame, separately.
    let orb = Orb::new().with_n_features(2000);
    g.bench_function("stage_detect_1080p_2000", |b| {
        b.iter(|| orb.detect(black_box(&big)))
    });
    let detected = orb.detect(&big);
    g.bench_function("stage_orientations_1080p_2000", |b| {
        b.iter(|| {
            let mut kps = detected.clone();
            orb.compute_orientations(black_box(&big), &mut kps);
            kps
        })
    });
    let mut oriented = detected.clone();
    orb.compute_orientations(&big, &mut oriented);
    g.bench_function("stage_extract_1080p_2000", |b| {
        b.iter(|| orb.extract(black_box(&big), &oriented))
    });
    g.finish();
}

fn bench_fast(c: &mut Criterion) {
    let small = frame_640();
    let mut g = c.benchmark_group("fast");
    g.sample_size(30);
    g.bench_function("detect_640x480_t20", |b| {
        b.iter(|| fast_detect(black_box(&small), 20, usize::MAX))
    });
    let kps = fast_detect(&small, 20, usize::MAX);
    g.bench_function("nms_640x480_t20", |b| {
        b.iter(|| non_max_suppression(black_box(kps.clone()), &small, 20))
    });
    let big = frame_1080(&small);
    g.bench_function("detect_1080p_t20", |b| {
        b.iter(|| fast_detect(black_box(&big), 20, usize::MAX))
    });
    let kps_big = fast_detect(&big, 20, usize::MAX);
    eprintln!("fast candidates: 640={} 1080={}", kps.len(), kps_big.len());
    g.bench_function("nms_1080p_t20", |b| {
        b.iter(|| non_max_suppression(black_box(kps_big.clone()), &big, 20))
    });
    g.bench_function("orb_stage_gaussian_blur_1080p", |b| {
        b.iter(|| cv_imgproc::gaussian_blur(black_box(&big), 2.0))
    });
    g.finish();
}

fn bench_sift(c: &mut Criterion) {
    let small = frame_640();
    let cpu = CpuBackend::new().unwrap();
    let device = ComputeDevice::Cpu(&cpu);
    let tensor: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(small.as_raw().clone(), TensorShape::new(1, 480, 640)).unwrap();
    let sift = Sift::new();
    let mut g = c.benchmark_group("sift");
    g.sample_size(10).measurement_time(Duration::from_secs(8));
    g.bench_function("detect_and_compute_640x480", |b| {
        b.iter(|| {
            sift.detect_and_compute(&device, black_box(&tensor))
                .unwrap()
        })
    });
    g.finish();
}

fn orb_pair() -> (Descriptors, Descriptors) {
    let small = frame_640();
    let other = shifted(&small, 3, 2);
    let (_, d1) = orb_detect_and_compute(&small, 1000);
    let (_, d2) = orb_detect_and_compute(&other, 1000);
    (d1, d2)
}

fn sift_pair() -> (Descriptors, Descriptors) {
    let small = frame_640();
    let other = shifted(&small, 3, 2);
    let cpu = CpuBackend::new().unwrap();
    let device = ComputeDevice::Cpu(&cpu);
    let sift = Sift::new();
    let t1: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(small.as_raw().clone(), TensorShape::new(1, 480, 640)).unwrap();
    let t2: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(other.as_raw().clone(), TensorShape::new(1, 480, 640)).unwrap();
    let (_, d1) = sift.detect_and_compute(&device, &t1).unwrap();
    let (_, d2) = sift.detect_and_compute(&device, &t2).unwrap();
    (d1, d2)
}

fn bench_matching(c: &mut Criterion) {
    let (q, t) = orb_pair();
    let (sq, st) = sift_pair();
    let mut g = c.benchmark_group("matching");
    g.sample_size(20);
    let hamming = Matcher::new(MatchType::BruteForceHamming).with_cross_check();
    g.bench_function(format!("bf_hamming_cross_{}x{}", q.len(), t.len()), |b| {
        b.iter(|| hamming.match_descriptors(black_box(&q), black_box(&t)))
    });
    let ratio = Matcher::new(MatchType::BruteForceHamming).with_ratio_test(0.8);
    g.bench_function(format!("bf_hamming_ratio_{}x{}", q.len(), t.len()), |b| {
        b.iter(|| ratio.match_descriptors(black_box(&q), black_box(&t)))
    });
    let l2 = Matcher::new(MatchType::BruteForceL2).with_ratio_test(0.8);
    g.bench_function(format!("bf_l2_sift_{}x{}", sq.len(), st.len()), |b| {
        b.iter(|| l2.match_descriptors(black_box(&sq), black_box(&st)))
    });
    g.bench_function(format!("knn2_hamming_{}x{}", q.len(), t.len()), |b| {
        b.iter(|| knn_match(black_box(&q), black_box(&t), 2))
    });
    let mut flann = FlannMatcher::new().with_ratio_test(0.8);
    flann.train(&t);
    g.bench_function(format!("flann_{}x{}", q.len(), t.len()), |b| {
        b.iter(|| flann.match_descriptors(black_box(&q)))
    });
    g.finish();
}

criterion_group!(benches, bench_orb, bench_fast, bench_sift, bench_matching);
criterion_main!(benches);
