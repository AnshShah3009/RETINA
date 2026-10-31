//! GPU LSD radix sort must be a correct, stable sort.
//!
//! An LSD radix sort is only correct if every digit pass is stable: keys equal
//! in the current digit must keep the order the previous passes gave them.
//! The scatter used to rank keys within a workgroup with `atomicAdd`, whose
//! order is unspecified, so later passes scrambled the lower digits.
//!
//! Tests skip when no adapter is available.

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::radix_sort::{radix_sort_key_value_u32, radix_sort_u32};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU radix sort test (no adapter): {}", e);
            None
        }
    }
}

/// xorshift32 stream - deterministic pseudo-random keys.
fn random_keys(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

fn gpu_sort(gpu: &GpuContext, keys: &[u32]) -> Vec<u32> {
    let n = keys.len();
    let input: CpuTensor<u32> = Tensor::from_vec(keys.to_vec(), TensorShape::new(1, 1, n)).unwrap();
    let out = radix_sort_u32(gpu, &input.to_gpu_ctx(gpu).unwrap())
        .unwrap()
        .to_cpu_ctx(gpu)
        .unwrap();
    out.storage.as_slice().unwrap().to_vec()
}

fn gpu_sort_kv(gpu: &GpuContext, keys: &[u32], vals: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    let k: CpuTensor<u32> = Tensor::from_vec(keys.to_vec(), TensorShape::new(1, 1, n)).unwrap();
    let v: CpuTensor<u32> = Tensor::from_vec(vals.to_vec(), TensorShape::new(1, 1, n)).unwrap();
    let (ks, vs) = radix_sort_key_value_u32(
        gpu,
        &k.to_gpu_ctx(gpu).unwrap(),
        &v.to_gpu_ctx(gpu).unwrap(),
    )
    .unwrap();
    (
        ks.to_cpu_ctx(gpu)
            .unwrap()
            .storage
            .as_slice()
            .unwrap()
            .to_vec(),
        vs.to_cpu_ctx(gpu)
            .unwrap()
            .storage
            .as_slice()
            .unwrap()
            .to_vec(),
    )
}

fn assert_sorted_equal(label: &str, got: &[u32], keys: &[u32]) {
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(got.len(), expected.len(), "{label}: length");
    let bad = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let first = got.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        bad,
        0,
        "{label}: {bad} of {} positions differ from sort() (first at {first:?})",
        got.len()
    );
}

#[test]
fn radix_sort_1000_random_keys_matches_cpu_sort() {
    let Some(gpu) = require_gpu() else { return };
    let keys = random_keys(1000, 0x1234_5678);
    assert_sorted_equal("1000 keys", &gpu_sort(gpu, &keys), &keys);
}

#[test]
fn radix_sort_odd_sizes_match_cpu_sort() {
    let Some(gpu) = require_gpu() else { return };
    for &n in &[1usize, 2, 255, 256, 257, 511, 4097] {
        let keys = random_keys(n, n as u32 * 7 + 3);
        assert_sorted_equal(&format!("n={n}"), &gpu_sort(gpu, &keys), &keys);
    }
}

#[test]
fn radix_sort_a_million_keys_with_many_duplicates_matches_cpu_sort() {
    let Some(gpu) = require_gpu() else { return };
    let n = 1_000_000usize;
    // Mix: fully random keys, a few hundred distinct values (heavy
    // duplication), and keys that share all but one digit.
    let r = random_keys(n, 0xDEAD_BEEF);
    let keys: Vec<u32> = r
        .iter()
        .enumerate()
        .map(|(i, &x)| match i % 3 {
            0 => x,
            1 => (x % 300).wrapping_mul(0x0101_0101),
            _ => 0xAB00_00CD | ((x & 0xFF) << 8),
        })
        .collect();
    assert_sorted_equal("1e6 keys", &gpu_sort(gpu, &keys), &keys);
}

#[test]
fn radix_sort_key_value_is_stable_and_permutes_values_with_keys() {
    let Some(gpu) = require_gpu() else { return };
    let n = 300_000usize;
    // Few distinct keys -> long runs of equal keys whose values must stay in
    // input order. Values are the input indices.
    let keys: Vec<u32> = random_keys(n, 42)
        .iter()
        .map(|x| (x % 1000).wrapping_mul(0x0100_0101))
        .collect();
    let vals: Vec<u32> = (0..n as u32).collect();

    let (ks, vs) = gpu_sort_kv(gpu, &keys, &vals);

    let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
    expected.sort_by_key(|&(k, _)| k); // stable
    let exp_k: Vec<u32> = expected.iter().map(|p| p.0).collect();
    let exp_v: Vec<u32> = expected.iter().map(|p| p.1).collect();

    assert_eq!(ks, exp_k, "keys not sorted");
    let consistent = vs.iter().zip(&ks).all(|(&v, &k)| keys[v as usize] == k);
    assert!(consistent, "values not permuted together with their keys");
    let unstable = vs.iter().zip(&exp_v).filter(|(a, b)| a != b).count();
    assert_eq!(unstable, 0, "{unstable} values out of stable order");
}

#[test]
fn latent_radix_sort_module_still_validates() {
    // radix_sort.wgsl has no host today (ShaderLibrary only); make sure the
    // stable-scatter edit keeps it a valid module.
    let Some(gpu) = require_gpu() else { return };
    let lib = cv_hal::gpu_kernels::shaders::ShaderLibrary::new();
    let src = *lib.get("radix_sort").unwrap();
    gpu.create_compute_pipeline(src, "histogram");
    gpu.create_compute_pipeline(src, "scatter");
}

#[test]
fn morton_normals_pipeline_recovers_sphere_normals() {
    // compute_normals_morton_gpu orders points by Morton code with the KV
    // radix sort and builds an LBVH over the sorted codes; scrambled ordering
    // yields neighbourhoods from all over the cloud. On a unit sphere every
    // normal must be (anti)parallel to its point.
    let Some(gpu) = require_gpu() else { return };
    let n = 20_000usize;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let points: Vec<nalgebra::Vector3<f32>> = (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).sqrt();
            let t = golden * i as f32;
            nalgebra::Vector3::new(r * t.cos(), y, r * t.sin())
        })
        .collect();
    let normals =
        cv_hal::gpu_kernels::pointcloud_gpu::compute_normals_morton_gpu(gpu, &points, 10).unwrap();
    assert_eq!(normals.len(), n);
    let bad = points
        .iter()
        .zip(&normals)
        .filter(|(p, nrm)| !(nrm.dot(p).abs() > 0.95))
        .count();
    assert_eq!(bad, 0, "{bad} of {n} normals not radial");
}
