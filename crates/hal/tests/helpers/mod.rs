//! Test helper utilities for cv-hal tests
//!
//! Provides common utilities for testing including:
//! - Tensor creation helpers
//! - CPU/GPU comparison utilities
//! - Random test data generators
//! - Reference data fixtures

use cv_core::storage::CpuStorage;
use cv_core::tensor::{DataType, Tensor};
use cv_core::TensorShape;
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;

/// Try to get GPU context, initializing if needed.
/// Returns None if no GPU is available.
pub fn get_gpu_context() -> Option<&'static GpuContext> {
    // Try to get global context first
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }

    // Initialize if not initialized yet
    match pollster::block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(_) => None,
    }
}

/// Get GPU context with verbose output showing which backend is used.
/// Prints the backend type to help verify GPU execution.
pub fn get_gpu_context_verbose(test_name: &str) -> Option<&'static GpuContext> {
    match get_gpu_context() {
        Some(ctx) => {
            println!(
                "✓ {}: Using GPU backend: {:?}",
                test_name,
                ctx.backend_type()
            );
            Some(ctx)
        }
        None => {
            println!("✗ {}: Skipping - no GPU available", test_name);
            None
        }
    }
}

/// Simple pseudo-random number generator for tests
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f32) / (u32::MAX >> 9) as f32
    }

    pub fn next_f64(&mut self) -> f64 {
        self.next_f32() as f64
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next_f32() * 256.0) as u8
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_f32() as u32 * u32::MAX
    }
}

/// Create a test tensor with sequential f32 values
pub fn sequential_f32_tensor(w: usize, h: usize, c: usize) -> Tensor<f32, CpuStorage<f32>> {
    let size = w * h * c;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    create_f32_tensor(&data, w, h, c)
}

/// Create a test tensor with random f32 values
pub fn random_f32_tensor(w: usize, h: usize, c: usize, seed: u64) -> Tensor<f32, CpuStorage<f32>> {
    let size = w * h * c;
    let mut rng = SimpleRng::new(seed);
    let data: Vec<f32> = (0..size).map(|_| rng.next_f32() * 100.0).collect();
    create_f32_tensor(&data, w, h, c)
}

/// Create a test tensor with random u8 values
pub fn random_u8_tensor(w: usize, h: usize, c: usize, seed: u64) -> Tensor<u8, CpuStorage<u8>> {
    let size = w * h * c;
    let mut rng = SimpleRng::new(seed);
    let data: Vec<u8> = (0..size).map(|_| rng.next_u8()).collect();
    create_u8_tensor(&data, w, h, c)
}

/// Create a constant f32 tensor
pub fn constant_f32_tensor(
    value: f32,
    w: usize,
    h: usize,
    c: usize,
) -> Tensor<f32, CpuStorage<f32>> {
    let size = w * h * c;
    let data = vec![value; size];
    create_f32_tensor(&data, w, h, c)
}

/// Create a constant u8 tensor
pub fn constant_u8_tensor(value: u8, w: usize, h: usize, c: usize) -> Tensor<u8, CpuStorage<u8>> {
    let size = w * h * c;
    let data = vec![value; size];
    create_u8_tensor(&data, w, h, c)
}

/// Create an f32 tensor from raw data
pub fn create_f32_tensor(
    data: &[f32],
    w: usize,
    h: usize,
    c: usize,
) -> Tensor<f32, CpuStorage<f32>> {
    let storage = CpuStorage::from_vec(data.to_vec()).unwrap();
    Tensor {
        storage,
        shape: TensorShape::new(c, h, w),
        dtype: DataType::F32,
        _phantom: std::marker::PhantomData,
    }
}

/// Create a u8 tensor from raw data
pub fn create_u8_tensor(data: &[u8], w: usize, h: usize, c: usize) -> Tensor<u8, CpuStorage<u8>> {
    let storage = CpuStorage::from_vec(data.to_vec()).unwrap();
    Tensor {
        storage,
        shape: TensorShape::new(c, h, w),
        dtype: DataType::U8,
        _phantom: std::marker::PhantomData,
    }
}

/// Compute mean squared error between two f32 arrays
pub fn compute_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

/// Compute peak signal-to-noise ratio
pub fn compute_psnr(a: &[f32], b: &[f32]) -> f32 {
    let mse = compute_mse(a, b);
    if mse < 1e-10 {
        return f32::MAX;
    }
    let max_val = a.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
    20.0 * (max_val / mse.sqrt()).log10()
}

/// Check if two tensors are approximately equal
pub fn tensors_close(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
}

/// Compute variance of a dataset
pub fn compute_variance(data: &[f32]) -> f32 {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

/// Compute standard deviation
pub fn compute_std(data: &[f32]) -> f32 {
    compute_variance(data).sqrt()
}

/// Generate a Gaussian kernel for testing
pub fn gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = Vec::with_capacity(size * size);
    let half = size / 2;
    let mut sum = 0.0f32;

    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32) - (half as f32);
            let dy = (y as f32) - (half as f32);
            let value = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            sum += value;
        }
    }

    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

/// Test patterns for algorithm validation

pub mod patterns {
    /// Create a checkerboard pattern
    pub fn checkerboard(w: usize, h: usize, c: usize, square_size: usize) -> Vec<f32> {
        let mut data = vec![0.0; w * h * c];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let val = if ((x / square_size) + (y / square_size)) % 2 == 0 {
                    1.0
                } else {
                    0.0
                };
                for ch in 0..c {
                    data[idx * c + ch] = val;
                }
            }
        }
        data
    }

    /// Create a gradient pattern
    pub fn gradient(w: usize, h: usize, c: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(w * h * c);
        for y in 0..h {
            for x in 0..w {
                let val = (x as f32) + (y as f32) * 0.1;
                for _ in 0..c {
                    data.push(val);
                }
            }
        }
        data
    }

    /// Create a radial gradient
    pub fn radial_gradient(w: usize, h: usize, c: usize) -> Vec<f32> {
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();
        let mut data = Vec::with_capacity(w * h * c);

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = 1.0 - (dist / max_dist);
                for _ in 0..c {
                    data.push(val);
                }
            }
        }
        data
    }

    /// Create a step edge
    pub fn step_edge(w: usize, h: usize, c: usize, edge_x: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(w * h * c);
        for _ in 0..h {
            for x in 0..w {
                let val = if x >= edge_x { 1.0 } else { 0.0 };
                for _ in 0..c {
                    data.push(val);
                }
            }
        }
        data
    }

    /// Create a circle pattern
    pub fn circle(w: usize, h: usize, c: usize, cx: usize, cy: usize, radius: usize) -> Vec<f32> {
        let mut data = vec![0.0; w * h * c];
        let r2 = (radius * radius) as f32;

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist2 = dx * dx + dy * dy;
                let idx = y * w + x;
                for ch in 0..c {
                    data[idx * c + ch] = if dist2 <= r2 { 1.0 } else { 0.0 };
                }
            }
        }
        data
    }
}
