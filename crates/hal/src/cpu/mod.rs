use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, Result};
use cv_core::{storage::Storage, Float, Tensor, TensorShape};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use wide::*;

pub mod border;
pub mod simd;
pub mod utils;

static NEXT_CPU_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Debug)]
pub struct CpuBackend {
    device_id: DeviceId,
    num_threads: usize,
    simd_available: bool,
}

impl CpuBackend {
    pub fn new() -> Option<Self> {
        let num_threads = std::env::var("RUSTCV_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(rayon::current_num_threads);

        let device_id = DeviceId(NEXT_CPU_ID.fetch_add(1, Ordering::Relaxed));

        Some(Self {
            device_id,
            num_threads,
            simd_available: true,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn is_available() -> bool {
        true
    }
}

#[allow(clippy::needless_range_loop)]
pub fn gaussian_kernel_1d<T: Float>(sigma: T, size: usize) -> Vec<T> {
    let mut kernel = vec![T::ZERO; size];
    let radius = size / 2;
    let mut sum = T::ZERO;
    let two = T::from_f32(2.0);
    for i in 0..size {
        let x = T::from_f32(i as f32 - radius as f32);
        kernel[i] = (-(x * x) / (two * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for i in 0..size {
        kernel[i] /= sum;
    }
    kernel
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn supports(&self, capability: Capability) -> bool {
        match capability {
            Capability::Compute => true,
            Capability::Simd => self.simd_available,
            Capability::TensorCore => false,
            Capability::RayTracing => false,
        }
    }

    fn queue(&self, _queue_type: QueueType) -> QueueId {
        QueueId(0)
    }

    fn preferred_queue(&self) -> QueueType {
        QueueType::Compute
    }
}

impl CpuBackend {
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn convolve_separable<T: Float + bytemuck::Pod + std::fmt::Debug>(
        &self,
        src: &[T],
        dst: &mut [T],
        w: usize,
        h: usize,
        kx: &[T],
        ky: &[T],
    ) -> Result<()> {
        let rx = kx.len() / 2;
        let ry = ky.len() / 2;

        let pool = cv_core::BufferPool::global();
        let required_bytes = w * h * std::mem::size_of::<T>();
        let mut intermediate_vec = pool.get(required_bytes);

        if intermediate_vec.capacity() < required_bytes {
            return Err(crate::Error::MemoryError(format!(
                "Buffer pool returned insufficient buffer for separable convolution (capacity {} < required {})",
                intermediate_vec.capacity(), required_bytes
            )));
        }

        intermediate_vec.resize(required_bytes, 0);
        let intermediate: &mut [T] =
            bytemuck::cast_slice_mut(&mut intermediate_vec[..required_bytes]);

        // Horizontal pass
        intermediate
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row_inter)| {
                let row_src = &src[y * w..(y + 1) * w];
                for x in 0..w {
                    let mut sum = T::ZERO;
                    for i in 0..kx.len() {
                        let sx = (x as isize + i as isize - rx as isize).clamp(0, w as isize - 1)
                            as usize;
                        sum += row_src[sx] * kx[i];
                    }
                    row_inter[x] = sum;
                }
            });

        // Vertical pass
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_dst)| {
            for x in 0..w {
                let mut sum = T::ZERO;
                for j in 0..ky.len() {
                    let sy =
                        (y as isize + j as isize - ry as isize).clamp(0, h as isize - 1) as usize;
                    sum += intermediate[sy * w + x] * ky[j];
                }
                row_dst[x] = sum;
            }
        });

        pool.return_buffer(intermediate_vec);
        Ok(())
    }
}

include!("compute_context_impl.rs");
