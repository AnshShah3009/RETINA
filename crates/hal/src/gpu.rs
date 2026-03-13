use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, DeviceId, SubmissionIndex};
use cv_core::{storage::Storage, Tensor};
use futures::executor::block_on;
use futures::FutureExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{Backends, Device, Instance, PowerPreference, Queue, RequestAdapterOptions};

static GLOBAL_CONTEXT: OnceLock<crate::Result<GpuContext>> = OnceLock::new();

/// Shared GPU Context containing Device and Queue.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pipeline_cache: Arc<std::sync::Mutex<std::collections::HashMap<String, wgpu::ComputePipeline>>>,
    last_submission: Arc<AtomicU64>,
}

impl GpuContext {
    /// Safely downcasts a GpuStorage result to the requested generic storage S.
    /// Uses TypeId checks to avoid allocations when S is GpuStorage.
    #[allow(dead_code)]
    pub(crate) fn downcast_storage<
        T: Clone + Copy + std::fmt::Debug + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
    >(
        &self,
        result_gpu: Tensor<T, crate::storage::GpuStorage<T>>,
    ) -> crate::Result<Tensor<T, S>> {
        use std::marker::PhantomData;

        // Always use downcasting for safety.
        // The performance cost is negligible compared to GPU execution time.
        let storage_any = result_gpu.storage.as_any();

        if let Some(storage_s) = storage_any.downcast_ref::<S>() {
            Ok(Tensor {
                storage: storage_s.clone(),
                shape: result_gpu.shape,
                dtype: result_gpu.dtype,
                _phantom: PhantomData,
            })
        } else {
            Err(crate::Error::InvalidInput(
                "Failed to downcast GPU result to original storage type".into(),
            ))
        }
    }
}

impl ComputeContext for GpuContext {
    fn backend_type(&self) -> BackendType {
        BackendType::WebGPU
    }

    fn device_id(&self) -> DeviceId {
        DeviceId(0)
    }

    fn wait_idle(&self) -> crate::Result<()> {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        Ok(())
    }

    fn last_submission_index(&self) -> SubmissionIndex {
        SubmissionIndex(self.last_submission.load(Ordering::SeqCst))
    }

    fn convolve_2d<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        kernel: &Tensor<T, S>,
        border_mode: BorderMode<T>,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(input_storage), Some(kernel_storage)) = (
                input.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                kernel.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            ) {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };
                let kernel_gpu = Tensor {
                    storage: kernel_storage.clone(),
                    shape: kernel.shape,
                    dtype: kernel.dtype,
                    _phantom: PhantomData,
                };

                let border_f32: crate::context::BorderMode<f32> = match border_mode {
                    crate::context::BorderMode::Constant(v) => {
                        crate::context::BorderMode::Constant(v.to_f32())
                    }
                    crate::context::BorderMode::Replicate => crate::context::BorderMode::Replicate,
                    crate::context::BorderMode::Reflect => crate::context::BorderMode::Reflect,
                    crate::context::BorderMode::Wrap => crate::context::BorderMode::Wrap,
                };
                let result_gpu = crate::gpu_kernels::convolve::convolve_2d(
                    self,
                    &input_gpu,
                    &kernel_gpu,
                    border_f32,
                )?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU convolve_2d only supports f32".into(),
            ))
        }
    }

    fn dispatch<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        shader_source: &str,
        buffers: &[&Tensor<u8, S>],
        uniforms: &[u8],
        workgroups: (u32, u32, u32),
    ) -> crate::Result<()> {
        use crate::storage::GpuStorage;
        let pipeline = self.create_compute_pipeline(shader_source, "main");

        let mut gpu_buffers = Vec::with_capacity(buffers.len());
        for tensor in buffers {
            let storage = tensor
                .storage
                .as_any()
                .downcast_ref::<GpuStorage<u8>>()
                .ok_or_else(|| {
                    crate::Error::InvalidInput("Dispatch requires GpuStorage tensors".into())
                })?;
            gpu_buffers.push(storage.buffer());
        }

        let mut entries = Vec::with_capacity(gpu_buffers.len() + 1);
        for (i, buf) in gpu_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let _uniform_buf;
        if !uniforms.is_empty() {
            _uniform_buf = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Dispatch Uniforms"),
                    contents: uniforms,
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            ));
            entries.push(wgpu::BindGroupEntry {
                binding: gpu_buffers.len() as u32,
                resource: _uniform_buf.as_ref().unwrap().as_entire_binding(),
            });
        } else {
            _uniform_buf = None;
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dispatch Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.submit(encoder);

        Ok(())
    }

    fn threshold<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        thresh: T,
        max_value: T,
        typ: ThresholdType,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
            let input_gpu = crate::GpuTensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu =
                crate::gpu_kernels::threshold::threshold(self, &input_gpu, thresh, max_value, typ)?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into(),
            ))
        }
    }

    fn sobel<
        T: cv_core::Float + bytemuck::Pod + std::fmt::Debug + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> crate::Result<(Tensor<T, S>, Tensor<T, S>)> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let (gx_gpu, gy_gpu) =
                    crate::gpu_kernels::sobel::sobel(self, &input_gpu, dx, dy, ksize)?;
                let gx_s = gx_gpu
                    .storage
                    .as_any()
                    .downcast_ref::<S>()
                    .ok_or_else(|| crate::Error::InvalidInput("Failed to downcast gx".into()))?
                    .clone();
                let gy_s = gy_gpu
                    .storage
                    .as_any()
                    .downcast_ref::<S>()
                    .ok_or_else(|| crate::Error::InvalidInput("Failed to downcast gy".into()))?
                    .clone();
                Ok((
                    Tensor {
                        storage: gx_s,
                        shape: gx_gpu.shape,
                        dtype: gx_gpu.dtype,
                        _phantom: PhantomData,
                    },
                    Tensor {
                        storage: gy_s,
                        shape: gy_gpu.shape,
                        dtype: gy_gpu.dtype,
                        _phantom: PhantomData,
                    },
                ))
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU sobel only supports f32".into(),
            ))
        }
    }

    fn canny<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        low_threshold: T,
        high_threshold: T,
    ) -> crate::Result<Tensor<T, S>> {
        use std::marker::PhantomData;

        if let Some(input_storage) = input
            .storage
            .as_any()
            .downcast_ref::<crate::storage::GpuStorage<f32>>()
        {
            let input_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::canny::canny(
                self,
                &input_gpu,
                low_threshold.to_f32(),
                high_threshold.to_f32(),
            )?;
            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast canny GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn hough_lines<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        rho: T,
        theta: T,
        threshold: u32,
    ) -> crate::Result<Vec<cv_core::HoughLine>> {
        use std::marker::PhantomData;

        if let Some(input_storage) = input
            .storage
            .as_any()
            .downcast_ref::<crate::storage::GpuStorage<f32>>()
        {
            let input_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let lines = crate::gpu_kernels::hough::hough_lines(
                self,
                &input_gpu,
                rho.to_f32(),
                theta.to_f32(),
                threshold,
            )?;
            Ok(bytemuck::cast_vec(lines))
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn hough_circles<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        min_radius: T,
        max_radius: T,
        threshold: u32,
    ) -> crate::Result<Vec<cv_core::HoughCircle>> {
        use std::marker::PhantomData;

        if let Some(input_storage) = input
            .storage
            .as_any()
            .downcast_ref::<crate::storage::GpuStorage<f32>>()
        {
            let input_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let circles = crate::gpu_kernels::hough_circles::hough_circles(
                self,
                &input_gpu,
                min_radius.to_f32(),
                max_radius.to_f32(),
                threshold,
            )?;
            Ok(bytemuck::cast_vec(circles))
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn match_template<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        template: &Tensor<T, S>,
        method: TemplateMatchMethod,
    ) -> crate::Result<Tensor<T, OS>> {
        use std::marker::PhantomData;

        if let (Some(img_storage), Some(templ_storage)) = (
            image
                .storage
                .as_any()
                .downcast_ref::<crate::storage::GpuStorage<f32>>(),
            template
                .storage
                .as_any()
                .downcast_ref::<crate::storage::GpuStorage<f32>>(),
        ) {
            let img_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: img_storage.clone(),
                shape: image.shape,
                dtype: image.dtype,
                _phantom: PhantomData,
            };
            let templ_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: templ_storage.clone(),
                shape: template.shape,
                dtype: template.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::template_matching::match_template(
                self, &img_gpu, &templ_gpu, method,
            )?;
            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<OS>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast match_template GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn detect_objects<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _threshold: T,
    ) -> crate::Result<Vec<cv_core::Detection>> {
        Err(crate::Error::NotSupported(
            "GPU detect_objects not yet implemented".into(),
        ))
    }

    fn stereo_match<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        left: &Tensor<T, S>,
        right: &Tensor<T, S>,
        params: &StereoMatchParams,
    ) -> crate::Result<Tensor<T, OS>> {
        use std::marker::PhantomData;

        if let (Some(l_storage), Some(r_storage)) = (
            left.storage
                .as_any()
                .downcast_ref::<crate::storage::GpuStorage<f32>>(),
            right
                .storage
                .as_any()
                .downcast_ref::<crate::storage::GpuStorage<f32>>(),
        ) {
            let l_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: l_storage.clone(),
                shape: left.shape,
                dtype: left.dtype,
                _phantom: PhantomData,
            };
            let r_gpu: crate::GpuTensor<f32> = cv_core::Tensor {
                storage: r_storage.clone(),
                shape: right.shape,
                dtype: right.dtype,
                _phantom: PhantomData,
            };

            let result_gpu =
                crate::gpu_kernels::stereo::stereo_match(self, &l_gpu, &r_gpu, params)?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<OS>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn triangulate_points<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _proj_left: &[[T; 4]; 3],
        _proj_right: &[[T; 4]; 3],
        _points_left: &Tensor<T, S>,
        _points_right: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, OS>> {
        Err(crate::Error::NotSupported(
            "GPU triangulate_points not yet implemented".into(),
        ))
    }

    fn find_chessboard_corners<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _image: &Tensor<T, S>,
        _pattern_size: (usize, usize),
    ) -> crate::Result<Vec<cv_core::KeyPoint>> {
        Err(crate::Error::NotSupported(
            "GPU find_chessboard_corners not yet implemented".into(),
        ))
    }

    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> crate::Result<Tensor<u8, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(input_storage), Some(kernel_storage)) = (
            input.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
            kernel.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
        ) {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };
            let kernel_gpu = Tensor {
                storage: kernel_storage.clone(),
                shape: kernel.shape,
                dtype: kernel.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::morphology::morphology(
                self,
                &input_gpu,
                typ,
                &kernel_gpu,
                iterations,
            )?;
            // Safety: we know T is compatible here if we reached this point with GpuStorage<u8>
            // but this is tricky with the generic T.
            // Given the HAL structure, we'll likely only call this with T=f32 or u8.
            // If it's f32 and we have u8 storage, we have a problem.
            // Let's assume for now morphology on GPU is f32-compatible or we use casts.
            // Actually, the previous code used GpuStorage<u8>.
            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "Morphology requires GpuStorage<u8>".into(),
            ))
        }
    }

    fn warp<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        matrix: &[[T; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let matrix_f32: [[f32; 3]; 3] = [
                    [
                        matrix[0][0].to_f32(),
                        matrix[0][1].to_f32(),
                        matrix[0][2].to_f32(),
                    ],
                    [
                        matrix[1][0].to_f32(),
                        matrix[1][1].to_f32(),
                        matrix[1][2].to_f32(),
                    ],
                    [
                        matrix[2][0].to_f32(),
                        matrix[2][1].to_f32(),
                        matrix[2][2].to_f32(),
                    ],
                ];

                let result_gpu =
                    crate::gpu_kernels::warp::warp(self, &input_gpu, &matrix_f32, new_shape, typ)?;
                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast warp GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU warp only supports f32".into(),
            ))
        }
    }

    fn nms<T: cv_core::Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        window_size: usize,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::nms::nms_pixel(
                    self,
                    &input_gpu,
                    threshold.to_f32(),
                    window_size,
                )?;
                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast nms GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU nms only supports f32".into(),
            ))
        }
    }

    fn nms_boxes<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> crate::Result<Vec<usize>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                crate::gpu_kernels::nms::nms_boxes(self, &input_gpu, iou_threshold.to_f32())
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU nms_boxes only supports f32".into(),
            ))
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn nms_rotated_boxes<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> crate::Result<Vec<usize>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;

        // Fallback: Download to CPU and compute
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                // Read data to CPU
                let size = input.storage.len() * 4;
                let raw_data = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                    self.device.clone(),
                    &self.queue,
                    input_storage.buffer(),
                    0,
                    size,
                ))?;

                let rows = input.shape.height;
                let cols = input.shape.width;
                if cols != 6 {
                    return Err(crate::Error::InvalidInput(
                        "NMS Rotated Boxes expects (N, 6) tensor".into(),
                    ));
                }

                let mut boxes: Vec<(usize, f32)> = (0..rows)
                    .map(|i| {
                        (i, raw_data[i * 6 + 5]) // (index, score)
                    })
                    .collect();

                boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut kept = Vec::new();
                let mut suppressed = vec![false; rows];

                for i in 0..boxes.len() {
                    let (idx1, _) = boxes[i];
                    if suppressed[idx1] {
                        continue;
                    }

                    kept.push(idx1);
                    let r1 = cv_core::RotatedRect::new(
                        raw_data[idx1 * 6],
                        raw_data[idx1 * 6 + 1],
                        raw_data[idx1 * 6 + 2],
                        raw_data[idx1 * 6 + 3],
                        raw_data[idx1 * 6 + 4],
                    );

                    for j in (i + 1)..boxes.len() {
                        let (idx2, _) = boxes[j];
                        if suppressed[idx2] {
                            continue;
                        }

                        let r2 = cv_core::RotatedRect::new(
                            raw_data[idx2 * 6],
                            raw_data[idx2 * 6 + 1],
                            raw_data[idx2 * 6 + 2],
                            raw_data[idx2 * 6 + 3],
                            raw_data[idx2 * 6 + 4],
                        );

                        if cv_core::rotated_iou(&r1, &r2) > iou_threshold.to_f32() {
                            suppressed[idx2] = true;
                        }
                    }
                }
                Ok(kept)
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU nms_rotated_boxes fallback only supports f32".into(),
            ))
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn nms_polygons<T: cv_core::Float + 'static>(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[T],
        iou_threshold: T,
    ) -> crate::Result<Vec<usize>> {
        // CPU implementation since polygons are already on CPU
        let n = polygons.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if scores.len() != n {
            return Err(crate::Error::InvalidInput(
                "Scores length must match polygons length".into(),
            ));
        }

        let mut items: Vec<(usize, T)> = (0..n).map(|i| (i, scores[i])).collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; n];

        for i in 0..n {
            let (idx1, _) = items[i];
            if suppressed[idx1] {
                continue;
            }

            kept.push(idx1);
            let p1 = &polygons[idx1];

            for j in (i + 1)..n {
                let (idx2, _) = items[j];
                if suppressed[idx2] {
                    continue;
                }

                let p2 = &polygons[idx2];
                if cv_core::polygon_iou(p1, p2) > iou_threshold.to_f32() {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    fn pointcloud_normals<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = points.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: points.shape,
                    dtype: points.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu =
                    crate::gpu_kernels::pointcloud::compute_normals(self, &input_gpu, k_neighbors)?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU pointcloud_normals only supports f32".into(),
            ))
        }
    }

    fn tsdf_integrate<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        depth_image: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        voxel_volume: &mut Tensor<T, S>,
        voxel_size: T,
        truncation: T,
    ) -> crate::Result<()> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(depth_storage), Some(voxel_storage)) = (
                depth_image
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>(),
                voxel_volume
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>(),
            ) {
                let depth_gpu = Tensor {
                    storage: depth_storage.clone(),
                    shape: depth_image.shape,
                    dtype: depth_image.dtype,
                    _phantom: PhantomData,
                };
                let mut voxel_gpu = Tensor {
                    storage: voxel_storage.clone(),
                    shape: voxel_volume.shape,
                    dtype: voxel_volume.dtype,
                    _phantom: PhantomData,
                };

                let camera_pose_f32: [[f32; 4]; 4] = [
                    [
                        camera_pose[0][0].to_f32(),
                        camera_pose[0][1].to_f32(),
                        camera_pose[0][2].to_f32(),
                        camera_pose[0][3].to_f32(),
                    ],
                    [
                        camera_pose[1][0].to_f32(),
                        camera_pose[1][1].to_f32(),
                        camera_pose[1][2].to_f32(),
                        camera_pose[1][3].to_f32(),
                    ],
                    [
                        camera_pose[2][0].to_f32(),
                        camera_pose[2][1].to_f32(),
                        camera_pose[2][2].to_f32(),
                        camera_pose[2][3].to_f32(),
                    ],
                    [
                        camera_pose[3][0].to_f32(),
                        camera_pose[3][1].to_f32(),
                        camera_pose[3][2].to_f32(),
                        camera_pose[3][3].to_f32(),
                    ],
                ];
                let intrinsics_f32: [f32; 4] = [
                    intrinsics[0].to_f32(),
                    intrinsics[1].to_f32(),
                    intrinsics[2].to_f32(),
                    intrinsics[3].to_f32(),
                ];

                crate::gpu_kernels::tsdf::integrate(
                    self,
                    &depth_gpu,
                    &camera_pose_f32,
                    &intrinsics_f32,
                    &mut voxel_gpu,
                    voxel_size.to_f32(),
                    truncation.to_f32(),
                )
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU tsdf_integrate only supports f32".into(),
            ))
        }
    }

    fn tsdf_raycast<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        tsdf_volume: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        image_size: (u32, u32),
        depth_range: (T, T),
        voxel_size: T,
        truncation: T,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = tsdf_volume
                .storage
                .as_any()
                .downcast_ref::<GpuStorage<f32>>()
            {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: tsdf_volume.shape,
                    dtype: tsdf_volume.dtype,
                    _phantom: PhantomData,
                };

                let camera_pose_f32: [[f32; 4]; 4] = [
                    [
                        camera_pose[0][0].to_f32(),
                        camera_pose[0][1].to_f32(),
                        camera_pose[0][2].to_f32(),
                        camera_pose[0][3].to_f32(),
                    ],
                    [
                        camera_pose[1][0].to_f32(),
                        camera_pose[1][1].to_f32(),
                        camera_pose[1][2].to_f32(),
                        camera_pose[1][3].to_f32(),
                    ],
                    [
                        camera_pose[2][0].to_f32(),
                        camera_pose[2][1].to_f32(),
                        camera_pose[2][2].to_f32(),
                        camera_pose[2][3].to_f32(),
                    ],
                    [
                        camera_pose[3][0].to_f32(),
                        camera_pose[3][1].to_f32(),
                        camera_pose[3][2].to_f32(),
                        camera_pose[3][3].to_f32(),
                    ],
                ];
                let intrinsics_f32: [f32; 4] = [
                    intrinsics[0].to_f32(),
                    intrinsics[1].to_f32(),
                    intrinsics[2].to_f32(),
                    intrinsics[3].to_f32(),
                ];
                let depth_range_f32 = (depth_range.0.to_f32(), depth_range.1.to_f32());

                let result_gpu = crate::gpu_kernels::tsdf::raycast(
                    self,
                    &input_gpu,
                    &camera_pose_f32,
                    &intrinsics_f32,
                    image_size,
                    depth_range_f32,
                    voxel_size.to_f32(),
                    truncation.to_f32(),
                )?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU tsdf_raycast only supports f32".into(),
            ))
        }
    }

    fn tsdf_extract_mesh<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        voxel_volume: &Tensor<T, S>,
        voxel_size: T,
        iso_level: T,
        max_triangles: u32,
    ) -> crate::Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = voxel_volume
                .storage
                .as_any()
                .downcast_ref::<GpuStorage<f32>>()
            {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: voxel_volume.shape,
                    dtype: voxel_volume.dtype,
                    _phantom: PhantomData,
                };

                crate::gpu_kernels::marching_cubes::extract_mesh(
                    self,
                    &input_gpu,
                    voxel_size.to_f32(),
                    iso_level.to_f32(),
                    max_triangles,
                )
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU tsdf_extract_mesh only supports f32".into(),
            ))
        }
    }

    fn optical_flow_lk<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        prev_pyramid: &[Tensor<T, S>],
        next_pyramid: &[Tensor<T, S>],
        points: &[[T; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> crate::Result<Vec<[T; 2]>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Verify all tensors in pyramids are GpuStorage<f32>
            let mut prev_gpu = Vec::with_capacity(prev_pyramid.len());
            for t in prev_pyramid {
                if let Some(s) = t.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                    prev_gpu.push(Tensor {
                        storage: s.clone(),
                        shape: t.shape,
                        dtype: t.dtype,
                        _phantom: PhantomData,
                    });
                } else {
                    return Err(crate::Error::InvalidInput(
                        "GpuContext requires GpuStorage tensors".into(),
                    ));
                }
            }

            let mut next_gpu = Vec::with_capacity(next_pyramid.len());
            for t in next_pyramid {
                if let Some(s) = t.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                    next_gpu.push(Tensor {
                        storage: s.clone(),
                        shape: t.shape,
                        dtype: t.dtype,
                        _phantom: PhantomData,
                    });
                } else {
                    return Err(crate::Error::InvalidInput(
                        "GpuContext requires GpuStorage tensors".into(),
                    ));
                }
            }

            // Safety: T == f32 verified above; points slice is &[T] = &[f32]
            let points_f32: &[[f32; 2]] = unsafe {
                std::slice::from_raw_parts(points.as_ptr() as *const [f32; 2], points.len() / 2)
            };
            let results = crate::gpu_kernels::optical_flow::lucas_kanade(
                self,
                &prev_gpu,
                &next_gpu,
                points_f32,
                window_size,
                max_iters,
            )?;

            // Safety: we verified T == f32, so Vec<[f32;2]> and Vec<[T;2]> have identical layout.
            let results_t: Vec<[T; 2]> = unsafe { std::mem::transmute(results) };
            Ok(results_t)
        } else {
            Err(crate::Error::NotSupported(
                "GPU optical_flow_lk only supports f32".into(),
            ))
        }
    }

    fn pyramid_down<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::pyramid::pyramid_down(self, &input_gpu)?;
                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast pyramid_down result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU pyramid_down only supports f32".into(),
            ))
        }
    }

    fn cvt_color<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        code: ColorConversion,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
            let input_gpu = crate::GpuTensor::<T> {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::color::color_convert(self, &input_gpu, code)?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn resize<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        new_shape: (usize, usize),
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
            let input_gpu = crate::GpuTensor::<T> {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::resize::resize(
                self,
                &input_gpu,
                new_shape.0 as u32,
                new_shape.1 as u32,
            )?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn bilateral_filter<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        d: i32,
        sigma_color: T,
        sigma_space: T,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
            let input_gpu = crate::GpuTensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::bilateral::bilateral_filter(
                self,
                &input_gpu,
                d,
                sigma_color,
                sigma_space,
            )?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn fast_detect<
        T: cv_core::Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        non_max_suppression: bool,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
            let input_gpu = crate::GpuTensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::fast::fast_detect(
                self,
                &input_gpu,
                threshold,
                non_max_suppression,
            )?;

            let storage_any = result_gpu.storage.as_any();
            if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                Ok(Tensor {
                    storage: storage_s.clone(),
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput(
                    "Failed to downcast GPU result".into(),
                ))
            }
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn gaussian_blur<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        sigma: T,
        k_size: usize,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let blurred_f32 = crate::gpu_kernels::convolve::gaussian_blur(
                    self,
                    &input_gpu,
                    sigma.to_f32(),
                    k_size,
                )?;

                let storage_any = blurred_f32.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: blurred_f32.shape,
                        dtype: blurred_f32.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU gaussian_blur only supports f32".into(),
            ))
        }
    }

    fn subtract<
        T: Clone + Copy + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(a_storage), Some(b_storage)) = (
                a.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                b.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            ) {
                let a_gpu = Tensor {
                    storage: a_storage.clone(),
                    shape: a.shape,
                    dtype: a.dtype,
                    _phantom: PhantomData,
                };
                let b_gpu = Tensor {
                    storage: b_storage.clone(),
                    shape: b.shape,
                    dtype: b.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::subtract::subtract(self, &a_gpu, &b_gpu)?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU subtract only supports f32 for now".into(),
            ))
        }
    }

    fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> crate::Result<cv_core::Matches> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(q_storage), Some(t_storage)) = (
            query.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
            train.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
        ) {
            let q_gpu = Tensor {
                storage: q_storage.clone(),
                shape: query.shape,
                dtype: query.dtype,
                _phantom: PhantomData,
            };
            let t_gpu = Tensor {
                storage: t_storage.clone(),
                shape: train.shape,
                dtype: train.dtype,
                _phantom: PhantomData,
            };

            crate::gpu_kernels::matching::match_descriptors(self, &q_gpu, &t_gpu, ratio_threshold)
        } else {
            Err(crate::Error::InvalidInput(
                "GpuContext requires GpuStorage tensors".into(),
            ))
        }
    }

    fn sift_extrema<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        dog_prev: &Tensor<T, S>,
        dog_curr: &Tensor<T, S>,
        dog_next: &Tensor<T, S>,
        threshold: T,
        edge_threshold: T,
    ) -> crate::Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        use crate::storage::GpuStorage;
        use crate::tensor_ext::TensorToCpu;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(p_storage), Some(c_storage), Some(n_storage)) = (
                dog_prev.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                dog_curr.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                dog_next.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            ) {
                let p_gpu = Tensor {
                    storage: p_storage.clone(),
                    shape: dog_prev.shape,
                    dtype: dog_prev.dtype,
                    _phantom: PhantomData,
                };
                let c_gpu = Tensor {
                    storage: c_storage.clone(),
                    shape: dog_curr.shape,
                    dtype: dog_curr.dtype,
                    _phantom: PhantomData,
                };
                let n_gpu = Tensor {
                    storage: n_storage.clone(),
                    shape: dog_next.shape,
                    dtype: dog_next.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::sift::sift_extrema(
                    self,
                    &p_gpu,
                    &c_gpu,
                    &n_gpu,
                    threshold.to_f32(),
                    edge_threshold.to_f32(),
                )?;
                result_gpu.to_cpu_ctx(self)
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU sift_extrema only supports f32".into(),
            ))
        }
    }

    fn compute_sift_descriptors<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> crate::Result<cv_core::Descriptors> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(img_storage) = image.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let img_gpu = Tensor {
                    storage: img_storage.clone(),
                    shape: image.shape,
                    dtype: image.dtype,
                    _phantom: PhantomData,
                };

                // 1. Upload keypoints to GPU
                let kp_data: Vec<[f32; 4]> = keypoints
                    .keypoints
                    .iter()
                    .map(|kp| [kp.x as f32, kp.y as f32, kp.size as f32, kp.angle as f32])
                    .collect();

                let kp_buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SIFT Keypoints Buffer"),
                        contents: bytemuck::cast_slice(&kp_data),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                // 2. Output buffer for descriptors (num_kps * 128 * 4 bytes)
                let out_byte_size = (kp_data.len() * 128 * 4) as u64;
                let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SIFT Descriptors Output"),
                    size: out_byte_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                let params = [
                    image.shape.width as u32,
                    image.shape.height as u32,
                    kp_data.len() as u32,
                    0u32,
                ];
                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("SIFT Desc Params"),
                            contents: bytemuck::cast_slice(&params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let shader_source = include_str!("../shaders/sift_descriptor.wgsl");
                let pipeline = self.create_compute_pipeline(shader_source, "main");

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SIFT Desc Bind Group"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: img_gpu.storage.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: kp_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    let x = (kp_data.len() as u32).div_ceil(64);
                    pass.dispatch_workgroups(x, 1, 1);
                }
                self.submit(encoder);

                // 3. Read back and convert to Descriptor objects
                let raw_descs: Vec<f32> =
                    pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                        self.device.clone(),
                        &self.queue,
                        &output_buffer,
                        0,
                        out_byte_size as usize,
                    ))?;

                let mut descs = Vec::with_capacity(keypoints.len());
                for i in 0..keypoints.len() {
                    let start = i * 128;
                    let data: Vec<u8> = raw_descs[start..start + 128]
                        .iter()
                        .map(|&v| (v * 512.0).min(255.0) as u8)
                        .collect();
                    descs.push(cv_core::Descriptor::new(data, keypoints.keypoints[i]));
                }

                Ok(cv_core::Descriptors { descriptors: descs })
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU compute_sift_descriptors only supports f32".into(),
            ))
        }
    }

    fn icp_correspondences<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        src: &Tensor<T, S>,
        tgt: &Tensor<T, S>,
        max_dist: T,
    ) -> crate::Result<Vec<(usize, usize, T)>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(src_storage), Some(tgt_storage)) = (
                src.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                tgt.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            ) {
                let src_gpu = Tensor {
                    storage: src_storage.clone(),
                    shape: src.shape,
                    dtype: src.dtype,
                    _phantom: PhantomData,
                };
                let tgt_gpu = Tensor {
                    storage: tgt_storage.clone(),
                    shape: tgt.shape,
                    dtype: tgt.dtype,
                    _phantom: PhantomData,
                };

                let res = crate::gpu_kernels::icp::icp_correspondences(
                    self,
                    &src_gpu,
                    &tgt_gpu,
                    max_dist.to_f32(),
                )?;

                // Convert Vec<(usize, usize, f32)> to Vec<(usize, usize, T)>
                Ok(res
                    .into_iter()
                    .map(|(a, b, d)| (a, b, T::from_f32(d)))
                    .collect())
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU icp_correspondences only supports f32".into(),
            ))
        }
    }

    fn icp_accumulate<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source: &Tensor<T, S>,
        target: &Tensor<T, S>,
        target_normals: &Tensor<T, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<T>,
    ) -> crate::Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(src_storage), Some(tgt_storage), Some(norm_storage)) = (
                source.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                target.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                target_normals
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>(),
            ) {
                let src_gpu = Tensor {
                    storage: src_storage.clone(),
                    shape: source.shape,
                    dtype: source.dtype,
                    _phantom: PhantomData,
                };
                let tgt_gpu = Tensor {
                    storage: tgt_storage.clone(),
                    shape: target.shape,
                    dtype: target.dtype,
                    _phantom: PhantomData,
                };
                let n_gpu = Tensor {
                    storage: norm_storage.clone(),
                    shape: target_normals.shape,
                    dtype: target_normals.dtype,
                    _phantom: PhantomData,
                };

                let transform_f32: nalgebra::Matrix4<f32> = transform.map(|v| v.to_f32());

                let (m6, v6) = crate::gpu_kernels::icp::icp_accumulate(
                    self,
                    &src_gpu,
                    &tgt_gpu,
                    &n_gpu,
                    correspondences,
                    &transform_f32,
                )?;

                Ok((m6.map(|v| T::from_f32(v)), v6.map(|v| T::from_f32(v))))
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU icp_accumulate only supports f32".into(),
            ))
        }
    }

    fn dense_icp_step<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source_depth: &Tensor<T, S>,
        target_data: &Tensor<T, S>,
        intrinsics: &[T; 4],
        initial_guess: &nalgebra::Matrix4<T>,
        max_dist: T,
        max_angle: T,
    ) -> crate::Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(src_storage), Some(tgt_storage)) = (
                source_depth
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>(),
                target_data
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>(),
            ) {
                let src_gpu = Tensor {
                    storage: src_storage.clone(),
                    shape: source_depth.shape,
                    dtype: source_depth.dtype,
                    _phantom: PhantomData,
                };
                let tgt_gpu = Tensor {
                    storage: tgt_storage.clone(),
                    shape: target_data.shape,
                    dtype: target_data.dtype,
                    _phantom: PhantomData,
                };

                let intrinsics_f32 = [
                    intrinsics[0].to_f32(),
                    intrinsics[1].to_f32(),
                    intrinsics[2].to_f32(),
                    intrinsics[3].to_f32(),
                ];
                let guess_f32 = initial_guess.map(|v| v.to_f32());

                let (m6, v6) = crate::gpu_kernels::icp::dense_step(
                    self,
                    &src_gpu,
                    &tgt_gpu,
                    &intrinsics_f32,
                    &guess_f32,
                    max_dist.to_f32(),
                    max_angle.to_f32(),
                )?;

                Ok((m6.map(|v| T::from_f32(v)), v6.map(|v| T::from_f32(v))))
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU dense_icp_step only supports f32".into(),
            ))
        }
    }

    fn akaze_diffusion<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        k: T,
        tau: T,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::akaze::akaze_diffusion(
                    self,
                    &input_gpu,
                    k.to_f32(),
                    tau.to_f32(),
                )?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU akaze_diffusion only supports f32".into(),
            ))
        }
    }

    fn akaze_derivatives<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> crate::Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let (lx_gpu, ly_gpu, ldet_gpu) =
                    crate::gpu_kernels::akaze::akaze_derivatives(self, &input_gpu)?;

                let lx_any = lx_gpu.storage.as_any();
                let ly_any = ly_gpu.storage.as_any();
                let ldet_any = ldet_gpu.storage.as_any();

                if let (Some(lx_s), Some(ly_s), Some(ldet_s)) = (
                    lx_any.downcast_ref::<S>(),
                    ly_any.downcast_ref::<S>(),
                    ldet_any.downcast_ref::<S>(),
                ) {
                    Ok((
                        Tensor {
                            storage: lx_s.clone(),
                            shape: lx_gpu.shape,
                            dtype: lx_gpu.dtype,
                            _phantom: PhantomData,
                        },
                        Tensor {
                            storage: ly_s.clone(),
                            shape: ly_gpu.shape,
                            dtype: ly_gpu.dtype,
                            _phantom: PhantomData,
                        },
                        Tensor {
                            storage: ldet_s.clone(),
                            shape: ldet_gpu.shape,
                            dtype: ldet_gpu.dtype,
                            _phantom: PhantomData,
                        },
                    ))
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU results".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU akaze_derivatives only supports f32".into(),
            ))
        }
    }

    fn akaze_contrast_k<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> crate::Result<T> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };

                let k = crate::gpu_kernels::akaze::akaze_contrast_k(self, &input_gpu)?;
                Ok(T::from_f32(k))
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU akaze_contrast_k only supports f32".into(),
            ))
        }
    }

    fn spmv<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[T],
        x: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(x_storage) = x.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let x_gpu = Tensor {
                    storage: x_storage.clone(),
                    shape: x.shape,
                    dtype: x.dtype,
                    _phantom: PhantomData,
                };

                let values_f32: &[f32] = bytemuck::cast_slice(values);

                let result_gpu = crate::gpu_kernels::sparse::spmv(
                    self,
                    row_ptr,
                    col_indices,
                    values_f32,
                    &x_gpu,
                )?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU spmv only supports f32".into(),
            ))
        }
    }
    fn mog2_update<
        T: cv_core::Float + bytemuck::Pod + 'static,
        S1: Storage<T> + 'static,
        S2: Storage<u32> + 'static,
    >(
        &self,
        frame: &Tensor<T, S1>,
        model: &mut Tensor<T, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params<T>,
    ) -> crate::Result<()> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(frame_storage), Some(model_storage), Some(mask_storage)) = (
                frame.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                model.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                mask.storage.as_any().downcast_ref::<GpuStorage<u32>>(),
            ) {
                let frame_gpu = Tensor {
                    storage: frame_storage.clone(),
                    shape: frame.shape,
                    dtype: frame.dtype,
                    _phantom: PhantomData,
                };
                let mut model_gpu = Tensor {
                    storage: model_storage.clone(),
                    shape: model.shape,
                    dtype: model.dtype,
                    _phantom: PhantomData,
                };
                let mut mask_gpu = Tensor {
                    storage: mask_storage.clone(),
                    shape: mask.shape,
                    dtype: mask.dtype,
                    _phantom: PhantomData,
                };

                // Convert Mog2Params<T> to Mog2Params<f32> for the kernel
                let params_f32 = crate::context::Mog2Params {
                    width: params.width,
                    height: params.height,
                    n_mixtures: params.n_mixtures,
                    alpha: params.alpha.to_f32(),
                    var_threshold: params.var_threshold.to_f32(),
                    background_ratio: params.background_ratio.to_f32(),
                    var_init: params.var_init.to_f32(),
                    var_min: params.var_min.to_f32(),
                    var_max: params.var_max.to_f32(),
                    _padding: params._padding,
                };

                crate::gpu_kernels::mog2_update(
                    self,
                    &frame_gpu,
                    &mut model_gpu,
                    &mut mask_gpu,
                    &params_f32,
                )
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU mog2_update only supports f32".into(),
            ))
        }
    }

    fn pointcloud_transform<
        T: cv_core::Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        transform: &[[T; 4]; 4],
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::any::TypeId;
        use std::marker::PhantomData;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let Some(input_storage) = points.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = crate::GpuTensor {
                    storage: input_storage.clone(),
                    shape: points.shape,
                    dtype: points.dtype,
                    _phantom: PhantomData::<f32>,
                };

                // Safe: we verified T is f32 via TypeId check above
                let transform_f32: &[[f32; 4]; 4] =
                    unsafe { &*(transform as *const [[T; 4]; 4] as *const [[f32; 4]; 4]) };

                let result_gpu = crate::gpu_kernels::pointcloud_transform::pointcloud_transform(
                    self,
                    &input_gpu,
                    transform_f32,
                )?;

                let storage_any = result_gpu.storage.as_any();
                if let Some(storage_s) = storage_any.downcast_ref::<S>() {
                    Ok(Tensor {
                        storage: storage_s.clone(),
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast GPU pointcloud_transform result".into(),
                    ))
                }
            } else {
                Err(crate::Error::InvalidInput(
                    "GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into(),
                ))
            }
        } else {
            Err(crate::Error::NotSupported(
                "GPU pointcloud_transform only supports f32".into(),
            ))
        }
    }
}

impl GpuContext {
    /// Get the global GPU context. Returns an error if not yet initialized.
    pub fn global() -> crate::Result<&'static GpuContext> {
        GLOBAL_CONTEXT
            .get()
            .ok_or_else(|| {
                crate::Error::InitError(
                    "GPU Context not initialized. Call init_global() first.".into(),
                )
            })?
            .as_ref()
            .map_err(|e| crate::Error::InitError(e.to_string()))
    }

    /// Initialize the global GPU context asynchronously.
    pub async fn init_global() -> crate::Result<&'static GpuContext> {
        let res = GLOBAL_CONTEXT.get_or_init(|| {
            // We still need a sync way to call the async new_async if we are in init_global
            // But init_global is itself async, so we can await it.
            // Wait, get_or_init doesn't support async closures.
            // We use a different pattern: initialize outside and then set.
            Box::pin(Self::new_async())
                .now_or_never()
                .unwrap_or_else(|| {
                    // If it wasn't ready immediately, we have a problem with get_or_init.
                    // Modern OnceCell/OnceLock don't easily support async.
                    // For now, we'll keep the block_on inside the init_global ONLY if called from a non-async thread,
                    // or use a mutex to guard the async init.
                    block_on(Self::new_async())
                })
        });
        res.as_ref()
            .map_err(|e| crate::Error::InitError(e.to_string()))
    }

    /// Initialize a new GPU context (synchronous wrapper).
    pub fn new() -> crate::Result<Self> {
        block_on(Self::new_async())
    }

    /// Initialize a new GPU context asynchronously.
    pub async fn new_async() -> crate::Result<Self> {
        Self::new_with_policy(PowerPreference::HighPerformance).await
    }

    pub async fn new_with_policy(preference: PowerPreference) -> crate::Result<Self> {
        // Create instance with conservative flags to avoid driver panics on integrated GPUs
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default()
                .difference(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                crate::Error::InitError(format!("Failed to find a suitable GPU adapter: {}", e))
            })?;

        Self::from_adapter(adapter).await
    }

    pub async fn from_adapter(adapter: wgpu::Adapter) -> crate::Result<Self> {
        // Request device
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("CV-HAL Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| crate::Error::InitError(format!("Failed to create GPU device: {}", e)))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            last_submission: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Check if a GPU is available.
    pub fn is_available() -> bool {
        block_on(Self::is_available_async())
    }

    /// Check if a GPU is available asynchronously.
    pub async fn is_available_async() -> bool {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        !instance
            .enumerate_adapters(Backends::all())
            .await
            .is_empty()
    }

    /// Enumerate all available adapters.
    pub async fn enumerate_adapters() -> Vec<wgpu::Adapter> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        instance.enumerate_adapters(Backends::all()).await
    }

    /// Get reference to device (convenience method)
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get Arc to device
    pub fn device_arc(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Get Arc to queue
    pub fn queue_arc(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    /// Submit a command encoder (convenience method)
    pub fn submit(&self, encoder: wgpu::CommandEncoder) -> SubmissionIndex {
        let index = self.last_submission.fetch_add(1, Ordering::SeqCst) + 1;
        self.queue.submit(std::iter::once(encoder.finish()));
        SubmissionIndex(index)
    }

    /// Create a simplified compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let cache_key = format!("{}:{}", shader_source, entry_point);
        if let Ok(cache) = self.pipeline_cache.lock() {
            if let Some(pipeline) = cache.get(&cache_key) {
                return pipeline.clone();
            }
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        if let Ok(mut cache) = self.pipeline_cache.lock() {
            cache.insert(cache_key, pipeline.clone());
        }

        pipeline
    }

    /// Get a pooled buffer.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        crate::gpu_kernels::buffer_utils::global_pool().get(&self.device, size, usage)
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        crate::gpu_kernels::buffer_utils::global_pool().return_buffer(&self.device, buffer, usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new();
        match ctx {
            Ok(c) => println!("GPU Context created: {:?}", c.device),
            Err(e) => println!("GPU initialization failed (expected on some CI): {}", e),
        }
    }

    #[test]
    fn test_gaussian_blur_gpu_parity() {
        let gpu = if let Ok(g) = GpuContext::global() {
            g
        } else {
            return;
        };
        let cpu = crate::cpu::CpuBackend::new().unwrap();

        let width = 64usize;
        let height = 64usize;
        let mut data = vec![0f32; width * height];
        for i in 0..data.len() {
            data[i] = (i % 255) as f32;
        }

        let shape = cv_core::TensorShape::new(1, height, width);
        let tensor_cpu = cv_core::CpuTensor::from_vec(data, shape).unwrap();

        // GPU execution
        use crate::tensor_ext::{TensorToCpu, TensorToGpu};
        let tensor_gpu = tensor_cpu.to_gpu_ctx(gpu).unwrap();
        let blurred_gpu = gpu.gaussian_blur(&tensor_gpu, 1.5, 7).unwrap();
        let res_gpu = blurred_gpu.to_cpu_ctx(gpu).unwrap();

        // CPU execution
        let res_cpu = cpu.gaussian_blur(&tensor_cpu, 1.5, 7).unwrap();

        // Check equality
        let slice_gpu = res_gpu.as_slice().unwrap();
        let slice_cpu = res_cpu.as_slice().unwrap();

        let mut diff_count = 0;
        for i in 0..slice_gpu.len() {
            if (slice_gpu[i] as i32 - slice_cpu[i] as i32).abs() > 1 {
                diff_count += 1;
            }
        }

        assert!(
            diff_count < (width * height) / 100,
            "Too many differences between GPU and CPU blur: {}",
            diff_count
        );
    }
}
