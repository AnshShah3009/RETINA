use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, ThresholdType, WarpType,
};
use crate::cpu::CpuBackend;
use crate::gpu::GpuContext;
use crate::mlx::MlxContext;
use crate::Result;
use cv_core::{storage::Storage, Tensor};
use std::sync::OnceLock;

static CPU_CONTEXT: OnceLock<CpuBackend> = OnceLock::new();
static MLX_CONTEXT: OnceLock<MlxContext> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
pub enum ComputeDevice<'a> {
    Cpu(&'a CpuBackend),
    Gpu(&'a GpuContext),
    Mlx(&'a MlxContext),
}

impl<'a> ComputeDevice<'a> {
    pub fn backend_type(&self) -> crate::BackendType {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.backend_type(),
            ComputeDevice::Gpu(gpu) => gpu.backend_type(),
            ComputeDevice::Mlx(mlx) => mlx.backend_type(),
        }
    }

    pub fn device_id(&self) -> crate::DeviceId {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.device_id(),
            ComputeDevice::Gpu(gpu) => gpu.device_id(),
            ComputeDevice::Mlx(mlx) => mlx.device_id(),
        }
    }

    pub fn convolve_2d<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Gpu(gpu) => gpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Mlx(mlx) => mlx.convolve_2d(input, kernel, border_mode),
        }
    }

    pub fn threshold<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.threshold(input, thresh, max_value, typ),
            ComputeDevice::Gpu(gpu) => gpu.threshold(input, thresh, max_value, typ),
            ComputeDevice::Mlx(mlx) => mlx.threshold(input, thresh, max_value, typ),
        }
    }

    pub fn sobel<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.sobel(input, dx, dy, ksize),
            ComputeDevice::Gpu(gpu) => gpu.sobel(input, dx, dy, ksize),
            ComputeDevice::Mlx(mlx) => mlx.sobel(input, dx, dy, ksize),
        }
    }

    pub fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.morphology(input, typ, kernel, iterations),
            ComputeDevice::Gpu(gpu) => gpu.morphology(input, typ, kernel, iterations),
            ComputeDevice::Mlx(mlx) => mlx.morphology(input, typ, kernel, iterations),
        }
    }

    pub fn warp<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.warp(input, matrix, new_shape, typ),
            ComputeDevice::Gpu(gpu) => gpu.warp(input, matrix, new_shape, typ),
            ComputeDevice::Mlx(mlx) => mlx.warp(input, matrix, new_shape, typ),
        }
    }

    pub fn nms<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms(input, threshold, window_size),
            ComputeDevice::Gpu(gpu) => gpu.nms(input, threshold, window_size),
            ComputeDevice::Mlx(mlx) => mlx.nms(input, threshold, window_size),
        }
    }

    pub fn nms_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_boxes(input, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_boxes(input, iou_threshold),
            ComputeDevice::Mlx(mlx) => mlx.nms_boxes(input, iou_threshold),
        }
    }

    pub fn nms_rotated_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_rotated_boxes(input, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_rotated_boxes(input, iou_threshold),
            ComputeDevice::Mlx(mlx) => mlx.nms_rotated_boxes(input, iou_threshold),
        }
    }

    pub fn nms_polygons(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[f32],
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_polygons(polygons, scores, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_polygons(polygons, scores, iou_threshold),
            ComputeDevice::Mlx(mlx) => mlx.nms_polygons(polygons, scores, iou_threshold),
        }
    }

    pub fn pointcloud_transform<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.pointcloud_transform(points, transform),
            ComputeDevice::Gpu(gpu) => gpu.pointcloud_transform(points, transform),
            ComputeDevice::Mlx(mlx) => mlx.pointcloud_transform(points, transform),
        }
    }

    pub fn pointcloud_normals<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.pointcloud_normals(points, k_neighbors),
            ComputeDevice::Gpu(gpu) => gpu.pointcloud_normals(points, k_neighbors),
            ComputeDevice::Mlx(mlx) => mlx.pointcloud_normals(points, k_neighbors),
        }
    }

    pub fn tsdf_integrate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        depth_image: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        voxel_volume: &mut Tensor<f32, S>,
        voxel_size: f32,
        truncation: f32,
    ) -> Result<()> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.tsdf_integrate(
                depth_image,
                camera_pose,
                intrinsics,
                voxel_volume,
                voxel_size,
                truncation,
            ),
            ComputeDevice::Gpu(gpu) => gpu.tsdf_integrate(
                depth_image,
                camera_pose,
                intrinsics,
                voxel_volume,
                voxel_size,
                truncation,
            ),
            ComputeDevice::Mlx(mlx) => mlx.tsdf_integrate(
                depth_image,
                camera_pose,
                intrinsics,
                voxel_volume,
                voxel_size,
                truncation,
            ),
        }
    }

    pub fn tsdf_raycast<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        tsdf_volume: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        image_size: (u32, u32),
        depth_range: (f32, f32),
        voxel_size: f32,
        truncation: f32,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.tsdf_raycast(
                tsdf_volume,
                camera_pose,
                intrinsics,
                image_size,
                depth_range,
                voxel_size,
                truncation,
            ),
            ComputeDevice::Gpu(gpu) => gpu.tsdf_raycast(
                tsdf_volume,
                camera_pose,
                intrinsics,
                image_size,
                depth_range,
                voxel_size,
                truncation,
            ),
            ComputeDevice::Mlx(mlx) => mlx.tsdf_raycast(
                tsdf_volume,
                camera_pose,
                intrinsics,
                image_size,
                depth_range,
                voxel_size,
                truncation,
            ),
        }
    }

    pub fn tsdf_extract_mesh<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        tsdf_volume: &Tensor<f32, S>,
        voxel_size: f32,
        iso_level: f32,
        max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        match self {
            ComputeDevice::Cpu(cpu) => {
                cpu.tsdf_extract_mesh(tsdf_volume, voxel_size, iso_level, max_triangles)
            }
            ComputeDevice::Gpu(gpu) => {
                gpu.tsdf_extract_mesh(tsdf_volume, voxel_size, iso_level, max_triangles)
            }
            ComputeDevice::Mlx(mlx) => {
                mlx.tsdf_extract_mesh(tsdf_volume, voxel_size, iso_level, max_triangles)
            }
        }
    }

    pub fn optical_flow_lk<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        prev_pyramid: &[Tensor<f32, S>],
        next_pyramid: &[Tensor<f32, S>],
        points: &[[f32; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> Result<Vec<[f32; 2]>> {
        match self {
            ComputeDevice::Cpu(cpu) => {
                cpu.optical_flow_lk(prev_pyramid, next_pyramid, points, window_size, max_iters)
            }
            ComputeDevice::Gpu(gpu) => {
                gpu.optical_flow_lk(prev_pyramid, next_pyramid, points, window_size, max_iters)
            }
            ComputeDevice::Mlx(mlx) => {
                mlx.optical_flow_lk(prev_pyramid, next_pyramid, points, window_size, max_iters)
            }
        }
    }

    pub fn cvt_color<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.cvt_color(input, code),
            ComputeDevice::Gpu(gpu) => gpu.cvt_color(input, code),
            ComputeDevice::Mlx(mlx) => mlx.cvt_color(input, code),
        }
    }

    pub fn resize<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.resize(input, new_shape),
            ComputeDevice::Gpu(gpu) => gpu.resize(input, new_shape),
            ComputeDevice::Mlx(mlx) => mlx.resize(input, new_shape),
        }
    }

    pub fn bilateral_filter<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.bilateral_filter(input, d, sigma_color, sigma_space),
            ComputeDevice::Gpu(gpu) => gpu.bilateral_filter(input, d, sigma_color, sigma_space),
            ComputeDevice::Mlx(mlx) => mlx.bilateral_filter(input, d, sigma_color, sigma_space),
        }
    }

    pub fn fast_detect<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.fast_detect(input, threshold, non_max_suppression),
            ComputeDevice::Gpu(gpu) => gpu.fast_detect(input, threshold, non_max_suppression),
            ComputeDevice::Mlx(mlx) => mlx.fast_detect(input, threshold, non_max_suppression),
        }
    }

    pub fn gaussian_blur<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        sigma: f32,
        k_size: usize,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.gaussian_blur(input, sigma, k_size),
            ComputeDevice::Gpu(gpu) => gpu.gaussian_blur(input, sigma, k_size),
            ComputeDevice::Mlx(mlx) => mlx.gaussian_blur(input, sigma, k_size),
        }
    }

    pub fn subtract<
        T: Clone + Copy + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.subtract(a, b),
            ComputeDevice::Gpu(gpu) => gpu.subtract(a, b),
            ComputeDevice::Mlx(mlx) => mlx.subtract(a, b),
        }
    }

    pub fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.match_descriptors(query, train, ratio_threshold),
            ComputeDevice::Gpu(gpu) => gpu.match_descriptors(query, train, ratio_threshold),
            ComputeDevice::Mlx(mlx) => mlx.match_descriptors(query, train, ratio_threshold),
        }
    }

    pub fn sift_extrema<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        dog_prev: &Tensor<f32, S>,
        dog_curr: &Tensor<f32, S>,
        dog_next: &Tensor<f32, S>,
        threshold: f32,
        edge_threshold: f32,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        match self {
            ComputeDevice::Cpu(cpu) => {
                cpu.sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold)
            }
            ComputeDevice::Gpu(gpu) => {
                gpu.sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold)
            }
            ComputeDevice::Mlx(mlx) => {
                mlx.sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold)
            }
        }
    }

    pub fn compute_sift_descriptors<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        image: &Tensor<f32, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.compute_sift_descriptors(image, keypoints),
            ComputeDevice::Gpu(gpu) => gpu.compute_sift_descriptors(image, keypoints),
            ComputeDevice::Mlx(mlx) => mlx.compute_sift_descriptors(image, keypoints),
        }
    }

    pub fn icp_correspondences<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        src: &Tensor<f32, S>,
        tgt: &Tensor<f32, S>,
        max_dist: f32,
    ) -> Result<Vec<(usize, usize, f32)>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.icp_correspondences(src, tgt, max_dist),
            ComputeDevice::Gpu(gpu) => gpu.icp_correspondences(src, tgt, max_dist),
            ComputeDevice::Mlx(mlx) => mlx.icp_correspondences(src, tgt, max_dist),
        }
    }

    pub fn icp_accumulate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        source: &Tensor<f32, S>,
        target: &Tensor<f32, S>,
        target_normals: &Tensor<f32, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<f32>,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        match self {
            ComputeDevice::Cpu(cpu) => {
                cpu.icp_accumulate(source, target, target_normals, correspondences, transform)
            }
            ComputeDevice::Gpu(gpu) => {
                gpu.icp_accumulate(source, target, target_normals, correspondences, transform)
            }
            ComputeDevice::Mlx(mlx) => {
                mlx.icp_accumulate(source, target, target_normals, correspondences, transform)
            }
        }
    }

    pub fn dense_icp_step<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        source_depth: &Tensor<f32, S>,
        target_data: &Tensor<f32, S>,
        intrinsics: &[f32; 4],
        initial_guess: &nalgebra::Matrix4<f32>,
        max_dist: f32,
        max_angle: f32,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.dense_icp_step(
                source_depth,
                target_data,
                intrinsics,
                initial_guess,
                max_dist,
                max_angle,
            ),
            ComputeDevice::Gpu(gpu) => gpu.dense_icp_step(
                source_depth,
                target_data,
                intrinsics,
                initial_guess,
                max_dist,
                max_angle,
            ),
            ComputeDevice::Mlx(mlx) => mlx.dense_icp_step(
                source_depth,
                target_data,
                intrinsics,
                initial_guess,
                max_dist,
                max_angle,
            ),
        }
    }

    pub fn akaze_diffusion<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        k: f32,
        tau: f32,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_diffusion(input, k, tau),
            ComputeDevice::Gpu(gpu) => gpu.akaze_diffusion(input, k, tau),
            ComputeDevice::Mlx(mlx) => mlx.akaze_diffusion(input, k, tau),
        }
    }

    pub fn akaze_derivatives<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_derivatives(input),
            ComputeDevice::Gpu(gpu) => gpu.akaze_derivatives(input),
            ComputeDevice::Mlx(mlx) => mlx.akaze_derivatives(input),
        }
    }

    pub fn akaze_contrast_k<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<f32> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_contrast_k(input),
            ComputeDevice::Gpu(gpu) => gpu.akaze_contrast_k(input),
            ComputeDevice::Mlx(mlx) => mlx.akaze_contrast_k(input),
        }
    }

    pub fn spmv<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[f32],
        x: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.spmv(row_ptr, col_indices, values, x),
            ComputeDevice::Gpu(gpu) => gpu.spmv(row_ptr, col_indices, values, x),
            ComputeDevice::Mlx(mlx) => mlx.spmv(row_ptr, col_indices, values, x),
        }
    }

    pub fn mog2_update<S1: Storage<f32> + 'static, S2: Storage<u32> + 'static>(
        &self,
        frame: &Tensor<f32, S1>,
        model: &mut Tensor<f32, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params,
    ) -> Result<()> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.mog2_update(frame, model, mask, params),
            ComputeDevice::Gpu(gpu) => gpu.mog2_update(frame, model, mask, params),
            ComputeDevice::Mlx(mlx) => mlx.mog2_update(frame, model, mask, params),
        }
    }

    /// Get a pooled GPU buffer if this is a GPU device.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        match self {
            ComputeDevice::Gpu(gpu) => Ok(gpu.get_buffer(size, usage)),
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => Err(crate::Error::NotSupported(
                "GPU buffer pooling not available".into(),
            )),
        }
    }

    /// Return a buffer to the pool if this is a GPU device.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) -> Result<()> {
        match self {
            ComputeDevice::Gpu(gpu) => {
                gpu.return_buffer(buffer, usage);
                Ok(())
            }
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => Err(crate::Error::NotSupported(
                "GPU buffer pooling not available".into(),
            )),
        }
    }
}

/// Get a compute device by its ID.
pub fn get_device_by_id(id: crate::DeviceId) -> Result<ComputeDevice<'static>> {
    // Check CPU
    if let Some(cpu) = CPU_CONTEXT.get() {
        if cpu.device_id() == id {
            return Ok(ComputeDevice::Cpu(cpu));
        }
    } else {
        // Try to init if it's the only one
        if let Some(cpu_backend) = CpuBackend::new() {
            let cpu = CPU_CONTEXT.get_or_init(move || cpu_backend);
            if cpu.device_id() == id {
                return Ok(ComputeDevice::Cpu(cpu));
            }
        }
    }

    // Check GPU
    if let Ok(gpu) = GpuContext::global() {
        if gpu.device_id() == id {
            return Ok(ComputeDevice::Gpu(gpu));
        }
    }

    // Check MLX
    if let Some(mlx) = MLX_CONTEXT.get() {
        if mlx.device_id() == id {
            return Ok(ComputeDevice::Mlx(mlx));
        }
    }

    Err(crate::Error::DeviceError(format!(
        "Device {:?} not found in global contexts",
        id
    )))
}

/// Get the best available compute device.
pub fn get_device() -> Result<ComputeDevice<'static>> {
    if let Some(mlx) = MlxContext::new() {
        return Ok(ComputeDevice::Mlx(MLX_CONTEXT.get_or_init(|| mlx)));
    }

    match GpuContext::global() {
        Ok(gpu) => Ok(ComputeDevice::Gpu(gpu)),
        Err(_) => {
            let cpu_backend = CpuBackend::new()
                .ok_or_else(|| crate::Error::InitError("CPU backend unavailable".into()))?;
            let cpu = CPU_CONTEXT.get_or_init(|| cpu_backend);
            Ok(ComputeDevice::Cpu(cpu))
        }
    }
}
