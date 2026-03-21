use crate::context::{
    BorderMode, ColorConversion, ComputeContext, Mog2Params, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::cpu::CpuBackend;
use crate::{BackendType, DeviceId, Result};
use cv_core::{storage::Storage, Float, Tensor};

/// Experimental MLX context for Apple Silicon.
///
/// Operations delegate to a CPU fallback backend when native MLX acceleration
/// is not available. The `pointcloud_normals` method retains its specialised
/// Metal/GPU-or-CPU path for f32 data.
///
/// **WARNING:** Currently untested on actual hardware.
#[derive(Debug)]
pub struct MlxContext {
    pub device_id: DeviceId,
    /// CPU backend used as a fallback for operations not yet accelerated on MLX.
    cpu_fallback: CpuBackend,
}

impl MlxContext {
    pub fn new() -> Option<Self> {
        #[cfg(feature = "mlx")]
        {
            // Placeholder for actual mlx-rs initialization.
            // Use DeviceId(2) to avoid colliding with CpuBackend (DeviceId 0) and
            // GpuContext (DeviceId 0) in the device registry HashMap.
            Some(Self {
                device_id: DeviceId(2),
                cpu_fallback: CpuBackend::new()?,
            })
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = DeviceId(2);
            None
        }
    }
}

impl ComputeContext for MlxContext {
    fn backend_type(&self) -> BackendType {
        BackendType::Mlx
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    fn last_submission_index(&self) -> crate::SubmissionIndex {
        crate::SubmissionIndex(0)
    }

    fn convolve_2d<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        kernel: &Tensor<T, S>,
        border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.convolve_2d(input, kernel, border_mode)
    }

    fn dispatch<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        name: &str,
        buffers: &[&Tensor<u8, S>],
        uniforms: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        self.cpu_fallback
            .dispatch(name, buffers, uniforms, workgroups)
    }

    fn threshold<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        thresh: T,
        max_value: T,
        typ: ThresholdType,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.threshold(input, thresh, max_value, typ)
    }

    fn sobel<
        T: Float + bytemuck::Pod + std::fmt::Debug + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>)> {
        self.cpu_fallback.sobel(input, dx, dy, ksize)
    }

    fn canny<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        low_threshold: T,
        high_threshold: T,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback
            .canny(input, low_threshold, high_threshold)
    }

    fn hough_lines<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        rho: T,
        theta: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>> {
        self.cpu_fallback.hough_lines(input, rho, theta, threshold)
    }

    fn hough_circles<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        min_radius: T,
        max_radius: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>> {
        self.cpu_fallback
            .hough_circles(input, min_radius, max_radius, threshold)
    }

    fn match_template<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        template: &Tensor<T, S>,
        method: TemplateMatchMethod,
    ) -> Result<Tensor<T, OS>> {
        self.cpu_fallback.match_template(image, template, method)
    }

    fn detect_objects<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
    ) -> Result<Vec<cv_core::Detection>> {
        self.cpu_fallback.detect_objects(input, threshold)
    }

    fn stereo_match<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        left: &Tensor<T, S>,
        right: &Tensor<T, S>,
        params: &StereoMatchParams,
    ) -> Result<Tensor<T, OS>> {
        self.cpu_fallback.stereo_match(left, right, params)
    }

    fn triangulate_points<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        proj_left: &[[T; 4]; 3],
        proj_right: &[[T; 4]; 3],
        points_left: &Tensor<T, S>,
        points_right: &Tensor<T, S>,
    ) -> Result<Tensor<T, OS>> {
        self.cpu_fallback
            .triangulate_points(proj_left, proj_right, points_left, points_right)
    }

    fn find_chessboard_corners<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        pattern_size: (usize, usize),
    ) -> Result<Vec<cv_core::KeyPoint>> {
        self.cpu_fallback
            .find_chessboard_corners(image, pattern_size)
    }

    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        self.cpu_fallback.morphology(input, typ, kernel, iterations)
    }

    fn warp<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        matrix: &[[T; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.warp(input, matrix, new_shape, typ)
    }

    fn nms<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        window_size: usize,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.nms(input, threshold, window_size)
    }

    fn nms_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        self.cpu_fallback.nms_boxes(input, iou_threshold)
    }

    fn nms_rotated_boxes<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        self.cpu_fallback.nms_rotated_boxes(input, iou_threshold)
    }

    fn nms_polygons<T: Float + 'static>(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[T],
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        self.cpu_fallback
            .nms_polygons(polygons, scores, iou_threshold)
    }

    fn pyramid_down<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.pyramid_down(input)
    }

    fn pointcloud_transform<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        transform: &[[T; 4]; 4],
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.pointcloud_transform(points, transform)
    }

    /// Compute point-cloud normals on Apple Silicon.
    ///
    /// Retains the specialised Metal/GPU-or-CPU path for f32; delegates to the
    /// CPU fallback for other float types.
    fn pointcloud_normals<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<T, S>> {
        use crate::Error;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let src = points.storage.as_slice().ok_or_else(|| {
                Error::InvalidInput("Points not on CPU -- transfer to CPU first".into())
            })?;
            let src_f32: &[f32] = bytemuck::cast_slice(src);
            let num_points = points.shape.height;
            let k = (k_neighbors as usize)
                .max(3)
                .min(num_points.saturating_sub(1));

            // Convert flat vec4 storage -> nalgebra Vector3 slice.
            let vecs: Vec<nalgebra::Vector3<f32>> = src_f32
                .chunks(4)
                .take(num_points)
                .map(|c| nalgebra::Vector3::new(c[0], c[1], c[2]))
                .collect();

            // Try Metal (via wgpu GpuContext); fall back to fast CPU path.
            let normals =
                crate::gpu_kernels::pointcloud::compute_normals_morton_gpu_or_cpu(&vecs, k as u32);

            // Write normals back to output storage (same type S).
            let mut out = S::new(num_points * 4, T::ZERO).map_err(Error::MemoryError)?;
            if let Some(dst) = out.as_mut_slice() {
                let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                for (i, n) in normals.iter().enumerate() {
                    dst_f32[i * 4] = n.x;
                    dst_f32[i * 4 + 1] = n.y;
                    dst_f32[i * 4 + 2] = n.z;
                    dst_f32[i * 4 + 3] = 0.0;
                }
            }
            Ok(Tensor {
                storage: out,
                shape: points.shape,
                dtype: points.dtype,
                _phantom: std::marker::PhantomData,
            })
        } else {
            // Fall back to CPU for non-f32 types.
            self.cpu_fallback.pointcloud_normals(points, k_neighbors)
        }
    }

    fn tsdf_integrate<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        depth_image: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        voxel_volume: &mut Tensor<T, S>,
        voxel_size: T,
        truncation: T,
    ) -> Result<()> {
        self.cpu_fallback.tsdf_integrate(
            depth_image,
            camera_pose,
            intrinsics,
            voxel_volume,
            voxel_size,
            truncation,
        )
    }

    fn tsdf_raycast<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        tsdf_volume: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        image_size: (u32, u32),
        depth_range: (T, T),
        voxel_size: T,
        truncation: T,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.tsdf_raycast(
            tsdf_volume,
            camera_pose,
            intrinsics,
            image_size,
            depth_range,
            voxel_size,
            truncation,
        )
    }

    fn tsdf_extract_mesh<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        tsdf_volume: &Tensor<T, S>,
        voxel_size: T,
        iso_level: T,
        max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        self.cpu_fallback
            .tsdf_extract_mesh(tsdf_volume, voxel_size, iso_level, max_triangles)
    }

    fn optical_flow_lk<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        prev_pyramid: &[Tensor<T, S>],
        next_pyramid: &[Tensor<T, S>],
        points: &[[T; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> Result<Vec<[T; 2]>> {
        self.cpu_fallback.optical_flow_lk(
            prev_pyramid,
            next_pyramid,
            points,
            window_size,
            max_iters,
        )
    }

    fn cvt_color<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        code: ColorConversion,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.cvt_color(input, code)
    }

    fn resize<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.resize(input, new_shape)
    }

    fn bilateral_filter<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        d: i32,
        sigma_color: T,
        sigma_space: T,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback
            .bilateral_filter(input, d, sigma_color, sigma_space)
    }

    fn fast_detect<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        non_max_suppression: bool,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback
            .fast_detect(input, threshold, non_max_suppression)
    }

    fn gaussian_blur<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        sigma: T,
        k_size: usize,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.gaussian_blur(input, sigma, k_size)
    }

    fn subtract<
        T: Float + 'static + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.subtract(a, b)
    }

    fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        self.cpu_fallback
            .match_descriptors(query, train, ratio_threshold)
    }

    fn sift_extrema<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        dog_prev: &Tensor<T, S>,
        dog_curr: &Tensor<T, S>,
        dog_next: &Tensor<T, S>,
        threshold: T,
        edge_threshold: T,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        self.cpu_fallback
            .sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold)
    }

    fn compute_sift_descriptors<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        self.cpu_fallback.compute_sift_descriptors(image, keypoints)
    }

    fn icp_correspondences<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        src: &Tensor<T, S>,
        tgt: &Tensor<T, S>,
        max_dist: T,
    ) -> Result<Vec<(usize, usize, T)>> {
        self.cpu_fallback.icp_correspondences(src, tgt, max_dist)
    }

    fn icp_accumulate<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source: &Tensor<T, S>,
        target: &Tensor<T, S>,
        target_normals: &Tensor<T, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<T>,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        self.cpu_fallback
            .icp_accumulate(source, target, target_normals, correspondences, transform)
    }

    fn dense_icp_step<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source_depth: &Tensor<T, S>,
        target_data: &Tensor<T, S>,
        intrinsics: &[T; 4],
        initial_guess: &nalgebra::Matrix4<T>,
        max_dist: T,
        max_angle: T,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        self.cpu_fallback.dense_icp_step(
            source_depth,
            target_data,
            intrinsics,
            initial_guess,
            max_dist,
            max_angle,
        )
    }

    fn akaze_diffusion<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        k: T,
        tau: T,
        diffusivity: crate::context::DiffusivityType,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback
            .akaze_diffusion(input, k, tau, diffusivity)
    }

    fn akaze_derivatives<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)> {
        self.cpu_fallback.akaze_derivatives(input)
    }

    fn akaze_contrast_k<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<T> {
        self.cpu_fallback.akaze_contrast_k(input)
    }

    fn spmv<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[T],
        x: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        self.cpu_fallback.spmv(row_ptr, col_indices, values, x)
    }

    fn mog2_update<
        T: Float + bytemuck::Pod + 'static,
        S1: Storage<T> + 'static,
        S2: Storage<u32> + 'static,
    >(
        &self,
        frame: &Tensor<T, S1>,
        model: &mut Tensor<T, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &Mog2Params<T>,
    ) -> Result<()> {
        self.cpu_fallback.mog2_update(frame, model, mask, params)
    }

    fn remap<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
        S2: Storage<f32> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _map_x: &Tensor<f32, S2>,
        _map_y: &Tensor<f32, S2>,
        _interpolation: crate::context::Interpolation,
        _border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        Err(crate::Error::NotSupported(
            "remap not yet implemented on MLX backend".into(),
        ))
    }

    fn undistort<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _intrinsics: &cv_core::CameraIntrinsics,
        _distortion: &cv_core::Distortion,
        _rectification: &nalgebra::Matrix3<f64>,
        _new_intrinsics: &cv_core::CameraIntrinsics,
        _interpolation: crate::context::Interpolation,
        _border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        Err(crate::Error::NotSupported(
            "undistort not yet implemented on MLX backend".into(),
        ))
    }
}
