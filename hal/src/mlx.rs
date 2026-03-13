use crate::context::{
    BorderMode, ColorConversion, ComputeContext, Mog2Params, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, DeviceId, Error, Result};
use cv_core::{storage::Storage, Float, Tensor};

/// Experimental MLX Context for Apple Silicon
/// WARNING: Currently untested on actual hardware.
#[derive(Debug)]
pub struct MlxContext {
    pub device_id: DeviceId,
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
        _input: &Tensor<T, S>,
        _kernel: &Tensor<T, S>,
        _border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX convolve_2d not implemented".into(),
        ))
    }

    fn dispatch<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _name: &str,
        _buffers: &[&Tensor<u8, S>],
        _uniforms: &[u8],
        _workgroups: (u32, u32, u32),
    ) -> Result<()> {
        Err(Error::NotSupported("MLX dispatch not implemented".into()))
    }

    fn threshold<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _thresh: T,
        _max_value: T,
        _typ: ThresholdType,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX threshold not implemented".into()))
    }

    fn sobel<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _dx: i32,
        _dy: i32,
        _ksize: usize,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>)> {
        Err(Error::NotSupported("MLX sobel not implemented".into()))
    }

    fn canny<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _low_threshold: T,
        _high_threshold: T,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX canny not implemented".into()))
    }

    fn hough_lines<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _rho: T,
        _theta: T,
        _threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>> {
        Err(Error::NotSupported(
            "MLX hough_lines not implemented".into(),
        ))
    }

    fn hough_circles<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _min_radius: T,
        _max_radius: T,
        _threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>> {
        Err(Error::NotSupported(
            "MLX hough_circles not implemented".into(),
        ))
    }

    fn match_template<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _image: &Tensor<T, S>,
        _template: &Tensor<T, S>,
        _method: TemplateMatchMethod,
    ) -> Result<Tensor<T, OS>> {
        Err(Error::NotSupported(
            "MLX match_template not implemented".into(),
        ))
    }

    fn detect_objects<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _threshold: T,
    ) -> Result<Vec<cv_core::Detection>> {
        Err(Error::NotSupported(
            "MLX detect_objects not implemented".into(),
        ))
    }

    fn stereo_match<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _left: &Tensor<T, S>,
        _right: &Tensor<T, S>,
        _params: &StereoMatchParams,
    ) -> Result<Tensor<T, OS>> {
        Err(Error::NotSupported(
            "MLX stereo_match not implemented".into(),
        ))
    }

    fn triangulate_points<
        T: Float + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _proj_left: &[[T; 4]; 3],
        _proj_right: &[[T; 4]; 3],
        _points_left: &Tensor<T, S>,
        _points_right: &Tensor<T, S>,
    ) -> Result<Tensor<T, OS>> {
        Err(Error::NotSupported(
            "MLX triangulate_points not implemented".into(),
        ))
    }

    fn find_chessboard_corners<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _image: &Tensor<T, S>,
        _pattern_size: (usize, usize),
    ) -> Result<Vec<cv_core::KeyPoint>> {
        Err(Error::NotSupported(
            "MLX find_chessboard_corners not implemented".into(),
        ))
    }

    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _typ: MorphologyType,
        _kernel: &Tensor<u8, S>,
        _iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX morphology not implemented".into()))
    }

    fn warp<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _matrix: &[[T; 3]; 3],
        _new_shape: (usize, usize),
        _typ: WarpType,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX warp not implemented".into()))
    }

    fn nms<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _threshold: T,
        _window_size: usize,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX nms not implemented".into()))
    }

    fn nms_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _iou_threshold: T,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported("MLX nms_boxes not implemented".into()))
    }

    fn nms_rotated_boxes<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _iou_threshold: T,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported(
            "MLX nms_rotated_boxes not implemented".into(),
        ))
    }

    fn nms_polygons<T: Float + 'static>(
        &self,
        _polygons: &[cv_core::Polygon],
        _scores: &[T],
        _iou_threshold: T,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported(
            "MLX nms_polygons not implemented".into(),
        ))
    }

    fn pyramid_down<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX pyramid_down not implemented".into(),
        ))
    }

    fn pointcloud_transform<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _points: &Tensor<T, S>,
        _transform: &[[T; 4]; 4],
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX pointcloud_transform not implemented".into(),
        ))
    }

    /// Compute point-cloud normals on Apple Silicon.
    fn pointcloud_normals<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<T, S>> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let src = points.storage.as_slice().ok_or_else(|| {
                Error::InvalidInput("Points not on CPU — transfer to CPU first".into())
            })?;
            let src_f32: &[f32] = bytemuck::cast_slice(src);
            let num_points = points.shape.height;
            let k = (k_neighbors as usize)
                .max(3)
                .min(num_points.saturating_sub(1));

            // Convert flat vec4 storage → nalgebra Vector3 slice.
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
            Err(Error::NotSupported(
                "MLX pointcloud_normals only supports f32".into(),
            ))
        }
    }

    fn tsdf_integrate<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _depth_image: &Tensor<T, S>,
        _camera_pose: &[[T; 4]; 4],
        _intrinsics: &[T; 4],
        _voxel_volume: &mut Tensor<T, S>,
        _voxel_size: T,
        _truncation: T,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "MLX tsdf_integrate not implemented".into(),
        ))
    }

    fn tsdf_raycast<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _tsdf_volume: &Tensor<T, S>,
        _camera_pose: &[[T; 4]; 4],
        _intrinsics: &[T; 4],
        _image_size: (u32, u32),
        _depth_range: (T, T),
        _voxel_size: T,
        _truncation: T,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX tsdf_raycast not implemented".into(),
        ))
    }

    fn tsdf_extract_mesh<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _tsdf_volume: &Tensor<T, S>,
        _voxel_size: T,
        _iso_level: T,
        _max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        Err(Error::NotSupported(
            "MLX tsdf_extract_mesh not implemented".into(),
        ))
    }

    fn optical_flow_lk<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _prev_pyramid: &[Tensor<T, S>],
        _next_pyramid: &[Tensor<T, S>],
        _points: &[[T; 2]],
        _window_size: usize,
        _max_iters: u32,
    ) -> Result<Vec<[T; 2]>> {
        Err(Error::NotSupported(
            "MLX optical_flow_lk not implemented".into(),
        ))
    }

    fn cvt_color<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _code: ColorConversion,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX cvt_color not implemented".into()))
    }

    fn resize<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _new_shape: (usize, usize),
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX resize not implemented".into()))
    }

    fn bilateral_filter<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _d: i32,
        _sigma_color: T,
        _sigma_space: T,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX bilateral_filter not implemented".into(),
        ))
    }

    fn fast_detect<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
        _threshold: T,
        _non_max_suppression: bool,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX fast_detect not implemented".into(),
        ))
    }

    fn gaussian_blur<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _sigma: T,
        _k_size: usize,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX gaussian_blur not implemented".into(),
        ))
    }

    fn subtract<
        T: Float + 'static + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _a: &Tensor<T, S>,
        _b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX subtract not implemented".into()))
    }

    fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _query: &Tensor<u8, S>,
        _train: &Tensor<u8, S>,
        _ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        Err(Error::NotSupported(
            "MLX match_descriptors not implemented".into(),
        ))
    }

    fn sift_extrema<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _dog_prev: &Tensor<T, S>,
        _dog_curr: &Tensor<T, S>,
        _dog_next: &Tensor<T, S>,
        _threshold: T,
        _edge_threshold: T,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        Err(Error::NotSupported(
            "MLX sift_extrema not implemented".into(),
        ))
    }

    fn compute_sift_descriptors<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _image: &Tensor<T, S>,
        _keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        Err(Error::NotSupported(
            "MLX compute_sift_descriptors not implemented".into(),
        ))
    }

    fn icp_correspondences<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _src: &Tensor<T, S>,
        _tgt: &Tensor<T, S>,
        _max_dist: T,
    ) -> Result<Vec<(usize, usize, T)>> {
        Err(Error::NotSupported(
            "MLX icp_correspondences not implemented".into(),
        ))
    }

    fn icp_accumulate<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _source: &Tensor<T, S>,
        _target: &Tensor<T, S>,
        _target_normals: &Tensor<T, S>,
        _correspondences: &[(u32, u32)],
        _transform: &nalgebra::Matrix4<T>,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        Err(Error::NotSupported(
            "MLX icp_accumulate not implemented".into(),
        ))
    }

    fn dense_icp_step<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _source_depth: &Tensor<T, S>,
        _target_data: &Tensor<T, S>,
        _intrinsics: &[T; 4],
        _initial_guess: &nalgebra::Matrix4<T>,
        _max_dist: T,
        _max_angle: T,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        Err(Error::NotSupported(
            "MLX dense_icp_step not implemented".into(),
        ))
    }

    fn akaze_diffusion<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _input: &Tensor<T, S>,
        _k: T,
        _tau: T,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported(
            "MLX akaze_diffusion not implemented".into(),
        ))
    }

    fn akaze_derivatives<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)> {
        Err(Error::NotSupported(
            "MLX akaze_derivatives not implemented".into(),
        ))
    }

    fn akaze_contrast_k<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        _input: &Tensor<T, S>,
    ) -> Result<T> {
        Err(Error::NotSupported(
            "MLX akaze_contrast_k not implemented".into(),
        ))
    }

    fn spmv<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        _row_ptr: &[u32],
        _col_indices: &[u32],
        _values: &[T],
        _x: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        Err(Error::NotSupported("MLX spmv not implemented".into()))
    }

    fn mog2_update<T: Float + 'static, S1: Storage<T> + 'static, S2: Storage<u32> + 'static>(
        &self,
        _frame: &Tensor<T, S1>,
        _model: &mut Tensor<T, S1>,
        _mask: &mut Tensor<u32, S2>,
        _params: &Mog2Params<T>,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "MLX mog2_update not implemented".into(),
        ))
    }
}
