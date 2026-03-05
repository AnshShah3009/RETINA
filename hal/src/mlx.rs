use crate::context::{
    BorderMode, ColorConversion, ComputeContext, Mog2Params, MorphologyType, StereoMatchParams,
    TemplateMatchMethod, ThresholdType, WarpType,
};
use crate::{BackendType, DeviceId, Error, Result};
use cv_core::{storage::Storage, Tensor};

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
            // Placeholder for actual mlx-rs initialization
            Some(Self {
                device_id: DeviceId(0),
            })
        }
        #[cfg(not(feature = "mlx"))]
        {
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

    fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _kernel: &Tensor<f32, S>,
        _border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
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

    fn threshold<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _thresh: u8,
        _max_value: u8,
        _typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX threshold not implemented".into()))
    }

    fn sobel<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _dx: i32,
        _dy: i32,
        _ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        Err(Error::NotSupported("MLX sobel not implemented".into()))
    }

    fn canny<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _low_threshold: f32,
        _high_threshold: f32,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX canny not implemented".into()))
    }

    fn hough_lines<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _rho: f32,
        _theta: f32,
        _threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>> {
        Err(Error::NotSupported(
            "MLX hough_lines not implemented".into(),
        ))
    }

    fn hough_circles<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _min_radius: f32,
        _max_radius: f32,
        _threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>> {
        Err(Error::NotSupported(
            "MLX hough_circles not implemented".into(),
        ))
    }

    fn match_template<S: Storage<u8> + 'static, OS: Storage<f32> + 'static>(
        &self,
        _image: &Tensor<u8, S>,
        _template: &Tensor<u8, S>,
        _method: TemplateMatchMethod,
    ) -> Result<Tensor<f32, OS>> {
        Err(Error::NotSupported(
            "MLX match_template not implemented".into(),
        ))
    }

    fn detect_objects<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _threshold: f32,
    ) -> Result<Vec<cv_core::Detection>> {
        Err(Error::NotSupported(
            "MLX detect_objects not implemented".into(),
        ))
    }

    fn stereo_match<S: Storage<u8> + 'static, OS: Storage<f32> + 'static>(
        &self,
        _left: &Tensor<u8, S>,
        _right: &Tensor<u8, S>,
        _params: &StereoMatchParams,
    ) -> Result<Tensor<f32, OS>> {
        Err(Error::NotSupported(
            "MLX stereo_match not implemented".into(),
        ))
    }

    fn triangulate_points<S: Storage<f32> + 'static, OS: Storage<f32> + 'static>(
        &self,
        _proj_left: &[[f32; 4]; 3],
        _proj_right: &[[f32; 4]; 3],
        _points_left: &Tensor<f32, S>,
        _points_right: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, OS>> {
        Err(Error::NotSupported(
            "MLX triangulate_points not implemented".into(),
        ))
    }

    fn find_chessboard_corners<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _image: &Tensor<u8, S>,
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

    fn warp<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _matrix: &[[f32; 3]; 3],
        _new_shape: (usize, usize),
        _typ: WarpType,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX warp not implemented".into()))
    }

    fn nms<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _threshold: f32,
        _window_size: usize,
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported("MLX nms not implemented".into()))
    }

    fn nms_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported("MLX nms_boxes not implemented".into()))
    }

    fn nms_rotated_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported(
            "MLX nms_rotated_boxes not implemented".into(),
        ))
    }

    fn nms_polygons(
        &self,
        _polygons: &[cv_core::Polygon],
        _scores: &[f32],
        _iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        Err(Error::NotSupported(
            "MLX nms_polygons not implemented".into(),
        ))
    }

    fn pointcloud_transform<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _points: &Tensor<f32, S>,
        _transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported(
            "MLX pointcloud_transform not implemented".into(),
        ))
    }

    fn pointcloud_normals<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _points: &Tensor<f32, S>,
        _k_neighbors: u32,
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported(
            "MLX pointcloud_normals not implemented".into(),
        ))
    }

    fn tsdf_integrate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _depth_image: &Tensor<f32, S>,
        _camera_pose: &[[f32; 4]; 4],
        _intrinsics: &[f32; 4],
        _voxel_volume: &mut Tensor<f32, S>,
        _voxel_size: f32,
        _truncation: f32,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "MLX tsdf_integrate not implemented".into(),
        ))
    }

    fn tsdf_raycast<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _tsdf_volume: &Tensor<f32, S>,
        _camera_pose: &[[f32; 4]; 4],
        _intrinsics: &[f32; 4],
        _image_size: (u32, u32),
        _depth_range: (f32, f32),
        _voxel_size: f32,
        _truncation: f32,
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported(
            "MLX tsdf_raycast not implemented".into(),
        ))
    }

    fn tsdf_extract_mesh<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _tsdf_volume: &Tensor<f32, S>,
        _voxel_size: f32,
        _iso_level: f32,
        _max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        Err(Error::NotSupported(
            "MLX tsdf_extract_mesh not implemented".into(),
        ))
    }

    fn optical_flow_lk<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _prev_pyramid: &[Tensor<f32, S>],
        _next_pyramid: &[Tensor<f32, S>],
        _points: &[[f32; 2]],
        _window_size: usize,
        _max_iters: u32,
    ) -> Result<Vec<[f32; 2]>> {
        Err(Error::NotSupported(
            "MLX optical_flow_lk not implemented".into(),
        ))
    }

    fn cvt_color<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _code: ColorConversion,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX cvt_color not implemented".into()))
    }

    fn resize<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported("MLX resize not implemented".into()))
    }

    fn bilateral_filter<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _d: i32,
        _sigma_color: f32,
        _sigma_space: f32,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported(
            "MLX bilateral_filter not implemented".into(),
        ))
    }

    fn fast_detect<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _threshold: u8,
        _non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported(
            "MLX fast_detect not implemented".into(),
        ))
    }

    fn gaussian_blur<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        _input: &Tensor<u8, S>,
        _sigma: f32,
        _k_size: usize,
    ) -> Result<Tensor<u8, S>> {
        Err(Error::NotSupported(
            "MLX gaussian_blur not implemented".into(),
        ))
    }

    fn subtract<
        T: Clone + Copy + bytemuck::Pod + std::fmt::Debug,
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

    fn sift_extrema<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _dog_prev: &Tensor<f32, S>,
        _dog_curr: &Tensor<f32, S>,
        _dog_next: &Tensor<f32, S>,
        _threshold: f32,
        _edge_threshold: f32,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        Err(Error::NotSupported(
            "MLX sift_extrema not implemented".into(),
        ))
    }

    fn compute_sift_descriptors<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _image: &Tensor<f32, S>,
        _keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        Err(Error::NotSupported(
            "MLX compute_sift_descriptors not implemented".into(),
        ))
    }

    fn icp_correspondences<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _src: &Tensor<f32, S>,
        _tgt: &Tensor<f32, S>,
        _max_dist: f32,
    ) -> Result<Vec<(usize, usize, f32)>> {
        Err(Error::NotSupported(
            "MLX icp_correspondences not implemented".into(),
        ))
    }

    fn icp_accumulate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _source: &Tensor<f32, S>,
        _target: &Tensor<f32, S>,
        _target_normals: &Tensor<f32, S>,
        _correspondences: &[(u32, u32)],
        _transform: &nalgebra::Matrix4<f32>,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        Err(Error::NotSupported(
            "MLX icp_accumulate not implemented".into(),
        ))
    }

    fn dense_icp_step<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _source_depth: &Tensor<f32, S>,
        _target_data: &Tensor<f32, S>,
        _intrinsics: &[f32; 4],
        _initial_guess: &nalgebra::Matrix4<f32>,
        _max_dist: f32,
        _max_angle: f32,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        Err(Error::NotSupported(
            "MLX dense_icp_step not implemented".into(),
        ))
    }

    fn akaze_diffusion<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _k: f32,
        _tau: f32,
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported(
            "MLX akaze_diffusion not implemented".into(),
        ))
    }

    fn akaze_derivatives<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
    ) -> Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)> {
        Err(Error::NotSupported(
            "MLX akaze_derivatives not implemented".into(),
        ))
    }

    fn akaze_contrast_k<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
    ) -> Result<f32> {
        Err(Error::NotSupported(
            "MLX akaze_contrast_k not implemented".into(),
        ))
    }

    fn spmv<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        _row_ptr: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _x: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported("MLX spmv not implemented".into()))
    }

    fn mog2_update<S1: Storage<f32> + 'static, S2: Storage<u32> + 'static>(
        &self,
        _frame: &Tensor<f32, S1>,
        _model: &mut Tensor<f32, S1>,
        _mask: &mut Tensor<u32, S2>,
        _params: &Mog2Params,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "MLX mog2_update not implemented".into(),
        ))
    }
}
