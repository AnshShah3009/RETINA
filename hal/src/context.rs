use crate::{BackendType, DeviceId, Result};
use cv_core::{storage::Storage, Tensor};

/// A unified context for executing compute operations.
///
/// This trait abstracts over different compute backends (CPU, CUDA, Vulkan/WebGPU),
/// allowing high-level algorithms to be written in a backend-agnostic way.
pub trait ComputeContext: Send + Sync {
    /// Get the backend type of this context
    fn backend_type(&self) -> BackendType;

    /// Get the unique device ID
    fn device_id(&self) -> DeviceId;

    /// Wait for all pending operations to complete
    fn wait_idle(&self) -> Result<()>;

    /// Get the index of the last submitted GPU operation
    fn last_submission_index(&self) -> crate::SubmissionIndex;

    // --- Core Operations ---

    /// Execute a 2D convolution
    fn convolve_2d<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>>;

    /// Execute a generic compute shader/kernel
    ///
    /// * `name`: Name of the kernel (e.g., "gaussian_blur")
    /// * `buffers`: List of buffers (input/output)
    /// * `uniforms`: Uniform data (constants)
    /// * `workgroups`: (x, y, z) dispatch size
    fn dispatch<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        name: &str,
        buffers: &[&Tensor<u8, S>],
        uniforms: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Execute a threshold operation
    fn threshold<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>>;

    /// Execute a Sobel operator
    fn sobel<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)>;

    /// Execute Canny edge detection
    fn canny<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        low_threshold: f32,
        high_threshold: f32,
    ) -> Result<Tensor<u8, S>>;

    /// Execute Hough line transform
    fn hough_lines<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        rho: f32,
        theta: f32,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>>;

    /// Execute Hough circle transform
    fn hough_circles<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        min_radius: f32,
        max_radius: f32,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>>;

    /// Template matching
    fn match_template<
        S: Storage<u8> + 'static,
        OS: Storage<f32> + cv_core::StorageFactory<f32> + 'static,
    >(
        &self,
        image: &Tensor<u8, S>,
        template: &Tensor<u8, S>,
        method: TemplateMatchMethod,
    ) -> Result<Tensor<f32, OS>>;

    /// Execute a morphological operation
    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>>;

    /// Execute a warp operation (affine or perspective)
    fn warp<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<u8, S>>;

    /// Execute Non-Maximum Suppression
    fn nms<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> Result<Tensor<f32, S>>;

    /// Execute Bounding Box NMS
    /// input: (N, 5) tensor [x1, y1, x2, y2, score]
    fn nms_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Execute Rotated Bounding Box NMS
    /// input: (N, 6) tensor [cx, cy, w, h, angle, score]
    fn nms_rotated_boxes<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Execute Polygon NMS
    /// input: list of polygons, list of scores
    fn nms_polygons(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[f32],
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Transform a point cloud
    fn pointcloud_transform<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>>;

    /// Compute normals for a point cloud
    fn pointcloud_normals<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<f32, S>>;

    /// TSDF Volume Integration
    fn tsdf_integrate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        depth_image: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        voxel_volume: &mut Tensor<f32, S>,
        voxel_size: f32,
        truncation: f32,
    ) -> Result<()>;

    /// Execute object detection (e.g., via a loaded ONNX/TensorRT model)
    fn detect_objects<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: f32,
    ) -> Result<Vec<cv_core::Detection>>;

    fn tsdf_raycast<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        tsdf_volume: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        image_size: (u32, u32),
        depth_range: (f32, f32),
        voxel_size: f32,
        truncation: f32,
    ) -> Result<Tensor<f32, S>>;

    fn tsdf_extract_mesh<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        tsdf_volume: &Tensor<f32, S>,
        voxel_size: f32,
        iso_level: f32,
        max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>>;

    fn optical_flow_lk<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        prev_pyramid: &[Tensor<f32, S>],
        next_pyramid: &[Tensor<f32, S>],
        points: &[[f32; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> Result<Vec<[f32; 2]>>;

    /// Color space conversion
    fn cvt_color<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> Result<Tensor<u8, S>>;

    /// Resize an image
    fn resize<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>>;

    /// Bilateral Filter
    fn bilateral_filter<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> Result<Tensor<u8, S>>;

    /// FAST Keypoint Detection
    /// Returns a score map (1 channel, same size as input)
    fn fast_detect<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>>;

    /// Gaussian Blur
    fn gaussian_blur<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        sigma: f32,
        k_size: usize,
    ) -> Result<Tensor<u8, S>>;

    /// Elementwise Subtraction (A - B)
    /// Input: Signed output often needed for DoG, but we might use f32 or i16.
    /// For SIFT DoG, we usually use f32.
    fn subtract<
        T: Clone + Copy + bytemuck::Pod + std::fmt::Debug,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>>;

    /// Feature Matching
    fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches>;

    /// SIFT Local Extrema Detection
    /// Finds local maxima/minima in 3x3x3 scale-space neighborhood.
    /// Returns a U8 score map on CPU for refinement.
    fn sift_extrema<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        dog_prev: &Tensor<f32, S>,
        dog_curr: &Tensor<f32, S>,
        dog_next: &Tensor<f32, S>,
        threshold: f32,
        edge_threshold: f32,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>>;

    /// SIFT Descriptor Computation
    /// Returns a descriptor tensor (num_kps x 128)
    fn compute_sift_descriptors<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        image: &Tensor<f32, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors>;

    /// ICP Point Correspondence
    /// Finds nearest neighbors from src points to tgt points
    fn icp_correspondences<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        src: &Tensor<f32, S>,
        tgt: &Tensor<f32, S>,
        max_dist: f32,
    ) -> Result<Vec<(usize, usize, f32)>>;

    /// ICP Jacobian Accumulation
    /// Accumulates J^T * J and J^T * r for point-to-plane ICP
    fn icp_accumulate<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        source: &Tensor<f32, S>,
        target: &Tensor<f32, S>,
        target_normals: &Tensor<f32, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<f32>,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)>;

    fn dense_icp_step<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        source_depth: &Tensor<f32, S>,
        target_data: &Tensor<f32, S>,
        intrinsics: &[f32; 4],
        initial_guess: &nalgebra::Matrix4<f32>,
        max_dist: f32,
        max_angle: f32,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)>;

    /// AKAZE Non-linear Diffusion step
    fn akaze_diffusion<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        k: f32,
        tau: f32,
    ) -> Result<Tensor<f32, S>>;

    /// AKAZE Derivatives and Hessian Determinant
    fn akaze_derivatives<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)>;

    /// Compute AKAZE Contrast K factor (70th percentile)
    fn akaze_contrast_k<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<f32>;

    /// Sparse Matrix-Vector Multiply (y = A * x)
    fn spmv<S: Storage<f32> + cv_core::StorageFactory<f32> + 'static>(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[f32],
        x: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, S>>;

    fn mog2_update<S1: Storage<f32> + 'static, S2: Storage<u32> + 'static>(
        &self,
        frame: &Tensor<f32, S1>,
        model: &mut Tensor<f32, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &Mog2Params,
    ) -> Result<()>;

    /// Compute disparity map from stereo pairs
    fn stereo_match<
        S: Storage<u8> + 'static,
        OS: Storage<f32> + cv_core::StorageFactory<f32> + 'static,
    >(
        &self,
        left: &Tensor<u8, S>,
        right: &Tensor<u8, S>,
        params: &StereoMatchParams,
    ) -> Result<Tensor<f32, OS>>;

    /// Triangulate 3D points from 2D correspondences
    fn triangulate_points<
        S: Storage<f32> + 'static,
        OS: Storage<f32> + cv_core::StorageFactory<f32> + 'static,
    >(
        &self,
        proj_left: &[[f32; 4]; 3],
        proj_right: &[[f32; 4]; 3],
        points_left: &Tensor<f32, S>,
        points_right: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, OS>>;

    /// Accelerated chessboard corner detection
    fn find_chessboard_corners<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        image: &Tensor<u8, S>,
        pattern_size: (usize, usize),
    ) -> Result<Vec<cv_core::KeyPoint>>;
}

#[derive(Debug, Clone, Copy)]
pub struct StereoMatchParams {
    pub method: StereoMatchMethod,
    pub min_disparity: i32,
    pub num_disparities: i32,
    pub block_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoMatchMethod {
    BlockMatching,
    SemiGlobalMatching,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Mog2Params {
    pub width: u32,
    pub height: u32,
    pub n_mixtures: u32,
    pub alpha: f32,
    pub var_threshold: f32,
    pub background_ratio: f32,
    pub var_init: f32,
    pub var_min: f32,
    pub var_max: f32,
    pub _padding: [u32; 3], // Align to 16 bytes for WGSL
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversion {
    RgbToGray,
    BgrToGray,
    GrayToRgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpType {
    Affine,
    Perspective,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyType {
    Erode,
    Dilate,
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    Constant(f32),
    Replicate,
    Reflect,
    Wrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateMatchMethod {
    SqDiff,
    SqDiffNormed,
    Ccorr,
    CcorrNormed,
    Ccoeff,
    CcoeffNormed,
}
