use crate::{BackendType, DeviceId, Result};
use cv_core::{storage::Storage, Float, Tensor};

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
    fn convolve_2d<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        kernel: &Tensor<T, S>,
        border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>>;

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
    fn threshold<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        thresh: T,
        max_value: T,
        typ: ThresholdType,
    ) -> Result<Tensor<T, S>>;

    /// Execute a Sobel operator
    fn sobel<
        T: Float + bytemuck::Pod + std::fmt::Debug + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>)>;

    /// Execute Canny edge detection
    fn canny<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        low_threshold: T,
        high_threshold: T,
    ) -> Result<Tensor<T, S>>;

    /// Execute Hough line transform
    fn hough_lines<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        rho: T,
        theta: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughLine>>;

    /// Execute Hough circle transform
    fn hough_circles<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        min_radius: T,
        max_radius: T,
        threshold: u32,
    ) -> Result<Vec<cv_core::HoughCircle>>;

    /// Template matching
    fn match_template<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        template: &Tensor<T, S>,
        method: TemplateMatchMethod,
    ) -> Result<Tensor<T, OS>>;

    /// Execute a morphological operation
    fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>>;

    /// Execute a warp operation (affine or perspective)
    fn warp<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        matrix: &[[T; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<T, S>>;

    /// Execute Non-Maximum Suppression
    fn nms<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        window_size: usize,
    ) -> Result<Tensor<T, S>>;

    /// Execute Bounding Box NMS
    /// input: (N, 5) tensor [x1, y1, x2, y2, score]
    fn nms_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>>;

    /// Execute Rotated Bounding Box NMS
    /// input: (N, 6) tensor [cx, cy, w, h, angle, score]
    fn nms_rotated_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>>;

    /// Execute Polygon NMS
    /// input: list of polygons, list of scores
    fn nms_polygons<T: Float + 'static>(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[T],
        iou_threshold: T,
    ) -> Result<Vec<usize>>;

    /// Transform a point cloud
    fn pointcloud_transform<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        transform: &[[T; 4]; 4],
    ) -> Result<Tensor<T, S>>;

    /// Compute normals for a point cloud
    fn pointcloud_normals<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<T, S>>;

    /// TSDF Volume Integration
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
    ) -> Result<()>;

    /// Execute object detection (e.g., via a loaded ONNX/TensorRT model)
    fn detect_objects<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
    ) -> Result<Vec<cv_core::Detection>>;

    #[allow(clippy::too_many_arguments)]
    fn tsdf_raycast<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        tsdf_volume: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        image_size: (u32, u32),
        depth_range: (T, T),
        voxel_size: T,
        truncation: T,
    ) -> Result<Tensor<T, S>>;

    fn tsdf_extract_mesh<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        tsdf_volume: &Tensor<T, S>,
        voxel_size: T,
        iso_level: T,
        max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>>;

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
    ) -> Result<Vec<[T; 2]>>;

    /// Color space conversion
    fn cvt_color<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        code: ColorConversion,
    ) -> Result<Tensor<T, S>>;

    /// Resize an image
    fn resize<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<T, S>>;

    /// Create an image pyramid level (downsample)
    fn pyramid_down<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>>;

    /// Bilateral Filter
    fn bilateral_filter<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        d: i32,
        sigma_color: T,
        sigma_space: T,
    ) -> Result<Tensor<T, S>>;

    /// FAST Keypoint Detection
    /// Returns a score map (1 channel, same size as input)
    fn fast_detect<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        non_max_suppression: bool,
    ) -> Result<Tensor<T, S>>;

    /// Gaussian Blur
    fn gaussian_blur<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        sigma: T,
        k_size: usize,
    ) -> Result<Tensor<T, S>>;

    /// Elementwise Subtraction (A - B)
    /// Input: Signed output often needed for DoG, but we might use f32 or i16.
    /// For SIFT DoG, we usually use f32.
    fn subtract<
        T: Float + 'static + bytemuck::Pod + std::fmt::Debug,
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
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>>;

    /// SIFT Descriptor Computation
    /// Returns a descriptor tensor (num_kps x 128)
    fn compute_sift_descriptors<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors>;

    /// ICP Point Correspondence
    /// Finds nearest neighbors from src points to tgt points
    fn icp_correspondences<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        src: &Tensor<T, S>,
        tgt: &Tensor<T, S>,
        max_dist: T,
    ) -> Result<Vec<(usize, usize, T)>>;

    /// ICP Jacobian Accumulation
    /// Accumulates J^T * J and J^T * r for point-to-plane ICP
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
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)>;

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
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)>;

    /// AKAZE Non-linear Diffusion step
    fn akaze_diffusion<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        k: T,
        tau: T,
    ) -> Result<Tensor<T, S>>;

    /// AKAZE Derivatives and Hessian Determinant
    #[allow(clippy::type_complexity)]
    fn akaze_derivatives<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)>;

    /// Compute AKAZE Contrast K factor (70th percentile)
    fn akaze_contrast_k<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<T>;

    /// Sparse Matrix-Vector Multiply (y = A * x)
    fn spmv<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[T],
        x: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>>;

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
    ) -> Result<()>;

    /// Compute disparity map from stereo pairs
    fn stereo_match<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + 'static,
        OS: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        left: &Tensor<T, S>,
        right: &Tensor<T, S>,
        params: &StereoMatchParams,
    ) -> Result<Tensor<T, OS>>;

    /// Triangulate 3D points from 2D correspondences
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
    ) -> Result<Tensor<T, OS>>;

    /// Accelerated chessboard corner detection
    fn find_chessboard_corners<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
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
#[derive(Debug, Clone, Copy)]
pub struct Mog2Params<T: Float> {
    pub width: u32,
    pub height: u32,
    pub n_mixtures: u32,
    pub alpha: T,
    pub var_threshold: T,
    pub background_ratio: T,
    pub var_init: T,
    pub var_min: T,
    pub var_max: T,
    pub _padding: [u32; 3], // Align to 16 bytes for WGSL
}

// Safety: All fields are Pod when T: Pod. Repr(C) guarantees no padding surprises.
unsafe impl<T: Float + bytemuck::Pod> bytemuck::Pod for Mog2Params<T> {}
unsafe impl<T: Float + bytemuck::Zeroable> bytemuck::Zeroable for Mog2Params<T> {}

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
pub enum BorderMode<T: Float> {
    Constant(T),
    Replicate,
    Reflect,
    /// Reflect without duplicating the border pixel (period = 2n-2).
    /// This is the default border mode in OpenCV (`BORDER_REFLECT_101`).
    Reflect101,
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
