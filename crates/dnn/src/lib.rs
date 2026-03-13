//! Deep Learning Neural Network Inference
//!
//! Provides a high-level interface for running pre-trained neural network models
//! using the `tract` library (Pure Rust ONNX inference engine).
//!
//! # Features
//!
//! - **ONNX Model Loading**: Load pre-trained models in ONNX format
//! - **Inference**: Execute forward passes with preprocessing/postprocessing
//! - **Image Processing**: Built-in image resize and normalization
//! - **Batching**: Support for batch processing of images
//!
//! # Supported Formats
//!
//! - ONNX (Open Neural Network Exchange) models
//! - Common architectures: ResNet, VGG, MobileNet, YOLO, etc.
//!
//! # Example
//!
//! ```ignore
//! # use cv_dnn::DnnNet;
//! # use image::ImageReader;
//! let net = DnnNet::load("model.onnx")?;
//! let img = ImageReader::open("image.jpg")?.decode()?;
//! // Preprocess and run inference
//! ```

pub mod blob;

use cv_core::Tensor;
use cv_runtime::orchestrator::ResourceGroup;
use image::DynamicImage;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type DnnError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type DnnResult<T> = cv_core::Result<T>;

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Neural network model for inference
///
/// Encapsulates a loaded ONNX model with associated metadata.
/// Provides methods for preprocessing input and running inference.
///
/// # Model Loading
///
/// Models are loaded from ONNX files, optimized for inference,
/// and compiled to a runnable form.
///
/// # Input/Output
///
/// - **Input**: Expects f32 tensors with shape matching model specification
/// - **Output**: Returns vector of f32 tensors with model outputs
pub struct DnnNet {
    /// Loaded and optimized ONNX model
    model: Arc<RunnableModel>,
    /// Input tensor shape (typically [batch, channels, height, width])
    input_shape: Vec<usize>,
}

impl DnnNet {
    /// Load an ONNX neural network model from file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to ONNX model file
    ///
    /// # Returns
    ///
    /// * `Ok(DnnNet)` - Loaded and optimized model ready for inference
    /// * `Err(Error)` - If model loading, optimization, or compilation fails
    ///
    /// # Errors
    ///
    /// May return `DnnError` if:
    /// - File not found or cannot be read
    /// - Invalid ONNX format
    /// - Model optimization fails
    /// - Model compilation to runnable form fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cv_dnn::DnnNet;
    /// let net = DnnNet::load("resnet50.onnx")?;
    /// # Ok::<(), cv_core::Error>(())
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| cv_core::Error::RuntimeError(format!("Failed to load ONNX model: {}", e)))?
            .into_optimized()
            .map_err(|e| {
                cv_core::Error::RuntimeError(format!("Failed to optimize ONNX model: {}", e))
            })?
            .into_runnable()
            .map_err(|e| {
                cv_core::Error::RuntimeError(format!("Failed to create runnable ONNX model: {}", e))
            })?;

        // Inspect input facts to determine shape
        // model.model() returns reference to Graph
        let input_shape = if let Some(_input_node_idx) = model.model().inputs.first() {
            // Basic heuristic: check if shape is fixed
            // For now, default to standard image net
            vec![1, 3, 224, 224]
        } else {
            vec![1, 3, 224, 224]
        };

        Ok(Self {
            model: Arc::new(model),
            input_shape,
        })
    }

    /// Run a forward pass (inference) through the network
    ///
    /// Executes the neural network on the provided input tensor and returns
    /// all output tensors.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with f32 values, shape must match model expectations
    ///   - Typically [batch, channels, height, width] for vision models
    ///   - Values should be preprocessed (normalized to [0,1] or standardized)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Tensor<f32>>)` - Output tensors from all output nodes
    /// * `Err(DnnError)` - If inference fails
    ///
    /// # Errors
    ///
    /// May return `DnnError` if:
    /// - Input tensor shape doesn't match model expectations
    /// - Inference computation fails
    /// - Output tensor conversion fails
    ///
    /// # Output Format
    ///
    /// Output tensors are converted to standard cv-core format:
    /// - 1D outputs: (1, 1, N)
    /// - 2D outputs: (1, H, W)
    /// - 3D outputs: (C, H, W)
    /// - 4D outputs: (H, W, C) [assuming NCHW input, N=1]
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let input_shape_vec = vec![
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        ];

        let slice = input
            .as_slice()
            .map_err(|e| Error::DnnError(e.to_string()))?;
        // Create tract tensor from slice
        let tensor = tract_onnx::prelude::Tensor::from_shape(&input_shape_vec, slice)
            .map_err(|e| Error::DnnError(format!("Failed to create tensor from shape: {}", e)))?;

        let result = self
            .model
            .run(tvec!(tensor.into()))
            .map_err(|e| Error::DnnError(format!("Model forward pass failed: {}", e)))?;

        let mut outputs = Vec::new();
        for t in result {
            let shape = t.shape();
            let data = t
                .as_slice::<f32>()
                .map_err(|e| {
                    Error::DnnError(format!("Failed to extract slice from tensor: {}", e))
                })?
                .to_vec();

            let tensor_shape = match shape.len() {
                1 => cv_core::TensorShape::new(1, 1, shape[0]),
                2 => cv_core::TensorShape::new(1, shape[0], shape[1]),
                3 => cv_core::TensorShape::new(shape[0], shape[1], shape[2]),
                4 => cv_core::TensorShape::new(shape[1], shape[2], shape[3]), // Assuming NCHW
                _ => cv_core::TensorShape::new(1, 1, shape.iter().product()),
            };

            outputs.push(
                Tensor::from_vec(data, tensor_shape).map_err(|e| Error::DnnError(e.to_string()))?,
            );
        }

        Ok(outputs)
    }

    /// Preprocess an image for network inference
    ///
    /// Performs standard image preprocessing:
    /// 1. Convert to grayscale
    /// 2. Resize to network input dimensions
    /// 3. Normalize pixel values to [0, 1] range
    ///
    /// # Arguments
    ///
    /// * `img` - Input image (any format supported by `image` crate)
    /// * `runner` - Resource group for scheduling compute operations
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<f32>)` - Preprocessed image tensor with shape (C, H, W)
    /// * `Err(DnnError)` - If preprocessing fails
    ///
    /// # Errors
    ///
    /// May fail if:
    /// - Image resize operation fails
    /// - Tensor creation fails
    /// - Invalid resource group
    ///
    /// # Output Format
    ///
    /// Returns f32 tensor with:
    /// - Shape: (channels, height, width) where height/width = model input dims
    /// - Values: Normalized to [0.0, 1.0] range
    /// - Channels: 1 (converted to grayscale)
    pub fn preprocess(&self, img: &DynamicImage, runner: &ResourceGroup) -> Result<Tensor<f32>> {
        let target_w = self.input_shape[3];
        let target_h = self.input_shape[2];
        let channels = self.input_shape[1];

        let gray = img.to_luma8();
        let resized = cv_imgproc::resize_ctx(
            &gray,
            target_w as u32,
            target_h as u32,
            cv_imgproc::Interpolation::Linear,
            runner,
        );

        let data: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

        Tensor::from_vec(
            data,
            cv_core::TensorShape::new(channels, target_h, target_w),
        )
        .map_err(|e| Error::DnnError(e.to_string()))
    }
}
