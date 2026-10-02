#![forbid(unsafe_code)]
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

/// Derive the network input shape from the model's declared input fact.
///
/// The shape must be fully static (every dimension a positive integer, rank 4)
/// because `forward` and `preprocess` both index it positionally
/// (`self.input_shape[0..4]`) and cannot supply a value for a symbolic or
/// absent dimension.
///
/// # Returns
/// `Ok(shape)` when the model's first input fact pins a rank-4 shape,
/// otherwise `Err` explaining exactly what could not be determined.
fn fixed_rank4_input_shape(model: &RunnableModel) -> Result<Vec<usize>> {
    let outlet = *model
        .model()
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidInput("Model declares no input nodes".into()))?;

    let fact = model.model().outlet_fact(outlet).map_err(|e| {
        Error::InvalidInput(format!(
            "Could not read the model's input fact ({outlet:?}): {e}"
        ))
    })?;

    let dims: &[tract_onnx::prelude::TDim] = fact.shape.dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "Model input must be a rank-4 tensor [batch, channels, height, width], got rank {} \
             (shape {:?})",
            dims.len(),
            dims
        )));
    }
    let mut shape = Vec::with_capacity(4);
    for (axis, dim) in dims.iter().enumerate() {
        match dim.as_i64() {
            Some(v) if v > 0 => shape.push(v as usize),
            Some(v) => {
                return Err(Error::InvalidInput(format!(
                    "Model input dimension {axis} is {v}; every dimension must be a positive \
                     integer"
                )))
            }
            None => {
                return Err(Error::InvalidInput(format!(
                    "Model input dimension {axis} is not statically known ({dim}); DnnNet cannot \
                     resolve symbolic or unknown dimensions"
                )))
            }
        }
    }
    Ok(shape)
}

impl DnnNet {
    /// Load an ONNX neural network model from file.
    ///
    /// The network input shape is read from the model's own input fact. It is
    /// never guessed: a model whose input is not a fully static rank-4 tensor
    /// is rejected with an explicit error (this used to be silently hardcoded
    /// to `[1, 3, 224, 224]` in both branches of a dead `if`, so any other model
    /// either raised a `RuntimeError` on the first forward pass or silently got
    /// a layout mismatch).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to ONNX model file
    ///
    /// # Returns
    ///
    /// * `Ok(DnnNet)` - Loaded and optimized model ready for inference
    /// * `Err(Error)` - If model loading, optimization, or compilation fails, or
    ///   if the input shape cannot be determined
    ///
    /// # Errors
    ///
    /// May return `Error` if:
    /// - File not found or cannot be read
    /// - Invalid ONNX format
    /// - Model optimization fails
    /// - Model compilation to runnable form fails
    /// - The model's input is missing, dynamic, or not rank 4
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
            .model_for_path(&path)
            .map_err(|e| cv_core::Error::RuntimeError(format!("Failed to load ONNX model: {}", e)))?
            .into_optimized()
            .map_err(|e| {
                cv_core::Error::RuntimeError(format!("Failed to optimize ONNX model: {}", e))
            })?
            .into_runnable()
            .map_err(|e| {
                cv_core::Error::RuntimeError(format!("Failed to create runnable ONNX model: {}", e))
            })?;

        let input_shape = fixed_rank4_input_shape(&model)?;

        Ok(Self {
            model: Arc::new(model),
            input_shape,
        })
    }

    /// The network input shape, as read from the model at load time.
    ///
    /// Format: `[batch, channels, height, width]`.
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
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
        let input_shape_vec: Vec<usize> = self.input_shape.clone();
        if input_shape_vec.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "DnnNet was loaded with a rank-{} input shape {:?}; forward requires a rank-4 \
                 [batch, channels, height, width] shape",
                input_shape_vec.len(),
                input_shape_vec
            )));
        }

        let slice = input
            .as_slice()
            .map_err(|e| Error::RuntimeError(e.to_string()))?;
        // Create tract tensor from slice
        let tensor =
            tract_onnx::prelude::Tensor::from_shape(&input_shape_vec, slice).map_err(|e| {
                Error::RuntimeError(format!("Failed to create tensor from shape: {}", e))
            })?;

        let result = self
            .model
            .run(tvec!(tensor.into()))
            .map_err(|e| Error::RuntimeError(format!("Model forward pass failed: {}", e)))?;

        let mut outputs = Vec::new();
        for t in result {
            let shape = t.shape();
            let data = t
                .as_slice::<f32>()
                .map_err(|e| {
                    Error::RuntimeError(format!("Failed to extract slice from tensor: {}", e))
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
                Tensor::from_vec(data, tensor_shape)
                    .map_err(|e| Error::RuntimeError(e.to_string()))?,
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
    /// - Shape: (channels, height, width) where channels/height/width are the
    ///   model's own input dims (read from the model, not assumed)
    /// - Values: Normalized to [0.0, 1.0] range
    /// - The single luma plane is replicated across `channels`
    pub fn preprocess(&self, img: &DynamicImage, runner: &ResourceGroup) -> Result<Tensor<f32>> {
        let (channels, target_h, target_w) = self.input_chw().ok_or_else(|| {
            Error::InvalidInput(format!(
                "DnnNet input shape {:?} is not rank-4 [batch, channels, height, width]",
                self.input_shape
            ))
        })?;
        preprocess_grayscale(img, channels, target_h, target_w, runner)
    }

    /// The model's `[channels, height, width]` input dims, or `None` if the
    /// loaded shape is not rank 4.
    pub fn input_chw(&self) -> Option<(usize, usize, usize)> {
        match self.input_shape.as_slice() {
            [_, c, h, w] => Some((*c, *h, *w)),
            _ => None,
        }
    }
}

/// Convert an image to a normalized `(channels, height, width)` f32 tensor.
///
/// The image is converted to a single luma plane and that plane is replicated
/// across `channels` so the tensor's element count matches the model's input
/// shape. Previously the luma plane was wrapped in a 3-channel shape, which made
/// `Tensor::from_vec` fail with a `DimensionMismatch` for the standard
/// `[1, 3, H, W]` input shape.
fn preprocess_grayscale(
    img: &DynamicImage,
    channels: usize,
    target_h: usize,
    target_w: usize,
    runner: &ResourceGroup,
) -> Result<Tensor<f32>> {
    let channels = channels.max(1);

    let gray = img.to_luma8();
    let resized = cv_imgproc::resize_ctx(
        &gray,
        target_w as u32,
        target_h as u32,
        cv_imgproc::Interpolation::Linear,
        runner,
    );

    let mut data: Vec<f32> = Vec::with_capacity(target_w * target_h * channels);
    for &v in resized.as_raw().iter() {
        let normalized = v as f32 / 255.0;
        for _ in 0..channels {
            data.push(normalized);
        }
    }

    Tensor::from_vec(
        data,
        cv_core::TensorShape::new(channels, target_h, target_w),
    )
    .map_err(|e| Error::RuntimeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    fn test_preprocess_produces_tensor_matching_model_shape() {
        // Regression: the luma plane must be replicated to `channels` so the
        // tensor length matches TensorShape::new(channels, h, w).
        let group = cv_runtime::orchestrator::scheduler()
            .expect("scheduler")
            .get_default_group()
            .expect("default group");
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, image::Luma([128u8])));

        let tensor = preprocess_grayscale(&img, 3, 4, 4, &group).expect("preprocess failed");
        assert_eq!(tensor.shape.channels, 3);
        assert_eq!(tensor.shape.height, 4);
        assert_eq!(tensor.shape.width, 4);
        assert_eq!(tensor.as_slice().expect("slice").len(), 3 * 4 * 4);
    }
}
