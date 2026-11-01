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
//! - **Batching**: A batch of 1 is supported. An output batch greater than 1 is
//!   rejected: cv-core tensors are CHW and have no batch axis.
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
                )));
            }
            None => {
                return Err(Error::InvalidInput(format!(
                    "Model input dimension {axis} is not statically known ({dim}); DnnNet cannot \
                     resolve symbolic or unknown dimensions"
                )));
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
    /// - 4D outputs: (C, H, W) when the NCHW batch dimension is 1. A batch other
    ///   than 1 is an error; the tensor type cannot store it.
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

        let (channels, height, width) =
            (input_shape_vec[1], input_shape_vec[2], input_shape_vec[3]);
        let got = (input.shape.channels, input.shape.height, input.shape.width);
        if got != (channels, height, width) {
            return Err(Error::InvalidInput(format!(
                "input tensor shape (channels, height, width) = {got:?} does not match the \
                 model's (channels, height, width) = {:?}; the flat buffer would be silently \
                 reinterpreted as the model's layout",
                (channels, height, width)
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

            let dims: Vec<usize> = (0..shape.len()).map(|i| shape[i]).collect();
            outputs.push(tensor_from_model_output(&dims, data)?);
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

/// Wrap a model output in a cv-core tensor.
///
/// `cv_core::TensorShape` is CHW only: it has no batch axis. A rank-4 NCHW
/// output with batch 1 is stored as `(C, H, W)`. A larger batch cannot be
/// represented, and is rejected here — building `(C, H, W)` anyway makes
/// `Tensor::from_vec` fail with a length mismatch that blames the buffer
/// instead of the unsupported batch.
fn tensor_from_model_output(dims: &[usize], data: Vec<f32>) -> Result<Tensor<f32>> {
    let tensor_shape = match dims.len() {
        1 => cv_core::TensorShape::new(1, 1, dims[0]),
        2 => cv_core::TensorShape::new(1, dims[0], dims[1]),
        3 => cv_core::TensorShape::new(dims[0], dims[1], dims[2]),
        4 => {
            let batch = dims[0];
            if batch != 1 {
                return Err(Error::InvalidInput(format!(
                    "batch size {batch} is unsupported: cv_core::Tensor is CHW and has no batch \
                     dimension, so an NCHW output cannot be stored (shape {dims:?})"
                )));
            }
            cv_core::TensorShape::new(dims[1], dims[2], dims[3])
        }
        _ => cv_core::TensorShape::new(1, 1, dims.iter().product()),
    };

    Tensor::from_vec(data, tensor_shape).map_err(|e| Error::RuntimeError(e.to_string()))
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

    // Planar (CHW): the tensor is labelled `TensorShape::new(channels, h, w)`
    // and handed to tract as NCHW, so the luma plane is repeated once per
    // channel. Repeating each *pixel* `channels` times wrote interleaved HWC
    // data under a CHW label: plane 0 held the first third of the image with
    // every pixel tripled, and every multi-channel model saw a scrambled
    // input. Single-channel models were unaffected, which hid it.
    let plane: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    let mut data: Vec<f32> = Vec::with_capacity(plane.len() * channels);
    for _ in 0..channels {
        data.extend_from_slice(&plane);
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

    #[test]
    fn test_preprocess_is_planar_chw() {
        // Each channel plane must be the whole image in row-major order, not
        // the first third of it with every pixel repeated.
        let group = cv_runtime::orchestrator::scheduler()
            .expect("scheduler")
            .get_default_group()
            .expect("default group");
        let src = GrayImage::from_fn(4, 4, |x, y| image::Luma([(16 * (y * 4 + x)) as u8]));
        let img = DynamicImage::ImageLuma8(src.clone());

        let tensor = preprocess_grayscale(&img, 3, 4, 4, &group).expect("preprocess failed");
        let data = tensor.as_slice().expect("slice");
        let plane: Vec<f32> = src.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
        for c in 0..3 {
            assert_eq!(&data[c * 16..(c + 1) * 16], plane.as_slice(), "plane {c}");
        }
    }

    #[test]
    fn rank4_batch_of_one_loads_as_chw() {
        let data = vec![0.25f32; 3 * 4 * 5];
        let tensor = tensor_from_model_output(&[1, 3, 4, 5], data.clone()).expect("batch 1");
        assert_eq!(tensor.shape.channels, 3);
        assert_eq!(tensor.shape.height, 4);
        assert_eq!(tensor.shape.width, 5);
        assert_eq!(tensor.as_slice().expect("slice"), data.as_slice());
    }

    // -- a minimal in-memory ONNX Identity model, so `forward` can be exercised
    // without a binary fixture ------------------------------------------------

    fn pb_varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            out.push(b);
            if v == 0 {
                break;
            }
        }
    }

    fn pb_key(field: u64, wire: u64, out: &mut Vec<u8>) {
        pb_varint((field << 3) | wire, out);
    }

    fn pb_len(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
        pb_key(field, 2, out);
        pb_varint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
    }

    fn pb_int(field: u64, v: i64, out: &mut Vec<u8>) {
        pb_key(field, 0, out);
        pb_varint(v as u64, out);
    }

    fn pb_str(field: u64, s: &str, out: &mut Vec<u8>) {
        pb_len(field, s.as_bytes(), out);
    }

    fn pb_value_info(name: &str, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &d in dims {
            let mut dim = Vec::new();
            pb_int(1, d, &mut dim);
            pb_len(1, &dim, &mut shape);
        }
        let mut tensor = Vec::new();
        pb_int(1, 1, &mut tensor); // elem_type = FLOAT
        pb_len(2, &shape, &mut tensor);
        let mut ty = Vec::new();
        pb_len(1, &tensor, &mut ty);
        let mut vi = Vec::new();
        pb_str(1, name, &mut vi);
        pb_len(2, &ty, &mut vi);
        vi
    }

    fn identity_onnx(dims: &[i64]) -> Vec<u8> {
        let mut node = Vec::new();
        pb_str(1, "x", &mut node);
        pb_str(2, "y", &mut node);
        pb_str(4, "Identity", &mut node);

        let mut graph = Vec::new();
        pb_len(1, &node, &mut graph);
        pb_str(2, "g", &mut graph);
        pb_len(11, &pb_value_info("x", dims), &mut graph);
        pb_len(12, &pb_value_info("y", dims), &mut graph);

        let mut opset = Vec::new();
        pb_int(2, 13, &mut opset);

        let mut model = Vec::new();
        pb_int(1, 7, &mut model); // ir_version
        pb_len(7, &graph, &mut model);
        pb_len(8, &opset, &mut model);
        model
    }

    fn runnable_identity(dims: &[i64]) -> RunnableModel {
        let bytes = identity_onnx(dims);
        let mut reader: &[u8] = &bytes;
        tract_onnx::onnx()
            .model_for_read(&mut reader)
            .expect("parse")
            .into_optimized()
            .expect("optimize")
            .into_runnable()
            .expect("runnable")
    }

    #[test]
    fn a_tensor_with_the_wrong_shape_is_rejected_not_reinterpreted() {
        let net = DnnNet {
            model: Arc::new(runnable_identity(&[1, 1, 2, 3])),
            input_shape: vec![1, 1, 2, 3],
        };
        // The model wants CHW (1, 2, 3). This tensor is (1, 3, 2): the same six
        // values, but a different image - a transposed one. Re-labelling the
        // flat buffer as (1, 2, 3) silently returns a different picture.
        let input = cv_core::Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            cv_core::TensorShape::new(1, 3, 2),
        )
        .expect("input tensor");

        let err = net
            .forward(&input)
            .expect_err("a tensor of a different shape must not be accepted");
        let msg = err.to_string();
        assert!(
            msg.contains("shape"),
            "error should name the shape mismatch, got {msg}"
        );
    }

    #[test]
    fn a_matching_tensor_still_runs_through_identity() {
        let net = DnnNet {
            model: Arc::new(runnable_identity(&[1, 1, 2, 3])),
            input_shape: vec![1, 1, 2, 3],
        };
        let data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let input = cv_core::Tensor::from_vec(data.clone(), cv_core::TensorShape::new(1, 2, 3))
            .expect("input tensor");

        let outputs = net.forward(&input).expect("matching shape runs");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape.channels, 1);
        assert_eq!(outputs[0].shape.height, 2);
        assert_eq!(outputs[0].shape.width, 3);
        assert_eq!(outputs[0].as_slice().expect("slice"), data.as_slice());
    }

    #[test]
    fn rank4_batch_of_two_is_rejected_before_a_lying_shape() {
        // NCHW (2, 3, 4, 5) has 120 values. Dropping the batch and labelling
        // the buffer (3, 4, 5) asks `Tensor::from_vec` for 60 and returns
        // `DimensionMismatch` ("Data size mismatch") instead of saying that a
        // batch greater than 1 cannot be stored.
        let err = tensor_from_model_output(&[2, 3, 4, 5], vec![0.0f32; 2 * 3 * 4 * 5])
            .expect_err("batch 2 is not representable");
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("batch"),
            "error should name the unsupported batch, got {msg}"
        );
        assert!(
            !msg.contains("Data size mismatch"),
            "shape must not be built so the length check is the one that fails: {msg}"
        );
    }
}
