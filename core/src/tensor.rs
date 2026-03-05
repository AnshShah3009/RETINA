use crate::storage::{CpuStorage, Storage};
use std::default::Default;
use std::fmt;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    U8,
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    pub fn hw(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn chw(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn len(&self) -> usize {
        self.channels
            .saturating_mul(self.height)
            .saturating_mul(self.width)
    }

    pub fn checked_len(&self) -> Option<usize> {
        self.channels
            .checked_mul(self.height)
            .and_then(|partial| partial.checked_mul(self.width))
    }

    pub fn is_1d(&self) -> bool {
        self.height == 1 && self.width == 1
    }

    pub fn is_2d(&self) -> bool {
        self.channels == 1
    }

    pub fn is_3d(&self) -> bool {
        self.channels > 1
    }
}

/// N-dimensional array abstraction.
///
/// **Layout Convention:**
/// `rust-cv-native` strictly uses the **CHW (Channel-Height-Width)** layout (also known as channel-first).
/// Data is stored contiguously in memory with Width as the fastest-varying dimension,
/// followed by Height, and then Channels.
///
/// For a 3D tensor with dimensions (C, H, W), the element at (c, h, w) is located at:
/// `index = c * (H * W) + h * W + w`
#[derive(Debug, Clone)]
pub struct Tensor<T: Clone + Copy + 'static, S: Storage<T> = CpuStorage<T>> {
    pub storage: S,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub _phantom: PhantomData<T>,
}

pub type CpuTensor<T> = Tensor<T, CpuStorage<T>>;

impl<T: Clone + Copy + fmt::Debug + 'static> CpuTensor<T> {
    /// Extract a sub-tensor from this tensor.
    ///
    /// The sub-tensor is defined by ranges for channels, height, and width.
    /// This operation creates a new tensor with a copy of the data.
    pub fn slice(
        &self,
        c_range: std::ops::Range<usize>,
        h_range: std::ops::Range<usize>,
        w_range: std::ops::Range<usize>,
    ) -> crate::Result<Self> {
        if c_range.end > self.shape.channels
            || h_range.end > self.shape.height
            || w_range.end > self.shape.width
        {
            return Err(crate::Error::InvalidInput(
                "Slice range out of bounds".into(),
            ));
        }

        let new_channels = c_range.len();
        let new_height = h_range.len();
        let new_width = w_range.len();
        let new_shape = TensorShape::new(new_channels, new_height, new_width);

        let mut new_data = Vec::with_capacity(new_shape.len());
        let src_data = self.as_slice()?;

        for c in c_range {
            for h in h_range.clone() {
                let start_idx =
                    c * self.shape.height * self.shape.width + h * self.shape.width + w_range.start;
                let end_idx = start_idx + new_width;
                new_data.extend_from_slice(&src_data[start_idx..end_idx]);
            }
        }

        Self::from_vec(new_data, new_shape)
    }

    /// Concatenate multiple tensors along a specified dimension.
    ///
    /// * `tensors`: List of tensors to concatenate.
    /// * `dim`: Dimension to join along (0=C, 1=H, 2=W).
    pub fn concat(tensors: &[&Self], dim: usize) -> crate::Result<Self> {
        if tensors.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Cannot concat empty tensor list".into(),
            ));
        }

        let first_shape = tensors[0].shape;
        let mut new_shape = first_shape;

        // Verify other dimensions match
        for (i, t) in tensors.iter().enumerate().skip(1) {
            match dim {
                0 => {
                    // Concat along Channels
                    if t.shape.height != first_shape.height || t.shape.width != first_shape.width {
                        return Err(crate::Error::DimensionMismatch(format!(
                            "Shape mismatch at index {}: {:?} vs {:?}",
                            i, t.shape, first_shape
                        )));
                    }
                    new_shape.channels += t.shape.channels;
                }
                1 => {
                    // Concat along Height
                    if t.shape.channels != first_shape.channels
                        || t.shape.width != first_shape.width
                    {
                        return Err(crate::Error::DimensionMismatch(format!(
                            "Shape mismatch at index {}: {:?} vs {:?}",
                            i, t.shape, first_shape
                        )));
                    }
                    new_shape.height += t.shape.height;
                }
                2 => {
                    // Concat along Width
                    if t.shape.channels != first_shape.channels
                        || t.shape.height != first_shape.height
                    {
                        return Err(crate::Error::DimensionMismatch(format!(
                            "Shape mismatch at index {}: {:?} vs {:?}",
                            i, t.shape, first_shape
                        )));
                    }
                    new_shape.width += t.shape.width;
                }
                _ => {
                    return Err(crate::Error::InvalidInput(
                        "Invalid dimension for concat (must be 0, 1, or 2)".into(),
                    ))
                }
            }
        }

        let mut new_data = Vec::with_capacity(new_shape.len());

        match dim {
            0 => {
                // CHW layout: Concatenating along C is just appending the vectors
                for t in tensors {
                    new_data.extend_from_slice(t.as_slice()?);
                }
            }
            1 => {
                // Concatenating along H: Must interleave channels
                for c in 0..new_shape.channels {
                    for t in tensors {
                        let c_offset = c * t.shape.height * t.shape.width;
                        let size = t.shape.height * t.shape.width;
                        new_data.extend_from_slice(&t.as_slice()?[c_offset..c_offset + size]);
                    }
                }
            }
            2 => {
                // Concatenating along W: Must interleave channels and rows
                for c in 0..new_shape.channels {
                    for h in 0..new_shape.height {
                        for t in tensors {
                            let idx = c * t.shape.height * t.shape.width + h * t.shape.width;
                            new_data.extend_from_slice(&t.as_slice()?[idx..idx + t.shape.width]);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        Self::from_vec(new_data, new_shape)
    }
}

impl<T: Clone + Copy + fmt::Debug + 'static, S: crate::storage::StorageFactory<T>> Tensor<T, S> {
    pub fn from_vec(data: Vec<T>, shape: TensorShape) -> crate::Result<Self> {
        if data.len() != shape.len() {
            return Err(crate::Error::DimensionMismatch(format!(
                "Data size mismatch: got {}, expected {}",
                data.len(),
                shape.len()
            )));
        }
        let dtype = match std::any::type_name::<T>() {
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            "f64" => DataType::F64,
            _ => DataType::F32,
        };
        Ok(Self {
            storage: S::from_vec(data).map_err(crate::Error::MemoryError)?,
            shape,
            dtype,
            _phantom: PhantomData,
        })
    }

    pub fn reshape(&self, new_shape: TensorShape) -> crate::Result<Self> {
        let len = self.storage.len();

        if len != new_shape.len() {
            return Err(crate::Error::DimensionMismatch(format!(
                "Cannot reshape: size mismatch ({} != {})",
                len,
                new_shape.len()
            )));
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        })
    }

    pub fn as_slice(&self) -> crate::Result<&[T]> {
        self.storage
            .as_slice()
            .ok_or_else(|| crate::Error::RuntimeError("Data not on CPU".into()))
    }

    pub fn as_mut_slice(&mut self) -> crate::Result<&mut [T]> {
        self.storage
            .as_mut_slice()
            .ok_or_else(|| crate::Error::RuntimeError("Data not on CPU".into()))
    }

    pub fn index(&self, c: usize, h: usize, w: usize) -> crate::Result<T> {
        if c >= self.shape.channels || h >= self.shape.height || w >= self.shape.width {
            return Err(crate::Error::RuntimeError("Index out of bounds".into()));
        }
        // Enforce packed CHW layout: length must match shape
        if self.storage.len() != self.shape.len() {
            return Err(crate::Error::RuntimeError(
                "Tensor storage size mismatch".into(),
            ));
        }
        let idx = c * self.shape.height * self.shape.width + h * self.shape.width + w;
        Ok(self.as_slice()?[idx])
    }

    pub fn index_mut(&mut self, c: usize, h: usize, w: usize) -> crate::Result<&mut T> {
        if c >= self.shape.channels || h >= self.shape.height || w >= self.shape.width {
            return Err(crate::Error::RuntimeError("Index out of bounds".into()));
        }
        let (height, width) = (self.shape.height, self.shape.width);
        let idx = c * height * width + h * width + w;
        Ok(&mut self.as_mut_slice()?[idx])
    }

    /// Safe access returning Result
    pub fn get(&self, c: usize, h: usize, w: usize) -> crate::Result<T> {
        self.index(c, h, w)
    }

    pub fn get_mut(&mut self, c: usize, h: usize, w: usize) -> crate::Result<&mut T> {
        self.index_mut(c, h, w)
    }

    /// Create a new tensor with the same metadata but different storage.
    pub fn with_storage<S2: Storage<T>>(self, storage: S2) -> Tensor<T, S2> {
        Tensor {
            storage,
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        }
    }

    /// Attempt to map the storage of this tensor to another type using a closure.
    pub fn try_map_storage<S2: Storage<T>, E, F>(
        self,
        f: F,
    ) -> std::result::Result<Tensor<T, S2>, E>
    where
        F: FnOnce(S) -> std::result::Result<S2, E>,
    {
        Ok(Tensor {
            storage: f(self.storage)?,
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        })
    }
}

impl<T: Clone + Copy + Default + fmt::Debug + 'static> Tensor<T> {
    pub fn new(shape: TensorShape) -> crate::Result<Self> {
        let dtype = match std::any::type_name::<T>() {
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            "f64" => DataType::F64,
            _ => DataType::F32,
        };

        Ok(Self {
            storage: CpuStorage::new(shape.len(), T::default())
                .map_err(crate::Error::RuntimeError)?,
            shape,
            dtype,
            _phantom: PhantomData,
        })
    }

    pub fn zeros(shape: TensorShape) -> crate::Result<Self> {
        Self::new(shape)
    }

    pub fn ones(shape: TensorShape) -> crate::Result<Self>
    where
        T: One,
    {
        let mut t = Self::new(shape)?;
        if let Some(s) = t.storage.as_mut_slice() {
            for v in s.iter_mut() {
                *v = T::one();
            }
        }
        Ok(t)
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}
impl One for f64 {
    fn one() -> Self {
        1.0
    }
}
impl One for u8 {
    fn one() -> Self {
        1
    }
}
impl One for u16 {
    fn one() -> Self {
        1
    }
}
impl One for u32 {
    fn one() -> Self {
        1
    }
}
impl One for i32 {
    fn one() -> Self {
        1
    }
}

impl Tensor<f32, CpuStorage<f32>> {
    /// SIMD-accelerated element-wise addition.
    pub fn add(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(crate::Error::RuntimeError("Tensor shape mismatch".into()));
        }
        let a = self.as_slice()?;
        let b = other.as_slice()?;
        let mut res = vec![0.0f32; a.len()];

        {
            let mut a_chunks = a.chunks_exact(8);
            let mut b_chunks = b.chunks_exact(8);
            let mut res_chunks = res.chunks_exact_mut(8);

            for ((a8, b8), r8) in (&mut a_chunks).zip(&mut b_chunks).zip(&mut res_chunks) {
                let va = wide::f32x8::new(a8.try_into().expect("Chunk size guaranteed to be 8"));
                let vb = wide::f32x8::new(b8.try_into().expect("Chunk size guaranteed to be 8"));
                let vr = va + vb;
                r8.copy_from_slice(&<[f32; 8]>::from(vr));
            }

            let rem_a = a_chunks.remainder();
            let rem_b = b_chunks.remainder();
            let rem_res = res_chunks.into_remainder();

            for i in 0..rem_a.len() {
                rem_res[i] = rem_a[i] + rem_b[i];
            }
        }

        Self::from_vec(res, self.shape)
    }

    /// SIMD-accelerated element-wise subtraction.
    pub fn sub(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(crate::Error::RuntimeError("Tensor shape mismatch".into()));
        }
        let a = self.as_slice()?;
        let b = other.as_slice()?;
        let mut res = vec![0.0f32; a.len()];

        {
            let mut a_chunks = a.chunks_exact(8);
            let mut b_chunks = b.chunks_exact(8);
            let mut res_chunks = res.chunks_exact_mut(8);

            for ((a8, b8), r8) in (&mut a_chunks).zip(&mut b_chunks).zip(&mut res_chunks) {
                let va = wide::f32x8::new(a8.try_into().expect("Chunk size guaranteed to be 8"));
                let vb = wide::f32x8::new(b8.try_into().expect("Chunk size guaranteed to be 8"));
                let vr = va - vb;
                r8.copy_from_slice(&<[f32; 8]>::from(vr));
            }

            let rem_a = a_chunks.remainder();
            let rem_b = b_chunks.remainder();
            let rem_res = res_chunks.into_remainder();

            for i in 0..rem_a.len() {
                rem_res[i] = rem_a[i] - rem_b[i];
            }
        }

        Self::from_vec(res, self.shape)
    }

    /// SIMD-accelerated element-wise multiplication.
    pub fn mul(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(crate::Error::RuntimeError("Tensor shape mismatch".into()));
        }
        let a = self.as_slice()?;
        let b = other.as_slice()?;
        let mut res = vec![0.0f32; a.len()];

        {
            let mut a_chunks = a.chunks_exact(8);
            let mut b_chunks = b.chunks_exact(8);
            let mut res_chunks = res.chunks_exact_mut(8);

            for ((a8, b8), r8) in (&mut a_chunks).zip(&mut b_chunks).zip(&mut res_chunks) {
                let va = wide::f32x8::new(a8.try_into().expect("Chunk size guaranteed to be 8"));
                let vb = wide::f32x8::new(b8.try_into().expect("Chunk size guaranteed to be 8"));
                let vr = va * vb;
                r8.copy_from_slice(&<[f32; 8]>::from(vr));
            }

            let rem_a = a_chunks.remainder();
            let rem_b = b_chunks.remainder();
            let rem_res = res_chunks.into_remainder();

            for i in 0..rem_a.len() {
                rem_res[i] = rem_a[i] * rem_b[i];
            }
        }

        Self::from_vec(res, self.shape)
    }
}

/// CPU-specific convenience APIs for Tensor<T, CpuStorage<T>>.
///
/// This impl block provides convenience methods for working with CPU-based tensors,
/// enabling easier access patterns and migration from older code.
impl<T: Clone + Copy + Default + fmt::Debug + 'static> Tensor<T, CpuStorage<T>> {
    /// Get an immutable reference to the underlying data as a slice.
    ///
    /// This is a convenience wrapper that extracts the data from CPU storage
    /// without requiring pattern matching on `as_slice()`.
    ///
    /// # Panics
    ///
    /// Panics if the storage is not a CpuStorage (should never happen in normal use).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], shape)?;
    /// let data = tensor.cpu_as_slice();
    /// assert_eq!(data.len(), 3);
    /// ```
    pub fn cpu_as_slice(&self) -> &[T] {
        self.storage
            .as_slice()
            .expect("CpuStorage should always provide a slice")
    }

    /// Get a mutable reference to the underlying data as a slice.
    ///
    /// This is a convenience wrapper that extracts mutable data from CPU storage.
    ///
    /// # Panics
    ///
    /// Panics if the storage is not a CpuStorage (should never happen in normal use).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], shape)?;
    /// let data = tensor.cpu_as_mut_slice();
    /// data[0] = 42.0;
    /// ```
    pub fn cpu_as_mut_slice(&mut self) -> &mut [T] {
        self.storage
            .as_mut_slice()
            .expect("CpuStorage should always provide a mutable slice")
    }

    /// Clone the stored data into a new vector.
    ///
    /// This creates an independent copy of the tensor's data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], shape)?;
    /// let vec = tensor.cpu_to_vec();
    /// assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn cpu_to_vec(&self) -> Vec<T> {
        self.storage.to_vec()
    }
}

impl Tensor<f32> {
    pub fn from_image_gray(data: &[u8], width: usize, height: usize) -> crate::Result<Self> {
        let mut float_data = Vec::with_capacity(width * height);
        for &pixel in data {
            float_data.push(pixel as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(1, height, width))
    }

    pub fn from_image_rgb(data: &[u8], width: usize, height: usize) -> crate::Result<Self> {
        let mut float_data = Vec::with_capacity(3 * width * height);
        for chunk in data.chunks(3) {
            float_data.push(chunk[0] as f32 / 255.0);
            float_data.push(chunk[1] as f32 / 255.0);
            float_data.push(chunk[2] as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(3, height, width))
    }

    pub fn to_image_gray(&self) -> crate::Result<Vec<u8>> {
        if !self.shape.is_2d() {
            return Err(crate::Error::RuntimeError(
                "Tensor must be 2D for gray image".into(),
            ));
        }
        Ok(self
            .as_slice()?
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect())
    }
}

impl<T: Clone + Copy + fmt::Debug + 'static, S: Storage<T>> fmt::Display for Tensor<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {}, {:?})",
            self.shape.channels,
            self.shape.height,
            self.shape.width,
            self.storage.device()
        )
    }
}

pub type Tensor3f = Tensor<f32>;
pub type Tensor4f = Tensor<f64>;

pub fn create_tensor_2d<T: Clone + Copy + Default + fmt::Debug + 'static>(
    height: usize,
    width: usize,
) -> crate::Result<Tensor<T>> {
    Tensor::new(TensorShape::new(1, height, width))
}

pub fn create_tensor_3d<T: Clone + Copy + Default + fmt::Debug + 'static>(
    channels: usize,
    height: usize,
    width: usize,
) -> crate::Result<Tensor<T>> {
    Tensor::new(TensorShape::new(channels, height, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod tensor_shape_tests {
        use super::*;

        #[test]
        fn test_tensor_shape_new() {
            let shape = TensorShape::new(3, 100, 200);

            assert_eq!(shape.channels, 3);
            assert_eq!(shape.height, 100);
            assert_eq!(shape.width, 200);
        }

        #[test]
        fn test_tensor_shape_len() {
            let shape = TensorShape::new(3, 100, 200);
            assert_eq!(shape.len(), 3 * 100 * 200);
        }

        #[test]
        fn test_tensor_shape_hw() {
            let shape = TensorShape::new(3, 100, 200);
            let (h, w) = shape.hw();
            assert_eq!(h, 100);
            assert_eq!(w, 200);
        }

        #[test]
        fn test_tensor_shape_chw() {
            let shape = TensorShape::new(3, 100, 200);
            let (c, h, w) = shape.chw();
            assert_eq!(c, 3);
            assert_eq!(h, 100);
            assert_eq!(w, 200);
        }

        #[test]
        fn test_tensor_shape_is_1d() {
            assert!(TensorShape::new(100, 1, 1).is_1d());
            assert!(!TensorShape::new(1, 100, 1).is_1d());
        }

        #[test]
        fn test_tensor_shape_is_2d() {
            assert!(TensorShape::new(1, 100, 200).is_2d());
            assert!(!TensorShape::new(3, 100, 200).is_2d());
        }

        #[test]
        fn test_tensor_shape_is_3d() {
            assert!(TensorShape::new(3, 100, 200).is_3d());
        }
    }

    mod tensor_creation_tests {
        use super::*;

        #[test]
        fn test_tensor_zeros() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            assert_eq!(tensor.shape.channels, 1);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 10);
        }

        #[test]
        fn test_tensor_ones() {
            let tensor: Tensor<f32> = Tensor::ones(TensorShape::new(1, 2, 2)).unwrap();

            let slice = tensor.as_slice().unwrap();
            for &v in slice {
                assert!((v - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn test_tensor_from_vec() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let tensor: Tensor<f32> =
                Tensor::from_vec(data.clone(), TensorShape::new(1, 2, 2)).unwrap();

            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice, &data[..]);
        }

        #[test]
        fn test_tensor_from_vec_wrong_size() {
            let data = vec![1.0f32, 2.0, 3.0];
            let result: Result<Tensor<f32>, _> = Tensor::from_vec(data, TensorShape::new(1, 2, 2));

            assert!(result.is_err());
        }
    }

    mod tensor_slice_tests {
        use super::*;

        #[test]
        fn test_slice_valid() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let slice = tensor.slice(0..1, 2..5, 3..7).unwrap();
            assert_eq!(slice.shape.height, 3);
            assert_eq!(slice.shape.width, 4);
        }

        #[test]
        fn test_slice_out_of_bounds() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let result = tensor.slice(0..1, 0..20, 0..10);
            assert!(result.is_err());
        }

        #[test]
        fn test_slice_empty() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let result = tensor.slice(0..1, 5..5, 0..10);
            assert!(result.is_ok());
        }

        #[test]
        fn test_slice_copy() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            {
                let data = tensor.as_mut_slice().unwrap();
                for v in data.iter_mut() {
                    *v = 42.0;
                }
            }

            let slice = tensor.slice(0..1, 2..5, 3..7).unwrap();
            let slice_data = slice.as_slice().unwrap();
            for &v in slice_data {
                assert!((v - 42.0).abs() < 1e-5);
            }
        }
    }

    mod tensor_concat_tests {
        use super::*;

        fn create_tensor_with_value(shape: TensorShape, val: f32) -> Tensor<f32> {
            let mut t = Tensor::zeros(shape).unwrap();
            let slice = t.as_mut_slice().unwrap();
            for v in slice.iter_mut() {
                *v = val;
            }
            t
        }

        #[test]
        fn test_concat_along_channels() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 2), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(2, 2, 2), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 0).unwrap();

            assert_eq!(result.shape.channels, 3);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.width, 2);
        }

        #[test]
        fn test_concat_along_height() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 4), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 3, 4), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 1).unwrap();

            assert_eq!(result.shape.channels, 1);
            assert_eq!(result.shape.height, 5);
            assert_eq!(result.shape.width, 4);
        }

        #[test]
        fn test_concat_along_width() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 3), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 2, 4), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 2).unwrap();

            assert_eq!(result.shape.channels, 1);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.width, 7);
        }

        #[test]
        fn test_concat_empty_list() {
            let tensors: Vec<&Tensor<f32>> = vec![];

            let result = Tensor::concat(&tensors, 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_concat_shape_mismatch() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 3), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 4, 5), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_concat_invalid_dim() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 2), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 2, 2), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 3);
            assert!(result.is_err());
        }
    }

    mod tensor_operations_tests {
        use super::*;

        #[test]
        fn test_index() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(3, 4, 5)).unwrap();

            {
                let slice = tensor.as_mut_slice().unwrap();
                slice[0] = 1.0;
                slice[1] = 2.0;
            }

            let idx0 = tensor.index(0, 0, 0).unwrap();
            let idx1 = tensor.index(0, 0, 1).unwrap();

            assert!((idx0 - 1.0).abs() < 1e-5);
            assert!((idx1 - 2.0).abs() < 1e-5);
        }

        #[test]
        fn test_index_out_of_bounds() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 2, 2)).unwrap();

            let result = tensor.index(2, 0, 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_index_mut() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 2, 2)).unwrap();

            *tensor.index_mut(0, 0, 0).unwrap() = 5.0;

            assert!((tensor.index(0, 0, 0).unwrap() - 5.0).abs() < 1e-5);
        }

        #[test]
        fn test_reshape() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor: Tensor<f32> = Tensor::from_vec(data, TensorShape::new(1, 2, 3)).unwrap();

            let reshaped = tensor.reshape(TensorShape::new(1, 3, 2)).unwrap();

            assert_eq!(reshaped.shape.height, 3);
            assert_eq!(reshaped.shape.width, 2);
        }
    }

    mod tensor_image_tests {
        use super::*;

        #[test]
        fn test_from_image_gray() {
            let data = vec![0u8, 128, 255];
            let result = Tensor::<f32>::from_image_gray(&data, 3, 1).unwrap();

            assert_eq!(result.shape.width, 3);
            assert_eq!(result.shape.height, 1);
            assert_eq!(result.shape.channels, 1);
        }

        #[test]
        fn test_from_image_gray_wrong_size() {
            let data = vec![0u8, 128];
            let result = Tensor::<f32>::from_image_gray(&data, 2, 2);

            assert!(result.is_err());
        }

        #[test]
        fn test_from_image_rgb() {
            let data = vec![255u8, 0, 0, 0, 255, 0];
            let result = Tensor::<f32>::from_image_rgb(&data, 1, 2).unwrap();

            assert_eq!(result.shape.width, 1);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.channels, 3);
        }
    }

    mod helper_function_tests {
        use super::*;

        #[test]
        fn test_create_tensor_2d() {
            let tensor: Tensor<f32> = create_tensor_2d(10, 20).unwrap();

            assert_eq!(tensor.shape.channels, 1);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 20);
        }

        #[test]
        fn test_create_tensor_3d() {
            let tensor: Tensor<f32> = create_tensor_3d(3, 10, 20).unwrap();

            assert_eq!(tensor.shape.channels, 3);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 20);
        }
    }

    mod tensor_storage_generic_tests {
        use super::*;
        use crate::storage::DeviceType;

        #[test]
        fn test_tensor_cpu_storage_creation() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape = TensorShape::new(1, 2, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            assert_eq!(tensor.storage.len(), 6);
            assert_eq!(tensor.shape, shape);
            assert_eq!(tensor.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn test_tensor_storage_access() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let mut tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Test immutable access through storage
            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice.len(), 4);

            // Test mutable access through storage
            let mut_slice = tensor.as_mut_slice().unwrap();
            mut_slice[0] = 10.0;

            // Verify mutation
            let updated = tensor.as_slice().unwrap();
            assert!((updated[0] - 10.0).abs() < 1e-5);
        }

        #[test]
        fn test_tensor_from_vec_cpu_storage() {
            let data = vec![5.0f32; 12];
            let shape = TensorShape::new(3, 2, 2);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Verify the storage is CpuStorage
            assert_eq!(tensor.storage.len(), 12);
            assert_eq!(tensor.storage.device(), DeviceType::Cpu);
        }

        #[test]
        fn test_tensor_slice_access_f32() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Should be able to access slice through existing API
            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice, &data[..]);
        }

        #[test]
        fn test_tensor_mut_slice_access_f32() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let mut tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Should be able to access mutable slice through existing API
            {
                let mut_slice = tensor.as_mut_slice().unwrap();
                for v in mut_slice.iter_mut() {
                    *v *= 2.0;
                }
            }

            let updated = tensor.as_slice().unwrap();
            assert!((updated[0] - 2.0).abs() < 1e-5);
            assert!((updated[1] - 4.0).abs() < 1e-5);
            assert!((updated[2] - 6.0).abs() < 1e-5);
        }

        #[test]
        fn test_tensor_reshape_generic_storage() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape1 = TensorShape::new(1, 2, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape1).unwrap();

            let shape2 = TensorShape::new(1, 3, 2);
            let reshaped = tensor.reshape(shape2).unwrap();

            assert_eq!(reshaped.shape, shape2);
            assert_eq!(reshaped.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn test_tensor_storage_device_type() {
            let data = vec![1.0f32; 24];
            let shape = TensorShape::new(2, 3, 4);
            let tensor: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(data, shape).unwrap();

            // Verify the storage is CPU-based
            assert_eq!(tensor.storage.device(), DeviceType::Cpu);
        }

        #[test]
        fn test_tensor_storage_capacity() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(data, shape).unwrap();

            // Verify storage capacity matches data length
            assert_eq!(tensor.storage.capacity(), 3);
            assert_eq!(tensor.storage.len(), 3);
        }

        #[test]
        fn test_tensor_with_storage_conversion() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let tensor1: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Create a new storage with same data
            let storage2: CpuStorage<f32> = CpuStorage::from_vec(data).unwrap();

            // Use with_storage to change storage
            let tensor2 = tensor1.with_storage(storage2);

            // Should have new storage but same shape and data
            assert_eq!(tensor2.shape, shape);
            assert_eq!(tensor2.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    mod cpu_convenience_api_tests {
        use super::*;

        #[test]
        fn test_cpu_as_slice() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data.clone(), shape).unwrap();

            let slice = tensor.cpu_as_slice();
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_cpu_as_mut_slice() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let mut tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data, shape).unwrap();

            let slice = tensor.cpu_as_mut_slice();
            slice[0] = 42.0;
            slice[1] = 99.0;

            assert_eq!(tensor.cpu_as_slice()[0], 42.0);
            assert_eq!(tensor.cpu_as_slice()[1], 99.0);
        }

        #[test]
        fn test_cpu_to_vec() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let shape = TensorShape::new(1, 1, 5);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data.clone(), shape).unwrap();

            let vec = tensor.cpu_to_vec();
            assert_eq!(vec, data);

            // Verify it's a clone, not the same reference
            assert_eq!(vec, tensor.cpu_as_slice());
        }

        #[test]
        fn test_cpu_to_vec_independence() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data, shape).unwrap();

            let mut vec = tensor.cpu_to_vec();
            vec[0] = 999.0;

            // Modifying the vec should not affect the tensor
            assert_ne!(vec[0], tensor.cpu_as_slice()[0]);
            assert_eq!(tensor.cpu_as_slice()[0], 1.0);
        }

        #[test]
        fn test_cpu_convenience_with_different_types() {
            let data_i32 = vec![1i32, 2, 3, 4];
            let shape = TensorShape::new(2, 2, 1);
            let tensor = Tensor::<i32, CpuStorage<i32>>::from_vec(data_i32.clone(), shape).unwrap();

            assert_eq!(tensor.cpu_as_slice(), &[1, 2, 3, 4]);
            assert_eq!(tensor.cpu_to_vec(), data_i32);
        }
    }
}
