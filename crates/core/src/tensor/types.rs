use crate::storage::{CpuStorage, Storage};
use std::marker::PhantomData;

#[cfg(feature = "half-precision")]
use half;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    U8,
    U16,
    U32,
    I32,
    F32,
    F64,
    #[cfg(feature = "half-precision")]
    F16,
    #[cfg(feature = "half-precision")]
    BF16,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::F64 => 8,
            #[cfg(feature = "half-precision")]
            DataType::F16 | DataType::BF16 => 2,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            DataType::F32 | DataType::F64 => true,
            #[cfg(feature = "half-precision")]
            DataType::F16 | DataType::BF16 => true,
            _ => false,
        }
    }

    pub fn from_type<T: 'static>() -> crate::Result<DataType> {
        use std::any::TypeId;
        let t = TypeId::of::<T>();
        if t == TypeId::of::<u8>() {
            Ok(DataType::U8)
        } else if t == TypeId::of::<u16>() {
            Ok(DataType::U16)
        } else if t == TypeId::of::<u32>() {
            Ok(DataType::U32)
        } else if t == TypeId::of::<i32>() {
            Ok(DataType::I32)
        } else if t == TypeId::of::<f32>() {
            Ok(DataType::F32)
        } else if t == TypeId::of::<f64>() {
            Ok(DataType::F64)
        } else {
            #[cfg(feature = "half-precision")]
            {
                if t == TypeId::of::<half::f16>() {
                    return Ok(DataType::F16);
                } else if t == TypeId::of::<half::bf16>() {
                    return Ok(DataType::BF16);
                }
            }
            Err(crate::Error::InvalidInput(format!(
                "Unsupported tensor data type: {}",
                std::any::type_name::<T>()
            )))
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

    pub fn is_empty(&self) -> bool {
        self.channels == 0 || self.height == 0 || self.width == 0
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

pub type Tensor3f = Tensor<f32>;
pub type Tensor4f = Tensor<f64>;
