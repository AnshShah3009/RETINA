use crate::BufferHandle;
use std::any::Any;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumeration of supported compute devices.
pub enum DeviceType {
    /// CPU (system RAM)
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Vulkan GPU
    Vulkan,
    /// Apple Metal GPU
    Metal,
    /// Windows DirectML GPU
    Dml,
    /// NVIDIA TensorRT
    TensorRT,
}

/// A unified trait for storage backends across different compute devices.
///
/// This trait defines the interface for tensor storage, allowing different
/// backends (CPU, GPU, etc.) to provide a consistent abstraction while
/// optimizing for their specific characteristics.
///
/// # Required Methods
///
/// - `handle()`: Get a unique identifier for this storage
/// - `capacity()`: Total number of elements this storage can hold
/// - `shape()`: Multi-dimensional shape information (if applicable)
/// - `len()`: Number of elements currently stored
/// - `is_empty()`: Convenience method for `len() == 0`
/// - `data_type_name()`: String representation of the stored type
/// - `as_any()`: Downcasting support for accessing concrete types
pub trait Storage<T: 'static>: Debug + Clone {
    /// Returns the unique handle for this storage.
    fn handle(&self) -> BufferHandle;

    /// Returns the total capacity (maximum number of elements).
    fn capacity(&self) -> usize;

    /// Returns the multi-dimensional shape of the data, if applicable.
    ///
    /// For a flat buffer, this returns an empty slice.
    /// For multi-dimensional data, returns shape dimensions in order.
    fn shape(&self) -> &[usize];

    /// Returns the number of elements currently stored.
    fn len(&self) -> usize;

    /// Returns `true` if this storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a human-readable name of the stored data type.
    fn data_type_name(&self) -> &'static str;

    /// Returns a reference to this storage as a dynamic `Any` for downcasting.
    ///
    /// This enables downcasting to the concrete storage type when needed.
    fn as_any(&self) -> &dyn Any;

    /// (Backward compat) Returns the device type this storage resides on.
    /// Default implementation returns Cpu; override for other devices.
    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }

    /// (Backward compat) Get the data as a slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    /// Default: None. Override in CPU implementations.
    fn as_slice(&self) -> Option<&[T]> {
        None
    }

    /// (Backward compat) Get the data as a mutable slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    /// Default: None. Override in CPU implementations.
    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }
}

/// Marker trait for CPU-based storage implementations.
///
/// Types implementing this marker indicate that data is stored in system RAM
/// and can be accessed directly via shared memory patterns.
pub trait CpuStorageMarker {
    /// The element type stored in this CPU storage.
    type Element;

    /// Convert to immutable slice for direct memory access.
    fn as_slice(&self) -> &[Self::Element];

    /// Convert to mutable slice for direct memory access.
    fn as_mut_slice(&mut self) -> &mut [Self::Element];

    /// Create a new vector by cloning the stored data.
    fn to_vec_cpu(&self) -> Vec<Self::Element>
    where
        Self::Element: Clone;
}

/// Marker trait for GPU-based storage implementations.
///
/// Types implementing this marker indicate that data is stored on a GPU
/// device and requires explicit transfer operations for CPU access.
pub trait GpuStorageMarker {
    /// The element type stored in this GPU storage.
    type Element;

    /// Transfer data from GPU to CPU.
    fn to_cpu(&self) -> std::result::Result<Vec<Self::Element>, String>
    where
        Self::Element: Clone;
}

/// Factory trait for creating storage instances.
///
/// This trait provides backward-compatible factory methods for creating
/// storage instances from data without explicitly calling constructor.
pub trait StorageFactory<T: 'static>: Storage<T> + Sized {
    /// Create storage from a vector of data.
    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String>;

    /// Create storage with a given size and default value.
    fn new(size: usize, default_value: T) -> std::result::Result<Self, String>;
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counter for generating unique BufferHandle IDs.
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique BufferHandle ID.
#[inline]
fn generate_handle_id() -> u64 {
    NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst)
}

/// Standard CPU-based storage using `Vec<T>` directly.
///
/// CpuStorage provides direct memory access to stored data on the CPU with
/// handle-based identification and multi-dimensional shape support.
#[derive(Debug, Clone)]
pub struct CpuStorage<T> {
    /// Unique handle for this storage instance
    handle: BufferHandle,
    /// Shape information (empty for flat buffers)
    shape: Vec<usize>,
    /// The actual data
    data: Vec<T>,
}

impl<T: Clone + Debug + 'static> Storage<T> for CpuStorage<T> {
    fn handle(&self) -> BufferHandle {
        self.handle
    }

    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn data_type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn as_slice(&self) -> Option<&[T]> {
        Some(&self.data)
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }
}

impl<T: Clone + Debug + 'static> CpuStorage<T> {
    /// Create a new CpuStorage with default-initialized data.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape dimensions (empty for flat buffer)
    ///
    /// # Returns
    ///
    /// A Result containing the new CpuStorage or an error string if T cannot be default-constructed.
    pub fn new_with_shape(shape: Vec<usize>) -> std::result::Result<Self, String>
    where
        T: Default,
    {
        let capacity = shape.iter().product::<usize>();
        let data = (0..capacity).map(|_| T::default()).collect();

        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape,
            data,
        })
    }

    /// (Backward compat) Create a new CpuStorage with size and default value.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements
    /// * `default_value` - Value to initialize all elements with
    pub fn new(size: usize, default_value: T) -> std::result::Result<Self, String> {
        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape: vec![],
            data: vec![default_value; size],
        })
    }

    /// (Backward compat) Create CpuStorage from an existing vector.
    /// Uses empty shape for backward compatibility.
    ///
    /// # Arguments
    ///
    /// * `data` - The vector containing the actual data
    pub fn from_vec(data: Vec<T>) -> std::result::Result<Self, String> {
        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape: vec![],
            data,
        })
    }

    /// Create CpuStorage from an existing vector with shape validation.
    ///
    /// # Arguments
    ///
    /// * `data` - The vector containing the actual data
    /// * `shape` - Shape dimensions (must match data length)
    ///
    /// # Returns
    ///
    /// A Result containing the new CpuStorage or an error if shape doesn't match data length.
    pub fn from_vec_with_shape(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> std::result::Result<Self, String> {
        let expected_len = shape.iter().product::<usize>();
        let actual_len = data.len();

        if actual_len != expected_len {
            return Err(format!(
                "Shape mismatch: shape {:?} expects {} elements, but data has {}",
                shape, expected_len, actual_len
            ));
        }

        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape,
            data,
        })
    }

    /// Convert storage to a vector by cloning the data.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T: Clone + Debug + 'static> CpuStorageMarker for CpuStorage<T> {
    type Element = T;

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn to_vec_cpu(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T: Clone + Debug + 'static> StorageFactory<T> for CpuStorage<T> {
    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String> {
        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape: vec![],
            data,
        })
    }

    fn new(size: usize, default_value: T) -> std::result::Result<Self, String> {
        Ok(CpuStorage {
            handle: BufferHandle(generate_handle_id()),
            shape: vec![],
            data: vec![default_value; size],
        })
    }
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;

    // ===== Storage<T> Trait Tests =====

    #[test]
    fn test_storage_trait_len() {
        let storage = CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_storage_trait_is_empty() {
        let empty = CpuStorage::<f32>::from_vec(vec![]).unwrap();
        assert!(empty.is_empty());

        let nonempty = CpuStorage::from_vec(vec![1.0f32]).unwrap();
        assert!(!nonempty.is_empty());
    }

    #[test]
    fn test_storage_trait_data_type_name() {
        let storage_f32 = CpuStorage::from_vec(vec![1.0f32]).unwrap();
        let storage_i32 = CpuStorage::from_vec(vec![1i32]).unwrap();

        assert!(storage_f32.data_type_name().contains("f32"));
        assert!(storage_i32.data_type_name().contains("i32"));
    }

    #[test]
    fn test_storage_trait_downcast_to_any() {
        let storage = CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0]).unwrap();
        let any = storage.as_any();

        // Verify it can be downcast
        let downcasted = any.downcast_ref::<CpuStorage<f32>>();
        assert!(downcasted.is_some());

        let casted = downcasted.unwrap();
        assert_eq!(casted.len(), 3);
    }

    // ===== CpuStorage<T> Specific Tests =====

    #[test]
    fn test_cpu_storage_new_with_default() {
        let storage = CpuStorage::<f32>::new_with_shape(vec![3, 4]).unwrap();
        assert_eq!(storage.len(), 12); // 3 * 4
        assert_eq!(storage.shape(), &[3, 4]);
    }

    #[test]
    fn test_cpu_storage_from_vec_valid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = CpuStorage::from_vec_with_shape(data.clone(), vec![2, 2]).unwrap();

        assert_eq!(storage.len(), 4);
        assert_eq!(storage.shape(), &[2, 2]);
        assert_eq!(storage.to_vec(), data);
    }

    #[test]
    fn test_cpu_storage_from_vec_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = CpuStorage::from_vec_with_shape(data, vec![2, 2]); // expects 4 elements

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Shape mismatch"));
    }

    #[test]
    fn test_cpu_storage_handle_uniqueness() {
        let storage1 = CpuStorage::from_vec(vec![1.0f32]).unwrap();
        let storage2 = CpuStorage::from_vec(vec![2.0f32]).unwrap();

        assert_ne!(storage1.handle(), storage2.handle());
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let data = vec![1, 2, 3, 4, 5];
        let storage = CpuStorage::from_vec(data.clone()).unwrap();

        let result = storage.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_cpu_storage_clone_shares_data() {
        let storage1 = CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0]).unwrap();
        let storage2 = storage1.clone();

        // Both should have the same handle (they're clones)
        assert_eq!(storage1.handle(), storage2.handle());

        // Data should be equal
        assert_eq!(storage1.to_vec(), storage2.to_vec());
    }

    #[test]
    fn test_cpu_storage_capacity() {
        let storage = CpuStorage::from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(storage.capacity(), 5);
    }

    #[test]
    fn test_cpu_storage_empty_shape() {
        let storage = CpuStorage::from_vec(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(storage.shape(), &[]);
    }

    #[test]
    fn test_cpu_storage_multidimensional_shape() {
        let data = vec![1.0f32; 24];
        let storage = CpuStorage::from_vec_with_shape(data, vec![2, 3, 4]).unwrap();

        assert_eq!(storage.shape(), &[2, 3, 4]);
        assert_eq!(storage.len(), 24);
    }
}
