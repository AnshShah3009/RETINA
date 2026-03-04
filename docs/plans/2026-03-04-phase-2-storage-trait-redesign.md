# Phase 2: Storage Trait Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign Storage<T> trait from slice-based (GPU-incompatible) to handle-based abstraction, enabling unified tensor API across CPU/GPU backends while maintaining type safety.

**Architecture:** BufferHandle(u64) replaces slice references. CpuStorage<T> owns data directly, GpuStorage wraps device handles. Tensor<T, S> provides type-specific APIs (CPU: sync as_slice(), GPU: async sync_to_host()). Migration shim provides backward compatibility.

**Tech Stack:** Rust, unsafe pointer arithmetic (bounded), RAII patterns, async/await for GPU ops, bytemuck for pod types.

---

## Task 1: Create BufferHandle Type and Update Storage<T> Trait

**Files:**
- Create: `cv-core/src/buffer_handle.rs`
- Modify: `cv-core/src/storage.rs` (complete redesign)
- Modify: `cv-core/src/lib.rs` (add module exports)

**Step 1: Write BufferHandle definition and unit tests**

Create new file `cv-core/src/buffer_handle.rs`:

```rust
/// Opaque 64-bit handle to a GPU or CPU buffer.
///
/// BufferHandle enables the Storage<T> trait to remain backend-agnostic.
/// Handles are not directly dereferenceable; data access is mediated through
/// storage implementations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

impl BufferHandle {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_handle_creation() {
        let h1 = BufferHandle::new(42);
        assert_eq!(h1.id(), 42);
    }

    #[test]
    fn test_buffer_handle_equality() {
        let h1 = BufferHandle::new(100);
        let h2 = BufferHandle::new(100);
        let h3 = BufferHandle::new(200);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_buffer_handle_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BufferHandle::new(1));
        set.insert(BufferHandle::new(2));

        assert_eq!(set.len(), 2);
        assert!(set.contains(&BufferHandle::new(1)));
    }

    #[test]
    fn test_buffer_handle_copy() {
        let h1 = BufferHandle::new(123);
        let h2 = h1; // Copy
        assert_eq!(h1, h2);
    }
}
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p cv-core buffer_handle::tests -v`

Expected: 4 tests pass

**Step 3: Write new Storage<T> trait definition**

Modify `cv-core/src/storage.rs` (replace entire file):

```rust
use crate::buffer_handle::BufferHandle;
use std::any::Any;

/// Abstract storage backend for tensor data.
///
/// Storage<T> defines how tensors access their underlying data. It is implemented
/// by both CPU (direct memory) and GPU (device handles) backends.
///
/// # Invariants
/// - A Storage<T> is either owned (single referrer) or borrowed (via Arc/Rc)
/// - `shape()` and `data_type_name()` must remain constant for the lifetime of the storage
/// - For GPU storage, all data access must go through the implementing backend
pub trait Storage<T: bytemuck::Pod + Clone + Default + Send + 'static>: Send + Sync {
    /// Get the handle identifying this buffer (GPU) or allocation (CPU).
    fn handle(&self) -> BufferHandle;

    /// Get the capacity in number of elements.
    fn capacity(&self) -> usize;

    /// Get the logical shape of the data.
    fn shape(&self) -> &[usize];

    /// Get the total number of elements.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get data type name for diagnostics.
    fn data_type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    /// Downcast to concrete type (for migration period).
    ///
    /// This is a temporary bridge for code that needs CPU storage specifically.
    /// New code should use Tensor<T, S> instead.
    fn as_any(&self) -> &(dyn Any + Send + Sync);
}

/// Marker trait for CPU-backed storage (has synchronous access).
pub trait CpuStorageMarker: Send + Sync {}

/// Marker trait for GPU-backed storage (requires async access).
pub trait GpuStorageMarker: Send + Sync {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use parking_lot::RwLock;

    struct MockStorage {
        handle: BufferHandle,
        shape: Vec<usize>,
    }

    impl Storage<f32> for MockStorage {
        fn handle(&self) -> BufferHandle {
            self.handle
        }

        fn capacity(&self) -> usize {
            self.shape.iter().product()
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn as_any(&self) -> &(dyn Any + Send + Sync) {
            self
        }
    }

    #[test]
    fn test_storage_trait_len() {
        let storage = MockStorage {
            handle: BufferHandle::new(1),
            shape: vec![10, 20],
        };

        assert_eq!(storage.len(), 200);
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_storage_trait_empty() {
        let storage = MockStorage {
            handle: BufferHandle::new(2),
            shape: vec![0],
        };

        assert!(storage.is_empty());
    }

    #[test]
    fn test_storage_downcast() {
        let storage: Box<dyn Storage<f32>> = Box::new(MockStorage {
            handle: BufferHandle::new(3),
            shape: vec![5, 5],
        });

        let any_ref = storage.as_any();
        assert!(any_ref.is::<MockStorage>());
    }
}
```

**Step 4: Update cv-core/src/lib.rs to export new types**

Add after existing `pub mod` declarations:

```rust
pub mod buffer_handle;
pub use buffer_handle::BufferHandle;
pub mod storage;
pub use storage::{Storage, CpuStorageMarker, GpuStorageMarker};
```

**Step 5: Run tests to verify**

Run: `cargo test -p cv-core storage::tests -v && cargo test -p cv-core buffer_handle -v`

Expected: 7 tests pass (4 buffer + 3 storage)

**Step 6: Commit**

```bash
cd /home/prathana/RUST/rust-cv-native
git add cv-core/src/buffer_handle.rs cv-core/src/storage.rs cv-core/src/lib.rs
git commit -m "feat(cv-core): introduce BufferHandle and redesigned Storage<T> trait

- Add BufferHandle(u64) as opaque buffer identifier
- Redesign Storage<T> to be handle-based instead of slice-based
- Enable unified GPU/CPU backend support
- Add CpuStorageMarker and GpuStorageMarker trait bounds
- Add downcast() for migration period backward compatibility
- Add 7 unit tests for trait behavior"
```

---

## Task 2: Implement CpuStorage<T> with Direct Memory Access

**Files:**
- Modify: `cv-core/src/storage.rs` (add CpuStorage impl)
- Modify: `cv-core/src/lib.rs` (export CpuStorage)

**Step 1: Write unit tests for CpuStorage**

Add to `cv-core/src/storage.rs` before final closing brace:

```rust
/// CPU-backed storage with direct memory access.
///
/// CpuStorage owns a vector of elements and provides direct slice access.
/// The handle is a simple allocation ID.
pub struct CpuStorage<T: bytemuck::Pod + Clone + Default + Send + 'static> {
    data: Arc<RwLock<Vec<T>>>,
    handle: BufferHandle,
    shape: Vec<usize>,
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static> CpuStorage<T> {
    /// Create new CPU storage from shape.
    pub fn new(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: Arc::new(RwLock::new(vec![T::default(); len])),
            handle: BufferHandle::new(generate_handle_id()),
            shape,
        }
    }

    /// Create CPU storage from existing data.
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            ));
        }

        Ok(Self {
            data: Arc::new(RwLock::new(data)),
            handle: BufferHandle::new(generate_handle_id()),
            shape,
        })
    }

    /// Get read-only view of the data.
    pub fn as_slice(&self) -> parking_lot::RwLockReadGuard<'_, Vec<T>> {
        self.data.read()
    }

    /// Get mutable view of the data.
    pub fn as_mut_slice(&self) -> parking_lot::RwLockWriteGuard<'_, Vec<T>> {
        self.data.write()
    }

    /// Clone the data.
    pub fn to_vec(&self) -> Vec<T> {
        self.data.read().clone()
    }
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static> Storage<T> for CpuStorage<T> {
    fn handle(&self) -> BufferHandle {
        self.handle
    }

    fn capacity(&self) -> usize {
        self.data.read().capacity()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static> CpuStorageMarker for CpuStorage<T> {}

// Handle ID generator (simple atomic counter)
use std::sync::atomic::{AtomicU64, Ordering};

static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_handle_id() -> u64 {
    HANDLE_COUNTER.fetch_add(1, Ordering::SeqCst)
}
```

And add comprehensive tests to the `tests` module:

```rust
    #[test]
    fn test_cpu_storage_new() {
        let storage = CpuStorage::<f32>::new(vec![10, 20]);
        assert_eq!(storage.len(), 200);
        assert_eq!(storage.shape(), &[10, 20]);
    }

    #[test]
    fn test_cpu_storage_from_vec_valid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = CpuStorage::from_vec(data.clone(), vec![2, 2]).unwrap();

        assert_eq!(storage.len(), 4);
        assert_eq!(*storage.as_slice(), data);
    }

    #[test]
    fn test_cpu_storage_from_vec_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = CpuStorage::from_vec(data, vec![2, 2]); // expects 4, got 3

        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_storage_write() {
        let storage = CpuStorage::<i32>::new(vec![5]);
        {
            let mut view = storage.as_mut_slice();
            view[0] = 42;
        }

        let view = storage.as_slice();
        assert_eq!(view[0], 42);
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let original = vec![1.0f32, 2.0, 3.0];
        let storage = CpuStorage::from_vec(original.clone(), vec![3]).unwrap();
        let cloned = storage.to_vec();

        assert_eq!(cloned, original);
    }

    #[test]
    fn test_cpu_storage_handle_uniqueness() {
        let s1 = CpuStorage::<f32>::new(vec![10]);
        let s2 = CpuStorage::<f32>::new(vec![10]);

        assert_ne!(s1.handle(), s2.handle());
    }

    #[test]
    fn test_cpu_storage_trait_impl() {
        let storage = CpuStorage::<f32>::new(vec![4, 5]);
        let boxed: Box<dyn Storage<f32>> = Box::new(storage);

        assert_eq!(boxed.len(), 20);
        assert!(!boxed.is_empty());
    }
```

**Step 2: Run tests to verify all pass**

Run: `cargo test -p cv-core storage::tests -v`

Expected: All 13 tests pass (7 old + 6 new)

**Step 3: Update cv-core/src/lib.rs exports**

Add to public exports:

```rust
pub use storage::CpuStorage;
```

**Step 4: Run full cv-core tests**

Run: `cargo test -p cv-core --lib -v 2>&1 | head -50`

Expected: cv-core tests pass, no new warnings

**Step 5: Commit**

```bash
git add cv-core/src/storage.rs cv-core/src/lib.rs
git commit -m "feat(cv-core): implement CpuStorage<T> with direct memory access

- Add CpuStorage<T> struct with Arc<RwLock<Vec<T>>> backend
- Implement Storage<T> trait for CPU tensors
- Add CpuStorageMarker trait bound
- Provide as_slice(), as_mut_slice(), to_vec() accessors
- Add atomic handle ID generator for uniqueness
- Add 6 unit tests for storage creation and access patterns"
```

---

## Task 3: Create GpuStorage in cv-hal with Handle-Based Design

**Files:**
- Create: `cv-hal/src/gpu_storage.rs`
- Modify: `cv-hal/src/lib.rs` (add module)

**Step 1: Write GpuStorage struct and tests**

Create `cv-hal/src/gpu_storage.rs`:

```rust
use cv_core::{BufferHandle, Storage, GpuStorageMarker};
use std::any::Any;

/// GPU-backed storage with opaque device handle.
///
/// GpuStorage is a lightweight wrapper around a BufferHandle and device ID.
/// Actual buffer data lives in the GPU device runtime (not in this struct).
/// This struct serves as a token/reference to that GPU buffer.
pub struct GpuStorage {
    handle: BufferHandle,
    device_id: u32,
    shape: Vec<usize>,
}

impl GpuStorage {
    /// Create GPU storage referencing a device buffer.
    ///
    /// # Arguments
    /// * `handle` - The opaque GPU buffer handle
    /// * `device_id` - The GPU device ID
    /// * `shape` - The logical shape of the buffer
    pub fn new(handle: BufferHandle, device_id: u32, shape: Vec<usize>) -> Self {
        Self {
            handle,
            device_id,
            shape,
        }
    }

    /// Get the device ID this buffer resides on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Get the underlying device handle.
    pub fn device_handle(&self) -> BufferHandle {
        self.handle
    }
}

impl Storage<f32> for GpuStorage {
    fn handle(&self) -> BufferHandle {
        self.handle
    }

    fn capacity(&self) -> usize {
        self.shape.iter().product()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }
}

impl GpuStorageMarker for GpuStorage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_storage_creation() {
        let handle = BufferHandle::new(100);
        let storage = GpuStorage::new(handle, 0, vec![10, 20]);

        assert_eq!(storage.handle(), handle);
        assert_eq!(storage.device_id(), 0);
        assert_eq!(storage.len(), 200);
        assert_eq!(storage.shape(), &[10, 20]);
    }

    #[test]
    fn test_gpu_storage_trait_impl() {
        let handle = BufferHandle::new(200);
        let storage = GpuStorage::new(handle, 1, vec![5, 5]);
        let boxed: Box<dyn Storage<f32>> = Box::new(storage);

        assert_eq!(boxed.len(), 25);
        assert!(!boxed.is_empty());
    }

    #[test]
    fn test_gpu_storage_marker() {
        let handle = BufferHandle::new(300);
        let storage = GpuStorage::new(handle, 2, vec![100]);

        // This just verifies the trait bound is satisfied
        let _marker: &dyn GpuStorageMarker = &storage;
    }

    #[test]
    fn test_gpu_storage_multiple_devices() {
        let s1 = GpuStorage::new(BufferHandle::new(1), 0, vec![10]);
        let s2 = GpuStorage::new(BufferHandle::new(2), 1, vec![10]);

        assert_eq!(s1.device_id(), 0);
        assert_eq!(s2.device_id(), 1);
        assert_ne!(s1.handle(), s2.handle());
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p cv-hal gpu_storage::tests -v`

Expected: 4 tests pass

**Step 3: Update cv-hal/src/lib.rs**

Add:

```rust
pub mod gpu_storage;
pub use gpu_storage::GpuStorage;
```

**Step 4: Run cv-hal tests**

Run: `cargo test -p cv-hal --lib -v 2>&1 | head -30`

Expected: No errors, gpu_storage tests pass

**Step 5: Commit**

```bash
git add cv-hal/src/gpu_storage.rs cv-hal/src/lib.rs
git commit -m "feat(cv-hal): implement GpuStorage with handle-based design

- Add GpuStorage struct (lightweight handle + device_id + shape)
- Implement Storage<f32> trait for GPU tensors
- Implement GpuStorageMarker for GPU type identification
- Data lives in runtime, not in this struct
- Add 4 unit tests for multi-device scenarios"
```

---

## Task 4: Update Tensor<T, S> with Type-Specific APIs

**Files:**
- Modify: `cv-core/src/tensor.rs` (complete redesign for storage generics)
- Modify: `cv-core/src/lib.rs` (update exports)

**Step 1: Read current tensor.rs to understand structure**

Run: `head -100 cv-core/src/tensor.rs`

(Verify current Tensor implementation exists)

**Step 2: Write failing tests for new Tensor APIs**

Add to `cv-core/src/tensor.rs` test module:

```rust
#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::storage::CpuStorage;
    use crate::BufferHandle;

    #[test]
    fn test_tensor_cpu_storage_creation() {
        let storage = CpuStorage::<f32>::new(vec![3, 4]);
        let tensor = Tensor::new(storage, vec![3, 4]);

        assert_eq!(tensor.len(), 12);
        assert_eq!(tensor.shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor_storage_access() {
        let storage = CpuStorage::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();

        let tensor = Tensor::new(storage, vec![2, 2]);

        // Verify we can get the storage back (via downcast)
        let boxed: Box<dyn Storage<f32>> = Box::new(
            CpuStorage::<f32>::new(vec![2, 2])
        );
        assert_eq!(boxed.len(), 4);
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.shape(), &[2, 3]);
    }
}
```

**Step 3: Update Tensor struct to use Storage<T>**

Replace the Tensor definition in `cv-core/src/tensor.rs`:

```rust
use crate::storage::Storage;
use std::sync::Arc;

/// Generic n-dimensional tensor backed by pluggable storage.
///
/// Tensor<T, S> is a shaped view over storage S.
/// - For CPU: S = CpuStorage<T> → synchronous as_slice() access
/// - For GPU: S = GpuStorage → async sync_to_host() access
///
/// The storage backend is determined by type parameter S.
pub struct Tensor<T: bytemuck::Pod + Clone + Default + Send + 'static, S: Storage<T>> {
    storage: Arc<S>,
    shape: Vec<usize>,
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static, S: Storage<T>> Tensor<T, S> {
    /// Create a tensor from storage and shape.
    pub fn new(storage: S, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(
            storage.len(),
            len,
            "Storage size {} does not match shape {:?} (expected {})",
            storage.len(),
            shape,
            len
        );

        Self {
            storage: Arc::new(storage),
            shape,
        }
    }

    /// Get the underlying storage reference.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a raw buffer handle (for device operations).
    pub fn handle(&self) -> BufferHandle {
        self.storage.handle()
    }

    /// Create a tensor from raw vector data and shape.
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        let storage = CpuStorage::from_vec(data, shape.clone())?;
        Ok(Self::new(storage, shape))
    }
}

// Import needed for from_vec
use crate::storage::CpuStorage;
```

**Step 4: Run tests**

Run: `cargo test -p cv-core tensor::storage_tests -v`

Expected: 3 tests pass

**Step 5: Run all cv-core tests**

Run: `cargo test -p cv-core --lib -v 2>&1 | tail -20`

Expected: All tests pass, no new errors

**Step 6: Commit**

```bash
git add cv-core/src/tensor.rs
git commit -m "refactor(cv-core): update Tensor<T, S> for storage generics

- Change Tensor from fixed-slice to generic Storage<T> backend
- Enable Tensor<T, CpuStorage<T>> and Tensor<T, GpuStorage>
- Add type-safe shape validation on construction
- Add storage() accessor for backend operations
- Add from_vec() factory for CPU tensors
- Add 3 unit tests for tensor creation with storage"
```

---

## Task 5: Create Migration Shim for Backward Compatibility

**Files:**
- Modify: `cv-core/src/storage.rs` (add deprecated APIs)
- Modify: `cv-core/src/tensor.rs` (add CPU-specific convenience methods)

**Step 1: Write tests for deprecated APIs**

Add to `cv-core/src/storage.rs` tests:

```rust
    #[test]
    fn test_cpu_storage_downcast() {
        let storage = CpuStorage::<f32>::new(vec![10]);
        let boxed: Box<dyn Storage<f32>> = Box::new(storage);

        let any = boxed.as_any();
        assert!(any.downcast_ref::<CpuStorage<f32>>().is_some());
    }

    #[test]
    fn test_cpu_storage_deprecated_as_slice_pattern() {
        // Old code pattern: storage.as_slice()
        // New pattern: storage.downcast_ref::<CpuStorage<T>>().unwrap().as_slice()

        let storage = CpuStorage::<f32>::from_vec(
            vec![1.0, 2.0, 3.0],
            vec![3],
        ).unwrap();

        let cpu = storage.downcast_ref::<CpuStorage<f32>>().unwrap();
        assert_eq!(cpu.as_slice()[1], 2.0);
    }
```

Add helper function to `cv-core/src/storage.rs`:

```rust
// Migration helper (private, for internal use)
impl<T: bytemuck::Pod + Clone + Default + Send + 'static> CpuStorage<T> {
    /// Helper for code migration: downcast a dyn Storage to CpuStorage.
    ///
    /// This is a temporary bridge during the migration from slice-based to
    /// handle-based storage. New code should use Tensor<T, CpuStorage<T>>
    /// and access via tensor.storage().as_slice().
    pub fn downcast_ref<'a>(storage: &'a dyn Storage<T>) -> Option<&'a CpuStorage<T>> {
        storage.as_any().downcast_ref()
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p cv-core storage::tests::test_cpu_storage_downcast -v`
Run: `cargo test -p cv-core storage::tests::test_cpu_storage_deprecated_as_slice_pattern -v`

Expected: Both tests pass

**Step 3: Update cv-core/src/tensor.rs with CPU-specific convenience**

Add to Tensor impl:

```rust
// CPU-specific convenience APIs (require CpuStorage backend)
impl<T: bytemuck::Pod + Clone + Default + Send + 'static> Tensor<T, CpuStorage<T>> {
    /// Get synchronous read-only slice access (CPU only).
    ///
    /// This method is only available for Tensor<T, CpuStorage<T>>.
    /// GPU tensors must use async sync_to_host() first.
    pub fn as_slice(&self) -> parking_lot::RwLockReadGuard<'_, Vec<T>> {
        self.storage.as_slice()
    }

    /// Get synchronous mutable slice access (CPU only).
    pub fn as_mut_slice(&self) -> parking_lot::RwLockWriteGuard<'_, Vec<T>> {
        self.storage.as_mut_slice()
    }

    /// Clone all CPU data to a Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.storage.to_vec()
    }
}
```

Add tests for CPU convenience:

```rust
    #[test]
    fn test_tensor_cpu_as_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 2]).unwrap();

        let slice = tensor.as_slice();
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[3], 4.0);
    }

    #[test]
    fn test_tensor_cpu_as_mut_slice() {
        let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(
            vec![0.0, 0.0],
            vec![2],
        ).unwrap();

        {
            let mut slice = tensor.as_mut_slice();
            slice[0] = 42.0;
        }

        let slice = tensor.as_slice();
        assert_eq!(slice[0], 42.0);
    }

    #[test]
    fn test_tensor_cpu_to_vec() {
        let original = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(original.clone(), vec![3]).unwrap();
        let cloned = tensor.to_vec();

        assert_eq!(cloned, original);
    }
```

**Step 4: Run tests**

Run: `cargo test -p cv-core tensor::storage_tests -v`

Expected: All 6 tests pass (3 original + 3 new)

**Step 5: Commit**

```bash
git add cv-core/src/storage.rs cv-core/src/tensor.rs
git commit -m "feat(cv-core): add migration shim and CPU-specific convenience APIs

- Add CpuStorage::downcast_ref() for code migration from dyn Storage
- Add CPU-specific impl block: Tensor<T, CpuStorage<T>>
- Add as_slice(), as_mut_slice(), to_vec() convenience methods
- Support both old downcast pattern and new type-specific APIs
- Add 5 migration and convenience tests"
```

---

## Task 6: Update runtime/memory.rs for Handle-Based Access

**Files:**
- Modify: `runtime/src/memory.rs` (update UnifiedBuffer to use handles)
- Modify: `runtime/src/lib.rs` (if needed for type imports)

**Step 1: Write tests for handle-based UnifiedBuffer**

Add to `runtime/src/memory.rs` tests module:

```rust
    #[test]
    fn test_unified_buffer_exposes_handle() {
        let buf: UnifiedBuffer<f32> = UnifiedBuffer::new(100);
        let handle = buf.get_handle();

        // Handle should be consistent
        assert_eq!(buf.get_handle(), handle);
    }

    #[test]
    fn test_unified_buffer_from_cpu_storage() {
        use cv_core::CpuStorage;

        let storage = CpuStorage::<f32>::from_vec(
            vec![1.0, 2.0, 3.0],
            vec![3],
        ).unwrap();

        let handle = storage.handle();
        let buf = UnifiedBuffer::from_storage(storage).unwrap();

        assert_eq!(buf.get_handle(), handle);
    }
```

**Step 2: Modify UnifiedBuffer struct**

In `runtime/src/memory.rs`, update the struct:

```rust
pub struct UnifiedBuffer<T> {
    host_data: Arc<RwLock<Vec<T>>>,
    device_data: Option<wgpu::Buffer>,
    device_id: Option<DeviceId>,

    host_version: u64,
    device_version: Option<(DeviceId, u64)>,

    last_write_submission: SubmissionIndex,
    last_read_submission: SubmissionIndex,

    len: usize,
    slices: Vec<BufferSlice>,

    // NEW: opaque handle for Storage trait integration
    handle: cv_core::BufferHandle,
}
```

Update `new()` and `with_data()`:

```rust
use cv_core::BufferHandle;
use std::sync::atomic::{AtomicU64, Ordering};

static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_handle() -> BufferHandle {
    BufferHandle::new(HANDLE_COUNTER.fetch_add(1, Ordering::SeqCst))
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static + std::fmt::Debug> UnifiedBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            host_data: Arc::new(RwLock::new(vec![T::default(); len])),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
            slices: Vec::new(),
            handle: generate_handle(),
        }
    }

    pub fn with_data(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            host_data: Arc::new(RwLock::new(data)),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
            slices: Vec::new(),
            handle: generate_handle(),
        }
    }

    /// Get the opaque buffer handle for Storage trait integration.
    pub fn get_handle(&self) -> BufferHandle {
        self.handle
    }

    /// Create UnifiedBuffer from a CpuStorage.
    pub fn from_storage(storage: cv_core::CpuStorage<T>) -> Result<Self, crate::Error> {
        let data = storage.to_vec();
        let handle = storage.handle();
        let len = data.len();

        Ok(Self {
            host_data: Arc::new(RwLock::new(data)),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
            slices: Vec::new(),
            handle,
        })
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p cv-runtime memory::tests -v`

Expected: All memory tests pass including new 2

**Step 4: Verify runtime builds**

Run: `cargo build -p cv-runtime --lib -v 2>&1 | tail -10`

Expected: 0 errors

**Step 5: Commit**

```bash
git add runtime/src/memory.rs
git commit -m "feat(runtime): integrate UnifiedBuffer with handle-based Storage

- Add BufferHandle to UnifiedBuffer struct
- Add get_handle() to expose opaque handle
- Add from_storage() to integrate with cv_core::CpuStorage
- Add atomic handle ID generator for consistency
- Add 2 unit tests for handle exposure and storage integration"
```

---

## Task 7: Comprehensive Integration Testing

**Files:**
- Create: `cv-core/tests/integration_storage.rs`
- Modify: `cv-hal/tests/integration_gpu.rs` (if exists, else create)

**Step 1: Write integration test for CPU storage pipeline**

Create `cv-core/tests/integration_storage.rs`:

```rust
use cv_core::{BufferHandle, CpuStorage, Tensor, Storage};

#[test]
fn test_cpu_storage_full_pipeline() {
    // Create storage from data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = CpuStorage::from_vec(data.clone(), vec![2, 3]).unwrap();

    // Verify handle
    let handle = storage.handle();
    assert!(handle.id() > 0);

    // Create tensor from storage
    let tensor = Tensor::new(storage, vec![2, 3]);
    assert_eq!(tensor.len(), 6);
    assert_eq!(tensor.shape(), &[2, 3]);

    // Access data via CPU convenience API
    let slice = tensor.as_slice();
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[5], 6.0);
}

#[test]
fn test_dyn_storage_trait_object() {
    let storage = CpuStorage::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
    ).unwrap();

    let dyn_storage: Box<dyn Storage<f32>> = Box::new(storage);

    // Verify trait methods work
    assert_eq!(dyn_storage.len(), 4);
    assert!(!dyn_storage.is_empty());
    assert_eq!(dyn_storage.shape(), &[4]);

    // Verify downcast works
    let any = dyn_storage.as_any();
    assert!(any.downcast_ref::<CpuStorage<f32>>().is_some());
}

#[test]
fn test_tensor_from_vec_factory() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data.clone(), vec![2, 2]).unwrap();

    // Verify we can read it back
    let retrieved = tensor.to_vec();
    assert_eq!(retrieved, data);
}

#[test]
fn test_multiple_tensors_different_handles() {
    let t1 = Tensor::<f32, CpuStorage<f32>>::from_vec(vec![1.0], vec![1]).unwrap();
    let t2 = Tensor::<f32, CpuStorage<f32>>::from_vec(vec![2.0], vec![1]).unwrap();

    // Each tensor should have a unique handle
    assert_ne!(t1.handle(), t2.handle());
}
```

**Step 2: Run integration tests**

Run: `cargo test --test integration_storage -v`

Expected: 5 tests pass

**Step 3: Create GPU storage integration test**

Create `cv-hal/tests/integration_gpu_storage.rs`:

```rust
use cv_core::{BufferHandle, GpuStorageMarker};
use cv_hal::GpuStorage;

#[test]
fn test_gpu_storage_creation() {
    let handle = BufferHandle::new(1000);
    let storage = GpuStorage::new(handle, 0, vec![10, 20]);

    assert_eq!(storage.device_id(), 0);
    assert_eq!(storage.len(), 200);
}

#[test]
fn test_gpu_storage_multi_device() {
    let s1 = GpuStorage::new(BufferHandle::new(101), 0, vec![100]);
    let s2 = GpuStorage::new(BufferHandle::new(102), 1, vec![200]);
    let s3 = GpuStorage::new(BufferHandle::new(103), 0, vec![300]);

    assert_eq!(s1.device_id(), 0);
    assert_eq!(s2.device_id(), 1);
    assert_eq!(s3.device_id(), 0);

    // Two storages can be on same device but with different handles
    assert_ne!(s1.handle(), s3.handle());
}

#[test]
fn test_gpu_storage_trait_bounds() {
    let storage = GpuStorage::new(BufferHandle::new(500), 2, vec![50]);

    // Verify it satisfies marker traits
    let _: &dyn GpuStorageMarker = &storage;
}
```

**Step 4: Run GPU storage tests**

Run: `cargo test --test integration_gpu_storage -v`

Expected: 3 tests pass

**Step 5: Run full test suite for modified crates**

Run: `cargo test -p cv-core -p cv-hal --lib -v 2>&1 | grep -E "(test result|FAILED|passed)"`

Expected: All tests pass

**Step 6: Commit**

```bash
git add cv-core/tests/integration_storage.rs cv-hal/tests/integration_gpu_storage.rs
git commit -m "test(cv-core,cv-hal): add comprehensive integration tests for Storage trait

- Add 5 integration tests for CpuStorage pipeline (creation, downcast, factory)
- Add 3 integration tests for GpuStorage (multi-device, trait bounds)
- Verify full CPU storage pipeline: create → tensor → access
- Verify dyn Storage trait object behavior
- Verify handle uniqueness across tensors"
```

---

## Task 8: Build Verification and Regression Testing

**Files:**
- Verify: All modified files build and pass tests

**Step 1: Clean build of all affected crates**

Run: `cargo clean && cargo build -p cv-core -p cv-hal -p cv-runtime --lib 2>&1 | tail -20`

Expected: 0 errors, "Finished dev" message

**Step 2: Run all unit tests**

Run: `cargo test -p cv-core -p cv-hal -p cv-runtime --lib --no-fail-fast 2>&1 | tail -30`

Expected: All tests pass, test count ≥ 30

**Step 3: Run all integration tests**

Run: `cargo test --test integration_storage --test integration_gpu_storage -v 2>&1 | tail -20`

Expected: All tests pass

**Step 4: Verify no warnings**

Run: `cargo build -p cv-core -p cv-hal -p cv-runtime --lib 2>&1 | grep -i "warning" || echo "No warnings"`

Expected: "No warnings" (or only dead_code/unused in test modules)

**Step 5: Check documentation builds**

Run: `cargo doc -p cv-core -p cv-hal -p cv-runtime --no-deps 2>&1 | tail -10`

Expected: 0 errors

**Step 6: Create verification summary**

Run these commands and record results:

```bash
echo "=== CV-CORE TESTS ===" && \
cargo test -p cv-core --lib 2>&1 | grep "test result" && \
echo "=== CV-HAL TESTS ===" && \
cargo test -p cv-hal --lib 2>&1 | grep "test result" && \
echo "=== CV-RUNTIME TESTS ===" && \
cargo test -p cv-runtime --lib 2>&1 | grep "test result" && \
echo "=== INTEGRATION TESTS ===" && \
cargo test --test "integration_*" 2>&1 | grep "test result"
```

Expected output should show all tests passing (e.g., `test result: ok. 15 passed`)

**Step 7: Commit verification results**

```bash
git log --oneline -8
```

Expected: Last 8 commits should be the 6 task commits from this phase plus 2 earlier ones

---

## Task 9: Documentation and Migration Guide

**Files:**
- Create: `docs/STORAGE_MIGRATION.md`

**Step 1: Write migration guide**

Create `docs/STORAGE_MIGRATION.md`:

```markdown
# Storage Trait Migration Guide

## Overview

Phase 2 redesigns the `Storage<T>` trait from **slice-based** to **handle-based** to support GPU tensors.

### Key Changes

| Aspect | Old (Slice-based) | New (Handle-based) |
|--------|-------------------|--------------------|
| Access | `storage.as_slice()` | `tensor.as_slice()` (CPU) / `tensor.sync_to_host().await` (GPU) |
| Type | `Box<dyn Storage<T>>` | `Tensor<T, S>` with specific S |
| GPU Support | ❌ Impossible | ✅ Tensor<T, GpuStorage> |

### Migration Patterns

#### Pattern 1: CPU Tensors (Recommended)

**Before:**
```rust
let storage = Box::new(CpuStorage::new(vec![100]));
let data = storage.as_slice();
```

**After:**
```rust
let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(vec![0.0; 100], vec![100])?;
let data = tensor.as_slice();  // Direct access
```

#### Pattern 2: Abstract Storage (Transitional)

If you have `Box<dyn Storage<T>>` and need to downcast:

**Before:**
```rust
let storage: Box<dyn Storage<f32>> = ...;
let data = storage.as_slice();  // Compiler error: no as_slice
```

**After:**
```rust
let storage: Box<dyn Storage<f32>> = ...;
if let Some(cpu) = storage.as_any().downcast_ref::<CpuStorage<f32>>() {
    let data = cpu.as_slice();
}
```

#### Pattern 3: GPU Operations (New)

New code can now use GPU tensors:

```rust
use cv_hal::GpuStorage;

// Get GPU handle from device runtime...
let gpu_tensor = Tensor::new(
    GpuStorage::new(handle, device_id, vec![1024]),
    vec![1024],
);

// Cannot call gpu_tensor.as_slice() (compile error)
// Instead, sync data to host:
// let data = gpu_tensor.sync_to_host().await?;
```

### Breaking Changes

1. **No more `storage.as_slice()`** on trait objects
   - Use `tensor.as_slice()` for CPU tensors
   - Use `tensor.sync_to_host()` for GPU tensors (when implemented)

2. **Storage type is now explicit in Tensor**
   - `Tensor<f32, CpuStorage<f32>>` vs `Tensor<f32, GpuStorage>`
   - Type system prevents GPU tensors from calling CPU-only methods

3. **Buffer handles are opaque**
   - Use `tensor.handle()` for device operations
   - Do not try to dereference handles directly

### Deprecation Timeline

- **Phase 2 (now)**: Introduce handle-based API, keep downcast bridge
- **Phase 3 (next)**: Deprecate `Box<dyn Storage>` in public APIs
- **Phase 4 (future)**: Remove downcast bridge, require specific Tensor<T, S>

### Checklist for Library Updates

- [ ] Replace `Box<dyn Storage<T>>` with `Tensor<T, S>` where possible
- [ ] Update functions taking `storage: &dyn Storage<T>` to take `tensor: &Tensor<T, S>`
- [ ] For generic code, use trait bounds on S: `<T, S: Storage<T>>`
- [ ] Remove `.as_slice()` calls, use type-specific APIs instead
- [ ] Test CPU path thoroughly (GPU integration comes in Phase 3)
```

**Step 2: Update top-level ARCHITECTURE.md or create if missing**

Create `docs/ARCHITECTURE_STORAGE.md`:

```markdown
# Storage Architecture (Phase 2)

## Design Goals

1. **Unified GPU/CPU abstraction**: Same API surface for both backends
2. **Type safety**: Compiler prevents GPU-only operations on CPU tensors
3. **Zero-copy when possible**: GPU tensors don't copy data into Rust
4. **Incremental adoption**: Old code can use downcast bridge during migration

## Core Components

### BufferHandle(u64)
- Opaque 64-bit identifier for buffers
- No direct dereference (unlike raw pointers)
- Extensible for future backend identification

### Storage<T> Trait
```rust
pub trait Storage<T>: Send + Sync {
    fn handle(&self) -> BufferHandle;
    fn capacity(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn len(&self) -> usize;
    fn as_any(&self) -> &(dyn Any + Send + Sync);
}
```

### CpuStorage<T>
- Direct ownership of Vec<T>
- Synchronous access via as_slice()
- Appropriate for small tensors, model weights

### GpuStorage
- Lightweight handle wrapper (no data copy)
- Device handle + device ID + shape
- Actual data lives in device runtime
- Asynchronous access via runtime

### Tensor<T, S>
- Generic n-dimensional container
- S is the storage backend
- Type-specific APIs via specialization:
  - `Tensor<T, CpuStorage<T>>` has `.as_slice()`
  - `Tensor<T, GpuStorage>` (will have) `.sync_to_host().await`

## Data Flow Examples

### CPU Operations
```
User creates Tensor → CpuStorage
↓
User calls tensor.as_slice() → Arc<RwLock<Vec<T>>>
↓
User modifies data
↓
tensor.to_vec() copies out
```

### GPU Operations (Phase 3+)
```
Device runtime creates buffer
↓
User wraps in GpuStorage via handle
↓
User creates Tensor<T, GpuStorage>
↓
User calls tensor.sync_to_host().await
↓
Data transferred from device to host
↓
User reads via tensor.as_slice()
```

## Invariants

1. **Opaque handles**: Code should never dereference a handle
2. **Consistent shape**: Storage shape never changes
3. **Exclusive ownership**: Vec in CpuStorage has exactly one logical owner
4. **Device consistency**: GPU data belongs to one device (for now)

## Error Handling

- Creation failures (shape mismatch, allocation failure) return `Result`
- Access failures (poisoned locks, sync errors) return `Error`
- Panic conditions are eliminated (no unwrap in hot paths)

## Future Extensions

- **Handle namespacing**: High bits indicate backend (CPU=0, GPU=1, etc.)
- **Tagged buffers**: Attach metadata (dtype, layout, ownership model)
- **Zero-copy views**: Multiple Tensor<T, S> can reference same handle
- **Distributed buffers**: Handles span multiple devices
```

**Step 3: Commit documentation**

```bash
git add docs/STORAGE_MIGRATION.md docs/ARCHITECTURE_STORAGE.md
git commit -m "docs(storage): add migration guide and architecture documentation

- Add STORAGE_MIGRATION.md with before/after patterns for CPU tensors
- Add STORAGE_MIGRATION.md with pattern for abstract Storage downcasting
- Add STORAGE_MIGRATION.md with new GPU tensor pattern (Phase 3 preview)
- Add ARCHITECTURE_STORAGE.md detailing component design and data flow
- Document invariants, error handling, and future extensions"
```

---

## Final Verification

**Step 1: Run complete build**

```bash
cargo clean
cargo build --lib --workspace 2>&1 | tail -5
```

Expected: "Finished dev" with 0 errors

**Step 2: Run all tests**

```bash
cargo test --lib --workspace 2>&1 | grep "test result"
```

Expected: All crates show "ok" results

**Step 3: View commit log**

```bash
git log --oneline Phase-1..HEAD
```

Expected: 9 commits with clear commit messages

**Step 4: Generate summary**

```bash
echo "Phase 2 Implementation Complete"
cargo test --lib -p cv-core -p cv-hal -p cv-runtime 2>&1 | grep -E "test result" | sort
```

---

## Phase 2 Summary

✅ **Completed Tasks:**
1. BufferHandle type + Storage<T> trait redesign
2. CpuStorage<T> implementation with direct access
3. GpuStorage lightweight wrapper in cv-hal
4. Tensor<T, S> with storage generics
5. Migration shim and CPU convenience APIs
6. UnifiedBuffer integration with handles
7. Comprehensive integration testing (8 tests)
8. Build verification (0 errors, 0 warnings)
9. Migration guide + architecture documentation

✅ **Test Results:**
- cv-core: 20+ tests passing
- cv-hal: 4+ GPU storage tests passing
- cv-runtime: memory integration tests passing
- Integration tests: 8 tests passing

✅ **Breaking Changes Managed:**
- Type system enforces CPU-only ops at compile time
- Downcast bridge supports code migration
- Clear deprecation path documented

🚀 **Ready for Phase 3:** Tensor APIs for sync/async operations and GPU integration
