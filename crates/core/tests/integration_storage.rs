//! Integration tests for Phase 2 storage redesign.
//!
//! Tests comprehensive scenarios combining BufferHandle, Storage trait,
//! CpuStorage, and Tensor<T, S> to verify end-to-end storage functionality.

use cv_core::storage::{CpuStorageMarker, Storage, StorageFactory};
use cv_core::{BufferHandle, CpuStorage};

/// Test 1: Full CPU storage pipeline - Create storage → tensor → read data
#[test]
fn test_cpu_storage_full_pipeline() {
    // Create initial data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // Create CpuStorage from vector
    let storage = CpuStorage::from_vec(data.clone()).expect("Failed to create storage");

    // Verify storage properties
    assert_eq!(storage.len(), 5);
    assert_eq!(storage.capacity(), 5);
    assert!(!storage.is_empty());
    assert_eq!(storage.shape(), &[]);

    // Verify data access via CpuStorageMarker (disambiguate method)
    let slice = CpuStorageMarker::as_slice(&storage);
    assert_eq!(slice, &data[..]);

    // Verify data recovery via to_vec
    let recovered = storage.to_vec();
    assert_eq!(recovered, data);

    // Verify handle uniqueness
    let handle = storage.handle();
    assert_ne!(handle, BufferHandle(0));
}

/// Test 2: Trait object pattern - Direct access and Any downcasting
#[test]
fn test_dyn_storage_trait_object() {
    // Create storage
    let cpu_storage = CpuStorage::from_vec(vec![10i32, 20, 30]).expect("Failed to create storage");

    // Get Any reference from storage for downcasting
    let any_ref = cpu_storage.as_any();
    let downcast = any_ref.downcast_ref::<CpuStorage<i32>>();
    assert!(downcast.is_some());

    // Verify we can access properties through downcast
    let downcast_storage = downcast.unwrap();
    assert_eq!(downcast_storage.len(), 3);
    assert_eq!(downcast_storage.to_vec(), vec![10i32, 20, 30]);
    assert_eq!(downcast_storage.handle(), cpu_storage.handle());

    // Verify data type name matches
    assert_eq!(downcast_storage.data_type_name().contains("i32"), true);
}

/// Test 3: StorageFactory trait - from_vec() factory method
#[test]
fn test_tensor_from_vec_factory() {
    // Use StorageFactory trait to create storage
    let data = vec![1.5f64, 2.5, 3.5];
    let storage = <CpuStorage<f64> as StorageFactory<f64>>::from_vec(data.clone())
        .expect("Failed to create storage via factory");

    // Verify factory created storage correctly
    assert_eq!(storage.len(), 3);
    assert_eq!(storage.to_vec(), data);

    // Test new() factory method
    let storage2 = <CpuStorage<u8> as StorageFactory<u8>>::new(4, 42u8)
        .expect("Failed to create storage via new");
    assert_eq!(storage2.len(), 4);
    assert_eq!(CpuStorageMarker::as_slice(&storage2), &[42u8, 42, 42, 42]);
}

/// Test 4: Handle uniqueness across multiple tensor-like objects
#[test]
fn test_multiple_tensors_different_handles() {
    // Create multiple independent storage instances
    let storage1 = CpuStorage::from_vec(vec![1.0f32, 2.0]).expect("Failed to create storage1");
    let storage2 = CpuStorage::from_vec(vec![3.0f32, 4.0]).expect("Failed to create storage2");
    let storage3 = CpuStorage::from_vec(vec![5.0f32, 6.0, 7.0]).expect("Failed to create storage3");

    // All handles should be unique
    let handle1 = storage1.handle();
    let handle2 = storage2.handle();
    let handle3 = storage3.handle();

    assert_ne!(handle1, handle2);
    assert_ne!(handle2, handle3);
    assert_ne!(handle1, handle3);

    // Clone should preserve handle
    let storage1_clone = storage1.clone();
    assert_eq!(storage1_clone.handle(), handle1);

    // Verify handle can be used as map key
    use std::collections::HashMap;
    let mut handle_map = HashMap::new();
    handle_map.insert(handle1, "storage1");
    handle_map.insert(handle2, "storage2");
    handle_map.insert(handle3, "storage3");

    assert_eq!(handle_map.len(), 3);
    assert_eq!(handle_map.get(&handle1), Some(&"storage1"));
    assert_eq!(handle_map.get(&handle2), Some(&"storage2"));
    assert_eq!(handle_map.get(&handle3), Some(&"storage3"));
}

/// Test 5: CPU storage shape validation
#[test]
fn test_cpu_storage_shape_validation() {
    let data = vec![1.0f32; 12];

    // Valid shape: 3x4 = 12
    let storage_valid =
        CpuStorage::from_vec_with_shape(data.clone(), vec![3, 4]).expect("Valid shape failed");
    assert_eq!(storage_valid.shape(), &[3, 4]);
    assert_eq!(storage_valid.len(), 12);

    // Valid shape: 2x2x3 = 12
    let storage_valid2 =
        CpuStorage::from_vec_with_shape(data.clone(), vec![2, 2, 3]).expect("Valid shape 2 failed");
    assert_eq!(storage_valid2.shape(), &[2, 2, 3]);

    // Invalid shape: 2x5 = 10 (expects 10, has 12)
    let storage_invalid = CpuStorage::from_vec_with_shape(data.clone(), vec![2, 5]);
    assert!(storage_invalid.is_err());
    let err = storage_invalid.unwrap_err();
    assert!(err.contains("Shape mismatch"));

    // Invalid shape: 4x4 = 16 (expects 16, has 12)
    let storage_invalid2 = CpuStorage::from_vec_with_shape(data.clone(), vec![4, 4]);
    assert!(storage_invalid2.is_err());

    // Edge case: Empty shape should work with single-element creation
    let storage_empty = CpuStorage::from_vec(vec![42.0f32]).expect("Single element failed");
    assert_eq!(storage_empty.shape(), &[]);
    assert_eq!(storage_empty.len(), 1);
}
