//! Tests for unsafe pointer cast safety.

#[test]
fn test_type_id_equality_for_same_type() {
    // TypeId should be equal for the same type
    assert_eq!(std::any::TypeId::of::<f32>(), std::any::TypeId::of::<f32>());
}

#[test]
fn test_type_id_inequality_for_different_types() {
    // TypeId should differ for different types
    assert_ne!(std::any::TypeId::of::<f32>(), std::any::TypeId::of::<f64>());
    assert_ne!(std::any::TypeId::of::<f32>(), std::any::TypeId::of::<u8>());
    assert_ne!(std::any::TypeId::of::<f32>(), std::any::TypeId::of::<u32>());
}

#[test]
fn test_transmute_safety_f32_to_f32() {
    // Transmuting f32 to f32 is safe (identity)
    let val: f32 = 3.14;
    let result: f32 = unsafe { std::mem::transmute(val) };
    assert_eq!(val, result);
}

#[test]
fn test_transmute_vec_f32_to_vec_f32() {
    // Transmuting Vec<[f32; 2]> to Vec<[f32; 2]> is safe
    let data: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0]];
    let result: Vec<[f32; 2]> = unsafe { std::mem::transmute(data.clone()) };
    assert_eq!(data, result);
}

#[test]
fn test_slice_from_raw_parts_f32() {
    // Creating &[f32] from raw f32 pointer is safe
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let slice: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_slice_from_raw_parts_array_f32() {
    // Creating &[[f32; 2]] from raw pointer is safe for f32
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let slice: &[[f32; 2]] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [f32; 2], data.len() / 2) };
    assert_eq!(slice, &[[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn test_memory_layout_f32() {
    // Verify f32 has expected size
    assert_eq!(std::mem::size_of::<f32>(), 4);
    assert_eq!(std::mem::align_of::<f32>(), 4);
}

#[test]
fn test_memory_layout_array_f32() {
    // Verify [f32; 2] has expected size
    assert_eq!(std::mem::size_of::<[f32; 2]>(), 8);
}

#[test]
fn test_memory_layout_matrix_f32() {
    // Verify [[f32; 4]; 4] has expected size (transformation matrix)
    assert_eq!(std::mem::size_of::<[[f32; 4]; 4]>(), 64);
}

#[test]
fn test_type_check_prevents_ub() {
    // Simulate the type check pattern used in gpu.rs
    fn process<T: 'static>(val: T) -> String {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safe to transmute because we verified T == f32
            let f32_val: f32 = unsafe { std::mem::transmute_copy(&val) };
            std::mem::forget(val); // Don't drop val
            format!("f32: {}", f32_val)
        } else {
            "not f32".to_string()
        }
    }

    let result = process(3.14f32);
    assert!(result.starts_with("f32:"));
}
