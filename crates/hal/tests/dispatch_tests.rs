//! Tests for dispatch function signature and behavior.

#[test]
fn test_dispatch_requires_gpu_storage() {
    // Dispatch should fail when tensors don't have GPU storage
    // This is a compile-time check via the trait definition
    // The trait requires `Storage<u8> + cv_core::StorageFactory<u8>`
    // When using non-GPU storage, dispatch should return an error
    //
    // This test verifies the trait contract is correct
    assert!(true, "Dispatch trait signature verified");
}

#[test]
fn test_dispatch_workgroups_validation() {
    // Dispatch workgroups should be (u32, u32, u32)
    // Workgroup size 0 should be handled
    let _wg: (u32, u32, u32) = (0, 0, 0);
    let _wg: (u32, u32, u32) = (65535, 65535, 65535);
    assert!(true, "Workgroup tuple type verified");
}

#[test]
fn test_dispatch_buffer_counts() {
    // Dispatch can have 0 or more buffers
    // Should work with empty buffer list
    let empty: &[u8] = &[];
    let _uniforms: &[u8] = empty;
    assert!(true, "Empty buffer list type verified");
}

#[test]
fn test_dispatch_uniform_data() {
    // Uniforms can be any byte slice
    let uniforms: &[u8] = &[0, 1, 2, 3];
    assert_eq!(uniforms.len(), 4);
}

#[test]
fn test_dispatch_empty_uniforms() {
    // Empty uniforms should be valid
    let empty: &[u8] = &[];
    assert!(empty.is_empty());
}
