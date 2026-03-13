//! BufferHandle type for unified storage identification.
//!
//! Provides a lightweight, copy-able identifier for storage buffers across
//! different backends (CPU, GPU, etc.).

use std::fmt::Debug;

/// A lightweight handle to identify storage buffers uniquely.
///
/// BufferHandle provides a copy-able, hashable identifier for storage buffers,
/// enabling efficient handle-based access patterns without holding references.
///
/// # Traits
///
/// - **Copy, Clone**: Handle can be freely copied without allocation
/// - **Debug, PartialEq, Eq, Hash**: Support debugging and use as map keys
///
/// # Example
///
/// ```
/// use cv_core::BufferHandle;
///
/// let handle1 = BufferHandle(1);
/// let handle2 = handle1; // Copy semantics
/// assert_eq!(handle1, handle2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_handle_creation() {
        let handle = BufferHandle(42);
        assert_eq!(handle.0, 42);
    }

    #[test]
    fn test_buffer_handle_equality() {
        let handle1 = BufferHandle(42);
        let handle2 = BufferHandle(42);
        let handle3 = BufferHandle(43);

        assert_eq!(handle1, handle2);
        assert_ne!(handle1, handle3);
    }

    #[test]
    fn test_buffer_handle_copy() {
        let handle1 = BufferHandle(100);
        let handle2 = handle1;

        // Both should have the same value
        assert_eq!(handle1, handle2);
        // Both should still be usable (Copy semantics)
        assert_eq!(handle1.0, handle2.0);
    }

    #[test]
    fn test_buffer_handle_hashable() {
        use std::collections::HashSet;

        let handle1 = BufferHandle(1);
        let handle2 = BufferHandle(2);
        let handle3 = BufferHandle(1);

        let mut set = HashSet::new();
        set.insert(handle1);
        set.insert(handle2);
        set.insert(handle3); // Duplicate of handle1

        // Should only have 2 unique handles
        assert_eq!(set.len(), 2);
        assert!(set.contains(&handle1));
        assert!(set.contains(&handle2));
    }
}
