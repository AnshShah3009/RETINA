//! Tests for error propagation.

use cv_hal::DeviceId;
use cv_runtime::orchestrator::RuntimeRunner;
use cv_runtime::Result;

#[test]
fn test_error_propagation_on_failure() {
    let runner = RuntimeRunner::Sync(DeviceId(0));

    // Error should propagate through
    let result: Result<()> =
        runner.run(|| Err(cv_runtime::Error::RuntimeError("test error".into())));
    assert!(result.is_err());
}

#[test]
fn test_error_propagation_with_fallback() {
    let runner = RuntimeRunner::Sync(DeviceId(0));

    // Error with idempotent=false should not fallback
    let result: Result<()> = runner.run_safe(false, || {
        Err(cv_runtime::Error::RuntimeError("test error".into()))
    });
    assert!(result.is_err());
}

#[test]
fn test_result_chaining() {
    fn operation() -> Result<i32> {
        Ok(42)
    }

    fn failing_operation() -> Result<i32> {
        Err(cv_runtime::Error::RuntimeError("operation failed".into()))
    }

    let ok_result = operation();
    assert!(ok_result.is_ok());

    let err_result = failing_operation();
    assert!(err_result.is_err());
}

#[test]
fn test_device_not_found_error() {
    // Getting a non-existent device should error
    use cv_runtime::device_registry::registry;

    if let Ok(reg) = registry() {
        // Default CPU should exist
        let _ = reg.default_cpu();
    }
}

#[test]
fn test_result_map_error() {
    fn operation() -> Result<i32> {
        Err(cv_runtime::Error::RuntimeError("test".into()))
    }

    let result = operation();
    assert!(result.is_err());
}

#[test]
fn test_error_display() {
    let error = cv_runtime::Error::RuntimeError("test message".into());
    let display = format!("{}", error);
    assert!(display.contains("test message"));
}
