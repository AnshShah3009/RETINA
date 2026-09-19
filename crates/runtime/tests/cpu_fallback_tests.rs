//! Tests for CPU fallback behavior.

use cv_hal::DeviceId;
use cv_runtime::orchestrator::get_execution_mode;
use cv_runtime::orchestrator::{best_runner, try_best_runner, RuntimeRunner};
use cv_runtime::Result;

#[test]
fn test_runtime_runner_sync() {
    // Sync runner executes on calling thread
    let runner = RuntimeRunner::Sync(DeviceId(0));
    let result = runner.run(|| 42);
    assert_eq!(result, 42);
}

#[test]
fn test_runtime_runner_run_safe() {
    // run_safe should return Ok on success
    let runner = RuntimeRunner::Sync(DeviceId(0));
    let result: Result<i32> = runner.run_safe(true, || Ok(42));
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_runtime_runner_device_id() {
    let device_id = DeviceId(42);
    let runner = RuntimeRunner::Sync(device_id);
    assert_eq!(runner.device_id(), device_id);
}

#[test]
fn test_best_runner_available() {
    // best_runner should return some runner
    let runner = best_runner();
    assert!(runner.is_ok());
}

#[test]
fn test_try_best_runner() {
    let runner = try_best_runner();
    assert!(runner.is_ok());
}

#[test]
fn test_execution_mode_default() {
    let mode = get_execution_mode();
    // Default mode should be Normal
    assert!(matches!(
        mode,
        cv_runtime::orchestrator::ExecutionMode::Normal
    ));
}

#[test]
fn test_fallback_logic() {
    // Test that fallback logic exists
    let runner = RuntimeRunner::Sync(DeviceId(0));

    // When error occurs with idempotent=false, should not fallback
    let result: Result<i32> = runner.run_safe(false, || {
        Err(cv_runtime::Error::RuntimeError("test error".into()))
    });
    assert!(result.is_err());
}

#[test]
fn test_fallback_success_no_fallback() {
    // When operation succeeds, no fallback should happen
    let runner = RuntimeRunner::Sync(DeviceId(0));
    let result: Result<i32> = runner.run_safe(true, || Ok(42));
    assert_eq!(result.unwrap(), 42);
}
