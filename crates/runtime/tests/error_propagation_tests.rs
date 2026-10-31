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
    // Getting a non-existent device should report absence.
    use cv_runtime::device_registry::registry;

    // This test previously could not fail: it bound `default_cpu()` to `_` and
    // asserted nothing at all, so it passed whether or not the registry worked.
    // A test named for an error path that never checks the error path is worse
    // than no test - it reads as coverage in any report counting tests.
    let reg = registry().expect("the device registry should initialise");

    // Control: the default CPU device must exist and be the one the registry
    // nominates. Without this the absence check below could pass vacuously if
    // the registry were simply empty.
    let cpu = reg.default_cpu();
    assert!(
        reg.get_device(cpu.id()).is_some(),
        "control: the default CPU device {:?} should be retrievable by id",
        cpu.id()
    );

    // A device id that cannot exist: the registry only ever holds ids it was
    // given, so one far outside any real allocation must be absent. Derived from
    // what is actually registered rather than hardcoded, so this cannot become a
    // false assertion if ids are renumbered.
    let max_id = reg
        .all_devices()
        .iter()
        .map(|d| d.id().0)
        .max()
        .unwrap_or(0);
    let absent = DeviceId(max_id + 1000);

    assert!(
        reg.get_device(absent).is_none(),
        "looking up {absent:?} returned a device, but the highest id in use is \
         {max_id} - so that device cannot exist"
    );

    // And the same through the group's own lookup, which is the path a caller
    // would actually take.
    assert!(
        !reg.all_devices().iter().any(|d| d.id() == absent),
        "{absent:?} unexpectedly appears in all_devices()"
    );
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
