use cv_hal::DeviceId;
use cv_runtime::orchestrator::{reset_execution_mode, set_execution_mode, RuntimeRunner};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static TEST_ISOLATION: Mutex<()> = Mutex::new(());

fn with_isolation<F>(f: F)
where
    F: FnOnce(),
{
    let _guard = TEST_ISOLATION.lock();
    reset_execution_mode();
    f();
}

#[test]
fn test_runner_sync_basic() {
    with_isolation(|| {
        let runner = RuntimeRunner::Sync(DeviceId(0));
        let result = runner.run(|| 42);
        assert_eq!(result, 42);
    });
}

#[test]
fn test_runner_sync_returns_result() {
    with_isolation(|| {
        let runner = RuntimeRunner::Sync(DeviceId(1));
        let result = runner.run(|| "hello world");
        assert_eq!(result, "hello world");
    });
}

#[test]
fn test_runner_sync_closure_send() {
    with_isolation(|| {
        let runner = RuntimeRunner::Sync(DeviceId(0));
        let data = Arc::new(AtomicUsize::new(0));
        let d = data.clone();
        let result = runner.run(move || {
            d.fetch_add(1, Ordering::SeqCst);
            42
        });
        assert_eq!(result, 42);
        assert_eq!(data.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn test_runner_device_id() {
    with_isolation(|| {
        let runner = RuntimeRunner::Sync(DeviceId(42));
        assert_eq!(runner.device_id(), DeviceId(42));
    });
}

#[test]
fn test_run_safe_success() {
    with_isolation(|| {
        let runner = RuntimeRunner::Sync(DeviceId(0));
        let result: std::result::Result<i32, cv_runtime::Error> = runner.run_safe(true, || Ok(42));
        assert_eq!(result.unwrap(), 42);
    });
}

#[test]
fn test_run_safe_error_strict() {
    with_isolation(|| {
        set_execution_mode(cv_runtime::orchestrator::ExecutionMode::Strict);
        let runner = RuntimeRunner::Sync(DeviceId(1));

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let result: std::result::Result<(), cv_runtime::Error> = runner.run_safe(true, move || {
            c.fetch_add(1, Ordering::SeqCst);
            Err(cv_runtime::Error::RuntimeError("test error".into()))
        });

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn test_run_safe_error_adaptive() {
    with_isolation(|| {
        set_execution_mode(cv_runtime::orchestrator::ExecutionMode::Adaptive(
            cv_runtime::orchestrator::AdaptiveLevel::Basic,
        ));
        let runner = RuntimeRunner::Sync(DeviceId(999));

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let result: std::result::Result<(), cv_runtime::Error> = runner.run_safe(true, move || {
            c.fetch_add(1, Ordering::SeqCst);
            Err(cv_runtime::Error::RuntimeError("test error".into()))
        });

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn test_run_safe_non_idempotent() {
    with_isolation(|| {
        set_execution_mode(cv_runtime::orchestrator::ExecutionMode::Adaptive(
            cv_runtime::orchestrator::AdaptiveLevel::Basic,
        ));
        let runner = RuntimeRunner::Sync(DeviceId(999));

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let result: std::result::Result<(), cv_runtime::Error> =
            runner.run_safe(false, move || {
                c.fetch_add(1, Ordering::SeqCst);
                Err(cv_runtime::Error::RuntimeError("test error".into()))
            });

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}
