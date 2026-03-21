use cv_hal::DeviceId;
use cv_runtime::orchestrator::{
    get_execution_mode, reset_execution_mode, set_execution_mode, AdaptiveLevel, ExecutionMode,
};
use cv_runtime::GroupPolicy;
use parking_lot::Mutex;
use std::sync::Arc;

static TEST_ISOLATION: Mutex<()> = Mutex::new(());

fn create_test_scheduler() -> Arc<cv_runtime::orchestrator::TaskScheduler> {
    Arc::new(cv_runtime::orchestrator::TaskScheduler::new())
}

fn with_isolation<F>(f: F)
where
    F: FnOnce(),
{
    let _guard = TEST_ISOLATION.lock();
    reset_execution_mode();
    f();
}

#[test]
fn test_execution_mode_strict_no_fallback() {
    with_isolation(|| {
        let scheduler = create_test_scheduler();
        let _ = scheduler.create_group("default", num_cpus::get(), None, GroupPolicy::default());

        set_execution_mode(ExecutionMode::Strict);
        let gpu_id = DeviceId(1);
        let runner = cv_runtime::orchestrator::RuntimeRunner::Sync(gpu_id);

        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = std::sync::Arc::new(AtomicUsize::new(0));

        let c = counter.clone();
        let _result: Result<(), cv_runtime::Error> = runner.run_safe(true, || {
            c.fetch_add(1, Ordering::SeqCst);
            Err(cv_runtime::Error::RuntimeError(
                "Simulated GPU failure".into(),
            ))
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn test_execution_mode_adaptive_fallback() {
    with_isolation(|| {
        let _ = cv_runtime::registry().unwrap();
        let runner = cv_runtime::orchestrator::RuntimeRunner::Sync(DeviceId(999));

        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Basic));

        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = std::sync::Arc::new(AtomicUsize::new(0));

        let c = counter.clone();
        let _result: Result<(), cv_runtime::Error> = runner.run_safe(true, || {
            c.fetch_add(1, Ordering::SeqCst);
            Err(cv_runtime::Error::RuntimeError(
                "Simulated GPU failure".into(),
            ))
        });

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn test_execution_mode_aggressive_latency_hint() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Normal);
        let mode_normal = get_execution_mode();
        assert!(matches!(mode_normal, ExecutionMode::Normal));

        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Aggressive));
        let mode_aggressive = get_execution_mode();

        match mode_aggressive {
            ExecutionMode::Adaptive(AdaptiveLevel::Aggressive) => {}
            _ => panic!("Expected Adaptive(Aggressive), got {:?}", mode_aggressive),
        }
    });
}

#[test]
fn test_execution_mode_strict_variant() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Strict);
        assert_eq!(get_execution_mode(), ExecutionMode::Strict);
    });
}

#[test]
fn test_execution_mode_normal_variant() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Normal);
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    });
}

#[test]
fn test_execution_mode_adaptive_basic_variant() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Basic));
        match get_execution_mode() {
            ExecutionMode::Adaptive(level) => assert_eq!(level, AdaptiveLevel::Basic),
            _ => panic!("Expected Adaptive(Basic)"),
        }
    });
}

#[test]
fn test_execution_mode_adaptive_aggressive_variant() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Aggressive));
        match get_execution_mode() {
            ExecutionMode::Adaptive(level) => assert_eq!(level, AdaptiveLevel::Aggressive),
            _ => panic!("Expected Adaptive(Aggressive)"),
        }
    });
}

#[test]
fn test_execution_mode_reset() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Strict);
        assert_eq!(get_execution_mode(), ExecutionMode::Strict);

        reset_execution_mode();
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    });
}

#[test]
fn test_execution_mode_isolation_between_tests() {
    with_isolation(|| {
        set_execution_mode(ExecutionMode::Strict);
        assert_eq!(get_execution_mode(), ExecutionMode::Strict);
    });
}
