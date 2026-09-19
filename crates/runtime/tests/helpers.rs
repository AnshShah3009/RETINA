use parking_lot::Mutex;
use std::sync::Arc;

static TEST_ISOLATION: Mutex<()> = Mutex::new(());

pub fn with_test_isolation<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = TEST_ISOLATION.lock();
    cv_runtime::orchestrator::reset_execution_mode();
    f()
}

pub fn reset_runtime_state() {
    cv_runtime::orchestrator::reset_execution_mode();
}

pub fn create_test_scheduler() -> Arc<cv_runtime::orchestrator::TaskScheduler> {
    Arc::new(cv_runtime::orchestrator::TaskScheduler::new())
}

pub struct TestContext {
    _guard: parking_lot::MutexGuard<'static, ()>,
}

impl TestContext {
    pub fn new() -> Self {
        let guard = TEST_ISOLATION.lock();
        reset_runtime_state();
        Self { _guard: guard }
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        reset_runtime_state();
    }
}
