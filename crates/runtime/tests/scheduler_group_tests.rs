use cv_runtime::orchestrator::{reset_execution_mode, TaskPriority};
use cv_runtime::GroupPolicy;
use parking_lot::Mutex;
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

fn create_scheduler() -> Arc<cv_runtime::orchestrator::TaskScheduler> {
    Arc::new(cv_runtime::orchestrator::TaskScheduler::new())
}

#[test]
fn test_create_group_default_cpu() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        let group = s.create_group("test", 2, None, policy).unwrap();
        assert_eq!(group.name, "test");
    });
}

#[test]
fn test_create_group_specific_device() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();
        let cpu_id = cv_runtime::registry().unwrap().default_cpu().id();

        let group = s
            .create_group_with_device("test", 2, None, policy, cpu_id)
            .unwrap();
        assert_eq!(group.name, "test");
    });
}

#[test]
fn test_create_group_with_cores() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        let cores = vec![0, 1];
        let group = s
            .create_group("test", 2, Some(cores.clone()), policy)
            .unwrap();
        assert_eq!(group.name, "test");
    });
}

#[test]
fn test_remove_group() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        s.create_group("test", 2, None, policy).unwrap();
        assert!(s.get_group("test").unwrap().is_some());

        let removed = s.remove_group("test").unwrap();
        assert!(removed.is_some());
        assert!(s.get_group("test").unwrap().is_none());
    });
}

#[test]
fn test_remove_nonexistent() {
    with_isolation(|| {
        let s = create_scheduler();
        let removed = s.remove_group("nonexistent").unwrap();
        assert!(removed.is_none());
    });
}

#[test]
fn test_get_group() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        s.create_group("test", 2, None, policy).unwrap();
        let retrieved = s.get_group("test").unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test");
    });
}

#[test]
fn test_get_nonexistent_group() {
    with_isolation(|| {
        let s = create_scheduler();
        let retrieved = s.get_group("nonexistent").unwrap();
        assert!(retrieved.is_none());
    });
}

#[test]
fn test_group_policy_background() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy {
            priority: TaskPriority::Background,
            allow_work_stealing: false,
            allow_dynamic_scaling: false,
        };

        let group = s.create_group("bg", 2, None, policy).unwrap();
        assert_eq!(group.policy.priority, TaskPriority::Background);
    });
}

#[test]
fn test_group_policy_high() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy {
            priority: TaskPriority::High,
            allow_work_stealing: true,
            allow_dynamic_scaling: true,
        };

        let group = s.create_group("high", 2, None, policy).unwrap();
        assert_eq!(group.policy.priority, TaskPriority::High);
    });
}

#[test]
fn test_multiple_groups() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        s.create_group("g1", 2, None, policy.clone()).unwrap();
        s.create_group("g2", 2, None, policy.clone()).unwrap();
        s.create_group("g3", 2, None, policy).unwrap();

        assert!(s.get_group("g1").unwrap().is_some());
        assert!(s.get_group("g2").unwrap().is_some());
        assert!(s.get_group("g3").unwrap().is_some());
    });
}

#[test]
fn test_group_load_tracking() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        let group = s.create_group("test", 2, None, policy).unwrap();
        assert_eq!(group.load(), 0);
    });
}

#[test]
fn test_group_num_threads() {
    with_isolation(|| {
        let s = create_scheduler();
        let policy = GroupPolicy::default();

        let group = s.create_group("test", 4, None, policy).unwrap();
        assert_eq!(group.num_threads(), 4);
    });
}
