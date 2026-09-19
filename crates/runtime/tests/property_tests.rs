use cv_runtime::orchestrator::{
    get_execution_mode, reset_execution_mode, set_execution_mode, AdaptiveLevel, ExecutionMode,
};
use parking_lot::Mutex;
use proptest::prelude::*;

static TEST_ISOLATION: Mutex<()> = Mutex::new(());

fn with_isolation<F>(f: F)
where
    F: FnOnce(),
{
    let _guard = TEST_ISOLATION.lock();
    reset_execution_mode();
    f();
}

proptest! {
    #[test]
    fn test_execution_mode_strict_and_normal(a in any::<bool>()) {
        with_isolation(|| {
            if a {
                set_execution_mode(ExecutionMode::Strict);
                assert_eq!(get_execution_mode(), ExecutionMode::Strict);
            } else {
                set_execution_mode(ExecutionMode::Normal);
                assert_eq!(get_execution_mode(), ExecutionMode::Normal);
            }
        });
    }

    #[test]
    fn test_execution_mode_clone(mode in any::<u8>()) {
        let basic = ExecutionMode::Adaptive(AdaptiveLevel::Basic);
        let aggressive = ExecutionMode::Adaptive(AdaptiveLevel::Aggressive);

        assert_ne!(basic, aggressive);

        let mode = if mode % 3 == 0 {
            ExecutionMode::Strict
        } else if mode % 3 == 1 {
            ExecutionMode::Normal
        } else if mode % 2 == 0 {
            ExecutionMode::Adaptive(AdaptiveLevel::Basic)
        } else {
            ExecutionMode::Adaptive(AdaptiveLevel::Aggressive)
        };

        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_adaptive_level_boolean(use_basic in any::<bool>()) {
        let basic = ExecutionMode::Adaptive(AdaptiveLevel::Basic);
        let aggressive = ExecutionMode::Adaptive(AdaptiveLevel::Aggressive);

        assert_ne!(basic, aggressive);

        let mode = if use_basic {
            ExecutionMode::Adaptive(AdaptiveLevel::Basic)
        } else {
            ExecutionMode::Adaptive(AdaptiveLevel::Aggressive)
        };

        match mode {
            ExecutionMode::Adaptive(AdaptiveLevel::Basic) => {},
            ExecutionMode::Adaptive(AdaptiveLevel::Aggressive) => {},
            _ => panic!("Expected Adaptive mode"),
        }
    }

    #[test]
    fn test_execution_mode_debug_roundtrip(mode in any::<u8>()) {
        with_isolation(|| {
            let exec_mode = if mode % 4 == 0 {
                ExecutionMode::Strict
            } else if mode % 4 == 1 {
                ExecutionMode::Normal
            } else if mode % 4 == 2 {
                ExecutionMode::Adaptive(AdaptiveLevel::Basic)
            } else {
                ExecutionMode::Adaptive(AdaptiveLevel::Aggressive)
            };

            set_execution_mode(exec_mode);
            let debug_str = format!("{:?}", get_execution_mode());
            assert!(!debug_str.is_empty(), "Debug format should not be empty");
        });
    }

    #[test]
    fn test_execution_mode_state_persistence(persists in any::<bool>()) {
        with_isolation(|| {
            reset_execution_mode();
            assert_eq!(get_execution_mode(), ExecutionMode::Normal);

            set_execution_mode(ExecutionMode::Strict);

            if persists {
                assert_eq!(get_execution_mode(), ExecutionMode::Strict);
            } else {
                reset_execution_mode();
                assert_eq!(get_execution_mode(), ExecutionMode::Normal);
            }
        });
    }
}

#[test]
fn test_execution_mode_default_normal() {
    with_isolation(|| {
        reset_execution_mode();
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    });
}

#[test]
fn test_adaptive_level_count() {
    let basic = ExecutionMode::Adaptive(AdaptiveLevel::Basic);
    let aggressive = ExecutionMode::Adaptive(AdaptiveLevel::Aggressive);

    assert_ne!(basic, aggressive);

    match basic {
        ExecutionMode::Adaptive(AdaptiveLevel::Basic) => {}
        _ => panic!("Expected Basic"),
    }
    match aggressive {
        ExecutionMode::Adaptive(AdaptiveLevel::Aggressive) => {}
        _ => panic!("Expected Aggressive"),
    }
}

#[test]
fn test_execution_mode_all_variants() {
    let strict = ExecutionMode::Strict;
    let normal = ExecutionMode::Normal;
    let basic = ExecutionMode::Adaptive(AdaptiveLevel::Basic);
    let aggressive = ExecutionMode::Adaptive(AdaptiveLevel::Aggressive);

    assert_ne!(strict, normal);
    assert_ne!(strict, basic);
    assert_ne!(strict, aggressive);
    assert_ne!(normal, basic);
    assert_ne!(normal, aggressive);
    assert_ne!(basic, aggressive);
}
